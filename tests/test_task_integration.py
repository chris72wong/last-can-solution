"""End-to-end offline task checks through the real application facade.

The camera, controller, task worker and command loop all run together. Socket
creation is forbidden throughout these tests; no robot is contacted.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import socket
import threading
import time
import unittest
import uuid
from unittest.mock import patch

from lastcan.config import ConfigStore, ROOT
from lastcan.runtime import Runtime


def wait_for(predicate, timeout=2.0, heartbeat=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if heartbeat:
            heartbeat()
        if predicate():
            return True
        time.sleep(.02)
    return bool(predicate())


class TaskIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.network_guard = patch.object(socket, "socket", side_effect=AssertionError(
            "Offline integration test attempted to create a network socket"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)
        self.temp_base = ROOT / ".cache" / "task-integration"
        self.temp_path = self.temp_base / uuid.uuid4().hex
        self.temp_path.mkdir(parents=True)
        self.addCleanup(self.cleanup_directory)
        self.runtime = None

    def cleanup_directory(self):
        target = self.temp_path.resolve()
        if target.exists() and target.is_relative_to(self.temp_base.resolve()):
            shutil.rmtree(target)

    def make_runtime(self, path=None):
        self.runtime = Runtime(demo=True, config_path=path or self.temp_path / "robot.json")
        self.addCleanup(self.runtime.wait_closed)
        self.addCleanup(self.runtime.close)
        self.assertTrue(wait_for(lambda: self.runtime.get_detection() is not None))
        return self.runtime

    @staticmethod
    def servos(runtime):
        return [args[0] for kind, args in runtime.transport.commands if kind == "servos"]

    @staticmethod
    def nonzero_velocities(runtime):
        return [args for kind, args in runtime.transport.commands
                if kind == "velocity" and any(args)]

    def test_full_grab_drive_home_and_horizontal_drop_with_live_worker_threads(self):
        runtime = self.make_runtime()
        initial = runtime.get_detection()
        self.assertTrue(initial.simulated)
        self.assertEqual(len(initial.boxes), 2)
        self.assertTrue(runtime.grab(), runtime.tasks.error)
        runtime.set_drive_action("left", True)  # rejected during automatic control
        self.assertEqual(runtime._actions, set())
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy, timeout=28,
                                 heartbeat=runtime.heartbeat), runtime.tasks.status)
        self.assertEqual(runtime.tasks.error, "", runtime.tasks.status)
        self.assertIn("verify secure grip", runtime.tasks.status)
        velocities = self.nonzero_velocities(runtime)
        self.assertTrue(any(v[2] < 0 for v in velocities), "Alignment never turned toward the can")
        self.assertTrue(any(v[0] > 0 for v in velocities), "Alignment never approached the can")
        expected_grab = [runtime.config.data["poses"][name]["positions"] for name in
                         ("home", "open", "reach", "grip", "lift", "carry")]
        actual_grab = [{str(k): v for k, v in p.items()} for p in self.servos(runtime)]
        self.assertEqual(actual_grab, expected_grab)

        runtime.heartbeat()  # release/rearm after the completed task
        runtime.set_drive_action("back", True)
        self.assertTrue(wait_for(lambda: any(v[0] < 0 for v in self.nonzero_velocities(runtime)),
                                 heartbeat=runtime.heartbeat))
        runtime.set_drive_action("back", False)
        self.assertTrue(runtime.drop(), runtime.tasks.error)
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy, timeout=4,
                                 heartbeat=runtime.heartbeat), runtime.tasks.status)
        self.assertEqual(runtime.tasks.error, "", runtime.tasks.status)
        expected_drop = [runtime.config.data["poses"][name]["positions"] for name in
                         ("release_horizontal", "open", "home")]
        actual_drop = [{str(k): v for k, v in p.items()} for p in self.servos(runtime)[6:]]
        self.assertEqual(actual_drop, expected_drop)
        self.assertIn("verify horizontal release", runtime.tasks.status)
        self.assertEqual(runtime.controller.owner, "stopped")
        self.assertFalse(runtime.config.path.exists(), "Simulation leaked into physical calibration")

    def test_operator_stop_cancels_active_sequence_and_never_queues_remaining_poses(self):
        runtime = self.make_runtime()
        runtime.config.data["poses"]["home"]["duration"] = .5
        self.assertTrue(runtime.grab())
        self.assertTrue(wait_for(lambda: len(self.servos(runtime)) == 1))
        runtime.stop("Operator Stop integration test")
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy))
        # Give both the task worker and periodic command worker several cycles.
        time.sleep(.25)
        self.assertEqual(len(self.servos(runtime)), 1)
        self.assertIn("Cancelled", runtime.tasks.status)
        self.assertEqual(runtime._actions, set())
        self.assertEqual(runtime.transport.commands[-1], ("velocity", (0., 0., 0.)))

    def test_drop_opens_only_gripper_and_preserves_horizontal_arm_pose_until_release(self):
        runtime = self.make_runtime()
        poses = runtime.config.data["poses"]
        self.assertEqual(set(poses["open"]["positions"]), {"5"})
        self.assertEqual(set(poses["grip"]["positions"]), {"5"})
        self.assertEqual(poses["release_horizontal"]["positions"]["5"],
                         poses["grip"]["positions"]["5"],
                         "Moving into horizontal pose must not release the can")
        self.assertTrue(runtime.drop(), runtime.tasks.error)
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy, timeout=3))
        commands = self.servos(runtime)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[1], {5: poses["open"]["positions"]["5"]})
        positions = dict(commands[0])
        horizontal_joints = {sid: value for sid, value in positions.items() if sid != 5}
        positions.update(commands[1])
        self.assertEqual({sid: value for sid, value in positions.items() if sid != 5}, horizontal_joints)
        self.assertEqual(commands[2], {int(k): v for k, v in poses["home"]["positions"].items()})

    def test_disconnect_during_alignment_aborts_and_reconnect_does_not_resume(self):
        runtime = self.make_runtime()
        self.assertTrue(runtime.grab())
        self.assertTrue(wait_for(lambda: self.nonzero_velocities(runtime), timeout=4))
        runtime.disconnect()
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy and not runtime.transport.connected))
        before_reconnect = len(runtime.transport.commands)
        servo_count = len(self.servos(runtime))
        runtime.connect()
        self.assertTrue(wait_for(lambda: runtime.transport.connected and runtime.get_detection() is not None,
                                 timeout=4, heartbeat=runtime.heartbeat))
        time.sleep(.25)
        self.assertFalse(runtime.tasks.busy)
        self.assertEqual(len(self.servos(runtime)), servo_count)
        new_velocities = [args for kind, args in runtime.transport.commands[before_reconnect:]
                          if kind == "velocity"]
        self.assertTrue(new_velocities)
        self.assertTrue(all(not any(v) for v in new_velocities))
        self.assertEqual(runtime._actions, set())

    def test_transport_session_loss_during_task_aborts_even_with_automatic_reconnect(self):
        runtime = self.make_runtime()
        runtime.config.data["poses"]["home"]["duration"] = .5
        self.assertTrue(runtime.grab())
        self.assertTrue(wait_for(lambda: len(self.servos(runtime)) == 1))
        session = runtime.transport.session_id
        runtime.transport.close()  # simulate a socket loss, still entirely in memory
        self.assertTrue(wait_for(lambda: not runtime.tasks.busy))
        self.assertTrue(wait_for(lambda: runtime.transport.connected and
                                 runtime.transport.session_id != session, timeout=4))
        self.assertEqual(len(self.servos(runtime)), 1)
        self.assertTrue(runtime.tasks.error)
        self.assertEqual(runtime._actions, set())
        self.assertEqual(runtime.transport.commands[-1], ("velocity", (0., 0., 0.)))

    def test_demo_calibration_cannot_overwrite_an_existing_real_configuration(self):
        path = self.temp_path / "existing-real-robot.json"
        real_store = ConfigStore(path)
        real_store.data["robot"]["ip"] = "192.0.2.42"
        real_store.save()
        original = path.read_bytes()
        runtime = self.make_runtime(path)
        for servo in runtime.config.data["servos"]:
            runtime.set_servo(servo["id"], 500, duration=1)
        self.assertTrue(wait_for(lambda: len(runtime.controller.commanded_positions) == 5))
        runtime.save_pose("synthetic_pose", 1, 0, True)
        runtime.save_alignment(.55, .4, .04, .03, 1, 1)
        runtime.save_settings()
        runtime.close()
        self.assertTrue(runtime.wait_closed())
        self.assertEqual(path.read_bytes(), original)
        reloaded = ConfigStore(path)
        self.assertEqual(reloaded.data["robot"]["ip"], "192.0.2.42")
        self.assertEqual(reloaded.data["servos"], [])
        self.assertEqual(reloaded.data["poses"], {})
        self.assertIsNone(reloaded.data["alignment"]["target_x"])

    def test_runtime_stop_is_nonblocking_and_invalidates_queued_servos_during_slow_send(self):
        runtime = self.make_runtime()
        entered, release = threading.Event(), threading.Event()
        original = runtime.transport.move_servos
        def slow_move(positions, duration):
            entered.set()
            release.wait(1)
            original(positions, duration)
        runtime.transport.move_servos = slow_move
        runtime.set_servo(1, 510, duration=1)
        runtime.set_servo(2, 520, duration=1)
        self.assertTrue(entered.wait(.5))
        try:
            started = time.monotonic()
            runtime.stop("Stop during an in-flight servo command")
            runtime.heartbeat()
            self.assertLess(time.monotonic() - started, .1)
        finally:
            release.set()
        self.assertTrue(wait_for(lambda: self.servos(runtime)))
        time.sleep(.25)
        commands = self.servos(runtime)
        self.assertEqual(commands, [{1: 510}], "A stale second servo command escaped Stop")
        self.assertEqual(runtime.transport.commands[-1], ("velocity", (0., 0., 0.)))


if __name__ == "__main__":
    unittest.main()
