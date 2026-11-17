"""Offline behavior tests. All servo calibration here is SYNTHETIC."""
import copy
import math
import threading
import time
import unittest
from types import SimpleNamespace

from lastcan.config import DEFAULT_CONFIG
from lastcan.control import (ControlError, FakeTransport, RobotController,
                             RosbridgeTransport, TaskRunner)
from lastcan.types import Detection, DetectionPacket, drive_axes


def calibrated_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["robot"]["topics_verified"] = True
    config["servos"] = [dict(id=1, name="SYNTHETIC TEST JOINT", min=400, max=600, verified=True)]
    names = ["home", "open", "reach", "grip", "lift", "carry", "release_horizontal"]
    config["poses"] = {name: dict(positions={"1": 450 + i * 10}, duration=0.05,
                                 pause=0, verified=True) for i, name in enumerate(names)}
    config["alignment"].update(calibrated=True, target_x=0.5, target_height=0.4,
                               x_tolerance=0.03, height_tolerance=0.03,
                               turn_sign=1, forward_sign=1, pulse_s=0.02,
                               settle_s=0.02, timeout_s=0.8, target_loss_s=0.15,
                               max_frame_age_s=0.2, stable_frames=2)
    return config


def packet(frame_id=1, x=0.5, height=0.4, captured_at=None, boxes=None, simulated=False):
    box = Detection((x - 0.04) * 640, (0.5 - height / 2) * 360,
                    (x + 0.04) * 640, (0.5 + height / 2) * 360, 0.95)
    return DetectionPacket(frame_id, time.monotonic() if captured_at is None else captured_at,
                           time.monotonic(), SimpleNamespace(size=(640, 360)),
                           (box,) if boxes is None else boxes, simulated)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.config = calibrated_config()
        self.transport = FakeTransport(self.config)
        self.transport.connect()
        self.controller = RobotController(self.transport, self.config)

    def test_live_constructor_never_connects(self):
        transport = RosbridgeTransport(self.config)
        self.assertFalse(transport.connected)
        self.assertIsNone(transport._socket)

    def test_all_driving_axes_and_opposing_inputs(self):
        expected = {"forward": (0.12, 0, 0), "back": (-0.12, 0, 0),
                    "left": (0, 0.12, 0), "right": (0, -0.12, 0),
                    "rotate_left": (0, 0, 0.35), "rotate_right": (0, 0, -0.35)}
        for action, axes in expected.items():
            with self.subTest(action=action):
                result = drive_axes({action}, 0.12, 0.35)
                self.assertEqual(result, axes)
                self.assertTrue(self.controller.set_manual(*result))
                self.controller.tick()
                self.assertEqual(self.transport.commands[-1], ("velocity", axes))
        self.assertEqual(drive_axes({"forward", "back", "left", "right"}, .12, .35), (0, 0, 0))

    def test_release_and_stop_zero_immediately(self):
        self.controller.set_manual(.12, 0, 0)
        self.controller.tick()
        self.controller.set_manual(0, 0, 0)
        self.controller.tick()
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.controller.set_manual(.12, 0, 0)
        self.controller.stop("Focus lost")
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.assertEqual(self.controller.owner, "stopped")

    def test_manual_deadman_expires_and_requires_release(self):
        self.controller.set_manual(.12, 0, 0)
        self.controller.tick(time.monotonic() + 2)
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.assertFalse(self.controller.set_manual(.12, 0, 0))
        self.controller.set_manual(0, 0, 0)
        self.assertTrue(self.controller.set_manual(.12, 0, 0))

    def test_reconnect_does_not_replay_held_input_or_positions(self):
        self.controller.move_manual_servo(1, 500)
        self.controller.set_manual(.12, 0, 0)
        self.controller.tick()
        self.transport.close()
        self.transport.connect()  # even a disconnect between ticks is detected by session ID
        self.controller.tick()
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.assertFalse(self.controller.set_manual(.12, 0, 0))
        self.assertEqual(self.controller.commanded_positions, {})
        self.controller.set_manual(0, 0, 0)
        self.assertTrue(self.controller.set_manual(.12, 0, 0))

    def test_task_exclusively_owns_velocity_even_against_manual_zero(self):
        self.assertTrue(self.controller.acquire())
        self.controller.set_automatic(.05, 0, 0)
        self.assertFalse(self.controller.set_manual(0, 0, 0))
        self.assertFalse(self.controller.set_manual(-.12, 0, 0))
        self.controller.clear_manual()
        self.controller.tick()
        self.assertEqual(self.transport.commands[-1], ("velocity", (.05, 0, 0)))
        with self.assertRaises(ControlError):
            self.controller.move_manual_servo(1, 500)
        self.controller.stop()
        with self.assertRaises(ControlError):
            self.controller.set_automatic(.05, 0, 0)

    def test_automatic_intent_also_expires(self):
        self.controller.acquire()
        self.controller.set_automatic(.05, 0, 0, lease_s=.1)
        self.controller.tick(time.monotonic() + 1)
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))

    def test_acquire_and_deferred_stop_do_not_do_socket_io(self):
        self.assertTrue(self.controller.acquire())
        self.assertEqual(self.transport.commands, [])
        self.controller.set_automatic(.05, 0, 0)
        count = len(self.transport.commands)
        self.controller.stop("GUI stop", publish=False)
        self.assertEqual(len(self.transport.commands), count)
        self.controller.tick()
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))

    def test_unverified_topics_never_move(self):
        self.config["robot"]["topics_verified"] = False
        self.assertFalse(self.controller.set_manual(.12, 0, 0))
        self.assertFalse(self.controller.acquire())
        with self.assertRaises(ControlError):
            self.controller.move_manual_servo(1, 500)
        self.assertFalse(any(command[0] == "servos" for command in self.transport.commands))

    def test_manual_servo_limits_and_commanded_not_measured(self):
        self.controller.move_manual_servo(1, 530, 1.5)
        self.assertEqual(self.transport.commands[-1], ("servos", ({1: 530}, 1.5)))
        self.assertEqual(self.controller.commanded_positions, {1: 530})
        for position in (399, 601, float("nan"), 500.1):
            with self.subTest(position=position), self.assertRaises(ControlError):
                self.controller.move_manual_servo(1, position)
        self.config["servos"][0]["verified"] = False
        with self.assertRaises(ControlError):
            self.controller.move_manual_servo(1, 500)

    def test_invalid_velocity_rejected(self):
        for value in (float("nan"), float("inf"), "0.12", True):
            with self.subTest(value=value), self.assertRaises(ControlError):
                self.controller.set_manual(value, 0, 0)

    def test_gui_stop_and_heartbeat_never_wait_for_an_inflight_network_write(self):
        entered, release = threading.Event(), threading.Event()
        original = self.transport.publish_velocity
        def slow_send(lx, ly, az):
            if any((lx, ly, az)):
                entered.set()
                release.wait(1)
            original(lx, ly, az)
        self.transport.publish_velocity = slow_send
        self.controller.set_manual(.12, 0, 0)
        worker = threading.Thread(target=self.controller.tick)
        worker.start()
        self.assertTrue(entered.wait(.5))
        try:
            started = time.monotonic()
            self.assertFalse(self.controller.set_manual(0, 0, 0))
            self.assertFalse(self.controller.acquire())
            self.controller.clear_manual()
            self.controller.stop("GUI Stop during write", publish=False)
            self.assertLess(time.monotonic() - started, .1)
        finally:
            release.set()
            worker.join(1)
        # That already-issued write may finish, but every later tick is zero.
        self.controller.tick()
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.assertFalse(self.controller.set_manual(.12, 0, 0))

    def test_stale_queued_servo_guard_prevents_send(self):
        with self.assertRaisesRegex(ControlError, "Queued servo command"):
            self.controller.move_manual_servo(1, 500, guard=lambda: False)
        self.assertEqual(self.transport.commands, [])


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.config = calibrated_config()
        self.transport = FakeTransport(self.config)
        self.transport.connect()
        self.controller = RobotController(self.transport, self.config)
        self.frame = 0
        self.callback = self.fresh_packet
        self.runner = TaskRunner(self.controller, self.config, lambda: self.callback())

    def fresh_packet(self):
        self.frame += 1
        return packet(self.frame)

    def tearDown(self):
        self.runner.cancel("Test cleanup")
        self.runner.join(1)

    def wait_done(self, timeout=2):
        self.runner.join(timeout)
        self.assertFalse(self.runner.busy, self.runner.status)

    def servo_commands(self):
        return [command[1][0] for command in self.transport.commands if command[0] == "servos"]

    def test_missing_calibration_rejects_before_any_motion(self):
        self.config["alignment"]["calibrated"] = False
        self.assertFalse(self.runner.start_grab())
        self.assertIn("Calibrate", self.runner.error)
        self.assertEqual(self.transport.commands, [])

    def test_bad_late_pose_prevalidated_before_home_moves(self):
        self.config["poses"]["carry"]["positions"]["1"] = 999
        self.assertFalse(self.runner.start_grab())
        self.assertEqual(self.transport.commands, [])

    def test_missing_target_rejected_without_arm_motion(self):
        self.callback = lambda: None
        self.assertFalse(self.runner.start_grab())
        self.assertIn("fresh visible", self.runner.error)
        self.assertEqual(self.transport.commands, [])

    def test_grab_alignment_success_and_exact_pose_order(self):
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertEqual(self.runner.error, "")
        expected = [self.config["poses"][name]["positions"]["1"] for name in
                    ("home", "open", "reach", "grip", "lift", "carry")]
        self.assertEqual([command[1] for command in self.servo_commands()], expected)
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))
        self.assertIn("verify secure grip", self.runner.status)

    def test_drop_works_without_detector_and_is_horizontal_then_open(self):
        self.config["alignment"]["calibrated"] = False
        self.callback = lambda: None
        self.assertTrue(self.runner.start_drop())
        self.wait_done()
        expected = [self.config["poses"][name]["positions"]["1"] for name in
                    ("release_horizontal", "open", "home")]
        self.assertEqual([command[1] for command in self.servo_commands()], expected)

    def test_drop_missing_horizontal_pose_cannot_run(self):
        del self.config["poses"]["release_horizontal"]
        self.assertFalse(self.runner.start_drop())
        self.assertIn("release_horizontal", self.runner.error)
        self.assertEqual(self.transport.commands, [])

    def test_overlap_and_manual_movement_are_rejected(self):
        self.assertTrue(self.runner.start_grab())
        self.assertFalse(self.runner.start_drop())
        self.assertFalse(self.controller.set_manual(.12, 0, 0))
        with self.assertRaises(ControlError):
            self.controller.move_manual_servo(1, 500)
        self.runner.cancel("Operator Stop")
        self.wait_done()

    def test_cancel_prevents_remaining_poses(self):
        self.config["poses"]["home"]["duration"] = .3
        self.assertTrue(self.runner.start_grab())
        deadline = time.monotonic() + .5
        while not self.servo_commands() and time.monotonic() < deadline:
            time.sleep(.005)
        self.runner.cancel("Operator Stop")
        self.wait_done()
        self.assertEqual(len(self.servo_commands()), 1)
        self.assertIn("Cancelled", self.runner.status)
        self.assertEqual(self.transport.commands[-1], ("velocity", (0, 0, 0)))

    def test_disconnect_aborts_before_remaining_poses(self):
        self.config["poses"]["home"]["duration"] = .3
        self.assertTrue(self.runner.start_grab())
        deadline = time.monotonic() + .5
        while not self.servo_commands() and time.monotonic() < deadline:
            time.sleep(.005)
        self.transport.close()
        self.transport.connect()
        self.wait_done()
        self.assertIn("connection changed", self.runner.error)
        self.assertEqual(len(self.servo_commands()), 1)

    def test_target_loss_does_not_switch_to_a_different_can(self):
        count = 0
        def moving_away():
            nonlocal count
            count += 1
            return packet(count, x=.5 if count <= 2 else .95, height=.2)
        self.callback = moving_away
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertIn("Target lost", self.runner.error)
        self.assertEqual(len(self.servo_commands()), 2)  # home/open only

    def test_initial_target_is_retained_across_home_open(self):
        calls = 0
        def replaced_target():
            nonlocal calls
            calls += 1
            return packet(calls, x=.25 if calls == 1 else .75, height=.4)
        self.callback = replaced_target
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertIn("Target lost", self.runner.error)
        self.assertEqual(len(self.servo_commands()), 2)

    def test_stale_frames_abort_before_reaching(self):
        calls = 0
        def then_stale():
            nonlocal calls
            calls += 1
            return packet(calls, captured_at=time.monotonic() - (0 if calls == 1 else 5))
        self.callback = then_stale
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertIn("stale", self.runner.error.lower())
        self.assertEqual(len(self.servo_commands()), 2)

    def test_same_frame_cannot_satisfy_stability(self):
        first = packet()
        self.callback = lambda: first
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertIn("Target lost", self.runner.error)
        self.assertEqual(len(self.servo_commands()), 2)

    def test_alignment_timeout_is_bounded(self):
        self.config["alignment"].update(timeout_s=.12, target_loss_s=1)
        self.callback = lambda: packet(self._next(), x=.62, height=.2)
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertIn("timed out", self.runner.error)
        self.assertEqual(len(self.servo_commands()), 2)

    def _next(self):
        self.frame += 1
        return self.frame

    def test_pulse_pause_fresh_frame_alignment_approaches_then_grabs(self):
        def observed():
            nonzero = [c[1] for c in self.transport.commands
                       if c[0] == "velocity" and any(c[1])]
            if not nonzero:
                x, height = .65, .2
            elif nonzero[-1][2]:
                x, height = .5, .2
            else:
                x, height = .5, .4
            return packet(self._next(), x=x, height=height)
        self.callback = observed
        self.assertTrue(self.runner.start_grab())
        self.wait_done()
        self.assertEqual(self.runner.error, "")
        velocities = [c[1] for c in self.transport.commands if c[0] == "velocity"]
        self.assertTrue(any(v[2] < 0 for v in velocities))
        self.assertTrue(any(v[0] > 0 for v in velocities))
        self.assertEqual(len(self.servo_commands()), 6)

    def test_simulated_detections_cannot_drive_real_transport(self):
        self.transport.is_fake = False  # no real sockets are ever created
        self.callback = lambda: packet(self._next(), simulated=True)
        self.assertFalse(self.runner.start_grab())
        self.assertIn("Simulated", self.runner.error)
        self.assertEqual(self.transport.commands, [])

    def test_nearest_target_and_ambiguity(self):
        previous = (.5, .5, .08, .3)
        central = Detection(300, 125, 340, 235, .9)
        distant = Detection(560, 125, 620, 235, .99)
        selected = TaskRunner._select((distant, central), (640, 360), previous, .2)
        self.assertIs(selected[0], central)
        nearby_twin = Detection(302, 125, 342, 235, .9)
        with self.assertRaisesRegex(ControlError, "ambiguous"):
            TaskRunner._select((central, nearby_twin), (640, 360), previous, .2)


if __name__ == "__main__":
    unittest.main()
