from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import shutil
import time
import unittest
import uuid
from unittest.mock import patch

from PIL import Image

from lastcan.camera import JpegBuffer
from lastcan.config import ConfigStore, DEFAULT_CONFIG, ROOT
from lastcan.types import Detection, DetectionPacket, FramePacket, LatestValue, drive_axes, fit_geometry


def wait_until(predicate, timeout=2.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return True
        time.sleep(.025)
    return False


class WorkspaceTemporaryDirectory:
    """Default-mode mkdir avoids restricted Windows sandbox ACLs on mode 0700."""
    def __init__(self):
        self.base = ROOT / ".cache" / "test-tmp"
        self.path = self.base / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.name = str(self.path)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()

    def cleanup(self):
        target = self.path.resolve()
        if target.exists() and target.is_relative_to(self.base.resolve()):
            shutil.rmtree(target)


def temporary_directory():
    return WorkspaceTemporaryDirectory()


class PacketAndCameraTests(unittest.TestCase):
    def test_jpeg_split_markers_and_multiple_frames(self):
        parser = JpegBuffer()
        self.assertEqual(parser.feed(b"boundary\xff"), [])
        self.assertEqual(parser.feed(b"\xd8one\xff"), [])
        self.assertEqual(parser.feed(b"\xd9\r\n\xff\xd8two\xff\xd9"),
                         [b"\xff\xd8one\xff\xd9", b"\xff\xd8two\xff\xd9"])

    def test_corrupt_camera_input_does_not_grow_without_bound(self):
        parser = JpegBuffer(limit=64)
        parser.feed(b"\xff\xd8" + b"x" * 1000)
        self.assertLessEqual(len(parser.buffer), 64)
        self.assertEqual(parser.feed(b"noise\xff\xd8ok\xff\xd9"), [b"\xff\xd8ok\xff\xd9"])

    def test_letterboxed_image_and_boxes_share_transform(self):
        scale, x, y = fit_geometry((640, 360), (800, 600))
        self.assertEqual((scale, x, y), (1.25, 0, 75))
        box = Detection(64, 36, 128, 72, .9)
        transformed = (box.x1 * scale + x, box.y1 * scale + y,
                       box.x2 * scale + x, box.y2 * scale + y)
        self.assertEqual(transformed, (80, 120, 160, 165))

    def test_all_six_directions_and_conflicting_keys(self):
        expected = {"forward": (.12, 0, 0), "back": (-.12, 0, 0),
                    "left": (0, .12, 0), "right": (0, -.12, 0),
                    "rotate_left": (0, 0, .35), "rotate_right": (0, 0, -.35)}
        for action, velocity in expected.items():
            self.assertEqual(drive_axes({action}, .12, .35), velocity)
        self.assertEqual(drive_axes({"left", "right", "forward", "back"}, .12, .35), (0, 0, 0))

    def test_latest_slot_overwrites_old_frames(self):
        slot = LatestValue()
        for i in range(100):
            slot.put(i)
        self.assertEqual(slot.get(), 99)
        slot.put(None)
        self.assertIsNone(slot.get())

    def test_config_default_no_fabricated_physical_calibration(self):
        with temporary_directory() as directory:
            path = Path(directory) / "robot.json"
            store = ConfigStore(path)
            self.assertEqual(store.data["servos"], [])
            self.assertEqual(store.data["poses"], {})
            self.assertIsNone(store.data["alignment"]["target_x"])
            store.save()
            self.assertEqual(json.loads(path.read_text())["servos"], [])


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, demo=True):
        from lastcan.runtime import Runtime
        temporary = temporary_directory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "robot.json"
        runtime = Runtime(demo=demo, config_path=path,
                          model_path=None if demo else str(Path(temporary.name) / "missing-model.pt"))
        self.addCleanup(runtime.wait_closed)
        self.addCleanup(runtime.close)
        return runtime, path

    def test_demo_cannot_open_network_and_does_not_write_real_config(self):
        with patch.object(socket, "socket", side_effect=AssertionError("Demo touched the network")):
            runtime, path = self.make_runtime()
            self.assertTrue(wait_until(lambda: runtime.get_detection() is not None))
            runtime.save_settings()
            runtime.save_alignment(.5, .4, .04, .03, 1, 1)
            self.assertFalse(path.exists())
            self.assertTrue(runtime.get_detection().simulated)
            self.assertTrue(runtime.transport.is_fake)

    def test_release_focus_style_stop_and_input_lease(self):
        runtime, _ = self.make_runtime()
        self.assertTrue(wait_until(lambda: runtime.transport.connected))
        runtime.heartbeat()
        runtime.set_drive_action("forward", True)
        self.assertTrue(wait_until(lambda: any(k == "velocity" and a[0] > 0 for k, a in runtime.transport.commands)))
        runtime.set_drive_action("forward", False)
        self.assertTrue(wait_until(lambda: runtime.transport.commands[-1] == ("velocity", (0., 0., 0.))))
        runtime.set_drive_action("left", True)
        runtime.stop("Window lost focus")
        self.assertTrue(wait_until(lambda: runtime.transport.commands[-1] == ("velocity", (0., 0., 0.))))
        runtime.heartbeat()
        runtime.set_drive_action("forward", True)
        self.assertTrue(wait_until(lambda: runtime.controller.owner == "manual"))
        time.sleep(.6)  # no GUI heartbeat: the manual command lease must expire
        self.assertEqual(runtime.transport.commands[-1], ("velocity", (0., 0., 0.)))

    def test_display_uses_the_image_that_produced_the_boxes(self):
        runtime, _ = self.make_runtime()
        runtime.camera.stop()
        now = time.monotonic()
        old_image = Image.new("RGB", (640, 360), "red")
        latest_image = Image.new("RGB", (640, 360), "blue")
        detection = DetectionPacket(10, now, now, old_image, (Detection(1, 2, 3, 4, .9),), True)
        runtime._frames.put(FramePacket(11, now + .01, latest_image))
        runtime._detections.put(detection)
        self.assertIs(runtime.get_display().image, old_image)
        runtime._detections.put(DetectionPacket(9, now - 5, now, old_image, (), True))
        self.assertIs(runtime.get_display().image, latest_image)

    def test_close_finishes_with_zero_and_disconnects_fake_robot(self):
        runtime, _ = self.make_runtime()
        runtime.heartbeat()
        runtime.set_drive_action("forward", True)
        time.sleep(.12)
        runtime.close()
        runtime.wait_closed()
        self.assertFalse(runtime.transport.connected)
        velocities = [args for kind, args in runtime.transport.commands if kind == "velocity"]
        self.assertEqual(velocities[-1], (0., 0., 0.))

    def test_default_runtime_stays_disconnected_without_robot(self):
        runtime, _ = self.make_runtime(demo=False)
        time.sleep(.1)
        self.assertFalse(runtime.transport.connected)
        runtime.set_drive_action("forward", True)
        self.assertEqual(runtime._actions, set())
        self.assertFalse(runtime.grab())
        self.assertIn("trained can detector", runtime.message)

    def test_recording_pose_uses_explicit_joint_subset_and_rejects_outdated_limits(self):
        runtime, path = self.make_runtime()
        runtime.controller.commanded_positions.update({1: 500, 2: 500, 3: 500, 4: 500, 5: 600})
        with self.assertRaisesRegex(ValueError, "gripper"):
            runtime.save_pose("open", 1.0, .2, True)
        runtime.save_pose("open", 1.0, .2, True, servo_ids=[5])
        self.assertEqual(runtime.config.data["poses"]["open"]["positions"], {"5": 600})
        with self.assertRaisesRegex(ValueError, "home"):
            runtime.save_pose("home", 1.0, .2, True, servo_ids=[5])
        runtime.config.data["servos"][-1]["max"] = 550
        with self.assertRaisesRegex(ValueError, "updated limits"):
            runtime.save_pose("grip", 1.0, .2, True, servo_ids=[5])
        self.assertFalse(path.exists())

    def test_changing_robot_endpoint_invalidates_previous_physical_calibration(self):
        runtime, path = self.make_runtime()
        runtime.disconnect()
        self.assertTrue(wait_until(lambda: not runtime.transport.connected))
        runtime.config.data["robot"]["ip"] = "different-robot"
        runtime.save_settings()
        self.assertTrue(all(not s["verified"] for s in runtime.config.data["servos"]))
        self.assertTrue(all(not p["verified"] for p in runtime.config.data["poses"].values()))
        self.assertFalse(runtime.config.data["alignment"]["calibrated"])
        self.assertFalse(runtime.grab())
        self.assertFalse(runtime.drop())
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
