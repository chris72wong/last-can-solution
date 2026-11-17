"""Offline UI and coordinate checks. No real robot transport is constructed."""
from __future__ import annotations

import copy
import gc
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from lastcan.config import ConfigStore, DEFAULT_CONFIG
from lastcan.gui import Cockpit, canvas_box, fresh_largest_box
from lastcan.types import Detection, DetectionPacket


class FakeRuntime:
    def __init__(self, folder: Path):
        self.demo = True
        self.config = ConfigStore(folder / "robot.json")
        self.message = "Offline test"
        self.calls = []
        self.packet = None
        self.state = {"connected": True, "connection_status": "Simulated connection",
            "camera_status": "Simulated camera", "detector_status": "Simulated detections",
            "task_status": "Idle", "task_error": "", "busy": False,
            "owner": "stopped", "commanded_positions": {}, "calibrated": False,
            "model_available": True, "detection_count": 0,
            "grab_errors": ["Missing calibration"], "drop_errors": ["Missing poses"]}

    def heartbeat(self): self.calls.append(("heartbeat",))
    def snapshot(self): return self.state.copy()
    def get_display(self): return self.packet
    def get_detection(self): return self.packet
    def set_drive_action(self, action, pressed): self.calls.append(("drive", action, pressed))
    def stop(self, reason="Stopped"): self.calls.append(("stop", reason))
    def grab(self): self.calls.append(("grab",))
    def drop(self): self.calls.append(("drop",))
    def close(self): self.calls.append(("close",))
    def connect(self): self.calls.append(("connect",))
    def disconnect(self): self.calls.append(("disconnect",))
    def save_frame(self): self.calls.append(("save_frame",))
    def set_confidence(self, value): self.calls.append(("confidence", value))
    def set_servo(self, sid, position, duration=1): self.calls.append(("servo", sid, position, duration))
    def save_settings(self): self.calls.append(("settings",))
    def save_pose(self, *args, **kwargs): self.calls.append(("pose", *args, kwargs))
    def save_alignment(self, **kwargs): self.calls.append(("alignment", kwargs))


class CoordinateTests(unittest.TestCase):
    def test_letterbox_coordinates_for_multiple_boxes(self):
        # 640x360 in 800x600 -> scale 1.25, top offset 75.
        a = canvas_box(Detection(0, 0, 640, 360, .9), (640, 360), (800, 600))
        b = canvas_box(Detection(100, 100, 200, 200, .8), (640, 360), (800, 600))
        self.assertEqual(a, (0, 75, 800, 525))
        self.assertEqual(b, (125, 200, 250, 325))

    def test_capture_largest_fresh_box_and_reject_wrong_mode_and_stale(self):
        packet = DetectionPacket(1, 10, 10.1, Image.new("RGB", (640, 360)),
            (Detection(0, 0, 10, 10, .9), Detection(200, 100, 440, 280, .8)), True)
        self.assertEqual(fresh_largest_box(packet, 1, True, now=10.5), (.5, .5))
        with self.assertRaisesRegex(ValueError, "mode"):
            fresh_largest_box(packet, 1, False, now=10.5)
        with self.assertRaisesRegex(ValueError, "stale"):
            fresh_largest_box(packet, 1, True, now=12)
        with self.assertRaises(ValueError):
            fresh_largest_box(None, 1, True, now=12)


class CockpitTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"No desktop Tk display: {exc}")
        self.root.withdraw()
        # No disk writes: FakeRuntime.save_settings is deliberately in-memory.
        self.runtime = FakeRuntime(Path(__file__).resolve().parent / "unused-gui-config")
        self.app = Cockpit(self.root, self.runtime)

    def tearDown(self):
        app = getattr(self, "app", None)
        try:
            if app is not None:
                app.close()
        finally:
            # Tk interpreters, variables and images must be finalized on their
            # creating thread. Later tests start camera workers that may trigger
            # Python's cyclic GC, so don't leave destroyed GUI cycles for them.
            self.app = None
            self.root = None
            self.runtime = None
            del app
            gc.collect()

    def test_startup_never_moves_and_missing_calibration_disables_tasks(self):
        self.assertFalse(any(call[0] in ("drive", "servo", "grab", "drop") for call in self.runtime.calls))
        self.assertEqual(str(self.app.grab_button["state"]), "disabled")
        self.assertEqual(str(self.app.drop_button["state"]), "disabled")

    def test_retrying_connection_can_be_cancelled_before_robot_is_connected(self):
        self.runtime.demo = False
        self.runtime.state.update(connected=False, connection_requested=True,
                                  connection_status="Retrying unavailable robot")
        self.root.after_cancel(self.app._after_id)
        self.app._refresh()
        self.assertEqual(str(self.app.connect_button["state"]), "disabled")
        self.assertEqual(str(self.app.disconnect_button["state"]), "normal")
        self.app.disconnect_button.invoke()
        self.assertIn(("disconnect",), self.runtime.calls)
        self.runtime.state["connection_requested"] = False
        self.root.after_cancel(self.app._after_id)
        self.app._refresh()
        self.assertEqual(str(self.app.connect_button["state"]), "normal")
        self.assertEqual(str(self.app.disconnect_button["state"]), "disabled")

    def test_mouse_and_key_sources_release_independently(self):
        self.app._set_source("forward", "mouse", True)
        self.app._set_source("forward", "key:w", True)
        self.app._release_mouse()
        drives = [call for call in self.runtime.calls if call[0] == "drive"]
        self.assertEqual(drives, [("drive", "forward", True)])
        self.app._key_release(SimpleNamespace(keysym="w"))
        self.assertEqual(self.runtime.calls[-1], ("drive", "forward", False))

    def test_keys_do_not_drive_or_start_tasks_while_typing(self):
        entry = tk.Entry(self.root)
        with patch.object(self.root, "focus_get", return_value=entry):
            for key in ("w", "s", "a", "d", "q", "e", "g", "h", "space"):
                self.app._key_press(SimpleNamespace(keysym=key))
        self.assertFalse(any(call[0] in ("drive", "grab", "drop", "stop") for call in self.runtime.calls))
        # Escape is always an emergency action, including while typing.
        with patch.object(self.root, "focus_get", return_value=entry):
            self.app._key_press(SimpleNamespace(keysym="Escape"))
        self.assertEqual(self.runtime.calls[-1], ("stop", "Operator stop"))

    def test_all_six_keyboard_directions_and_disabled_shortcuts(self):
        with patch.object(self.root, "focus_get", return_value=None):
            for key, action in (("w", "forward"), ("s", "back"), ("a", "left"),
                                ("d", "right"), ("q", "rotate_left"), ("e", "rotate_right")):
                self.app._key_press(SimpleNamespace(keysym=key))
                self.assertEqual(self.runtime.calls[-1], ("drive", action, True))
                self.app._key_release(SimpleNamespace(keysym=key))
                self.assertEqual(self.runtime.calls[-1], ("drive", action, False))
            self.app._key_press(SimpleNamespace(keysym="g"))
            self.app._key_press(SimpleNamespace(keysym="h"))
        self.assertFalse(any(call[0] in ("grab", "drop") for call in self.runtime.calls))

    def test_stop_clears_every_held_direction(self):
        self.app._set_source("forward", "key:w", True)
        self.app._set_source("rotate_left", "mouse", True)
        self.app._stop()
        self.assertFalse(any(self.app._drive_sources.values()))
        self.assertIn(("drive", "forward", False), self.runtime.calls)
        self.assertIn(("drive", "rotate_left", False), self.runtime.calls)
        self.assertEqual(self.runtime.calls[-1], ("stop", "Operator stop"))

    def test_entire_window_focus_loss_stops(self):
        self.app._set_source("left", "mouse", True)
        self.app._check_focus()
        self.assertEqual(self.runtime.calls[-1], ("stop", "Window lost focus"))

    def test_joint_rebuild_does_not_send_and_button_is_explicit(self):
        self.runtime.config.data["servos"] = [{"id": 7, "name": "Synthetic joint", "min": 430, "max": 560, "verified": True}]
        self.app._rebuild_servos()
        self.assertFalse(any(call[0] == "servo" for call in self.runtime.calls))
        widgets = self.app._servo_widgets[7]
        self.assertEqual(widgets["target"].get(), "")
        widgets["target"].set("500")
        self.app._send_servo(7, widgets["target"], tk.StringVar(self.root, value="1.5"))
        self.assertEqual(self.runtime.calls[-1], ("servo", 7, 500, 1.5))

    def test_render_all_boxes_using_packet_image(self):
        packet = DetectionPacket(4, time.monotonic(), time.monotonic(), Image.new("RGB", (640, 360)),
            (Detection(20, 20, 60, 80, .9), Detection(100, 50, 160, 130, .8)), True)
        self.app._render(packet, force=True)
        # Two box outlines + two label rectangles + the simulation banner.
        rectangles = [item for item in self.app.canvas.find_all() if self.app.canvas.type(item) == "rectangle"]
        self.assertEqual(len(rectangles), 5)

    def test_settings_failure_rolls_back_config(self):
        old = copy.deepcopy(self.runtime.config.data)
        def fail(): raise ValueError("disk unavailable")
        self.runtime.save_settings = fail
        with self.assertRaisesRegex(ValueError, "disk"):
            self.app._edit_config(lambda data: data["robot"].update(ip="changed"))
        self.assertEqual(self.runtime.config.data, old)

    def test_open_pose_requires_explicit_gripper_ids_and_passes_subset(self):
        self.app.pose_name.set("open")
        self.app.pose_joint_ids.set("")
        self.app._save_pose()
        self.assertFalse(any(call[0] == "pose" for call in self.runtime.calls))
        self.assertIn("gripper", self.app._last_message)
        self.app.pose_joint_ids.set("7, 8")
        self.app._save_pose()
        pose_calls = [call for call in self.runtime.calls if call[0] == "pose"]
        self.assertEqual(pose_calls[-1][-1], {"servo_ids": [7, 8]})

    def test_calibration_tabs_scroll_at_minimum_window_size(self):
        self.root.deiconify()
        self.root.geometry("1080x740")
        # The pose form is longer than the minimum-height viewport and must
        # remain reachable through the dedicated scrollbar.
        content = self.app.pose_tree.master
        canvas = content.master
        wrapper = canvas.master
        calibration_tabs = wrapper.master
        main_tabs = calibration_tabs.master.master
        main_tabs.select(calibration_tabs.master)
        calibration_tabs.select(wrapper)
        self.root.update()
        bounds = tuple(float(v) for v in canvas.cget("scrollregion").split())
        self.assertGreater(bounds[3], canvas.winfo_height())
        canvas.yview_moveto(1)
        self.assertGreater(canvas.yview()[0], 0)


if __name__ == "__main__":
    unittest.main()
