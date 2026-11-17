"""Application facade: Tk callbacks never perform inference or network I/O."""
from __future__ import annotations

import copy
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time

from .camera import DemoCamera, MjpegCamera
from .config import ConfigStore, ROOT
from .control import FakeTransport, RobotController, RosbridgeTransport, TaskRunner
from .types import DetectionPacket, FramePacket, LatestValue, drive_axes


def demo_config() -> dict:
    """Invented simulation fixtures, never written into real robot.json."""
    from .config import DEFAULT_CONFIG
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["robot"]["topics_verified"] = True
    names = ("SIM base", "SIM shoulder", "SIM elbow", "SIM wrist", "SIM gripper")
    config["servos"] = [{"id": i + 1, "name": name, "min": 200, "max": 800, "verified": True}
                        for i, name in enumerate(names)]
    for name in ("home", "open", "reach", "grip", "lift", "carry", "release_horizontal"):
        positions = {str(i): 500 for i in range(1, 6)}
        if name in ("open", "reach"):
            positions["5"] = 700
        if name in ("grip", "lift", "carry", "release_horizontal"):
            positions["5"] = 400
        if name == "reach":
            positions["2"], positions["3"] = 620, 630
        if name == "release_horizontal":
            positions["4"] = 650
        if name in ("open", "grip"):
            positions = {"5": positions["5"]}  # open in place; do not swing the arm during release
        config["poses"][name] = {"positions": positions, "duration": 0.18,
                                 "pause": 0.02, "verified": True}
    config["alignment"].update(calibrated=True, target_x=0.5, target_height=0.40,
                               x_tolerance=0.04, height_tolerance=0.03,
                               turn_sign=1, forward_sign=1, stable_frames=2,
                               pulse_s=0.12, settle_s=0.12, timeout_s=35.0)
    return config


class Runtime:
    def __init__(self, demo: bool = False, config_path: str | Path | None = None,
                 model_path: str | None = None):
        self.demo = demo
        self.config = ConfigStore(config_path)
        if demo:
            self.config.data = demo_config()
        if model_path:
            self.config.data["vision"]["model"] = model_path
        validate_settings(self.config.data)
        self._saved = self.config.snapshot()
        self.message = "Simulation only. No robot connection." if demo else "Ready. Connect when the robot is prepared."
        self.transport = FakeTransport(self.config.data) if demo else RosbridgeTransport(self.config.data)
        if demo:
            self.transport.connect()  # in-memory operation; no socket exists
        self.controller = RobotController(self.transport, self.config.data)
        self._frames: LatestValue[FramePacket] = LatestValue()
        self._detections: LatestValue[DetectionPacket] = LatestValue()
        self._vision_generation = 0
        self.tasks = TaskRunner(self.controller, self.config.data, self.get_detection)
        self.camera = (DemoCamera(self.transport, self._on_demo_packet) if demo else
                       MjpegCamera(self.config.data, self._on_live_frame))
        self.detector = None
        self._detector_status = "Synthetic detections / no trained model" if demo else "Checking trained model..."
        self._actions: set[str] = set()
        self._lock = threading.RLock()
        self._pending_servos: dict[int, tuple[int, float, int]] = {}
        self._epoch = 0
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._shutdown_thread = None
        self._want_connected = demo
        self._connection_status = "SIMULATED connection" if demo else "Disconnected"
        self._last_connected = bool(self.transport.connected)
        self._last_attempt = 0.0
        self._io_thread = threading.Thread(target=self._io_loop, name="robot-command-loop", daemon=True)
        if demo:
            self.camera.start()
        self._io_thread.start()
        self._vision_thread = None
        if not demo:
            self._vision_thread = threading.Thread(target=self._vision_loop, name="detector", daemon=True)
            self._vision_thread.start()

    def _on_demo_packet(self, packet: DetectionPacket) -> None:
        self._frames.put(FramePacket(packet.frame_id, packet.captured_at, packet.image))
        self._detections.put(packet)

    def _on_live_frame(self, packet: FramePacket) -> None:
        with self._lock:
            if self._want_connected and self.transport.connected and not self._stop.is_set():
                self._frames.put(packet)

    def _clear_camera_session(self) -> None:
        with self._lock:
            self._vision_generation += 1
            self._frames.put(None)
            self._detections.put(None)

    def connect(self) -> None:
        self.stop("Connecting")
        self._want_connected = True
        self._last_attempt = 0.0
        self.message = "Connecting..." if not self.demo else "Simulation connected"

    def disconnect(self) -> None:
        self.stop("Disconnected")
        self._want_connected = False
        self._clear_camera_session()
        self.message = "Disconnected; reconnecting never resumes previous motion."

    def set_drive_action(self, action: str, pressed: bool) -> None:
        with self._lock:
            if not pressed:
                self._actions.discard(action)
            elif self.tasks.busy:
                self.message = "Manual driving is locked during a task. Press Stop to cancel."
            elif not self.transport.connected:
                self.message = "Connect before driving."
            elif not self.config.data["robot"].get("topics_verified"):
                self.message = "Verify the robot topics in Calibration before driving."
            else:
                self._actions.add(action)
        self.heartbeat()

    def heartbeat(self) -> None:
        with self._lock:
            actions = self._actions.copy()
        # No manual calls while automatic control owns the command stream.
        if not self.tasks.busy and self.transport.connected and self._want_connected:
            motion = self.config.data["motion"]
            self.controller.set_manual(*drive_axes(actions, float(motion["linear_speed"]),
                                                   float(motion["angular_speed"])))

    def stop(self, reason: str = "Stopped") -> None:
        with self._lock:
            self._actions.clear()
            self._pending_servos.clear()
            self._epoch += 1
        self.tasks.cancel(reason)
        self.controller.stop(reason, publish=False)
        self.message = reason + ". Already-issued servo moves may still finish."

    def grab(self) -> bool:
        if not self.demo and not (self.detector and self.detector.available):
            self.message = "Grab requires the trained can detector. Train/install models/best.pt and restart."
            return False
        if not self._want_connected or not self.transport.connected or self._stop.is_set():
            self.message = "Connect before starting Grab."
            return False
        with self._lock:
            self._actions.clear()
            self._pending_servos.clear()
            self._epoch += 1
        self.controller.clear_manual()
        result = self.tasks.start_grab()
        self.message = self.tasks.status
        return result

    def drop(self) -> bool:
        if not self._want_connected or not self.transport.connected or self._stop.is_set():
            self.message = "Connect before starting Drop."
            return False
        with self._lock:
            self._actions.clear()
            self._pending_servos.clear()
            self._epoch += 1
        self.controller.clear_manual()
        result = self.tasks.start_drop()
        self.message = self.tasks.status
        return result

    def set_servo(self, servo_id: int, position: int, duration: float = 1.0) -> None:
        self._require_idle()
        if not self.transport.connected:
            raise ValueError("Connect before commanding a servo.")
        if not self.config.data["robot"].get("topics_verified"):
            raise ValueError("Verify robot topics before commanding a servo.")
        servo = next((s for s in self.config.data["servos"] if s["id"] == int(servo_id)), None)
        if not servo or not servo.get("verified"):
            raise ValueError("Register and verify this servo first.")
        if not servo["min"] <= int(position) <= servo["max"]:
            raise ValueError("Position is outside this servo's verified safe range.")
        if not math.isfinite(float(duration)) or not 0.1 <= float(duration) <= 15:
            raise ValueError("Move duration must be 0.1 to 15 seconds; use 1–2 while learning.")
        with self._lock:
            self._actions.clear()
            self.controller.clear_manual()
            self._pending_servos[int(servo_id)] = (int(position), float(duration), self._epoch)

    def _io_loop(self) -> None:
        while not self._stop.is_set():
            try:
                now = time.monotonic()
                # Retire observations from the lost socket BEFORE opening a
                # replacement; a fast reconnect must not expose old detections.
                if self._last_connected and not self.transport.connected:
                    with self._lock:
                        self._actions.clear()
                        self._pending_servos.clear()
                        self._epoch += 1
                    self.tasks.cancel("Connection lost")
                    self.controller.stop("Connection lost", publish=False)
                    self._clear_camera_session()
                    self._last_connected = False
                    self._connection_status = "Connection lost; retrying"
                if self._want_connected and not self.transport.connected and now - self._last_attempt > 2:
                    self._last_attempt = now
                    self._connection_status = "Connecting..."
                    if self.transport.connect():
                        with self._lock:
                            self._actions.clear()
                            self._pending_servos.clear()
                        self.controller.stop("Connected: movement cleared")
                        self._last_connected = True
                        self._clear_camera_session()
                        self._connection_status = "Connected" if not self.demo else "SIMULATED connection"
                        self.camera.start()
                    else:
                        self._connection_status = "Unavailable: " + str(getattr(self.transport, "last_error", "connection failed"))
                if not self._want_connected and self.transport.connected:
                    self.controller.stop("Disconnected")
                    self.transport.close()
                    self.camera.stop()
                    self._frames.put(None)
                    self._detections.put(None)
                    self._connection_status = "Disconnected"
                connected = bool(self.transport.connected)
                if connected and self._want_connected and not self.demo:
                    # Recheck worker liveness after quick disconnect/reconnect.
                    # A prior reader may have exited just as start() returned.
                    self.camera.start()
                if connected != self._last_connected:
                    with self._lock:
                        self._actions.clear()
                        self._pending_servos.clear()
                        self._epoch += 1
                    self.tasks.cancel("Connection changed")
                    self.controller.stop("Connection changed")
                    self._last_connected = connected
                    self._clear_camera_session()
                    if not connected and self._want_connected:
                        self._connection_status = "Connection lost: " + str(getattr(self.transport, "last_error", "retrying"))
                with self._lock:
                    pending = dict(self._pending_servos)
                    self._pending_servos.clear()
                for servo_id, (position, duration, epoch) in pending.items():
                    if epoch == self._epoch and not self.tasks.busy and self._want_connected:
                        self.controller.move_manual_servo(servo_id, position, duration,
                            guard=lambda: epoch == self._epoch and not self.tasks.busy and self._want_connected)
                self.controller.tick()  # leases use the time AFTER any blocking connection/send
            except Exception as exc:
                self.message = f"Robot command failed: {exc}"
                self.tasks.cancel(str(exc))
                with self._lock:
                    self._actions.clear()
                    self._pending_servos.clear()
                try:
                    self.controller.stop(str(exc))
                except Exception:
                    pass
            try:
                period = 1 / max(1, float(self.config.data["motion"]["publish_hz"]))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                period = .1  # a rejected staged settings edit must not kill the publisher
            self._stop.wait(period)

    def _vision_loop(self) -> None:
        try:
            from .vision import Detector
            vision = self.config.data["vision"]
            self.detector = Detector(self.config.resolve(vision["model"]),
                                     imgsz=int(vision["imgsz"]), confidence=float(vision["confidence"]))
            self.detector.load()
            self._detector_status = self.detector.status
        except Exception as exc:
            self._detector_status = f"Detector unavailable: {exc}"
            return
        last_id = -1
        while not self._stop.is_set():
            with self._lock:
                generation = self._vision_generation
                frame = self._frames.get()
            if frame and frame.frame_id != last_id:
                last_id = frame.frame_id
                try:
                    packet = self.detector.predict(frame)
                    with self._lock:
                        if (not self._stop.is_set() and self._want_connected and
                                generation == self._vision_generation):
                            self._detections.put(packet)
                    self._detector_status = self.detector.status
                except Exception as exc:
                    self._detector_status = f"Prediction failed: {exc}"
            self._stop.wait(0.01)

    def get_detection(self) -> DetectionPacket | None:
        return self._detections.get()

    def get_display(self) -> DetectionPacket | FramePacket | None:
        frame, detection = self._frames.get(), self._detections.get()
        if detection and not detection.error and time.monotonic() - detection.captured_at <= float(self.config.data["vision"]["max_frame_age_s"]):
            return detection
        return frame

    def set_confidence(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or not 0.01 <= value <= 0.99:
            raise ValueError("Confidence must be 0.01 to 0.99.")
        self.config.data["vision"]["confidence"] = value
        if self.detector:
            self.detector.confidence = value

    def save_frame(self) -> Path | None:
        frame = self._frames.get()
        if not frame:
            raise ValueError("No camera frame is available yet.")
        folder = ROOT / "captures" / ("demo" if self.demo else "robot")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_f{frame.frame_id}.jpg")
        frame.image.save(path, quality=95)
        path.with_suffix(".json").write_text(json.dumps({"frame_id": frame.frame_id,
            "received_monotonic": frame.captured_at, "simulated": self.demo,
            "note": "Robot receipt time, not a synchronized sensor timestamp."}, indent=2), encoding="utf-8")
        self.message = f"Saved {path.name}"
        return path

    def _require_idle(self) -> None:
        if self.tasks.busy:
            raise ValueError("Stop the active task before changing calibration or arm commands.")

    def save_pose(self, name: str, duration: float, pause: float, verified: bool,
                  servo_ids: list[int] | None = None) -> None:
        self._require_idle()
        name = name.strip()
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Use letters, numbers, underscores, or hyphens for a pose name.")
        duration, pause = float(duration), float(pause)
        if not all(math.isfinite(v) for v in (duration, pause)) or not 0.1 <= duration <= 15 or not 0 <= pause <= 15:
            raise ValueError("Duration must be 0.1–15 seconds and pause 0–15 seconds.")
        configured = self.config.data["servos"]
        if name in ("open", "grip") and servo_ids is None:
            raise ValueError("For open/grip, select only the gripper joint IDs so the rest of the arm stays still.")
        if servo_ids is not None:
            selected = set(servo_ids)
            if not selected or not selected.issubset({s["id"] for s in configured}):
                raise ValueError("Choose IDs of configured joints, or leave blank for every joint.")
            servos = [s for s in configured if s["id"] in selected]
        else:
            servos = configured
        if name == "home" and len(servos) != len(configured):
            raise ValueError("The home pose must specify every configured joint.")
        commanded = dict(self.controller.commanded_positions)
        if not servos or any(not s.get("verified") or s["id"] not in commanded for s in servos):
            raise ValueError("Command every selected verified joint before saving its pose; positions are not measured.")
        if any(not s["min"] <= commanded[s["id"]] <= s["max"] for s in servos):
            raise ValueError("A last-commanded value is outside the updated limits; command the selected joints again.")
        positions = {str(s["id"]): int(commanded[s["id"]]) for s in servos}
        self.config.data["poses"][name] = {"positions": positions, "duration": duration,
                                            "pause": pause, "verified": bool(verified)}
        self._persist()
        self.message = f"Saved {'SIMULATED ' if self.demo else ''}pose '{name}' from commanded positions."

    def save_alignment(self, target_x: float, target_height: float, x_tolerance: float,
                       height_tolerance: float, turn_sign: int, forward_sign: int) -> None:
        self._require_idle()
        values = [float(v) for v in (target_x, target_height, x_tolerance, height_tolerance)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Alignment values must be finite.")
        if not .01 <= values[0] <= .99 or not .01 <= values[1] <= .95 or not all(.001 <= v <= .20 for v in values[2:]):
            raise ValueError("Use pickup x 0.01–0.99, height 0.01–0.95, and tolerances 0.001–0.20.")
        if int(turn_sign) not in (-1, 1) or int(forward_sign) not in (-1, 1):
            raise ValueError("Direction signs must be +1 or -1, verified with the wheels raised.")
        self.config.data["alignment"].update(calibrated=True, target_x=values[0], target_height=values[1],
            x_tolerance=values[2], height_tolerance=values[3], turn_sign=int(turn_sign), forward_sign=int(forward_sign))
        self._persist()
        self.message = "Saved alignment calibration" + (" (simulation only)" if self.demo else "")

    def save_settings(self) -> None:
        self._require_idle()
        validate_settings(self.config.data)
        old = {k: v for k, v in self._saved["robot"].items() if k != "topics_verified"}
        new = {k: v for k, v in self.config.data["robot"].items() if k != "topics_verified"}
        if self.transport.connected and old != new:
            raise ValueError("Disconnect before changing the robot address or topics.")
        if old != new:
            # Physical calibration belongs to one camera/robot. Reusing it for
            # a new endpoint without checking the arm is unsafe.
            for servo in self.config.data["servos"]:
                servo["verified"] = False
            for pose in self.config.data["poses"].values():
                pose["verified"] = False
            self.config.data["alignment"]["calibrated"] = False
            self._clear_camera_session()
        self.stop("Settings saved")
        self._persist()
        self.message = ("Robot changed: verify joints, poses, and alignment again" if old != new else "Settings saved")
        if self.demo:
            self.message += " in memory; real calibration untouched"

    def _persist(self) -> None:
        if not self.demo:
            self.config.save()
        self._saved = self.config.snapshot()

    def snapshot(self) -> dict:
        packet = self._detections.get()
        errors = self.tasks.calibration_errors("grab")
        return {"connected": bool(self.transport.connected), "connection_requested": self._want_connected,
                "connection_status": self._connection_status,
                "camera_status": self.camera.status, "detector_status": self._detector_status,
                "task_status": self.tasks.status, "task_error": self.tasks.error,
                "busy": self.tasks.busy, "owner": self.controller.owner,
                "commanded_positions": dict(self.controller.commanded_positions),
                "calibrated": not errors, "grab_errors": errors,
                "drop_errors": self.tasks.calibration_errors("drop"),
                "model_available": self.demo or bool(self.detector and self.detector.available),
                "detection_count": len(packet.boxes) if packet and not packet.error else 0}

    def close(self) -> None:
        if self._shutdown_thread is not None:
            return
        self._want_connected = False
        self.stop("Closing")
        self._stop.set()
        self._shutdown_thread = threading.Thread(target=self._shutdown, name="robot-shutdown", daemon=False)
        self._shutdown_thread.start()

    def _shutdown(self) -> None:
        try:
            self.tasks.join(2.0)
            self._io_thread.join(timeout=4.0)
            # The periodic loop is now stopped: explicitly flush the final zero
            # BEFORE closing the socket, even when no arm task was running.
            self.controller.stop("Closing", publish=True)
            self.transport.close()
            self.camera.stop()
            if self._vision_thread:
                self._vision_thread.join()  # outside Tk; finish an in-flight local model load/prediction
        finally:
            self._closed.set()

    def wait_closed(self, timeout: float | None = 8.0) -> bool:
        """Wait outside Tk's event loop (entry point and automated checks)."""
        return self._closed.wait(timeout)


def validate_settings(config: dict) -> None:
    robot = config["robot"]
    if not str(robot["ip"]).strip() or any(c in str(robot["ip"]) for c in "/\\ ?#"):
        raise ValueError("Enter a robot hostname or IP address, without a URL.")
    for key in ("rosbridge_port", "video_port"):
        if not 1 <= int(robot[key]) <= 65535:
            raise ValueError("Ports must be between 1 and 65535.")
    for key in ("camera_topic", "cmd_vel_topic", "servo_topic"):
        if not str(robot[key]).startswith("/") or any(c.isspace() for c in str(robot[key])):
            raise ValueError("Topics must start with '/' and contain no spaces.")
    ids = set()
    for servo in config["servos"]:
        sid = int(servo["id"])
        if sid in ids or not 1 <= sid <= 253:
            raise ValueError("Servo IDs must be unique bus IDs 1–253, verified on the robot.")
        ids.add(sid)
        if not str(servo["name"]).strip() or not 0 <= int(servo["min"]) < int(servo["max"]) <= 1000:
            raise ValueError("Each servo needs a name and a measured safe range within 0–1000.")
    motion = config["motion"]
    for key in ("linear_speed", "angular_speed", "publish_hz", "manual_lease_s"):
        if not math.isfinite(float(motion[key])) or float(motion[key]) <= 0:
            raise ValueError("Motion settings must be finite and positive.")
