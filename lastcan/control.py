"""Robot transport, exclusive wheel ownership, and cancellable calibrated tasks.

This module never connects at import or construction. FakeTransport is the
only transport used by offline tests/demo. Servo positions below are accepted
only from explicitly verified user calibration; there are no real arm presets.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from typing import Callable

from .types import Detection, DetectionPacket


class ControlError(RuntimeError):
    """A command cannot safely proceed with current connection/calibration."""


class TaskCancelled(ControlError):
    pass


def _number(value, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ControlError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ControlError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ControlError(f"{name} must be at most {maximum}")
    return result


class FakeTransport:
    """In-memory transport. It cannot create sockets or reach the robot."""
    is_fake = True

    def __init__(self, config: dict):
        self.config = config
        self.connected = False
        self.session_id = 0
        self.last_error = ""
        self.commands: list[tuple] = []
        self._lock = threading.RLock()

    def connect(self) -> bool:
        with self._lock:
            self.session_id += 1
            self.connected = True
            self.last_error = ""
        return True

    def close(self) -> None:
        with self._lock:
            self.connected = False

    def publish_velocity(self, lx: float, ly: float, az: float) -> None:
        with self._lock:
            if not self.connected:
                raise ControlError("Fake robot is disconnected")
            self.commands.append(("velocity", (float(lx), float(ly), float(az))))

    def move_servos(self, positions: dict[int, int], duration: float) -> None:
        with self._lock:
            if not self.connected:
                raise ControlError("Fake robot is disconnected")
            self.commands.append(("servos", (dict(positions), float(duration))))


class RosbridgeTransport:
    """One shared rosbridge socket with a receive worker and serialized writes."""
    is_fake = False

    def __init__(self, config: dict):
        self.config = config
        self.connected = False
        self.session_id = 0
        self.last_error = "Not connected"
        self._socket = None
        self._lock = threading.RLock()
        self._reader = None

    def connect(self) -> bool:
        self.close()
        try:
            import websocket
            robot = self.config["robot"]
            ws = websocket.create_connection(
                f"ws://{robot['ip']}:{int(robot['rosbridge_port'])}", timeout=2,
                enable_multithread=True)
            ws.settimeout(0.5)
            with self._lock:
                self._socket = ws
                self.session_id += 1
                session = self.session_id
                self.connected = True
                for topic, message_type in ((robot["cmd_vel_topic"], robot["cmd_vel_type"]),
                                             (robot["servo_topic"], robot["servo_type"])):
                    self._send({"op": "advertise", "topic": topic, "type": message_type})
                self.last_error = ""
            self._reader = threading.Thread(target=self._receive,
                                            args=(ws, session), daemon=True,
                                            name="rosbridge-receive")
            self._reader.start()
            return True
        except Exception as exc:
            self.last_error = f"Cannot connect: {exc}"
            self.close()
            return False

    def _receive(self, ws, session: int) -> None:
        import websocket
        while self.connected and self.session_id == session:
            try:
                message = ws.recv()
                if not message:
                    raise ConnectionError("rosbridge closed the connection")
                # Consume status/service responses so the socket cannot fill.
                # Advertise/publish errors are reported instead of ignored.
                try:
                    status = json.loads(message)
                    if status.get("op") == "status" and status.get("level") == "error":
                        raise ConnectionError(status.get("msg", "rosbridge error"))
                except (ValueError, TypeError):
                    pass
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                with self._lock:
                    if self.session_id == session:
                        self.last_error = f"Connection lost: {exc}"
                        self.connected = False
                break

    def _send(self, message: dict) -> None:
        with self._lock:
            if not self.connected or self._socket is None:
                raise ControlError("Robot is disconnected")
            try:
                self._socket.send(json.dumps(message, allow_nan=False))
            except Exception as exc:
                self.connected = False
                self.last_error = f"Send failed: {exc}"
                raise ControlError(self.last_error) from exc

    def publish_velocity(self, lx: float, ly: float, az: float) -> None:
        self._send({"op": "publish", "topic": self.config["robot"]["cmd_vel_topic"],
                    "msg": {"linear": {"x": lx, "y": ly, "z": 0.0},
                            "angular": {"x": 0.0, "y": 0.0, "z": az}}})

    def move_servos(self, positions: dict[int, int], duration: float) -> None:
        self._send({"op": "publish", "topic": self.config["robot"]["servo_topic"],
                    "msg": {"duration": duration,
                            "position": [{"id": int(sid), "position": int(position)}
                                         for sid, position in positions.items()]}})

    def close(self) -> None:
        with self._lock:
            self.connected = False
            self.session_id += 1
            ws, self._socket = self._socket, None
        if ws is not None:
            try:
                ws.close(timeout=0.5)
            except Exception:
                pass


class RobotController:
    """Serializes physical commands and grants exactly one wheel owner.

    A reconnect disarms old manual input until a zero-input heartbeat is seen.
    Both manual and automatic wheel intents expire, so a dead worker cannot
    leave its last nonzero intent being published indefinitely.
    """

    def __init__(self, transport, config: dict):
        self.transport = transport
        self.config = config
        self.owner = "stopped"
        self.commanded_positions: dict[int, int] = {}
        self.last_error = ""
        self.stop_reason = "Stopped"
        self._lock = threading.RLock()
        self._halt_requested = threading.Event()
        self._velocity = (0.0, 0.0, 0.0)
        self._expires = 0.0
        self._connection = self._connection_identity()
        self._manual_armed = bool(transport.connected)

    def _connection_identity(self):
        return (bool(self.transport.connected), getattr(self.transport, "session_id", 0))

    def _observe_connection(self) -> bool:
        current = self._connection_identity()
        if current != self._connection:
            self._connection = current
            self._velocity = (0.0, 0.0, 0.0)
            self._expires = 0.0
            self.owner = "stopped"
            self._manual_armed = False
            self.commanded_positions.clear()  # no state feedback; reconnect means unknown
            self.stop_reason = "Connection changed; release driving controls"
        return current[0]

    def _topics_ready(self) -> bool:
        return self.config.get("robot", {}).get("topics_verified") is True

    def _apply_requested_stop(self) -> None:
        """Apply a nonblocking GUI stop after an in-flight write releases its lock."""
        if self._halt_requested.is_set():
            self._halt_requested.clear()
            self.owner = "stopped"
            self._velocity = (0.0, 0.0, 0.0)
            self._expires = 0.0
            self._manual_armed = False

    def set_manual(self, lx: float, ly: float, az: float) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._apply_requested_stop()
            connected = self._observe_connection()
            if self.owner == "task":
                return False
            values = self._checked_velocity(lx, ly, az)
            if not any(values):
                self._manual_armed = connected
                self._velocity = values
                self._expires = 0.0
                self.owner = "stopped"
                return connected
            if not connected or not self._topics_ready() or not self._manual_armed:
                return False
            self.owner = "manual"
            self._velocity = values
            self._expires = time.monotonic() + _number(
                self.config["motion"].get("manual_lease_s", 0.4), "manual lease", 0.05, 2.0)
            return True
        finally:
            self._lock.release()

    def _checked_velocity(self, lx: float, ly: float, az: float):
        linear = _number(self.config["motion"]["linear_speed"], "linear speed", 0, 0.3)
        angular = _number(self.config["motion"]["angular_speed"], "angular speed", 0, 1.0)
        return tuple(max(-limit, min(limit, _number(value, "velocity")))
                     for value, limit in ((lx, linear), (ly, linear), (az, angular)))

    def clear_manual(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._apply_requested_stop()
            if self.owner != "task":
                self._velocity = (0.0, 0.0, 0.0)
                self._expires = 0.0
                self.owner = "stopped"
                self._manual_armed = bool(self.transport.connected)
        finally:
            self._lock.release()

    def stop(self, reason: str = "Stopped", publish: bool = True) -> None:
        self.stop_reason = reason
        self._halt_requested.set()
        # GUI cancellation must not wait behind a network write. The command
        # worker observes the event before sending any further nonzero intent.
        if not self._lock.acquire(blocking=publish):
            return
        try:
            self._apply_requested_stop()
            self._observe_connection()
            if publish:
                self._publish((0.0, 0.0, 0.0))
        finally:
            self._lock.release()

    def _publish(self, velocity) -> bool:
        if self._halt_requested.is_set():
            self._apply_requested_stop()
            velocity = (0.0, 0.0, 0.0)
        if not self.transport.connected:
            return False
        try:
            self.transport.publish_velocity(*velocity)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._observe_connection()
            return False

    def tick(self, now: float | None = None) -> None:
        with self._lock:
            self._apply_requested_stop()
            if not self._observe_connection():
                return
            now = time.monotonic() if now is None else now
            if not self._topics_ready() or now >= self._expires:
                self._velocity = (0.0, 0.0, 0.0)
                if self.owner == "manual":
                    self.owner = "stopped"
                    self._manual_armed = False
            self._publish(self._velocity)

    def acquire(self, owner: str = "task") -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._apply_requested_stop()
            if owner != "task":
                raise ValueError("Only task ownership may be acquired")
            if not self._observe_connection() or not self._topics_ready() or self.owner == "task":
                return False
            self.owner = "task"
            self._velocity = (0.0, 0.0, 0.0)
            self._expires = 0.0
            self._manual_armed = False
            # Acquiring ownership is CPU-only and may be called by tkinter.
            # The task worker publishes zero before issuing any arm command.
            return True
        finally:
            self._lock.release()

    def set_automatic(self, lx: float, ly: float, az: float, lease_s: float = 0.4) -> None:
        with self._lock:
            self._apply_requested_stop()
            if not self._observe_connection() or self.owner != "task":
                raise ControlError("Automatic movement no longer owns the robot")
            if not self._topics_ready():
                raise ControlError("Robot topics have not been verified")
            self._velocity = self._checked_velocity(lx, ly, az)
            self._expires = time.monotonic() + _number(lease_s, "automatic lease", 0.01, 2.0)
            if not self._publish(self._velocity):
                raise ControlError(self.last_error or "Robot disconnected")

    def release(self, owner: str = "task") -> None:
        with self._lock:
            if self.owner == owner:
                self.stop("Task finished; release driving controls")
            elif self.owner == "stopped":
                # Cancellation may already have revoked ownership without
                # network I/O; the finishing worker also flushes its zero.
                self._publish((0.0, 0.0, 0.0))

    def servo_definitions(self) -> dict[int, dict]:
        definitions = {}
        servos = self.config.get("servos", [])
        if not isinstance(servos, list):
            raise ControlError("Servo configuration must be a list")
        for servo in servos:
            if not isinstance(servo, dict):
                raise ControlError("Each servo definition must be an object")
            sid = servo.get("id")
            if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1 or sid > 253:
                raise ControlError("Each servo needs a verified integer ID from 1 to 253")
            if sid in definitions:
                raise ControlError(f"Duplicate servo ID {sid}")
            if servo.get("verified") is not True:
                raise ControlError(f"Servo {sid} has not been verified")
            low = _number(servo.get("min"), f"Servo {sid} minimum", 0, 1000)
            high = _number(servo.get("max"), f"Servo {sid} maximum", 0, 1000)
            if not low.is_integer() or not high.is_integer() or low >= high:
                raise ControlError(f"Servo {sid} needs integer safe limits with minimum < maximum")
            definitions[sid] = servo
        if not definitions:
            raise ControlError("No verified arm servos configured")
        return definitions

    def validate_pose(self, positions: dict, duration: float) -> dict[int, int]:
        _number(duration, "Servo duration", 0.05, 30)
        definitions = self.servo_definitions()
        if not isinstance(positions, dict) or not positions:
            raise ControlError("A pose needs at least one commanded servo position")
        result = {}
        for key, value in positions.items():
            try:
                sid = int(key)
            except (ValueError, TypeError) as exc:
                raise ControlError(f"Invalid servo ID {key}") from exc
            if isinstance(key, bool) or str(sid) != str(key):
                raise ControlError(f"Invalid servo ID {key}")
            if sid not in definitions:
                raise ControlError(f"Servo {sid} is not configured and verified")
            position = _number(value, f"Servo {sid} position")
            servo = definitions[sid]
            if not position.is_integer() or not servo["min"] <= position <= servo["max"]:
                raise ControlError(f"Servo {sid} position is outside verified limits")
            result[sid] = int(position)
        return result

    def _move(self, positions: dict, duration: float, task: bool,
              guard: Callable[[], bool] | None = None) -> None:
        with self._lock:
            self._apply_requested_stop()
            if guard is not None and not guard():
                raise ControlError("Queued servo command was cancelled")
            if not self._observe_connection() or not self._topics_ready():
                raise ControlError("Connect and verify robot topics before moving servos")
            if (self.owner == "task") != task:
                raise ControlError("Arm is owned by an active task" if not task else "Task was stopped")
            checked = self.validate_pose(positions, duration)
            if not task:
                self.stop("Manual servo movement")
            if self._halt_requested.is_set():
                self._apply_requested_stop()
                raise ControlError("Servo command cancelled before sending")
            if guard is not None and not guard():
                raise ControlError("Queued servo command was cancelled before sending")
            try:
                self.transport.move_servos(checked, float(duration))
            except Exception as exc:
                self._observe_connection()
                raise ControlError(str(exc)) from exc
            self.commanded_positions.update(checked)

    def move_manual_servo(self, sid: int, position: int, duration: float = 1.0,
                          guard: Callable[[], bool] | None = None) -> None:
        self._move({sid: position}, duration, task=False, guard=guard)

    def move_task_servos(self, positions: dict, duration: float) -> None:
        self._move(positions, duration, task=True)


class TaskRunner:
    """Runs one fully prevalidated task at a time, outside the GUI thread."""

    def __init__(self, controller: RobotController, config: dict,
                 get_detection: Callable[[], DetectionPacket | None]):
        self.controller = controller
        self.config = config
        self.get_detection = get_detection
        self.busy = False
        self.status = "Idle"
        self.error = ""
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._cancel_reason = "Cancelled"
        self._thread = None
        self._task_config = None
        self._task_session = None
        self._initial_target = None

    def calibration_errors(self, kind: str = "grab") -> list[str]:
        try:
            self._validate(kind, self.config)
            return []
        except (ControlError, KeyError, TypeError, ValueError) as exc:
            return [str(exc)]

    def _validate(self, kind: str, config: dict) -> list[tuple[str, dict]]:
        if config.get("robot", {}).get("topics_verified") is not True:
            raise ControlError("Verify the robot topic names and message types first")
        definitions = self.controller.servo_definitions()
        sequence = config.get("sequences", {}).get(kind)
        if not isinstance(sequence, list) or not sequence or not all(isinstance(n, str) for n in sequence):
            raise ControlError(f"Configure the {kind} pose sequence")
        required = (["home", "open", "reach", "grip", "lift", "carry"] if kind == "grab"
                    else ["release_horizontal", "open", "home"])
        try:
            indices = [sequence.index(name) for name in required]
        except ValueError as exc:
            raise ControlError(f"{kind.title()} requires poses: {', '.join(required)}") from exc
        if indices != sorted(indices) or (kind == "grab" and sequence[0] != "home"):
            raise ControlError(f"The {kind} poses must run in order: {', '.join(required)}")
        steps = []
        for name in sequence:
            pose = config.get("poses", {}).get(name)
            if not isinstance(pose, dict) or pose.get("verified") is not True:
                raise ControlError(f"Record and verify the '{name}' pose first")
            self.controller.validate_pose(pose.get("positions"), pose.get("duration"))
            _number(pose.get("pause", 0.0), f"Pause for {name}", 0, 30)
            if name == "home" and {int(s) for s in pose["positions"]} != set(definitions):
                raise ControlError("The home pose must specify every verified arm servo")
            steps.append((name, copy.deepcopy(pose)))
        if kind == "grab":
            a = config.get("alignment", {})
            if a.get("calibrated") is not True:
                raise ControlError("Calibrate visual pickup alignment with the arm in home/open pose")
            _number(a.get("target_x"), "Pickup target x", 0.01, 0.99)
            _number(a.get("target_height"), "Pickup target height", 0.01, 0.95)
            _number(a.get("x_tolerance"), "Pickup x tolerance", 0.001, 0.20)
            _number(a.get("height_tolerance"), "Pickup height tolerance", 0.001, 0.20)
            for sign in ("turn_sign", "forward_sign"):
                if a.get(sign) not in (-1, 1) or isinstance(a.get(sign), bool):
                    raise ControlError(f"Verify {sign}: it must be -1 or 1")
            for name, low, high in (("turn_speed", 0.01, 0.5), ("forward_speed", 0.01, 0.15),
                                    ("pulse_s", 0.02, 0.5), ("settle_s", 0.02, 2),
                                    ("timeout_s", 0.1, 120), ("target_loss_s", 0.05, 5),
                                    ("max_frame_age_s", 0.05, 2), ("tracking_gate", 0.01, 0.5)):
                _number(a.get(name), name, low, high)
            stable = _number(a.get("stable_frames", 3), "Stable frames", 1, 10)
            if not stable.is_integer():
                raise ControlError("Stable frames must be an integer")
        return steps

    def start_grab(self) -> bool:
        return self._start("grab")

    def start_drop(self) -> bool:
        return self._start("drop")

    def _start(self, kind: str) -> bool:
        with self._lock:
            if self.busy:
                self.error = "Another grab/drop routine is already running"
                return False
            try:
                config = copy.deepcopy(self.config)
                steps = self._validate(kind, config)
                if kind == "grab":
                    packet = self.get_detection()
                    if (packet is None or packet.error or not packet.boxes or
                            not -0.05 <= time.monotonic() - packet.captured_at <=
                            config["alignment"]["max_frame_age_s"]):
                        raise ControlError("Grab requires a fresh visible target detection")
                    if packet.simulated and not getattr(self.controller.transport, "is_fake", False):
                        raise ControlError("Simulated detections cannot control the real robot")
                    size = getattr(packet.image, "size", (0, 0))
                    if not isinstance(size, tuple) or len(size) != 2 or min(size) <= 0:
                        raise ControlError("Detection frame has no valid image size")
                    selected = self._select(packet.boxes, size, None, config["alignment"]["tracking_gate"])
                    if selected is None:
                        raise ControlError("Grab requires a valid target detection")
                    self._initial_target = selected[1]
                if not self.controller.acquire("task"):
                    raise ControlError("Robot is disconnected or already owned by another task")
            except (ControlError, KeyError, TypeError, ValueError) as exc:
                self.error = str(exc)
                self.status = f"Cannot {kind}: {self.error}"
                return False
            self._task_config = config
            self._task_session = getattr(self.controller.transport, "session_id", 0)
            self._cancel.clear()
            self.error = ""
            self.status = f"Preparing {kind}"
            self.busy = True
            self._thread = threading.Thread(target=self._run, args=(kind, steps), daemon=True,
                                            name=f"robot-{kind}")
            self._thread.start()
            return True

    def cancel(self, reason: str = "Cancelled") -> None:
        with self._lock:
            self._cancel_reason = reason
            self._cancel.set()
            # CPU-only: the periodic publisher or finishing task worker sends
            # zero. Do not perform socket I/O inside a tkinter callback.
            self.controller.stop(reason, publish=False)

    def join(self, timeout: float = 1.0) -> None:
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _check(self) -> None:
        if self._cancel.is_set():
            raise TaskCancelled(self._cancel_reason)
        if (not self.controller.transport.connected or
                getattr(self.controller.transport, "session_id", 0) != self._task_session):
            raise ControlError("Robot disconnected or its connection changed during the routine")
        if self.controller.owner != "task":
            raise TaskCancelled("Routine stopped; task no longer owns the robot")

    def _wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check()
            self._cancel.wait(min(0.02, max(0, end - time.monotonic())))
        self._check()

    def _play(self, steps: list[tuple[str, dict]], kind: str) -> None:
        for name, pose in steps:
            self._check()
            self.status = f"{kind.title()}: {name}"
            self.controller.move_task_servos(pose["positions"], pose["duration"])
            self._wait(pose["duration"] + pose.get("pause", 0.0))

    def _run(self, kind: str, steps: list[tuple[str, dict]]) -> None:
        try:
            self._check()
            self.controller.set_automatic(0, 0, 0)
            if kind == "grab":
                # Home/open before aligning, because a camera mounted on the
                # arm changes its geometry when the arm changes pose.
                split = next(i for i, (name, _) in enumerate(steps) if name == "reach")
                self._play(steps[:split], kind)
                self._align(min_capture_at=time.monotonic())
                self._play(steps[split:], kind)
            else:
                self._play(steps, kind)
            self.status = ("Grab sequence complete; verify secure grip" if kind == "grab"
                           else "Drop sequence complete; verify horizontal release")
            self.error = ""
        except TaskCancelled as exc:
            self.status = f"Cancelled: {exc}"
            self.error = str(exc)
        except Exception as exc:
            self.status = f"{kind.title()} failed: {exc}"
            self.error = str(exc)
        finally:
            self.controller.release("task")
            with self._lock:
                self.busy = False

    @staticmethod
    def _normal(box: Detection, size: tuple[int, int]) -> tuple[float, float, float, float]:
        w, h = size
        return (box.center[0] / w, box.center[1] / h,
                (box.x2 - box.x1) / w, (box.y2 - box.y1) / h)

    @classmethod
    def _select(cls, boxes: tuple[Detection, ...], size: tuple[int, int], previous,
                gate: float) -> tuple[Detection, tuple[float, float, float, float]] | None:
        valid = [b for b in boxes if b.area > 0 and all(math.isfinite(v) for v in
                 (b.x1, b.y1, b.x2, b.y2, b.confidence))]
        if not valid:
            return None
        if previous is None:
            chosen = max(valid, key=lambda b: (b.area, b.confidence))
            return chosen, cls._normal(chosen, size)
        candidates = []
        for box in valid:
            current = cls._normal(box, size)
            distance = math.hypot(current[0] - previous[0], current[1] - previous[1])
            ratio = current[3] / max(previous[3], 0.001)
            if distance <= gate and 0.4 <= ratio <= 2.5:
                candidates.append((distance, box, current))
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[0])
        if len(candidates) > 1 and candidates[1][0] <= candidates[0][0] + 0.015:
            raise ControlError("Target identity is ambiguous; separate the cans and retry")
        return candidates[0][1], candidates[0][2]

    def _align(self, min_capture_at: float) -> None:
        self.status = "Aligning to selected can"
        a = self._task_config["alignment"]
        start = last_seen = time.monotonic()
        last_frame_id = None
        # Retain the can selected on the Grab press across the preparation
        # poses. If the camera shifts too far, fail rather than select another.
        previous = self._initial_target
        stable = 0
        last_problem = "No fresh target detection"
        while True:
            self._check()
            now = time.monotonic()
            if now - start > a["timeout_s"]:
                raise ControlError("Alignment timed out; reposition the robot and retry")
            if now - last_seen > a["target_loss_s"]:
                raise ControlError(f"Target lost: {last_problem}")
            packet = self.get_detection()
            if packet is None or packet.frame_id == last_frame_id:
                self._wait(0.02)
                continue
            last_frame_id = packet.frame_id
            if packet.simulated and not getattr(self.controller.transport, "is_fake", False):
                raise ControlError("Simulated detections cannot control the real robot")
            age = now - packet.captured_at
            if packet.error or age < -0.05 or age > a["max_frame_age_s"]:
                last_problem = packet.error or "Camera frame is stale"
                stable = 0
                self._wait(0.02)
                continue
            if packet.captured_at < min_capture_at:
                last_problem = "Waiting for a frame captured after the last movement"
                self._wait(0.02)
                continue
            size = getattr(packet.image, "size", (0, 0))
            if not isinstance(size, tuple) or len(size) != 2 or min(size) <= 0:
                raise ControlError("Detection frame has no valid image size")
            selected = self._select(packet.boxes, size, previous, a["tracking_gate"])
            if selected is None:
                last_problem = "Selected can no longer matches a detection"
                stable = 0
                self._wait(0.02)
                continue
            _, previous = selected
            last_seen = now
            x, _, _, height = previous
            x_error = x - a["target_x"]
            height_error = a["target_height"] - height
            if abs(x_error) <= a["x_tolerance"] and abs(height_error) <= a["height_tolerance"]:
                stable += 1
                self.controller.set_automatic(0, 0, 0)
                if stable >= int(a.get("stable_frames", 3)):
                    return
                self._wait(0.02)
                continue
            stable = 0
            if height_error < -a["height_tolerance"]:
                raise ControlError("Can is closer than the calibrated pickup position; reposition and retry")
            if abs(x_error) > a["x_tolerance"]:
                velocity = (0, 0, -math.copysign(a["turn_speed"], x_error) * a["turn_sign"])
            else:
                velocity = (a["forward_speed"] * a["forward_sign"], 0, 0)
            self.controller.set_automatic(*velocity, lease_s=a["pulse_s"] + 0.1)
            self._wait(a["pulse_s"])
            self.controller.set_automatic(0, 0, 0)
            self._wait(a["settle_s"])
            # A post-settle fresh observation is mandatory before another pulse.
            min_capture_at = time.monotonic()
