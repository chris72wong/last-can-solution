"""Portable configuration. No real servo limits or poses ship pre-calibrated."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = {
    "robot": {
        "ip": "192.168.149.1", "rosbridge_port": 9090, "video_port": 8080,
        "camera_topic": "/depth_cam/rgb/image_raw",
        "cmd_vel_topic": "/cmd_vel", "cmd_vel_type": "geometry_msgs/msg/Twist",
        "servo_topic": "/ros_robot_controller/bus_servo/set_position",
        "servo_type": "ros_robot_controller_msgs/msg/ServosPosition",
        "topics_verified": False,
    },
    "motion": {"linear_speed": 0.12, "angular_speed": 0.35,
               "publish_hz": 10, "manual_lease_s": 0.4},
    "vision": {"model": "models/best.pt", "confidence": 0.25,
               "imgsz": 640, "max_frame_age_s": 1.0},
    "servos": [],
    "poses": {},
    "sequences": {
        "grab": ["home", "open", "reach", "grip", "lift", "carry"],
        "drop": ["release_horizontal", "open", "home"],
    },
    "alignment": {
        "calibrated": False, "target_x": None, "target_height": None,
        "x_tolerance": None, "height_tolerance": None,
        "turn_sign": None, "forward_sign": None,
        "turn_speed": 0.18, "forward_speed": 0.05,
        "pulse_s": 0.12, "settle_s": 0.25,
        "timeout_s": 25.0, "target_loss_s": 1.0,
        "max_frame_age_s": 1.0, "tracking_gate": 0.20,
        "stable_frames": 3,
    },
}


class ConfigStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else ROOT / "config" / "robot.json"
        self._lock = threading.RLock()
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8-sig"))
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(self.data.get(key), dict):
                    self.data[key].update(value)
                else:
                    self.data[key] = value

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)

    def resolve(self, path: str) -> Path:
        result = Path(path)
        return result if result.is_absolute() else ROOT / result
