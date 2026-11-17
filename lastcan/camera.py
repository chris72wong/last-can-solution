"""MJPEG capture and a completely offline, explicitly synthetic camera."""
from __future__ import annotations

import io
import math
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from .types import Detection, DetectionPacket, FramePacket


class JpegBuffer:
    """Bounded incremental parser, also handling split JPEG markers."""

    def __init__(self, limit: int = 4 * 1024 * 1024):
        self.buffer = b""
        self.limit = limit

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer += data
        images = []
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                self.buffer = self.buffer[-1:]
                break
            if start:
                self.buffer = self.buffer[start:]
            end = self.buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.buffer) > self.limit:
                    self.buffer = self.buffer[-1:]
                break
            images.append(self.buffer[:end + 2])
            self.buffer = self.buffer[end + 2:]
        return images


class MjpegCamera:
    def __init__(self, config: dict, on_frame: Callable[[FramePacket], None]):
        self.config = config
        self.on_frame = on_frame
        self.status = "Camera disconnected"
        self._stop = threading.Event()
        self._thread = None
        self._response = None
        self._frame_id = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()  # never revive a reader for an old IP/topic
        self._thread = threading.Thread(target=self._run, name="mjpeg-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        response = self._response
        if response:
            try:
                response.close()
            except OSError:
                pass
        if self._thread and self._thread.ident is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        self.status = "Camera disconnected"

    def _run(self) -> None:
        while not self._stop.is_set():
            robot = self.config["robot"].copy()
            url = (f"http://{robot['ip']}:{int(robot['video_port'])}/stream?" +
                   urllib.parse.urlencode({"topic": robot["camera_topic"], "type": "mjpeg", "quality": 70}))
            try:
                self.status = "Connecting camera..."
                with urllib.request.urlopen(url, timeout=3.0) as response:
                    self._response = response
                    parser = JpegBuffer()
                    while not self._stop.is_set():
                        read = getattr(response, "read1", response.read)
                        chunk = read(16384)
                        if self._stop.is_set():
                            break
                        if not chunk:
                            raise ConnectionError("Camera stream ended")
                        # Decode only the newest JPEG from this read.
                        images = parser.feed(chunk)
                        if images:
                            try:
                                image = Image.open(io.BytesIO(images[-1])).convert("RGB")
                                image.load()
                            except (OSError, ValueError):
                                continue
                            self._frame_id += 1
                            if self._stop.is_set():
                                break
                            self.on_frame(FramePacket(self._frame_id, time.monotonic(), image))
                            self.status = f"Live camera | {image.width} x {image.height}"
            except Exception as exc:
                if not self._stop.is_set():
                    self.status = f"Camera unavailable: {exc}"
                    self._stop.wait(1.0)
            finally:
                self._response = None


class DemoCamera:
    """Synthetic target moves in response to the fake transport, never a network."""

    def __init__(self, transport, on_packet: Callable[[DetectionPacket], None]):
        self.transport = transport
        self.on_packet = on_packet
        self.status = "SIMULATED camera | no robot connection"
        self._stop = threading.Event()
        self._thread = None
        self._command_index = 0
        self._velocity = (0.0, 0.0, 0.0)
        self.x, self.height = 0.70, 0.23
        try:
            self.font = ImageFont.truetype("segoeui.ttf", 13)
            self.big_font = ImageFont.truetype("segoeuib.ttf", 20)
        except OSError:
            self.font = self.big_font = ImageFont.load_default()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="synthetic-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.ident is not None:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        frame_id = 0
        previous = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            dt, previous = min(0.2, now - previous), now
            commands = list(self.transport.commands)
            for kind, args in commands[self._command_index:]:
                if kind == "velocity":
                    self._velocity = args
            self._command_index = len(commands)
            vx, vy, turn = self._velocity
            if not self.transport.connected:
                vx = vy = turn = 0.0
            self.x = max(0.12, min(0.88, self.x + turn * dt * 1.4 - vy * dt))
            self.height = max(0.12, min(0.65, self.height + vx * dt * 1.6))
            frame_id += 1
            image, boxes = self.render(now)
            self.on_packet(DetectionPacket(frame_id, now, time.monotonic(), image, boxes, simulated=True))
            self._stop.wait(0.07)

    def render(self, now: float) -> tuple[Image.Image, tuple[Detection, ...]]:
        image = Image.new("RGB", (640, 360), "#16222d")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 145, 640, 360), fill="#273644")
        draw.line((0, 145, 640, 145), fill="#50616f", width=2)
        for bottom in range(-320, 1000, 100):
            draw.line((320, 145, bottom, 360), fill="#354a59")
        for y in (165, 194, 235, 291, 355):
            draw.line((0, y, 640, y), fill="#354a59")
        draw.rounded_rectangle((26, 92, 124, 145), 5, fill="#21323d", outline="#527082")
        draw.text((42, 108), "LAB / 2087", font=self.font, fill="#7b9baa")
        draw.rectangle((482, 60, 599, 144), fill="#1d303e", outline="#416070")
        draw.line((540, 60, 540, 144), fill="#416070")
        boxes = []
        for cx, height, confidence in ((0.18, 0.16, 0.91), (self.x, self.height, 0.97)):
            h = height * 360
            w = h * 0.38
            bottom = 190 + height * 185
            x1, y1, x2, y2 = cx * 640 - w / 2, bottom - h, cx * 640 + w / 2, bottom
            draw.ellipse((x1 - 8, y2 - 3, x2 + 10, y2 + 9), fill="#19232c")
            draw.rounded_rectangle((x1, y1, x2, y2), radius=max(2, int(w * .15)), fill="#d5dee4")
            draw.rectangle((x1 + w * .5, y1 + h * .08, x2 - 1, y2 - h * .06), fill="#1763a0")
            draw.rectangle((x1 + 2, y1 + h * .38, x2 - 2, y1 + h * .64), fill="#e6b642")
            draw.ellipse((x1, y1 - 2, x2, y1 + max(4, h * .04)), fill="#b9c7d0", outline="#eff6fa")
            boxes.append(Detection(x1, y1, x2, y2, confidence))
        draw.rectangle((0, 0, 640, 39), fill="#352a13")
        draw.text((16, 9), "SIMULATED CAMERA / SYNTHETIC DETECTIONS", font=self.font, fill="#f7cf76")
        draw.text((18, 329), "Offline sandbox: no frames or commands reach a robot", font=self.font, fill="#a7bdcb")
        draw.line((320, 150, 320, 295), fill="#607b8c")
        draw.line((305, 221, 335, 221), fill="#607b8c")
        return image, tuple(boxes)
