"""Camera reconnect regression checks; HTTP is fully mocked."""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from lastcan.camera import MjpegCamera
from lastcan.config import DEFAULT_CONFIG
from lastcan.control import FakeTransport
from lastcan.types import Detection, DetectionPacket, FramePacket


def jpeg(colour):
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(stream, format="JPEG")
    return stream.getvalue()


class MockStream:
    def __init__(self, data):
        self.data = data
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def read(self, size):
        time.sleep(.01)
        return b"" if self.closed else self.data

    read1 = read


class CameraLifecycleTests(unittest.TestCase):
    def test_rover_topic_keeps_slashes_and_stream_bypasses_system_proxy(self):
        payload = jpeg("green")
        requests = []

        class RoverVideoServer(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                # Reproduce the rover: an escaped topic returns no frames.
                body = payload if "topic=/depth_cam/rgb/image_raw&" in self.path else b""
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), RoverVideoServer)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["robot"].update(ip="127.0.0.1", video_port=server.server_port)
        frames = []
        received = threading.Event()

        def on_frame(frame):
            frames.append(frame)
            received.set()

        camera = None
        worker.start()
        try:
            with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1",
                                         "HTTP_PROXY": "http://127.0.0.1:1",
                                         "no_proxy": "", "NO_PROXY": ""}):
                camera = MjpegCamera(config, on_frame)
                camera.start()
                self.assertTrue(received.wait(2), camera.status)
                self.assertEqual(frames[0].image.size, (16, 16))
                self.assertIn("topic=/depth_cam/rgb/image_raw&", requests[0])
        finally:
            if camera:
                camera.stop()
            server.shutdown()
            server.server_close()
            worker.join(1)

    def test_old_inflight_connection_cannot_revive_after_robot_address_changes(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["robot"]["ip"] = "old-robot"
        frames, urls = [], []
        request_started, finish_old_request = threading.Event(), threading.Event()
        old_stream, new_stream = MockStream(jpeg("red")), MockStream(jpeg("blue"))

        def urlopen(url, timeout):
            urls.append(url)
            if "old-robot" in url:
                request_started.set()
                finish_old_request.wait(2)
                return old_stream
            return new_stream

        camera = MjpegCamera(config, frames.append)
        with patch.object(camera._opener, "open", urlopen):
            try:
                camera.start()
                self.assertTrue(request_started.wait(1))
                camera.stop()  # old urlopen still pending, so no response to close
                self.assertTrue(camera._thread.is_alive())
                config["robot"]["ip"] = "new-robot"
                camera.start()
                self.assertTrue(camera._stop.is_set(), "start revived the old reader's cancellation")
                finish_old_request.set()
                camera._thread.join(1)
                self.assertFalse(camera._thread.is_alive())
                self.assertEqual(frames, [], "Old robot emitted a frame after it was stopped")
                # Runtime retries start on its command loop once the old reader exits.
                camera.start()
                deadline = time.monotonic() + 1
                while not frames and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(frames)
                self.assertEqual(len(urls), 2)
                self.assertIn("old-robot", urls[0])
                self.assertIn("new-robot", urls[1])
                red, green, blue = frames[0].image.getpixel((0, 0))
                self.assertGreater(blue, red, "Displayed image came from the previous robot")
            finally:
                finish_old_request.set()
                camera.stop()


class InferenceSessionTests(unittest.TestCase):
    def test_session_clear_and_prediction_publication_are_atomic(self):
        from lastcan.runtime import Runtime
        at_publication, release_publication = threading.Event(), threading.Event()
        transition_done = threading.Event()

        class PassiveCamera:
            def __init__(self, *args): self.status = "Mocked camera"
            def start(self): pass
            def stop(self): pass

        class ImmediateDetector:
            def __init__(self, *args, **kwargs):
                self.available, self.status, self.confidence = True, "Mocked detector", .25
            def load(self): pass
            def predict(self, frame):
                return DetectionPacket(frame.frame_id, frame.captured_at, time.monotonic(),
                                       frame.image, (Detection(1, 1, 20, 30, .9),))

        def wait_for(predicate, timeout=2):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate(): return True
                time.sleep(.01)
            return bool(predicate())

        config_path = Path(__file__).resolve().parents[1] / ".cache" / "atomic-session-missing.json"
        with patch("lastcan.runtime.RosbridgeTransport", FakeTransport), \
             patch("lastcan.runtime.MjpegCamera", PassiveCamera), \
             patch("lastcan.vision.Detector", ImmediateDetector), \
             patch.object(socket, "socket", side_effect=AssertionError("Network is forbidden")):
            runtime = Runtime(demo=False, config_path=config_path)
            original_put = runtime._detections.put
            def paused_publication(value):
                if value is not None and value.frame_id == 41:
                    at_publication.set()
                    release_publication.wait(3)
                original_put(value)
            runtime._detections.put = paused_publication
            def reconnect():
                runtime.disconnect()
                wait_for(lambda: not runtime.transport.connected)
                runtime.connect()
                wait_for(lambda: runtime.transport.connected)
                transition_done.set()
            transition = threading.Thread(target=reconnect, daemon=True)
            try:
                runtime.connect()
                self.assertTrue(wait_for(lambda: runtime.transport.connected))
                runtime._frames.put(FramePacket(41, time.monotonic(), Image.new("RGB", (640, 360))))
                self.assertTrue(at_publication.wait(1))
                transition.start()
                # With an atomic guard, the transition waits for this publication
                # then clears it. Without one, it completes first and the old
                # publication wrongly appears after reconnect.
                transition_done.wait(.3)
                release_publication.set()
                self.assertTrue(transition_done.wait(3))
                time.sleep(.05)
                self.assertIsNone(runtime.get_detection(), "An old result was published after session clear")
            finally:
                release_publication.set()
                if transition.ident is not None:
                    transition.join(3)
                runtime.close()
                self.assertTrue(runtime.wait_closed())

    def test_old_inflight_prediction_is_discarded_after_disconnect_reconnect(self):
        from lastcan.runtime import Runtime
        entered, release_prediction = threading.Event(), threading.Event()

        class PassiveCamera:
            def __init__(self, *args):
                self.status = "Mocked camera; no HTTP requests"
            def start(self):
                pass
            def stop(self):
                pass

        class DelayedDetector:
            def __init__(self, *args, **kwargs):
                self.available = True
                self.status = "Mocked detector"
                self.confidence = .25
            def load(self):
                pass
            def predict(self, frame):
                if frame.frame_id == 41:
                    entered.set()
                    release_prediction.wait(3)
                return DetectionPacket(frame.frame_id, frame.captured_at, time.monotonic(),
                                       frame.image, (Detection(10, 10, 30, 60, .9),))

        def wait_for(predicate, timeout=2):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return True
                time.sleep(.01)
            return bool(predicate())

        config_path = Path(__file__).resolve().parents[1] / ".cache" / "inference-lifecycle-missing.json"
        with patch("lastcan.runtime.RosbridgeTransport", FakeTransport), \
             patch("lastcan.runtime.MjpegCamera", PassiveCamera), \
             patch("lastcan.vision.Detector", DelayedDetector), \
             patch.object(socket, "socket", side_effect=AssertionError("Network is forbidden")):
            runtime = Runtime(demo=False, config_path=config_path)
            try:
                runtime.connect()
                self.assertTrue(wait_for(lambda: runtime.transport.connected))
                old_session = runtime.transport.session_id
                runtime._frames.put(FramePacket(41, time.monotonic(), Image.new("RGB", (640, 360), "red")))
                self.assertTrue(entered.wait(1))
                runtime.disconnect()
                self.assertTrue(wait_for(lambda: not runtime.transport.connected))
                runtime.connect()
                self.assertTrue(wait_for(lambda: runtime.transport.connected))
                self.assertNotEqual(old_session, runtime.transport.session_id)
                release_prediction.set()
                time.sleep(.12)
                self.assertIsNone(runtime.get_detection(), "Old connection prediction reappeared after reconnect")
                runtime._frames.put(FramePacket(42, time.monotonic(), Image.new("RGB", (640, 360), "blue")))
                self.assertTrue(wait_for(lambda: runtime.get_detection() is not None))
                self.assertEqual(runtime.get_detection().frame_id, 42)
            finally:
                release_prediction.set()
                runtime.close()
                self.assertTrue(runtime.wait_closed())

    def test_stop_during_a_blocking_read_discards_the_returned_jpeg(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        frames = []
        reading, release_read = threading.Event(), threading.Event()

        class DelayedStream(MockStream):
            def read(self, size):
                reading.set()
                release_read.wait(2)
                return self.data
            read1 = read

        stream = DelayedStream(jpeg("red"))
        camera = MjpegCamera(config, frames.append)
        with patch.object(camera._opener, "open", return_value=stream):
            try:
                camera.start()
                self.assertTrue(reading.wait(1))
                camera.stop()
                release_read.set()
                camera._thread.join(1)
                self.assertEqual(frames, [])
            finally:
                release_read.set()
                camera.stop()


if __name__ == "__main__":
    unittest.main()
