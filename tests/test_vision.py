import importlib
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
from PIL import Image

from lastcan.types import FramePacket
from lastcan.vision import Detector


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parent.parent / ".cache" / ("vision-test-" + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.frame = FramePacket(27, 123.5, Image.new("RGB", (640, 360), (255, 0, 0)))

    def tearDown(self):
        cache = (Path(__file__).resolve().parent.parent / ".cache").resolve()
        target = self.folder.resolve()
        if not target.is_relative_to(cache) or target == cache:
            raise AssertionError(f"Refusing recursive cleanup outside this test's cache: {target}")
        shutil.rmtree(target)

    def ready_detector(self, corners, confidence):
        detector = Detector(self.folder / "local.pt")
        detector.available = detector._attempted = True
        result = SimpleNamespace(boxes=SimpleNamespace(
            xyxy=SimpleNamespace(tolist=lambda: corners),
            conf=SimpleNamespace(tolist=lambda: confidence)))
        detector._model = SimpleNamespace(predict=Mock(return_value=[result]))
        return detector

    def test_missing_weights_never_import_or_download_yolo(self):
        detector = Detector(self.folder / "missing.pt")
        with patch.dict(sys.modules, {"ultralytics": None, "torch": None}):
            self.assertFalse(detector.load())
            self.assertFalse(detector.load())
            result = detector.predict(self.frame)
        self.assertEqual(result.boxes, ())
        self.assertIn("Missing trained weights", result.error)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_rgb_to_bgr_and_matching_source_identity(self):
        detector = self.ready_detector([[10, 20, 50, 80]], [.95])
        result = detector.predict(self.frame)
        source = detector._model.predict.call_args.args[0]
        self.assertEqual(source[0, 0].tolist(), [0, 0, 255])
        self.assertTrue(source.flags.c_contiguous)
        self.assertIs(result.image, self.frame.image)
        self.assertEqual((result.frame_id, result.captured_at), (27, 123.5))
        self.assertFalse(result.simulated)

    def test_all_cans_keep_original_frame_coordinates(self):
        detector = self.ready_detector([[12, 18, 100, 150], [400, 100, 600, 320]], [.9, .85])
        result = detector.predict(self.frame)
        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(result.boxes[1].center, (500, 210))
        self.assertEqual(result.boxes[1].y2, 320)

    def test_invalid_boxes_and_below_threshold_are_discarded(self):
        detector = self.ready_detector([[-10, -10, 800, 900], [5, 2, 4, 8],
                                        [1, 1, float("nan"), 3], [2, 2, 8, 8]], [.9, .9, .9, .1])
        result = detector.predict(self.frame)
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual((result.boxes[0].x1, result.boxes[0].x2, result.boxes[0].y2), (0, 640, 360))

    def test_inference_error_never_reuses_old_boxes(self):
        detector = self.ready_detector([[1, 1, 8, 8]], [.95])
        self.assertEqual(len(detector.predict(self.frame).boxes), 1)
        detector._model.predict.side_effect = RuntimeError("GPU test failure")
        result = detector.predict(FramePacket(28, 124, self.frame.image))
        self.assertEqual(result.frame_id, 28)
        self.assertEqual(result.boxes, ())
        self.assertIn("GPU test failure", result.error)
        detector._model.predict.side_effect = None
        recovered = detector.predict(FramePacket(29, 125, self.frame.image))
        self.assertEqual(len(recovered.boxes), 1)
        self.assertIn("ready", detector.status)

    def test_load_warms_up_once_and_selects_cuda(self):
        weights = self.folder / "trained.pt"
        weights.write_text("mock backend weights", encoding="utf-8")
        model = SimpleNamespace(names={0: "redbull"}, predict=Mock())
        factory = Mock(return_value=model)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {"torch": torch, "ultralytics": SimpleNamespace(YOLO=factory)}):
            detector = Detector(weights)
            self.assertFalse(detector.available)
            self.assertTrue(detector.load())
            self.assertTrue(detector.load())
        factory.assert_called_once_with(str(weights.resolve()), task="detect")
        self.assertEqual(model.predict.call_count, 1)
        self.assertEqual(model.predict.call_args.kwargs["device"], 0)

    def test_generic_multiclass_weights_are_rejected(self):
        weights = self.folder / "wrong.pt"
        weights.write_text("mock", encoding="utf-8")
        model = SimpleNamespace(names={0: "person", 1: "bicycle"}, predict=Mock())
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                                    "ultralytics": SimpleNamespace(YOLO=lambda *a, **k: model)}):
            detector = Detector(weights)
            self.assertFalse(detector.load())
        self.assertIn("one-class can detector", detector.status)
        model.predict.assert_not_called()

    def test_dataset_counts_backgrounds_and_multiple_target_labels(self):
        from train_cans import inspect_split
        images = self.folder / "train" / "images"
        labels = self.folder / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        (images / "positive.jpg").write_bytes(b"fixture; decoded by YOLO at training time")
        (images / "empty.jpg").write_bytes(b"fixture")
        (labels / "positive.txt").write_text("0 0.3 0.5 0.1 0.2\n0 0.7 0.5 0.1 0.2\n", encoding="utf-8")
        (labels / "empty.txt").write_text("", encoding="utf-8")
        counts = inspect_split(images)
        self.assertEqual((counts["images"], counts["positive_images"], counts["negative_images"], counts["boxes"]), (2, 1, 1, 2))

    def test_one_class_classifier_is_not_accepted_as_a_detector(self):
        weights = self.folder / "classifier.pt"
        weights.write_text("mock", encoding="utf-8")
        model = SimpleNamespace(names={0: "redbull"}, task="classify", predict=Mock())
        with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                                    "ultralytics": SimpleNamespace(YOLO=lambda *a, **k: model)}):
            detector = Detector(weights)
            self.assertFalse(detector.load())
        self.assertIn("detection weights", detector.status)
        model.predict.assert_not_called()

    def test_dataset_rejects_missing_background_label_and_wrong_class(self):
        from train_cans import inspect_split
        images = self.folder / "train" / "images"
        labels = self.folder / "train" / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        (images / "one.jpg").write_bytes(b"fixture")
        with self.assertRaisesRegex(FileNotFoundError, "explicit label"):
            inspect_split(images)
        (labels / "one.txt").write_text("1 0.3 0.5 0.1 0.2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class 0"):
            inspect_split(images)


if __name__ == "__main__":
    unittest.main()
