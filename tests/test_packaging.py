from pathlib import Path
import unittest

from tests.test_runtime import temporary_directory
from tools.package_submission import included_files


class SubmissionTests(unittest.TestCase):
    def test_only_can_weights_and_source_not_nested_training_artifacts(self):
        with temporary_directory() as directory:
            root = Path(directory)
            paths = ["main.py", "models/best.pt", "yolo11n.pt", ".venv/site.py",
                     "data/train/labels/can.txt", "runs/cans/v1/weights/best.pt",
                     "tools/train_mnist_sample.py", "tools/data/mnist/labels/one.txt",
                     "tools/runs/mnist/weights/best.pt", "lastcan/__pycache__/x.pyc",
                     "config/robot.json", "tests/test_example.py"]
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            selected = {p.relative_to(root).as_posix() for p in included_files(root)}
            self.assertEqual(selected, {"main.py", "models/best.pt", "tools/train_mnist_sample.py",
                                        "config/robot.json", "tests/test_example.py"})


if __name__ == "__main__":
    unittest.main()
