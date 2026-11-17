"""Exercise the actual cockpit without a robot and record concrete desktop checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import tkinter as tk
    from PIL import ImageGrab
    from lastcan.gui import Cockpit
    from lastcan.runtime import Runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", action="store_true", help="normal GUI; NEVER clicks Connect")
    args = parser.parse_args()
    path = ROOT / "config" / "robot.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    runtime = Runtime(demo=not args.normal)
    root = tk.Tk()
    app = Cockpit(root, runtime)
    evidence = ROOT / "verification"
    evidence.mkdir(exist_ok=True)
    result = {}
    started = time.monotonic()

    def inspect():
        # A normal run with real weights should prove the background loader is
        # usable, rather than recording a successful window before it finishes.
        if (args.normal and runtime.config.resolve(runtime.config.data["vision"]["model"]).is_file()
                and not runtime.snapshot()["model_available"] and time.monotonic() - started < 15):
            root.after(250, inspect)
            return
        try:
            snapshot = runtime.snapshot()
            packet = runtime.get_display()
            result.update(mode="normal-disconnected" if args.normal else "offline-demo",
                window=[root.winfo_width(), root.winfo_height()],
                canvas=[app.canvas.winfo_width(), app.canvas.winfo_height()],
                rendered_frame=getattr(packet, "frame_id", None),
                boxes=len(packet.boxes) if hasattr(packet, "boxes") else 0,
                joint_controls=len(app._servo_widgets), connected=snapshot["connected"],
                task=snapshot["task_status"], detector=snapshot["detector_status"],
                model_available=snapshot["model_available"],
                servo_motion_at_startup=False, nonzero_drive_at_startup=False)
            if runtime.demo:
                result["servo_motion_at_startup"] = any(k == "servos" for k, v in runtime.transport.commands)
                result["nonzero_drive_at_startup"] = any(k == "velocity" and any(v) for k, v in runtime.transport.commands)
                if not packet or not getattr(packet, "simulated", False) or result["boxes"] < 1:
                    raise AssertionError("Demo did not deliver labelled synthetic detections")
                if result["servo_motion_at_startup"] or result["nonzero_drive_at_startup"]:
                    raise AssertionError("Startup unexpectedly issued a movement")
            elif runtime.transport.connected:
                raise AssertionError("Normal smoke unexpectedly connected to the robot")
            elif runtime.config.resolve(runtime.config.data["vision"]["model"]).is_file() and not snapshot["model_available"]:
                raise AssertionError("Trained weights exist but did not load: " + snapshot["detector_status"])
            try:
                # Capture only our own window, never unrelated desktop content.
                shot = ImageGrab.grab(window=int(root.winfo_id()))
                screenshot = evidence / ("cockpit-normal.png" if args.normal else "cockpit-demo.png")
                shot.save(screenshot)
                result["screenshot"] = screenshot.name
            except Exception as exc:
                result["screenshot"] = f"Unavailable in this desktop session: {type(exc).__name__}: {exc}"
            result["passed"] = True
        except Exception as exc:
            result["passed"] = False
            result["error"] = str(exc)
        finally:
            app.close()

    root.after(2500, inspect)
    root.mainloop()
    runtime.close()
    runtime.wait_closed()
    after = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    result["real_configuration_unchanged"] = before == after
    result["monotonic_completed"] = time.monotonic()
    output = evidence / ("gui-smoke-normal.json" if args.normal else "gui-smoke.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
