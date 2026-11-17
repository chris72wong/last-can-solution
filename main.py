"""Start the real cockpit disconnected, or explicitly opt into offline simulation."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="The Last Can | JetRover cockpit")
    parser.add_argument("--demo", action="store_true", help="fully offline synthetic camera and fake robot")
    parser.add_argument("--config", type=Path, help="configuration JSON (defaults to config/robot.json)")
    parser.add_argument("--model", help="override trained model path")
    parser.add_argument("--smoke-seconds", type=float, default=0, help="close GUI automatically for smoke checks")
    args = parser.parse_args()
    runtime = root = app = None
    try:
        import tkinter as tk
        from lastcan.gui import Cockpit
        from lastcan.runtime import Runtime
        runtime = Runtime(demo=args.demo, config_path=args.config, model_path=args.model)
        root = tk.Tk()
        app = Cockpit(root, runtime)
        if args.smoke_seconds > 0:
            root.after(int(args.smoke_seconds * 1000), app.close)
        root.mainloop()
        return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run .\\setup.ps1, then use .venv\\Scripts\\python.exe main.py --demo", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Cockpit could not start: {exc}", file=sys.stderr)
        return 1
    finally:
        if app is not None:
            app.close()
        elif root is not None:
            root.destroy()
        if runtime is not None:
            runtime.close()
            runtime.wait_closed(None)
        # Tk variables/interpreters must be collected on their owning thread,
        # not during a later allocation in a camera/model worker.
        app = root = None
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
