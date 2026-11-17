# The Last Can

Start with [HOW_TO_RUN.md](HOW_TO_RUN.md).

```powershell
.\.venv\Scripts\python.exe main.py --demo
```

The real robot starts disconnected and uncalibrated. Offline simulation has an
amber banner, separate in-memory calibration, and no robot network access.

See [HARDWARE_CHECKLIST.md](HARDWARE_CHECKLIST.md) for remaining physical
measurements and live tests, and [TRAINING_NOTES.md](TRAINING_NOTES.md) for actual
model training results and pending robot evaluation.

Based on the instructor's `ccorbett0116/last-can`, `last-can-gui`,
`last-can-eyes`, and `last-can-grabber` starter projects supplied with the lab.
Original downloaded projects were preserved. The adapted MNIST sample retains
its instructor provenance in the tools folder. This solution adds the integrated
cockpit, calibrated control tasks, training/evaluation pipeline, and offline tests.

Instructor references: [assignment and rubric](https://github.com/ccorbett0116/last-can),
[robot guide](https://github.com/ccorbett0116/last-can/blob/main/docs/robot_guide.md),
[GUI starter](https://github.com/ccorbett0116/last-can-gui),
[vision starter](https://github.com/ccorbett0116/last-can-eyes), and
[arm starter](https://github.com/ccorbett0116/last-can-grabber).
