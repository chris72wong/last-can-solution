# Build verification — 2026-11-17

## Delivered

Integrated tkinter cockpit, MJPEG camera, one loaded trained YOLO detector,
rosbridge transport, exclusive wheel-command arbitration, calibrated visual
alignment, grab/drop pose sequences, and calibration recording controls.
The original four starter projects were preserved.

Offline mode uses synthetic camera frames/detections and a socket-free fake
transport. Real configuration ships with unverified starter topics and no arm
IDs/ranges/poses or pickup calibration. No physical robot was connected or moved.

## Automated evidence

- **79 offline tests passed** together; see `test-results.txt`.
- Tests cover all driving directions, release/focus/close stops, exclusive task
  ownership, expiring movement leases, reconnection without replay, cancellation,
  slow/stalled writes, gripper-only release, frame geometry, and stale data.
- Camera lifecycle regressions reproduce old-IP requests and old predictions
  completing after reconnect; results are discarded at an atomic session boundary.
- Actual cockpit smoke checks passed in demo and normal disconnected modes.
  Window-specific captures were visually inspected. `gui-smoke-normal.json`
  confirms the genuine model loaded on CUDA:0 without connecting the robot.
- Missing-weight startup was also exercised before training completed.
- CUDA environment and dependency checks passed; no broken pip requirements.

## Genuine training and held-out results

Adapted the instructor MNIST script into a one-class can trainer, ran the
10-epoch sample, then trained `baseline_v5_20261117` for all **80 epochs** on the
provided `redbull_v5` dataset. Original splits: **1,684 train / 218 validation /
127 test**. Image size 640, batch 16, four workers, patience 20, flip probability
0.5, and CUDA:0. Actual settings and complete metrics are in `training-results.json`.

| Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.999259 | 0.982759 | 0.994719 | 0.759315 |
| Held-out test | 0.973997 | 0.998876 | 0.993701 | 0.770351 |

At confidence **0.25**, frame-level testing found a detection on **65/65 positive
frames**, and false boxes on **4/62 negative frames**. Frame hit counts do not
prove that every can in a multi-can image was boxed; the object-level metrics
above measure localization/detection performance separately.

The final model SHA256 is
`e8e02af4d225e100e4cc661d7392191e4461f0960d19a68d2d85ad769a687293`.
`model-inference-check.json` verifies the exact exported checkpoint through
the application's PIL-RGB → BGR → YOLO path on actual held-out positive and
negative images, retaining source frame IDs and matching file-input predictions.

## Rubric status and physical work remaining

| Area | Software evidence | Still requires the real robot |
| --- | --- | --- |
| GUI, 10 marks | Window, buttons/keys, overlays, configurable joint controls, task controls, status and offline tests | Actual camera, every drive direction, every arm joint, and live boxes |
| Vision, 15 marks | Genuine group-authorized training, weights, script, notes and held-out evaluation | Live detection across angles/distances/edges, multiple cans, wrong-can rejection and driving |
| Hunt, 10 marks | Synthetic full grab → return drive → horizontal drop, cancellation and failure handling | Measured alignment/poses, secure grip, actual return and horizontal release, repeatability |

Complete `HARDWARE_CHECKLIST.md`, fill group names and real test outcomes in
`TRAINING_NOTES.md`, tune confidence using live validation, and add verified
failure examples if needed. This build is **not claimed to be a successful live
robot mission**. Unknown physical values are deliberately unset.

## Packaging

The submission ZIP uses an explicit source/config/model allowlist and a SHA256
manifest. Recursive `data`, `runs`, cache, capture and virtual-environment trees
are excluded, including those under the copied MNIST sample. The generic
pretrained `yolo11n.pt` at the development root is excluded; `models/best.pt`
is the trained can checkpoint. Final ZIP verification is recorded separately
in `package-verification.json` after extraction and launch checks.
