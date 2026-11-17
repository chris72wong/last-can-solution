# Training notes

**Group:** fill in group member names before submission.
**Model file:** `models/best.pt` from `baseline_v5_20261117`.
**Recorded:** 2026-11-17T17:49:09.821212+00:00. Actual offline training; live robot testing is pending.

## 1. Model

Fine-tuned `yolo11n.pt` as a one-class YOLO detector. The nano
model is the lab's baseline and keeps live inference light. No larger model
comparison has been performed. The GUI loads the actual trained weights.

The instructor's MNIST sample completed its 10 epochs first (digit validation
mAP50 0.991690, mAP50-95 0.965382). Those results are separate from the can
results below; see `verification/mnist-sample-results.json`.

## 2. Settings

Command: `python train_cans.py --data data\redbull_v5 --epochs 80 --patience 20 --imgsz 640 --batch 16 --workers 4 --name baseline_v5_20261117`

| Setting | Value | Reason |
| --- | --- | --- |
| Epoch limit / completed | 80 / 80 | Baseline 80; patience 20 permits early stopping. |
| Image size | 640 | Preserve detail in 640-pixel-wide robot frames. |
| Batch / workers | 16 / 4 | GPU baseline, adjusted only for actual resource/startup errors. |
| Horizontal flip | 0.5 | A mirrored can remains a can. |
| Patience / seed | 20 / 0 | Repeatable baseline and bounded early stopping. |
| Other augmentation | Ultralytics defaults | Preserve a simple starting point; compare future changes one at a time. |
| Runtime confidence | 0.25 | Initial threshold; final live tuning is pending. |
| Device | CUDA:0 | Checked before training; see verification/environment.json. |

## 3. Data and split

Dataset directory: `redbull_v5`. Its existing
training/validation/test split was preserved; no test images were used for
training or parameter selection. For the supplied `redbull_v5` baseline,
the instructor's README says robot captures were split by time blocks and
test pictures are robot-camera frames. The v5 training split includes
rotated training-only copies. Record the provenance/splitting of any custom
dataset here after adding new group captures.

| Split | Images | With a target | Without a target | Labelled boxes |
| --- | --- | --- | --- | --- |
| train | 1684 | 1401 | 283 | 1410 |
| val | 218 | 174 | 44 | 174 |
| test | 127 | 65 | 62 | 75 |

## 4. Validation and held-out results

| Split | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| val | 0.99472 | 0.75931 | 0.99926 | 0.98276 |
| test | 0.99370 | 0.77035 | 0.97400 | 0.99888 |

On the held-out robot-frame split at confidence 0.25:
positive frames with at least one detection: **65 / 65**;
negative frames with a false box: **4 / 62**.
These frame counts do not prove every can in a multi-can image was detected;
the object-level metrics above assess localization. Live lighting, motion,
wrong can variants, and new robot viewpoints still require physical testing.

Predictions, training curves, and batch visualizations remain in the local
run folder (excluded from submission). Machine-readable provenance and
metrics are in `verification/training-results.json`.

## 5. What we changed after testing on the robot

| Run | Robot test observation | Change | Result |
| --- | --- | --- | --- |
| baseline_v5_20261117 | Pending: no live robot test performed | Initial provided-data baseline | Offline metrics above; live outcome pending |

Complete this section after real robot tests. Save failures through the GUI,
label all target cans and keep wrong cans unboxed, split new captures by
session, and retrain without mixing the held-out test frames into training.

## 6. Known weaknesses

This model has not been tested live by our group. Camera pose changes,
unseen can orientations/variants, motion blur, and multiple touching cans
remain unverified. The observed offline misses and false alarms are retained
in the prediction output and report rather than claimed as successful.

At confidence 0.25, false alarms occurred on `neg_00249.jpg`,
`neg_00261.jpg`, `neg_00267.jpg`, and `neg_00276.jpg`. Visual review of
`neg_00249.jpg` showed a trouser leg incorrectly boxed at confidence 0.80;
raising confidence alone will not reliably fix that case. Collect NEW
negative robot frames of people/legs and similar clutter if this happens
live; keep the current test split untouched.

Visual review of `rbt_00203.jpg` showed separate boxes on an upright and
a sideways target; `rbt_00216.jpg` showed separate boxes on two distant
targets, one at confidence 0.30. Increasing the threshold could lose that
distant target. These are saved-frame observations, not live robot tests.

The exported model was also checked through the GUI's exact PIL-RGB frame
interface on a positive and a negative test image. Frame identities and box
coordinates matched Ultralytics file-input prediction on CUDA:0; see
`verification/model-inference-check.json` for the actual model SHA256.
