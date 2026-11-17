# The Last Can

A Windows tkinter cockpit for manual JetRover driving, live one-class Red Bull
detection, visual pickup alignment, and calibrated grab/drop routines.

**Physical robot testing is pending.** The real configuration contains no guessed
servo IDs, limits, poses, or pickup targets. Automatic routines stay unavailable
until those measurements are recorded. The offline demo uses separate synthetic
calibration in memory and cannot open a robot connection.

## Start on this computer

Open PowerShell in this project's folder. Use its virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe main.py --demo
```

The amber demo labels mean both the image and detections are simulated. Drive,
Grab, Drop, and Stop operate only the in-memory robot. Demo calibration changes
are never saved into real `config/robot.json`.

Start the normal application, initially disconnected:

```powershell
.\.venv\Scripts\python.exe main.py
```

When the environment is activated, the equivalent commands are `python main.py`
and `python main.py --demo`. Close any second control GUI before live operation.

## Install on another lab PC

Requires 64-bit Python 3.12 with tkinter and the lab's NVIDIA GPU/driver. The setup
script creates `.venv`, installs the lab's CUDA PyTorch pins first, installs the
application packages, and checks the GPU. It does not change global packages.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe tools\check_environment.py
```

Use `setup.ps1 -DemoOnly` for the offline GUI without the CUDA/YOLO installation.
The inference and training tests require the full installation. The exact
environment used for this build is recorded in `verification/environment.json`
and `requirements-lock.txt`; setup uses the instructor's documented pins.

Manual equivalent, if preferred:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Live use and calibration

Follow `HARDWARE_CHECKLIST.md` in order. No live commands were issued during the
software build. All work runs on the lab PC; install nothing on the robot.

1. Power the robot and join its Wi-Fi. Default address/topic names are the starter
   values in **Calibrate > Robot**. Check them against your robot using the
   read-only topic utility and its camera server page, then mark them verified.
2. Click **Connect robot**. Missing robot/camera/model errors appear as status
   messages. A failed connection is retried while connection is requested.
   Disconnect cancels those retries. Changing the robot address/topics invalidates
   old servo/pose/alignment verification so it cannot be reused on another robot.
   Disconnect or Stop clears previous movement. Reconnection requires release
   of the controls; old movement is not resumed.

   If the camera reports **Camera stream ended**, check the RGB feed directly at
   `http://192.168.149.1:8080/stream?topic=/depth_cam/rgb/image_raw&type=mjpeg`
   (substitute your verified robot IP/topic). The rover's video server requires
   literal `/` characters in the topic; `%2F` can produce an empty stream. The
   application preserves these slashes and connects directly without a PC HTTP
   proxy. Restart the application after updating the camera code.
3. **Calibrate > Servos:** enter each verified bus ID, joint name, and measured
   safe range. Every registered joint gets an Operate control. Choose a target
   and click **Send**; creating widgets and moving the target slider alone do
   not issue commands. Use slow moves while learning the arm. Displayed values
   are last **commanded** positions, never measured feedback.
4. **Calibrate > Poses:** after positioning the joints, save their commanded
   values under `home`, `open`, `reach`, `grip`, `lift`, `carry`, and
   `release_horizontal`. Home must include all joints. **Open and grip must
   include only the gripper joint IDs**, so opening at the release pose does
   not simultaneously swing the rest of the arm. Keep the gripper closed in
   `release_horizontal`; the next `open` step releases it in place.
5. Prove a pickup with the can in a fixed position. With the camera/arm at the
   **home/open** pose, capture that can in **Calibrate > Align**. Enter measured
   normalized tolerances and verified motion signs, and save. Turn sign +1
   commands negative angular.z for a can to the right of target; -1 reverses
   that convention. Forward sign +1 uses positive linear.x; -1 reverses it.
6. Drive until the target is in view, then **Grab**. It selects the largest
   detection and retains that target, moves home/open, aligns using short
   motion pulses followed by fresh observations, then reaches/grips/lifts/carries.
   If moving home changes the view so much the target cannot be matched, the
   routine aborts; search and start with the camera in its calibrated pose.
7. Drive back manually. **Drop** moves to the horizontal-release pose with the
   gripper closed, opens it in place, then returns home.

The first target is the largest visible can. Other boxes remain visible, but
the routine rejects ambiguous target matches, stale/lost detections, timeout,
and a can closer than the calibrated pickup distance. Reposition and retry.

Keys: **W/S** forward/back; **A/D** strafe; **Q/E** rotate; **G** Grab; **H** Drop;
**Space** Stop. Driving/task keys are ignored while typing into forms; **Esc**
stops even while typing. Losing application focus clears driving and cancels
the task. Manual controls are locked during automatic routines.

Stop cancels remaining sequence commands and requests zero wheel velocity.
**An already-issued servo movement may still finish.** Closing the window
flushes a final zero before closing the connection. A lost Wi-Fi connection
cannot deliver a stop; verify the robot's own command watchdog with your TA.

## Train and evaluate

Download the supplied dataset while connected to the internet:

```powershell
.\.venv\Scripts\python.exe tools\download_dataset.py
.\.venv\Scripts\python.exe train_cans.py --data data\redbull_v5 --name baseline_v1
```

Defaults are yolo11n, 640 pixels, 80 epochs, patience 20, batch 16, CUDA:0,
horizontal flips, and four workers. The existing dataset split is preserved.
The script refuses CPU training, checks labels, adapts to actual GPU-memory or
Windows worker errors, validates best weights, evaluates the held-out test
split, saves predictions, and copies genuine weights to `models/best.pt`.
Restart the normal cockpit after replacing weights; the model is loaded once.

To recover an interrupted training launch, rerun the above command with a new
name. For an actual Windows worker startup problem, add `--workers 0`; for a
GPU-memory problem, lower `--batch`. Do not train on the robot's Jetson.

Optional independent robot-frame folders can be evaluated during training with
`--positive-frames <folder> --negative-frames <folder>`. These must be unseen
frames, not copies of training images. Save raw camera frames through the GUI;
real captures go to `captures/robot`, synthetic ones to `captures/demo`.

Evaluate independently collected unseen frames without retraining:

```powershell
.\.venv\Scripts\python.exe -m tools.evaluate_frames --positive captures\positives --negative captures\negatives
```

Sort those folders by whether a target is present. The evaluator saves boxed
predictions and measured hit/false-alarm counts; inspect every failure.

Training evidence is in `TRAINING_NOTES.md`, `verification/training-results.json`,
and the local `runs/cans/<run>/` folder. Live testing remains pending regardless
of offline validation scores. Keep new captures split by session/time, label
every target can, and use empty label files for scenes with no target.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --demo --smoke-seconds 3
```

Tests use fake transports and synthetic observations, including a full
grab/drive/drop mission, cancellation, reconnect behavior, stale frames, box
coordinates, GUI key handling, and missing weights/calibration. The GUI smoke
check needs a desktop session; it does not connect to a robot.

MJPEG frame timestamps are **PC receipt times**. The displayed receive-to-result
time and age do not measure synchronized sensor age or full Wi-Fi/server delay.

## Submission

Fill in group names, complete live results in `TRAINING_NOTES.md`, and finish the
physical checklist. Build a clean ZIP with:

```powershell
.\.venv\Scripts\python.exe tools\package_submission.py
```

The ZIP includes source, configuration, trained model when present, dependencies,
instructions, tests, and compact verification records. It excludes `.venv`,
datasets, captures, caches, and `runs`. Test the extracted package on the demo PC.
