# Measurements and live tests still required

**Status: no physical robot connection, arm calibration, or live mission has been
performed during this build.** Offline test results prove software behavior in a
simulation; they do not establish a successful pickup or detector performance on
the live feed.

## 1. Connection and directions

- [ ] Read the instructor robot guide; one operator controls movement.
- [ ] Join the robot Wi-Fi. Download dependencies/models before switching.
- [ ] Record the actual IP: __________
- [ ] Verify camera topic and stream in `http://<robot-ip>:8080`: __________
- [ ] Run `python tools/list_topics.py` and record the actual topics/types.
- [ ] Drive topic/type: __________
- [ ] Arm topic/type and message fields: __________
- [ ] Mark verified values in Calibrate > Robot, then Connect.
- [ ] With wheels raised, verify each drive button/key and release-to-stop.
- [ ] Verify Stop, focus loss, closing the app, and reconnection behavior.
- [ ] Confirm the robot's velocity watchdog/physical stopping procedure with TA;
      software cannot deliver zero over a disconnected network.

## 2. Map and register every arm joint

Find IDs and current positions through the robot/manual with the TA. Start near
the actual current position, not an assumed middle value. Move one joint slowly
at a time, and leave clearance from floor, chassis, camera, and other joints.
Do not leave a servo stalled or squeezing until it buzzes/heats up.

| ID | Joint | Current position source | Increasing moves toward | Safe min | Safe max | Verified by |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- [ ] Enter actual IDs/ranges in Calibrate > Servos, including the gripper.
- [ ] Confirm last-commanded values are not being mistaken for sensor readings.
- [ ] Record whether the camera moves with the arm: __________

## 3. Calibrate and prove poses

Use slow manual joint commands, visually inspect the pose, and save it through
Calibrate > Poses. A saved pose records commanded values only. Check the verified
box only after observing that pose successfully on this robot.

| Pose | Joint IDs included | Positions | Duration / pause | Physical result |
| --- | --- | --- | --- | --- |
| home (every joint) | | | | |
| open (gripper only) | | | | |
| reach | | | | |
| grip (gripper only) | | | | |
| lift | | | | |
| carry | | | | |
| release_horizontal (gripper closed) | | | | |

- [ ] Fixed-position pickup succeeds repeatedly without crushing/slipping.
- [ ] Carry pose holds the can through driving, turning, and stopping.
- [ ] Horizontal-release pose keeps the can closed in the gripper near the floor.
- [ ] The following open step releases in place; only then does home retract.
- [ ] Practise cancellation. An already-issued servo move may still finish.

## 4. Calibrate visual alignment

Place the can where the proven pickup works. Set arm/camera to the home/open
pose, capture the largest target box in Calibrate > Align, then record:

| Measurement | Actual value | How verified |
| --- | --- | --- |
| Normalized target center X | | |
| Normalized target box height | | |
| X tolerance inside successful pickup range | | |
| Height tolerance inside successful pickup range | | |
| Turn sign (+1 or -1) | | |
| Forward sign (+1 or -1) | | |

- [ ] Save verified alignment. Check from left/right/near/far starting positions.
- [ ] Confirm short pulses and pauses cope with actual camera delay.
- [ ] Hide target mid-approach: wheels stop and task fails clearly.
- [ ] Try multiple cans; confirm it retains its chosen target or aborts ambiguity.
- [ ] Start only with camera/arm in a view consistent with home/open calibration.

## 5. Live detector tests (13 of 15 vision marks)

Use the actual trained model and tune confidence on the live camera. Save useful
failure frames using Save frame; correct labels and retrain as needed. Keep
separate capture sessions for training, validation, and final testing.

| Case | Passed / attempts | Confidence | Failure-frame names / fix |
| --- | --- | --- | --- |
| Upright around 30 cm, 1 m, and 2 m | | | |
| Can turned around / top or bottom facing camera | | | |
| Can lying on its side | | | |
| Target at each frame edge | | | |
| Two or three targets, separate and touching | | | |
| Wrong can beside target: only target boxed | | | |
| Empty room / other objects: no target boxes | | | |
| Robot driving past target | | | |

Record actual model changes and both improvements and regressions in
TRAINING_NOTES.md. Do not replace pending live results with offline metrics.

## 6. Full mission and submission

- [ ] Start on the floor, target outside the camera view.
- [ ] Driver finds the target using this GUI; detect while moving and stopped.
- [ ] Press Grab once: robot aligns, approaches, picks up, and holds the can.
- [ ] Drive to start with the can still held.
- [ ] Press Drop once: controlled release near start, can horizontal on release.
- [ ] Repeat without restarts or manual rescues. Record driver, time, failures.
- [ ] Everyone can explain their code and operate the controls.
- [ ] Save a dated working backup; fill group names and genuine live notes.
- [ ] Package source, config, model, notes, HOW_TO_RUN; omit photos/runs/.venv.
- [ ] Test the extracted submission on the lab PC; make one group submission.

| Attempt | Driver | Model / config version | Complete? | Failure and change |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |
| | | | | |
