# Integration contract

Root owns `lastcan/types.py`, `config.py`, `camera.py`, `runtime.py`, `main.py`, integration tests/docs/package. Do not edit another agent's files without coordination.

Shared types in `lastcan.types`: `Detection(x1,y1,x2,y2,confidence)` with `.center`/`.area`; `FramePacket(frame_id,captured_at,image)`; `DetectionPacket(frame_id,captured_at,completed_at,image,boxes,simulated=False,error='')`; `LatestValue.put/get`. Timestamps are monotonic; image is full-resolution PIL RGB. Boxes belong to the exact image in DetectionPacket. `fit_geometry(source_size,target_size)` and `drive_axes(actions,linear,angular)` shared pure helpers.

ConfigStore has `.data` mutable dict, `.save()`, `.snapshot()`, `.resolve(path)`; actual schema is DEFAULT_CONFIG in config.py. Servo entries: `{id:int,name:str,min:int,max:int,verified:bool}`; poses keyed by name: `{positions:{str_id:int},duration:float,pause:float,verified:bool}`. Real defaults deliberately have NO servos/poses; physical alignment values are null. Demo config will be separate, synthetic, memory-only.

## Robot/control agent owns lastcan/control.py and tests/test_control.py

Export FakeTransport(config), RosbridgeTransport(config): `.connected`, `.connect()->bool`, `.close()`, `.publish_velocity(lx,ly,az)`, `.move_servos(positions,duration)`; fake transport records commands. config is FULL configuration dict. Live constructor must not connect automatically.

Export RobotController(transport, config): `.set_manual(lx,ly,az)->bool` renews manual lease; `.clear_manual()`, `.stop(reason='Stopped')`, `.tick(now=None)` called 10Hz by runtime worker; `.owner`, `.commanded_positions` dict[int,int]; `.move_manual_servo(id,position,duration=1.0)` validates config. Own ALL wheel arbitration; connection transitions clear movement; no automatic reconnect replay.

Export TaskRunner(controller, config, get_detection:Callable[[],DetectionPacket|None]): `.start_grab()->bool`, `.start_drop()->bool` spawn worker; `.cancel(reason='Cancelled')`; `.busy`, `.status`, `.error`; routines require real verified calibration (except separate synthetic demo config), validate sequences before any motion, stop on failures. Coordinate extra public API with root.

## Vision agent owns lastcan/vision.py, train_cans.py, tools training/setup/dataset helpers, requirements files, .venv, models, TRAINING_NOTES.md, tests/test_vision.py

Export Detector(model_path, imgsz=640, confidence=0.25): `.status` string, `.available` bool, `.confidence` adjustable; `.predict(frame:FramePacket)->DetectionPacket`. Lazy import/load once in WORKER not UI; handle missing weights without downloading a generic fallback. Consider `.load()` explicit; coordinate with root. No frame queue in module; runtime repeatedly calls predict on latest unique frame.

Manage local venv install, dataset download, sample run and actual baseline training. Do not edit source starters or root-owned files. Coordinate all install/training processes; log results and blockers truthfully.

## GUI agent owns lastcan/gui.py and tests/test_gui.py

Export Cockpit(root:tk.Tk, runtime). Runtime facade (root implements):
- `.demo` bool, `.config` ConfigStore (demo uses temporary in-memory store), `.message` str
- `.connect()`, `.disconnect()`, `.close()`; nonblocking callbacks to workers
- `.set_drive_action(action,pressed)` actions forward/back/left/right/rotate_left/rotate_right
- `.heartbeat()` GUI calls every 50ms; renew manual lease from current pressed actions
- `.stop(reason='Stopped')`, `.grab()`, `.drop()`
- `.set_servo(id,position,duration=1.0)` queues work, no network on Tk thread
- `.get_display()->DetectionPacket|FramePacket|None` exact matching annotated frame
- `.get_detection()->DetectionPacket|None`, `.save_frame()->Path|None`
- `.set_confidence(value)`
- `.save_pose(name,duration,pause,verified,servo_ids=None)` saves selected last-commanded positions; fail if unknown. Home includes all joints; open/grip must explicitly select only the gripper IDs.
- `.save_alignment(target_x,target_height,x_tolerance,height_tolerance,turn_sign,forward_sign)` normalized targets/tolerances; requires current real/fake detection appropriate to mode
- `.save_settings()` persists changes to config only while task idle
- `.snapshot()->dict` fields: connected, connection_status, camera_status, detector_status, task_status, task_error, busy, owner, commanded_positions, calibrated, model_available, detection_count

`Runtime.close()` starts nonblocking shutdown, and `wait_closed()` joins outside Tk's event loop. A final zero is flushed before the socket closes. `controller.stop(...,publish=False)` is a nonblocking intent used by GUI callbacks. Manual servo requests carry an epoch guard and are sent only from the command worker. TaskRunner starts/cancels are CPU-only; actual sends run in its worker.

GUI provides config fields for IP/topics, servo registration with verified ID/name/min/max, pose recording, alignment fields/capture. No motion on widget creation or config reload. Stop clears input, cancels tasks, zeros wheels. Keys W/S forward/back A/D strafe Q/E rotate G Grab H Drop Space Stop. Avoid driving while typing in form inputs. Focus loss clears manual input/cancels autonomous if entire app loses focus. Only main thread touches Tk.
