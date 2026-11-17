"""Tk cockpit. All robot IO and inference are delegated to the runtime.

Servo sliders stage targets; only an explicit Send command moves a joint.
The camera and every box always use the same frame packet and transform.
"""
from __future__ import annotations

import copy
import json
import math
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

from PIL import Image, ImageTk

from .types import Detection, DetectionPacket, FramePacket, fit_geometry

BG = "#09111f"
PANEL = "#121e30"
FIELD = "#1a2a40"
BORDER = "#263950"
TEXT = "#edf4fb"
MUTED = "#96abc2"
GREEN = "#50dfbd"
AMBER = "#f2ba69"
RED = "#fa7b83"

KEY_ACTIONS = {"w": "forward", "s": "back", "a": "left", "d": "right",
               "q": "rotate_left", "e": "rotate_right"}
POSE_NAMES = ("home", "open", "reach", "grip", "lift", "carry", "release_horizontal")


def canvas_box(box: Detection, source: tuple[int, int], target: tuple[int, int]) -> tuple[float, ...]:
    """Pure shared-frame coordinate conversion, including letterboxing."""
    scale, ox, oy = fit_geometry(source, target)
    return (box.x1 * scale + ox, box.y1 * scale + oy,
            box.x2 * scale + ox, box.y2 * scale + oy)


def is_text_input(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        return widget.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox", "Text"}
    except (AttributeError, tk.TclError):
        return False


def fresh_largest_box(packet: DetectionPacket | None, max_age: float, demo: bool,
                      now: float | None = None) -> tuple[float, float]:
    """Calibration cannot quietly use an old frame or cross simulation modes."""
    now = time.monotonic() if now is None else now
    if packet is None or packet.error or not packet.boxes:
        raise ValueError("No detected can. Put a can at the proven pickup position first.")
    if packet.simulated != demo:
        raise ValueError("This detection belongs to a different operating mode.")
    if now - packet.captured_at > max_age or packet.captured_at > now:
        raise ValueError("Camera frame is stale. Wait for a fresh detection.")
    box = max(packet.boxes, key=lambda candidate: candidate.area)
    width, height = packet.image.size
    return box.center[0] / width, (box.y2 - box.y1) / height


class Cockpit:
    def __init__(self, root: tk.Tk, runtime: Any):
        self.root, self.runtime = root, runtime
        self._closed = False
        self._after_id: str | None = None
        self._focus_after_ids: set[str] = set()
        self._photo: ImageTk.PhotoImage | None = None
        self._last_display: Any = None
        self._display_key: Any = None
        self._keys: set[str] = set()
        self._drive_sources: dict[str, set[str]] = {}
        self._servo_signature = ""
        self._servo_widgets: dict[int, dict[str, Any]] = {}
        self._scroll_regions: dict[tk.Misc, tk.Canvas] = {}
        self._last_message = ""
        self._runtime_message_seen = ""
        self.root.title("The Last Can | JetRover Mission Control")
        self.root.configure(bg=BG)
        self.root.geometry("1320x880")
        self.root.minsize(1080, 740)
        self.root.option_add("*Font", ("Segoe UI", 10))
        self._style()
        self._build()
        self._bind()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("TButton", background=FIELD, foreground=TEXT, borderwidth=0, padding=(12, 8))
        style.map("TButton", background=[("active", BORDER), ("disabled", PANEL)],
                  foreground=[("disabled", "#51667d")])
        style.configure("Accent.TButton", background=GREEN, foreground=BG, font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#91efd6"), ("disabled", BORDER)])
        style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT,
                        bordercolor=BORDER, padding=5)
        style.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                        arrowcolor=TEXT, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                        arrowcolor=TEXT, padding=4)
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(13, 9))
        style.map("TNotebook.Tab", background=[("selected", FIELD)], foreground=[("selected", GREEN)])
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=BG)
        style.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT,
                        borderwidth=0, rowheight=25)
        style.configure("Treeview.Heading", background=BORDER, foreground=TEXT, relief="flat")
        style.map("Treeview", background=[("selected", "#30536a")])
        style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=PANEL, arrowcolor=MUTED)

    @staticmethod
    def _label(parent: tk.Misc, text: str = "", *, muted: bool = False, **kwargs: Any) -> ttk.Label:
        return ttk.Label(parent, text=text, style="Muted.TLabel" if muted else "TLabel", **kwargs)

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(19, 13))
        identity = tk.Frame(header, bg=BG)
        identity.pack(side="left")
        tk.Label(identity, text="THE LAST CAN", bg=BG, fg=TEXT,
                 font=("Segoe UI", 24, "bold")).pack(anchor="w")
        tk.Label(identity, text="JETROVER  /  MISSION CONTROL", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(1, 0))
        self.stop_button = tk.Button(header, text="STOP  [SPACE]", bg="#79313e", fg="white",
                                     activebackground="#a13d4e", activeforeground="white",
                                     relief="flat", borderwidth=0, padx=22, pady=13,
                                     font=("Segoe UI", 11, "bold"), command=self._stop)
        self.stop_button.pack(side="right", padx=(14, 0))
        self.disconnect_button = ttk.Button(header, text="Disconnect", command=lambda: self._call(self.runtime.disconnect))
        self.disconnect_button.pack(side="right", padx=(8, 0))
        self.connect_button = ttk.Button(header, text="Connect robot", style="Accent.TButton",
                                         command=lambda: self._call(self.runtime.connect))
        self.connect_button.pack(side="right")
        mode = "SIMULATED DEMO · NO ROBOT IO" if self.runtime.demo else "LIVE MODE · CONNECT WHEN READY"
        tk.Label(header, text=mode, bg=BG, fg=AMBER if self.runtime.demo else GREEN,
                 font=("Segoe UI", 9, "bold")).pack(side="right", padx=22)

        self.status_frame = tk.Frame(self.root, bg=BG)
        self.status_frame.pack(fill="x", padx=24, pady=(0, 14))
        self.status_labels: dict[str, ttk.Label] = {}
        for index, (key, title) in enumerate((("connection", "01 / CONNECTION"),
                                             ("vision", "02 / VISION"), ("mission", "03 / MISSION"))):
            self.status_frame.columnconfigure(index, weight=1)
            card = ttk.Frame(self.status_frame, padding=(14, 9))
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 7, 0))
            self._label(card, title, muted=True, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            label = self._label(card, "Starting…", font=("Segoe UI", 10, "bold"), wraplength=360)
            label.pack(anchor="w", pady=(4, 0))
            self.status_labels[key] = label

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2, minsize=430)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        camera_top = ttk.Frame(left)
        camera_top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._label(camera_top, "ROVER CAMERA", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.frame_label = self._label(camera_top, "Waiting for a frame", muted=True, font=("Segoe UI", 9))
        self.frame_label.pack(side="right")
        self.canvas = tk.Canvas(left, background="#060d17", highlightthickness=1,
                                highlightbackground=BORDER, width=720, height=405)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render(self._last_display, force=True))
        camera_bottom = ttk.Frame(left)
        camera_bottom.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self._label(camera_bottom, "Confidence", muted=True).pack(side="left")
        self.confidence = tk.DoubleVar(value=float(self.runtime.config.data["vision"]["confidence"]))
        self.confidence_value = self._label(camera_bottom, f"{self.confidence.get():.2f}", width=4)
        self.confidence_value.pack(side="left", padx=8)
        confidence_slider = ttk.Scale(camera_bottom, from_=0.05, to=0.95, variable=self.confidence,
                                      command=self._confidence_changed)
        confidence_slider.pack(side="left", fill="x", expand=True, padx=(0, 14))
        ttk.Button(camera_bottom, text="Save frame", command=self._save_frame).pack(side="right")
        self.camera_status = self._label(left, "", muted=True, wraplength=680)
        self.camera_status.grid(row=3, column=0, sticky="w", pady=(8, 4))
        self.video_stats = self._label(left, "Boxes use the exact displayed frame. Frame age excludes camera/server delay.",
                                      muted=True, wraplength=680, font=("Segoe UI", 8))
        self.video_stats.grid(row=4, column=0, sticky="w")
        mission = ttk.Frame(left)
        mission.grid(row=5, column=0, sticky="ew", pady=(15, 0))
        mission.columnconfigure((0, 1), weight=1)
        self.grab_button = ttk.Button(mission, text="GRAB CAN  [G]", style="Accent.TButton",
                                      command=lambda: self._call(self.runtime.grab))
        self.grab_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.drop_button = ttk.Button(mission, text="DROP HORIZONTALLY  [H]", command=lambda: self._call(self.runtime.drop))
        self.drop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.task_hint = self._label(left, "", muted=True, wraplength=680, font=("Segoe UI", 9))
        self.task_hint.grid(row=6, column=0, sticky="w", pady=(10, 0))

        tabs = ttk.Notebook(body)
        tabs.grid(row=0, column=1, sticky="nsew")
        operate = ttk.Frame(tabs, padding=14)
        calibrate = ttk.Frame(tabs, padding=(10, 12))
        tabs.add(operate, text="Operate")
        tabs.add(calibrate, text="Calibrate")
        self._build_operate(operate)
        self._build_calibration(calibrate)

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=24, pady=(11, 14))
        self.footer = tk.Label(footer, text="Ready", bg=BG, fg=MUTED, anchor="w",
                               font=("Segoe UI", 9), wraplength=1120)
        self.footer.pack(fill="x")

    def _build_operate(self, parent: ttk.Frame) -> None:
        self._label(parent, "DRIVE", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._label(parent, "Hold a button or key. Release to stop that direction.", muted=True,
                    font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 9))
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        grid.columnconfigure((0, 1, 2), weight=1)
        positions = (("rotate_left", "↶  Q", 0, 0), ("forward", "↑  W", 0, 1),
                     ("rotate_right", "E  ↷", 0, 2), ("left", "←  A", 1, 0),
                     ("back", "↓  S", 1, 1), ("right", "D  →", 1, 2))
        self.drive_buttons = []
        for action, label, row, col in positions:
            button = tk.Button(grid, text=label, bg=FIELD, fg=TEXT, activebackground=BORDER,
                               activeforeground=GREEN, relief="flat", bd=0, pady=13,
                               font=("Segoe UI", 14, "bold"), takefocus=False)
            button.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
            button.bind("<ButtonPress-1>", lambda _e, name=action: self._set_source(name, "mouse", True))
            button.bind("<ButtonRelease-1>", lambda _e, name=action: self._set_source(name, "mouse", False))
            self.drive_buttons.append(button)
        self.owner_label = self._label(parent, "Wheel control: stopped", muted=True, font=("Segoe UI", 9))
        self.owner_label.pack(anchor="w", pady=(8, 17))
        self._label(parent, "ARM JOINTS", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._label(parent, "Commanded positions only · no position feedback", muted=True,
                    font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 9))
        arm_container = ttk.Frame(parent)
        arm_container.pack(fill="both", expand=True)
        self.arm_canvas = tk.Canvas(arm_container, background=PANEL, highlightthickness=0, width=370)
        arm_scroll = ttk.Scrollbar(arm_container, orient="vertical", command=self.arm_canvas.yview)
        self.arm_canvas.configure(yscrollcommand=arm_scroll.set)
        arm_scroll.pack(side="right", fill="y")
        self.arm_canvas.pack(side="left", fill="both", expand=True)
        self.arm_frame = ttk.Frame(self.arm_canvas)
        self._scroll_regions[self.arm_frame] = self.arm_canvas
        self._arm_window = self.arm_canvas.create_window((0, 0), window=self.arm_frame, anchor="nw")
        self.arm_frame.bind("<Configure>", lambda _e: self.arm_canvas.configure(scrollregion=self.arm_canvas.bbox("all")))
        self.arm_canvas.bind("<Configure>", lambda event: self.arm_canvas.itemconfigure(self._arm_window, width=event.width))

    def _build_calibration(self, parent: ttk.Frame) -> None:
        self._label(parent, "Save measurements from your robot.", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._label(parent, "Demo changes stay in memory. Live poses start uncalibrated.", muted=True,
                    wraplength=440, font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 11))
        tabs = ttk.Notebook(parent)
        tabs.pack(fill="both", expand=True)
        robot, servos, poses, alignment = (self._scrolling_tab(tabs, name)
                                         for name in ("Robot", "Servos", "Poses", "Align"))
        self._build_robot_settings(robot)
        self._build_servo_settings(servos)
        self._build_pose_settings(poses)
        self._build_alignment_settings(alignment)

    def _scrolling_tab(self, tabs: ttk.Notebook, name: str) -> ttk.Frame:
        wrapper = ttk.Frame(tabs)
        tabs.add(wrapper, text=name)
        canvas = tk.Canvas(wrapper, background=PANEL, highlightthickness=0, width=390)
        scrollbar = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        self._scroll_regions[content] = canvas
        self._scroll_regions[canvas] = canvas
        return content

    def _form_field(self, parent: ttk.Frame, row: int, label: str, value: Any, width: int = 23) -> tk.StringVar:
        self._label(parent, label, muted=True).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        variable = tk.StringVar(value="" if value is None else str(value))
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=5)
        parent.columnconfigure(1, weight=1)
        return variable

    def _build_robot_settings(self, parent: ttk.Frame) -> None:
        robot = self.runtime.config.data["robot"]
        self.robot_fields: dict[str, tk.StringVar] = {}
        for row, (key, label) in enumerate((("ip", "Robot IP"), ("rosbridge_port", "Command port"),
                                           ("video_port", "Video port"), ("camera_topic", "Camera topic"),
                                           ("cmd_vel_topic", "Drive topic"), ("servo_topic", "Servo topic"))):
            self.robot_fields[key] = self._form_field(parent, row, label, robot[key])
        self.topics_verified = tk.BooleanVar(value=bool(robot.get("topics_verified")))
        ttk.Checkbutton(parent, text="I verified topics/types on this robot", variable=self.topics_verified).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(12, 6))
        ttk.Button(parent, text="Save connection settings", command=self._save_robot_settings).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=8)
        self._label(parent, "Disconnect before changing connection settings. Reconnect to apply them.\n\n"
                    "Directions and servo IDs must be checked on the real robot before use.", muted=True,
                    wraplength=375, font=("Segoe UI", 9)).grid(row=8, column=0, columnspan=2, sticky="w", pady=10)

    def _build_servo_settings(self, parent: ttk.Frame) -> None:
        self.servo_fields = {}
        for row, (key, label) in enumerate((("id", "Servo ID"), ("name", "Joint name"),
                                           ("min", "Verified minimum"), ("max", "Verified maximum"))):
            self.servo_fields[key] = self._form_field(parent, row, label, "")
        self.servo_verified = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="ID and safe range physically verified", variable=self.servo_verified).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=8)
        buttons = ttk.Frame(parent)
        buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(3, 10))
        ttk.Button(buttons, text="Add / update joint", command=self._save_servo).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Remove ID", command=self._remove_servo).pack(side="left", padx=(7, 0))
        self.servo_tree = ttk.Treeview(parent, columns=("id", "name", "range", "verified"), show="headings", height=5)
        for key, label, width in (("id", "ID", 35), ("name", "Joint", 110), ("range", "Range", 90), ("verified", "Checked", 65)):
            self.servo_tree.heading(key, text=label)
            self.servo_tree.column(key, width=width, stretch=key == "name")
        self.servo_tree.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.servo_tree.bind("<<TreeviewSelect>>", self._select_servo)
        self._label(parent, "Register only IDs you have identified. Editing joints invalidates saved pose/alignment checks. "
                    "Sliders stage a target; Send explicitly commands it.", muted=True, wraplength=375,
                    font=("Segoe UI", 9)).grid(row=7, column=0, columnspan=2, sticky="w", pady=12)

    def _build_pose_settings(self, parent: ttk.Frame) -> None:
        self._label(parent, "Pose name", muted=True).grid(row=0, column=0, sticky="w", pady=5)
        self.pose_name = tk.StringVar(value="home")
        ttk.Combobox(parent, textvariable=self.pose_name, values=POSE_NAMES, width=22).grid(row=0, column=1, sticky="ew")
        self.pose_joint_ids = self._form_field(parent, 1, "Joint IDs (blank=all)", "")
        self.pose_duration = self._form_field(parent, 2, "Move duration (s)", 1.5)
        self.pose_pause = self._form_field(parent, 3, "Extra pause (s)", 0.2)
        self.pose_verified = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Pose and clearance physically checked", variable=self.pose_verified).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=10)
        ttk.Button(parent, text="Save current commanded pose", command=self._save_pose).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.pose_tree = ttk.Treeview(parent, columns=("pose", "joints", "duration", "verified"), show="headings", height=7)
        for key, label, width in (("pose", "Pose", 135), ("joints", "IDs", 70), ("duration", "Seconds", 65), ("verified", "Checked", 60)):
            self.pose_tree.heading(key, text=label)
            self.pose_tree.column(key, width=width, stretch=key == "pose")
        self.pose_tree.grid(row=6, column=0, columnspan=2, sticky="nsew")
        self.pose_tree.bind("<<TreeviewSelect>>", self._select_pose)
        self._label(parent, "For OPEN and GRIP, enter only the gripper joint ID(s). These partial poses must not move "
                    "the arm away from the horizontal-release position. Other poses can include all joints. "
                    "Values are commanded, not measured feedback.",
                    muted=True, wraplength=375, font=("Segoe UI", 9)).grid(row=7, column=0, columnspan=2, sticky="w", pady=12)

    def _build_alignment_settings(self, parent: ttk.Frame) -> None:
        alignment = self.runtime.config.data["alignment"]
        self.alignment_fields = {}
        for row, (key, label) in enumerate((("target_x", "Pickup center X / width"),
                                           ("target_height", "Pickup box H / height"),
                                           ("x_tolerance", "X tolerance (fraction)"),
                                           ("height_tolerance", "Height tolerance"))):
            self.alignment_fields[key] = self._form_field(parent, row, label, alignment.get(key))
        self.alignment_signs = {}
        for row, (key, label) in enumerate((("turn_sign", "Turn response sign"), ("forward_sign", "Forward response sign")), start=4):
            self._label(parent, label, muted=True).grid(row=row, column=0, sticky="w", pady=5)
            value = alignment.get(key)
            variable = tk.StringVar(value="" if value is None else str(int(value)))
            ttk.Combobox(parent, textvariable=variable, values=("1", "-1"), state="readonly", width=8).grid(
                row=row, column=1, sticky="ew", pady=5)
            self.alignment_signs[key] = variable
        ttk.Button(parent, text="Capture largest detected can", command=self._capture_alignment).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(12, 6))
        ttk.Button(parent, text="Save verified pickup calibration", command=self._save_alignment).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=6)
        self._label(parent, "First prove a pickup with the can in a fixed position. Capture in HOME/OPEN pose: Grab "
                    "returns there before aligning. Turn sign +1 turns right for a can right of target; −1 reverses it. "
                    "Enter measured tolerances and verify movement signs. "
                    "The largest visible can is selected initially.", muted=True, wraplength=375,
                    font=("Segoe UI", 9)).grid(row=8, column=0, columnspan=2, sticky="w", pady=12)

    def _bind(self) -> None:
        self.root.bind("<KeyPress>", self._key_press, add="+")
        self.root.bind("<KeyRelease>", self._key_release, add="+")
        self.root.bind("<ButtonRelease-1>", lambda _e: self._release_mouse(), add="+")
        self.root.bind("<FocusOut>", self._schedule_focus_check, add="+")
        self.root.bind("<MouseWheel>", self._mouse_wheel, add="+")

    def _mouse_wheel(self, event: tk.Event) -> str | None:
        widget = event.widget
        while widget is not None:
            canvas = self._scroll_regions.get(widget)
            if canvas is not None:
                steps = int(-event.delta / 120) or (-1 if event.delta > 0 else 1)
                canvas.yview_scroll(steps, "units")
                return "break"
            widget = getattr(widget, "master", None)
        return None

    def _set_source(self, action: str, source: str, pressed: bool) -> None:
        sources = self._drive_sources.setdefault(action, set())
        was_pressed = bool(sources)
        if pressed:
            sources.add(source)
        else:
            sources.discard(source)
        if was_pressed != bool(sources):
            self.runtime.set_drive_action(action, bool(sources))

    def _release_mouse(self) -> None:
        for action in tuple(self._drive_sources):
            self._set_source(action, "mouse", False)

    def _clear_inputs(self) -> None:
        for action, sources in self._drive_sources.items():
            if sources:
                self.runtime.set_drive_action(action, False)
            sources.clear()
        self._keys.clear()

    def _key_press(self, event: tk.Event) -> str | None:
        key = event.keysym.lower()
        if key == "escape":
            self._stop()
            return "break"
        if is_text_input(self.root.focus_get()):
            return None
        if key == "space":
            self._stop()
            return "break"
        if key in self._keys:
            return "break" if key in KEY_ACTIONS or key in {"g", "h"} else None
        self._keys.add(key)
        if key in KEY_ACTIONS:
            self._set_source(KEY_ACTIONS[key], f"key:{key}", True)
            return "break"
        if key == "g":
            if str(self.grab_button["state"]) == "normal":
                self._call(self.runtime.grab)
            else:
                self._last_message = self.task_hint.cget("text") or "Grab is unavailable. Check calibration and detector status."
            return "break"
        if key == "h":
            if str(self.drop_button["state"]) == "normal":
                self._call(self.runtime.drop)
            else:
                self._last_message = self.task_hint.cget("text") or "Drop is unavailable. Check calibration."
            return "break"
        return None

    def _key_release(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        self._keys.discard(key)
        if key in KEY_ACTIONS:
            self._set_source(KEY_ACTIONS[key], f"key:{key}", False)

    def _schedule_focus_check(self, _event: tk.Event | None = None) -> None:
        if self._closed:
            return
        callback_id: str | None = None
        def check() -> None:
            if callback_id is not None:
                self._focus_after_ids.discard(callback_id)
            self._check_focus()
        callback_id = self.root.after(30, check)
        self._focus_after_ids.add(callback_id)

    def _check_focus(self) -> None:
        if self._closed:
            return
        try:
            focus = self.root.focus_displayof()
            if focus is None or focus.winfo_toplevel() != self.root:
                self._clear_inputs()
                self.runtime.stop("Window lost focus")
        except tk.TclError:
            pass

    def _stop(self) -> None:
        self._clear_inputs()
        self._call(self.runtime.stop, "Operator stop")

    def _call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            previous = self._last_message
            result = function(*args, **kwargs)
            if self._last_message == previous:
                self._last_message = str(getattr(self.runtime, "message", ""))
            self._runtime_message_seen = str(getattr(self.runtime, "message", ""))
            return result
        except Exception as exc:
            self._last_message = str(exc)
            if hasattr(self, "footer"):
                self.footer.configure(text=self._last_message, fg=RED)
            return None

    def _confidence_changed(self, value: str) -> None:
        confidence = float(value)
        self.confidence_value.configure(text=f"{confidence:.2f}")
        self._call(self.runtime.set_confidence, confidence)

    def _save_frame(self) -> None:
        result = self._call(self.runtime.save_frame)
        if result:
            self._last_message = f"Saved camera frame: {result}"

    def _ensure_idle(self, *, disconnected: bool = False) -> None:
        snapshot = self.runtime.snapshot()
        if snapshot.get("busy"):
            raise ValueError("Stop the active routine before changing calibration.")
        if disconnected and snapshot.get("connected") and not self.runtime.demo:
            raise ValueError("Disconnect before changing robot connection settings.")

    def _edit_config(self, edit: Any, *, disconnected: bool = False) -> bool:
        self._ensure_idle(disconnected=disconnected)
        previous = copy.deepcopy(self.runtime.config.data)
        try:
            edit(self.runtime.config.data)
            result = self.runtime.save_settings()
            if result is False:
                raise ValueError(getattr(self.runtime, "message", "Configuration was not saved."))
        except Exception:
            self.runtime.config.data.clear()
            self.runtime.config.data.update(previous)
            raise
        self._last_message = "Saved calibration in demo memory." if self.runtime.demo else "Saved configuration."
        return True

    def _save_robot_settings(self) -> None:
        def action() -> None:
            values = {key: var.get().strip() for key, var in self.robot_fields.items()}
            if not values["ip"] or any(not values[key].startswith("/") for key in ("camera_topic", "cmd_vel_topic", "servo_topic")):
                raise ValueError("Enter a robot address and topic names beginning with /.")
            for key in ("rosbridge_port", "video_port"):
                values[key] = int(values[key])
                if not 1 <= values[key] <= 65535:
                    raise ValueError("Ports must be between 1 and 65535.")
            values["topics_verified"] = self.topics_verified.get()
            self._edit_config(lambda data: data["robot"].update(values), disconnected=True)
        self._call(action)

    @staticmethod
    def _invalidate_calibration(data: dict) -> None:
        data["alignment"]["calibrated"] = False
        for pose in data.get("poses", {}).values():
            pose["verified"] = False

    def _save_servo(self) -> None:
        def action() -> None:
            servo = {key: variable.get().strip() for key, variable in self.servo_fields.items()}
            servo.update({key: int(servo[key]) for key in ("id", "min", "max")})
            if not servo["name"] or not 1 <= servo["id"] <= 253 or not 0 <= servo["min"] < servo["max"] <= 1000:
                raise ValueError("Provide a joint name, ID 1–253, and 0 ≤ minimum < maximum ≤ 1000.")
            servo["verified"] = self.servo_verified.get()
            def edit(data: dict) -> None:
                data["servos"] = [item for item in data["servos"] if int(item["id"]) != servo["id"]] + [servo]
                data["servos"].sort(key=lambda item: int(item["id"]))
                self._invalidate_calibration(data)
            self._edit_config(edit)
            self._servo_signature = ""
        self._call(action)

    def _remove_servo(self) -> None:
        def action() -> None:
            servo_id = int(self.servo_fields["id"].get())
            def edit(data: dict) -> None:
                if not any(int(servo["id"]) == servo_id for servo in data["servos"]):
                    raise ValueError("Select an existing servo ID to remove.")
                data["servos"] = [servo for servo in data["servos"] if int(servo["id"]) != servo_id]
                self._invalidate_calibration(data)
            self._edit_config(edit)
            self._servo_signature = ""
        self._call(action)

    def _select_servo(self, _event: tk.Event | None = None) -> None:
        selection = self.servo_tree.selection()
        if not selection:
            return
        servo_id = int(selection[0])
        for servo in self.runtime.config.data["servos"]:
            if int(servo["id"]) == servo_id:
                for key, variable in self.servo_fields.items():
                    variable.set(str(servo[key]))
                self.servo_verified.set(bool(servo.get("verified")))
                break

    def _save_pose(self) -> None:
        def action() -> None:
            self._ensure_idle()
            name = self.pose_name.get().strip()
            duration, pause = float(self.pose_duration.get()), float(self.pose_pause.get())
            if not name or not all(math.isfinite(value) for value in (duration, pause)) or not 0.1 <= duration <= 15 or not 0 <= pause <= 15:
                raise ValueError("Enter a pose name, duration 0.1–15 seconds, and pause 0–15 seconds.")
            ids_text = self.pose_joint_ids.get().replace(",", " ").strip()
            ids = [int(value) for value in ids_text.split()] if ids_text else None
            if ids is not None and (len(set(ids)) != len(ids) or any(not 1 <= sid <= 253 for sid in ids)):
                raise ValueError("Enter unique joint IDs separated by commas or spaces.")
            if name in ("open", "grip") and ids is None:
                raise ValueError("For open/grip, enter only the gripper joint ID(s), so opening never repositions the arm.")
            self.runtime.save_pose(name, duration, pause, self.pose_verified.get(), servo_ids=ids)
            self._update_pose_list()
        self._call(action)

    def _capture_alignment(self) -> None:
        def action() -> None:
            self._ensure_idle()
            target_x, target_height = fresh_largest_box(self.runtime.get_detection(),
                float(self.runtime.config.data["alignment"]["max_frame_age_s"]), self.runtime.demo)
            self.alignment_fields["target_x"].set(f"{target_x:.4f}")
            self.alignment_fields["target_height"].set(f"{target_height:.4f}")
            self._last_message = "Captured target only. Enter verified tolerances/signs, then save calibration."
        self._call(action)

    def _save_alignment(self) -> None:
        def action() -> None:
            self._ensure_idle()
            values = {key: float(variable.get()) for key, variable in self.alignment_fields.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Calibration values must be finite numbers.")
            if not .01 <= values["target_x"] <= .99 or not .01 <= values["target_height"] <= .95:
                raise ValueError("Pickup center X must be 0.01–0.99 and height 0.01–0.95.")
            if any(not .001 <= values[key] <= .20 for key in ("x_tolerance", "height_tolerance")):
                raise ValueError("Enter measured tolerances from 0.001 to 0.20.")
            values.update({key: int(variable.get()) for key, variable in self.alignment_signs.items()})
            if any(values[key] not in (-1, 1) for key in self.alignment_signs):
                raise ValueError("Verify both movement signs: choose +1 or −1.")
            self.runtime.save_alignment(**values)
        self._call(action)

    def _rebuild_servos(self) -> None:
        for child in self.arm_frame.winfo_children():
            child.destroy()
        self._servo_widgets.clear()
        servos = self.runtime.config.data["servos"]
        self.servo_tree.delete(*self.servo_tree.get_children())
        if not servos:
            self._label(self.arm_frame, "No joints configured", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 7))
            self._label(self.arm_frame, "Open Calibrate → Servos to register the IDs and safe ranges you measured. "
                        "No real servo values are assumed.", muted=True, wraplength=360).pack(anchor="w")
        for servo in servos:
            sid = int(servo["id"])
            checked = bool(servo.get("verified"))
            self.servo_tree.insert("", "end", iid=str(sid), values=(sid, servo["name"],
                f"{servo['min']}–{servo['max']}", "yes" if checked else "no"))
            row = ttk.Frame(self.arm_frame, padding=(0, 6, 5, 12))
            row.pack(fill="x")
            title = ttk.Frame(row)
            title.pack(fill="x")
            self._label(title, f"{servo['name']}  /  ID {sid}", font=("Segoe UI", 10, "bold")).pack(side="left")
            self._label(title, "verified" if checked else "unverified", foreground=GREEN if checked else AMBER,
                        font=("Segoe UI", 8)).pack(side="right")
            commanded = self._label(row, "Commanded: unknown", muted=True, font=("Segoe UI", 9))
            commanded.pack(anchor="w", pady=(3, 4))
            target = tk.StringVar(value="")
            slider_var = tk.DoubleVar(value=(float(servo["min"]) + float(servo["max"])) / 2)
            slider = ttk.Scale(row, from_=servo["min"], to=servo["max"], variable=slider_var,
                               command=lambda value, var=target: var.set(str(round(float(value)))))
            slider.pack(fill="x")
            bottom = ttk.Frame(row)
            bottom.pack(fill="x", pady=(5, 0))
            self._label(bottom, "Target", muted=True, font=("Segoe UI", 9)).pack(side="left")
            entry = ttk.Entry(bottom, textvariable=target, width=6)
            entry.pack(side="left", padx=(5, 10))
            duration = tk.StringVar(value="1.5")
            ttk.Entry(bottom, textvariable=duration, width=4).pack(side="left")
            self._label(bottom, "s", muted=True).pack(side="left", padx=(3, 6))
            button = ttk.Button(bottom, text="Send", command=lambda servo_id=sid, var=target, dur=duration:
                self._send_servo(servo_id, var, dur))
            button.pack(side="right")
            self._servo_widgets[sid] = {"commanded": commanded, "button": button,
                                        "target": target, "verified": checked}
        self._update_pose_list()

    def _send_servo(self, servo_id: int, target: tk.StringVar, duration: tk.StringVar) -> None:
        def action() -> None:
            if not target.get().strip():
                raise ValueError("Choose a target position first; startup does not assume the joint's position.")
            seconds = float(duration.get())
            if not math.isfinite(seconds) or not 0.1 <= seconds <= 15:
                raise ValueError("Servo move duration must be 0.1–15 seconds; use 1–2 seconds when learning.")
            self.runtime.set_servo(servo_id, int(target.get()), seconds)
        self._call(action)

    def _update_pose_list(self) -> None:
        self.pose_tree.delete(*self.pose_tree.get_children())
        for name, pose in self.runtime.config.data.get("poses", {}).items():
            joints = ",".join(str(sid) for sid in pose.get("positions", {}))
            self.pose_tree.insert("", "end", iid=name, values=(name, joints, pose.get("duration", "?"), "yes" if pose.get("verified") else "no"))

    def _select_pose(self, _event: tk.Event | None = None) -> None:
        selection = self.pose_tree.selection()
        if not selection:
            return
        name = selection[0]
        pose = self.runtime.config.data.get("poses", {}).get(name)
        if pose is not None:
            self.pose_name.set(name)
            self.pose_joint_ids.set(", ".join(str(sid) for sid in pose.get("positions", {})))
            self.pose_duration.set(str(pose.get("duration", 1.5)))
            self.pose_pause.set(str(pose.get("pause", .2)))
            self.pose_verified.set(bool(pose.get("verified")))

    def _render(self, packet: FramePacket | DetectionPacket | None, *, force: bool = False) -> None:
        if self._closed:
            return
        self._last_display = packet
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        key = (getattr(packet, "frame_id", None), getattr(packet, "completed_at", None), width, height)
        if key == self._display_key and not force:
            return
        self._display_key = key
        self.canvas.delete("all")
        if packet is None:
            self._photo = None
            self.canvas.create_line(width / 2 - 30, height / 2 - 34, width / 2 + 30, height / 2 - 34, fill=BORDER, width=2)
            self.canvas.create_oval(width / 2 - 16, height / 2 - 50, width / 2 + 16, height / 2 - 18, outline=BORDER, width=2)
            self.canvas.create_text(width / 2, height / 2 + 12, text="AWAITING CAMERA", fill=MUTED,
                                    font=("Segoe UI", 13, "bold"))
            self.canvas.create_text(width / 2, height / 2 + 40, text="Connect the rover, or launch with --demo", fill=MUTED,
                                    font=("Segoe UI", 10))
            return
        source = packet.image.size
        scale, ox, oy = fit_geometry(source, (width, height))
        image = packet.image.resize((max(1, round(source[0] * scale)), max(1, round(source[1] * scale))), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image, master=self.root)
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)
        if isinstance(packet, DetectionPacket):
            color = AMBER if packet.simulated else GREEN
            for box in packet.boxes:
                x1, y1, x2, y2 = canvas_box(box, source, (width, height))
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                label = f"{'SIM' if packet.simulated else 'RED BULL'}  {box.confidence:.2f}"
                label_y = max(oy, y1 - 23)
                self.canvas.create_rectangle(x1, label_y, x1 + 122, label_y + 22, fill=PANEL, outline="")
                self.canvas.create_text(x1 + 6, label_y + 11, anchor="w", text=label, fill=color,
                                        font=("Segoe UI", 9, "bold"))
        if self.runtime.demo:
            self.canvas.create_rectangle(ox + 12, oy + 12, ox + 228, oy + 40, fill="#573e23", outline="")
            self.canvas.create_text(ox + 24, oy + 26, text="SIMULATED · NO REAL DETECTOR", anchor="w",
                                    fill="#ffdaa2", font=("Segoe UI", 9, "bold"))

    def _refresh(self) -> None:
        if self._closed:
            return
        try:
            self.runtime.heartbeat()
            snapshot = self.runtime.snapshot()
            connected, busy = bool(snapshot.get("connected")), bool(snapshot.get("busy"))
            self.status_labels["connection"].configure(text=snapshot.get("connection_status", "Disconnected"),
                foreground=GREEN if connected else AMBER)
            self.status_labels["vision"].configure(text=snapshot.get("detector_status", "No detector loaded"),
                foreground=GREEN if snapshot.get("model_available") else AMBER)
            self.status_labels["mission"].configure(text=snapshot.get("task_status", "Idle"),
                foreground=RED if snapshot.get("task_error") else GREEN if busy else TEXT)
            self.camera_status.configure(text=snapshot.get("camera_status", "Camera disconnected"))
            self.owner_label.configure(text=f"Wheel control: {snapshot.get('owner', 'stopped')}")
            requested = bool(snapshot.get("connection_requested", connected))
            self.connect_button.configure(state="disabled" if connected or requested or self.runtime.demo else "normal")
            self.disconnect_button.configure(state="normal" if (connected or requested) and not self.runtime.demo else "disabled")
            grab_errors = snapshot.get("grab_errors", [])
            drop_errors = snapshot.get("drop_errors", [])
            self.grab_button.configure(state="normal" if connected and not busy and not grab_errors and snapshot.get("model_available") else "disabled")
            self.drop_button.configure(state="normal" if connected and not busy and not drop_errors else "disabled")
            hints = []
            if not connected:
                hints.append("Connect to the rover to operate it.")
            if grab_errors:
                hints.append("Grab: " + "; ".join(str(item) for item in grab_errors[:2]))
            if drop_errors:
                hints.append("Drop: " + "; ".join(str(item) for item in drop_errors[:1]))
            if busy:
                hints = ["Routine running. Manual controls are locked. Stop cancels remaining commands."]
            self.task_hint.configure(text=" ".join(hints) or "Ready. Find a can, Grab, drive home, then Drop horizontally.")
            for button in self.drive_buttons:
                button.configure(state="normal" if connected and not busy else "disabled")
            signature = json.dumps(self.runtime.config.data["servos"], sort_keys=True)
            if signature != self._servo_signature:
                self._servo_signature = signature
                self._rebuild_servos()
            commanded = snapshot.get("commanded_positions", {})
            for sid, widgets in self._servo_widgets.items():
                value = commanded.get(sid, commanded.get(str(sid)))
                widgets["commanded"].configure(text=f"Commanded: {value if value is not None else 'unknown'}")
                widgets["button"].configure(state="normal" if connected and not busy and widgets["verified"] else "disabled")
            packet = self.runtime.get_display()
            self._render(packet)
            if packet is not None:
                age = max(0.0, time.monotonic() - packet.captured_at)
                self.frame_label.configure(text=f"{'SIM  /  ' if self.runtime.demo else ''}FRAME {packet.frame_id} · {age:.1f}s old")
                inference = (packet.completed_at - packet.captured_at) * 1000 if isinstance(packet, DetectionPacket) else None
                count = len(packet.boxes) if isinstance(packet, DetectionPacket) else 0
                self.video_stats.configure(text=f"{count} can{'s' if count != 1 else ''} · " +
                    (f"receive-to-result {inference:.0f} ms · " if inference is not None else "") +
                    "frame age excludes camera/server delay.")
                if age > float(self.runtime.config.data["vision"]["max_frame_age_s"]):
                    self.frame_label.configure(foreground=RED, text=f"STALE FRAME · {age:.1f}s since receipt")
                else:
                    self.frame_label.configure(foreground=MUTED)
            runtime_message = str(getattr(self.runtime, "message", ""))
            if runtime_message != self._runtime_message_seen:
                self._last_message = runtime_message
                self._runtime_message_seen = runtime_message
            message = self._last_message or runtime_message
            self.footer.configure(text=message or "W/S drive · A/D strafe · Q/E rotate · G grab · H drop · Space stop · Esc stops even while typing",
                                  fg=MUTED)
        except Exception as exc:
            self.footer.configure(text=f"Cockpit update: {exc}", fg=RED)
        self._after_id = self.root.after(50, self._refresh)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = set(self._focus_after_ids)
        if self._after_id is not None:
            pending.add(self._after_id)
        for callback_id in pending:
            try:
                self.root.after_cancel(callback_id)
            except tk.TclError:
                pass
        self._focus_after_ids.clear()
        self._after_id = None
        # ImageTk owns Tcl objects; release it on the Tk-creating thread, not
        # during a later garbage collection initiated by a camera worker.
        self._photo = None
        self._last_display = None
        self._clear_inputs()
        try:
            self.runtime.close()
        finally:
            self.root.destroy()
