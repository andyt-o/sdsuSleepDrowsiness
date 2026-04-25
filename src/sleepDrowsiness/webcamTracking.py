from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import mediapipe as mp
import numpy as np
import polars as pl
import xgboost as xgb

from src.sleepDrowsiness.training.createDataset import SCHEMA


CAMERA_INDEX = 0
DEFAULT_REFRESH_MS = 10
DROWSY_THRESHOLD = 0.5
MAX_ALERT_LEVEL = 100.0
SUPPORTED_VIDEO_TYPES = (
    ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv *.m4v"),
    ("MP4 files", "*.mp4"),
    ("All files", "*.*"),
)


@dataclass
class TrackingState:
    label: str = "Starting webcam"
    confidence: float | None = None
    detail: str = "Waiting for the first frame."
    is_drowsy: bool = False


class DrowsinessTracker:
    def __init__(self, model_name: str, webcam_fps: int, webcam_dims: tuple[int, int]):
        self.fps = webcam_fps
        self.width, self.height = webcam_dims
        self.model = self._load_model(model_name)
        self.feature_names = [column for column in SCHEMA if column != "label"]
        self.feature_schema = {column: pl.Float32 for column in self.feature_names}
        self.local_data = self._create_input_row()

        self._result_lock = threading.Lock()
        self._latest_result = None
        self.landmarker = self._create_landmarker()

    def close(self) -> None:
        self.landmarker.close()

    def predict_frame(self, frame: np.ndarray) -> TrackingState:
        timestamp_ms = int(time.time() * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.landmarker.detect_async(mp_image, timestamp_ms)

        with self._result_lock:
            result = self._latest_result

        if result is None or not result.face_blendshapes:
            return TrackingState(label="Can't Detect Face", detail="No face detected.")

        blendshapes = result.face_blendshapes[0]
        for blendshape in blendshapes:
            self.local_data[blendshape.category_name] = blendshape.score

        score = self._predict_score()
        is_drowsy = score < DROWSY_THRESHOLD
        label = "Drowsy" if is_drowsy else "Awake"
        detail = f"Face detected. Model score: {score:.3f}"

        return TrackingState(
            label=label,
            confidence=score,
            detail=detail,
            is_drowsy=is_drowsy,
        )

    def handle_callback(self, result, _image, _timestamp) -> None:
        with self._result_lock:
            self._latest_result = result

    def _create_landmarker(self):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="./face_landmarker.task",
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=True,
            num_faces=1,
            result_callback=self.handle_callback,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _create_input_row(self) -> dict:
        return {column: 0.0 for column in self.feature_names}

    def _load_model(self, model_name: str) -> xgb.Booster:
        model = xgb.Booster()
        model.load_model(Path(model_name))
        return model

    def _predict_score(self) -> float:
        data = pl.DataFrame(data=self.local_data, schema=self.feature_schema)
        prediction = self.model.predict(xgb.DMatrix(data))
        return float(np.asarray(prediction).ravel()[0])


class DrowsinessApp:
    def __init__(self, tracker: DrowsinessTracker):
        self.tracker = tracker
        self.cap: cv2.VideoCapture | None = None
        self.camera_indexes = self.find_cameras()
        self.alert_level = 0.0
        self.drowsy_started_at: float | None = None
        self.last_progress_update = time.monotonic()

        self.root = tk.Tk()
        self.root.title("Sleep Drowsiness")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_text = tk.StringVar(value="Starting webcam")
        self.detail_text = tk.StringVar(value="Waiting for webcam frames.")
        self.score_text = tk.StringVar(value="Score: --")
        self.alert_text = tk.StringVar(value="Drowsiness alert: 0%")
        self.source_text = tk.StringVar(value="No source selected")
        self.camera_choice = tk.StringVar()

        self.video_label: ttk.Label | None = None
        self.alert_bar: ttk.Progressbar | None = None
        self.camera_dropdown: ttk.Combobox | None = None
        self._photo = None
        self._running = True

        self._build_layout()
        self._start_default_source()

    def run(self) -> None:
        self._update_frame()
        self.root.mainloop()

    def close(self) -> None:
        self._running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.tracker.close()
        self.root.destroy()

    def _build_layout(self) -> None:
        self.root.configure(bg="#111827")
        self._configure_styles()

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")

        video_frame = ttk.Frame(
            main_frame,
            style="Video.TFrame",
            width=self.tracker.width,
            height=self.tracker.height,
        )
        video_frame.grid(row=0, column=0)
        video_frame.grid_propagate(False)

        self.video_label = ttk.Label(video_frame)
        self.video_label.place(
            x=0, y=0, width=self.tracker.width, height=self.tracker.height
        )

        control_panel = ttk.Frame(
            main_frame,
            padding=(18, 18, 18, 18),
            style="Panel.TFrame",
            width=300,
        )
        control_panel.grid(row=0, column=1, sticky="ns")
        control_panel.grid_propagate(False)

        ttk.Label(
            control_panel,
            text="Control Panel",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 20))

        ttk.Label(
            control_panel,
            textvariable=self.status_text,
            font=("Segoe UI", 28, "bold"),
            wraplength=250,
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        ttk.Label(
            control_panel,
            textvariable=self.detail_text,
            font=("Segoe UI", 11),
            wraplength=250,
        ).grid(row=2, column=0, sticky="w", pady=(0, 18))

        ttk.Label(
            control_panel,
            textvariable=self.score_text,
            font=("Segoe UI", 12),
        ).grid(row=3, column=0, sticky="w", pady=(0, 22))

        ttk.Label(
            control_panel,
            text="Webcam",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

        camera_values = [f"Webcam {index}" for index in self.camera_indexes]
        self.camera_dropdown = ttk.Combobox(
            control_panel,
            textvariable=self.camera_choice,
            values=camera_values,
            state="readonly" if camera_values else "disabled",
        )
        self.camera_dropdown.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.camera_dropdown.bind("<<ComboboxSelected>>", self._select_webcam)

        ttk.Button(
            control_panel,
            text="Open Video File",
            command=self._select_video_file,
        ).grid(row=6, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(
            control_panel,
            textvariable=self.source_text,
            font=("Segoe UI", 10),
            wraplength=250,
        ).grid(row=7, column=0, sticky="w", pady=(0, 24))

        ttk.Label(
            control_panel,
            text="Drowsiness Duration",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=8, column=0, sticky="w", pady=(0, 8))

        self.alert_bar = ttk.Progressbar(
            control_panel,
            orient="horizontal",
            length=250,
            mode="determinate",
            maximum=MAX_ALERT_LEVEL,
        )
        self.alert_bar.grid(row=9, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(
            control_panel,
            textvariable=self.alert_text,
            font=("Segoe UI", 11),
        ).grid(row=10, column=0, sticky="w")

        ttk.Button(
            control_panel,
            text="Quit",
            command=self.close,
        ).grid(row=11, column=0, sticky="ew", pady=(36, 0))

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#111827")
        style.configure("Video.TFrame", background="#020617")
        style.configure("Panel.TFrame", background="#1f2937")
        style.configure("TLabel", background="#1f2937", foreground="#f9fafb")
        style.configure("TButton", padding=(12, 8))
        style.configure(
            "Horizontal.TProgressbar",
            background="#f97316",
            troughcolor="#374151",
            bordercolor="#374151",
            lightcolor="#f97316",
            darkcolor="#ea580c",
        )

    def _create_webcam(self, camera_index: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FPS, self.tracker.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.tracker.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.tracker.height)
        return cap

    def _start_default_source(self) -> None:
        if self.camera_indexes:
            default_index = (
                CAMERA_INDEX
                if CAMERA_INDEX in self.camera_indexes
                else self.camera_indexes[0]
            )
            self._open_webcam(default_index)
            return

        self._show_status(
            TrackingState(
                label="No Webcam Found",
                detail="Open a video file to test the app without a webcam.",
            )
        )
        self.source_text.set("No webcam found. Use Open Video File.")

    def _select_webcam(self, _event=None) -> None:
        selected = self.camera_choice.get()
        if not selected:
            return

        camera_index = int(selected.replace("Webcam ", ""))
        self._open_webcam(camera_index)

    def _select_video_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=SUPPORTED_VIDEO_TYPES,
        )
        if not path:
            return

        self._open_video_file(Path(path))

    def _open_webcam(self, camera_index: int) -> None:
        cap = self._create_webcam(camera_index)
        if not cap.isOpened():
            cap.release()
            self._show_status(
                TrackingState(
                    label="Webcam Error",
                    detail=f"Webcam {camera_index} could not be opened.",
                )
            )
            return

        self._set_capture(cap)
        self.camera_choice.set(f"Webcam {camera_index}")
        self.source_text.set(f"Using Webcam {camera_index}")
        self._reset_alert_level()

    def _open_video_file(self, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            self._show_status(
                TrackingState(
                    label="Video Error",
                    detail=f"Could not open {path.name}.",
                )
            )
            return

        self._set_capture(cap)
        self.camera_choice.set("")
        self.source_text.set(f"Using video: {path.name}")
        self._reset_alert_level()

    def _set_capture(self, cap: cv2.VideoCapture) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = cap

    def _update_frame(self) -> None:
        if not self._running:
            return

        if self.cap is None or not self.cap.isOpened():
            self.root.after(250, self._update_frame)
            return

        success, frame = self.cap.read()
        if not success:
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(1, self._update_frame)
                return

            self._show_status(
                TrackingState(
                    label="Source Error",
                    detail="The selected source is not loading frames.",
                )
            )
            self.root.after(250, self._update_frame)
            return

        frame = cv2.resize(frame, (self.tracker.width, self.tracker.height))
        state = self.tracker.predict_frame(frame)
        self._show_status(state)
        self._update_alert_level(state)
        self._show_frame(frame)

        self.root.after(DEFAULT_REFRESH_MS, self._update_frame)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, encoded = cv2.imencode(".ppm", rgb_frame)
        if not success:
            return

        self._photo = tk.PhotoImage(data=encoded.tobytes(), format="PPM")
        if self.video_label is not None:
            self.video_label.configure(image=self._photo)

    def _show_status(self, state: TrackingState) -> None:
        self.status_text.set(state.label)
        self.detail_text.set(state.detail)

        if state.confidence is None:
            self.score_text.set("Score: --")
        else:
            self.score_text.set(f"Score: {state.confidence:.3f}")

    def _update_alert_level(self, state: TrackingState) -> None:
        now = time.monotonic()
        elapsed = now - self.last_progress_update
        self.last_progress_update = now

        if state.is_drowsy:
            if self.drowsy_started_at is None:
                self.drowsy_started_at = now
            drowsy_seconds = now - self.drowsy_started_at
            fill_rate = 12.0 + min(drowsy_seconds * 7.0, 48.0)
            self.alert_level += fill_rate * elapsed
        else:
            self.drowsy_started_at = None
            self.alert_level -= 18.0 * elapsed

        self.alert_level = max(0.0, min(MAX_ALERT_LEVEL, self.alert_level))

        if self.alert_bar is not None:
            self.alert_bar.configure(value=self.alert_level)
        self.alert_text.set(f"Drowsiness alert: {self.alert_level:.0f}%")

    def _reset_alert_level(self) -> None:
        self.alert_level = 0.0
        self.drowsy_started_at = None
        self.last_progress_update = time.monotonic()
        if self.alert_bar is not None:
            self.alert_bar.configure(value=self.alert_level)
        self.alert_text.set("Drowsiness alert: 0%")

    def find_cameras(self) -> list[int]:
        available_cameras = []
        for index in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
            cap.release()
        return available_cameras


class setupModel:
    def __init__(self, modelName: str, webcamFPS: int, webcamDims: tuple[int, int]):
        self.tracker = DrowsinessTracker(modelName, webcamFPS, webcamDims)
        self.app = DrowsinessApp(self.tracker)
        self.app.run()

    def findCameras(self):
        return self.app.find_cameras()
