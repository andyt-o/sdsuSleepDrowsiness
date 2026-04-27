from pathlib import Path
import mediapipe as mp
import cv2
import numpy as np
import polars as pl
import xgboost as xgb
import time
import threading
import sounddevice as sd
from src.sleepDrowsiness.training.createDataset import SCHEMA

DROWSY_THRESHOLD_SECONDS = 1.5  # Time before triggering the drowsy alert
ALERT_REPEAT_SECONDS = (
    1  # Delay between each beep repetition (longer = less aggressive)
)
SAMPLE_RATE = 44100  # Audio sample rate in Hz


class setupModel:
    def __init__(self, modelName: str, webcamFPS: int, webcamDims: tuple):
        OPTIONS = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="./face_landmarker.task"
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_face_detection_confidence=0.25,
            output_face_blendshapes=True,
            num_faces=1,
            result_callback=self.handleCallback,
        )
        self.MODEL = xgb.Booster()
        self.MODEL.load_model(Path(modelName))
        self.FPS = webcamFPS
        self.WIDTH, self.HEIGHT = webcamDims
        self.LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(OPTIONS)
        self.RES = None
        self.LOCALDATA = SCHEMA.copy()
        self.LOCALDATA.pop("label")

        # --- Time tracking ---
        self.drowsy_since: float | None = None  # Timestamp when drowsy state started
        self.alert_active: bool = False  # Whether the alert is currently active
        self.alert_last_played: float = 0.0  # Timestamp of the last beep played

        # --- Pre-generate the soft alert sound ---
        self.ALERT_SOUND = self._generate_soft_chime()

        self.createWebcam()

    def _generate_soft_chime(self) -> np.ndarray:
        """
        Generate a soft two-tone chime using sine waves and a smooth fade envelope.
        Blends two harmonically related frequencies for a gentle, bell-like sound.
        Much softer than a raw buzzer beep.

        @returns   np.ndarray — Float32 mono audio samples ready for sounddevice
        """
        duration = 1.2  # seconds — long enough to be noticed, short enough to not annoy
        freq1 = 440.0  # Hz — A4, warm and soft
        freq2 = 528.0  # Hz — slightly above, creates a pleasant harmonic blend
        volume = 0.4  # Keep it gentle (0.0 to 1.0)

        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Mix two sine waves for a richer, bell-like tone
        wave = np.sin(2 * np.pi * freq1 * t) * 0.6 + np.sin(2 * np.pi * freq2 * t) * 0.4

        # Smooth fade-in / fade-out envelope to eliminate clicks and harsh edges
        # Attack: first 10% of the sound fades in
        # Release: last 40% of the sound fades out gradually
        envelope = np.ones(n_samples)
        attack_end = int(0.10 * n_samples)
        release_start = int(0.60 * n_samples)

        envelope[:attack_end] = np.linspace(0, 1, attack_end)  # gentle fade in
        envelope[release_start:] = np.linspace(
            1, 0, n_samples - release_start
        )  # long fade out

        wave = wave * envelope * volume
        return wave.astype(np.float32)

    def _play_alert(self):
        """
        Play the soft chime using sounddevice in a blocking call.
        Must be called inside a daemon thread to avoid blocking the webcam loop.

        @requires  self.ALERT_SOUND — Pre-generated numpy audio array
        @returns   None
        """
        sd.play(self.ALERT_SOUND, samplerate=SAMPLE_RATE)
        sd.wait()  # Wait for playback to finish before the thread exits

    def createWebcam(self):
        """
        Main webcam loop: captures frames, runs MediaPipe + XGBoost inference,
        updates the drowsy timer, triggers the sound alert, and renders the overlay.

        @requires  self.MODEL      — Loaded XGBoost booster
        @requires  self.LANDMARKER — MediaPipe FaceLandmarker in LIVE_STREAM mode
        @returns   None
        """
        print(self.findCameras())
        webcam = cv2.VideoCapture(0)

        while webcam.isOpened():
            suc, frame = webcam.read()
            webcam.set(cv2.CAP_PROP_FPS, self.FPS)
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

            if not suc:
                print(
                    "Error: Webcam is not loading anything, please check your hardware."
                )
                break

            timestamp = int(time.time() * 1000)
            mpImage = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            self.LANDMARKER.detect_async(mpImage, timestamp)

            modelOutput = None
            instant_label = None  # What the model predicts for the current frame

            if self.RES is not None:
                y = self.RES.face_blendshapes
                if len(y) > 0:
                    # Fill feature dict with blendshape scores
                    for i in y[0]:
                        self.LOCALDATA[i.category_name] = i.score
                    modelOutput = self.MODEL.predict(
                        xgb.DMatrix(pl.DataFrame(data=self.LOCALDATA))
                    )
                    # Probability < 0.3 → Drowsy, >= 0.3 → Awake
                    instant_label = "Drowsy" if modelOutput < 0.3 else "Awake"

            # --- Time-based drowsy logic ---
            now = time.time()
            display_label, color = self._update_drowsy_timer(instant_label, now)

            # --- Sound alert: play chime every ALERT_REPEAT_SECONDS while alert is active ---
            if (
                self.alert_active
                and (now - self.alert_last_played) >= ALERT_REPEAT_SECONDS
            ):
                self.alert_last_played = now
                threading.Thread(target=self._play_alert, daemon=True).start()

            # --- Display ---
            # Line 1: final state label (with 2s delay applied)
            cv2.putText(
                frame,
                display_label,
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                3,
                cv2.LINE_AA,
            )

            # Line 2: raw model score for debugging
            if modelOutput is not None:
                score_text = f"Score: {float(modelOutput[0]):.3f}  [{instant_label}]"
                cv2.putText(
                    frame,
                    score_text,
                    (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

            # Line 3: progress bar showing time elapsed since drowsy started (0 → 1.5s)
            if self.drowsy_since is not None:
                elapsed = min(now - self.drowsy_since, DROWSY_THRESHOLD_SECONDS)
                bar_width = int((elapsed / DROWSY_THRESHOLD_SECONDS) * 300)
                cv2.rectangle(frame, (25, 110), (325, 130), (50, 50, 50), -1)
                cv2.rectangle(
                    frame, (25, 110), (25 + bar_width, 130), (0, 100, 255), -1
                )
                cv2.putText(
                    frame,
                    f"{elapsed:.1f}s / {DROWSY_THRESHOLD_SECONDS}s",
                    (25, 148),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("Sleep Drowsiness", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        webcam.release()

    def _update_drowsy_timer(self, instant_label: str | None, now: float):
        """
        Update the drowsiness timer and return the display label and BGR color.
        The alert only activates after DROWSY_THRESHOLD_SECONDS of continuous drowsy detection.

        @requires  instant_label: str | None — Current frame prediction ("Drowsy", "Awake", or None)
        @requires  now          : float      — Current timestamp in seconds
        @assigns   self.drowsy_since      — Set on first drowsy frame, cleared on awake
        @assigns   self.alert_active      — True once threshold is exceeded
        @assigns   self.alert_last_played — Reset to 0 when driver wakes up
        @returns   tuple(str, tuple) — (display text, BGR color)
        """
        if instant_label == "Drowsy":
            # Start the timer on the first drowsy frame
            if self.drowsy_since is None:
                self.drowsy_since = now

            elapsed = now - self.drowsy_since
            if elapsed >= DROWSY_THRESHOLD_SECONDS:
                self.alert_active = True
            # Under the threshold: keep showing Awake to avoid premature alerts
        else:
            # Driver is awake or face not detected: reset everything
            self.drowsy_since = None
            self.alert_active = False
            self.alert_last_played = 0.0

        if instant_label is None:
            return "Can't Detect Face", (100, 100, 100)

        if self.alert_active:
            return "DROWSY !", (0, 0, 255)  # Bright red
        else:
            return "Awake", (0, 220, 0)  # Green

    def findCameras(self):
        """
        Scan the first 5 camera indices and return those that are accessible.

        @returns list[int] — List of available camera indices
        """
        availableCameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                availableCameras.append(i)
                cap.release()
        return availableCameras

    def handleCallback(self, r, i, t):
        """
        MediaPipe async callback: stores the latest detection result.

        @requires  r — MediaPipe FaceLandmarkerResult
        @assigns   self.RES — Updated with the latest result each frame
        @returns   None
        """
        self.RES = r
