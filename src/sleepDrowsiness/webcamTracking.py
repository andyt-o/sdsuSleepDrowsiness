from pathlib import Path
import mediapipe as mp
import cv2
import numpy as np
import polars as pl
import xgboost as xgb
import time


class setupModel:
    def __init__(self, modelName: str, webcamFPS: int, webcamDims: tuple):
        OPTIONS = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="./face_landmarker.task"
            ),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=True,
            num_faces=1,
            result_callback=self.handleCallback,
        )
        self.MODEL = xgb.Booster().load_model((Path(modelName)))
        self.FPS = webcamFPS
        self.WIDTH, self.HEIGHT = webcamDims
        self.LANDMARKER = mp.tasks.vision.FaceLandmarker.create_from_options(OPTIONS)
        self.RES = None
        self.createWebcam()

    def createWebcam(self):
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
            print(self.RES.face_blendshapes)

            cv2.imshow("Sleep Drowsiness", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        webcam.release()

    def findCameras(self):
        availableCameras = []
        for i in range(25):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                availableCameras.append(i)
                cap.release()
        return availableCameras

    def handleCallback(self, r, i, t):
        self.RES = r
