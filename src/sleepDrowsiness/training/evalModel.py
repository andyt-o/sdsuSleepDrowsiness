from pathlib import Path
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from .createDataset import SCHEMA
import xgboost as xgb
import mediapipe as mp
import cv2
import numpy as np
import polars as pl
import os

load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="./face_landmarker.task"),
    output_face_blendshapes=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.2,
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)


class evalModel:
    def __init__(self, dataset):
        self.MODEL = self.loadModel(self.findModel())
        self.DATASET = self.loadDataset(dataset)["train"]
        self.inputData = SCHEMA
        self.testData()

    def loadDataset(self, dataset: str):
        x = load_dataset(dataset, token=HUGGING_FACE, streaming=True)
        x: Dataset = x.shuffle(seed=1)
        return x

    def findModel(self):
        modelPath = Path("src/sleepDrowsiness/models")
        files = [f for f in modelPath.glob("*") if f.is_file()]
        newestModel = max(files, key=lambda f: f.stat().st_ctime)
        return newestModel

    def loadModel(self, model):
        x = xgb.Booster()
        x.load_model(model)
        return x

    def testData(self):
        for i in self.DATASET:
            currImage = np.ascontiguousarray(np.array(i["image"]))
            cv2Image = cv2.cvtColor(currImage, cv2.COLOR_RGB2BGR)
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)

            res = landmarker.detect(mpImage).face_blendshapes

            if len(res) > 0:
                for i in res[0]:
                    self.inputData[i.category_name] = i.score
                print(self.inputData)

            cv2.putText(
                cv2Image,
                f"This person is {self.DATASET.features['label'].int2str(i['label'])}",
                (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.425,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("evalModel", cv2Image)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("e"):
                    break


def evaluateModel():
    testFile = evalModel("akahana/Driver-Drowsiness-Dataset")


if __name__ == "__main__":

    testFile = evalModel("akahana/Driver-Drowsiness-Dataset")
    try:
        print("\n")
    except Exception:
        print("Error: Model could not be found to test.")
