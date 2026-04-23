from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path, PureWindowsPath
from dotenv import load_dotenv
import polars as pl
import numpy as np
import mediapipe as mp
import xgboost as xgb
import matplotlib as mpl
import datetime as dt
import cv2
import os

load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")

dataPath = Path("src/sleepDrowsiness/data")
dataName = dt.datetime.now().strftime("%H:%M:S_%m-%d-%Y")

mediaPipePath = "./face_landmarker.task"
baseOptions = mp.tasks.BaseOptions(model_asset_path=mediaPipePath)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=baseOptions,
    output_face_blendshapes=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.2,
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

SCHEMA = {
    "_neutral": pl.Float16,
    "browDownLeft": pl.Float16,
    "browDownRight": pl.Float16,
    "browInnerUp": pl.Float16,
    "browOuterUpLeft": pl.Float16,
    "browOuterUpRight": pl.Float16,
    "cheekPuff": pl.Float16,
    "cheekSquintLeft": pl.Float16,
    "cheekSquintRight": pl.Float16,
    "eyeBlinkLeft": pl.Float16,
    "eyeBlinkRight": pl.Float16,
    "eyeLookDownLeft": pl.Float16,
    "eyeLookDownRight": pl.Float16,
    "eyeLookInLeft": pl.Float16,
    "eyeLookInRight": pl.Float16,
    "eyeLookOutLeft": pl.Float16,
    "eyeLookOutRight": pl.Float16,
    "eyeLookUpLeft": pl.Float16,
    "eyeLookUpRight": pl.Float16,
    "eyeSquintLeft": pl.Float16,
    "eyeSquintRight": pl.Float16,
    "eyeWideLeft": pl.Float16,
    "eyeWideRight": pl.Float16,
    "jawForward": pl.Float16,
    "jawLeft": pl.Float16,
    "jawOpen": pl.Float16,
    "jawRight": pl.Float16,
    "mouthClose": pl.Float16,
    "mouthDimpleLeft": pl.Float16,
    "mouthDimpleRight": pl.Float16,
    "mouthFrownLeft": pl.Float16,
    "mouthFrownRight": pl.Float16,
    "mouthFunnel": pl.Float16,
    "mouthLeft": pl.Float16,
    "mouthLowerDownLeft": pl.Float16,
    "mouthLowerDownRight": pl.Float16,
    "mouthPressLeft": pl.Float16,
    "mouthPressRight": pl.Float16,
    "mouthPucker": pl.Float16,
    "mouthRight": pl.Float16,
    "mouthRollLower": pl.Float16,
    "mouthRollUpper": pl.Float16,
    "mouthShrugLower": pl.Float16,
    "mouthShrugUpper": pl.Float16,
    "mouthSmileLeft": pl.Float16,
    "mouthSmileRight": pl.Float16,
    "mouthStretchLeft": pl.Float16,
    "mouthStretchRight": pl.Float16,
    "mouthUpperUpLeft": pl.Float16,
    "mouthUpperUpRight": pl.Float16,
    "noseSneerLeft": pl.Float16,
    "noseSneerRight": pl.Float16,
    "label": pl.Int8,
}
X = SCHEMA.copy()


class HuggingFace:
    def __init__(self, dataset: str, device: str, streaming: bool):
        self.dataset = dataset
        self.device = device
        self.streaming = streaming
        self.localdata: DatasetDict = None
        self.DATAFRAME = pl.DataFrame(schema=SCHEMA)

    def streamData(self, num: int, split: str = "train"):
        self.localdata = load_dataset(
            self.dataset, streaming=self.streaming, token=HUGGING_FACE
        )

        self.localdata = self.localdata.shuffle(seed=5, buffer_size=100)

        try:
            self.localdata = self.localdata[split]
            print(f"Success: Loading from {split} worked.")
        except Exception:
            print(
                f"Error: Loading from {split} failed, loading the first available split."
            )
            self.localdata = self.localdata[next(iter(self.localdata.keys()))]

        for i in self.localdata:
            currImage = np.ascontiguousarray(np.array(i["image"]))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
            res = landmarker.detect(mpImage).face_blendshapes

            if len(res) > 0:
                num -= 1
                for j in res[0]:
                    X[j.category_name] = j.score
                X["label"] = i["label"]
                self.DATAFRAME = self.DATAFRAME.vstack(pl.from_dict(X, schema=SCHEMA))

            if num <= 0:
                print("Finished.")
                landmarker.close()
                self.DATAFRAME = self.DATAFRAME.unique()
                self.DATAFRAME.rechunk()
                self.DATAFRAME.write_csv(dataPath / f"csvOutput_{dataName}")
                self.DATAFRAME.write_parquet(dataPath / f"parquetOutput_{dataName}")
                return

            """cv2.imshow(
                "currentIteration",
                cv2.putText(
                    cv2.cvtColor(currImage, cv2.COLOR_RGB2BGR),
                    f"Current Image is: {self.localdata.features['label'].int2str(i['label'])}",
                    (25, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                ),
            )

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print(self.DATAFRAME)
                    self.DATAFRAME.rechunk()
                    landmarker.close()
                    print(self.DATAFRAME)
                    self.DATAFRAME.write_csv(dataPath / f"csvOutput_{dataName}")
                    self.DATAFRAME.write_parquet(dataPath / f"parquetOutput_{dataName}")
                    return
                elif key == ord("e"):
                    break"""


def createDataset(platform: str, dataset: str, streaming=False):
    if not Path(mediaPipePath).exists():
        print("Error: Mediapipe Landmarker task is not present in the root directory.")
        return
    if platform == 0:
        HF_DATASET = HuggingFace(dataset, os.name, streaming)
        if streaming:
            print("Streaming Data")
            HF_DATASET.streamData(1000)
