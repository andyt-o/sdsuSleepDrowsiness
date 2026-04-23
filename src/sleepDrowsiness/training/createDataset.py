from datasets import load_dataset, DatasetDict, load_dataset_builder
from pathlib import Path, PureWindowsPath
from dotenv import load_dotenv
import polars as pl
import numpy as np
import mediapipe as mp
import datetime as dt
import cv2
import os

load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")

dataPath = (
    Path("src/sleepDrowsiness/data")
    if os.name == "posix"
    else Path("src\sleepDrowsiness\data")
)
dataName = dt.datetime.now().strftime("%H_%M_%m_%d_%Y")

mediaPipePath = "./face_landmarker.task"
baseOptions = mp.tasks.BaseOptions(model_asset_path=mediaPipePath)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=baseOptions,
    output_face_blendshapes=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.2,
)

SCHEMA = {
    "_neutral": pl.Float32,
    "browDownLeft": pl.Float32,
    "browDownRight": pl.Float32,
    "browInnerUp": pl.Float32,
    "browOuterUpLeft": pl.Float32,
    "browOuterUpRight": pl.Float32,
    "cheekPuff": pl.Float32,
    "cheekSquintLeft": pl.Float32,
    "cheekSquintRight": pl.Float32,
    "eyeBlinkLeft": pl.Float32,
    "eyeBlinkRight": pl.Float32,
    "eyeLookDownLeft": pl.Float32,
    "eyeLookDownRight": pl.Float32,
    "eyeLookInLeft": pl.Float32,
    "eyeLookInRight": pl.Float32,
    "eyeLookOutLeft": pl.Float32,
    "eyeLookOutRight": pl.Float32,
    "eyeLookUpLeft": pl.Float32,
    "eyeLookUpRight": pl.Float32,
    "eyeSquintLeft": pl.Float32,
    "eyeSquintRight": pl.Float32,
    "eyeWideLeft": pl.Float32,
    "eyeWideRight": pl.Float32,
    "jawForward": pl.Float32,
    "jawLeft": pl.Float32,
    "jawOpen": pl.Float32,
    "jawRight": pl.Float32,
    "mouthClose": pl.Float32,
    "mouthDimpleLeft": pl.Float32,
    "mouthDimpleRight": pl.Float32,
    "mouthFrownLeft": pl.Float32,
    "mouthFrownRight": pl.Float32,
    "mouthFunnel": pl.Float32,
    "mouthLeft": pl.Float32,
    "mouthLowerDownLeft": pl.Float32,
    "mouthLowerDownRight": pl.Float32,
    "mouthPressLeft": pl.Float32,
    "mouthPressRight": pl.Float32,
    "mouthPucker": pl.Float32,
    "mouthRight": pl.Float32,
    "mouthRollLower": pl.Float32,
    "mouthRollUpper": pl.Float32,
    "mouthShrugLower": pl.Float32,
    "mouthShrugUpper": pl.Float32,
    "mouthSmileLeft": pl.Float32,
    "mouthSmileRight": pl.Float32,
    "mouthStretchLeft": pl.Float32,
    "mouthStretchRight": pl.Float32,
    "mouthUpperUpLeft": pl.Float32,
    "mouthUpperUpRight": pl.Float32,
    "noseSneerLeft": pl.Float32,
    "noseSneerRight": pl.Float32,
    "label": pl.Int8,
}


class HuggingFace:
    def __init__(self, dataset: str, device: str, streaming: bool):
        self.dataset = dataset
        self.device = device
        self.streaming = streaming
        self.localdata: DatasetDict = None
        self.DATAFRAME = pl.DataFrame(schema=SCHEMA)
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def processData(self, data):
        X = SCHEMA.copy()
        N = len(data["label"])

        for i in SCHEMA:
            X[i] = [0 for _ in range(N)]

        for i in range(N):
            currImage = np.ascontiguousarray(np.array(data["image"][i]))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
            res = self.landmarker.detect(mpImage)
            X["label"][i] = data["label"][i]
            for j in res.face_blendshapes[0]:
                X[j.category_name][i] = j.score

        """
        for i in range(len(data["label"])):
            currImage = np.ascontiguousarray(np.array(data["image"][i]))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
            res = landmarker.detect(mpImage)
            for j in res.face_blendshapes[0]:
                if i == 0:
                    X[j.category_name] = [j.score]
                else:
                    X[j.category_name].append(j.score)
            if i == 0:
                X["label"] = [data["label"][i]]
            else:
                X["label"].append(data["label"][i])
        """
        return X

    def streamData(self, num: int, split: str = "train"):
        self.localdata = load_dataset(
            self.dataset, streaming=self.streaming, token=HUGGING_FACE
        )

        if num == -1:
            x = load_dataset_builder(self.dataset)
            num = x.info.splits["train"].num_examples
        self.localdata = self.localdata.shuffle(seed=5, buffer_size=100)

        try:
            self.localdata = self.localdata[split]
            print(f"Success: Loading from {split} worked.")
        except Exception:
            print(
                f"Error: Loading from {split} failed, loading the first available split."
            )
            self.localdata = self.localdata[next(iter(self.localdata.keys()))]

        processedData = self.localdata.map(
            self.processData,
            batched=True,
            batch_size=1000,
            remove_columns=["image", "label"],
        )

        _num = num
        for sample in processedData.take(num):
            _num -= 1
            print(f"Loading... {_num} remaining.")
            self.DATAFRAME = self.DATAFRAME.vstack(pl.from_dict(sample, schema=SCHEMA))

        self.DATAFRAME.unique()
        self.DATAFRAME.rechunk()
        self.DATAFRAME.write_csv(dataPath / f"csvOutput_{dataName}")
        self.DATAFRAME.write_parquet(dataPath / f"parquetOutput_{dataName}")


def createDataset(platform: str, dataset: str, streaming=False, num=1):
    if not Path(mediaPipePath).exists():
        print("Error: Mediapipe Landmarker task is not present in the root directory.")
        return
    if platform == 0:
        HF_DATASET = HuggingFace(dataset, os.name, streaming)
        if streaming:
            print("Streaming Data")
            HF_DATASET.streamData(num)

        """currImage = np.ascontiguousarray(np.array(i["image"]))
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
        res = landmarker.detect(mpImage).face_blendshapes

        if len(res) > 0:
            for j in res[0]:
                X[j.category_name] = j.score
            X["label"] = i["label"]
            self.DATAFRAME = self.DATAFRAME.vstack(pl.from_dict(X, schema=SCHEMA))
        
        print("Finished.")
        landmarker.close()
        self.DATAFRAME = self.DATAFRAME.unique()
        self.DATAFRAME.rechunk()
        self.DATAFRAME.write_csv(dataPath / f"csvOutput_{dataName}")
        self.DATAFRAME.write_parquet(dataPath / f"parquetOutput_{dataName}")
        return

        cv2.imshow(
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
