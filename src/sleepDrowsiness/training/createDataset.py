from datasets import load_dataset, DatasetDict, load_dataset_builder
from pathlib import Path, PureWindowsPath
from dotenv import load_dotenv
import polars as pl
import numpy as np
import mediapipe as mp
import datetime as dt
import cv2
import os

# Load environment variables from the .env file
load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")
if HUGGING_FACE is None:
    raise ValueError("Missing HUGGING_FACE token. Add it to .env")
os.environ["HF_TOKEN"] = HUGGING_FACE

# Output path for processed data files (cross-platform)
dataPath = (
    Path(r"src/sleepDrowsiness/data")
    if os.name == "posix"
    else Path(r"src\sleepDrowsiness\data")
)

# Timestamp used to name output files uniquely
dataName = dt.datetime.now().strftime("%H_%M_%m_%d_%Y")

# Path to the MediaPipe Face Landmarker model file
mediaPipePath = "./face_landmarker.task"

# Configure MediaPipe Face Landmarker for static image processing
baseOptions = mp.tasks.BaseOptions(model_asset_path=mediaPipePath)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=baseOptions,
    output_face_blendshapes=True,  # We need blendshapes as features
    running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Static image mode (not live stream)
    num_faces=1,  # Only detect one face per image
    min_face_detection_confidence=0.2,  # Low threshold to maximize detections
)

# Schema defining the 52 blendshape features + label used across the whole project.
# Each key is a MediaPipe blendshape name; values are their Polars dtype.
# label: 0 = Awake, 1 = Drowsy
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
    """
    Handles dataset loading, feature extraction, and export
    from either a Hugging Face Hub dataset or a local image folder.
    """

    # @requires  dataset  : str  — Hugging Face dataset name
    # @requires  device   : str  — OS name (os.name), used for path handling
    # @requires  streaming: bool — Whether to stream the dataset or download it fully
    # @assigns   self.DATAFRAME  — Empty Polars DataFrame with SCHEMA
    # @assigns   self.landmarker — MediaPipe FaceLandmarker instance
    # @returns   None
    def __init__(self, dataset: str, device: str, streaming: bool):
        self.dataset = dataset
        self.device = device
        self.streaming = streaming
        self.localdata: DatasetDict = None
        self.DATAFRAME = pl.DataFrame(schema=SCHEMA)  # Accumulates extracted features
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.counter = 0  # Tracks total images processed

    # @requires  data : dict — Batch from HuggingFace dataset with keys "image" and "label"
    # @assigns   X    — Dict of blendshape scores and labels for the batch
    # @returns   dict — Feature dict compatible with SCHEMA (to be stacked into DATAFRAME)
    def processData(self, data):
        """
        Batch processing function called by dataset.map().
        Extracts 52 blendshape features from each image using MediaPipe.
        Images where no face is detected are left as zeros.
        """
        X = SCHEMA.copy()
        N = len(data["label"])
        self.counter += N
        print(N)

        # Initialize all feature lists with zeros for the batch size
        for i in SCHEMA:
            X[i] = [0 for _ in range(N)]

        for i in range(N):
            # Convert PIL image to a contiguous numpy array (required by MediaPipe)
            currImage = np.ascontiguousarray(np.array(data["image"][i]))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
            res = self.landmarker.detect(mpImage)

            if len(res.face_blendshapes) > 0:
                # Face detected: store the label and all blendshape scores
                X["label"][i] = data["label"][i]
                for j in res.face_blendshapes[0]:
                    X[j.category_name][i] = j.score
            # No face detected: row stays at zeros (filtered during training)

        return X

    # @requires  num  : int — Number of samples to take (-1 = full dataset)
    # @requires  split: str — Dataset split to use (default: "train")
    # @assigns   self.DATAFRAME — Populated with extracted blendshape features
    # @returns   None — Writes .csv and .parquet files to dataPath
    def streamData(self, num: int, split: str = "train"):
        """
        Streams the Hugging Face dataset, extracts features batch by batch,
        and saves the result as CSV and Parquet files.
        Shuffles the dataset first to avoid ordering bias.
        """
        self.localdata = load_dataset(
            self.dataset, streaming=self.streaming, token=HUGGING_FACE
        )

        # If num == -1, fetch the full dataset size from the builder metadata
        if num == -1:
            x = load_dataset_builder(self.dataset)
            num = x.info.splits["train"].num_examples

        # Shuffle to reduce ordering bias during training
        self.localdata = self.localdata.shuffle(seed=5, buffer_size=100)

        # Try the requested split, fall back to the first available one
        try:
            self.localdata = self.localdata[split]
            print(f"Success: Loading from {split} worked.")
        except Exception:
            print(
                f"Error: Loading from {split} failed, loading the first available split."
            )
            self.localdata = self.localdata[next(iter(self.localdata.keys()))]

        # Map feature extraction over the dataset in batches of 1000
        processedData = self.localdata.map(
            self.processData,
            batched=True,
            batch_size=1000,
            remove_columns=["image", "label"],  # Drop raw columns after extraction
        )

        # Accumulate extracted samples into the main DataFrame
        for sample in processedData.take(num):
            self.DATAFRAME = self.DATAFRAME.vstack(pl.from_dict(sample, schema=SCHEMA))

        # Deduplicate, optimize memory layout, then export
        self.DATAFRAME.unique()
        self.DATAFRAME.rechunk()
        self.DATAFRAME.write_csv(dataPath / f"csvOutput_{dataName}")
        self.DATAFRAME.write_parquet(dataPath / f"parquetOutput_{dataName}")

    # @requires  dir : Path — Path to a local folder structured as imagefolder
    #                         (subfolders = class names: awake/, drowsy/)
    # @assigns   self.DATAFRAME — Populated with extracted blendshape features
    # @returns   None — Writes a .parquet test file to dataPath
    def createFromDirectory(self, dir: Path):
        """
        Loads images from a local directory structured as an image folder dataset.
        Extracts blendshape features for each image and saves the result as Parquet.
        Used for local testing instead of streaming from Hugging Face.
        """
        # Load images using HuggingFace's imagefolder loader (auto-detects class labels)
        self.localdata = load_dataset("imagefolder", data_dir=dir)
        X = SCHEMA.copy()

        for i in self.localdata["train"]:
            currImage = np.ascontiguousarray(np.array(i["image"]))
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)
            res = self.landmarker.detect(mpImage)

            if len(res.face_blendshapes) > 0:
                # Store each blendshape score and the image label
                for j in res.face_blendshapes[0]:
                    X[j.category_name] = j.score
                X["label"] = i["label"]
                self.DATAFRAME = self.DATAFRAME.vstack(pl.from_dict(X, schema=SCHEMA))

        self.DATAFRAME.unique()
        self.DATAFRAME.rechunk()
        self.DATAFRAME.write_parquet(dataPath / f"testParquet_{dataName}")


# @requires  platform : int  — 0 = Hugging Face stream, 1 = local directory
# @requires  dataset  : str  — HuggingFace dataset name (used when platform == 0)
# @requires  streaming: bool — Whether to stream data (default: False)
# @requires  num      : int  — Number of samples to process (default: 1)
# @returns   None — Delegates to HuggingFace.streamData or HuggingFace.createFromDirectory
def createDataset(platform: str, dataset: str, streaming=False, num=1):
    """
    Entry point for dataset creation.
    Dispatches to the correct loading strategy based on the platform argument:
      - platform 0: stream from Hugging Face Hub
      - platform 1: load from a local image folder (./data)
    """
    # Safety check: MediaPipe model file must exist before running
    if not Path(mediaPipePath).exists():
        print("Error: Mediapipe Landmarker task is not present in the root directory.")
        return

    if platform == 0:
        HF_DATASET = HuggingFace(dataset, os.name, streaming)
        if streaming:
            print("Streaming Data")
            HF_DATASET.streamData(num)

    elif platform == 1:
        HF_DATASET = HuggingFace(dataset, os.name, False)
        # Use OS-appropriate path separator for the local data folder
        if os.name == "nt":
            HF_DATASET.createFromDirectory(Path(r".\data"))
        else:
            HF_DATASET.createFromDirectory(Path("./data"))
