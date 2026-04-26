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

# Load the Hugging Face token from the .env file
load_dotenv()
HUGGING_FACE = os.getenv("HUGGING_FACE")


class evalModel:
    """
    Evaluates a trained XGBoost model against a Hugging Face image dataset.
    Streams images one by one, extracts blendshape features via MediaPipe,
    runs inference, and displays the prediction vs ground truth in a cv2 window.
    """

    # @requires  dataset: str — Hugging Face dataset name (e.g. "akahana/Driver-Drowsiness-Dataset")
    # @assigns   self.landmarker — MediaPipe FaceLandmarker in IMAGE mode
    # @assigns   self.MODEL      — Most recently trained XGBoost booster
    # @assigns   self.DATASET    — Streamed and shuffled train split
    # @assigns   self.inputData  — Feature dict (SCHEMA without the label column)
    # @returns   None — Immediately starts evaluation via self.testData()
    def __init__(self, dataset):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="./face_landmarker.task"
            ),
            output_face_blendshapes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Static image mode
            num_faces=1,
            min_face_detection_confidence=0.2,               # Low threshold to maximize detections
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.MODEL = self.loadModel(self.findModel())
        self.DATASET = self.loadDataset(dataset)["train"]

        # Use the full schema minus the label column as the inference input template
        self.inputData = SCHEMA
        self.inputData.pop("label")
        print(self.inputData)

        self.testData()

    # @requires  dataset: str — Hugging Face dataset name
    # @returns   DatasetDict — Shuffled streaming dataset (seed=1 for reproducibility)
    def loadDataset(self, dataset: str):
        """
        Load and shuffle a Hugging Face dataset in streaming mode.
        Streaming avoids downloading the full dataset to disk.
        """
        x = load_dataset(dataset, token=HUGGING_FACE, streaming=True)
        x: Dataset = x.shuffle(seed=1)
        return x

    # @requires  src/sleepDrowsiness/models/ — Must contain at least one .ubj model file
    # @returns   Path — Path to the most recently created model file
    def findModel(self):
        """
        Scan the models directory and return the most recently created model file.
        Uses file birth time (st_birthtime) to determine recency.
        """
        modelPath = Path("src/sleepDrowsiness/models")
        files = [f for f in modelPath.glob("*") if f.is_file()]
        newestModel = max(files, key=lambda f: f.stat().st_birthtime)
        print("Model being used is: ", newestModel)
        return newestModel

    # @requires  model: Path — Path to a valid XGBoost .ubj model file
    # @returns   xgb.Booster — Loaded XGBoost booster ready for inference
    def loadModel(self, model):
        """Load and return an XGBoost booster from the given model file path."""
        x = xgb.Booster()
        x.load_model(model)
        return x

    # @requires  self.DATASET   — Streaming dataset with "image" and "label" fields
    # @requires  self.MODEL     — Loaded XGBoost booster
    # @requires  self.inputData — Feature dict to fill with blendshape scores
    # @returns   None — Displays each image in a cv2 window with prediction overlay
    #            Press 'e' to go to the next image, 'q' to quit
    def testData(self):
        """
        Iterate over the dataset, run inference on each image, and display results.
        Tracks running accuracy across all evaluated images.
        Waits for user input between images:
          - 'e' → next image
          - 'q' → quit and close window
        """
        x = 0         # Total number of images evaluated
        accuracy = 0  # Number of correct predictions

        for i in self.DATASET:
            x += 1

            # Convert PIL image to a contiguous numpy array (required by MediaPipe and OpenCV)
            currImage = np.ascontiguousarray(np.array(i["image"]))
            cv2Image = cv2.cvtColor(currImage, cv2.COLOR_RGB2BGR)
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=currImage)

            # Extract blendshape features using MediaPipe
            res = self.landmarker.detect(mpImage).face_blendshapes
            if len(res) > 0:
                for j in res[0]:
                    self.inputData[j.category_name] = j.score
            # If no face detected, inputData keeps its previous values

            # Run XGBoost inference — output is a probability (< 0.5 = Drowsy, >= 0.5 = Awake)
            modelOutput = self.MODEL.predict(
                xgb.DMatrix(pl.DataFrame(data=self.inputData))
            )

            # Compare prediction to ground truth and update accuracy counter
            if (0 if modelOutput < 0.5 else 1) == i["label"]:
                accuracy += 1

            print(
                f"Model suggests that Image {x} is {modelOutput}, {self.DATASET.features['label'].int2str(0 if modelOutput < 0.5 else 1)}\nCurrent Accuracy is {accuracy / x}"
            )

            # Overlay ground truth label on the image (cyan)
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
            # Overlay model prediction on the image (white)
            cv2.putText(
                cv2Image,
                f"Model: {self.DATASET.features['label'].int2str(0 if modelOutput < 0.5 else 1)}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.425,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("evalModel", cv2Image)

            # Wait for user input before moving to the next image
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return
                elif key == ord("e"):
                    break  # Move to the next image


def evaluateModel():
    """
    Entry point for model evaluation.
    Runs the evaluation against the akahana dataset by default.

    Accuracy benchmarks observed across different datasets:
      Training dataset: akahana/Driver-Drowsiness-Dataset
        - akahana/Driver-Drowsiness-Dataset : 97% (same distribution as training)
        - n7i5x9/driver-drowsiness-dataset  : 60% (different distribution)
        - c3rl/yawning-people               : 52% (very different domain)
    """
    testFile = evalModel("akahana/Driver-Drowsiness-Dataset")


if __name__ == "__main__":
    # Run evaluation directly when the script is executed as a standalone module
    testFile = evalModel("akahana/Driver-Drowsiness-Dataset")
    try:
        print("\n")
    except Exception:
        print("Error: Model could not be found to test.")