## Driver drowsiness detection

A real-time driver drowsiness detection system using MediaPipe, OpenCV, Polars, and XGBoost. The system analyzes facial landmarks and blendshapes from a webcam feed to classify the driver as Awake or Drowsy and triggers a soft audio alert when drowsiness is detected for more than 1.5 seconds.

---

## How It Works

A driver is classified as **drowsy** when the model detects fatigue indicators such as:
- Prolonged eye closure (`eyeBlinkLeft`, `eyeBlinkRight`)
- Yawning / wide mouth opening (`jawOpen`, `mouthFunnel`, etc.)

The system extracts **52 facial blendshape coefficients** per frame using MediaPipe's Face Landmarker, feeds them into a trained XGBoost binary classifier, and overlays the prediction live on the video feed.

---

## How to Run

It requires a Python version higher than 3.13.

1. brew install libomp
2. Create and activate a virtual environment
python3 -m venv .venv
# On macOS/Linux : 
source .venv/bin/activate. 
# On Windows: 
.venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables
Create a .env file in the root directory and add your Hugging Face token:

```bash
touch .env
```

```env
HUGGING_FACE=hf_your_token_here
```

5.  Run the application
python3 main.py
(Note: On macOS, if you encounter OpenMP errors, run export KMP_DUPLICATE_LIB_OK=TRUE before execution).

Press **`q`** to quit the webcam window, or **`Ctrl+C`** to stop the process entirely.

---

## Project Structure 
IA_project/
│
├── .venv
├── data / train #Contains raw images for local testing
│     ├── awake/
│     └── drowsy/
├── src/sleepDrowsiness
│     ├── training/
│     │   ├── createDataset.py      # Feature extraction → Parquet/CSV
│     │   ├── trainModel.py        # XGBoost model training
│     │   └── evalModel.py      # Model evaluation on test images
│     ├── webcamTracking.py   # Real-time webcam inference
│     ├── data/     # Processed data: Stores Parquet/CSV feature files
│     └── models/    # Trained XGBoost models (.ubj)
│
├── main.py             # Entry point
├── face_landmarker.task   #MediaPipe Face Landmarker model
├── requirements.txt
├── pyproject.toml
├── .env            # API Tokens
└── README.md

## Technical Pipeline

Webcam / Dataset Images
        ↓
MediaPipe Face Landmarker  →  52 blendshape coefficients extracted
        ↓
Polars DataFrame  →  Stored as .parquet / .csv for training
        ↓
XGBoost Binary Classifier  →  Awake (0) or Drowsy (1)
        ↓
Live OpenCV overlay  →  Prediction + confidence score + Progress bar 

**Acceleration:** XGBoost automatically leverages **CUDA** (NVIDIA) or **MPS** (Apple Silicon) when available, falling back to CPU.

**Threshold:** A probability **< 0.5** → Drowsy, **≥ 0.5** → Awake.

---

## createDataset.py 

This module is responsible for building the dataset used for training the model.

It uses MediaPipe, an IA model, to extract facial landmarks and blendshapes from images, such as eye movements, mouth position, and facial expressions.

The extracted features are then:
- converted into structured numerical data
- stored using Polars
saved as .csv and .parquet files for later training

Output: a structured dataset of facial features + labels (drowsy / awake)

## trainModel.py

This module is used to train the machine learning model.

It loads the processed dataset and trains a classification model using XGBoost (model for binary classification.)

The model learns to predict whether a person is:
- drowsy (1)
- awake (0)

After training, the model is saved in .ubj format for later use.

## evalModel.py

This module is used to evaluate the trained model’s performance.

It:
- loads a trained XGBoost model
- processes test images using MediaPipe
- extracts facial features
- compares predictions with real labels

It also computes accuracy and displays real-time prediction results.

Output: model accuracy + visual evaluation results

## webcamTracking.py

This module enables real-time drowsiness detection using a webcam.

It:
- captures live video using OpenCV
- extracts facial landmarks using MediaPipe
- feeds features into the trained XGBoost model to get a drowsiness probability each frame
- applies a **1.5-second time filter** before triggering the alert, to avoid false positives from normal blinks or brief eye movements
- plays a **soft audio chime** via `sounddevice` 
- resets all timers immediately as soon as the driver opens their eyes
- displays the final state (`Awake` / `DROWSY !`), the raw model score, and a progress bar live on the video stream

## Model Accuracy Benchmarks
Results observed when training on akahana/Driver-Drowsiness-Dataset:

Test Dataset and Accuracy:
akahana/Driver-Drowsiness-Dataset -> 97%
n7i5x9/driver-drowsiness-dataset -> 60%
c3rl/yawning-people -> 52%