from pathlib import Path, PureWindowsPath
import xgboost as xgb
import polars as pl
import torch
import pyarrow.parquet as pq
import datetime as dt
import os

# @requires  src/sleepDrowsiness/data/ — Must contain at least one .parquet feature file
# @requires  src/sleepDrowsiness/models/ — Output directory for the trained model
# @assigns   MODEL — Trained XGBoost booster saved as .ubj to the models directory
# @returns   None — Saves the model to disk
def trainModel():
    """
    Trains a binary XGBoost classifier on the most recently generated parquet dataset.
    Automatically selects the best available compute device (CUDA / MPS / CPU).
    Saves the trained model as a timestamped .ubj file.
    """

    # Select the best available compute device for XGBoost acceleration
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    try:
        recentParquet = Path("src/sleepDrowsiness/data")
        print(os.name)

        # Build the output model path using OS-appropriate separators and a timestamp
        if os.name == "nt":
            saveLocation = (
                PureWindowsPath(r"src\sleepDrowsiness\models")
                / f"XMODEL_{dt.datetime.now().strftime('%H-%M-S_%m-%d-%Y')}"
            )
        else:
            saveLocation = (
                Path("src/sleepDrowsiness/models")
                / f"XMODEL_{dt.datetime.now().strftime('%H:%M:S_%m-%d-%Y')}"
            )

        # Find the most recently created parquet file in the data directory
        files = [f for f in recentParquet.glob("*") if f.is_file()]
        recentParquet = Path(max(files, key=lambda f: f.stat().st_birthtime))
        print(recentParquet)

        # Read column names from the parquet schema, then load the full dataframe
        parquetCols = pq.read_schema(recentParquet).names
        DATAFRAME = pl.read_parquet(recentParquet, columns=parquetCols)

        # Remove the last column (label) from the feature list — it's used separately as the target
        parquetCols.pop()

    except Exception:
        print("Error: Cannot locate the parquet file.")

    # XGBoost hyperparameters for binary classification
    PARAMS = {
        "max_depth": 5,          # Maximum tree depth — controls model complexity
        "eta": 0.15,             # Learning rate — smaller = more conservative updates
        "objective": "binary:logistic",  # Binary classification with probability output
        "eval_metric": "auc",    # Area Under the Curve — better metric than accuracy for imbalanced data
        "device": DEVICE,
    }

    if DEVICE == "cuda" or DEVICE == "mps":
        # QuantileDMatrix is optimized for GPU/MPS — faster memory transfer and histogram computation
        MAIN_DATA = xgb.QuantileDMatrix(
            DATAFRAME[parquetCols], label=DATAFRAME["label"]
        )
        PARAMS["tree_method"] = "hist"  # Required for GPU-accelerated training
    else:
        # Standard DMatrix for CPU training
        MAIN_DATA = xgb.DMatrix(DATAFRAME[parquetCols], label=DATAFRAME["label"])

        # Train for 2500 rounds — sufficient for convergence on this dataset size
        MODEL = xgb.train(PARAMS, MAIN_DATA, num_boost_round=2500)
        MODEL.save_model(f"{saveLocation}.ubj")