from pathlib import Path, PureWindowsPath
import xgboost as xgb
import polars as pl
import torch
import pyarrow.parquet as pq
import datetime as dt
import os


def trainModel():
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    try:
        recentParquet = Path("src/sleepDrowsiness/data")
        print(os.name)
        if os.name == "nt":
            saveLocation = (
                PureWindowsPath(r"src\sleepDrowsiness\models")
                / f"XMODEL_{dt.datetime.now().strftime(
                "%H-%M-S_%m-%d-%Y")}"
            )
        else:
            saveLocation = (
                Path("src/sleepDrowsiness/models")
                / f"XMODEL_{dt.datetime.now().strftime(
                "%H:%M:S_%m-%d-%Y")}"
            )
        files = [f for f in recentParquet.glob("*") if f.is_file()]
        recentParquet = Path(max(files, key=lambda f: f.stat().st_birthtime))
        print(recentParquet)
        parquetCols = pq.read_schema(recentParquet).names
        DATAFRAME = pl.read_parquet(recentParquet, columns=parquetCols)
        parquetCols.pop()
    except Exception:
        print("Error: Cannot locate the parquet file.")

    PARAMS = {
        "max_depth": 5,
        "eta": 0.15,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "device": DEVICE,
    }

    if DEVICE == "cuda" or DEVICE == "mps":
        MAIN_DATA = xgb.QuantileDMatrix(
            DATAFRAME[parquetCols], label=DATAFRAME["label"]
        )
        PARAMS["tree_method"] = "hist"
    else:
        MAIN_DATA = xgb.DMatrix(DATAFRAME[parquetCols], label=DATAFRAME["label"])

        MODEL = xgb.train(PARAMS, MAIN_DATA, num_boost_round=2500)
        MODEL.save_model(f"{saveLocation}.ubj")
