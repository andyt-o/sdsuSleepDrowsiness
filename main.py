import src.sleepDrowsiness as sdModel
import src.sleepDrowsiness.webcamTracking as mainApp
import os

# 0 is hugging face
# 1 is kaggle
if __name__ == "__main__":
    if os.name == "nt":
        s = r"src\sleepDrowsiness\models\ALAKHANA_1K.ubj"
    else:
        s = r"src/sleepDrowsiness/models/ALAKHANA_1K.ubj"
    # sdModel.trainModel()
    # sdModel.evaluateModel()
    # sdModel.createDataset(0, "n7i5x9/driver-drowsiness-dataset", True)
    mainApp = mainApp.setupModel(s, 24, (1280, 720))
