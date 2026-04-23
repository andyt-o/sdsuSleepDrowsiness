import src.sleepDrowsiness as sdModel

# 0 is hugging face
# 1 is kaggle
if __name__ == "__main__":
    sdModel.createDataset(0, "akahana/Driver-Drowsiness-Dataset", True, -1)
    # sdModel.evaluateModel()
    # sdModel.createDataset(0, "n7i5x9/driver-drowsiness-dataset", True)
