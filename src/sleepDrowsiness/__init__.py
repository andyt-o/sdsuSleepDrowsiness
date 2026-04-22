import os
from dotenv import load_dotenv
from .training.createDataset import createDataset

load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE")
print("Loaded Successfully")
