import os
import json
import dotenv

dotenv.load_dotenv(dotenv_path=".env")

LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)
TENSOR_SHAPE = (3,) + IMAGE_SIZE
BATCH_SIZE = 256
GPUS = 0
NUM_EPOCHS = 1
NUM_CLASSES = 9

BUCKET_NAME = "smart-trash-storage"
STAGE_DATA_NAME = "data_train/data.tar.gz"
LOCAL_DATA_DIR = "./data"
LOCAL_SPLIT_DATA_DIR = "./data_split"
TRAIN_DATA_DIR = f"{LOCAL_SPLIT_DATA_DIR}/train"
TEST_DATA_DIR = f"{LOCAL_SPLIT_DATA_DIR}/val"

GOOGLE_ACCOUNT_CREDENTIALS = json.loads(
    os.environ.get("GOOGLE_SERVICE_ACCOUNT_CREDENTIALS", "{}")
)
NEPTUNE_TOKEN = os.environ.get("NEPTUNE_TOKEN")
