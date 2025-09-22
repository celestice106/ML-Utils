# config.py

# ========== GENERAL ==========
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG_MODE = False  # Set True for verbose logging

# ========== PATHS ==========
BASE_DIR = "/content"
DATA_DIR = f"{BASE_DIR}/data"
MODEL_DIR = f"{BASE_DIR}/models"
LOG_DIR = f"{BASE_DIR}/logs"
EXPORT_DIR = f"{BASE_DIR}/exports"

# ========== TRAINING ==========
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
LR_SCHEDULER = True
LR_STEP_SIZE = 5
LR_GAMMA = 0.5

# ========== MODEL ==========
INPUT_SIZE = 784  # For MNIST
HIDDEN_UNITS = [128, 64]
OUTPUT_SIZE = 10
#ACTIVATION = "relu"
RELU = "relu"
TANH = "tanh"
SIGMOID = "sigmoid"
DROPOUT = 0.3

# ========== AUGMENTATION ==========
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "rotation": 10,
    "horizontal_flip": False,
    "normalize": True
}

# ========== LOGGING & TRACKING ==========
USE_TENSORBOARD = True
SAVE_MODEL_EACH_EPOCH = False
LOG_INTERVAL = 100  # steps

# ========== EXPORT ==========
EXPORT_FORMAT = "onnx"  # Options: 'pt', 'onnx', 'pickle'
EXPORT_NAME = f"{PROJECT_NAME}_v1"

# ========== MONETIZATION PREP ==========
ENABLE_API_EXPORT = True
API_ENDPOINT = "https://your-api.com/predict"
INCLUDE_LICENSE = True
LICENSE_TYPE = "MIT"
