"""
project settings - all notebooks use the same values.
"""
from pathlib import Path

# paths 
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
STYLES_CSV = RAW_DATA_DIR / "styles.csv"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DIR = PROCESSED_DATA_DIR / "training"
VALIDATION_DIR = PROCESSED_DATA_DIR / "validation"
TEST_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# classes considered
CLASSES = ["Tshirts", "Shirts", "Casual Shoes", "Sports Shoes", "Watches", "Handbags"]
NUM_CLASSES = len(CLASSES)

# data preparation
IMAGES_PER_CLASS = 1000      
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.1

# reproducibility
RANDOM_SEED = 42

# data pipeline
IMG_WIDTH = 60
IMG_HEIGHT = 80
NUM_CHANNELS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 50
