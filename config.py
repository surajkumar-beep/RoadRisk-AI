import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "roadrisk_ai_secret_key")
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "road_accident_dataset.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models")
    REPORT_PATH = os.path.join(BASE_DIR, "reports")
    DEBUG = True