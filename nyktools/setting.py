import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.getenv('DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))

FEATURE_DIR = os.path.join(DATA_DIR, 'feature')
os.makedirs(FEATURE_DIR, exist_ok=True)

PROCESSED_ROOT = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_ROOT, exist_ok=True)

DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)
