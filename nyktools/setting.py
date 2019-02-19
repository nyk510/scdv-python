import os

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = '/data'

FEATURE_DIR = os.path.join(DATA_DIR, 'feature')
os.makedirs(FEATURE_DIR, exist_ok=True)

PROCESSED_ROOT = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_ROOT, exist_ok=True)

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

PREDICT_FOR_TEST_DIR = os.path.join(DATA_DIR, 'predict_test')
os.makedirs(PREDICT_FOR_TEST_DIR, exist_ok=True)
