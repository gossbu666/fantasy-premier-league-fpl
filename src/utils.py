import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configurations
FPL_API_URL = "https://fantasy.premierleague.com/api"

# Model configurations
POSITIONS = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
FEATURE_HORIZONS = [1, 3, 5, 10, 38]

# Model parameters
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': [200, 400],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
    },
    'xgboost': {
        'n_estimators': [300, 600],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
    }
}

def ensure_data_dirs():
    """Create necessary data directories"""
    dirs = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"✓ Directory ready: {dir_path}")

