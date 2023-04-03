import os
from pathlib import Path

PROJ_CODE = 'AG'

BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
PROJ_DIR = BASE_DIR / 'proj'

MODEL_DIR = PROJ_DIR / 'model'
CACHE_DIR = PROJ_DIR / 'cache'
DATA_DIR = PROJ_DIR / 'data'
FIG_DIR = PROJ_DIR / 'fig'
SAVE_DIR = PROJ_DIR / 'save'
OUTPUT_DIR = PROJ_DIR / 'output'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
