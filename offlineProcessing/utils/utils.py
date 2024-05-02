from time import time
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

BODY_POSE_DIR = os.path.join(DATA_DIR, 'body_pose')

# cretae empty folder structure if repo is cloned for the first time
for folder in [OUTPUT_DIR, DATA_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)