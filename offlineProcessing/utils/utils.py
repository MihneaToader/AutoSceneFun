from time import time
import os
import platform

OPERATING_SYSTEM = platform.system()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')

BODY_POSE_DIR = os.path.join(ROOT_DIR, 'body_pose')

# External repos
EXTERNAL_REPOS_DIR = os.path.join(ROOT_DIR, 'external_repos')
PYVIZ_DIR = os.path.join(EXTERNAL_REPOS_DIR, 'PyViz3D')

# create empty folder structure if repo is cloned for the first time
# for folder in [OUTPUT_DIR, DATA_DIR]:
#     if not os.path.exists(folder):
#         os.makedirs(folder, exist_ok=True)