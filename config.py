"""Central configuration for the whole project (paths, pose grid, camera, model)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    _d.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset.npz"
MODEL_PATH = MODEL_DIR / "pose_net.pt"

# ---- Pose discretisation (Table I of the paper) -----------------------------
# Pose = follower position in the leader frame (x fwd, y left, z up) + roll about
# the camera boresight. Axis order everywhere: x, y, z, phi.
POSE_LOW = np.array([-100.0, -50.0, -20.0, -45.0])
POSE_HIGH = np.array([50.0, 50.0, 20.0, 45.0])
N_BINS = np.array([10, 10, 8, 6])
NUM_CLASSES = int(np.prod(N_BINS))  # 4800
BIN_WIDTH = (POSE_HIGH - POSE_LOW) / N_BINS  # [15, 10, 5, 15]

# ---- Camera (follower, gimballed to always look at the leader) -------------
IMG_W, IMG_H = 1280, 960
FOCAL_PX = 1000.0
MIN_RANGE = 20.0  # m: reject poses closer than this (near collision)

# ---- Imperfect "feature detector" simulation --------------------------------
PIXEL_NOISE_STD = 1.0   # px, Gaussian noise on detected feature locations
MISS_PROB = 0.02        # probability a geometrically visible feature is missed

# ---- Network (paper: 42 -> 100 -> 100 -> 4800) ------------------------------
NUM_FEATURES = 14
INPUT_DIM = NUM_FEATURES * 3  # (u, v, visible) per feature = 42
HIDDEN = 100
