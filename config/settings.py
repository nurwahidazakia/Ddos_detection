"""
Konfigurasi global proyek.

Semua path, parameter, dan konstanta terpusat di sini.
Tidak ada nilai ajaib (magic number) di modul lain.
"""

from pathlib import Path

# ==============================================================================
# PATH PROYEK
# ==============================================================================

# Root proyek
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CICDDOS2019_DIR = RAW_DATA_DIR / "cicddos2019"
CICIDS2017_DIR = RAW_DATA_DIR / "cicids2017"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = PROCESSED_DIR / "cleaned"
TRAIN_DIR = PROCESSED_DIR / "train"
TEST_DIR = PROCESSED_DIR / "test"
REPORTS_DIR = DATA_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "traffic_sgd.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"

# Config files
CONFIG_DIR = PROJECT_ROOT / "config"
LABEL_MAPPING_PATH = CONFIG_DIR / "label_mapping.json"
FEATURE_CONFIG_PATH = CONFIG_DIR / "selected_features.json"

# Database
DB_PATH = PROJECT_ROOT / "data" / "ddos_monitoring.db"

# Logs
LOG_DIR = PROJECT_ROOT / "logs"

# ==============================================================================
# PARAMETER MACHINE LEARNING
# ==============================================================================

# Random seed untuk reproduksibilitas
RANDOM_SEED = 42

# Train/test split ratio
TEST_SIZE = 0.2

# Chunk size untuk pembacaan CSV (RAM terbatas ~7.4 GB)
CHUNK_SIZE = 50_000

# Korelasi threshold untuk penghapusan fitur redundan
CORRELATION_THRESHOLD = 0.95

# Decision Tree hyperparameter search space
DT_PARAM_GRID = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None],
}

# Cross-validation folds
CV_FOLDS = 5

# ==============================================================================
# KOLOM YANG DIHAPUS (BUKAN FITUR)
# ==============================================================================

# Kolom metadata/ID yang tidak boleh menjadi fitur model
DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Flow Bytes/s",   # Sering mengandung Inf
    "Flow Packets/s", # Sering mengandung Inf
    "SimillarHTTP",
    "Unnamed: 0",
]

# Nama kolom label (berbeda antar versi dataset)
LABEL_COLUMN_NAMES = ["Label", " Label"]

# ==============================================================================
# PARAMETER MONITORING
# ==============================================================================

# Flask server
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True  # False untuk production

# SQLite retensi data (hari)
DB_RETENTION_DAYS = 30

# Interval polling dashboard (detik)
DASHBOARD_REFRESH_INTERVAL = 5

# ==============================================================================
# PARAMETER CICFLOWMETER
# ==============================================================================

# Path ke CICFlowMeter (sesuaikan setelah instalasi)
CICFLOWMETER_PATH = None  # Diisi setelah CICFlowMeter diinstall

# Output directory untuk flow CSV dari CICFlowMeter
FLOW_OUTPUT_DIR = DATA_DIR / "flows"
