# config.py
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

DEFAULT_PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    'pH': (0.0, 14.0), 'DO': (0.0, 30.0), 'TN': (0.0, 100.0),
    'TP': (0.0, 50.0), 'Tur': (0.0, 1000.0), 'PI': (0.0, 100.0),
    'Cond': (0.0, 3000.0), 'DOC': (0.0, 200.0), 'Temp': (-5.0, 45.0)
}

@dataclass
class Config:
    RAW_DATA_FILE: str = r"processed_taihu_data_with_coords.csv"
    METEO_FILE: str = ""

    # <=0 means "use all qualified stations"
    TOP_K_LAKES: int = -1
    MIN_EFFECTIVE_STEPS: int = 1000

    # ==============================
    # Station selection
    #   - "geo": pick stations geographically related (nearest to an anchor / centroid)
    #   - "manual": user-provided station IDs (can be Chinese station names)
    # ==============================
    STATION_SELECT_MODE: str = "geo"  # geo / manual
    MANUAL_STATION_IDS: List[str] = field(default_factory=lambda: [])
    # If set and exists in data, use this station as geographic anchor; otherwise use centroid.
    GEO_ANCHOR_ID: str = ""
    # If provided (both not None), use these coordinates as anchor.
    GEO_ANCHOR_LON: Optional[float] = None
    GEO_ANCHOR_LAT: Optional[float] = None
    # Optional: max radius (km). None means no radius limit.
    GEO_RADIUS_KM: Optional[float] = None

    FEATURE_COLS: List[str] = field(default_factory=lambda: ["Temp", "DO", "TN", "Cond", "pH", "PI", "TP", "Tur"])
    TARGET_FEATURES: List[str] = field(default_factory=lambda: ["DO","Tur","TN","TP","PI","Cond"])
    PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: DEFAULT_PHYSICAL_LIMITS)

    RESAMPLE_FREQ: str = '4h'
    CAUSAL_IMPUTE: bool = True  # If True, use past-only imputation (avoid future leakage)
    # Missing-value strategy:
    # - "linear": time interpolation + boundary fill + train-mean fallback
    # - "spatial": IDW from nearby stations at same timestamp + train-mean fallback
    # - "mice": iterative imputation (sklearn) + train-mean fallback
    # - "hybrid": short-gap linear (<=1 day) -> spatial IDW -> MICE
    #             -> historical seasonal mean fallback (month-hour + week-hour climatology)
    #             -> train-mean final safeguard
    IMPUTE_METHOD: str = "hybrid"
    # Hybrid short-gap threshold (hours). 24h means at 4h sampling: <=6 steps.
    HYBRID_SHORT_GAP_MAX_HOURS: float = 24.0
    # Limit for linear interpolation gap length (steps). None means unlimited.
    # At 4h frequency, 18 means up to 3 days.
    INTERP_LIMIT_STEPS: Optional[int] = 18
    # Spatial interpolation settings
    SPATIAL_K: int = 4
    SPATIAL_POWER: float = 2.0
    # MICE settings
    MICE_MAX_ITER: int = 20
    MICE_RANDOM_SEED: int = 2025
    SEQ_LEN: int = 30
    PRED_LEN: int = 3

    # ==============================
    # Experimental design (research-grade)
    # Strict Train/Val/Test split (per-station, time-ordered).
    # NOTE: Only Val is used for LR scheduling / early-stopping / best-epoch selection.
    #       Test is used ONLY once after training.
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.1
    # Warm-up overlap for Val/Test (only overlaps the *input history*, not the labels)
    # Recommended: overlap = SEQ_LEN
    SPLIT_OVERLAP: int = 30

    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    DROPOUT_RATE: float = 0.3
    DELTA_LOSS_WEIGHT: float = 0.35

    # ===== CNN-specific defaults (avoid constant-output / mean-regression) =====
    CNN_LR: float = 1e-3
    CNN_WEIGHT_DECAY: float = 1e-4
    CNN_DROPOUT: float = 0.1
    CNN_LAYERS: int = 6
    CNN_KERNEL_SIZE: int = 3

    EARLY_STOP_PATIENCE: int = 10

    HIDDEN_DIM: int = 64
    NUM_LAYERS: int = 3
    # iTransformer multi-head attention
    # NOTE: must divide HIDDEN_DIM.
    NUM_HEADS: int = 4

    AUG_TIMES: int = 1
    NOISE_SCALE: float = 0.005
    # Data augmentation switches for two training modes:
    # - baseline: cnn/tcn/lstm/itransformer/patchtst
    # - gnn: stgcn/dcrnn/stgcn_fusion
    AUG_ENABLE_BASELINE: bool = True
    AUG_ENABLE_GNN: bool = True

    TCN_KERNEL_SIZE: int = 3

    # Optional explainability plots (SHAP + Captum) during post-processing.
    # Keep False by default to avoid extra dependency/runtime overhead.
    ENABLE_XAI_PLOTS: bool = False

    # Output management:
    # True  -> merge most non-image artifacts into one run_bundle.json
    # False -> keep legacy split JSON files (runtime_env/splits/scalers/history...)
    COMPACT_OUTPUTS: bool = True

    # ===== GNN / ST model params =====
    GCN_HIDDEN_DIM: int = 64
    GCN_LAYERS: int = 2
    RNN_HIDDEN_DIM: int = 64

    # CSV column names in current dataset
    NODE_ID_COL: str = "ID"
    TIME_COL: str = "Date"
    LON_COL: str = "lon"
    LAT_COL: str = "lat"

    GRAPH_TYPE: str = "knn"
    KNN_K: int = 6
    KNN_SIGMA_KM: float = 20.0

    # ===== STFusionNet: adaptive adjacency + multi-branch temporal fusion =====
    USE_ADAPTIVE_ADJ: bool = True
    ADAPT_EMB_DIM: int = 16
    ADJ_STATIC_WEIGHT: float = 1.0
    ADJ_ADAPT_WEIGHT: float = 0.5

    FUSION_HIDDEN_DIM: int = 64
    TEMP_CNN_KERNEL: int = 3

    # Current model name.
    MODEL_NAME: str = "itransformer"  # cnn / tcn / lstm / itransformer / patchtst / stgcn / dcrnn / stgcn_fusion
    # STFusionNet ablation switches
    # TEMPORAL_BRANCH_MODE: all / cnn / lstm / tcn
    TEMPORAL_BRANCH_MODE: str = "all"
    # FUSION_MODE: gate / avg / concat
    FUSION_MODE: str = "gate"


    # ==============================
    # Per-model hyperparameters (for convenient manual tuning)
    # - For baseline models, the runner will NOT do grid-search; it will use the
    #   corresponding dictionary below to override the global defaults.
    # - For the new model (STFusionNet / stgcn_fusion), grid-search will still
    #   sweep SEQ_LEN/BATCH_SIZE/HIDDEN (+ optional heads) based on GRID_*.
    # ==============================
    MODEL_PARAMS: Dict[str, Dict[str, object]] = field(default_factory=lambda: {
        # Baselines
        "cnn": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-3,
            "WEIGHT_DECAY": 1e-4,
            "DROPOUT_RATE": 0.1,
            "MAX_EPOCHS": 50,
        },
        "tcn": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        "lstm": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        "itransformer": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "HIDDEN_DIM": 64,
            "NUM_HEADS": 4,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        "patchtst": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "HIDDEN_DIM": 64,
            "NUM_HEADS": 4,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        "stgcn": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "GCN_HIDDEN_DIM": 64,
            "RNN_HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        "dcrnn": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "GCN_HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
        # New model (STFusionNet alias)
        "stgcn_fusion": {
            "SEQ_LEN": 30,
            "BATCH_SIZE": 64,
            "GCN_HIDDEN_DIM": 64,
            "FUSION_HIDDEN_DIM": 64,
            "LEARNING_RATE": 1e-4,
            "WEIGHT_DECAY": 1e-3,
            "DROPOUT_RATE": 0.3,
            "MAX_EPOCHS": 50,
        },
    })


    # ==============================
    # Multi-model runner + auto hyper-parameter tuning
    # ==============================
    # Run multiple models in one command, e.g.
    # ["lstm", "tcn", "cnn", "itransformer", "stgcn", "stgcn_fusion"]
    RUN_MODELS: List[str] = field(default_factory=lambda: [])

    # Whether to enable auto hyper-parameter search.
    AUTO_TUNE: bool = True

    # ==============================
    # STFusionNet (stgcn_fusion / stfusionnet) tuning mode:
    #   - "search": tune then train
    #   - "default": train directly with one preset
    # Notes:
    #   - only affects stgcn_fusion / stfusionnet
    #   - can be overridden by CLI --stf_mode
    # ==============================
    STFUSIONNET_TUNE_MODE: str = "search"  # search / default
    # Max tuning combinations (<=0 means full grid).
    TUNE_TRIALS: int = 0
    # Shorter epochs/patience during tuning stage
    TUNE_MAX_EPOCHS: int = 30
    TUNE_EARLY_STOP_PATIENCE: int = 8
    # Tuning objective: val_nse (maximize) / val_rmse or val_mse (minimize)
    TUNE_OBJECTIVE: str = "val_nse"
    # Search backend: grid / random / optuna / hyperband
    # - optuna/hyperband require optional dependencies and will fallback if unavailable
    TUNE_SEARCH_METHOD: str = "grid"
    # Base random seed for tuning.
    TUNE_RANDOM_SEED: int = 2025
    # Keep all trial folders.
    KEEP_TRIAL_DIRS: bool = False
    # Extra suffix for run_id.
    RUN_TAG: str = ""

    # ------------------------------
    # Grid search space (paper-style)
    #   - SEQ_LEN: observation window length
    #   - BATCH_SIZE
    #   - HIDDEN size (state/embedding)
    # Default grid is modest (3*3*4=36)
    # ------------------------------
    GRID_SEQ_LENS: List[int] = field(default_factory=lambda: [18, 30, 42])
    GRID_BATCH_SIZES: List[int] = field(default_factory=lambda: [32, 64, 128])
    GRID_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    # Extended search dimensions (to avoid only 3D search).
    GRID_LEARNING_RATES: List[float] = field(default_factory=lambda: [5e-5, 1e-4, 3e-4, 1e-3])
    GRID_WEIGHT_DECAYS: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    GRID_DROPOUT_RATES: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    GRID_LR_SCHEDULERS: List[str] = field(default_factory=lambda: ["plateau", "cosine", "warmup_cosine"])
    GRID_WARMUP_EPOCHS: List[int] = field(default_factory=lambda: [2, 4])

    # Small extra grid for iTransformer attention heads.
    # Only combinations where (HIDDEN_DIM % NUM_HEADS == 0) will be tried.
    GRID_ITRANSFORMER_HEADS: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Runtime scheduler selection
    LR_SCHEDULER: str = "plateau"  # plateau / cosine / warmup_cosine
    WARMUP_EPOCHS: int = 3
    MIN_LR_RATIO: float = 0.1

    EXP_ROOT: str = "./Training_time_log"
    MODE: str = "train"
    LOAD_RUN_ID: str = ""

