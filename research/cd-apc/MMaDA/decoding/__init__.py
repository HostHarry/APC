from .config import VCHDDecodeConfig, vchd_config_from_dict
from .contrast import ContrastStats, compute_contrast_stats
from .decoder import visual_contrast_decode
from .history import (
    HistoryObservation,
    SparseHistory,
    history_adjusted_reliability,
    observe_sparse_history,
    sparse_distribution_from_dense,
    sparse_jsd,
)
from .mmada_adapter import (
    BranchKVCache,
    MMaDAVisualAccessAdapter,
    PairedKVCache,
    PairedLogits,
    build_paired_attention_bias,
)
from .vocabulary import build_valid_text_vocab
from .window import (
    CCAWState,
    WindowPressure,
    compute_window_pressure,
    pressure_adaptive_commit_budget,
    scope_next_hard_block,
    update_ccaw_state,
    update_inverse_ccaw_state,
)

__all__ = [
    "ContrastStats",
    "CCAWState",
    "BranchKVCache",
    "HistoryObservation",
    "MMaDAVisualAccessAdapter",
    "PairedKVCache",
    "PairedLogits",
    "SparseHistory",
    "VCHDDecodeConfig",
    "WindowPressure",
    "build_paired_attention_bias",
    "build_valid_text_vocab",
    "compute_contrast_stats",
    "compute_window_pressure",
    "history_adjusted_reliability",
    "observe_sparse_history",
    "pressure_adaptive_commit_budget",
    "scope_next_hard_block",
    "sparse_distribution_from_dense",
    "sparse_jsd",
    "update_ccaw_state",
    "update_inverse_ccaw_state",
    "vchd_config_from_dict",
    "visual_contrast_decode",
]
