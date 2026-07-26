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
from .lavida_adapter import (
    LaViDaVisualAccessAdapter,
    MMaDAVisualAccessAdapter,
    PairedLogits,
    TokenVisualAccessAdapter,
    build_paired_attention_bias,
    build_paired_attention_bias_from_mask,
    build_paired_prefix_attention_bias,
    infer_visual_mask_from_expanded_ids,
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
    "HistoryObservation",
    "LaViDaVisualAccessAdapter",
    "MMaDAVisualAccessAdapter",
    "PairedLogits",
    "SparseHistory",
    "TokenVisualAccessAdapter",
    "VCHDDecodeConfig",
    "WindowPressure",
    "build_paired_attention_bias",
    "build_paired_attention_bias_from_mask",
    "build_paired_prefix_attention_bias",
    "build_valid_text_vocab",
    "compute_contrast_stats",
    "compute_window_pressure",
    "history_adjusted_reliability",
    "infer_visual_mask_from_expanded_ids",
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
