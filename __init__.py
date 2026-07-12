"""
multivariate_sparse_connectivity
=================

Infer directed amplitude-amplitude cross-frequency coupling (AACFC)
from multichannel LFP / ECoG recordings with a sparse Lasso VAR and
band-dependent soft lag weights.

Quick start::

    import multivariate_sparse_connectivity as msc

    ctx = msc.load_session_features("recording.mat", orig_fs=1000.0, target_fs=100.0)
    res = msc.fit_weighted_lasso_soft_delays(
        ctx["X"], ctx["band_names"], ctx["bands"], ctx["fs_features"],
        morlet_q=ctx["morlet_q"],
    )
    A = res["A"]
"""

from . import dataproc, sysid, plots

from .dataproc import (
    load_mat_ecog,
    downsample_raw_ecog,
    extract_band_power,
    preprocess_band_power,
    load_session_features,
)
from .sysid import (
    compute_two_period_band_lags,
    compute_soft_delay_penalty_weights,
    build_full_lag_design_matrix,
    select_lambda_cross_validation,
    fit_weighted_lasso_soft_delays,
    lasso_nonzero_edge_mask,
    cross_both_edge_mask,
    decompose_r2_contributions,
    aggregate_lagged_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "dataproc",
    "sysid",
    "plots",
    "load_mat_ecog",
    "downsample_raw_ecog",
    "extract_band_power",
    "preprocess_band_power",
    "load_session_features",
    "compute_two_period_band_lags",
    "compute_soft_delay_penalty_weights",
    "build_full_lag_design_matrix",
    "select_lambda_cross_validation",
    "fit_weighted_lasso_soft_delays",
    "lasso_nonzero_edge_mask",
    "cross_both_edge_mask",
    "decompose_r2_contributions",
    "aggregate_lagged_matrix",
]
