"""
multivariate_sparse_connectivity.sysid
======================================

Multivariate sparse **amplitude-amplitude cross-frequency coupling**
(AACFC) on band-power (LFP envelope) features.

Given a preprocessed feature matrix ``X`` (channels × bands × time,
flattened to features × time), fits a time-lagged linear VAR whose
states are log-band-power envelopes.  Sparsity is enforced with
**Lasso** (L1); long delays are discouraged by a **band-dependent soft
lag weight matrix**.

Model
-----
Each target envelope ``x_i(t)`` (one channel–band amplitude feature) is
predicted from lagged amplitudes of all channels and bands:

    x_i(t) = Σ_l Σ_j A_{i,j,l} · x_j(t − l) + ε_i(t)

Coefficients are found by weighted Lasso (Gram coordinate descent,
same family as MATLAB ``lasso`` / glmnet):

- **Soft lag weights** — each band gets a trusted lag window
  ``two_period_lags[band]`` from its Morlet envelope timescale; weight
  ``w = 1`` inside that window, then grows with lag and centre
  frequency (``compute_soft_delay_penalty_weights``).
- **Global lambda** — when ``lambda_value=None``, purged blocked K-fold CV
  selects one lambda by minimising mean validation MSE across all targets
  (``select_lambda_cross_validation``).  Training rows within ``max_lag``
  after each validation block are excluded (lag leakage); optional
  ``cv_embargo`` drops extra rows.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any

import warnings
import numpy as np

from joblib import Parallel, delayed, effective_n_jobs
from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram
import scipy.sparse as _sp
import scipy.sparse.linalg as _spla


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-column R² with NaN for zero-variance targets.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n_samples, n_targets)

    Returns
    -------
    r2 : ndarray, shape (n_targets,)
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
        r2[ss_tot == 0] = np.nan
    return r2


def compute_two_period_band_lags(
    frequency_bands: Dict[str, Tuple[float, float]],
    fs_features: float,
    morlet_q: float = 6.0,
    min_periods: float = 2.0,
    max_samples: Optional[int] = None,
) -> Dict[str, int]:
    """
    After Morlet CWT the VAR states are log-power **envelopes**.  The
    relevant memory timescale is set by the envelope bandwidth, not the
    carrier frequency:

        BW_envelope = f_center / q
        lag = ceil(min_periods * fs_features / BW_envelope)

    Here ``q`` is the wavelet *quality factor* (number of cycles), the
    modelling quantity that sets the envelope timescale ``tau = q /
    f_center``.  Pass the **identical** ``morlet_q`` to this function and to
    ``extract_band_power`` so the wavelet bandwidth and the penalty timescale
    stay consistent.

    Parameters
    ----------
    frequency_bands : dict
        Band name → ``(f_lo, f_hi)`` in Hz.
    fs_features : float
        Sampling rate of the envelope-downsampled feature matrix (Hz).
    morlet_q : float
        Wavelet quality factor (number of cycles) defining the envelope
        timescale ``tau = q / f_center``.  Pass the **same** ``morlet_q``
        used in ``extract_band_power`` so the wavelet and the penalty agree.
    min_periods : float
        Number of envelope autocorrelation lengths to include (default 2).
    max_samples : int or None
        Optional cap on each band's lag in samples.

    Returns
    -------
    dict
        Band name → lag in samples at ``fs_features``.
    """
    if fs_features <= 0:
        raise ValueError("fs_features must be positive.")
    if morlet_q <= 0:
        raise ValueError("morlet_q must be positive.")

    band_lags = {}
    for band, (f_lo, f_hi) in frequency_bands.items():
        if f_lo <= 0 or f_hi <= f_lo:
            raise ValueError(f"Bad band {band}: ({f_lo}, {f_hi}).")
        f_center = 0.5 * (f_lo + f_hi)
        bw_env = f_center / morlet_q          # envelope bandwidth in Hz
        lag = int(np.ceil(min_periods * fs_features / bw_env))
        if max_samples is not None:
            lag = min(lag, max_samples)
        lag = max(1, lag)
        band_lags[band] = lag
    return band_lags


def compute_soft_delay_penalty_weights(
    n_channels: int,
    band_names: List[str],
    two_period_lags: Dict[str, int],
    frequency_bands: Dict[str, Tuple[float, float]],
    max_lag: int,
    long_lag_penalty_power: float = 1.5,
    min_weight: float = 1e-6,
    dtype=np.float32,
) -> np.ndarray:
    """
    Lasso column weights for the frequency-aware soft delay penalty.

    Within each band's two-period window (``two_period_lags[band]``) every
    lag column has weight 1.0.  Beyond that window the weight grows with
    normalised lag excess and is scaled by centre frequency so fast bands
    are penalised more strongly at long delays:

        w(lag, b) = 1 + (f_b / f_min) * ((lag - thr_b) / thr_b) ** p

    Column ordering matches ``build_full_lag_design_matrix``: lag blocks
    stacked as ``[lag1_feat0..lag1_featF, lag2_feat0.., ...]``, with
    features ordered ``(ch0_band0, ch0_band1, ..., ch1_band0, ...)``.

    Parameters
    ----------
    n_channels : int
    band_names : list of str
    two_period_lags : dict
        Per-band trusted lag counts from ``compute_two_period_band_lags``.
    frequency_bands : dict
        Band name → ``(f_lo, f_hi)`` in Hz (used for centre frequencies).
    max_lag : int
        Global VAR horizon in samples.
    long_lag_penalty_power : float
        Exponent ``p`` on normalised lag excess beyond each band's window.
    min_weight : float
        Floor applied to every weight (numerical stability).
    dtype : numpy dtype
        Output dtype (default ``float32``).

    Returns
    -------
    weights : ndarray, shape ``(n_features * max_lag,)``
        Flat weight vector for column scaling ``X_scaled = X_design / weights``.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")
    if long_lag_penalty_power <= 0:
        raise ValueError("long_lag_penalty_power must be positive.")

    f_center = np.array(
        [0.5 * (frequency_bands[b][0] + frequency_bands[b][1]) for b in band_names],
        dtype=np.float64,
    )
    if np.any(f_center <= 0):
        raise ValueError("All band centre frequencies must be positive.")
    band_scale = f_center / float(np.min(f_center))

    thr = np.tile(
        [max(1, int(two_period_lags[b])) for b in band_names],
        n_channels,
    ).astype(np.float64)
    scale = np.tile(band_scale, n_channels)

    lags = np.arange(1, max_lag + 1, dtype=np.float64)[:, None]
    excess = np.maximum(0.0, (lags - thr[None, :]) / thr[None, :])
    lag_w = 1.0 + scale[None, :] * (excess ** long_lag_penalty_power)
    return np.maximum(min_weight, lag_w).astype(dtype, copy=False).ravel()


def cross_both_edge_mask(
    edge_mask: np.ndarray,
    n_channels: int,
    n_bands: int,
) -> np.ndarray:
    """
    Restrict a feature×feature mask to cross-channel **and** cross-band pairs.

    Parameters
    ----------
    edge_mask : ndarray, shape ``(n_features, n_features)``
        Boolean edge mask (e.g. Lasso nonzero entries); rows = target,
        columns = source.
    n_channels, n_bands : int
        Feature layout ``n_features = n_channels * n_bands``.

    Returns
    -------
    mask : ndarray of bool, same shape as ``edge_mask``
    """
    sig = np.asarray(edge_mask, dtype=bool)
    n_features = n_channels * n_bands
    if sig.shape != (n_features, n_features):
        raise ValueError(
            f"edge_mask must be ({n_features}, {n_features}), got {sig.shape}."
        )
    feat = np.arange(n_features)
    ch, band = divmod(feat, n_bands)
    cross_ch = ch[:, None] != ch[None, :]
    cross_b = band[:, None] != band[None, :]
    return sig & cross_ch & cross_b


def lasso_nonzero_edge_mask(
    A: np.ndarray,
    eps: float = 1e-8,
    reduce: str = "max_abs",
) -> np.ndarray:
    """
    Feature×feature mask of Lasso-selected edges (any lag nonzero).

    Parameters
    ----------
    A : ndarray, shape ``(n_features, n_features * max_lag)``
        Fitted transition matrix.
    eps : float
        Coefficient magnitude threshold.
    reduce : str
        How to collapse lags: ``max_abs`` (default), ``sum_abs``, or ``any``.

    Returns
    -------
    mask : ndarray of bool, shape ``(n_features, n_features)``
        ``True`` when the source→target link survives sparsity at ``eps``.
    """
    A = np.asarray(A, dtype=float)
    n_features = A.shape[0]
    max_lag = A.shape[1] // n_features
    A3 = A.reshape(n_features, max_lag, n_features)
    if reduce == "max_abs":
        strength = np.abs(A3).max(axis=1)
    elif reduce == "sum_abs":
        strength = np.abs(A3).sum(axis=1)
    elif reduce == "any":
        strength = np.abs(A3).max(axis=1)
    else:
        raise ValueError("reduce must be 'max_abs', 'sum_abs', or 'any'.")
    return strength > eps


def build_full_lag_design_matrix(
    X: np.ndarray, max_lag: int, dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the lag-stacked VAR design matrix and one-step-ahead targets.

    For each time index ``t >= max_lag``, ``Y[t] = X[:, t]`` and the
    predictors are ``X[:, t-1], X[:, t-2], ..., X[:, t-max_lag]`` stacked
    as columns.

    Parameters
    ----------
    X : ndarray, shape ``(n_features, T)``
    max_lag : int
        Number of past lags included as predictors.
    dtype : numpy dtype
        Output dtype (default ``float32``).

    Returns
    -------
    X_design : ndarray, shape ``(T - max_lag, n_features * max_lag)``
        Column order:
        ``[lag1_feat0..lag1_featF, lag2_feat0..lag2_featF, ...]``.
    Y : ndarray, shape ``(T - max_lag, n_features)``
        One-step-ahead targets aligned with ``X_design`` rows.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=dtype))
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    n_feat, T = X.shape
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")
    if T <= max_lag:
        raise ValueError(f"T={T} <= max_lag={max_lag}.")

    Y = np.ascontiguousarray(X[:, max_lag:].T)
    X_design = np.empty((T - max_lag, n_feat * max_lag), dtype=dtype)
    for lag in range(1, max_lag + 1):
        X_design[:, (lag - 1) * n_feat : lag * n_feat] = X[:, max_lag - lag : T - lag].T
    return X_design, Y


# Sparse VAR core: weighted Lasso

def _fit_target_gram_cd(
    Q: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
    random_cd: bool = False,
    w_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Single-target Lasso via Gram coordinate descent (sklearn / glmnet family).

    Minimises the canonical Gram form

        (1/2) w^T Q w - q^T w + alpha * ||w||_1

    where ``Q = X^T X / n`` and ``q = X^T y / n`` for the (possibly
    column-scaled) design matrix.

    Parameters
    ----------
    Q : ndarray, shape ``(p, p)``
        Normalised Gram matrix (shared across targets when unchanged).
    q : ndarray, shape ``(p,)``
        Normalised ``X^T y`` vector for this target.
    y : ndarray, shape ``(n_samples,)``
        Target vector (passed through to the Cython kernel).
    alpha : float
        L1 penalty strength (λ in the weighted-Lasso pipeline).
    max_iter, tol : int, float
        Coordinate-descent stopping criteria.
    rng : numpy random Generator
        Used only when ``random_cd=True``.
    random_cd : bool
        If False (default), use cyclic coordinate descent.
    w_init : ndarray or None
        Warm-start coefficient vector from the previous (larger) λ.
        ``None`` → cold start at zero.

    Returns
    -------
    w : ndarray, shape ``(p,)``, float32
        Fitted coefficient vector in the scaled (Gram) space.
    """
    n_features = Q.shape[0]
    if Q.dtype == np.float64:
        Q64 = Q
    else:
        Q64 = np.ascontiguousarray(Q, dtype=np.float64)
    if q.dtype == np.float64:
        q64 = q
    else:
        q64 = np.ascontiguousarray(q, dtype=np.float64)
    if y.dtype == np.float64:
        y64 = y
    else:
        y64 = np.ascontiguousarray(y, dtype=np.float64)

    if w_init is None:
        w64 = np.zeros(n_features, dtype=np.float64)
    else:
        w64 = np.ascontiguousarray(w_init, dtype=np.float64).copy()

    rng_state = np.random.RandomState(int(rng.integers(0, 2**31 - 1)))
    enet_coordinate_descent_gram(
        w64,
        float(alpha),
        0.0,
        Q64,
        q64,
        y64,
        int(max_iter),
        float(tol),
        rng_state,
        1 if random_cd else 0,
        0,
    )
    return w64.astype(np.float32, copy=False)


def _effective_cv_n_jobs(n_jobs: int, n_predictors: int) -> int:
    """
    Cap parallel workers for CV / final fits on large Gram systems.

    When ``p`` (number of predictors) is large, many threaded Lasso solves
    can spike RAM even if they share the same ``Q`` matrix.

    Rules
    -----
    - ``p >= 8000`` → 1 worker (sequential targets)
    - ``p >= 4000`` → at most 2 workers
    - otherwise → ``effective_n_jobs(n_jobs)``

    Parameters
    ----------
    n_jobs : int
        Requested job count (``-1`` = all cores).
    n_predictors : int
        Number of columns in the design matrix / Gram matrix.

    Returns
    -------
    int
        Safe worker count for ``joblib.Parallel``.
    """
    jobs = effective_n_jobs(n_jobs)
    if n_predictors >= 8000:
        return 1
    if n_predictors >= 4000:
        return min(jobs, 2)
    return jobs


def _make_contiguous_cv_folds(n_samples: int, cv_folds: int) -> List[np.ndarray]:
    """
    Blocked, time-ordered CV fold indices (no shuffle).

    Splits ``0 .. n_samples-1`` into contiguous blocks so validation
    folds respect temporal order and avoid leakage across time.

    Parameters
    ----------
    n_samples : int
        Number of design-matrix rows.
    cv_folds : int
        Number of folds (must be ``>= 2`` and ``<= n_samples``).

    Returns
    -------
    folds : list of ndarray
        Each entry is the validation index array for one fold.
    """
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2.")
    if cv_folds > n_samples:
        raise ValueError(
            f"cv_folds={cv_folds} exceeds n_samples={n_samples}."
        )
    fold_sizes = np.full(cv_folds, n_samples // cv_folds, dtype=int)
    fold_sizes[: n_samples % cv_folds] += 1
    folds: List[np.ndarray] = []
    start = 0
    for size in fold_sizes:
        folds.append(np.arange(start, start + size, dtype=int))
        start += size
    return folds


def _purged_train_indices(
    n_samples: int,
    val_idx: np.ndarray,
    cv_purge_gap: int,
    cv_embargo: int = 0,
) -> np.ndarray:
    """
    Training design-row indices with purge + embargo around a validation block.

    Training rows immediately after a validation fold use lagged predictors
    that overlap validation targets; those rows are removed when
    ``cv_purge_gap >= max_lag``.
    """
    val_idx = np.asarray(val_idx, dtype=int)
    if val_idx.size == 0:
        return np.arange(n_samples, dtype=int)

    v_start = int(val_idx.min())
    v_end = int(val_idx.max()) + 1

    val_mask = np.zeros(n_samples, dtype=bool)
    val_mask[val_idx] = True

    purge_mask = np.zeros(n_samples, dtype=bool)
    gap_after = int(cv_purge_gap) + int(cv_embargo)
    if gap_after > 0:
        purge_mask[v_end : min(n_samples, v_end + gap_after)] = True
    if cv_embargo > 0:
        purge_mask[max(0, v_start - cv_embargo) : v_start] = True

    return np.where(~(val_mask | purge_mask))[0]


def _matlab_lambda_grid(
    XtY_T: np.ndarray,
    n_lambda: int,
    lambda_min_ratio: float,
) -> np.ndarray:
    """
    Log-spaced λ grid from λ_max down to ``lambda_min_ratio * λ_max``.

    ``λ_max = max |X^T y / n|`` across all targets — above this value
    the Lasso solution is exactly zero (KKT condition).  Matches the
    default path in MATLAB ``lasso``.

    Parameters
    ----------
    XtY_T : ndarray, shape ``(n_targets, p)``
        Normalised ``(X^T Y / n)^T`` rows, one per target.
    n_lambda : int
        Number of grid points (``>= 2``).
    lambda_min_ratio : float
        Ratio of smallest to largest λ (in ``(0, 1)``).

    Returns
    -------
    lambdas : ndarray, shape ``(n_lambda,)``
        Decreasing λ values (high → low).
    """
    if n_lambda < 2:
        raise ValueError("n_lambda must be >= 2.")
    if not (0.0 < lambda_min_ratio < 1.0):
        raise ValueError("lambda_min_ratio must be in (0, 1).")
    lam_max = float(np.max(np.abs(XtY_T)))
    if lam_max <= 0.0 or not np.isfinite(lam_max):
        lam_max = 1.0
    return np.geomspace(lam_max, lam_max * lambda_min_ratio, int(n_lambda))


def _prepare_cv_fold_cache(
    X_scaled: np.ndarray,
    Y: np.ndarray,
    folds: List[np.ndarray],
    n_samples: int,
    cv_purge_gap: int = 0,
    cv_embargo: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Precompute train-fold Gram objects once per CV split.

    Each fold stores a **single shared float64** ``Q`` matrix so target
    fits during CV do not re-allocate ``p × p`` arrays.

    Parameters
    ----------
    X_scaled : ndarray, shape ``(n_samples, p)``
        Column-scaled design matrix (``X_design / weights``).
    Y : ndarray, shape ``(n_samples, n_features)``
        One-step-ahead targets aligned with ``X_scaled``.
    folds : list of ndarray
        Validation index arrays from ``_make_contiguous_cv_folds``.
    n_samples : int
        Number of rows in ``X_scaled`` / ``Y``.
    cv_purge_gap : int
        Training rows removed after each validation block (lag leakage).
        Set to ``max_lag`` for a purged blocked CV.
    cv_embargo : int
        Extra training rows dropped adjacent to each validation block.

    Returns
    -------
    cache : list of tuple
        One entry per fold:

        ``(Q_tr, q_tr_T, y_tr_T, X_val, Y_val)``

        - ``Q_tr``       — ``(p, p)`` float64 Gram on the train split
        - ``q_tr_T``     — ``(n_targets, p)`` float64 ``X_tr^T Y_tr / n_tr``
        - ``y_tr_T``     — ``(n_targets, n_train)`` float64 train targets
        - ``X_val``      — ``(n_val, p)`` float32 validation predictors
        - ``Y_val``      — ``(n_val, n_features)`` validation targets
    """
    cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for val_idx in folds:
        tr_idx = _purged_train_indices(
            n_samples, val_idx, cv_purge_gap=cv_purge_gap, cv_embargo=cv_embargo,
        )
        if tr_idx.size == 0:
            raise ValueError(
                "Purged CV removed all training rows for a fold. "
                "Reduce cv_purge_gap / cv_embargo, use fewer cv_folds, "
                "or provide more data."
            )
        X_tr = X_scaled[tr_idx]
        X_val = np.ascontiguousarray(X_scaled[val_idx], dtype=np.float32)
        Y_tr = Y[tr_idx]
        Y_val = Y[val_idx]
        inv_n_tr = 1.0 / float(len(tr_idx))
        Q_tr = np.ascontiguousarray((X_tr.T @ X_tr) * inv_n_tr, dtype=np.float64)
        q_tr_T = np.ascontiguousarray((X_tr.T @ Y_tr).T * inv_n_tr, dtype=np.float64)
        y_tr_T = np.ascontiguousarray(Y_tr.T, dtype=np.float64)
        cache.append((Q_tr, q_tr_T, y_tr_T, X_val, Y_val))
    return cache


def _cv_eval_lambda_path(
    lambdas: np.ndarray,
    fold_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_targets: int,
    max_iter: int,
    tol: float,
    n_jobs: int,
    random_cd: bool,
    warm_state: Optional[List[List[Optional[np.ndarray]]]] = None,
    seed: int = 0,
    verbose: bool = False,
    progress_label: str = "lambda grid",
) -> Tuple[np.ndarray, List[List[Optional[np.ndarray]]]]:
    """
    Mean validation MSE along a decreasing λ path (warm-started).

    Evaluates each λ in order from high to low.  For every ``(fold, target)``
    the previous solution is passed as ``w_init`` to speed up coordinate
    descent (glmnet-style warm start).

    Parameters
    ----------
    lambdas : ndarray
        Decreasing λ values to evaluate.
    fold_cache : list
        Output of ``_prepare_cv_fold_cache``.
    n_targets : int
        Number of VAR targets (features).
    max_iter, tol : int, float
        Passed to ``_fit_target_gram_cd``.
    n_jobs : int
        Parallel workers for targets within each fold (auto-capped).
    random_cd : bool
        Random vs cyclic coordinate descent.
    warm_state : list of list or None
        Optional ``warm_state[fold][target]`` coefficient buffers updated
        in place.  ``None`` → fresh cold starts at the first λ.
    seed : int
        Base seed for per-target RNG streams.
    verbose : bool
        Print coarse progress every ~12.5% of the grid.
    progress_label : str
        Prefix for progress log lines.

    Returns
    -------
    cv_mse : ndarray, shape ``(len(lambdas),)``
        Mean validation MSE at each λ (pooled over folds and targets).
    warm_state : list of list
        Updated coefficient buffers for continued warm starting.
    """
    n_folds = len(fold_cache)
    if warm_state is None:
        warm_state = [[None] * n_targets for _ in range(n_folds)]

    cv_mse = np.empty(len(lambdas), dtype=np.float64)
    cv_jobs = _effective_cv_n_jobs(n_jobs, fold_cache[0][0].shape[0])

    for i, lam in enumerate(lambdas):
        fold_mses: List[float] = []
        alpha = float(lam)

        for fold_i, (Q_tr, q_tr_T, y_tr_T, X_val, Y_val) in enumerate(fold_cache):
            def _target_mse(t: int) -> float:
                w_init = warm_state[fold_i][t]
                rng_t = np.random.default_rng(seed + t + fold_i * n_targets + i * 9973)
                w = _fit_target_gram_cd(
                    Q_tr,
                    q_tr_T[t],
                    y_tr_T[t],
                    alpha,
                    max_iter,
                    tol,
                    rng_t,
                    random_cd,
                    w_init=w_init,
                )
                warm_state[fold_i][t] = w
                y_pred = X_val @ w
                return float(np.mean((Y_val[:, t] - y_pred) ** 2))

            if cv_jobs == 1:
                target_mses = [_target_mse(t) for t in range(n_targets)]
            else:
                target_mses = Parallel(
                    n_jobs=cv_jobs, backend="threading", pre_dispatch="2*n_jobs"
                )(delayed(_target_mse)(t) for t in range(n_targets))
            fold_mses.append(float(np.mean(target_mses)))

        cv_mse[i] = float(np.mean(fold_mses))
        if verbose and (i == 0 or (i + 1) % max(1, len(lambdas) // 8) == 0):
            print(f"[CV]   {progress_label}: {i + 1}/{len(lambdas)}")

    return cv_mse, warm_state


def _cv_score_lambda_cold(
    alpha: float,
    fold_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_targets: int,
    max_iter: int,
    tol: float,
    n_jobs: int,
    random_cd: bool,
    seed: int,
) -> float:
    """
    Single-λ CV score with cold starts.

    Used for the small refine grid around the coarse-path minimum where
    warm-start continuity from the coarse path is not guaranteed.

    Returns
    -------
    float
        Mean validation MSE at ``alpha``, pooled over folds and targets.
    """
    cv_mse, _ = _cv_eval_lambda_path(
        np.array([alpha], dtype=np.float64),
        fold_cache,
        n_targets,
        max_iter,
        tol,
        n_jobs,
        random_cd,
        warm_state=[[None] * n_targets for _ in range(len(fold_cache))],
        seed=seed,
        verbose=False,
    )
    return float(cv_mse[0])


def select_lambda_cross_validation(
    X_scaled: np.ndarray,
    Y: np.ndarray,
    XtY_T: np.ndarray,
    cv_folds: int = 5,
    n_lambda: int = 30,
    lambda_min_ratio: float = 1e-4,
    lambda_refine: bool = True,
    max_iter: int = 10000,
    tol: float = 1e-5,
    n_jobs: int = -1,
    random_cd: bool = False,
    cv_purge_gap: int = 0,
    cv_embargo: int = 0,
    verbose: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Select lambda by purged blocked K-fold CV.

    One global lambda minimises mean validation MSE pooled across all VAR
    targets and folds.  Training rows within ``cv_purge_gap`` samples after
    each validation block are excluded so lagged predictors do not overlap
    validation targets.  Optional ``cv_embargo`` drops additional rows for
    slow envelope autocorrelation.

    The coarse grid is evaluated along a warm-started path (high to low
    lambda).  When ``lambda_refine=True`` and ``n_lambda > 20``, up to 12
    extra cold-start points refine the neighbourhood of the coarse minimum.

    Parameters
    ----------
    X_scaled : ndarray, shape ``(n_samples, p)``
        Column-scaled design matrix passed to fold slicing.
    Y : ndarray, shape ``(n_samples, n_features)``
        One-step-ahead targets.
    XtY_T : ndarray, shape ``(n_features, p)``
        Full-data normalised ``(X^T Y / n)^T`` for building the λ grid.
    cv_folds : int
        Number of blocked time-ordered folds.
    n_lambda : int
        Target number of λ evaluations (coarse + optional refine).
    lambda_min_ratio : float
        Smallest λ = ``lambda_min_ratio * λ_max``.
    lambda_refine : bool
        If True and ``n_lambda > 20``, add a fine grid around the coarse
        CV minimum.
    max_iter, tol : int, float
        Coordinate-descent settings for each Lasso sub-problem.
    n_jobs : int
        Parallel workers for targets within each fold (auto-capped).
    random_cd : bool
        Random vs cyclic coordinate descent.
    cv_purge_gap : int
        Training rows purged after each validation block (use ``max_lag``).
    cv_embargo : int
        Extra rows dropped adjacent to each validation block.
    verbose : bool
        Print fold-cache size, progress, and selected lambda.

    Returns
    -------
    lambda_best : float
        CV-selected λ.
    lambda_grid : ndarray
        All evaluated λ values (sorted high → low).
    cv_mse : ndarray
        Mean validation MSE at each entry of ``lambda_grid``.
    """
    n_samples = X_scaled.shape[0]
    n_targets = Y.shape[1]
    n_predictors = X_scaled.shape[1]
    folds = _make_contiguous_cv_folds(n_samples, cv_folds)

    n_coarse = int(n_lambda) if not lambda_refine or n_lambda <= 20 else min(20, int(n_lambda))
    lambdas_coarse = _matlab_lambda_grid(XtY_T, n_coarse, lambda_min_ratio)

    if verbose:
        cv_jobs = _effective_cv_n_jobs(n_jobs, n_predictors)
        print(
            f"[CV] {cv_folds} purged blocked folds "
            f"(purge_gap={cv_purge_gap}, embargo={cv_embargo}), "
            f"{n_coarse} coarse lambda values "
            f"({lambdas_coarse[0]:.4e} -> {lambdas_coarse[-1]:.4e}), "
            f"{n_targets} targets, cv_n_jobs={cv_jobs}; "
            f"building fold Gram matrices ..."
        )

    fold_cache = _prepare_cv_fold_cache(
        X_scaled, Y, folds, n_samples,
        cv_purge_gap=cv_purge_gap, cv_embargo=cv_embargo,
    )
    if verbose:
        q_gb = sum(item[0].nbytes for item in fold_cache) / 1024**3
        print(f"[CV] fold cache: {len(fold_cache)} Q matrices ({q_gb:.2f} GB)")

    cv_mse, warm_state = _cv_eval_lambda_path(
        lambdas_coarse,
        fold_cache,
        n_targets,
        max_iter,
        tol,
        n_jobs,
        random_cd,
        verbose=verbose,
        progress_label="coarse lambda path",
    )

    lambdas_all = lambdas_coarse
    mse_all = cv_mse

    if lambda_refine and n_lambda > n_coarse:
        best_i = int(np.argmin(cv_mse))
        i_lo = min(best_i + 1, len(lambdas_coarse) - 1)
        i_hi = max(best_i - 1, 0)
        lam_lo = float(lambdas_coarse[i_lo])
        lam_hi = float(lambdas_coarse[i_hi])
        if lam_hi > lam_lo:
            n_fine = min(12, int(n_lambda) - n_coarse)
            lambdas_fine = np.geomspace(lam_hi, lam_lo, n_fine + 2)[1:-1]
            if verbose:
                print(
                    f"[CV] refining with {len(lambdas_fine)} extra lambda values "
                    f"between {lam_hi:.4e} and {lam_lo:.4e} ..."
                )
            mse_fine = np.empty(len(lambdas_fine), dtype=np.float64)
            for j, lam in enumerate(lambdas_fine):
                mse_fine[j] = _cv_score_lambda_cold(
                    float(lam),
                    fold_cache,
                    n_targets,
                    max_iter,
                    tol,
                    n_jobs,
                    random_cd,
                    seed=1000 + j,
                )
            lambdas_all = np.concatenate([lambdas_coarse, lambdas_fine])
            mse_all = np.concatenate([cv_mse, mse_fine])
            order = np.argsort(-lambdas_all)
            lambdas_all = lambdas_all[order]
            mse_all = mse_all[order]

    best_i = int(np.argmin(mse_all))
    lambda_best = float(lambdas_all[best_i])
    if verbose:
        print(
            f"[CV] selected lambda = {lambda_best:.6e}  "
            f"(CV MSE = {mse_all[best_i]:.6e})"
        )
    return lambda_best, lambdas_all, mse_all


def fit_weighted_lasso_soft_delays(
    X: np.ndarray,
    band_names: List[str],
    frequency_bands: Dict[str, Tuple[float, float]],
    fs_features: float,
    lambda_value: Optional[float] = None,
    max_lag: Optional[int] = None,
    max_lag_multiplier: float = 1.0,
    max_samples: Optional[int] = None,
    min_periods: float = 2.0,
    morlet_q: float = 6.0,
    long_lag_penalty_power: float = 1.5,
    max_iter: int = 50000,
    tol: float = 1e-5,
    n_jobs: int = -1,
    random_cd: bool = False,
    cv_folds: int = 5,
    n_lambda: int = 30,
    lambda_min_ratio: float = 1.5e-3,
    lambda_refine: bool = True,
    cv_purge_gap: Optional[int] = None,
    cv_embargo: int = 0,
    verbose: bool = True,
    prebuilt_design: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Fit a multivariate sparse AACFC model by weighted Lasso.

    States are band-power (amplitude) envelopes; nonzero coefficients in
    ``A`` encode directed amplitude-amplitude coupling across channels
    and frequency bands at specific lags.

    Pipeline
    --------
    1. Compute per-band two-period lags and soft delay penalty weights.
    2. Build the lag-stacked design matrix ``X_design`` and scale columns
       by ``1 / weights`` to implement weighted L1 in standard Lasso form.
    3. Precompute the full-data Gram matrix ``Q`` and ``X^T Y`` once.
    4. If ``lambda_value is None``, run purged blocked K-fold CV to pick lambda.
    5. Fit all targets in parallel at the chosen λ and back-transform
       coefficients to the original (unscaled) basis → transition matrix ``A``.

    λ selection
    -----------
    ``lambda_value=None`` (default) → ``select_lambda_cross_validation``
    with warm-started coarse path and optional refine step.
    Pass an explicit ``lambda_value > 0`` to skip CV.

    Solver
    ------
    sklearn Gram coordinate descent (MATLAB ``lasso`` / glmnet family).
    CV uses warm starts along the decreasing λ path; the final full-data
    fit uses a single cold start per target at the selected λ.

    Memory / parallelism
    --------------------
    ``X_scaled`` is freed after Gram formation.  Parallel target fits are
    auto-limited when ``p = n_features * max_lag`` is large (see
    ``_effective_cv_n_jobs``).

    Parameters
    ----------
    X : ndarray, shape ``(n_features, T)``
        Preprocessed feature matrix (channels × bands flattened).
    band_names : list of str
        Band labels in feature order ``(ch0_b0, ch0_b1, ..., ch1_b0, ...)``.
    frequency_bands : dict
        Band name → ``(f_lo, f_hi)`` in Hz.
    fs_features : float
        Sampling rate of ``X`` after envelope downsampling (Hz).
    lambda_value : float or None
        Fixed Lasso penalty, or ``None`` for CV (default).
    max_lag : int or None
        VAR order.  ``None`` → ``ceil(max_lag_multiplier * max(two_period_lags))``.
    max_lag_multiplier : float
        Scales the automatic ``max_lag`` choice.
    max_samples : int or None
        Cap on per-band two-period lags (samples).
    min_periods : float
        Envelope periods in ``compute_two_period_band_lags``.
    morlet_q : float
        Wavelet quality factor (number of cycles) setting the envelope
        timescale ``tau = q / f_center`` for the soft-delay penalty.  Must
        equal the ``morlet_q`` passed to ``extract_band_power``.
    long_lag_penalty_power : float
        Exponent ``p`` in ``compute_soft_delay_penalty_weights``.
    max_iter, tol : int, float
        Coordinate-descent stopping criteria.
    n_jobs : int
        Parallel workers (``-1`` = all cores, subject to auto cap).
    random_cd : bool
        Random vs cyclic coordinate descent.
    cv_folds : int
        Blocked K-fold count for λ selection.
    n_lambda : int
        Target number of CV λ evaluations.
    lambda_min_ratio : float
        Smallest CV λ as a fraction of λ_max.
    lambda_refine : bool
        Enable fine-grid refinement around the coarse CV minimum.
    cv_purge_gap : int or None
        Training rows purged after each CV validation block.  ``None``
        (default) uses ``max_lag`` so lagged predictors cannot overlap
        validation targets.
    cv_embargo : int
        Extra training rows dropped adjacent to each validation block
        (slow-envelope autocorrelation guard).
    verbose : bool
        Print lag structure, memory use, CV progress, and diagnostics.
    prebuilt_design : tuple or None
        Optional ``(X_design, Y)`` to skip design-matrix construction
        (must match ``n_features`` and ``max_lag``).

    Returns
    -------
    dict
        Keys include:

        - ``A`` — ``(n_features, n_features * max_lag)`` transition matrix
        - ``max_lag``, ``two_period_lags``, ``penalty_weights``
        - ``mean_r2``, ``per_target_r2``, ``spectral_radius``
        - ``lambda_value``, ``lambda_grid``, ``lambda_cv_mse``
          (grid/mse are ``None`` when λ was supplied manually)
        - ``long_lag_penalty_power``
        - ``n_channels``, ``n_bands``, ``n_features``, ``band_names``
    """
    if lambda_value is not None and lambda_value <= 0:
        raise ValueError("lambda_value must be > 0 when provided explicitly.")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (features, time), got {X.shape}.")
    if not np.all(np.isfinite(X)):
        warnings.warn("X has non-finite values — replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_features, T = X.shape
    n_bands = len(band_names)
    if n_bands == 0:
        raise ValueError("band_names is empty.")
    if n_features % n_bands != 0:
        raise ValueError(
            f"n_features={n_features} is not divisible by n_bands={n_bands}."
        )
    n_channels = n_features // n_bands

    missing = [b for b in band_names if b not in frequency_bands]
    if missing:
        raise ValueError(f"Bands missing from frequency_bands: {missing}")

    two_period_lags = compute_two_period_band_lags(frequency_bands, fs_features, morlet_q, min_periods, max_samples)

    if max_lag is None:
        max_lag = int(np.ceil(max_lag_multiplier * max(two_period_lags.values())))
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")
    if T <= max_lag:
        raise ValueError(f"Need T > max_lag, got T={T}, max_lag={max_lag}.")

    if verbose:
        print(f"[VAR] n_channels={n_channels}  n_bands={n_bands}  "
              f"n_features={n_features}  max_lag={max_lag}")
        for b in band_names:
            print(f"      two-period lag for {b:<10s}: "
                  f"{two_period_lags[b]} samples "
                  f"({two_period_lags[b]/fs_features:.2f} s)")

    if prebuilt_design is None:
        X_design, Y = build_full_lag_design_matrix(X, max_lag=max_lag, dtype=np.float32)
    else:
        X_design, Y = prebuilt_design
        X_design = np.ascontiguousarray(X_design, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)
        if X_design.shape[1] != n_features * max_lag:
            raise ValueError(
                f"prebuilt_design X_design has {X_design.shape[1]} columns; "
                f"expected {n_features * max_lag}."
            )
        if Y.shape[1] != n_features:
            raise ValueError(
                f"prebuilt_design Y has {Y.shape[1]} columns; "
                f"expected {n_features}."
            )
        if X_design.shape[0] != Y.shape[0]:
            raise ValueError(
                f"prebuilt_design row mismatch: X_design has "
                f"{X_design.shape[0]} rows but Y has {Y.shape[0]}."
            )
        if not (np.all(np.isfinite(X_design)) and np.all(np.isfinite(Y))):
            warnings.warn(
                "prebuilt_design has non-finite values — "
                "replacing with 0 to avoid solver crash.",
                UserWarning,
            )
            X_design = np.nan_to_num(X_design, nan=0.0, posinf=0.0, neginf=0.0)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    weights = compute_soft_delay_penalty_weights(
        n_channels=n_channels,
        band_names=band_names,
        two_period_lags=two_period_lags,
        frequency_bands=frequency_bands,
        max_lag=max_lag,
        long_lag_penalty_power=long_lag_penalty_power,
        dtype=np.float32,
    )
    n_samples = X_design.shape[0]

    # Solving in the rescaled space X/w is equivalent to weighted L1;
    # recover beta = beta_scaled / w afterwards.
    inv_w = (1.0 / weights).astype(np.float32)
    X_scaled = X_design * inv_w

    if verbose:
        gb = (X_scaled.nbytes + Y.nbytes) / 1024**3
        print(f"[VAR] design matrix: {X_scaled.shape}  ({gb:.2f} GB)")

    # Normalise by n_samples so the objective is the canonical Lasso
    #   (1/2) w^T Q w - q^T w + alpha ||w||_1,  Q = X^T X / n,  q = X^T y / n.
    # Conditioning of Q is then independent of n_samples, so the solver
    # behaves identically on the full series and on any train slice.
    # In this form alpha equals lambda_value.
    if verbose:
        print("[VAR] precomputing Gram and Xy ...")
    inv_n = 1.0 / float(n_samples)
    Q = np.ascontiguousarray(
        ((X_scaled.T @ X_scaled) * inv_n).astype(np.float64)
    )
    # Targets on axis 0 so each row is C-contiguous.
    XtY_T = np.ascontiguousarray(
        ((X_scaled.T @ Y) * inv_n).T.astype(np.float64)
    )
    Y64_T = np.ascontiguousarray(Y.T.astype(np.float64))
    if verbose:
        print(f"[VAR] Gram shape: {Q.shape}   "
              f"({Q.nbytes / 1024**3:.2f} GB)")

    lambda_grid: Optional[np.ndarray] = None
    lambda_cv_mse: Optional[np.ndarray] = None
    if lambda_value is None:
        purge_gap = int(max_lag if cv_purge_gap is None else cv_purge_gap)
        lambda_value, lambda_grid, lambda_cv_mse = select_lambda_cross_validation(
            X_scaled=X_scaled,
            Y=Y,
            XtY_T=XtY_T,
            cv_folds=cv_folds,
            n_lambda=n_lambda,
            lambda_min_ratio=lambda_min_ratio,
            lambda_refine=lambda_refine,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            random_cd=random_cd,
            cv_purge_gap=purge_gap,
            cv_embargo=cv_embargo,
            verbose=verbose,
        )
    elif verbose:
        print(f"[VAR] using user-supplied lambda = {lambda_value:.6e}")
    del X_scaled

    alpha = float(lambda_value)

    n_targets = Y.shape[1]
    fit_jobs = _effective_cv_n_jobs(n_jobs, Q.shape[0])
    if verbose:
        print(f"[VAR] fitting {n_targets} target regressions "
              f"(n_jobs={fit_jobs}) ...")
    seed_seq = np.random.SeedSequence(0)
    seeds = seed_seq.spawn(n_targets)

    def _job(t):
        rng_t = np.random.default_rng(seeds[t])
        return _fit_target_gram_cd(
            Q=Q,
            q=XtY_T[t],
            y=Y64_T[t],
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            rng=rng_t,
            random_cd=random_cd,
        )

    coefs_scaled = Parallel(
        n_jobs=fit_jobs, backend="threading", pre_dispatch="2*n_jobs"
    )(delayed(_job)(t) for t in range(n_targets))

    A_scaled = np.vstack(coefs_scaled).astype(np.float32)
    A = A_scaled * inv_w[None, :]

    Y_pred = X_design @ A.T
    r2 = _safe_r2(Y, Y_pred)
    sparsity = float(np.mean(np.abs(A) < 1e-8))
    if verbose:
        print(f"[VAR] mean R² = {np.nanmean(r2):.4f}   "
              f"sparsity = {sparsity:.4f}")
    if sparsity > 0.999:
        warnings.warn(
            f"Sparsity is {sparsity:.4f} — almost every coefficient is zero. "
            "The model is over-shrunk. Try one or more of:\n"
            "  - lower lambda_min_ratio or increase n_lambda in CV,\n"
            "  - pass an explicit lower lambda_value,\n"
            "  - lower long_lag_penalty_power (e.g. 1.0).",
            UserWarning,
        )

    # Spectral radius of companion matrix (stability diagnostic)
    rho = float("nan")
    try:
        rho = _var_spectral_radius(A, n_features, max_lag)
    except Exception as e:
        warnings.warn(f"Spectral radius computation failed: {e}", RuntimeWarning)
    if verbose:
        if np.isfinite(rho):
            label = "(STABLE)" if rho < 1.0 else "(UNSTABLE — increase lambda)"
        else:
            label = f"(not computed — companion matrix {n_features*max_lag}×{n_features*max_lag} exceeds size limit)"
        print(f"[VAR] spectral radius = {rho if np.isfinite(rho) else 'n/a'}  {label}")
    if rho >= 1.0 and np.isfinite(rho):
        warnings.warn(
            f"VAR spectral radius = {rho:.4f} >= 1.  The model is unstable "
            f"(free-running predictions will diverge).  Consider increasing "
            f"lambda_value or reducing max_lag.",
            UserWarning,
        )

    nonzero = lasso_nonzero_edge_mask(A)

    return {
        "A": A,
        "max_lag": max_lag,
        "two_period_lags": two_period_lags,
        "penalty_weights": weights,
        "mean_r2": float(np.nanmean(r2)),
        "per_target_r2": r2,
        "spectral_radius": rho,
        "n_channels": n_channels,
        "n_bands": n_bands,
        "n_features": n_features,
        "band_names": list(band_names),
        "lambda_value": float(lambda_value),
        "lambda_grid": lambda_grid,
        "lambda_cv_mse": lambda_cv_mse,
        "long_lag_penalty_power": float(long_lag_penalty_power),
        "nonzero_edges": nonzero,
        "lasso_nonzero_edges": nonzero,
    }


def _build_var_companion_sparse_csr(
    A: np.ndarray, n_features: int, max_lag: int,
) -> _sp.csr_matrix:
    """
    Build the VAR(p) companion matrix in sparse CSR format.

    Used internally by ``_var_spectral_radius`` for memory-efficient
    eigenvalue estimation on large systems.

    Parameters
    ----------
    A : ndarray, shape ``(n_features, n_features * max_lag)``
        Transition matrix with lag blocks
        ``[lag1, lag2, ..., lag_max_lag]`` along columns.
    n_features, max_lag : int
        State dimension and VAR order.

    Returns
    -------
    C_csr : scipy.sparse.csr_matrix, shape ``(N, N)``
        Companion matrix with ``N = n_features * max_lag``.
    """
    N = n_features * max_lag
    C_sp = _sp.lil_matrix((N, N), dtype=np.float64)
    A64 = np.asarray(A, dtype=np.float64)
    for lag_i in range(max_lag):
        sl = slice(lag_i * n_features, (lag_i + 1) * n_features)
        C_sp[:n_features, sl] = A64[:, sl]
    if max_lag > 1:
        C_sp[n_features:, : n_features * (max_lag - 1)] = _sp.eye(
            n_features * (max_lag - 1), dtype=np.float64
        )
    return C_sp.tocsr()


def _var_spectral_radius(
    A: np.ndarray, n_features: int, max_lag: int,
) -> float:
    """
    Spectral radius (largest-modulus eigenvalue) of the VAR companion matrix.

    Computed with ARPACK ``eigs(k=1, which='LM')`` on the sparse companion
    form, so it scales to large ``n_features * max_lag``.

    Parameters
    ----------
    A : ndarray, shape ``(n_features, n_features * max_lag)``
    n_features, max_lag : int

    Returns
    -------
    float
        Spectral radius ρ.  The VAR is stable when ρ < 1.
    """
    N = n_features * max_lag
    C_csr = _build_var_companion_sparse_csr(A, n_features, max_lag)
    vals = _spla.eigs(
        C_csr, k=1, which="LM", return_eigenvectors=False,
        maxiter=10 * N, tol=1e-6,
    )
    return float(np.max(np.abs(vals)))


def decompose_r2_contributions(
    A: np.ndarray,
    X: np.ndarray,
    max_lag: int,
    n_channels: int,
    n_bands: int,
) -> Dict[str, float]:
    """
    Decompose full-model R² by connectivity type.

    Masks subsets of ``A`` and recomputes one-step prediction R² to
    quantify how much variance each connectivity class explains:

    - ``auto``          — same channel and band (any lag)
    - ``cross_freq``    — same channel, different band
    - ``cross_channel`` — different channel, same band
    - ``cross_both``    — different channel and band
    - ``full``          — unmasked full model

    Parameters
    ----------
    A : ndarray, shape ``(n_features, n_features * max_lag)``
        Fitted transition matrix.
    X : ndarray, shape ``(n_features, T)``
        Preprocessed feature matrix used to build the design matrix.
    max_lag : int
        VAR order.
    n_channels, n_bands : int
        Layout of features as ``n_channels * n_bands``.

    Returns
    -------
    dict
        Mean R² values keyed by ``auto``, ``cross_freq``, ``cross_channel``,
        ``cross_both``, and ``full``.
    """
    n_features = n_channels * n_bands
    Xd, Y = build_full_lag_design_matrix(X, max_lag=max_lag)

    feat = np.arange(n_features)
    ch, band = divmod(feat, n_bands)
    same_ch = ch[:, None] == ch[None, :]
    same_b = band[:, None] == band[None, :]

    A3 = A.reshape(n_features, max_lag, n_features)

    def _mean_r2(mask: np.ndarray) -> float:
        A_m = (A3 * mask[:, None, :].astype(A.dtype)).reshape(
            n_features, n_features * max_lag
        )
        return float(np.nanmean(_safe_r2(Y, Xd @ A_m.T)))

    return {
        "auto": _mean_r2(same_ch & same_b),
        "cross_freq": _mean_r2(same_ch & ~same_b),
        "cross_channel": _mean_r2(~same_ch & same_b),
        "cross_both": _mean_r2(~same_ch & ~same_b),
        "full": float(np.nanmean(_safe_r2(Y, Xd @ A.T))),
    }


def _collapse_lag_tensor(A3: np.ndarray, reduce: str = "sum_abs") -> np.ndarray:
    if reduce == "sum_abs":
        return np.abs(A3).sum(axis=1)
    if reduce == "max_abs":
        return np.abs(A3).max(axis=1)
    if reduce == "l2":
        return np.sqrt((A3 ** 2).sum(axis=1))
    if reduce == "signed_sum":
        return A3.sum(axis=1)
    raise ValueError(
        f"Unknown reduce='{reduce}'. Use 'sum_abs', 'max_abs', 'l2' "
        f"or 'signed_sum'."
    )


def aggregate_lagged_matrix(
    A: np.ndarray,
    n_features: Optional[int] = None,
    max_lag: Optional[int] = None,
    reduce: str = "sum_abs",
) -> np.ndarray:
    """
    Collapse the lag dimension of a VAR coefficient matrix.

    The fitted ``A`` has shape ``(n_features, n_features * max_lag)`` with
    column ordering ``[lag1_block, lag2_block, ...]``.  Returns a single
    ``(n_features, n_features)`` interaction matrix where entry ``[i, j]``
    summarises how much feature j drives feature i across all lags.
    """
    A = np.asarray(A, dtype=float)
    if n_features is None:
        n_features = A.shape[0]
    if max_lag is None:
        if A.shape[1] % n_features != 0:
            raise ValueError(
                f"A.shape[1]={A.shape[1]} is not a multiple of "
                f"n_features={n_features}; pass max_lag explicitly."
            )
        max_lag = A.shape[1] // n_features

    blocks = A.reshape(n_features, max_lag, n_features)
    return _collapse_lag_tensor(blocks, reduce=reduce)
