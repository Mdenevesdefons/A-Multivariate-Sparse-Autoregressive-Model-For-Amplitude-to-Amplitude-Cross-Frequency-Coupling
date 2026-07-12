"""
Example: multivariate sparse amplitude-amplitude cross-frequency coupling (AACFC).

Runs the full pipeline end to end:

    load / synthesise LFP -> downsample -> Morlet band power (+COI)
         -> preprocess amplitudes -> Lasso AACFC -> plots

Synthetic LFP mode
------------------
Coupling is planted in a **continuous-time** band-envelope model: each band is
an amplitude-modulated oscillator and delayed envelope coupling is applied in
that generative system.  The resulting voltage trace is sampled at 1 kHz as if
recorded, then passed through the same Morlet + preprocessing + VAR pipeline
used for real data.

Usage
-----
    python example_usage.py
    python example_usage.py --mat path/to/recording.mat --var ECOG
"""

from __future__ import annotations

import argparse
import os
import sys

if "--no-show" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
for _p in (_REPO_ROOT, _PKG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import multivariate_sparse_connectivity as msc
from multivariate_sparse_connectivity import plots


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

MORLET_Q = 6.0

SYNTH_DURATION_S = 800.0
SYNTH_TARGET_FS = 100.0
SYNTH_DOWNSAMPLE_FACTOR = 5
SYNTH_NOISE_STD = 1e-2
SYNTH_COUPLING_BETA = 1e3
SYNTH_ENVELOPE_TAU_S = 0.5

BAND_CARRIER_HZ = {
    "delta": 2.5,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 38.0,
}

# Ground-truth edges (continuous-time delays in seconds):
# (src_ch, src_band, tgt_ch, tgt_band, lead_s)
SYNTHETIC_EDGES = [
    (0, "theta", 1, "gamma", 0.30),
    (2, "alpha", 3, "beta", 0.50),
    (3, "theta", 1, "alpha", 0.20),
    (0, "delta", 5, "beta", 1.00)
]


def snap_edges_to_feature_grid(
    edges: list[tuple],
    fs_features: float,
) -> list[tuple[int, str, int, str, float]]:
    """Snap planted continuous-time delays to integer lags at ``fs_features``."""
    snapped = []
    for src_ch, src_b, tgt_ch, tgt_b, lead_s in edges:
        lag = max(1, int(round(float(lead_s) * fs_features)))
        snapped.append((src_ch, src_b, tgt_ch, tgt_b, lag / fs_features))
    return snapped


def _feat_index(ch: int, band: str, band_names: list[str], n_bands: int) -> int:
    return ch * n_bands + band_names.index(band)


def make_synthetic_lfp(
    n_channels: int = 6,
    duration_s: float = SYNTH_DURATION_S,
    fs: float = 1000.0,
    seed: int = 0,
    edges: list[tuple] = SYNTHETIC_EDGES,
    noise_std: float = SYNTH_NOISE_STD,
    coupling_beta: float = SYNTH_COUPLING_BETA,
    envelope_tau_s: float = SYNTH_ENVELOPE_TAU_S,
) -> np.ndarray:
    """
    Continuous-time band-amplitude coupling model, sampled at ``fs``.

    Each active band on each channel is a narrowband oscillator whose envelope
    follows a stable AR(1) process.  Planted edges add delayed
    source-band envelope drive into target-band envelopes.  The recorded LFP is
    the sum of amplitude-modulated carriers plus measurement noise.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    band_names = list(BAND_CARRIER_HZ.keys())
    band_idx = {b: i for i, b in enumerate(band_names)}
    n_bands = len(band_names)

    bands_per_ch: dict[int, set[str]] = {ch: set() for ch in range(n_channels)}
    for src_ch, src_b, tgt_ch, tgt_b, _ in edges:
        bands_per_ch[src_ch].add(src_b)
        bands_per_ch[tgt_ch].add(tgt_b)

    dt = 1.0 / fs
    rho = float(np.exp(-dt / envelope_tau_s))
    innov_scale = float(np.sqrt(max(1e-12, 1.0 - rho ** 2)))

    A = np.zeros((n_channels, n_bands, n), dtype=np.float64)
    for ch in range(n_channels):
        for band in bands_per_ch[ch]:
            b_i = band_idx[band]
            eps = rng.standard_normal(n)
            env = np.zeros(n, dtype=np.float64)
            for ti in range(1, n):
                env[ti] = rho * env[ti - 1] + innov_scale * eps[ti]
            A[ch, b_i] = 0.6 * env / (np.std(env) + 1e-9)

    for src_ch, src_b, tgt_ch, tgt_b, lead_s in edges:
        lag = max(1, int(round(float(lead_s) * fs)))
        sf = band_idx[src_b]
        tf = band_idx[tgt_b]
        A[tgt_ch, tf, lag:] += coupling_beta * A[src_ch, sf, :-lag]

    X = np.zeros((n_channels, n), dtype=np.float64)
    for ch in range(n_channels):
        for band in bands_per_ch[ch]:
            b_i = band_idx[band]
            f0 = BAND_CARRIER_HZ[band]
            phase = rng.uniform(0.0, 2.0 * np.pi)
            X[ch] += A[ch, b_i] * np.sin(2.0 * np.pi * f0 * t + phase)
        X[ch] += noise_std * rng.standard_normal(n)

    return X


def _edge_lag_profile(
    A: np.ndarray,
    source: tuple[int, str],
    target: tuple[int, str],
    band_names: list[str],
    n_bands: int,
) -> np.ndarray:
    n_features = A.shape[0]
    max_lag = A.shape[1] // n_features
    sf = _feat_index(source[0], source[1], band_names, n_bands)
    tf = _feat_index(target[0], target[1], band_names, n_bands)
    A3 = A.reshape(n_features, max_lag, n_features)
    return A3[tf, :, sf]


def report_synthetic_edge_recovery(
    A: np.ndarray,
    edges: list[tuple],
    band_names: list[str],
    n_bands: int,
    fs_features: float,
) -> list[dict]:
    """Print recovery table and return per-edge summaries."""
    n_features = A.shape[0]
    strength = msc.aggregate_lagged_matrix(A, reduce="max_abs")
    nonzero = msc.lasso_nonzero_edge_mask(A)
    rank_matrix = np.where(nonzero, strength, 0.0)

    summaries = []
    print("\nGround-truth AAC edge recovery (synthetic):")
    print(f"  {'edge':<28s}  {'exp lag':>8s}  {'peak lag':>8s}  "
          f"{'err (ms)':>8s}  {'|A|':>7s}  {'rank':>6s}")
    print("  " + "-" * 72)

    for src_ch, src_b, tgt_ch, tgt_b, lead_s in edges:
        prof = _edge_lag_profile(
            A, (src_ch, src_b), (tgt_ch, tgt_b), band_names, n_bands,
        )
        peak_lag = int(np.argmax(np.abs(prof))) + 1
        exp_lag = max(1, int(round(lead_s * fs_features)))
        lag_err_ms = abs(peak_lag - exp_lag) / fs_features * 1000.0
        sf = _feat_index(src_ch, src_b, band_names, n_bands)
        tf = _feat_index(tgt_ch, tgt_b, band_names, n_bands)
        edge_strength = float(strength[tf, sf])
        rank = int((rank_matrix >= edge_strength).sum()) if edge_strength > 0 else -1
        label = f"{src_b}(ch{src_ch})->{tgt_b}(ch{tgt_ch})"
        print(
            f"  {label:<28s}  {exp_lag:8d}  {peak_lag:8d}  "
            f"{lag_err_ms:8.1f}  {edge_strength:7.4f}  {rank:6d}"
        )
        summaries.append({
            "edge": label,
            "source": (src_ch, src_b),
            "target": (tgt_ch, tgt_b),
            "expected_lag": exp_lag,
            "peak_lag": peak_lag,
            "lag_error_ms": lag_err_ms,
            "strength": edge_strength,
            "rank": rank,
        })
    return summaries


def _planted_edge_keys(edges: list[tuple]) -> set[tuple[int, str, int, str]]:
    return {(e[0], e[1], e[2], e[3]) for e in edges}


def _null_control_edge(
    planted_edges: list[tuple],
    n_channels: int,
    band_names: list[str],
) -> tuple[tuple[int, str], tuple[int, str]]:
    """Cross-channel x cross-band pair that was not planted."""
    planted = _planted_edge_keys(planted_edges)
    for tgt_ch in range(n_channels):
        for tgt_b in band_names:
            for src_ch in range(n_channels):
                for src_b in band_names:
                    if src_ch == tgt_ch and src_b == tgt_b:
                        continue
                    if (src_ch, src_b, tgt_ch, tgt_b) in planted:
                        continue
                    if src_ch != tgt_ch and src_b != tgt_b:
                        return (src_ch, src_b), (tgt_ch, tgt_b)
    raise RuntimeError("No unplanted cross-channel x cross-band pair found.")


def _top_lasso_edge(
    A: np.ndarray,
    n_channels: int,
    n_bands: int,
    band_names: list[str],
    planted_edges: list[tuple] | None = None,
) -> tuple[tuple[int, str], tuple[int, str]]:
    """Strongest Lasso cross-channel x cross-band link."""
    planted = _planted_edge_keys(planted_edges or [])
    strength = msc.aggregate_lagged_matrix(A, reduce="max_abs")
    mask = msc.cross_both_edge_mask(msc.lasso_nonzero_edge_mask(A), n_channels, n_bands)
    strength[~mask] = 0.0
    n_features = n_channels * n_bands
    for flat in np.argsort(strength.ravel())[::-1]:
        if strength.ravel()[flat] <= 0:
            break
        tgt = int(flat // n_features)
        src = int(flat % n_features)
        key = (src // n_bands, band_names[src % n_bands],
               tgt // n_bands, band_names[tgt % n_bands])
        if key not in planted:
            return (key[0], key[1]), (key[2], key[3])
    flat = int(np.argmax(strength))
    src = int(flat % n_features)
    tgt = int(flat // n_features)
    return (src // n_bands, band_names[src % n_bands]), (tgt // n_bands, band_names[tgt % n_bands])


def _planted_plot_edges(
    recovery: list[dict],
) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    order = sorted(recovery, key=lambda r: (r["rank"] if r["rank"] > 0 else 999, -r["strength"]))
    return [(r["source"], r["target"]) for r in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", default=None, help="Path to a MATLAB .mat recording.")
    parser.add_argument("--var", default=None, help="Variable name inside the .mat file.")
    parser.add_argument("--orig-fs", type=float, default=1000.0, help="Raw sampling rate (Hz).")
    parser.add_argument("--target-fs", type=float, default=SYNTH_TARGET_FS,
                        help="Downsample target (Hz).")
    parser.add_argument("--downsample-factor", type=int, default=SYNTH_DOWNSAMPLE_FACTOR,
                        help="Envelope block-average factor after band power.")
    parser.add_argument("--duration", type=float, default=SYNTH_DURATION_S,
                        help="Synthetic recording length (s); ignored for --mat.")
    parser.add_argument("--lambda", dest="lam", type=float, default=None,
                        help="Fixed Lasso lambda; omit for purged blocked CV on real data. "
                             "Synthetic LFP demo defaults to 1e-3.")
    parser.add_argument("--no-show", action="store_true", help="Save figures without displaying.")
    args = parser.parse_args()

    if args.lam is None and args.mat is None:
        args.lam = 1e-3
        print(f"Using default lambda={args.lam:g} for synthetic LFP demo.")

    show = not args.no_show
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    def fig_path(name: str) -> str:
        return os.path.join(fig_dir, name)

    band_names = list(BANDS.keys())
    n_bands = len(band_names)
    fs_features_nominal = args.target_fs / args.downsample_factor
    planted_edges = snap_edges_to_feature_grid(SYNTHETIC_EDGES, fs_features_nominal)
    recovery: list[dict] = []

    if args.mat:
        X_raw, meta = msc.load_mat_ecog(args.mat, var_name=args.var)
        print(f"Loaded {args.mat}: {X_raw.shape} (channels, time)")
        n_channels = X_raw.shape[0]
        X_ds, fs = msc.downsample_raw_ecog(X_raw, orig_fs=args.orig_fs, target_fs=args.target_fs)
        print(f"Downsampled: {X_ds.shape}, fs = {fs} Hz")
        band_power, band_names, coi = msc.extract_band_power(
            X_ds, sampling_rate=fs, frequency_bands=BANDS,
            method="morlet", n_freqs_per_band=20, morlet_q=MORLET_Q,
            trim_edge_periods=2.0, return_coi_stats=True,
        )
        print(f"band_power: {band_power.shape}; COI removed {coi['n_samples_removed']} samples")
        X, bp_clean = msc.preprocess_band_power(
            band_power, downsample_factor=args.downsample_factor,
            detrend_flag=True, zscore_flag=True,
        )
        fs_features = fs / args.downsample_factor
    else:
        n_channels = 6
        X_raw = make_synthetic_lfp(duration_s=args.duration, fs=args.orig_fs, edges=SYNTHETIC_EDGES)
        print(f"Synthetic LFP (continuous-time envelopes)  X_raw.shape={X_raw.shape}  fs={args.orig_fs} Hz")
        print("Planted edges (continuous-time delays, snapped to feature grid):")
        for e in planted_edges:
            lag = int(round(e[4] * fs_features_nominal))
            print(f"  {e[1]}(ch{e[0]}) -> {e[3]}(ch{e[2]}) @ {e[4]:.3f}s ({lag} samples @ {fs_features_nominal} Hz)")
        X_ds, fs = msc.downsample_raw_ecog(X_raw, orig_fs=args.orig_fs, target_fs=args.target_fs)
        band_power, band_names, coi = msc.extract_band_power(
            X_ds, sampling_rate=fs, frequency_bands=BANDS,
            method="morlet", n_freqs_per_band=20, morlet_q=MORLET_Q,
            trim_edge_periods=2.0, return_coi_stats=True,
        )
        X, bp_clean = msc.preprocess_band_power(
            band_power, downsample_factor=args.downsample_factor,
            detrend_flag=True, zscore_flag=True,
        )
        fs_features = fs / args.downsample_factor
        print(f"After Morlet + preprocess: X.shape={X.shape}, fs_features={fs_features} Hz")

    print(f"Feature matrix X: {X.shape}, fs_features = {fs_features} Hz")

    if bp_clean is not None:
        plots.plot_band_power_before_after(
            band_power, bp_clean, band_names, fs_features, args.downsample_factor,
            n_channels, channels_to_show=list(range(min(2, n_channels))),
            t_range=(0, min(60, bp_clean.shape[-1] / fs_features)),
            frequency_bands=BANDS,
            save_path=fig_path("band_power_before_after.png"), show=show,
        )

    max_samples = 500
    two_period_lags = msc.compute_two_period_band_lags(
        BANDS, fs_features, morlet_q=MORLET_Q, max_samples=max_samples,
    )
    max_lag_preview = max(two_period_lags.values())
    plots.plot_soft_lag_weights(
        two_period_lags, band_names, max_lag_preview,
        frequency_bands=BANDS,
        fs_features=fs_features,
        save_path=fig_path("soft_lag_weights.png"), show=show,
    )

    cv_embargo = max(1, int(two_period_lags.get("delta", 1)) // 4)

    res = msc.fit_weighted_lasso_soft_delays(
        X, band_names=band_names, frequency_bands=BANDS, fs_features=fs_features,
        lambda_value=args.lam,
        morlet_q=MORLET_Q,
        max_samples=max_samples,
        max_lag_multiplier=1.0,
        cv_folds=5,
        n_lambda=30,
        lambda_min_ratio=1e-4,
        lambda_refine=True,
        cv_embargo=cv_embargo,
        n_jobs=-1,
        verbose=True,
    )
    A = res["A"]
    max_lag = res["max_lag"]
    nonzero = res["nonzero_edges"]
    sig_cross_both = msc.cross_both_edge_mask(nonzero, n_channels, n_bands)
    coef_sparsity = float(np.mean(np.abs(A) < 1e-8))
    edge_sparsity = 1.0 - float(nonzero.mean())

    print(f"\nSelected lambda : {res['lambda_value']:.4e}")
    print(f"max_lag         : {max_lag} samples ({max_lag / fs_features:.2f} s)")
    print(f"Mean R2         : {res['mean_r2']:.4f}")
    print(f"Coefficient sparsity in A : {coef_sparsity:.4f}")
    print(f"Edge sparsity (feature x feature) : {edge_sparsity:.4f}")
    print(f"Lasso-nonzero cross-channel x cross-band edges: {int(sig_cross_both.sum())}")
    if args.lam is None:
        print(f"CV purge gap    : {max_lag} samples")
        print(f"CV embargo      : {cv_embargo} samples")
    print(f"Spectral radius : {res['spectral_radius']:.4f}")

    if not args.mat:
        recovery = report_synthetic_edge_recovery(
            A, planted_edges, band_names, n_bands, fs_features,
        )

    decomp = msc.decompose_r2_contributions(A, X, max_lag, n_channels, n_bands)
    print("\nR2 decomposition:")
    for k, v in decomp.items():
        print(f"  {k:<14s}: {v:.4f}")

    plots.plot_transition_matrix(
        A, band_names=band_names, n_channels=n_channels,
        title="AACFC matrix A",
        save_path=fig_path("transition_matrix.png"), show=show,
    )
    plots.plot_predicted_vs_actual(
        X, A, max_lag, band_names, fs_features, n_channels,
        frequency_bands=BANDS,
        channels_to_show=list(range(min(2, n_channels))),
        t_range=(0, min(30, X.shape[1] / fs_features)),
        save_path=fig_path("predicted_vs_actual.png"), show=show,
    )

    if recovery:
        best_planted = _planted_plot_edges(recovery)[0]
        s_ch, s_b = best_planted[0]
        t_ch, t_b = best_planted[1]
        best = recovery[0] if recovery[0]["source"] == best_planted[0] else recovery[
            min(range(len(recovery)), key=lambda i: recovery[i]["rank"] if recovery[i]["rank"] > 0 else 999)
        ]
        plots.plot_directed_edge_recovery(
            A, band_names=band_names, n_channels=n_channels,
            fs_features=fs_features, source=(s_ch, s_b), target=(t_ch, t_b),
            title=(f"Planted: {s_b}(ch{s_ch}) -> {t_b}(ch{t_ch})  "
                   f"[rank {best['rank']}, |A|={best['strength']:.3f}]"),
            save_path=fig_path("edge_planted.png"), show=show,
        )

        null_src, null_tgt = _null_control_edge(planted_edges, n_channels, band_names)
        ns, nb = null_src
        nt, tb = null_tgt
        null_prof = _edge_lag_profile(A, null_src, null_tgt, band_names, n_bands)
        null_peak = float(np.max(np.abs(null_prof)))
        plots.plot_directed_edge_recovery(
            A, band_names=band_names, n_channels=n_channels,
            fs_features=fs_features, source=null_src, target=null_tgt,
            title=(f"Control (not planted): {nb}(ch{ns}) -> {tb}(ch{nt})  "
                   f"[peak |A|={null_peak:.4f}]"),
            save_path=fig_path("edge_null_control.png"), show=show,
        )

        lasso_src, lasso_tgt = _top_lasso_edge(A, n_channels, n_bands, band_names, planted_edges)
        ls, lb = lasso_src
        lt, ltb = lasso_tgt
        plots.plot_directed_edge_recovery(
            A, band_names=band_names, n_channels=n_channels,
            fs_features=fs_features, source=lasso_src, target=lasso_tgt,
            title=(f"Top Lasso edge: {lb}(ch{ls}) -> {ltb}(ch{lt})"),
            save_path=fig_path("edge_top_lasso.png"), show=show,
        )

        plot_edges = _planted_plot_edges(recovery)[:2]
        plot_edges.append((null_src, null_tgt))
        plots.plot_edge_lag_profiles(
            A, band_names=band_names, n_channels=n_channels, fs_features=fs_features,
            edges=plot_edges,
            title="Lag profiles: planted vs control",
            two_period_lags=res["two_period_lags"],
            save_path=fig_path("edge_lag_profiles.png"), show=show,
        )

    plots.plot_frequency_frequency_interactions(
        A, n_channels, band_names,
        save_path=fig_path("frequency_frequency.png"), show=show,
    )
    plots.plot_channel_channel_interactions(
        A, n_channels, n_bands,
        save_path=fig_path("channel_channel.png"), show=show,
    )
    plots.plot_frequency_resolved_graph(
        A, n_channels=n_channels, n_bands=n_bands, band_names=band_names,
        edge_mask=nonzero, max_edges=10, label_top_k=10,
        title="Lasso-nonzero links with band labels",
        save_path=fig_path("connectivity_graph_freq.png"), show=show,
    )

    print(f"\nDone. Figures written to: {fig_dir}")


if __name__ == "__main__":
    main()
