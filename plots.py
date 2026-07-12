"""Visualisation helpers for the multivariate sparse AACFC pipeline."""

from __future__ import annotations
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any, Sequence
import os
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Circle

from .sysid import build_full_lag_design_matrix, cross_both_edge_mask


# Palette, colormaps, and journal styling

PALETTE: Dict[str, str] = {
    "blue": "#93B5CF",
    "pink": "#E8B4B8",
    "beige": "#E8DFD0",
    "lilac": "#C4B0D8",
    "terra": "#D4927A",
    "rose": "#C87888",
    "slate": "#A8A098",
    "cream": "#F7F3ED",
}


def _register_cmap(name: str, colors: List[str]) -> LinearSegmentedColormap:
    """Build and register a named matplotlib colormap."""
    cmap = LinearSegmentedColormap.from_list(name, colors)
    try:
        mpl.colormaps.register(cmap, name=name)
    except ValueError:
        mpl.colormaps.register(cmap, name=name, force=True)
    return cmap


CMAP_SEQ = _register_cmap("poster_seq", [
    PALETTE["cream"], PALETTE["beige"], PALETTE["pink"],
    PALETTE["lilac"], PALETTE["blue"], "#7A9CB8",
])
CMAP_DIV = _register_cmap("poster_div", [
    PALETTE["blue"], "#C8D8E8", PALETTE["beige"],
    "#E8C8B8", PALETTE["terra"],
])

CYCLE = [
    PALETTE["blue"], PALETTE["terra"], PALETTE["rose"],
    PALETTE["lilac"], PALETTE["pink"],
]
GRID_ALPHA = 0.35
EDGE_COLOR = "#6A6258"
PAPER_SAVE_PAD_INCHES = 0.02

PUB_RCPARAMS: Dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
    "font.size": 10,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "mathtext.fontset": "stix",
}
mpl.rcParams.update(PUB_RCPARAMS)

STANDARD_FIGSIZE: Tuple[float, float] = (10.0, 6.0)
STANDARD_FIGSIZE_SQUARE: Tuple[float, float] = (8.0, 8.0)


def _tick_fontsize(n_labels: int, max_size: int = 9, min_size: int = 5) -> int:
    """Scale tick-label font size down when many labels would overlap."""
    if n_labels <= 1:
        return max_size
    return max(min_size, min(max_size, int(520 // n_labels)))


def _band_axis_labels(
    band_names: Sequence[str],
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[str]:
    """Band names with optional Hz ranges for axis labels."""
    labels = []
    for b in band_names:
        if frequency_bands and b in frequency_bands:
            lo, hi = frequency_bands[b]
            labels.append(f"{b}\n({lo:g}-{hi:g} Hz)")
        else:
            labels.append(str(b))
    return labels


# Shared figure helpers

def _despine(ax, grid: bool = False):
    """Hide top/right spines; optional light y-grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=GRID_ALPHA, color=PALETTE["slate"])


def save_figure(fig, save_path: str, dpi: int = 300, **extra_kwargs) -> Optional[str]:
    """
    Write a figure to disk (PNG raster or PDF/EPS/SVG vector).

    Returns the path written, or ``None`` if saving failed (e.g. the target
    PDF is open in a viewer on Windows).
    """
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ext = os.path.splitext(save_path)[1].lower().lstrip(".")
    kwargs: Dict[str, Any] = {
        "bbox_inches": "tight",
        "pad_inches": PAPER_SAVE_PAD_INCHES,
        "facecolor": "white",
        "edgecolor": "none",
    }
    kwargs.update(extra_kwargs)
    if ext in ("pdf", "eps", "svg", "png"):
        kwargs["format"] = ext
    kwargs["dpi"] = dpi

    try:
        fig.savefig(save_path, **kwargs)
        return save_path
    except (PermissionError, OSError) as err:
        if not isinstance(err, PermissionError) and getattr(err, "errno", None) != 13:
            raise
        stem, suffix = os.path.splitext(save_path)
        alt = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
        try:
            fig.savefig(alt, **kwargs)
            warnings.warn(
                f"Could not overwrite locked figure:\n  {save_path}\n"
                f"Saved instead to:\n  {alt}",
                UserWarning,
            )
            return alt
        except (PermissionError, OSError):
            warnings.warn(f"Could not save figure (file locked):\n  {save_path}", UserWarning)
            return None


def _finalize_fig(fig, save_path: Optional[str] = None, show: bool = True,
                  dpi: int = 300, tight_layout: bool = True):
    """Shared finalizer: tight layout, optional save, optional display."""
    if tight_layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if save_path is not None:
        save_figure(fig, save_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _time_window_mask(
    t_axis: np.ndarray, t_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean mask and visible time coordinates for an optional zoom range."""
    if t_range is None:
        mask = np.ones(t_axis.shape[0], dtype=bool)
    else:
        mask = (t_axis >= t_range[0]) & (t_axis <= t_range[1])
    return mask, t_axis[mask]


def _make_feature_labels(
    n_channels: int, band_names: Sequence[str],
    channel_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """``channel-band`` label per VAR feature, in channel-major feature order."""
    if channel_labels is not None and len(channel_labels) >= n_channels:
        ch = [str(channel_labels[i]) for i in range(n_channels)]
    else:
        ch = [str(i + 1) for i in range(n_channels)]
    return [f"{ch[c]}-{b}" for c in range(n_channels) for b in band_names]


def plot_band_power_before_after(
    band_power: np.ndarray,
    band_power_processed: np.ndarray,
    band_names: List[str],
    fs_features: float,
    downsample_factor: int,
    n_channels: int,
    channels_to_show: Optional[List[int]] = None,
    t_range: Optional[Tuple[float, float]] = None,
    epsilon: float = 1e-10,
    figsize: Optional[Tuple[int, int]] = None,
    band_colors: Optional[List[str]] = None,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Side-by-side line plots of band-averaged power envelopes.

    Left: log10 after envelope decimation.  Right: fully preprocessed
    (detrend + z-score).
    """
    n_bands = len(band_names)
    if band_power_processed.shape[:2] != band_power.shape[:2]:
        raise ValueError(
            "band_power and band_power_processed must share (channels, n_bands); "
            f"got {band_power.shape[:2]} vs {band_power_processed.shape[:2]}"
        )

    bp = np.asarray(band_power, dtype=np.float32)
    if downsample_factor > 1:
        n_ch, n_b, n_t = bp.shape
        n_out = n_t // downsample_factor
        bp = bp[:, :, : n_out * downsample_factor]
        bp = bp.reshape(n_ch, n_b, n_out, downsample_factor).mean(axis=-1)
    bp_log = np.log10(bp + epsilon)

    proc = np.asarray(band_power_processed, dtype=np.float32)
    T = proc.shape[-1]
    t_axis = np.arange(T) / fs_features
    t_mask, t_vis = _time_window_mask(t_axis, t_range)

    if channels_to_show is None:
        channels_to_show = list(range(n_channels))
    n_show = len(channels_to_show)
    if figsize is None:
        figsize = STANDARD_FIGSIZE
    if band_colors is None:
        band_colors = [CYCLE[i % len(CYCLE)] for i in range(n_bands)]

    fig, axes = plt.subplots(n_show, 2, figsize=figsize, sharex=True)
    if n_show == 1:
        axes = np.array([axes])

    for row, ch in enumerate(channels_to_show):
        panels = (
            (bp_log[ch], "log10 power", "Extracted (mean in-band power)"),
            (proc[ch], "z-score", "Preprocessed (detrend + z-score)"),
        )
        for col, (data, ylab, _subtitle) in enumerate(panels):
            ax = axes[row, col]
            for b in range(n_bands):
                ax.plot(t_vis, data[b, t_mask], color=band_colors[b],
                        label=band_names[b], linewidth=0.9, alpha=0.9)
            ax.set_ylabel(f"Ch {ch}\n{ylab}" if col == 0 else ylab, fontsize=9)
            ax.grid(True, alpha=0.25)
            if row == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.85,
                          ncol=1 if n_bands <= 6 else 2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Band-power envelopes before and after preprocessing", fontsize=11, y=1.02)
    return _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=True)


# Model plots

def plot_soft_lag_weights(
    two_period_lags: Dict[str, int],
    band_names: List[str],
    max_lag: int,
    frequency_bands: Dict[str, Tuple[float, float]],
    long_lag_penalty_power: float = 1.5,
    fs_features: Optional[float] = None,
    figsize: Tuple[float, float] = STANDARD_FIGSIZE,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Preview of the soft band-dependent lag penalty weights used by the VAR.

    All lags ``1 .. max_lag`` stay in the design matrix; inside each band's
    two-period trusted window the Lasso column weight is 1.0 and beyond it
    the weight ramps up, discouraging (but not forbidding) long delays.
    See ``compute_soft_delay_penalty_weights`` for the exact formula.
    """
    from .sysid import compute_soft_delay_penalty_weights

    n_bands = len(band_names)
    caps = np.array([int(two_period_lags.get(b, 1)) for b in band_names])

    # One channel suffices: weights repeat identically across channels.
    weights_flat = compute_soft_delay_penalty_weights(
        n_channels=1,
        band_names=band_names,
        two_period_lags=two_period_lags,
        frequency_bands=frequency_bands,
        max_lag=max_lag,
        long_lag_penalty_power=long_lag_penalty_power,
        dtype=np.float64,
    )
    W = weights_flat.reshape(max_lag, n_bands)

    # Per-band normalisation of excess penalty (w - 1) so slow bands show a
    # visible colour ramp even when their trusted window is long.
    W_excess = np.maximum(W - 1.0, 0.0)
    W_disp = np.zeros_like(W_excess)
    for b in range(n_bands):
        peak = float(W_excess[:, b].max())
        if peak > 0:
            W_disp[:, b] = W_excess[:, b] / peak

    if fs_features is not None and fs_features > 0:
        y_extent = (1.0 / fs_features, max_lag / fs_features)
        y_label = "Lag (s)"
        cap_y = caps / fs_features
    else:
        y_extent = (0.5, max_lag + 0.5)
        y_label = "Lag (samples)"
        cap_y = caps.astype(float)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        W_disp, aspect="auto", origin="lower", cmap="poster_seq",
        interpolation="nearest", vmin=0.0, vmax=1.0,
        extent=[-0.5, n_bands - 0.5, y_extent[0], y_extent[1]], rasterized=True,
    )
    for b in range(n_bands):
        ax.plot(
            [b - 0.42, b + 0.42], [cap_y[b], cap_y[b]], color=EDGE_COLOR,
            linewidth=1.6, solid_capstyle="butt", zorder=3,
        )
        ax.text(
            b, y_extent[0] + 0.02 * (y_extent[1] - y_extent[0]),
            f"w=1\n<{caps[b]}", ha="center", va="bottom", fontsize=7, color=EDGE_COLOR,
        )
    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(band_names)
    ax.set_xlabel("Frequency band")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Soft lag penalty weights  (p={long_lag_penalty_power}; "
        f"per-band normalised excess above w=1)",
        fontsize=12,
    )
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Normalised excess penalty  (0 = w=1, 1 = max for band)")
    _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=False)
    return fig, W


def plot_transition_matrix(
    A: np.ndarray,
    title: str = "AACFC coefficient matrix A  (rows = target, cols = lagged source)",
    pct_clip: float = 99.0,
    cmap: str = "poster_div",
    figsize: Tuple[int, int] = STANDARD_FIGSIZE,
    band_names: Optional[List[str]] = None,
    n_channels: Optional[int] = None,
    channel_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """
    Plot the fitted AACFC coefficient matrix ``A``.

    Rows are target features (channel–band amplitudes at time *t*); columns are
    lagged source features at *t − 1 … t − max_lag*.  Nonzero entries are
    Lasso-selected coupling coefficients.

    ``pct_clip`` sets the symmetric colour limit to this percentile of ``|A|``
    so a few large coefficients do not wash out the rest.  Passing
    ``band_names`` and ``n_channels`` draws channel/lag separators and labels.
    """
    abs_vals = np.abs(A[np.abs(A) > 1e-12])
    if abs_vals.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.percentile(abs_vals, pct_clip)) or float(np.max(np.abs(A))) or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label=f"coefficient  (clip={pct_clip}th pct)")
    ax.set_title(title)

    if band_names is not None and n_channels is not None:
        n_bands = len(band_names)
        n_feat = n_channels * n_bands
        max_lag = A.shape[1] // n_feat
        for ch in range(1, n_channels):
            ax.axhline(ch * n_bands - 0.5, color="k", lw=0.4, alpha=0.4)
        for lag in range(1, max_lag):
            ax.axvline(lag * n_feat - 0.5, color="gray", lw=0.8, alpha=0.5)
        for lag in range(max_lag):
            for ch in range(1, n_channels):
                ax.axvline(lag * n_feat + ch * n_bands - 0.5, color="k", lw=0.4, alpha=0.4)

        feat_labels = _make_feature_labels(n_channels, band_names, channel_labels)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(feat_labels, fontsize=_tick_fontsize(n_feat))

        lag_centers = [lag * n_feat + (n_feat - 1) / 2 for lag in range(max_lag)]
        max_lag_ticks = 20
        lag_step = max(1, int(np.ceil(max_lag / max_lag_ticks)))
        tick_lags = list(range(0, max_lag, lag_step))
        ax.set_xticks([lag_centers[lag] for lag in tick_lags])
        ax.set_xticklabels([f"Lag {lag + 1}" for lag in tick_lags],
                           rotation=0, ha="center",
                           fontsize=_tick_fontsize(len(tick_lags), max_size=9))
    ax.set_xlabel("Source (lagged)")
    ax.set_ylabel("Target")
    return _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi)


def plot_predicted_vs_actual(
    X: np.ndarray,
    A: np.ndarray,
    max_lag: int,
    band_names: List[str],
    fs_features: float,
    n_channels: int,
    channels_to_show: Optional[List[int]] = None,
    t_range: Optional[Tuple[float, float]] = None,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    cmap: str = "poster_seq",
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
):
    """Heatmaps of observed, predicted, and prediction error band-power features."""
    n_bands = len(band_names)
    band_labels = _band_axis_labels(band_names, frequency_bands)
    Xd, Y = build_full_lag_design_matrix(X, max_lag=max_lag)
    Y_pred = Xd @ A.T

    T_pred = Y.shape[0]
    t_axis = (np.arange(T_pred) + max_lag) / fs_features
    t_mask, _ = _time_window_mask(t_axis, t_range)

    if channels_to_show is None:
        channels_to_show = list(range(n_channels))
    n_show = len(channels_to_show)
    if figsize is None:
        figsize = (12.0, 6.0)

    fig, axes = plt.subplots(n_show, 3, figsize=figsize, sharex=True, sharey="row")
    if n_show == 1:
        axes = axes[None, :]

    t_vis = t_axis[t_mask]
    ext = [t_vis[0], t_vis[-1], -0.5, n_bands - 0.5]
    ytick_fs = _tick_fontsize(n_bands, max_size=8, min_size=6)

    for ax_idx, ch in enumerate(channels_to_show):
        sl = slice(ch * n_bands, (ch + 1) * n_bands)
        actual = Y[t_mask, sl].T
        predicted = Y_pred[t_mask, sl].T
        error = predicted - actual
        vmin = min(actual.min(), predicted.min())
        vmax = max(actual.max(), predicted.max())
        axes[ax_idx, 0].imshow(actual, aspect="auto", cmap=cmap, origin="lower",
                               extent=ext, vmin=vmin, vmax=vmax)
        im1 = axes[ax_idx, 1].imshow(predicted, aspect="auto", cmap=cmap, origin="lower",
                                     extent=ext, vmin=vmin, vmax=vmax)
        err_vals = error[np.isfinite(error)]
        err_vmax = float(np.percentile(np.abs(err_vals), 99)) if err_vals.size else 1.0
        err_vmax = max(err_vmax, 1e-6)
        im2 = axes[ax_idx, 2].imshow(
            error, aspect="auto", cmap="poster_div", origin="lower",
            extent=ext, vmin=-err_vmax, vmax=err_vmax,
        )
        axes[ax_idx, 0].set_ylabel(f"Ch {ch}", fontsize=9)
        for col in range(3):
            axes[ax_idx, col].set_yticks(range(n_bands))
            if col == 0:
                axes[ax_idx, col].set_yticklabels(band_labels, fontsize=ytick_fs)
            else:
                axes[ax_idx, col].set_yticklabels([])
        plt.colorbar(im1, ax=axes[ax_idx, 1], fraction=0.03, pad=0.02)
        plt.colorbar(im2, ax=axes[ax_idx, 2], fraction=0.03, pad=0.02)

    axes[0, 0].set_title("Observed")
    axes[0, 1].set_title("Predicted")
    axes[0, 2].set_title("Error (pred - actual)")
    for col in range(3):
        axes[-1, col].set_xlabel("Time (s)")
    fig.suptitle(
        "Band-resolved one-step forecast  (rows = frequency bands)",
        fontsize=10, y=1.02,
    )
    return _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=True)


# Single directed edge (ground-truth recovery)

def plot_directed_edge_recovery(
    A: np.ndarray,
    band_names: Sequence[str],
    n_channels: int,
    fs_features: float,
    source: Tuple[int, str],
    target: Tuple[int, str],
    top_n_sources: int = 12,
    channel_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = STANDARD_FIGSIZE,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Show that the VAR recovered one specific directed edge, e.g.
    ``theta(ch0) -> gamma(ch1)``.

    Left panel: the fitted lag profile ``A[target, lag, source]`` (the learned
    impulse response of the edge) as a stem plot versus lag in milliseconds.
    A real edge shows nonzero weight at the coupling delay; a spurious one is
    flat at zero.

    Right panel: ranked incoming influence ``sum_lags |A[target, ., source_j]|``
    for every source feature predicting ``target``.  The queried ``source`` is
    highlighted; if the model captured the edge it sits at (or near) the top.

    Parameters
    ----------
    source, target : ``(channel_index, band_name)``
        The directed edge to inspect, ``source -> target``.
    """
    band_names = list(band_names)
    n_bands = len(band_names)
    n_features = n_channels * n_bands
    max_lag = A.shape[1] // n_features

    def _feat(ch: int, band: str) -> int:
        if band not in band_names:
            raise ValueError(f"Unknown band {band!r}; choices: {band_names}")
        if not (0 <= ch < n_channels):
            raise ValueError(f"Channel {ch} out of range [0, {n_channels}).")
        return ch * n_bands + band_names.index(band)

    src_ch, src_band = source
    tgt_ch, tgt_band = target
    src_feat = _feat(src_ch, src_band)
    tgt_feat = _feat(tgt_ch, tgt_band)

    feat_labels = _make_feature_labels(n_channels, band_names, channel_labels)
    src_label = feat_labels[src_feat]
    tgt_label = feat_labels[tgt_feat]

    A3 = A.reshape(n_features, max_lag, n_features)  # [target, lag, source]
    edge_profile = A3[tgt_feat, :, src_feat]                 # (max_lag,)
    incoming = np.abs(A3[tgt_feat]).sum(axis=0)              # (n_features,) sum over lags

    lags = np.arange(1, max_lag + 1)
    lag_ms = lags / fs_features * 1000.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    markerline, stemlines, baseline = ax0.stem(lag_ms, edge_profile)
    plt.setp(stemlines, color=PALETTE["blue"], linewidth=1.6)
    plt.setp(markerline, color=PALETTE["terra"], markersize=5)
    plt.setp(baseline, color=EDGE_COLOR, linewidth=0.8)
    peak = int(np.argmax(np.abs(edge_profile)))
    if np.abs(edge_profile[peak]) > 0:
        ax0.annotate(
            f"peak {edge_profile[peak]:+.3f}\nat {lag_ms[peak]:.0f} ms",
            xy=(lag_ms[peak], edge_profile[peak]),
            xytext=(0.62, 0.9 if edge_profile[peak] >= 0 else 0.1),
            textcoords="axes fraction", fontsize=8, color=EDGE_COLOR,
            ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=EDGE_COLOR, lw=0.9),
        )
    ax0.set_xlabel("Lag (ms)")
    ax0.set_ylabel("VAR coefficient  $A$")
    ax0.set_title(f"Lag profile  {src_label} → {tgt_label}\n(stem height = fitted A coefficient)", fontsize=11)
    _despine(ax0, grid=True)

    order = np.argsort(incoming)[::-1]
    keep = order[:top_n_sources]
    y = np.arange(len(keep))[::-1]
    colors = [PALETTE["terra"] if j == src_feat else PALETTE["blue"] for j in keep]
    ax1.barh(y, incoming[keep], color=colors, edgecolor=EDGE_COLOR, linewidth=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels([feat_labels[j] for j in keep], fontsize=_tick_fontsize(len(keep), max_size=7))
    ax1.set_xlabel(r"Incoming influence  $\sum_{\mathrm{lags}} |A|$")
    ax1.set_title(f"All sources predicting {tgt_label}\n(highlight = queried source)", fontsize=11)
    src_rank = int(np.where(order == src_feat)[0][0]) + 1
    ax1.text(
        0.97, 0.04,
        f"{src_label} ranked #{src_rank} / {n_features}",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
        color=EDGE_COLOR,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["cream"],
                  edgecolor=PALETTE["terra"], alpha=0.9),
    )
    _despine(ax1, grid=False)
    ax1.grid(axis="x", alpha=GRID_ALPHA, color=PALETTE["slate"])

    if title:
        fig.suptitle(title, fontsize=13)

    _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=False)
    return {
        "edge_profile": edge_profile,
        "lag_ms": lag_ms,
        "incoming_influence": incoming,
        "source_feature": src_feat,
        "target_feature": tgt_feat,
        "source_rank": src_rank,
        "n_features": n_features,
    }


# Collapsed connectivity summaries

def _aggregate_interaction_matrix(
    A: np.ndarray, n_features: int, max_lag: int,
    row_index: np.ndarray, col_index: np.ndarray,
    n_rows: int, n_cols: int, exclude_self_features: bool = False,
) -> np.ndarray:
    """Sum ``|A|`` over lags into ``M[row_index[i], col_index[j]]``."""
    M = np.zeros((n_rows, n_cols), dtype=np.float64)
    ii, jj = np.indices((n_features, n_features))
    row_idx = row_index[ii]
    col_idx = col_index[jj]
    if exclude_self_features:
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
    for lag in range(max_lag):
        block = np.abs(A[:, lag * n_features:(lag + 1) * n_features])
        np.add.at(M, (row_idx, col_idx), block[ii, jj])
    return M


def _annotate_heatmap(ax, M: np.ndarray, fmt: str = ".2f", fontsize: int = 8):
    """Write numeric annotations on a heatmap."""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=fontsize, color="black")


def _plot_collapsed_interaction_panels(
    M: np.ndarray, row_labels: List[str], col_labels: List[str], title: str,
    row_axis: str, col_axis: str, cbar_label: str = "Sum |A| (all lags)",
    cmap: str = "poster_seq", save_path: Optional[str] = None,
    show: bool = True, dpi: int = 300,
) -> Dict[str, Any]:
    """Two-panel figure: full aggregated influence + cross-only (diagonal masked)."""
    M = np.asarray(M, dtype=float)
    M_cross = M.copy()
    if M_cross.shape[0] == M_cross.shape[1]:
        np.fill_diagonal(M_cross, np.nan)

    n = len(row_labels)
    fig = plt.figure(figsize=STANDARD_FIGSIZE)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.06, 1, 0.06], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2], sharey=ax0)
    cax1 = fig.add_subplot(gs[0, 3])

    x_rot = 0 if n <= 6 else 45
    x_ha = "center" if x_rot == 0 else "right"

    im0 = ax0.imshow(M, cmap=cmap, origin="lower", aspect="auto")
    ax0.set_xlabel(col_axis)
    ax0.set_ylabel(row_axis)
    ax0.set_xticks(range(len(col_labels)))
    ax0.set_yticks(range(len(row_labels)))
    ax0.set_xticklabels(col_labels, rotation=x_rot, ha=x_ha,
                        fontsize=_tick_fontsize(len(col_labels), max_size=8))
    ax0.set_yticklabels(row_labels, fontsize=_tick_fontsize(len(row_labels), max_size=8))
    ax0.set_title("All source→target terms (incl. same band/channel)")
    _annotate_heatmap(ax0, M, fmt=".1f", fontsize=7)
    fig.colorbar(im0, cax=cax0, label=None)

    cross_pos = M_cross[np.isfinite(M_cross) & (M_cross > 0)]
    if cross_pos.size:
        vmin = max(float(np.percentile(cross_pos, 5)), 1e-12)
        vmax = float(np.nanmax(M_cross))
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    else:
        norm = None
    im1 = ax1.imshow(M_cross, cmap=cmap, origin="lower", aspect="auto", norm=norm)
    ax1.set_xlabel(col_axis)
    ax1.set_xticks(range(len(col_labels)))
    ax1.set_xticklabels(col_labels, rotation=x_rot, ha=x_ha)
    ax1.tick_params(axis="y", left=False, labelleft=False)
    ax1.set_title("Cross-only (same band/channel masked out)")
    _annotate_heatmap(ax1, M_cross, fmt=".2f", fontsize=7)
    fig.colorbar(im1, cax=cax1, label=cbar_label)

    _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=False)
    return {"full": M, "cross_only": M_cross}


def plot_frequency_frequency_interactions(
    A: np.ndarray,
    n_channels: int,
    band_names: List[str],
    title: str = "Frequency–frequency coupling  (sum |A| over channels & lags)",
    cmap: str = "poster_seq",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Aggregate ``sum_lags |A|`` across channels into a band x band matrix.

    ``M[target_band, source_band]`` measures how much each source band helps
    predict each target band.
    """
    n_bands = len(band_names)
    n_features = n_channels * n_bands
    max_lag = A.shape[1] // n_features
    feat = np.arange(n_features)
    M = _aggregate_interaction_matrix(
        A, n_features, max_lag, feat % n_bands, feat % n_bands, n_bands, n_bands,
    )
    return _plot_collapsed_interaction_panels(
        M, row_labels=list(band_names), col_labels=list(band_names), title=title,
        row_axis="Target band", col_axis="Source band", cbar_label="Sum |A|",
        cmap=cmap, save_path=save_path, show=show, dpi=dpi,
    )


def plot_channel_channel_interactions(
    A: np.ndarray,
    n_channels: int,
    n_bands: int,
    channel_labels: Optional[List[str]] = None,
    title: str = "Channel–channel coupling  (sum |A| over bands & lags)",
    cmap: str = "poster_seq",
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Aggregate ``sum_lags |A|`` across bands into a channel x channel matrix.

    ``M[target_ch, source_ch]`` measures predictive influence between channels.
    """
    n_features = n_channels * n_bands
    max_lag = A.shape[1] // n_features
    feat = np.arange(n_features)
    M = _aggregate_interaction_matrix(
        A, n_features, max_lag, feat // n_bands, feat // n_bands, n_channels, n_channels,
    )
    if channel_labels is None:
        channel_labels = [f"Ch{i:02d}" for i in range(n_channels)]
    return _plot_collapsed_interaction_panels(
        M, row_labels=list(channel_labels), col_labels=list(channel_labels), title=title,
        row_axis="Target channel", col_axis="Source channel", cbar_label="Sum |A|",
        cmap=cmap, save_path=save_path, show=show, dpi=dpi,
    )


def _circular_layout(n: int) -> np.ndarray:
    """``(n, 2)`` node coordinates on a unit circle, first node at the top."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2.0
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _resolve_node_positions(
    channel_positions: Optional[np.ndarray], n_channels: int,
) -> np.ndarray:
    """Validate supplied ``(n_channels, 2)`` coords, else circular layout."""
    if channel_positions is None:
        return _circular_layout(n_channels)
    pos = np.asarray(channel_positions, dtype=float)
    if pos.shape != (n_channels, 2):
        raise ValueError(f"channel_positions must be ({n_channels}, 2), got {pos.shape}.")
    return pos


def _draw_graph_nodes(
    ax, pos: np.ndarray, labels: Sequence[str], node_r: float, node_color: str,
) -> None:
    """Draw uniform-size channel nodes with centred labels."""
    for i in range(len(pos)):
        ax.add_patch(Circle(pos[i], node_r, facecolor=node_color, edgecolor=EDGE_COLOR,
                            lw=1.0, zorder=3))
        ax.text(pos[i, 0], pos[i, 1], str(labels[i]), ha="center", va="center",
                fontsize=8, zorder=4, color=EDGE_COLOR, fontweight="bold")


def _edge_shrink_points(ax, node_r: float, extra_points: float = 2.5) -> float:
    """
    Convert a node radius in data units to the ``shrinkA/shrinkB`` value (in
    points) that makes an arrow stop just outside the node circle.

    Requires the axes limits and aspect to be set first; forces a draw so the
    data->display transform is realised.
    """
    fig = ax.figure
    fig.canvas.draw()
    trans = ax.transData
    x0, y0 = trans.transform((0.0, 0.0))
    x1, y1 = trans.transform((float(node_r), 0.0))
    px = float(np.hypot(x1 - x0, y1 - y0))
    return px * 72.0 / float(fig.dpi) + extra_points


def _band_color_map(band_names: Sequence[str]) -> Dict[str, str]:
    """Stable categorical colour per frequency band."""
    return {b: CYCLE[i % len(CYCLE)] for i, b in enumerate(band_names)}


def _arc_point(p0: np.ndarray, p1: np.ndarray, rad: float, t: float = 0.5) -> np.ndarray:
    """
    Point at parameter ``t`` on matplotlib's ``arc3`` quadratic Bezier.

    ``arc3`` places the control point at ``M + rad*(dy, -dx)`` (``M`` the chord
    midpoint), so the curve bulges toward ``(dy, -dx)``.  Placing labels on the
    curve itself (rather than the chord) keeps them off the node circles and
    lets each arc's label sit on its own side.
    """
    d = p1 - p0
    ctrl = 0.5 * (p0 + p1) + rad * np.array([d[1], -d[0]])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def _feature_interaction_over_lags(A: np.ndarray, agg: str = "sum") -> np.ndarray:
    """
    Reduce ``|A|`` over lags into a feature x feature matrix ``S[target, source]``.

    ``agg="sum"`` accumulates every lag (sensitive but lets many tiny noise
    coefficients pile up); ``agg="max"`` keeps only the single largest-magnitude
    lag per pair (a much cleaner signal-vs-noise separator).
    """
    n_features = A.shape[0]
    max_lag = A.shape[1] // n_features
    A_abs = np.abs(A).reshape(n_features, max_lag, n_features)
    if agg == "sum":
        return A_abs.sum(axis=1)
    if agg == "max":
        return A_abs.max(axis=1)
    raise ValueError(f"agg must be 'sum' or 'max', got {agg!r}.")


def plot_frequency_resolved_graph(
    A: np.ndarray,
    n_channels: int,
    n_bands: int,
    band_names: Sequence[str],
    edge_mask: np.ndarray,
    channel_positions: Optional[np.ndarray] = None,
    channel_labels: Optional[Sequence[str]] = None,
    max_edges: Optional[int] = None,
    annotate: bool = True,
    label_top_k: Optional[int] = 5,
    title: str = "Channel graph with band labels  (Lasso-nonzero links)",
    node_color: str = PALETTE["beige"],
    figsize: Tuple[float, float] = STANDARD_FIGSIZE_SQUARE,
    curvature: float = 0.16,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Spatial channel graph with dominant source/target band labels per arrow.

    Each Lasso-nonzero cross-channel × cross-band link is coloured by its
    dominant source band and labelled ``src_band → tgt_band``.  Arrow width
    encodes peak ``|A|``.
    """
    band_names = list(band_names)
    S = _feature_interaction_over_lags(A, agg="max")
    sig = cross_both_edge_mask(edge_mask, n_channels, n_bands)

    pos = _resolve_node_positions(channel_positions, n_channels)
    if channel_labels is None:
        channel_labels = [f"Ch{i}" for i in range(n_channels)]
    band_colors = _band_color_map(band_names)

    def _block(tch: int, sch: int) -> np.ndarray:
        return S[tch * n_bands:(tch + 1) * n_bands, sch * n_bands:(sch + 1) * n_bands]

    def _sig_block(tch: int, sch: int) -> np.ndarray:
        return sig[tch * n_bands:(tch + 1) * n_bands, sch * n_bands:(sch + 1) * n_bands]

    # Mean band-interaction block over all off-diagonal channel pairs; removes
    # the systematic background before choosing each edge's band label.
    bg = np.zeros((n_bands, n_bands), dtype=float)
    n_pairs = 0
    for tch in range(n_channels):
        for sch in range(n_channels):
            if sch != tch:
                bg += _block(tch, sch)
                n_pairs += 1
    if n_pairs:
        bg /= n_pairs

    edges = []  # (src_ch, tgt_ch, weight, src_band, tgt_band)
    for tch in range(n_channels):
        for sch in range(n_channels):
            if sch == tch:
                continue
            block = _block(tch, sch)
            sblock = _sig_block(tch, sch)
            if not sblock.any():
                continue
            cand = np.where(sblock, block - bg, -np.inf)
            total = float(block.max())
            if total <= 0:
                continue
            tb, sb = np.unravel_index(int(np.argmax(cand)), block.shape)
            edges.append((sch, tch, total, band_names[sb], band_names[tb]))

    edges.sort(key=lambda e: e[2], reverse=True)
    if max_edges is not None:
        edges = edges[:max_edges]
    weights = np.array([e[2] for e in edges], dtype=float)

    node_strength = np.zeros(n_channels)
    for sch, tch, w, _sb, _tb in edges:
        node_strength[sch] += w
        node_strength[tch] += w

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    span = np.ptp(pos, axis=0)
    scale = float(np.max(span)) if np.max(span) > 0 else 1.0
    node_r = 0.06 * scale
    pad = node_r * 3.0
    ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
    ax.set_ylim(pos[:, 1].min() - pad, pos[:, 1].max() + pad)
    shrink = _edge_shrink_points(ax, node_r)
    w_hi = float(weights.max()) if weights.size else 1.0

    labeled_edges = list(edges)
    if label_top_k is not None:
        labeled_edges = labeled_edges[:label_top_k]
    label_set = {(e[0], e[1]) for e in labeled_edges}

    for sch, tch, w, sb, tb in sorted(edges, key=lambda e: e[2]):
        frac = w / w_hi
        arrow = FancyArrowPatch(
            pos[sch], pos[tch],
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>", mutation_scale=12 + 15 * frac,
            lw=1.2 + 3.2 * frac, color=band_colors[sb], alpha=0.6 + 0.35 * frac,
            shrinkA=shrink, shrinkB=shrink, zorder=2, capstyle="round",
        )
        ax.add_patch(arrow)
        if annotate and (sch, tch) in label_set:
            m = _arc_point(pos[sch], pos[tch], curvature, t=0.42)
            ax.text(
                m[0], m[1], f"{sb}\u2192{tb}", fontsize=7, ha="center", va="center",
                color=EDGE_COLOR, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=band_colors[sb], lw=0.8, alpha=0.9),
            )

    _draw_graph_nodes(ax, pos, channel_labels, node_r, node_color)
    ax.set_title(title, fontsize=13)

    handles = [Line2D([0], [0], color=band_colors[b], lw=3, label=b) for b in band_names]
    ax.legend(handles=handles, title="Source band", loc="upper left",
              frameon=True, framealpha=0.9, edgecolor="0.85", fontsize=8,
              title_fontsize=9)
    ax.text(
        0.5, -0.02,
        "Colour = dominant source band; label = src→tgt band pair.  "
        "Only Lasso-nonzero cross-channel × cross-band links.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color=EDGE_COLOR,
    )

    _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi, tight_layout=False)
    return {
        "edges": edges,
        "positions": pos,
        "node_strength": node_strength,
        "band_colors": band_colors,
    }


def plot_edge_lag_profiles(
    A: np.ndarray,
    band_names: Sequence[str],
    n_channels: int,
    fs_features: float,
    edges: Sequence[Tuple[Tuple[int, str], Tuple[int, str]]],
    two_period_lags: Optional[Dict[str, int]] = None,
    channel_labels: Optional[Sequence[str]] = None,
    ncols: int = 3,
    title: str = "Lag profile of one directed edge  (coefficients A[target, lag, source])",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Dict[str, Any]:
    """
    Lag profiles ``A[target, lag, source]`` for several directed edges.

    Each panel shows one edge as a stem plot versus lag in milliseconds.  The
    peak coefficient and its lag are annotated; when the lag profile is broad
    (effective spread larger than the source band's envelope timescale) the
    caption notes that the delay is *unresolved* — common for slow bands whose
    power envelope varies on a timescale longer than the coupling lead.
    """
    band_names = list(band_names)
    n_bands = len(band_names)
    n_features = n_channels * n_bands
    max_lag = A.shape[1] // n_features
    A3 = A.reshape(n_features, max_lag, n_features)  # [target, lag, source]
    feat_labels = _make_feature_labels(n_channels, band_names, channel_labels)

    def _feat(ch: int, band: str) -> int:
        if band not in band_names:
            raise ValueError(f"Unknown band {band!r}; choices: {band_names}")
        return ch * n_bands + band_names.index(band)

    lag_ms = np.arange(1, max_lag + 1) / fs_features * 1000.0
    n = len(edges)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = STANDARD_FIGSIZE

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    profiles = []
    lag_summaries = []
    for idx, (source, target) in enumerate(edges):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        sf = _feat(*source)
        tf = _feat(*target)
        prof = A3[tf, :, sf]
        profiles.append(prof)
        w = np.abs(prof)
        if w.sum() > 0:
            centroid_ms = float(np.sum(lag_ms * w) / w.sum())
            spread_ms = float(np.sqrt(np.sum(w * (lag_ms - centroid_ms) ** 2) / w.sum()))
        else:
            centroid_ms, spread_ms = float("nan"), float("nan")
        lag_summaries.append({"centroid_ms": centroid_ms, "spread_ms": spread_ms})

        markerline, stemlines, baseline = ax.stem(lag_ms, prof)
        plt.setp(stemlines, color=PALETTE["blue"], linewidth=1.3)
        plt.setp(markerline, color=PALETTE["terra"], markersize=4)
        plt.setp(baseline, color=EDGE_COLOR, linewidth=0.7)
        pk = int(np.argmax(np.abs(prof)))
        if np.abs(prof[pk]) > 1e-8:
            src_band = source[1]
            env_ms = (
                1000.0 * two_period_lags[src_band] / fs_features
                if two_period_lags and src_band in two_period_lags else None
            )
            if env_ms is not None and spread_ms > 0.5 * env_ms:
                note = (f"peak {prof[pk]:+.3f} @ {lag_ms[pk]:.0f} ms\n"
                        f"centroid {centroid_ms:.0f}±{spread_ms:.0f} ms (unresolved)")
            else:
                note = f"peak {prof[pk]:+.3f} @ {lag_ms[pk]:.0f} ms"
            ax.annotate(
                note, xy=(lag_ms[pk], prof[pk]), xytext=(0.5, 0.88),
                textcoords="axes fraction", fontsize=7.5, color=EDGE_COLOR,
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=EDGE_COLOR, lw=0.8),
            )
        ax.set_title(f"{feat_labels[sf]} -> {feat_labels[tf]}", fontsize=9)
        ax.set_xlabel("Lag (ms)")
        if c == 0:
            ax.set_ylabel("VAR coef $A$")
        _despine(ax, grid=True)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=13)
    _finalize_fig(fig, save_path=save_path, show=show, dpi=dpi)
    return {"lag_ms": lag_ms, "profiles": profiles, "edges": list(edges),
            "lag_summaries": lag_summaries}
