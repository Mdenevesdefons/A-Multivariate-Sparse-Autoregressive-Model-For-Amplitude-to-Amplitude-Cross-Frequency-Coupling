"""
Data loading and signal preprocessing for the multivariate sparse AACFC pipeline.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any, Union

import os
import re
import warnings
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import pywt


_MORLET_COI_FACTOR = np.sqrt(2.0)


def _cmor_bandwidth_from_q(q: float, central_freq: float = 1.0) -> float:
    """
    Convert a wavelet quality factor ``q`` (number of cycles) into pywt's
    complex-Morlet bandwidth parameter ``B`` for ``cmor{B}-{C}``.

    pywt defines ``cmorB-C`` as ``(1/sqrt(pi B)) exp(-t^2 / B) exp(2j pi C t)``,
    whose Gaussian time std is ``sqrt(B/2)`` and whose frequency std is
    ``sigma_f = 1 / (pi sqrt(2B))``.  The quality factor (cycle count) is
    therefore ``q = C / sigma_f = pi C sqrt(2B)``, which inverts to

        B = q**2 / (2 pi**2 C**2).

    With the conventional ``C = 1.0`` this gives ``B = q**2 / (2 pi**2)``
    (e.g. q = 6 cycles -> B ~ 1.824).
    """
    if q <= 0:
        raise ValueError("q (number of cycles) must be positive.")
    return float(q) ** 2 / (2.0 * np.pi ** 2 * float(central_freq) ** 2)


def _morlet_coi_half_width_samples(
    scale: float, trim_edge_periods: float = 1.0
) -> int:
    """One-sided COI half-width in samples."""
    if trim_edge_periods <= 0:
        return 0
    return int(np.ceil(trim_edge_periods * _MORLET_COI_FACTOR * float(scale)))


def _mask_morlet_cone_of_influence(
    power: np.ndarray,
    scales: np.ndarray,
    trim_edge_periods: float,
) -> np.ndarray:
    """
    Set CWT power inside the cone of influence to NaN (per scale row).

    Parameters
    ----------
    power : (n_scales, time), linear power |coef|^2
    scales : (n_scales,) pywt CWT scales matching ``power`` rows
    """
    if trim_edge_periods <= 0:
        return power
    out = power.astype(np.float64, copy=True)
    n_t = out.shape[1]
    coi = np.array(
        [_morlet_coi_half_width_samples(s, trim_edge_periods) for s in scales],
        dtype=int,
    )
    coi = np.minimum(coi, n_t // 2)
    t_idx = np.arange(n_t)
    invalid = (t_idx[None, :] < coi[:, None]) | (t_idx[None, :] >= n_t - coi[:, None])
    out[invalid] = np.nan
    return out


def _aggregate_cwt_power(power: np.ndarray, aggregation: str) -> np.ndarray:
    """
    Combine (n_scales, time) CWT power across scales, respecting NaN COI masks.

    Uses nansum for ``"sum"``.  For ``"mean"``, divides by the count of finite
    values only.
    """
    if aggregation == "sum":
        return np.nansum(power, axis=0)
    total = np.nansum(power, axis=0)
    count = np.sum(np.isfinite(power), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    mean[count == 0] = np.nan
    return mean


def _morlet_symmetric_coi_trim(
    frequency_bands: Dict[str, Tuple[float, float]],
    sampling_rate: float,
    central_frequency: float,
    trim_edge_periods: float,
) -> Tuple[int, str]:
    """Symmetric edge trim (samples) and the band that sets it (lowest f_lo)."""
    trim = 0
    limiting_band = ""
    for band, (f_lo, _) in frequency_bands.items():
        if f_lo <= 0:
            continue
        scale_lo = central_frequency * sampling_rate / f_lo
        band_trim = _morlet_coi_half_width_samples(scale_lo, trim_edge_periods)
        if band_trim > trim:
            trim = band_trim
            limiting_band = band
    return trim, limiting_band


def _hilbert_symmetric_coi_trim(
    frequency_bands: Dict[str, Tuple[float, float]],
    sampling_rate: float,
    trim_edge_periods: float,
    filter_order: int = 4,
) -> Tuple[int, str]:
    """Symmetric edge trim for Hilbert envelopes and the limiting band."""
    padlen = 3 * (2 * filter_order)
    trim = 0
    limiting_band = ""
    for band, (f_lo, _) in frequency_bands.items():
        if f_lo <= 0:
            continue
        period_trim = int(np.ceil(trim_edge_periods * sampling_rate / f_lo))
        band_trim = max(period_trim, padlen)
        if band_trim > trim:
            trim = band_trim
            limiting_band = band
    return trim, limiting_band


def _build_coi_trim_stats(
    n_samples_input: int,
    trim_samples_per_edge: int,
    sampling_rate: float,
    trim_edge_periods: float,
    method: str,
    limiting_band: str,
    trim_applied: bool,
) -> Dict[str, Any]:
    """Summary of samples discarded by symmetric COI edge trimming."""
    if not trim_applied or trim_samples_per_edge <= 0:
        n_removed = 0
        n_out = n_samples_input
        per_edge = 0
    else:
        per_edge = int(trim_samples_per_edge)
        n_removed = 2 * per_edge
        n_out = n_samples_input - n_removed

    fs = float(sampling_rate)
    return {
        "n_samples_input": int(n_samples_input),
        "n_samples_output": int(n_out),
        "n_samples_removed": int(n_removed),
        "n_samples_removed_per_edge": int(per_edge),
        "duration_removed_sec": float(n_removed / fs),
        "duration_removed_per_edge_sec": float(per_edge / fs),
        "fraction_removed": float(n_removed / n_samples_input) if n_samples_input else 0.0,
        "trim_edge_periods": float(trim_edge_periods),
        "method": method,
        "limiting_band": limiting_band,
        "trim_applied": bool(trim_applied),
    }


# I/O

def load_mat_ecog(
    mat_file_path: str,
    var_name: Optional[str] = None,
    transpose_if_needed: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a MATLAB ``.mat`` file containing one ECoG/LFP variable.

    Parameters
    ----------
    mat_file_path : str
        Path to the ``.mat`` file.
    var_name : str, optional
        Variable to read.  Defaults to the first non-metadata key.
    transpose_if_needed : bool
        If True (default) and the array is taller than it is wide, assume the
        longer axis is time and transpose to ``(channels, time)``.

    Returns
    -------
    data : ndarray, shape (channels, time)
    meta : dict
        Metadata for downstream tracing.
    """
    mat = sio.loadmat(mat_file_path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if var_name is None:
        if len(keys) == 0:
            raise ValueError(f"No data variables found in {mat_file_path}.")
        var_name = keys[0]

    data = np.asarray(mat[var_name], dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D array for {var_name!r}, got shape {data.shape}."
        )

    transposed = False
    if transpose_if_needed and data.shape[0] > data.shape[1]:
        warnings.warn(
            f"{var_name!r} has shape {data.shape}; assuming the longer "
            f"axis is time and transposing to (channels, time). Set "
            f"transpose_if_needed=False to disable.",
            UserWarning,
        )
        data = data.T
        transposed = True

    return data, {
        "keys": keys,
        "var_name": var_name,
        "shape": data.shape,
        "auto_transposed": transposed,
    }


# Raw preprocessing

def downsample_raw_ecog(
    ecog_data: np.ndarray,
    orig_fs: float,
    target_fs: float,
    axis: int = -1,
) -> Tuple[np.ndarray, float]:
    """
    Anti-aliased polyphase resampling to a lower sampling rate.

    Recommended ``target_fs >= 2 * f_max_of_interest``.  With <= 50 Hz of
    interest, 100-200 Hz is plenty.  The decimation factor must be an integer
    that approximates ``orig_fs / target_fs`` within 1%; otherwise the function
    raises, because a wrong effective sampling rate would silently corrupt the
    lag counts in ``compute_two_period_band_lags``.

    Returns
    -------
    out : ndarray
        Resampled signal.
    eff_fs : float
        Effective sampling rate after integer decimation.
    """
    if target_fs <= 0:
        raise ValueError("target_fs must be positive.")
    if target_fs >= orig_fs:
        return np.asarray(ecog_data, dtype=float), float(orig_fs)

    ratio = orig_fs / target_fs
    down = max(1, int(round(ratio)))
    eff_fs = orig_fs / down
    rel_err = abs(eff_fs - target_fs) / target_fs
    if rel_err > 0.01:
        raise ValueError(
            f"orig_fs / target_fs = {ratio:.4f} cannot be approximated "
            f"by an integer decimation within 1% (would give effective "
            f"fs = {eff_fs:.4f} Hz, rel err = {rel_err*100:.2f}%). "
            f"Pick a target_fs that divides orig_fs (e.g. {orig_fs/round(ratio):.2f} Hz)."
        )
    if abs(down - ratio) > 1e-6:
        warnings.warn(
            f"orig_fs / target_fs = {ratio:.4f} is not integer; "
            f"using down = {down} (effective fs = {eff_fs:.4f} Hz, "
            f"rel err = {rel_err*100:.3f}%). "
            f"Downstream code will use the effective fs.",
            UserWarning,
        )
    out = signal.resample_poly(ecog_data, up=1, down=down, axis=axis)
    return out, eff_fs


# Time-frequency decomposition

def extract_band_power(
    ecog_data: np.ndarray,
    sampling_rate: float,
    frequency_bands: Dict[str, Tuple[float, float]],
    method: str = "morlet",
    n_freqs_per_band: int = 12,
    morlet_q: float = 6.0,
    aggregation: str = "mean",
    trim_edge_periods: float = 2.0,
    return_coi_stats: bool = False,
) -> Union[
    Tuple[np.ndarray, List[str]],
    Tuple[np.ndarray, List[str], Dict[str, Any]],
]:
    """
    Extract band-limited power per channel.

    Parameters
    ----------
    ecog_data : (channels, time)
    sampling_rate : float
    frequency_bands : dict
        e.g. ``{"delta": (0.5, 4), "theta": (4, 8), ...}``.
    method : {"morlet", "hilbert"}
        - "morlet"  : continuous **complex** Morlet wavelet (cmor), so
                      ``|coefs|^2`` is the smooth analytic power envelope.
        - "hilbert" : 4th-order Butterworth band-pass + Hilbert envelope.
    n_freqs_per_band : int
        Number of log-spaced wavelet frequencies inside each band (Morlet only).
    morlet_q : float
        Wavelet **quality factor = number of cycles** (default 6.0).  The filter
        at centre frequency ``f0`` has bandwidth ``df = f0 / q`` and its power
        envelope fluctuates on a timescale ``tau = q / f0``.  Internally it is
        converted to pywt's ``cmor{B}-1.0`` bandwidth via ``B = q**2 / (2 pi**2)``.
        Use the **same** ``q`` in the VAR fit.
    aggregation : {"mean", "sum"}
        How to combine in-band wavelet powers.  ``"mean"`` (default) keeps the
        scale independent of ``n_freqs_per_band`` and comparable to Hilbert.
    trim_edge_periods : float
        Cone-of-influence margin for boundary artefacts.

        **Morlet:** each in-band CWT scale is masked where
        ``|t - edge| < trim_edge_periods * sqrt(2) * scale`` before the in-band
        ``nanmean``; a final symmetric trim removes the union of all bands' COI
        at their lowest in-band frequency so every band stays time-aligned.

        **Hilbert:** symmetric trim of ``max(trim_edge_periods`` cycles at each
        band's lower edge, filtfilt pad length).

        Set to ``0`` to disable all COI handling.
    return_coi_stats : bool
        If ``True``, also return a dict summarising how many samples were removed.

    Returns
    -------
    band_power : ndarray, shape (channels, n_bands, time'), float32
        Band-limited spectral power (linear units, NOT log).
    band_names : list of str
    coi_stats : dict, optional
        Present when ``return_coi_stats=True``.
    """
    if aggregation not in ("mean", "sum"):
        raise ValueError("aggregation must be 'mean' or 'sum'.")

    ecog_data = np.asarray(ecog_data, dtype=float)
    if ecog_data.ndim != 2:
        raise ValueError("ecog_data must be 2D (channels, time).")

    n_ch, n_t = ecog_data.shape
    band_names = list(frequency_bands.keys())
    n_b = len(band_names)
    band_power = np.zeros((n_ch, n_b, n_t), dtype=np.float32)

    cf = 1.0  # set below for the morlet branch; unused by hilbert
    if method == "morlet":
        # Complex Morlet ('cmor{B}-{C}', C = centre freq = 1.0).  Build B from
        # the requested quality factor q (cycles) so the wavelet has q cycles.
        morlet_B = _cmor_bandwidth_from_q(morlet_q, central_freq=1.0)
        wavelet = f"cmor{morlet_B}-1.0"
        # pywt mapping: scale = central_frequency / (freq * dt) = cf * fs / freq.
        cf = float(pywt.central_frequency(wavelet))
        for b, band in enumerate(band_names):
            f_lo, f_hi = frequency_bands[band]
            if f_lo <= 0 or f_hi <= f_lo:
                raise ValueError(f"Bad band {band}: ({f_lo}, {f_hi}).")
            freqs = np.geomspace(f_lo, f_hi, n_freqs_per_band)
            scales = cf * sampling_rate / freqs
            recovered = pywt.scale2frequency(wavelet, scales) * sampling_rate
            if not np.allclose(recovered, freqs, rtol=1e-6):
                raise RuntimeError(
                    f"pywt scale<->frequency mismatch for band {band}: "
                    f"requested {freqs}, got {recovered}."
                )
            for ch in range(n_ch):
                coefs, _ = pywt.cwt(
                    ecog_data[ch], scales, wavelet,
                    sampling_period=1.0 / sampling_rate,
                )
                power = np.abs(coefs) ** 2
                power = _mask_morlet_cone_of_influence(
                    power, scales, trim_edge_periods,
                )
                band_power[ch, b, :] = _aggregate_cwt_power(
                    power, aggregation,
                ).astype(np.float32)

    elif method == "hilbert":
        for b, band in enumerate(band_names):
            f_lo, f_hi = frequency_bands[band]
            nyq = sampling_rate / 2.0
            f_hi_safe = min(f_hi, nyq * 0.999)  # avoid aliasing near Nyquist
            sos = signal.butter(
                4, [f_lo, f_hi_safe], btype="bandpass", fs=sampling_rate, output="sos"
            )
            filtered = signal.sosfiltfilt(sos, ecog_data, axis=-1)
            band_power[:, b, :] = (
                np.abs(signal.hilbert(filtered, axis=-1)) ** 2
            ).astype(np.float32)

    else:
        raise ValueError("method must be 'morlet' or 'hilbert'.")

    trim_samples = 0
    limiting_band = ""
    trim_applied = False

    # Symmetric trim: union of per-band COI at both edges.
    if trim_edge_periods > 0:
        if method == "morlet":
            trim_samples, limiting_band = _morlet_symmetric_coi_trim(
                frequency_bands, sampling_rate, cf, trim_edge_periods,
            )
        else:
            trim_samples, limiting_band = _hilbert_symmetric_coi_trim(
                frequency_bands, sampling_rate, trim_edge_periods,
            )
        if 2 * trim_samples >= n_t:
            warnings.warn(
                f"trim_edge_periods={trim_edge_periods} would remove "
                f"2 x {trim_samples} = {2*trim_samples} samples from a "
                f"series of length {n_t}. Skipping edge trimming.",
                UserWarning,
            )
        else:
            band_power = band_power[:, :, trim_samples:-trim_samples]
            trim_applied = True

    coi_stats = _build_coi_trim_stats(
        n_samples_input=n_t,
        trim_samples_per_edge=trim_samples,
        sampling_rate=sampling_rate,
        trim_edge_periods=trim_edge_periods,
        method=method,
        limiting_band=limiting_band,
        trim_applied=trim_applied,
    )

    if return_coi_stats:
        return band_power, band_names, coi_stats
    return band_power, band_names


# Feature preprocessing

def preprocess_band_power(
    band_power: np.ndarray,
    downsample_factor: int = 1,
    epsilon: float = 1e-10,
    detrend_flag: bool = True,
    zscore_flag: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Envelope downsample (block-average) -> log10 -> detrend -> z-score -> flatten.

    Parameters
    ----------
    band_power : (channels, n_bands, time)
    downsample_factor : int
        If > 1, block-average the (non-negative) power envelope by this factor
        before ``log10``.  Block averaging keeps power >= 0 for the log.

    Returns
    -------
    X : ndarray, shape (channels * n_bands, time'), float32
        Feature matrix used by all downstream models.  Feature ordering:
        ``ch0_band0, ch0_band1, ..., ch1_band0, ...`` (channel-major).
    band_power_clean : ndarray, shape (channels, n_bands, time'), float32
    """
    bp = np.asarray(band_power, dtype=np.float32)

    if downsample_factor > 1:
        n_ch, n_b, n_t = bp.shape
        n_out = n_t // downsample_factor
        bp = bp[:, :, : n_out * downsample_factor]
        bp = bp.reshape(n_ch, n_b, n_out, downsample_factor).mean(axis=-1)

    bp = np.log10(bp + epsilon).astype(np.float32)

    if detrend_flag:
        bp = signal.detrend(bp, axis=-1).astype(np.float32)

    if zscore_flag:
        # Manual z-score with a zero-variance guard: a constant feature would
        # divide by zero; flooring std at eps leaves it at 0 (already mean-removed).
        mu = np.nanmean(bp, axis=-1, keepdims=True)
        sd = np.nanstd(bp, axis=-1, keepdims=True)
        sd = np.where(sd < epsilon, 1.0, sd)
        bp = ((bp - mu) / sd).astype(np.float32)

    n_ch, n_b, n_t = bp.shape
    X = bp.reshape(n_ch * n_b, n_t)
    return X, bp


# Optional convenience: full session bundle

def _parse_recording_name(name: str) -> Optional[Tuple[str, str]]:
    """Parse ``ECoG_A<subject>_S<session>`` from a file stem or full name."""
    m = re.search(r"ECoG_A(?P<subject>\w+?)_S(?P<session>\w+)", name, re.IGNORECASE)
    if m is None:
        return None
    return f"A{m.group('subject')}", f"S{m.group('session')}"


def load_session_features(
    mat_file: str,
    *,
    mat_var_name: Optional[str] = None,
    orig_fs: float = 1000.0,
    target_fs: float = 100.0,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = "morlet",
    n_freqs_per_band: int = 20,
    morlet_q: float = 6.0,
    trim_edge_periods: float = 2.0,
    downsample_factor: int = 1,
    random_seed: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load -> downsample -> band power -> preprocess features, in one call.

    Returns
    -------
    dict with keys: ``X`` (feature matrix), ``band_power``, ``band_names``,
    ``bands``, ``fs`` (post-downsample), ``fs_features`` (post envelope
    downsample), ``n_channels``, ``n_bands``, ``n_features``, plus parsed
    ``subject`` / ``session`` and ``meta``.
    """
    if frequency_bands is None:
        frequency_bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 50.0),
        }

    np.random.seed(random_seed)

    def _log(*args):
        if verbose:
            print(*args)

    X_raw, meta = load_mat_ecog(mat_file, var_name=mat_var_name)
    _log(f"Loaded raw shape: {X_raw.shape} (channels, time)")
    n_channels = X_raw.shape[0]

    X_ds, fs = downsample_raw_ecog(X_raw, orig_fs=orig_fs, target_fs=target_fs)
    _log(f"Downsampled: {X_ds.shape}, fs = {fs} Hz")

    band_power, band_names = extract_band_power(
        X_ds,
        sampling_rate=fs,
        frequency_bands=frequency_bands,
        method=method,
        n_freqs_per_band=n_freqs_per_band,
        morlet_q=morlet_q,
        trim_edge_periods=trim_edge_periods,
    )
    _log(f"band_power shape: {band_power.shape}  (channels, bands, time)")

    X, bp_clean = preprocess_band_power(
        band_power,
        downsample_factor=downsample_factor,
        detrend_flag=True,
        zscore_flag=True,
    )
    fs_features = fs / downsample_factor
    _log(f"Feature matrix X: {X.shape}  (channels*bands, time)")
    _log(f"Feature sampling rate: fs_features = {fs_features} Hz")

    stem = os.path.splitext(os.path.basename(mat_file))[0]
    parsed = _parse_recording_name(stem)
    subject = parsed[0] if parsed else None
    session = parsed[1] if parsed else None

    return dict(
        mat_file=mat_file,
        name=stem,
        subject=subject,
        session=session,
        X=X,
        X_ds=X_ds,
        band_power=band_power,
        bp_clean=bp_clean,
        band_names=band_names,
        bands=frequency_bands,
        fs=fs,
        fs_features=fs_features,
        n_channels=n_channels,
        n_bands=len(band_names),
        n_features=X.shape[0],
        morlet_q=morlet_q,
        downsample_factor=downsample_factor,
        meta=meta,
    )
