# calculate_wav_dual_entropy.py  (linear 0–1 spectrogram version)
# =================================================================
"""
Dual-entropy song detector for the “wav-pipeline”.

Assumes `generate_spectrogram_from_wav` returns a **linear, 0-to-1
normalised spectrogram** (after your perceptual **0.7 power** step).

Metrics:
    • Shannon spectral entropy  (bits) – computed on column-wise
      probabilities (power / column-sum).
    • Wiener entropy / spectral flatness  (log₁₀ geom / arith mean)
      – computed on the *raw* linear power matrix.

Each metric can be:
    • Gaussian-smoothed
    • Thresholded
    • Plotted alongside the original spectrogram

Returns a `SpectrogramEntropyData` object.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy as shannon_entropy

from load_wav_gen_spec import generate_spectrogram_from_wav  # your helper

ArrayLike = Union[np.ndarray, list]


# ────────────────────────────────────────────────────────────────────────
# Data container
# ────────────────────────────────────────────────────────────────────────
class SpectrogramEntropyData:
    def __init__(
        self,
        *,
        shannon_entropy_bits: ArrayLike,
        shannon_entropy_bits_smoothed: ArrayLike,
        wiener_entropy_log: ArrayLike,
        wiener_entropy_log_smoothed: ArrayLike,
        shannon_detected_song_times: List[Tuple[float, float]],
        wiener_detected_song_times: List[Tuple[float, float]],
        times: ArrayLike,
        frequencies: ArrayLike,
        spectrogram: ArrayLike,
    ) -> None:
        self.shannon_entropy_bits = shannon_entropy_bits
        self.shannon_entropy_bits_smoothed = shannon_entropy_bits_smoothed
        self.wiener_entropy_log = wiener_entropy_log
        self.wiener_entropy_log_smoothed = wiener_entropy_log_smoothed
        self.shannon_detected_song_times = shannon_detected_song_times
        self.wiener_detected_song_times = wiener_detected_song_times
        self.times = times
        self.frequencies = frequencies
        self.spectrogram = spectrogram


# ────────────────────────────────────────────────────────────────────────
# Helper – mask → (start, end) span list
# ────────────────────────────────────────────────────────────────────────
def mask_to_spans(mask: np.ndarray, t: np.ndarray) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    in_span = False
    for i, flag in enumerate(mask):
        if flag and not in_span:
            in_span, t0 = True, t[i]
        elif not flag and in_span:
            spans.append((t0, t[i]))
            in_span = False
    if in_span:
        spans.append((t0, t[-1]))
    return spans


# ────────────────────────────────────────────────────────────────────────
# MAIN – wav-only (linear-spectrogram) dual entropy
# ────────────────────────────────────────────────────────────────────────
def calculate_wav_spectrogram_entropy(
    wav_path: str | Path,
    *,
    smoothing_sigma: float = 200,
    shannon_threshold_bits: float = 8.0,
    wiener_threshold_log: float = -4,
    plot_shannon_figure: bool = True,
    plot_wiener_figure: bool = True,
    **spec_kw,  # forwarded to generate_spectrogram_from_wav
) -> SpectrogramEntropyData:
    """
    Parameters
    ----------
    wav_path : str | pathlib.Path
        Path to a .wav file.
    **spec_kw :
        Extra keyword args for `generate_spectrogram_from_wav`.
    """
    wav_path = Path(wav_path)
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"{wav_path} is not a .wav file")

    # 1) Build linear 0-1 spectrogram ---------------------------------------------
    spec_dict = generate_spectrogram_from_wav(wav_path, **spec_kw)
    S_norm    = spec_dict["spectrogram"]      # already linear 0–1
    f         = spec_dict["frequencies"]
    t         = spec_dict["times"]
    title_tag = wav_path.name

    eps = np.finfo(float).eps

    # 2) Shannon entropy -----------------------------------------------------------
    P_linear = S_norm + eps                                # linear power
    P_prob   = P_linear / P_linear.sum(axis=0, keepdims=True)

    sh_bits       = shannon_entropy(P_prob, base=2, axis=0)
    sh_bits_sm    = gaussian_filter1d(sh_bits, sigma=smoothing_sigma)
    sh_mask       = sh_bits_sm < shannon_threshold_bits
    sh_spans      = mask_to_spans(sh_mask, t)

    # 3) Wiener entropy ------------------------------------------------------------
    arith_mean    = P_linear.mean(axis=0) + eps
    geom_mean     = np.exp(np.log(P_linear + eps).mean(axis=0))
    wiener_log    = np.log10(geom_mean / arith_mean)
    wiener_sm     = gaussian_filter1d(wiener_log, sigma=smoothing_sigma)
    w_mask        = wiener_sm < wiener_threshold_log
    w_spans       = mask_to_spans(w_mask, t)

    # 4) Plot helper ---------------------------------------------------------------
    def _plot(raw, sm, thr, spans, ylabel, title):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw=dict(height_ratios=[1, 2]))
        ticks = np.linspace(t[0], t[-1], 6)

        axs[0].plot(t, raw, color="gray", alpha=0.4, label="Raw")
        axs[0].plot(t, sm,  color="black",
                    label=f"Smoothed (σ={smoothing_sigma})")
        axs[0].axhline(thr, color="red", ls="--", label="Threshold")
        for s_, e_ in spans:
            axs[0].axvspan(s_, e_, color="yellow", alpha=0.3)
        axs[0].set_ylabel(ylabel)
        axs[0].legend(fontsize="x-small", loc="upper right")

        axs[1].pcolormesh(t, f, S_norm, shading="auto", cmap="binary")
        axs[1].set_ylabel("Frequency (Hz)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylim(0, max(f))
        for s_, e_ in spans:
            axs[1].axvspan(s_, e_, color="yellow", alpha=0.3)

        for ax in axs:
            ax.set_xticks(ticks)
            ax.tick_params(axis="x", labelbottom=True)
        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
        fig.tight_layout()
        plt.show()

    if plot_shannon_figure:
        _plot(sh_bits, sh_bits_sm, shannon_threshold_bits, sh_spans,
              "Shannon entropy (bits)",
              f"Shannon Entropy – {title_tag}")

    if plot_wiener_figure:
        _plot(wiener_log, wiener_sm, wiener_threshold_log, w_spans,
              "log₁₀(Flatness)",
              f"Wiener Entropy – {title_tag}")

    # 5) Package results -----------------------------------------------------------
    return SpectrogramEntropyData(
        shannon_entropy_bits=sh_bits,
        shannon_entropy_bits_smoothed=sh_bits_sm,
        wiener_entropy_log=wiener_log,
        wiener_entropy_log_smoothed=wiener_sm,
        shannon_detected_song_times=sh_spans,
        wiener_detected_song_times=w_spans,
        times=t,
        frequencies=f,
        spectrogram=S_norm,
    )
