#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
**Dual‑entropy bird‑song detector**

Computes **Shannon spectral entropy** *and* **Wiener entropy (spectral flatness)**
from a pre‑computed spectrogram (.npz) or an in‑memory dict.  Each entropy
trace can be  
• Gaussian‑smoothed  
• Compared against a user threshold  
• Plotted (raw, smoothed, threshold, detected spans, + original spectrogram)

Returns a **`SpectrogramEntropyData`** instance bundling:

* raw & smoothed Shannon entropy
* raw & smoothed Wiener entropy
* detected song spans from each metric
* original spectrogram + axes

--------------------------------------------------------------------
Interactive use
---------------
```python
from calculate_dual_entropy import calculate_spectrogram_entropy
entropy_data = calculate_spectrogram_entropy(
    npz_path,
    smoothing_sigma=10,
    shannon_threshold_bits=4.0,
    wiener_threshold_log=-0.5,
)
print(entropy_data.shannon_detected_song_times)
```

CLI quick‑look
--------------
```bash
python calculate_dual_entropy.py segment.npz --sigma 10 \
     --sthresh 4.0 --wthresh -0.5
```
"""

from pathlib import Path
from typing import List, Tuple, Dict, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy as shannon_entropy

ArrayLike = Union[np.ndarray, list]

# ────────────────────────────────────────────────────────────────────────────
# Data container
# ────────────────────────────────────────────────────────────────────────────

class SpectrogramEntropyData:
    """Container for entropy features & detected song spans."""

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

# ────────────────────────────────────────────────────────────────────────────
# Utility to extract spans from a boolean mask
# ────────────────────────────────────────────────────────────────────────────

def mask_to_spans(mask: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    in_span = False
    for i, flag in enumerate(mask):
        if flag and not in_span:
            in_span, span_start = True, times[i]
        elif not flag and in_span:
            in_span = False
            spans.append((span_start, times[i]))
    if in_span:
        spans.append((span_start, times[-1]))
    return spans

# ────────────────────────────────────────────────────────────────────────────
# Main function
# ────────────────────────────────────────────────────────────────────────────

def calculate_npz_spectrogram_entropy(
    input_data: Union[str, dict],
    *,
    smoothing_sigma: float = 20,
    shannon_threshold_bits: float = 4.0,
    wiener_threshold_log: float = -0.5,
    plot_shannon_figure: bool = True,
    plot_wiener_figure: bool = True,
    segment_duration: float = 10,
) -> SpectrogramEntropyData:
    """Compute Shannon & Wiener entropy and return results as a data class."""

    eps = np.finfo(float).eps

    # ── 1. Load spectrogram ────────────────────────────────────────────────
    if isinstance(input_data, (str, bytes)):
        with np.load(input_data, allow_pickle=True) as data:
            spectrogram = data['s']
            frequencies = data['f'] if 'f' in data.files else np.arange(spectrogram.shape[0])
            times = data['t'] if 't' in data.files else np.linspace(0, segment_duration, spectrogram.shape[1], endpoint=False)
            name_for_plot = Path(input_data).stem
    elif isinstance(input_data, dict):
        spectrogram = input_data['spectrogram']
        frequencies = input_data['frequencies']
        times = input_data['times']
        name_for_plot = 'in‑memory input'
    else:
        raise TypeError('input_data must be a file path or dictionary')

    # Convert dB → linear power & normalise each column
    power_spec = 10 ** (spectrogram / 10) + eps
    power_spec /= power_spec.sum(axis=0, keepdims=True)

    # ── 2. Shannon entropy (bits) ──────────────────────────────────────────
    shannon_bits = shannon_entropy(power_spec, base=2, axis=0)
    shannon_bits_sm = gaussian_filter1d(shannon_bits, sigma=smoothing_sigma)
    sh_mask = shannon_bits_sm < shannon_threshold_bits
    sh_spans = mask_to_spans(sh_mask, times)

    # ── 3. Wiener entropy (log flatness) ───────────────────────────────────
    arithmetic_mean = power_spec.mean(axis=0) + eps
    geometric_mean = np.exp(np.log(power_spec).mean(axis=0))
    wiener_log = np.log10(geometric_mean / arithmetic_mean)
    wiener_log_sm = gaussian_filter1d(wiener_log, sigma=smoothing_sigma)
    w_mask = wiener_log_sm < wiener_threshold_log
    w_spans = mask_to_spans(w_mask, times)

    # ── 4. Plot helper ─────────────────────────────────────────────────────
    def _plot(tr_raw, tr_sm, thr, spans, ylabel, title):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                gridspec_kw=dict(height_ratios=[1, 2]))
        ticks = np.linspace(times[0], times[-1], 6)

        axs[0].plot(times, tr_raw, color='gray', alpha=0.4, label='Raw')
        axs[0].plot(times, tr_sm,  color='black', label=f'Smoothed (σ={smoothing_sigma})')
        axs[0].axhline(thr, color='red', ls='--', label='Threshold')
        for s, e in spans:
            axs[0].axvspan(s, e, color='yellow', alpha=0.3)
        axs[0].set_ylabel(ylabel)
        axs[0].legend(fontsize='x-small', loc='upper right')

        axs[1].pcolormesh(times, frequencies, spectrogram, shading='auto', cmap='binary')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylim(0, 500)
        for s, e in spans:
            axs[1].axvspan(s, e, color='yellow', alpha=0.3)

        for ax in axs:
            ax.set_xticks(ticks)
            ax.tick_params(axis='x', labelbottom=True)

        fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        fig.tight_layout()
        plt.show()

    if plot_shannon_figure:
        _plot(shannon_bits, shannon_bits_sm, shannon_threshold_bits, sh_spans,
              'Shannon entropy (bits)',
              f'Shannon Entropy & Detected Song for {name_for_plot}')

    if plot_wiener_figure:
        _plot(wiener_log, wiener_log_sm, wiener_threshold_log, w_spans,
              'log₁₀(Flatness)',
              f'Wiener Entropy & Detected Song for {name_for_plot}')

    # ── 5. Package into data class ─────────────────────────────────────────
    return SpectrogramEntropyData(
        shannon_entropy_bits=shannon_bits,
        shannon_entropy_bits_smoothed=shannon_bits_sm,
        wiener_entropy_log=wiener_log,
        wiener_entropy_log_smoothed=wiener_log_sm,
        shannon_detected_song_times=sh_spans,
        wiener_detected_song_times=w_spans,
        times=times,
        frequencies=frequencies,
        spectrogram=spectrogram,
    )

