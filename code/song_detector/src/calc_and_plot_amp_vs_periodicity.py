# batch_pipeline.py
# =================
"""
Batch wrapper that, for every .npz in a folder, runs:
   1) plot_NPZ_spectrogram_with_song_highlight  (optional visualisation)
   2) calculate_spectrogram_periodicity         (feature extraction)

Then draws TWO scatter-plots:
   • band-pass periodicity     vs. z-log amplitude
   • full-band periodicity     vs. z-log amplitude
Each point is coloured orange (songs == 1) or black (songs == 0) and a
logistic-regression decision boundary is overlaid.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression   # ← NEW

# ------------------------------------------------------------------
# import helpers (edit the module names if yours differ)
# ------------------------------------------------------------------
from plot_NPZ_spectrogram_with_song_highlight import (
    plot_NPZ_spectrogram_with_song_highlight,
)
from calculate_spectrogram_periodicity import calculate_spectrogram_periodicity


# ------------------------------------------------------------------
# wrapper
# ------------------------------------------------------------------
def batch_periodicity_vs_amplitude(
    folder: str | Path,
    *,
    plot_each_spec: bool = True,
    low_mod: float = 10,
    high_mod: float = 30,
    smoothing_sigma: float = 0.7,
    power_threshold: float = 0.2,
    segment_duration: float = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    x_all          : concatenated z_log_amp_trace
    y_band_all     : concatenated periodicity_bandpass
    y_full_all     : concatenated periodicity_full
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    # storage
    x_vecs: List[np.ndarray] = []
    y_band_vecs: List[np.ndarray] = []
    y_full_vecs: List[np.ndarray] = []
    song_masks: List[np.ndarray] = []

    # ── iterate over files ──────────────────────────────────────
    for npz_file in sorted(folder.glob("*.npz")):
        print(f"Processing {npz_file.name}")

        # (a) visualisation (optional)
        plot_NPZ_spectrogram_with_song_highlight(
            str(npz_file),
            plot_spectrogram=plot_each_spec
        )

        # (b) periodicity & feature extraction
        spec_data = calculate_spectrogram_periodicity(
            str(npz_file),
            plot_figures=True, #if you want the periodicity figures to show up
            low_mod=low_mod,
            high_mod=high_mod,
            smoothing_sigma=smoothing_sigma,
            power_threshold=power_threshold,
            segment_duration=segment_duration,
        )

        # align lengths
        x   = spec_data.z_log_amp_trace
        y_b = spec_data.periodicity_bandpass
        y_f = spec_data.periodicity_full
        s   = spec_data.songs.astype(bool)

        n = min(len(x), len(y_b), len(y_f), len(s))
        x_vecs.append(x[:n])
        y_band_vecs.append(y_b[:n])
        y_full_vecs.append(y_f[:n])
        song_masks.append(s[:n])

    # ── concatenate all files ───────────────────────────────────
    x_all      = np.concatenate(x_vecs)
    y_band_all = np.concatenate(y_band_vecs)
    y_full_all = np.concatenate(y_full_vecs)
    songs_all  = np.concatenate(song_masks).astype(int)   # 0/1
    colours    = np.where(songs_all, "orange", "black")

    # ── helper to draw scatter + decision line ──────────────────
    def scatter_with_boundary(x, y, title):
        # train logistic regression on the two features
        X = np.column_stack([x, y])
        clf = LogisticRegression(solver="lbfgs").fit(X, songs_all)

        # decision line:  w0 + w1*x + w2*y = 0
        w0, w1, w2 = clf.intercept_[0], *clf.coef_[0]
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = -(w0 + w1 * x_line) / w2

        # plot
        plt.scatter(x, y, s=7, c=colours, alpha=0.6)
        plt.plot(x_line, y_line, "b--", lw=2, label="decision boundary")
        plt.xlabel("z-scored log-amplitude")
        plt.ylabel("Periodicity")
        plt.title(title)
        plt.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="None",
                       markerfacecolor="orange", markersize=6, label="Song = 1"),
                Line2D([], [], marker="o", linestyle="None",
                       markerfacecolor="black",  markersize=6, label="Song = 0"),
                Line2D([], [], color="blue", ls="--", lw=2, label="Boundary"),
            ],
            frameon=False
        )
        plt.tight_layout()
        plt.show()

    # ── Figure 1: band-pass periodicity
    scatter_with_boundary(
        x_all, y_band_all,
        f"Band-pass periodicity ({low_mod}–{high_mod} Hz) vs. z-log amplitude"
    )

    # ── Figure 2: full-band periodicity
    scatter_with_boundary(
        x_all, y_full_all,
        "Full-band periodicity vs. z-log amplitude"
    )

    return x_all, y_band_all, y_full_all


# ------------------------------------------------------------------
# Hard-coded run
# ------------------------------------------------------------------
if __name__ == "__main__":
    folder_path = (
        "/Users/mirandahulsey-vincent/Documents/allPythonCode/"
        "BYOD_class/data_inputs/sample_GV_groundtruth_npzs"
    )

    batch_periodicity_vs_amplitude(
        folder=folder_path,
        plot_each_spec=True,
        low_mod=10,
        high_mod=30,
        smoothing_sigma=0.7,
        power_threshold=0.2,
        segment_duration=10,
    )
