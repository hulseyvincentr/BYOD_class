# batch_pipeline.py
# =================
"""
For every .npz in *folder*:
    1) plot_NPZ_spectrogram_with_song_highlight  (optional visualisation)
    2) calculate_spectrogram_amp_periodicity_entropy (feature extraction)

Outputs
-------
• 3-D scatter-plot (amp × band-pass periodicity × entropy) + decision plane
• 2-D scatter-plots with logistic-regression boundaries:
      (i)   amp  vs  band-pass periodicity
      (ii)  amp  vs  entropy
      (iii) band-pass periodicity  vs  entropy
      (iv)  full-band  periodicity vs  entropy
"""

from pathlib import Path
#from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401
from sklearn.linear_model import LogisticRegression

from plot_NPZ_spectrogram_with_song_highlight import (
    plot_NPZ_spectrogram_with_song_highlight,
)
from calculate_spectrogram_amp_periodicity_entropy import (
    calculate_spectrogram_amp_periodicity_entropy,
)


# ─────────────────────────────────────────────────────────────────
def batch_amp_periodicity_entropy(
    folder: str | Path,
    *,
    plot_each_spec: bool = False,
    low_mod: float = 10,
    high_mod: float = 30,
    smoothing_sigma: float = 0.7,
    power_threshold: float = 0.2,
    segment_duration: float = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    amp_all  : concatenated z_log_amp_trace
    per_bp   : concatenated band-pass periodicity
    ent_all  : concatenated Wiener-entropy trace
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    # storage
    amp_vecs, per_bp_vecs, per_full_vecs, ent_vecs, song_masks = (
        [] for _ in range(5)
    )

    # ── iterate over files ──────────────────────────────────────
    for npz_file in sorted(folder.glob("*.npz")):
        print(f"Processing {npz_file.name}")

        # (a) optional spectrogram preview
        plot_NPZ_spectrogram_with_song_highlight(
            npz_file, plot_spectrogram=plot_each_spec
        )

        # (b) feature extraction
        sd = calculate_spectrogram_amp_periodicity_entropy(
            npz_file,
            plot_figures=True,
            low_mod=low_mod,
            high_mod=high_mod,
            smoothing_sigma=smoothing_sigma,
            power_threshold=power_threshold,
            segment_duration=segment_duration,
        )

        # align lengths
        n = min(len(sd.z_log_amp_trace),
                len(sd.periodicity_bandpass),
                len(sd.periodicity_full),
                len(sd.wiener_entropy),
                len(sd.songs))
        amp_vecs.append(sd.z_log_amp_trace[:n])
        per_bp_vecs.append(sd.periodicity_bandpass[:n])
        per_full_vecs.append(sd.periodicity_full[:n])
        ent_vecs.append(sd.wiener_entropy[:n])
        song_masks.append(sd.songs[:n].astype(bool))

    # ── concatenate all files ───────────────────────────────────
    amp_all      = np.concatenate(amp_vecs)
    per_bp_all   = np.concatenate(per_bp_vecs)
    per_full_all = np.concatenate(per_full_vecs)
    ent_all      = np.concatenate(ent_vecs)
    songs_all    = np.concatenate(song_masks).astype(int)   # 0/1
    colours      = np.where(songs_all, "orange", "black")

    # ── helper: 2-D scatter with logistic boundary ─────────────
    def scatter2d(x, y, xlabel, ylabel, title):
        X = np.column_stack([x, y])
        clf = LogisticRegression(solver="lbfgs").fit(X, songs_all)
        w0, w1, w2 = clf.intercept_[0], *clf.coef_[0]

        x_line = np.linspace(x.min(), x.max(), 250)
        y_line = -(w0 + w1 * x_line) / w2

        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, c=colours, s=7, alpha=0.6)
        plt.plot(x_line, y_line, "b--", lw=2, label="Boundary")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="None",
                       markerfacecolor="orange", markersize=6, label="Song = 1"),
                Line2D([], [], marker="o", linestyle="None",
                       markerfacecolor="black",  markersize=6, label="Song = 0"),
                Line2D([], [], color="blue", ls="--", lw=2, label="Boundary"),
            ],
            frameon=False, fontsize="small"
        )
        plt.tight_layout()
        plt.show()

    # ── 3-D scatter & plane ────────────────────────────────────
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(amp_all, per_bp_all, ent_all, c=colours, s=7, alpha=0.6)
    ax.set_xlabel("z-log amplitude")
    ax.set_ylabel(f"Periodicity {low_mod}–{high_mod} Hz")
    ax.set_zlabel("Wiener entropy")
    ax.set_title("Amplitude × Periodicity × Entropy")

    # decision plane
    X3 = np.column_stack([amp_all, per_bp_all, ent_all])
    clf3 = LogisticRegression(solver="lbfgs").fit(X3, songs_all)
    w0, w1, w2, w3 = clf3.intercept_[0], *clf3.coef_[0]
    if abs(w3) > 1e-6:
        x_grid = np.linspace(amp_all.min(), amp_all.max(), 30)
        y_grid = np.linspace(per_bp_all.min(), per_bp_all.max(), 30)
        Xg, Yg = np.meshgrid(x_grid, y_grid)
        Zg = -(w0 + w1 * Xg + w2 * Yg) / w3
        ax.plot_surface(Xg, Yg, Zg, alpha=0.15, color="blue", edgecolor="none")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="None",
               markerfacecolor="orange", markersize=6, label="Song = 1"),
        Line2D([], [], marker="o", linestyle="None",
               markerfacecolor="black",  markersize=6, label="Song = 0"),
        Line2D([], [], color="blue", lw=3, label="Decision plane")
    ], frameon=False, loc="upper left")
    plt.tight_layout(); plt.show()

    # ── 2-D scatter-plots ──────────────────────────────────────
    scatter2d(
        amp_all, per_bp_all,
        "z-log amplitude", f"Periodicity {low_mod}–{high_mod} Hz",
        "Amplitude vs. band-pass periodicity"
    )
    scatter2d(
        amp_all, ent_all,
        "z-log amplitude", "Wiener entropy",
        "Amplitude vs. entropy"
    )
    scatter2d(
        per_bp_all, ent_all,
        f"Periodicity {low_mod}–{high_mod} Hz", "Wiener entropy",
        "Band-pass periodicity vs. entropy"
    )
    scatter2d(
        per_full_all, ent_all,
        "Full-band periodicity", "Wiener entropy",
        "Full-band periodicity vs. entropy"
    )

    return amp_all, per_bp_all, ent_all


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    folder_path = (
        "/Users/mirandahulsey-vincent/Documents/allPythonCode/"
        "BYOD_class/data_inputs/sample_GV_groundtruth_npzs"
    )

    batch_amp_periodicity_entropy(
        folder=folder_path,
        plot_each_spec=True,
        low_mod=10,
        high_mod=30,
        smoothing_sigma=0.7,
        power_threshold=0.2,
        segment_duration=10,
    )
