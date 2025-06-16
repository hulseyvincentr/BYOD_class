# -*- coding: utf-8 -*-

# segment_max_scatter.py
# ======================
"""
(1) Divide each recording into song vs. not-song segments using the
    ground-truth mask.
(2) For every segment, record the maximum of four features:
        • z_log_amp_band_smoothed
        • periodicity_bandpass
        • wiener_entropy_log
        • shannon_entropy_bits
(3) Scatter-plot those maxima, colour-coded by segment label.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression


# ──────────────────────────────────────────────────────────────────────────
def _contiguous_segments(mask: np.ndarray):
    """
    Yield (start_idx, end_idx, label) for every run of identical booleans
    in *mask*.  `end_idx` is inclusive.  Example: [0,0,1,1,1,0] →
        (0,1,False), (2,4,True), (5,5,False)
    """
    if mask.size == 0:
        return
    # indices where the value changes
    change = np.nonzero(np.diff(mask.astype(int)))[0] + 1
    # segment start positions (prepend 0, append len)
    seg_starts = np.concatenate(([0], change))
    seg_ends   = np.concatenate((change - 1, [len(mask) - 1]))
    for s, e in zip(seg_starts, seg_ends):
        yield s, e, bool(mask[s])    # label == mask value


def collect_segment_maxima(json_path: str | Path):
    """
    Returns four NumPy arrays (amp_max, per_max, wien_max, shan_max)
    and a boolean array 'is_song' (one entry per segment).
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    amp_max, per_max, wien_max, shan_max, labels = ([] for _ in range(5))

    for entry in data:
        m = entry["metrics"]

        amp   = np.asarray(m["z_log_amp_band_smoothed"])
        per   = np.asarray(m["periodicity_bandpass"])
        wien  = np.asarray(m["wiener_entropy_log"])
        shan  = np.asarray(m["shannon_entropy_bits"])
        mask  = np.asarray(m["groundtruth_songs"], dtype=bool)

        # truncate to common length (paranoia)
        n = min(len(amp), len(per), len(wien), len(shan), len(mask))
        amp, per, wien, shan, mask = (
            arr[:n] for arr in (amp, per, wien, shan, mask)
        )

        # iterate over contiguous song / not-song runs
        for s, e, is_song in _contiguous_segments(mask):
            amp_max .append(amp [s:e+1].max())
            per_max .append(per [s:e+1].max())
            wien_max.append(wien[s:e+1].max())
            shan_max.append(shan[s:e+1].max())
            labels  .append(is_song)

    return (
        np.asarray(amp_max),
        np.asarray(per_max),
        np.asarray(wien_max),
        np.asarray(shan_max),
        np.asarray(labels,   dtype=bool),
    )


# ──────────────────────────────────────────────────────────────────────────
def scatter2d(x, y, xlabel, ylabel, title, colours):
    """
    Generic 2-D scatter with a logistic-regression boundary and dynamic limits.
    """
    clf = LogisticRegression(solver="lbfgs").fit(
        np.column_stack([x, y]), (colours == "orange").astype(int)
    )
    w0, w1, w2 = clf.intercept_[0], *clf.coef_[0]
    x_line = np.linspace(x.min(), x.max(), 250)
    y_line = -(w0 + w1 * x_line) / w2

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, c=colours, s=20, alpha=0.7)
    plt.plot(x_line, y_line, "b--", lw=2, label="Boundary")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\n"
              f"y = -({w0:+.3f} + {w1:+.3f}·x) / {w2:+.3f}",
              fontsize="medium")

    # auto-pad entropy axes a little (if label contains 'entropy')
    if "entropy" in ylabel.lower():
        ypad = 0.05 * (y.max() - y.min() or 1)
        plt.ylim(y.min() - ypad, y.max() + ypad)

    plt.legend(handles=[
        Line2D([], [], marker="o", linestyle="None",
               markerfacecolor="orange", markersize=6, label="Song segment"),
        Line2D([], [], marker="o", linestyle="None",
               markerfacecolor="black",  markersize=6, label="Not-song segment"),
        Line2D([], [], color="blue", ls="--", lw=2, label="Boundary"),
    ], frameon=False, fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_segment_maxima(json_path: str | Path):
    amp, per, wien, shan, is_song = collect_segment_maxima(json_path)
    colours = np.where(is_song, "orange", "black")

    scatter2d(amp, per,  "max z-log amplitude",
              "max band-pass periodicity",
              "Segment maxima: amp vs. periodicity", colours)

    scatter2d(amp, wien, "max z-log amplitude",
              "max Wiener entropy", 
              "Segment maxima: amp vs. Wiener entropy", colours)

    scatter2d(amp, shan, "max z-log amplitude",
              "max Shannon entropy",
              "Segment maxima: amp vs. Shannon entropy", colours)


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    JSON_FILE = (
        "/Users/mirandahulsey-vincent/Documents/allPythonCode/"
        "BYOD_class/data_inputs/sample_GV_groundtruth_npzs_features.json"
    )
    plot_segment_maxima(JSON_FILE)
