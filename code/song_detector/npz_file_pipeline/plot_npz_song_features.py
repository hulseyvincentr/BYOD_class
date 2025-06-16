# plot_json_features.py
# ======================
"""
Load a pre-computed feature .json (from batch_extract), plot all acoustic
features (amplitude, periodicity, entropy) from each file, and colour-code
points by ground-truth song / non-song labels.

Now draws the SAME plots for *both* Wiener entropy (log) and Shannon entropy
(bits).
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
from sklearn.linear_model import LogisticRegression


def plot_json_feature_scatter(json_path: str | Path):
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())

    # storage
    amp_vecs, per_bp_vecs, per_full_vecs = ([] for _ in range(3))
    wien_vecs, shan_vecs, song_masks    = ([] for _ in range(3))

    # -------------------- collect + truncate --------------------
    for entry in data:
        m = entry["metrics"]

        amp        = np.array(m["z_log_amp_band_smoothed"])
        per_bp     = np.array(m["periodicity_bandpass"])
        per_full   = np.array(m["periodicity_full"])
        wien       = np.array(m["wiener_entropy_log"])
        shan       = np.array(m["shannon_entropy_bits"])
        song_mask  = np.array(m["groundtruth_songs"], dtype=bool)

        n = min(len(amp), len(per_bp), len(per_full), len(wien),
                len(shan), len(song_mask))

        amp_vecs .append(amp     [:n])
        per_bp_vecs.append(per_bp [:n])
        per_full_vecs.append(per_full[:n])
        wien_vecs .append(wien    [:n])
        shan_vecs .append(shan    [:n])
        song_masks.append(song_mask[:n])

    # concatenate to single long vectors
    amp_all      = np.concatenate(amp_vecs)
    per_bp_all   = np.concatenate(per_bp_vecs)
    per_full_all = np.concatenate(per_full_vecs)
    wien_all     = np.concatenate(wien_vecs)
    shan_all     = np.concatenate(shan_vecs)
    songs_all    = np.concatenate(song_masks).astype(int)
    colours      = np.where(songs_all, "orange", "black")

   # -------------------- helper: 2-D scatter --------------------
    def scatter2d(x, y, xlabel, ylabel, title):
        X  = np.column_stack([x, y])
        clf = LogisticRegression(solver="lbfgs").fit(X, songs_all)
        w0, w1, w2 = clf.intercept_[0], *clf.coef_[0]
    
        x_line = np.linspace(x.min(), x.max(), 250)
        y_line = -(w0 + w1 * x_line) / w2
    
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y, c=colours, s=7, alpha=0.6)
        plt.plot(x_line, y_line, "b--", lw=2, label="Boundary")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}\n"
                  f"y = -({w0:.2f} + {w1:.2f}x) / {w2:.2f}", fontsize="medium")
    
        # ── dynamic entropy limits ─────────────────────────────────
        if "entropy" in ylabel.lower():
            y_min, y_max = y.min(), y.max()
            pad = 0.05 * (y_max - y_min or 1)   # small 5 % cushion
            plt.ylim(y_min - pad, y_max + pad)
    
        plt.legend(handles=[
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="orange",
                   markersize=6, label="Song = 1"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="black",
                   markersize=6, label="Song = 0"),
            Line2D([], [], color="blue", ls="--", lw=2, label="Boundary")
        ], frameon=False, fontsize="small")
        plt.tight_layout()
        plt.show()


    # ───── helper: 3-D scatter + decision plane + title with equation ─────
    def scatter3d(ent_all, ent_label):
        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(amp_all, per_bp_all, ent_all, c=colours, s=7, alpha=0.6)
    
        # fit logistic model in 3-D
        X3   = np.column_stack([amp_all, per_bp_all, ent_all])
        clf3 = LogisticRegression(solver="lbfgs").fit(X3, songs_all)
        w0, w1, w2, w3 = clf3.intercept_[0], *clf3.coef_[0]
    
        # build decision plane (z = −(w0 + w1·x + w2·y)/w3)
        if abs(w3) > 1e-6:
            xg = np.linspace(amp_all.min(),    amp_all.max(),    30)
            yg = np.linspace(per_bp_all.min(), per_bp_all.max(), 30)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = -(w0 + w1 * Xg + w2 * Yg) / w3
            ax.plot_surface(Xg, Yg, Zg, alpha=0.15,
                            color="blue", edgecolor="none")
    
        # ── nice-looking equation string ────────────────────────────
        eqn = f"z = -({w0:+.3f} + {w1:+.3f}·x + {w2:+.3f}·y) / {w3:+.3f}"
    
        # labels + title with equation
        ax.set_xlabel("z-log amplitude")
        ax.set_ylabel("Band-pass periodicity")
        ax.set_zlabel(ent_label)
        ax.set_title(f"3-D: Amplitude × Periodicity × {ent_label}\n{eqn}",
                     fontsize="medium")
    
        # legend (unchanged)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="orange",
                   markersize=6, label="Song = 1"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="black",
                   markersize=6, label="Song = 0"),
            Line2D([], [], color="blue", lw=3, label="Decision plane")
        ], frameon=False, loc="upper left")
    
        plt.tight_layout()
        plt.show()


        
    # -------------------- plot helper for one entropy type -------
    def make_all_plots(ent_all, ent_label):
        scatter3d(ent_all, ent_label)

        scatter2d(amp_all,       ent_all, "z-log amplitude",
                  ent_label,     f"Amp vs. {ent_label}")
        scatter2d(per_bp_all,    ent_all, "Band-pass periodicity",
                  ent_label,     f"Periodicity vs. {ent_label}")
        scatter2d(per_full_all,  ent_all, "Full-band periodicity",
                  ent_label,     f"Full-band Periodicity vs. {ent_label}")

    # -------------------- generate both sets --------------------
    make_all_plots(wien_all,  "Wiener entropy")
    make_all_plots(shan_all,  "Shannon entropy (bits)")


if __name__ == "__main__":
    json_path = (
        "/Users/mirandahulsey-vincent/Documents/"
        "allPythonCode/BYOD_class/data_inputs/"
        "sample_GV_groundtruth_npzs_features.json"
    )
    plot_json_feature_scatter(json_path)
