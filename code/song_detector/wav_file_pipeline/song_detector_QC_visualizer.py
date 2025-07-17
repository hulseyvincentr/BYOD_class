# -*- coding: utf-8 -*-

#!/usr/bin/env python
"""
song_detector_QC_visualizer.py
==============================
Visual QC for *detected* song intervals.

Input  : <something>_detected_song_intervals.json
Output : PNG panels in <output_dir>/, plus a tiny legend PNG.

• One figure → up to `rows_per_fig` stacked timeline rows (default 6)
• One row    → 10‑second strip (ROW_DUR) that keeps wrapping until the
               entire recording is shown, red dashed line at the true end.
• Yellow span  = detected_song_times from the JSON
• Red dashed    = file boundary
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
from scipy.signal import spectrogram

ROW_DUR = 10.0  # seconds shown per timeline row


# ───────────────────────── helpers ──────────────────────────
def _draw_spec(ax, audio, sr, offset, dur, *, vmin=-90, vmax=-20):
    if dur <= 0:
        return
    f, t, S = spectrogram(
        audio[: int(round(dur * sr))],
        fs=sr, nperseg=1024, noverlap=512, scaling="spectrum", mode="magnitude"
    )
    S_db = 20 * np.log10(S + 1e-12)
    ax.pcolormesh(offset + t, f, S_db,
                  shading="auto", cmap="gray_r", vmin=vmin, vmax=vmax)


def _highlight(ax, spans: List[List[float]], pane_offset, seg_dur):
    for s, e in spans:
        if e <= 0 or s >= seg_dur:
            continue
        ax.axvspan(pane_offset + max(s, 0),
                   pane_offset + min(e, seg_dur),
                   color="yellow", alpha=0.45, zorder=5)


def _save_legend(out_dir: Path):
    fig, ax = plt.subplots(figsize=(3, 1.8))
    ax.axis("off")
    handles = [
        mpatches.Patch(color="yellow", alpha=0.45, label="Detected song span"),
        plt.Line2D([0], [0], color="red", lw=2, ls="--", label="File boundary"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center")
    fig.savefig(out_dir / "song_detector_QC_key.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_dir/'song_detector_QC_key.png'}")


# ───────────────────── main builder ─────────────────────
def build_QC_panels(json_path, output_dir, *, sr=44100, rows_per_fig=6):
    json_path, output_dir = Path(json_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open() as f:
        recs = json.load(f)

    # keep only recordings flagged as containing song
    recs = [r for r in recs if r["contains_song"]]
    if not recs:
        print("[WARN] No recordings marked as containing song.")
        return

    legend_done = False
    rec_idx, rec_prog, fig_no = 0, 0.0, 1

    while rec_idx < len(recs):
        fig, axes = plt.subplots(rows_per_fig, 1,
                                 figsize=(10, 2 * rows_per_fig),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)

        for ax in axes:
            if rec_idx >= len(recs):
                ax.axis("off")
                continue

            offset, titles = 0.0, []
            while offset < ROW_DUR and rec_idx < len(recs):
                rec = recs[rec_idx]
                wav_path = Path(rec["file_path"])

                # load the .wav only once per recording
                if rec_prog == 0.0:
                    audio_full, sr_file = sf.read(wav_path)
                    if audio_full.ndim > 1:
                        audio_full = audio_full.mean(axis=1)
                    if sr_file != sr:
                        print(f"[WARN] {wav_path.name}: "
                              f"sr {sr_file} != {sr}; using {sr_file}")
                        sr = sr_file

                remain = rec["duration_seconds"] - rec_prog
                seg_dur = min(remain, ROW_DUR - offset)

                # slice current segment
                start_sample = int(round(rec_prog * sr))
                seg_audio = audio_full[start_sample:
                                       start_sample + int(round(seg_dur * sr))]
                _draw_spec(ax, seg_audio, sr, offset, seg_dur)

                # highlight detected song spans
                shift = rec_prog
                _highlight(ax,
                           [(s - shift, e - shift)
                            for s, e in rec["detected_song_times"]],
                           offset, seg_dur)

                rec_prog += seg_dur
                offset += seg_dur

                finished = abs(rec_prog - rec["duration_seconds"]) < 1e-6
                if finished:
                    ax.axvline(offset, color="red", ls="--",
                               lw=1.0, zorder=6)
                    titles.append(wav_path.name)
                    rec_idx += 1
                    rec_prog = 0.0

            # cosmetics for this row
            ax.set_xlim(0, ROW_DUR)
            ax.set_ylim(0, 10_000)
            ax.set_yticks([0, 2500, 5000, 7500, 10_000])
            ax.set_ylabel("Freq [Hz]")
            ax.tick_params(labelsize=8)
            if titles:
                ax.set_title(" + ".join(titles), fontsize=9, pad=4)

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Song‑detector QC panel {fig_no}", fontsize=12)

        out_png = output_dir / f"song_detector_QC_panel_{fig_no:03d}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {out_png}")

        if not legend_done:
            _save_legend(output_dir)
            legend_done = True

        fig_no += 1


# ────────────────── CLI convenience ──────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Visual QC for detected song intervals."
    )
    p.add_argument("--json_path", required=True,
                   help="Path to *_detected_song_intervals.json")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save QC PNGs")
    p.add_argument("--sr", default=44100, type=int,
                   help="Sample‑rate for spectrograms (no resampling).")
    p.add_argument("--rows", default=6, type=int, dest="rows_per_fig",
                   help="Rows per figure.")
    args = p.parse_args()

    build_QC_panels(args.json_path, args.output_dir,
                    sr=args.sr, rows_per_fig=args.rows_per_fig)

# =============================================================================
# from song_detector_QC_visualizer import build_QC_panels
# json_path = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_clean/data_inputs/USA5510_unsegmented_songs/55_subsample_detected_song_intervals.json"
# output_dir = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_clean/data_inputs/USA5510_unsegmented_songs"
# build_QC_panels(json_path, output_dir,sr=44100,rows_per_fig=6)
# 
# =============================================================================
