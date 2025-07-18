#!/usr/bin/env python
# compare_and_save_mismatches.py
# ==============================
"""
Compare your song-detector JSON against George’s and (optionally) visualise the
files where they disagree.

Outputs
-------
1.  <output_json_path>  ·  JSON list of mismatches, each containing:
        file_name, file_path, duration_seconds,
        your_pipeline (bool), george_pipeline (bool),
        detected_song_times              (your spans, s),
        george_detected_song_times       (George spans, s)

2.  <qc_output_dir>/mismatch_panel_*.png    (if --viz / visualize_mistmatch=True)
    Spectrogram panels with colour-coded overlays:
        • yellow  → your detector flagged song, George did not  (FP)
        • red     → George flagged song, your detector did not  (FN)

3.  <qc_output_dir>/mismatch_QC_key.png     Legend explaining the colours.
"""

from __future__ import annotations
from pathlib import Path
import json, math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import soundfile as sf
from scipy.signal import spectrogram


# ───────────────────────────── compare & save ────────────────────────────────
def save_mismatches_with_segments(
    your_json_path,
    george_json_path,
    output_json_path,
    *,
    visualize_mistmatch: bool = False,     # ← flag requested by user
    qc_output_dir: str | Path | None = None,
    sr_target: int = 44_100,
    rows_per_fig: int = 6,
):
    """
    Compare two detectors and (optionally) create QC visualisations.

    Parameters
    ----------
    visualize_mistmatch
        If True, draw spectrograms of every mismatch.
    qc_output_dir
        Folder for the PNGs. If None, uses <output_json stem>_qc_panels.
    """
    # ── load JSON files ─────────────────────────────────────────────────────
    with open(your_json_path, "r") as f:
        your_data = json.load(f)
    with open(george_json_path, "r") as f:
        george_data = json.load(f)

    your_lookup = {
        rec["file_name"]: {
            "contains_song":        rec["contains_song"],
            "file_path":            rec.get("file_path", "unknown"),
            "duration_seconds":     rec.get("duration_seconds"),
            "detected_song_times":  rec.get("detected_song_times", []),
        }
        for rec in your_data
    }
    george_lookup = {rec["filename"]: rec for rec in george_data}

    # ── build mismatch list ─────────────────────────────────────────────────
    mismatches: list[dict] = []
    for fname, yrec in your_lookup.items():
        grec = george_lookup.get(fname)
        if grec is None:   # not in George’s larger set – ignore
            continue

        y_status = yrec["contains_song"]
        g_status = grec["song_present"]

        if y_status != g_status:
            g_spans_sec = (
                [
                    [seg["onset_ms"] / 1000.0, seg["offset_ms"] / 1000.0]
                    for seg in grec.get("segments", [])
                ]
                if g_status
                else []
            )

            mismatches.append(
                {
                    "file_name":                   fname,
                    "file_path":                   yrec["file_path"],
                    "duration_seconds":            yrec["duration_seconds"],
                    "your_pipeline":               y_status,
                    "george_pipeline":             g_status,
                    "detected_song_times":         yrec["detected_song_times"],
                    "george_detected_song_times":  g_spans_sec,
                }
            )

    # ── save mismatch JSON ─────────────────────────────────────────────────
    out_path = Path(output_json_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(mismatches, f, indent=2)

    print(
        f"✅ compared {len(your_lookup)} files; "
        f"found {len(mismatches)} mismatches.\n"
        f"💾 mismatch JSON saved → {out_path}"
    )

    # ── optional visualisation ──────────────────────────────────────────────
    if visualize_mistmatch and mismatches:
        if qc_output_dir is None:
            qc_output_dir = out_path.with_suffix("").parent / "qc_panels"
        build_mismatch_panels(
            mismatch_json=out_path,
            output_dir=qc_output_dir,
            sr_target=sr_target,
            rows_per_fig=rows_per_fig,
        )

    return out_path


# ─────────────────────────── QC panel generator ─────────────────────────────
ROW_DUR                 = 10.0                 # seconds per horizontal lane
SPEC_NPERSEG, SPEC_NOVERLAP = 1024, 512
SPEC_VMIN, SPEC_VMAX    = -90, -20             # dB colour range

_YELLOW = dict(color="yellow", alpha=0.40, zorder=6)   # FP    (your only)
_RED    = dict(color="red",    alpha=0.35, zorder=6)   # FN    (George only)
_BOUND  = dict(color="red",    ls="--",   lw=1.2, zorder=8)   # file edge


def build_mismatch_panels(
    mismatch_json: str | Path,
    output_dir: str | Path,
    *,
    sr_target: int = 44_100,
    rows_per_fig: int = 6,
):
    """
    Draw spectrogram QC panels with:
        • yellow overlays where only *your* detector finds song
        • red overlays where only George finds song
        • red dashed lines at every recording boundary
    The layout packs recordings back‑to‑back across rows of ROW_DUR seconds.
    """
    mismatch_json = Path(mismatch_json)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mismatch_json.open() as f:
        recs = json.load(f)
    if not recs:
        print("[WARN] No mismatches to plot.")
        return

    # helper caches ----------------------------------------------------------
    _audio_cache: dict[str, tuple[np.ndarray, int]] = {}  # wav_path → (audio, sr)

    def _load_audio(path: Path) -> tuple[np.ndarray, int]:
        """Read WAV (mono) and cache it."""
        if path.as_posix() not in _audio_cache:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            _audio_cache[path.as_posix()] = (audio, sr)
        return _audio_cache[path.as_posix()]

    # low‑level drawing helpers ---------------------------------------------
    def _draw_segment(ax, audio_seg, sr, x0):
        f, t, S = spectrogram(
            audio_seg,
            fs=sr,
            nperseg=SPEC_NPERSEG,
            noverlap=SPEC_NOVERLAP,
            scaling="spectrum",
            mode="magnitude",
        )
        S_db = 20 * np.log10(S + 1e-12)
        ax.pcolormesh(
            x0 + t, f, S_db,
            shading="auto", cmap="gray_r",
            vmin=SPEC_VMIN, vmax=SPEC_VMAX,
        )

    def _overlay(ax, spans, shift, x0, style):
        for s, e in spans:
            s_rel = s - shift
            e_rel = e - shift
            if e_rel <= 0 or s_rel >= SEG_DUR:
                continue
            ax.axvspan(x0 + max(s_rel, 0),
                       x0 + min(e_rel, SEG_DUR),
                       **style)

    # main panel engine ------------------------------------------------------
    SEG_DUR = ROW_DUR                     # alias for clarity
    fig_idx, rec_idx = 1, 0
    rec_progress = 0.0                    # seconds consumed within current rec

    while rec_idx < len(recs):
        # -------- new figure --------
        fig, axes = plt.subplots(
            rows_per_fig, 1,
            figsize=(10, 2 * rows_per_fig),
            sharex=True, constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for ax in axes:
            if rec_idx >= len(recs):
                ax.axis("off")
                continue

            offset = 0.0          # x position within this ROW_DUR lane
            titles = []           # recordings that appear in this axis

            while offset < ROW_DUR and rec_idx < len(recs):
                rec       = recs[rec_idx]
                wav_path  = Path(rec["file_path"])

                # ---- load / cache audio once per recording ----
                if rec_progress == 0.0:
                    audio_full, sr_file = _load_audio(wav_path)
                    sr_current = sr_file if sr_file != sr_target else sr_target
                else:
                    audio_full, sr_current = _load_audio(wav_path)

                # total duration from WAV – guarantees full coverage
                dur_total = len(audio_full) / sr_current
                remaining = dur_total - rec_progress
                seg_dur   = min(remaining, ROW_DUR - offset)

                # slice audio for current fragment
                start_idx = int(round(rec_progress * sr_current))
                end_idx   = start_idx + int(round(seg_dur * sr_current))
                audio_seg = audio_full[start_idx:end_idx]

                # draw spectrogram fragment
                _draw_segment(ax, audio_seg, sr_current, x0=offset)

                # overlays --------------------------------------------------
                if rec["george_pipeline"] and not rec["your_pipeline"]:
                    _overlay(ax,
                             rec["george_detected_song_times"],
                             shift=rec_progress,
                             x0=offset,
                             style=_RED)
                elif rec["your_pipeline"] and not rec["george_pipeline"]:
                    _overlay(ax,
                             rec["detected_song_times"],
                             shift=rec_progress,
                             x0=offset,
                             style=_YELLOW)

                rec_progress += seg_dur
                offset       += seg_dur

                finished = abs(rec_progress - dur_total) < 1e-6
                if finished:
                    ax.axvline(offset, **_BOUND)   # boundary line
                    titles.append(wav_path.name)
                    rec_idx     += 1
                    rec_progress = 0.0
                # else the same recording will continue in next lane slice

            # axis cosmetics -------------------------------------------------
            ax.set_xlim(0, ROW_DUR)
            ax.set_ylim(0, 10_000)
            ax.set_yticks([0, 2500, 5000, 7500, 10_000])
            ax.set_ylabel("Freq [Hz]")
            if titles:
                ax.set_title("   ⟡   ".join(titles), fontsize=9, pad=4)

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Mismatch QC panel {fig_idx}", fontsize=11)
        out_png = output_dir / f"mismatch_panel_{fig_idx:03d}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[INFO] saved {out_png.name}")
        fig_idx += 1

    _save_legend(output_dir)  # (same as before – unchanged)



def _save_legend(out_dir: Path):
    """PNG legend explaining yellow and red overlays."""
    key_path = out_dir / "mismatch_QC_key.png"
    if key_path.exists():
        return                      # already written in previous run

    fig, ax = plt.subplots(figsize=(3, 1.5))
    ax.axis("off")
    handles = [
        mpatches.Patch(**_YELLOW, label="Song detected by song feature detector only"),
        mpatches.Patch(**_RED,    label="Song detected by TweetyNET song detector only"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center")
    fig.savefig(key_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]   saved legend → {key_path.name}")


# ───────────────────────────── CLI wrapper ──────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Create mismatch JSON and optional QC PNGs."
    )
    p.add_argument("--yours",   required=True, type=Path,
                   help="Path to your *_detected_song_intervals.json*")
    p.add_argument("--georges", required=True, type=Path,
                   help="Path to George’s song-detection JSON")
    p.add_argument("--out",     required=True, type=Path,
                   help="Destination path for mismatch JSON")
    p.add_argument("--viz",     action="store_true", dest="visualize_mistmatch",
                   help="Generate spectrogram QC panels")
    p.add_argument("--qc_dir",  type=Path, default=None,
                   help="Directory for QC PNGs (default: *_qc_panels)")
    p.add_argument("--sr",      type=int, default=44_100,
                   help="Spectrogram sample-rate (overridden if WAV differs)")
    p.add_argument("--rows",    type=int, default=6, dest="rows_per_fig",
                   help="Spectrogram rows per PNG")
    args = p.parse_args()

    save_mismatches_with_segments(
        args.yours,
        args.georges,
        args.out,
        visualize_mistmatch=args.visualize_mistmatch,
        qc_output_dir=args.qc_dir,
        sr_target=args.sr,
        rows_per_fig=args.rows_per_fig,
    )



"""
your_json_path = "/Volumes/my_own_SSD/USA5288/0/0_detected_song_intervals.json"
george_json_path = "/Volumes/my_own_SSD/song_detection.json"
output_json_path = "/Volumes/my_own_SSD/USA5288/0/USA5288_0_mismatches.json"

save_mismatches_with_segments(
    your_json_path,
    george_json_path,
    output_json_path,
    visualize_mistmatch= True
)

print(f"{len(interval_files)} interval files created")
"""
