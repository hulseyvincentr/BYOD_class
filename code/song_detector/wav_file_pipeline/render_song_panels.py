# -*- coding: utf-8 -*-
# render_song_panels.py
"""
Render stacked spectrogram "panels" from WAVs described in a detector JSON.

JSON schema (yours, auto-detected):
  - filename: str  (e.g., "USA5288_45355.32428022_3_4_9_0_28.wav")
  - song_present: bool
  - segments: list of [start_sec, end_sec] (or list of {"start": s, "end": e})
  - spec_parameters: optional (unused here)

Core features
-------------
- Robust path handling: absolute or relative basenames under wav_dir (recursive search fallback).
- Optional filter `only_song_present=True` to include only entries that clearly contain song.
- Overlays yellow translucent spans for song intervals; optional `pad_before_after_sec`.
- Band-pass filter with Butterworth IIR (filtfilt) before spectrogram.
- Panels are fixed time-length slices; multiple panels per figure (stacked rows).
- Dashed red vertical lines at panel boundaries for visual anchoring.

Dependencies: numpy, scipy, matplotlib, soundfile
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional, List, Tuple

import json
import re
import math

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ──────────────────────────────────────────────────────────────────────────────
# Tunables (feel free to tweak)
# ──────────────────────────────────────────────────────────────────────────────
SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 512
SPEC_VMIN_DB, SPEC_VMAX_DB = -90, -20   # dB scale for display
CMAP = "gray_r"                         # spectrogram colormap
OVERLAY_ALPHA = 0.28                    # yellow overlay transparency
OVERLAY_COLOR = "yellow"

# Regex for WAV detection in strings
_WAV_RE = re.compile(r"\.wav$", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: schema normalization and path resolution
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_wav_path(entry: dict, wav_dir: Path, wav_key: str = "filename") -> Path:
    """
    Return a Path to the .wav for this entry.

    - If entry[wav_key] is absolute, use directly.
    - Else try wav_dir / value, then rglob(value's basename) under wav_dir.
    """
    if wav_key not in entry:
        raise KeyError(f"Expected key '{wav_key}' in entry (keys={list(entry.keys())[:10]}...)")
    v = entry[wav_key]
    if not isinstance(v, str) or not _WAV_RE.search(v):
        raise ValueError(f"Entry['{wav_key}'] should be a string ending in .wav (got: {v!r})")
    p = Path(v)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Absolute WAV not found: {p}")
        return p
    direct = wav_dir / v
    if direct.exists():
        return direct
    hits = list(wav_dir.rglob(Path(v).name))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot resolve '{v}' under {wav_dir}")


def _entry_has_song(entry: dict) -> bool:
    """Robust truthiness for song content."""
    if entry.get("song_present") is True:
        return True
    if entry.get("contains_song") is True:
        return True
    # Fallback to non-empty time windows:
    times = entry.get("segments") or entry.get("detected_song_times") or entry.get("song_times")
    return isinstance(times, list) and len(times) > 0


def _normalize_times(entry: dict) -> List[Tuple[float, float]]:
    """
    Return [(start_sec, end_sec), ...] from any of:
      - entry["segments"]              == [[s, e], ...]  or [{'start': s, 'end': e}, ...]
      - entry["detected_song_times"]   == same formats
      - entry["song_times"]            == same formats
    Assumes seconds.
    """
    raw = entry.get("segments") or entry.get("detected_song_times") or entry.get("song_times") or []
    out: List[Tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            s, e = float(item[0]), float(item[1])
            if e > s:
                out.append((s, e))
        elif isinstance(item, dict):
            if "start" in item and "end" in item:
                s, e = float(item["start"]), float(item["end"])
                if e > s:
                    out.append((s, e))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DSP helpers
# ──────────────────────────────────────────────────────────────────────────────
def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = max(1e-9, min(lowcut / nyq, 0.999999))
    high = max(low + 1e-9, min(highcut / nyq, 0.999999))
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _apply_bandpass(y: np.ndarray, fs: float, low_cut: float, high_cut: float) -> np.ndarray:
    """Zero-phase band-pass."""
    if low_cut is None or high_cut is None or low_cut <= 0 or high_cut <= 0 or high_cut <= low_cut:
        return y
    b, a = _butter_bandpass(low_cut, high_cut, fs, order=4)
    # filtfilt requires finite values
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    return filtfilt(b, a, y, method="gust")


def _slice_audio(y: np.ndarray, start_sec: float, end_sec: float, fs: float) -> np.ndarray:
    i0 = max(0, int(round(start_sec * fs)))
    i1 = min(len(y), int(round(end_sec * fs)))
    return y[i0:i1]


def _compute_spectrogram(y: np.ndarray, fs: float):
    """Return (T, F, S_db) where S_db is dB magnitude."""
    if len(y) == 0:
        # Avoid empty spec
        return np.array([0.0, 1e-3]), np.array([0.0, 1.0]), np.full((2, 2), SPEC_VMIN_DB, float)
    f, t, S = spectrogram(
        y,
        fs=fs,
        nperseg=SPEC_NPERSEG,
        noverlap=SPEC_NOVERLAP,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    # Convert to dB
    S_db = 20.0 * np.log10(np.maximum(S, 1e-12))
    return t, f, S_db


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────
def _draw_overlays(ax, panel_start: float, panel_end: float, windows: List[Tuple[float, float]]):
    """Draw yellow overlays clipped to [panel_start, panel_end]."""
    for s, e in windows:
        if e <= panel_start or s >= panel_end:
            continue
        s_clip = max(s, panel_start)
        e_clip = min(e, panel_end)
        ax.axvspan(
            s_clip - panel_start, e_clip - panel_start,
            color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA, lw=0
        )


def _format_axes(ax, dur: float, fs: float, title: Optional[str] = None):
    ax.set_xlim(0, dur)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    if title:
        ax.set_title(title, fontsize=10)
    # Dashed red lines at boundaries
    ax.axvline(0.0, color="red", ls="--", lw=0.8)
    ax.axvline(dur, color="red", ls="--", lw=0.8)


def _render_panels_for_file(
    wav_path: Path,
    out_dir: Path,
    segment_duration_sec: float,
    panels_per_fig: int,
    low_cut: float,
    high_cut: float,
    song_windows: List[Tuple[float, float]],
    pad_before_after_sec: float,
) -> List[Path]:
    """
    Render stacked spectrogram panels for a single WAV.
    Returns list of written figure paths.
    """
    # Load audio (mono)
    y, fs = sf.read(str(wav_path), always_2d=False)
    if y.ndim > 1:
        # mixdown to mono
        y = np.ascontiguousarray(np.mean(y, axis=1))
    else:
        y = np.ascontiguousarray(y)

    total_dur = len(y) / float(fs)
    # Band-pass once globally for speed (panel slices come from filtered y)
    y_f = _apply_bandpass(y, fs, low_cut, high_cut)

    # Pad overlays
    if pad_before_after_sec and pad_before_after_sec > 0:
        windows = [(max(0.0, s - pad_before_after_sec), min(total_dur, e + pad_before_after_sec))
                   for (s, e) in song_windows]
    else:
        windows = list(song_windows)

    # Panel starts
    n_panels = max(1, int(math.ceil(total_dur / segment_duration_sec)))
    starts = [i * segment_duration_sec for i in range(n_panels)]
    wrote: List[Path] = []

    # Batch panels per figure
    for batch_idx in range(0, n_panels, panels_per_fig):
        batch_starts = starts[batch_idx: batch_idx + panels_per_fig]
        n_rows = len(batch_starts)

        fig_h = max(2.0 * n_rows, 2.0)  # height scales with rows
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, fig_h), squeeze=False)
        axes = axes.ravel()

        for row_i, panel_start in enumerate(batch_starts):
            panel_end = min(total_dur, panel_start + segment_duration_sec)
            panel_dur = panel_end - panel_start

            # Slice filtered audio for spectrogram
            y_seg = _slice_audio(y_f, panel_start, panel_end, fs)
            t, f, S_db = _compute_spectrogram(y_seg, fs)

            ax = axes[row_i]
            # Time origin for each panel is 0..panel_dur
            # map spectrogram t (0..panel_dur) directly
            pcm = ax.pcolormesh(
                t, f, S_db,
                shading="auto", cmap=CMAP, vmin=SPEC_VMIN_DB, vmax=SPEC_VMAX_DB
            )

            # overlays
            _draw_overlays(ax, panel_start, panel_end, windows)
            # formatting
            title = f"{wav_path.name} · {panel_start:0.2f}–{panel_end:0.2f} s"
            _format_axes(ax, panel_dur, fs, title=title)

        # Colorbar on the right for the last axes
        cbar = fig.colorbar(pcm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Power (dB)")

        # Legend patch for overlays
        patch = mpatches.Patch(color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA, label="Detected song")
        axes[0].legend(handles=[patch], loc="upper right", frameon=True, fontsize=9)

        plt.tight_layout()

        # Write figure
        base = wav_path.stem
        out_name = f"{base}_panels_{batch_idx // panels_per_fig:03d}.png"
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        wrote.append(out_path)

    return wrote


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def process_detector_json(
    wav_dir: Union[str, Path],
    detector_json_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    segment_duration_sec: float = 60.0,
    panels_per_fig: int = 10,
    low_cut: float = 500.0,
    high_cut: float = 8000.0,
    only_song_present: bool = True,
    pad_before_after_sec: float = 0.0,
    wav_key: str = "filename",   # your JSON uses "filename"
) -> List[Path]:
    """
    Main entry: read detector JSON and render figures.

    Parameters
    ----------
    wav_dir : str | Path
        Root directory containing WAV files (or subfolders thereof).
    detector_json_path : str | Path
        Path to detector JSON (list of dicts).
    out_dir : str | Path | None
        Output directory for PNGs. Defaults to wav_dir / "panels".
    segment_duration_sec : float
        Duration of each panel in seconds (e.g., 60.0).
    panels_per_fig : int
        Number of stacked panels per figure.
    low_cut, high_cut : float
        Band-pass cutoff frequencies in Hz. If invalid, no filtering is applied.
    only_song_present : bool
        If True, only entries that clearly contain song are rendered.
    pad_before_after_sec : float
        Expand each overlay window by this margin on both sides (seconds).
    wav_key : str
        Field name holding the WAV name/path (default "filename").

    Returns
    -------
    List[Path]
        Paths to written figures.
    """
    wav_dir = Path(wav_dir).expanduser().resolve()
    detector_json_path = Path(detector_json_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else (wav_dir / "panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    with detector_json_path.open() as f:
        items = json.load(f)

    # Optional filter
    if only_song_present:
        items = [x for x in items if _entry_has_song(x)]

    written: List[Path] = []

    for entry in items:
        # Resolve WAV path (skip if not found)
        try:
            wav_path = _resolve_wav_path(entry, wav_dir, wav_key=wav_key)
        except Exception:
            continue

        # Overlay windows (seconds)
        song_windows = _normalize_times(entry)

        # Render and collect written files
        pngs = _render_panels_for_file(
            wav_path=wav_path,
            out_dir=out_dir,
            segment_duration_sec=segment_duration_sec,
            panels_per_fig=panels_per_fig,
            low_cut=low_cut,
            high_cut=high_cut,
            song_windows=song_windows,
            pad_before_after_sec=pad_before_after_sec,
        )
        written.extend(pngs)

    return written


"""
from pathlib import Path
from render_song_panels import process_detector_json

wav_dir       = Path("/Volumes/ROSE1-SSD/USA5288/")
detector_json = Path("/Volumes/ROSE1-SSD/USA5288/USA5288_song_detection.json")
out_dir       = Path("/Volumes/ROSE1-SSD/USA5288/panels")

written = process_detector_json(
    wav_dir=wav_dir,
    detector_json_path=detector_json,
    out_dir=out_dir,
    segment_duration_sec=60,
    panels_per_fig=10,
    low_cut=700,
    high_cut=7000,
    only_song_present=True,
    pad_before_after_sec=0.0,
    wav_key="filename",   # explicit, matches your JSON
)
print(f"Wrote {len(written)} figure(s). First few:", written[:3])

"""