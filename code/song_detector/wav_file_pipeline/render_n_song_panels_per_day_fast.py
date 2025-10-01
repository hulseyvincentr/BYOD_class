# render_n_song_panels_per_day_fast.py
# Generate per-day stacks of n spectrogram panels (first or random picks).
# - Gaussian nperseg=2048, hop=119, white background (percentile/gamma shaped)
# - Fixed y-lims 0..10 kHz
# - Filename above each panel in red; only bottom panel shows x-axis
# - Keeps rows with song_present==True even if 'segments' is empty
# - Fast basename index (case-insensitive)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable
import collections, re, json, math, random

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

# ── Spectrogram/display tunables (match your preferred look) ─────────────────
SPEC_NPERSEG   = 2048
SPEC_NOOVERLAP = SPEC_NPERSEG - 119             # hop = 119 samples
SPEC_WINDOW    = windows.gaussian(SPEC_NPERSEG, std=SPEC_NPERSEG/8)

Y_MIN = 0.0
Y_MAX = 10_000.0

# Robust display mapping → white background, darker for louder sound
PCTL_LO = 12.0    # raise = whiter bg (10–20 typical)
PCTL_HI = 99.5
GAMMA   = 0.75    # <1 brightens background; >1 darkens

CMAP = "gray"     # 0=black, 1=white (we'll invert in mapping to make loud darker)

# Labels
TITLE_COLOR   = "red"
TITLE_FONTSZ  = 8
OVERLAY_COLOR = "yellow"   # (not used here, kept for parity)
OVERLAY_ALPHA = 0.28

# ── Filename parsing: "USA5288_45355.33299256_3_4_9_14_59.wav" ───────────────
#  - Group(1) animal ID, Group(2) Excel serial (float-like string)
_NAME_RE = re.compile(r"^([A-Za-z0-9]+)_(\d+(?:\.\d+)?)_", re.IGNORECASE)

def _parse_name(basename: str) -> Optional[Tuple[str, float]]:
    m = _NAME_RE.match(basename)
    if not m: return None
    animal = m.group(1)
    serial = float(m.group(2))
    return animal, serial

# Excel serial date → date (no timezone). Excel's day 0 = 1899-12-30
from datetime import datetime, timedelta
def _excel_serial_to_date(serial: float):
    origin = datetime(1899, 12, 30)  # Excel system
    return (origin + timedelta(days=float(serial))).date()

# ── JSON helpers ─────────────────────────────────────────────────────────────
def _normalize_times(e: dict) -> List[Tuple[float, float]]:
    raw = e.get("segments") or e.get("detected_song_times") or e.get("song_times") or []
    out: List[Tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            s, d = float(item[0]), float(item[1])
            if d > s: out.append((s, d))
        elif isinstance(item, dict) and "start" in item and "end" in item:
            s, d = float(item["start"]), float(item["end"])
            if d > s: out.append((s, d))
    return out

def _get_wav_basename(e: dict, wav_key: str) -> Optional[str]:
    v = e.get(wav_key)
    if not isinstance(v, str): return None
    return Path(v).name

# ── WAV indexing (once; case-insensitive filenames) ──────────────────────────
def _index_wavs(wav_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    # Index common extensions (lower/upper)
    for ext in ("*.wav", "*.WAV"):
        for p in wav_dir.rglob(ext):
            idx[p.name.lower()] = p
    return idx

def _resolve_path_by_basename(basename: str, index: Dict[str, Path]) -> Optional[Path]:
    return index.get(basename.lower())

# ── Spectrogram helper: robust percentiles → white bg, loud darker ───────────
def _spec_disp(y: np.ndarray, fs: float):
    if y.size == 0:
        # Minimal image
        f = np.array([0.0, 1.0], float)
        t = np.array([0.0, 1e-3], float)
        S_disp = np.ones((2, 2), float)  # white
        return t, f, S_disp

    f, t, S = spectrogram(
        y, fs=fs,
        window=SPEC_WINDOW,
        nperseg=SPEC_NPERSEG,
        noverlap=SPEC_NOOVERLAP,
        detrend=False,
        scaling="spectrum",
    )
    S_db = 10.0 * np.log10(S + np.finfo(float).eps)

    lo = np.percentile(S_db, PCTL_LO)
    hi = np.percentile(S_db, PCTL_HI)
    if hi <= lo:
        lo, hi = S_db.min(), S_db.max() + 1e-6

    S_norm = np.clip((S_db - lo) / (hi - lo), 0.0, 1.0)
    # invert + gamma: background→white, loud→darker
    S_disp = (1.0 - S_norm) ** GAMMA
    return t, f, S_disp

# ── Panel render (single subplot) ────────────────────────────────────────────
def _render_panel(ax, y: np.ndarray, fs: float, animal: str, basename: str, start_s: float, end_s: float):
    t, f, S_disp = _spec_disp(y, fs)
    # Crop vertically 0..10 kHz
    m = (f >= Y_MIN) & (f <= Y_MAX)
    if not np.any(m): m = slice(None)
    f_view = f[m]
    S_view = S_disp[m, :]

    ax.imshow(
        S_view, origin="lower", aspect="auto", interpolation="nearest",
        extent=(0.0, (end_s - start_s), float(f_view[0]), float(f_view[-1])),
        cmap=CMAP, vmin=0.0, vmax=1.0
    )
    ax.set_facecolor("white")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(0.0, (end_s - start_s))
    ax.margins(x=0)
    # Filename title (red) above the panel
    ax.set_title(basename, color=TITLE_COLOR, fontsize=TITLE_FONTSZ, pad=2)
    ax.set_ylabel("Freq (Hz)")

# ── Load a segment quickly (mono) ────────────────────────────────────────────
def _load_segment(path: Path, start_s: float, end_s: float) -> Tuple[np.ndarray, float]:
    """Read [start_s, end_s) from WAV. Falls back to full read if needed."""
    # soundfile supports seeking by frames; read only what we need
    info = sf.info(str(path))
    fs = float(info.samplerate)
    i0 = max(0, int(round(start_s * fs)))
    i1 = max(i0, int(round(end_s * fs)))
    frames = i1 - i0
    if frames <= 0:
        return np.zeros(0, dtype=np.float64), fs
    with sf.SoundFile(str(path), "r") as f:
        f.seek(i0)
        y = f.read(frames, dtype="float64", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float64), fs

# ── Public API ────────────────────────────────────────────────────────────────
def render_n_song_panels_per_day(
    wav_dir: Path,
    detector_json_path: Path,
    out_dir: Path,
    n_per_day: int,
    selection: str = "first",       # or "random"
    seed: int = 42,
    segment_pad_sec: float = 0.7,   # pad around first segment (if present)
    min_panel_dur_sec: float = 4.0, # enforce minimum panel duration
    low_cut: Optional[float] = None,  # kept for signature parity (unused here)
    high_cut: Optional[float] = None, # we rely on the display band 0..10k
    wav_key: str = "filename",
    only_song_present: bool = True,
    dpi: int = 300,
    verbose: bool = True,
) -> List[Path]:

    wav_dir = Path(wav_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON
    rows = json.load(open(detector_json_path, "r"))
    if verbose:
        print(f"[json] rows={len(rows)} from {detector_json_path}")

    # Build index once (basenames → path)
    idx = _index_wavs(wav_dir)
    if verbose:
        print(f"[index] Found {len(idx)} wav files; indexed {len(idx)} names (case-insensitive)")

    # Bucket by day
    day_buckets: Dict[object, List[dict]] = collections.defaultdict(list)
    kept = 0
    missing = 0
    sample_missing: List[str] = []

    for e in rows:
        basename = _get_wav_basename(e, wav_key)
        if not basename:
            continue

        segs = _normalize_times(e)
        # IMPORTANT: keep if song_present True OR segments non-empty
        has_positive = (e.get("song_present") is True) or bool(segs)
        if only_song_present and not has_positive:
            continue

        p = _resolve_path_by_basename(basename, idx)
        if p is None:
            missing += 1
            if len(sample_missing) < 5:
                sample_missing.append(basename)
            continue

        parsed = _parse_name(basename)
        if not parsed:
            continue
        animal, serial = parsed
        day = _excel_serial_to_date(serial)

        e2 = dict(e)
        e2["_path"] = p
        e2["_basename"] = basename
        e2["_animal"] = animal
        e2["_serial"] = serial
        e2["_segments"] = segs  # may be empty; we'll fallback when rendering
        day_buckets[day].append(e2)
        kept += 1

    if verbose:
        print(f"[filter] kept={kept}, missing_files={missing}, days={len(day_buckets)}")
        if sample_missing:
            print("[filter] sample missing basenames:", sample_missing)

    if kept == 0:
        print("[warn] No usable items after filtering. Check wav_key/paths/segments.")
        return []

    # Selection per day
    rng = random.Random(seed)
    written: List[Path] = []

    for day, items in sorted(day_buckets.items(), key=lambda kv: kv[0]):
        # deterministic order: by serial then by basename
        items.sort(key=lambda d: (d["_serial"], d["_basename"]))
        if selection.lower() == "random":
            picks = items[:]
            rng.shuffle(picks)
            picks = picks[:n_per_day]
        else:
            picks = items[:n_per_day]
        if not picks:
            continue

        # Create a figure with len(picks) stacked panels
        n_rows = len(picks)
        fig_h = max(2.0 * n_rows, 2.0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(11.6, fig_h), squeeze=False, sharex=False)
        axes = axes.ravel()
        plt.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.08, hspace=0.0)

        for row_i, e in enumerate(picks):
            ax = axes[row_i]

            # Decide the panel time window
            info = sf.info(str(e["_path"]))
            fs = float(info.samplerate)
            total_dur = float(info.frames)/fs if info.frames and fs > 0 else 0.0
            segs = e["_segments"] or [(0.0, min(10.0, total_dur))]  # fallback if none
            s0, s1 = segs[0]
            # pad
            s0 = max(0.0, s0 - segment_pad_sec)
            s1 = min(total_dur, s1 + segment_pad_sec)
            # enforce minimum duration
            if (s1 - s0) < min_panel_dur_sec:
                mid = 0.5*(s0 + s1)
                s0 = max(0.0, mid - 0.5*min_panel_dur_sec)
                s1 = min(total_dur, mid + 0.5*min_panel_dur_sec)

            # Load just the needed samples
            y, fs_seg = _load_segment(e["_path"], s0, s1)
            # (Optional band-pass omitted for speed; 0–10 kHz view already enforced)

            _render_panel(ax, y, fs_seg, e["_animal"], e["_basename"], s0, s1)

            # Only bottom panel shows x-axis ticks/label
            if row_i == n_rows - 1:
                ax.set_xlabel("Time (s)")
                ax.tick_params(axis="x", which="both", labelbottom=True)
            else:
                ax.set_xlabel(None)
                ax.tick_params(axis="x", which="both", labelbottom=False)

        # File naming
        day_str = str(day)
        mode = "random" if selection.lower() == "random" else "first"
        out_name = f"{picks[0]['_animal']}_{day_str}_{mode}_n{n_rows}.png"
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        written.append(out_path)

    if verbose:
        print(f"[done] wrote {len(written)} file(s) to {out_dir}")

    return written




# ── Example usage ────────────────────────────────────────────────────────────
"""
from pathlib import Path
from render_n_song_panels_per_day_fast import render_n_song_panels_per_day

wav_dir       = Path("/Volumes/ROSE1-SSD/USA5288/")
detector_json = Path("/Volumes/ROSE1-SSD/USA5288/USA5288_song_detection.json")
out_dir       = Path("/Volumes/ROSE1-SSD/USA5288/panels_per_day_fast")

# FIRST n songs per day
written = render_n_song_panels_per_day(
    wav_dir=wav_dir,
    detector_json_path=detector_json,
    out_dir=out_dir,
    n_per_day=20,
    selection="first",      # or "random"
    seed=42,                # used for "random"
    segment_pad_sec=0.7,
    min_panel_dur_sec=4.0,
    low_cut=700, high_cut=7000,   # kept for parity; display is 0–10 kHz
    dpi=300,
    verbose=True,
)
print("Wrote:", len(written), "files")

"""
