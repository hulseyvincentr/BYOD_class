# -*- coding: utf-8 -*-
# render_song_panels.py  ·  streaming aggregated panels (full-width, bottom-only axes)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Tuple, Deque

import json, re, math, collections
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, spectrogram, resample_poly
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Display / DSP tunables ────────────────────────────────────────────────────
SPEC_NPERSEG = 1024
SPEC_NOVERLAP = 512
SPEC_VMIN_DB, SPEC_VMAX_DB = -90, -20
CMAP = "gray_r"
OVERLAY_ALPHA = 0.28
OVERLAY_COLOR = "yellow"
BOUNDARY_COLOR = "red"
BOUNDARY_LS = "--"
BOUNDARY_LW = 0.9
LABEL_FONTSIZE = 8
_WAV_RE = re.compile(r"\.wav$", re.IGNORECASE)

# ── JSON helpers ──────────────────────────────────────────────────────────────
def _entry_has_song(e: dict) -> bool:
    if e.get("song_present") is True: return True
    if e.get("contains_song") is True: return True
    t = e.get("segments") or e.get("detected_song_times") or e.get("song_times")
    return isinstance(t, list) and len(t) > 0

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

def _resolve_wav_path(entry: dict, wav_dir: Path, wav_key: str) -> Path:
    if wav_key not in entry:
        raise KeyError(f"Missing '{wav_key}' in entry (keys={list(entry.keys())[:10]}...)")
    v = entry[wav_key]
    if not isinstance(v, str) or not _WAV_RE.search(v):
        raise ValueError(f"Entry['{wav_key}'] must be a '.wav' string (got {v!r})")
    p = Path(v)
    if p.is_absolute():
        if not p.exists(): raise FileNotFoundError(p)
        return p
    cand = wav_dir / v
    if cand.exists(): return cand
    hits = list(wav_dir.rglob(Path(v).name))
    if hits: return hits[0]
    raise FileNotFoundError(f"Cannot resolve '{v}' under {wav_dir}")

# ── Timeline + audio ─────────────────────────────────────────────────────────
@dataclass
class TLFile:
    path: Path
    fs: float
    duration: float
    start: float
    end: float
    song_windows: List[Tuple[float, float]]
    _y_filtered: Optional[np.ndarray] = None
    _fs_loaded: Optional[float] = None

def _butter_bandpass(lowcut: float, highcut: float, fs: float, order=4):
    nyq = 0.5 * fs
    low = max(1e-9, min(lowcut / nyq, 0.999999))
    high = max(low + 1e-9, min(highcut / nyq, 0.999999))
    b, a = butter(order, [low, high], btype="band"); return b, a

def _apply_bandpass(y: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    if not (low and high) or high <= low or low <= 0 or high <= 0:
        return np.ascontiguousarray(y, dtype=np.float64)
    b, a = _butter_bandpass(low, high, fs)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    return filtfilt(b, a, y, method="gust")

def _load_and_prepare(tlf: TLFile, target_fs: float, low: float, high: float):
    if tlf._y_filtered is not None and tlf._fs_loaded == target_fs: return
    y, fs = sf.read(str(tlf.path), always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    y = _apply_bandpass(y, fs, low, high)
    if fs != target_fs:
        up = int(target_fs); dn = int(fs)
        g = math.gcd(up, dn); up //= g; dn //= g
        y = resample_poly(y, up, dn); fs = target_fs
    tlf._y_filtered = np.ascontiguousarray(y, dtype=np.float64); tlf._fs_loaded = fs

def _slice_y(tlf: TLFile, local_start: float, local_end: float) -> np.ndarray:
    assert tlf._y_filtered is not None and tlf._fs_loaded is not None
    fs = tlf._fs_loaded
    i0 = max(0, int(round(local_start * fs))); i1 = min(len(tlf._y_filtered), int(round(local_end * fs)))
    if i1 <= i0: return np.zeros(1, dtype=np.float64)
    return tlf._y_filtered[i0:i1]

def _compute_spectrogram(y: np.ndarray, fs: float):
    if len(y) == 0:
        return np.array([0.0, 1e-3]), np.array([0.0, 1.0]), np.full((2,2), SPEC_VMIN_DB, float)
    f, t, S = spectrogram(y, fs=fs, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP,
                          detrend=False, scaling="spectrum", mode="magnitude")
    S_db = 20.0 * np.log10(np.maximum(S, 1e-12)); return t, f, S_db

# ── Streaming prep ───────────────────────────────────────────────────────────
def _entry_iter(items: List[dict], wav_dir: Path, wav_key: str, only_song_present: bool):
    for e in items:
        if only_song_present and not _entry_has_song(e): continue
        try:
            path = _resolve_wav_path(e, wav_dir, wav_key)
            info = sf.info(str(path))
            fs = float(info.samplerate)
            dur = float(info.frames)/fs if info.frames and fs>0 else 0.0
            w = _normalize_times(e)
            yield (path, fs, dur, w)
        except Exception:
            continue

# ── Rendering helpers ─────────────────────────────────────────────────────────
def _collect_overlays(inter: List[TLFile], p0: float, p1: float) -> List[Tuple[float,float]]:
    spans: List[Tuple[float,float]] = []
    for f in inter:
        for s_local, e_local in f.song_windows:
            s = f.start + s_local; e = f.start + e_local
            if e <= p0 or s >= p1: continue
            s = max(p0, s); e = min(p1, e)
            if e > s: spans.append((s - p0, e - p0))
    return spans

def _collect_boundaries(inter: List[TLFile], p0: float, p1: float) -> List[Tuple[float,str,bool]]:
    out: List[Tuple[float,str,bool]] = []
    for f in inter:
        if p0 < f.start < p1: out.append((f.start - p0, f.path.name, True))
        if p0 < f.end   < p1: out.append((f.end   - p0, f.path.name, False))
    out.sort(key=lambda x: x[0]); return out

def _draw_panel(ax, y_panel: np.ndarray, fs: float, p0: float, p1: float,
                inter: List[TLFile], show_xlabel: bool, show_ylabel: bool):
    # Spectrogram
    t, f, S_db = _compute_spectrogram(y_panel, fs)
    ax.pcolormesh(t, f, S_db, shading="auto", cmap=CMAP, vmin=SPEC_VMIN_DB, vmax=SPEC_VMAX_DB)

    # Overlays
    for s_rel, e_rel in _collect_overlays(inter, p0, p1):
        ax.axvspan(s_rel, e_rel, color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA, lw=0)

    # File boundaries + labels
    for x, fname, is_start in _collect_boundaries(inter, p0, p1):
        ax.axvline(x, color=BOUNDARY_COLOR, ls=BOUNDARY_LS, lw=BOUNDARY_LW)
        if is_start:
            ax.text(x + 0.02*(p1-p0), f.max()*0.95, fname, fontsize=LABEL_FONTSIZE,
                    color=BOUNDARY_COLOR, va="top", ha="left", alpha=0.9)
        else:
            ax.plot([x, x], [f.max()*0.98, f.max()], color=BOUNDARY_COLOR, lw=BOUNDARY_LW)

    # Panel edges
    ax.axvline(0.0, color=BOUNDARY_COLOR, ls=BOUNDARY_LS, lw=BOUNDARY_LW)
    ax.axvline(p1 - p0, color=BOUNDARY_COLOR, ls=BOUNDARY_LS, lw=BOUNDARY_LW)

    # Formatting: full-width, bottom-only axes
    ax.set_xlim(0, p1 - p0)
    ax.margins(x=0)

    if show_xlabel:
        ax.set_xlabel("Time (s)")
        ax.tick_params(axis="x", which="both", labelbottom=True)
    else:
        ax.set_xlabel(None)
        ax.tick_params(axis="x", which="both", labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Freq (Hz)")
        ax.tick_params(axis="y", which="both", labelleft=True)
    else:
        ax.set_ylabel(None)
        ax.tick_params(axis="y", which="both", labelleft=False)

# ── Public API (STREAMING) ───────────────────────────────────────────────────
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
    wav_key: str = "filename",
    verbose: bool = True,
    max_files: Optional[int] = None,
    max_total_duration_sec: Optional[float] = None,
) -> List[Path]:
    """
    Streaming renderer. No colorbar or titles. Panels share x; only bottom shows x ticks/label.
    Only bottom shows y ticks/label; no vertical gap between panels.
    """
    wav_dir = Path(wav_dir).expanduser().resolve()
    detector_json_path = Path(detector_json_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve() if out_dir is not None else (wav_dir / "panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    with detector_json_path.open() as f:
        items = json.load(f)

    feed = _entry_iter(items, wav_dir, wav_key, only_song_present)

    buf: Deque[TLFile] = collections.deque()
    buf_end_time = 0.0
    total_files = 0
    written: List[Path] = []
    target_fs: Optional[float] = None

    batch_len = segment_duration_sec * panels_per_fig
    next_panel_start = 0.0
    batch_idx = 0

    def _append_next() -> bool:
        nonlocal buf_end_time, total_files, target_fs
        try:
            path, fs, dur, windows = next(feed)
        except StopIteration:
            return False
        if target_fs is None:
            target_fs = fs
        if pad_before_after_sec and pad_before_after_sec > 0:
            windows = [(max(0.0, s - pad_before_after_sec), max(0.0, e + pad_before_after_sec))
                       for (s, e) in windows]
        tlf = TLFile(path=path, fs=fs, duration=dur, start=buf_end_time,
                     end=buf_end_time + dur, song_windows=windows)
        buf.append(tlf); buf_end_time += dur; total_files += 1
        return True

    needed_end = next_panel_start + batch_len
    while buf_end_time < needed_end:
        if max_files is not None and total_files >= max_files: break
        if not _append_next(): break
        if max_total_duration_sec is not None and buf_end_time >= max_total_duration_sec: break

    if target_fs is None:
        if verbose: print("[render] No valid audio found.")
        return []

    while next_panel_start < buf_end_time:
        p0 = next_panel_start
        needed_end = p0 + batch_len
        while buf_end_time < needed_end:
            if max_files is not None and total_files >= max_files: break
            if not _append_next(): break
            if max_total_duration_sec is not None and buf_end_time >= max_total_duration_sec: break

        actual_p1 = min(buf_end_time, needed_end)
        panel_starts = [p0 + i*segment_duration_sec for i in range(panels_per_fig)
                        if p0 + i*segment_duration_sec < actual_p1]
        if not panel_starts: break

        n_rows = len(panel_starts)
        fig_h = max(2.0 * n_rows, 2.0)

        # sharex=True; remove vertical gaps with hspace=0; minimal margins so panels span width
        fig, axes = plt.subplots(n_rows, 1, figsize=(11.6, fig_h), squeeze=False, sharex=True)
        axes = axes.ravel()
        plt.subplots_adjust(left=0.06, right=0.995, top=0.995, bottom=0.08, hspace=0.0)

        buf_list = list(buf)
        idx = 0
        while idx < len(buf_list) and buf_list[idx].end <= panel_starts[0]:
            idx += 1

        for row_i, t_start in enumerate(panel_starts):
            t_end = min(t_start + segment_duration_sec, buf_end_time)
            ax = axes[row_i]

            # files intersecting this panel
            inter: List[TLFile] = []
            j = idx
            while j < len(buf_list) and buf_list[j].start < t_end:
                if buf_list[j].end > t_start: inter.append(buf_list[j])
                j += 1

            chunks: List[np.ndarray] = []
            for fobj in inter:
                _load_and_prepare(fobj, target_fs, low_cut, high_cut)
                s = max(fobj.start, t_start); e = min(fobj.end, t_end)
                chunks.append(_slice_y(fobj, s - fobj.start, e - fobj.start))
            y_panel = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float64)

            show_xlabel = (row_i == n_rows - 1)
            show_ylabel = (row_i == n_rows - 1)  # only bottom shows frequency scale
            _draw_panel(ax, y_panel, target_fs, t_start, t_end, inter, show_xlabel, show_ylabel)

            while idx < len(buf_list) and buf_list[idx].end <= t_start:
                idx += 1

        # Keep overlay legend (remove if you want)
        patch = mpatches.Patch(color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA, label="Detected song")
        axes[0].legend(handles=[patch], loc="upper right", frameon=True, fontsize=9)

        out_name = f"aggregated_panels_{batch_idx:03d}.png"
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        if verbose: print(f"[render] Wrote {out_path}")
        written.append(out_path)

        next_panel_start += batch_len
        while buf and buf[0].end <= next_panel_start: buf.popleft()
        batch_idx += 1

        if max_total_duration_sec is not None and next_panel_start >= max_total_duration_sec: break
        if not buf and next_panel_start >= buf_end_time: break

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
    panels_per_fig=5,
    low_cut=700,     # can match the view...
    high_cut=2000,   # ...but not required
    only_song_present=True,
    pad_before_after_sec=0.0,
    wav_key="filename",
    verbose=True,
    max_files=50,
)
print(f"Wrote {len(written)} figure(s). First few:", written[:3])

"""