from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json
import math
import random
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

# ── Spectrogram/display tunables ──────────────────────────────────────────────
SPEC_NPERSEG   = 2048
SPEC_NOOVERLAP = SPEC_NPERSEG - 119
SPEC_WINDOW    = windows.gaussian(SPEC_NPERSEG, std=SPEC_NPERSEG/8)

Y_MIN = 0.0
Y_MAX = 10_000.0
CMAP  = "gray"          # 0=black, 1=white (we invert below to get white background)

# Robust normalization (white background)
PCTL_LO = 12.0
PCTL_HI = 99.5
GAMMA   = 0.20          # <1 brightens background

# Styling
BOUNDARY_COLOR = "red"
BOUNDARY_LS    = "--"
BOUNDARY_LW    = 0.9
LABEL_FONTSIZE = 8


# ── Helpers ───────────────────────────────────────────────────────────────────
def _excel_serial_to_date(serial: float) -> datetime.date:
    origin = datetime(1899, 12, 30)
    return (origin + timedelta(days=float(serial))).date()

def _parse_day_from_basename(name: str) -> Optional[str]:
    """
    Expected basename pattern: {animal}_{excelSerial}_{...}.wav
    Returns 'YYYY-MM-DD' or None if it can't be parsed.
    """
    try:
        stem = Path(name).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        serial = float(parts[1])
        return _excel_serial_to_date(serial).isoformat()
    except Exception:
        return None

def _index_wavs(wav_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for p in wav_dir.rglob("*.wav"):
        index[p.name.lower()] = p
    for p in wav_dir.rglob("*.WAV"):
        index[p.name.lower()] = p
    return index

def _load_json_rows(json_path: Path) -> List[dict]:
    with json_path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    return data

def _pick_n(items: List[dict], n: int, mode: str, rng: random.Random) -> List[dict]:
    if mode.lower() == "random":
        if len(items) <= n:
            return items[:]
        picks = items[:]
        rng.shuffle(picks)
        return picks[:n]
    return items[:n]


# ── Audio → Spectrogram (white background, full file) ─────────────────────────
def _spectrogram_disp(y: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y) == 0:
        return np.array([0.0, 1e-3]), np.array([0.0, 1.0]), np.ones((2, 2), float)
    f, t, S = spectrogram(
        y, fs=fs,
        window=SPEC_WINDOW, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOOVERLAP,
        detrend=False, scaling="spectrum",
    )
    S_db = 10.0 * np.log10(S + np.finfo(float).eps)
    lo = np.percentile(S_db, PCTL_LO)
    hi = np.percentile(S_db, PCTL_HI)
    if hi <= lo:
        lo, hi = S_db.min(), S_db.max() + 1e-6
    S_norm = np.clip((S_db - lo) / (hi - lo), 0.0, 1.0)
    S_disp = (1.0 - S_norm) ** GAMMA
    return t, f, S_disp


# ── Blank panel ────────────────────────────────────────────────────────────────
def _render_blank_panel(ax: plt.Axes):
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_xticks([])
    ax.set_ylabel("Freq (Hz)")


# ── Main API (packed layout) ──────────────────────────────────────────────────
def render_n_song_panels_per_day(
    wav_dir: Path,
    detector_json_path: Path,
    out_dir: Path,
    n_per_day: int = 6,
    selection: str = "first",         # or "random"
    seed: int = 42,
    panels_per_png: int = 6,
    panel_duration_sec: float = 10.0, # fixed x-span per panel
    dpi: int = 300,
    only_song_present: bool = False,
    wav_key: str = "filename",
    verbose: bool = True,
) -> List[Path]:
    """
    Packs multiple recordings *sequentially* into each panel until `panel_duration_sec`
    is filled; then continues on the next panel. Red dashed lines mark the start and
    end of each recording wherever those fall inside a panel.
    """
    rng = random.Random(seed)
    wav_dir = wav_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index audio
    if verbose:
        print("[index] scanning wavs...")
    wav_index = _index_wavs(wav_dir)
    if verbose:
        print(f"[index] indexed {len(wav_index)} wav files")

    # Load JSON and bucket by day
    rows = _load_json_rows(detector_json_path)
    buckets: Dict[str, List[dict]] = {}
    missing: List[str] = []

    for r in rows:
        fn = r.get(wav_key)
        if not isinstance(fn, str):
            continue
        base = Path(fn).name
        day = _parse_day_from_basename(base)
        if day is None:
            continue
        if only_song_present and not bool(r.get("song_present")):
            continue
        p = wav_index.get(base.lower())
        if p is None:
            missing.append(base)
            continue
        buckets.setdefault(day, []).append({"_basename": base, "_path": p})

    if verbose:
        print(f"[filter] days={len(buckets)}, kept_rows={sum(len(v) for v in buckets.values())}, "
              f"missing_files={len(missing)}")
        if missing[:5]:
            print("[filter] sample missing:", missing[:5])

    written: List[Path] = []

    # Iterate days deterministically
    for day in sorted(buckets.keys()):
        items = buckets[day]
        items.sort(key=lambda d: d["_basename"])
        picks = _pick_n(items, n_per_day, selection, rng)

        # Precompute spectrograms once per selected recording
        packed = []
        for it in picks:
            wav_path = it["_path"]
            try:
                y, fs = sf.read(str(wav_path), always_2d=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                y = y.astype(float, copy=False)
                t, f_hz, S_disp = _spectrogram_disp(y, float(fs))
                rec_total = float(t[-1]) if t.size else 0.0
            except Exception:
                # placeholder on error
                t = np.array([0.0, 1e-3])
                f_hz = np.array([0.0, 1.0])
                S_disp = np.ones((2, 2), float)
                rec_total = 0.0
            packed.append(dict(path=wav_path, t=t, f=f_hz, S=S_disp, dur=rec_total))

        # Helper: new page & saving
        def _new_page(page_idx: int):
            fig_h = max(1.8 * panels_per_png, 2.0)
            fig, axes = plt.subplots(
                panels_per_png, 1, figsize=(11.6, fig_h), squeeze=False, sharex=False
            )
            axes = axes.ravel()
            for ax in axes:
                # standard look
                ax.set_facecolor("white")
                ax.set_ylim(Y_MIN, Y_MAX)
                ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
                ax.set_ylabel("Freq (Hz)")
                ax.set_xlim(0.0, float(panel_duration_sec))
            # x labels only on bottom
            for ax in axes[:-1]:
                ax.set_xlabel(None)
                ax.tick_params(axis="x", which="both", labelbottom=False)
            axes[-1].set_xlabel("Time (s)")
            return fig, axes

        def _save_page(fig, axes, page_idx: int, n_pages_hint: Optional[int] = None):
            title = f"{day} — {selection} {n_per_day} (packed)"
            if n_pages_hint is not None:
                title += f" · page {page_idx} of {n_pages_hint}"
            fig.suptitle(title, fontsize=12, y=0.995)
            plt.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.08, hspace=0.20)
            out_path = out_dir / f"{day}_{selection}_{n_per_day}_packed_part{page_idx}.png"
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
            written.append(out_path)
            if verbose:
                print("[write]", out_path)

        # Render packed
        if not packed:
            # emit a blank page
            fig, axes = _new_page(1)
            for ax in axes:
                _render_blank_panel(ax)
                ax.set_xlim(0.0, float(panel_duration_sec))
            _save_page(fig, axes, 1, 1)
            continue

        page_idx = 1
        fig, axes = _new_page(page_idx)
        ax_i = 0
        x_cursor = 0.0
        panels_used = 1  # count current panel

        def _advance_panel():
            nonlocal fig, axes, ax_i, x_cursor, page_idx, panels_used
            ax_i += 1
            if ax_i >= panels_per_png:
                _save_page(fig, axes, page_idx)
                page_idx += 1
                fig, axes = _new_page(page_idx)
                ax_i = 0
            x_cursor = 0.0
            panels_used += 1

        for rec in packed:
            t = rec["t"]; f_hz = rec["f"]; S_disp = rec["S"]; dur = rec["dur"]
            pos = 0.0
            first_chunk = True
            while pos < max(dur, 1e-9):
                # how much room left on the current panel
                room = panel_duration_sec - x_cursor
                take = min(room, dur - pos)
                # slice in time [pos, pos+take]
                if take > 0.0:
                    mask_t = (t >= pos) & (t <= pos + take + 1e-9)
                    if np.any(mask_t):
                        t_sel = t[mask_t]
                        S_sel = S_disp[:, mask_t]
                        # frequency crop
                        mask_f = (f_hz >= Y_MIN) & (f_hz <= Y_MAX)
                        if not np.any(mask_f):
                            mask_f = slice(None)
                        f_view = f_hz[mask_f]
                        S_view = S_sel[mask_f, :]

                        # extent maps into [x_cursor, x_cursor + draw_len]
                        draw_len = float(t_sel[-1] - t_sel[0]) if t_sel.size > 1 else take
                        axes[ax_i].imshow(
                            S_view, origin="lower", aspect="auto", interpolation="nearest",
                            extent=(x_cursor, x_cursor + draw_len,
                                    float(f_view[0]), float(f_view[-1])),
                            cmap=CMAP, vmin=0.0, vmax=1.0
                        )

                    # boundary markers inside this panel chunk
                    if first_chunk:
                        # start boundary
                        axes[ax_i].axvline(x_cursor, color=BOUNDARY_COLOR, ls=BOUNDARY_LS, lw=BOUNDARY_LW)
                        # filename label
                        axes[ax_i].text(
                            x_cursor + 0.02 * panel_duration_sec, 0.98 * Y_MAX,
                            rec["path"].name, fontsize=LABEL_FONTSIZE,
                            color=BOUNDARY_COLOR, va="top"
                        )

                    # end boundary if the recording ends in this chunk
                    if abs((pos + take) - dur) < 1e-9:
                        axes[ax_i].axvline(x_cursor + take, color=BOUNDARY_COLOR, ls=BOUNDARY_LS, lw=BOUNDARY_LW)

                    x_cursor += take
                    pos += take
                    first_chunk = False

                # advance to next panel if no room remains
                if x_cursor >= panel_duration_sec - 1e-12 and pos < dur - 1e-12:
                    _advance_panel()

            # if exactly filled the panel, move to a fresh one for the next recording
            if x_cursor >= panel_duration_sec - 1e-12:
                _advance_panel()

        # pad blanks to finish the last page neatly
        # (no need to know total pages ahead of time; we save the last page now)
        _save_page(fig, axes, page_idx)

    return written


# ── Example usage ────────────────────────────────────────────────────────────
# -*- coding: utf-8 -*-
"""
from pathlib import Path
from render_n_song_panels_per_day_fast import render_n_song_panels_per_day

wav_dir       = Path("/Volumes/ROSE1-SSD/USA5288/")
detector_json = Path("/Volumes/ROSE1-SSD/USA5288/USA5288_song_detection.json")
out_dir       = Path("/Volumes/ROSE1-SSD/USA5288/panels_per_day_fast")

written = render_n_song_panels_per_day(
    wav_dir=wav_dir,
    detector_json_path=detector_json,
    out_dir=out_dir,
    n_per_day=20,                 # select N recordings per day
    selection="first",            # 'first' or 'random'
    panels_per_png=6,             # rows per PNG
    panel_duration_sec=10.0,      # fixed timescale per panel
    dpi=300,
    only_song_present=True,       # require song_present in JSON
    verbose=True,
)
print(f"Wrote {len(written)} PNGs")
print("First few:", written[:3])
"""
