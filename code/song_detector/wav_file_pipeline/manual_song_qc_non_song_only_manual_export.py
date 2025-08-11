#!/usr/bin/env python
# manual_song_qc_non_song_only_manual_export.py
# =============================================
# Shows spectrograms ONLY for files with contains_song == False.
# Lets you add missed-song spans and writes ONE consolidated JSON:
#   <out_dir>/<input_json_stem>_manual_only.json
# Now with progress tracking + completion message.

from __future__ import annotations
from pathlib import Path
import json, math, sys, shutil, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import spectrogram, butter, sosfiltfilt, resample_poly

# ───────── tunables ─────────
ROW_SEC        = 60.0
HOP_SEC        = 5.0
SPEC_NPERSEG   = 1024
SPEC_NOVERLAP  = 512
SPEC_VMIN      = -90
SPEC_VMAX      = -20
FREQ_MAX_INIT  = 10_000
BANDPASS_HZ    = (700, 7000)
TARGET_SR      = 44_100
CONTEXT_ON_ADD = 0.0

# ───────── terminal helpers ─────────
def _supports_ansi() -> bool:
    try: return sys.stdout.isatty()
    except Exception: return False
_BOLD  = "\033[1m" if _supports_ansi() else ""
_DIM   = "\033[2m" if _supports_ansi() else ""
_RESET = "\033[0m" if _supports_ansi() else ""

def _safe_stem(p: Path) -> str: return p.stem.replace(" ", "_").replace(":", "-")

# ───────── audio / spec helpers ─────────
def _load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path)
    if x.ndim > 1: x = x.mean(axis=1)
    return x, sr

def _maybe_resample(x: np.ndarray, sr: int, target_sr: Optional[int]) -> tuple[np.ndarray, int]:
    if target_sr is None or target_sr == sr: return x, sr
    g = math.gcd(target_sr, sr); up, down = target_sr // g, sr // g
    return resample_poly(x, up, down), target_sr

def _maybe_bandpass(x: np.ndarray, sr: int, band: Optional[Tuple[float, float]]) -> np.ndarray:
    if band is None: return x
    lo, hi = max(1.0, band[0]), band[1]
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)

def _spec_db(x: np.ndarray, sr: int):
    f, t, S = spectrogram(x, fs=sr, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP,
                          scaling="spectrum", mode="magnitude")
    return f, t, 20*np.log10(S + 1e-12)

# ───────── backend / key plumbing ─────────
def _warn_if_noninteractive():
    backend = matplotlib.get_backend()
    print(f"[INFO] Matplotlib backend: {backend}")
    if "inline" in backend.lower():
        print("[WARN] Backend is 'inline' (non-interactive). In Spyder run %matplotlib qt or tk.")

def _disable_toolbar_keymaps():
    for k in [
        "keymap.fullscreen","keymap.save","keymap.grid","keymap.home","keymap.back",
        "keymap.forward","keymap.pan","keymap.zoom","keymap.quit","keymap.quit_all",
        "keymap.copy","keymap.yscale","keymap.xscale","keymap.tight","keymap.ylimit","keymap.xlimit",
    ]:
        if k in matplotlib.rcParams: matplotlib.rcParams[k] = []

def _post_figure_setup(fig):
    # Detach mpl default key handler; focus canvas; refocus on enter
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "key_press_handler_id"):
        try: fig.canvas.mpl_disconnect(mgr.key_press_handler_id)
        except Exception: pass
    try:
        from matplotlib.backends.qt_compat import QtCore
        fig.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        fig.canvas.setFocus()
    except Exception:
        try: fig.canvas.setFocus()
        except Exception: pass
    def _on_enter(_evt):
        try: fig.canvas.setFocus()
        except Exception: pass
    fig.canvas.mpl_connect("figure_enter_event", _on_enter)

def _install_qt_key_filter(fig, on_key_callback):
    """Qt fallback: capture Qt key events and forward to _on_key; suppress auto-repeat."""
    try:
        from matplotlib.backends.qt_compat import QtCore, QtGui
    except Exception:
        return
    canvas = fig.canvas
    key_map = {
        QtCore.Qt.Key_Left:  "left",
        QtCore.Qt.Key_Right: "right",
        QtCore.Qt.Key_Up:    "up",
        QtCore.Qt.Key_Down:  "down",
        QtCore.Qt.Key_A: "a", QtCore.Qt.Key_F: "f",
        QtCore.Qt.Key_N: "n", QtCore.Qt.Key_P: "p",
        QtCore.Qt.Key_Q: "q", QtCore.Qt.Key_S: "s",
        QtCore.Qt.Key_U: "u", QtCore.Qt.Key_Escape: "escape",
    }
    class _Evt:  # shim like mpl event
        def __init__(self, key: str): self.key = key
    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress:
                if isinstance(event, QtGui.QKeyEvent) and event.isAutoRepeat():
                    return True  # swallow repeats
                k = key_map.get(event.key())
                if k is None:
                    txt = event.text()
                    if txt: k = txt.lower()
                if k is not None:
                    on_key_callback(_Evt(k))
                    return True
            return False
    f = _Filter(canvas)
    canvas.installEventFilter(f)
    if not hasattr(fig, "_qt_key_filter_refs"):
        fig._qt_key_filter_refs = []
    fig._qt_key_filter_refs.append(f)
    print("[INFO] Qt key filter installed (auto-repeat suppressed).")

# ───────── data structures ─────────
@dataclass
class FileRecord:
    file_name: str
    file_path: str
    duration_seconds: float

@dataclass
class SessionState:
    file_idx: int = 0
    t0: float = 0.0
    freq_max: float = FREQ_MAX_INIT

# ───────── core viewer ─────────
class QCViewer:
    def __init__(
        self,
        your_json_path: Path,
        out_dir: Path,
        target_sr: Optional[int] = TARGET_SR,
        bandpass_hz: Optional[Tuple[float, float]] = BANDPASS_HZ,
        start_file: int = 0,
    ):
        self.in_json_path = Path(your_json_path)
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manual_only_path = self.out_dir / f"{self.in_json_path.stem}_manual_only.json"

        # Load JSON and build list of NON-SONG files
        self.full_records: List[dict] = json.loads(self.in_json_path.read_text())
        self._by_name: Dict[str, dict] = {r["file_name"]: r for r in self.full_records}
        self.files: List[FileRecord] = [
            FileRecord(r["file_name"], r.get("file_path",""), float(r.get("duration_seconds",0.0)))
            for r in self.full_records if not r.get("contains_song", False)
        ]
        if not self.files:
            raise RuntimeError("No files with contains_song == False were found in your JSON.")

        # State
        self.audio: dict[int, tuple[np.ndarray, int]] = {}
        self.target_sr = target_sr
        self.bandpass_hz = bandpass_hz
        self.state = SessionState(file_idx=start_file)

        # In-memory annotations only
        self.added_spans: List[Tuple[float, float]] = []
        self.added_spans_by_file: Dict[str, List[Tuple[float, float]]] = {}

        # Debounce to stop accidental double triggers
        self._last_key: Optional[str] = None
        self._last_key_t: float = 0.0
        self._debounce_s: float = 0.12

        # NEW: progress tracking (visited files)
        self.reviewed_idx: set[int] = set()

        _warn_if_noninteractive()
        _disable_toolbar_keymaps()

        self._load_file_audio(self.state.file_idx)

        self.fig, self.ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        self._connect_events()
        self._redraw()
        _post_figure_setup(self.fig)
        if "qt" in matplotlib.get_backend().lower():
            _install_qt_key_filter(self.fig, self._on_key)
        self._print_usage_banner()

    # ───────── IO ─────────
    def _load_file_audio(self, idx: int):
        self.added_spans = []
        rec = self.files[idx]; wav = Path(rec.file_path)
        if not wav.exists(): raise FileNotFoundError(f"Missing WAV: {wav}")
        x, sr = _load_audio_mono(wav); x, sr = _maybe_resample(x, sr, self.target_sr); x = _maybe_bandpass(x, sr, self.bandpass_hz)
        self.audio[idx] = (x, sr)
        dur = len(x)/sr; self.state.t0 = float(np.clip(self.state.t0, 0.0, max(0.0, dur - ROW_SEC)))

    def _write_manual_only_json(self):
        """Write ONE JSON with ONLY the files that have manually added spans."""
        manual_records = []
        for fname, spans in self.added_spans_by_file.items():
            if not spans: continue
            base = self._by_name.get(fname, {})
            manual_records.append({
                "file_name": fname,
                "file_path": base.get("file_path", ""),
                "duration_seconds": float(base.get("duration_seconds", 0.0)),
                "contains_song": True,
                "detected_song_times": [[float(s), float(e)] for (s, e) in spans],
            })
        self.manual_only_path.write_text(json.dumps(manual_records, indent=2))
        print(f"[OK] Wrote manual-only JSON → {self.manual_only_path}")

    # ───────── drawing ─────────
    def _draw_window(self):
        self.ax.clear()
        idx = self.state.file_idx; x, sr = self.audio[idx]
        dur = len(x)/sr; t0 = self.state.t0; t1 = min(dur, t0 + ROW_SEC)
        i0, i1 = int(round(t0*sr)), int(round(t1*sr)); seg = x[i0:i1]
        f, t, S_db = _spec_db(seg, sr)
        self.ax.pcolormesh(t + t0, f, S_db, shading="auto", cmap="gray_r", vmin=SPEC_VMIN, vmax=SPEC_VMAX)
        # overlays for current file only
        for s, e in self.added_spans:
            if e < t0 or s > t1: continue
            self.ax.add_patch(Rectangle((max(s,t0), 0), max(0.0, min(e,t1)-max(s,t0)),
                                        self.state.freq_max, facecolor=(1,0,0,0.30), edgecolor=None, zorder=7))
        self.ax.set_ylim(0, self.state.freq_max); self.ax.set_xlim(t0, t1)
        self.ax.set_ylabel("Freq [Hz]"); self.ax.set_xlabel("Time [s]")
        self.ax.legend([Rectangle((0,0),1,1,facecolor=(1,0,0,0.30), edgecolor=None)],
                       ["Manually added span"], frameon=False, loc="upper right")

    def _update_title(self):
        idx = self.state.file_idx
        total = len(self.files)
        reviewed = len(self.reviewed_idx)
        rec = self.files[idx]
        x, sr = self.audio[idx]
        dur = len(x) / sr
        t0 = self.state.t0
        t1 = min(dur, t0 + ROW_SEC)
        progress = f"[{idx + 1}/{total} | reviewed {reviewed}]"
        self.ax.set_title(
            f"{progress}  {rec.file_name}   {t0:.2f}–{t1:.2f}s   "
            "(←/→ scroll, a add span, u undo, n/p next/prev, s save, q save+quit)"
        )

    def _redraw(self):
        self._draw_window()
        # mark as reviewed + update title
        self.reviewed_idx.add(self.state.file_idx)
        self._update_title()
        self.fig.canvas.draw_idle()
        if len(self.reviewed_idx) == len(self.files):
            print("🎉 All files visited. Press 's' to write JSON, or 'q' to save and quit.")

    # ───────── navigation ─────────
    def _jump_window(self, direction: int):
        x, sr = self.audio[self.state.file_idx]; dur = len(x)/sr
        self.state.t0 = float(np.clip(self.state.t0 + direction*HOP_SEC, 0.0, max(0.0, dur-ROW_SEC)))
        self._redraw()

    def _jump_file(self, direction: int):
        # persist current file's spans in-memory
        rec = self.files[self.state.file_idx]
        self.added_spans_by_file[rec.file_name] = list(self.added_spans)

        prev = self.state.file_idx
        new  = int(np.clip(prev + direction, 0, len(self.files) - 1))
        if new == prev:
            if direction > 0:
                print("[INFO] Already at last file.")
            else:
                print("[INFO] Already at first file.")
            return

        self.state.file_idx = new
        self.state.t0 = 0.0
        self._load_file_audio(self.state.file_idx)
        self._redraw()

    # ───────── annotations ─────────
    def _add_span_by_clicks(self):
        print("Add span: click START then END in the plot (Esc to cancel).")
        coords: List[float] = []
        def onclick(event):
            if event.inaxes != self.ax: return
            coords.append(event.xdata)
            if len(coords) == 2:
                self.fig.canvas.mpl_disconnect(cid_click); self.fig.canvas.mpl_disconnect(cid_key)
                s, e = sorted(coords[:2]); s -= CONTEXT_ON_ADD; e += CONTEXT_ON_ADD
                s = max(0.0, s); e = max(s + 1e-3, e)
                self.added_spans.append((s, e))
                # mirror to dict-of-files (in memory only)
                rec = self.files[self.state.file_idx]
                self.added_spans_by_file.setdefault(rec.file_name, [])
                self.added_spans_by_file[rec.file_name] = list(self.added_spans)
                self._redraw(); print(f"Added span: {s:.2f}–{e:.2f}s")
        def onkey(event):
            if event.key == "escape":
                self.fig.canvas.mpl_disconnect(cid_click); self.fig.canvas.mpl_disconnect(cid_key); print("Add span cancelled.")
        cid_click = self.fig.canvas.mpl_connect("button_press_event", onclick)
        cid_key   = self.fig.canvas.mpl_connect("key_press_event", onkey)

    def _undo_last_span(self):
        if self.added_spans:
            s, e = self.added_spans.pop()
            rec = self.files[self.state.file_idx]
            self.added_spans_by_file[rec.file_name] = list(self.added_spans)
            print(f"Undo last span: {s:.2f}–{e:.2f}s"); self._redraw()

    # ───────── events ─────────
    def _connect_events(self):
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        # Debounce identical key within short interval
        now = time.time()
        if self._last_key == event.key and (now - self._last_key_t) < self._debounce_s:
            return
        self._last_key, self._last_key_t = event.key, now

        k = event.key
        if k == "right": self._jump_window(+1)
        elif k == "left": self._jump_window(-1)
        elif k == "up":   self.state.freq_max = min(20_000, self.state.freq_max + 1_000); self._redraw()
        elif k == "down": self.state.freq_max = max(2_000, self.state.freq_max - 1_000); self._redraw()
        elif k == "n":    self._jump_file(+1)
        elif k == "p":    self._jump_file(-1)
        elif k == "a":    self._add_span_by_clicks()
        elif k == "u":    self._undo_last_span()
        elif k == "s":
            self._write_manual_only_json()
            added = sum(1 for v in self.added_spans_by_file.values() if v)
            print(f"[SUMMARY] Visited {len(self.reviewed_idx)}/{len(self.files)} files; "
                  f"{added} file(s) with manually added spans.")
            print("Saved manual-only JSON.")
        elif k == "q":
            self._write_manual_only_json()
            added = sum(1 for v in self.added_spans_by_file.values() if v)
            print(f"[SUMMARY] Visited {len(self.reviewed_idx)}/{len(self.files)} files; "
                  f"{added} file(s) with manually added spans.")
            plt.close(self.fig)
        else:
            print(f"[key] {k} (no action bound)")

    # ───────── banner ─────────
    def _print_usage_banner(self):
        width = shutil.get_terminal_size((80, 20)).columns
        bar = "─" * max(40, min(100, width))
        total = len(self.files)
        in_json = str(self.in_json_path)
        msg = f"""
{_BOLD}Manual Song QC – Non-song files{_RESET}
Reviewing {total} file(s) from:
  {_DIM}{in_json}{_RESET}
Will write ONE JSON to:
  {_DIM}{self.manual_only_path}{_RESET}

{_BOLD}Keys{_RESET}
  ← / →   scroll window by {HOP_SEC:.1f}s      ↑ / ↓   change frequency ceiling (±1 kHz)
  n / p   next / previous file                 a        add span (two clicks: start then end)
  u       undo last span                       s        write manual-only JSON
  q       write JSON and quit

{_BOLD}Notes{_RESET}
  • Only files with contains_song = False are shown.
  • JSON includes ONLY files where you added at least one span.
  • Bandpass: {BANDPASS_HZ[0]}–{BANDPASS_HZ[1]} Hz   |   Target SR: {TARGET_SR if TARGET_SR else 'native'}
"""
        print(bar); print(msg.strip()); print(bar)

# ───────── public API ─────────
def run_manual_qc_non_song_manual_export(
    your_json_path: str | Path,
    out_dir: str | Path,
    *,
    target_sr: Optional[int] = TARGET_SR,
    bandpass_hz: Optional[Tuple[float, float]] = BANDPASS_HZ,
    start_file: int = 0,
):
    viewer = QCViewer(Path(your_json_path), Path(out_dir),
                      target_sr=target_sr, bandpass_hz=bandpass_hz, start_file=start_file)
    plt.show()

# ───────── CLI ─────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Manual spectrogram QC for non-song files; export one manual-only JSON.")
    p.add_argument("--yours",  required=True, type=Path, help="Path to your *_detected_song_intervals.json")
    p.add_argument("--outdir", required=True, type=Path, help="Output directory for the JSON")
    p.add_argument("--sr",     type=int, default=TARGET_SR, help="Target SR for spectrograms (<=0 to keep native)")
    p.add_argument("--no_bp",  action="store_true", help="Disable bandpass")
    p.add_argument("--start",  type=int, default=0, help="Start at file index")
    args = p.parse_args()
    sr = None if args.sr is not None and args.sr <= 0 else args.sr
    bp = None if args.no_bp else BANDPASS_HZ
    run_manual_qc_non_song_manual_export(args.yours, args.outdir, target_sr=sr, bandpass_hz=bp, start_file=args.start)


"""
%matplotlib qt
from manual_song_qc_non_song_only_manual_export import run_manual_qc_non_song_manual_export

run_manual_qc_non_song_manual_export(
    your_json_path="/Volumes/my_own_ssd/USA5288/0/0_detected_song_intervals.json",
    out_dir="/Volumes/my_own_ssd/USA5288/0/",
    target_sr=44100,
    bandpass_hz=(700, 7000),
)



"""