#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_song_features.py
========================

Combine **Amplitude & Periodicity** features with **Shannon & Wiener entropy**
features from a single `.npz` spectrogram.  Results are written to a JSON file
and also available programmatically via the returned path.

Typical notebook use
--------------------
```python
from extract_song_features import extract_song_features
json_path = extract_song_features('segment.npz')
```

Command‑line quick‑look
-----------------------
```bash
python extract_song_features.py segment.npz \
       --amp-sigma 15 --periodicity-thr 0.12 \
       --entropy-sigma 10 --sthresh 4.5 --wthresh -0.7
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ── add this folder to PYTHONPATH so sibling modules import cleanly ─────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # no harm if already present

# ── robust local imports (handles either filename + either function name) ──
from importlib import import_module

def _resolve_amp_func() -> Any:
    """Try both module names and both function names; return callable."""
    for mod_name in (
        "calculate_spectrogram_amp_and_periodicity",
        "calculate_npz_spectrogram_amplitude_and_periodicity",
    ):
        try:
            mod = import_module(mod_name)
            for fn_name in (
                "calculate_spectrogram_amp_and_periodicity",
                "calculate_spectrogram_amplitude_and_periodicity",
                "calculate_npz_spectrogram_amplitude_and_periodicity",
            ):
                if hasattr(mod, fn_name):
                    return getattr(mod, fn_name)
        except ModuleNotFoundError:
            continue
    raise ImportError("Could not locate amplitude/periodicity calculation function.")


def _resolve_entropy_func() -> Any:
    for mod_name in ("calculate_dual_entropy", "calculate_spectrogram_entropy"):
        try:
            mod = import_module(mod_name)
            if hasattr(mod, "calculate_spectrogram_entropy"):
                return getattr(mod, "calculate_spectrogram_entropy")
        except ModuleNotFoundError:
            continue
    raise ImportError("Could not locate entropy calculation function.")

_amp_func = _resolve_amp_func()
_ent_func = _resolve_entropy_func()

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _np_to_py(obj: Any) -> Any:
    """Recursively convert NumPy scalars / arrays to builtin Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _np_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_to_py(v) for v in obj]
    return obj

# ─────────────────────────────────────────────────────────────────────────────
# main driver
# ─────────────────────────────────────────────────────────────────────────────

def extract_song_features(
    npz_path: str | Path,
    *,
    # amplitude / periodicity params
    amp_sigma: float = 20.0,
    periodicity_sigma: float = 0.7,
    periodicity_thr: float = 0.15,
    # entropy params
    entropy_sigma: float = 20.0,
    shannon_thr_bits: float = 4.0,
    wiener_thr_log: float = -0.5,
    # plotting toggles
    plot_amplitude_figure: bool = True,
    plot_periodicity_figure: bool = True,
    plot_shannon_figure: bool = True,
    plot_wiener_figure: bool = True,
    # output
    output_json: str | None = None,
) -> Path:
    """Run both feature pipelines and dump combined JSON.

    Returns
    -------
    Path
        The path to the JSON file written.
    """

    npz_path = Path(npz_path)
    if output_json is None:
        output_json = npz_path.with_name(f"{npz_path.stem}_features.json")
    output_json = Path(output_json)

    # 1) amplitude + periodicity
    amp_per_data = _amp_func(
        npz_path,
        plot_amplitude_figure=plot_amplitude_figure,
        plot_periodicity_figure=plot_periodicity_figure,
        smoothing_sigma_amplitude=amp_sigma,
        smoothing_sigma_periodicity=periodicity_sigma,
        power_threshold=periodicity_thr,
    )

    # 2) Shannon + Wiener entropy
    entropy_data = _ent_func(
        npz_path,
        smoothing_sigma=entropy_sigma,
        shannon_threshold_bits=shannon_thr_bits,
        wiener_threshold_log=wiener_thr_log,
        plot_shannon_figure=plot_shannon_figure,
        plot_wiener_figure=plot_wiener_figure,
    )

    # Build combined dict (convert NumPy → Python lists for JSON)
    combined: Dict[str, Any] = {
        "source_file": str(npz_path),
        "amplitude_periodicity": _np_to_py(vars(amp_per_data)),
        "entropy": _np_to_py(vars(entropy_data)),
    }

    output_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"✔ Features written to {output_json}")
    return output_json

# ─────────────────────────────────────────────────────────────────────────────
# CLI guard  (allows %runfile with no args)  
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print(
            "\nThis script exposes a function `extract_song_features(npz_path, ...)`.\n"
            "Import it in Python or run from the command line, e.g.:\n"
            "    python extract_song_features.py recording.npz --sigma 10\n"
        )
        sys.exit(0)

    p = argparse.ArgumentParser(
        description="Extract amplitude/periodicity + entropy features to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz", help="Path to spectrogram .npz file")

    p.add_argument("--amp-sigma", type=float, default=20)
    p.add_argument("--periodicity-sigma", type=float, default=0.7)
    p.add_argument("--periodicity-thr", type=float, default=0.15)

    p.add_argument("--entropy-sigma", type=float, default=20)
    p.add_argument("--sthresh", type=float, default=4.0, help="Shannon threshold (bits)")
    p.add_argument("--wthresh", type=float, default=-0.5, help="Wiener log threshold")

    p.add_argument("--plot", action="store_true", help="Show diagnostic plots for both pipelines")
    p.add_argument("--out", help="Output JSON path (defaults to *_features.json)")

    args = p.parse_args()

    extract_song_features(
        args.npz,
        amp_sigma=args.amp_sigma,
        periodicity_sigma=args.periodicity_sigma,
        periodicity_thr=args.periodicity_thr,
        entropy_sigma=args.entropy_sigma,
        shannon_thr_bits=args.sthresh,
        wiener_thr_log=args.wthresh,
        plot_amp_periodicity=args.plot,
        plot_entropy=args.plot,
        output_json=args.out,
    )
