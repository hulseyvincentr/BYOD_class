#!/usr/bin/env python
"""
npz_feature_extraction_pipeline.py
==============================
Run two acoustic-feature extractors on one or many *.npz* spectrogram
files and (optionally) export their combined metrics to JSON.

The JSON now contains *full-length* lists instead of truncated strings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np                     # NEW – needed by _json_ready()

# ────────────────────────────────────────────────────────────────────
# Local imports
# ────────────────────────────────────────────────────────────────────
try:
    from calculate_npz_spectrogram_amp_and_periodicity import (
        calculate_npz_spectrogram_amplitude_and_periodicity,
    )
    from calculate_npz_spectrogram_entropy import (
        calculate_npz_spectrogram_entropy,
    )
except ModuleNotFoundError:
    print(
        "[ERROR] Could not import the two extractor modules. Ensure\n"
        "        *calculate_npz_spectrogram_amp_and_periodicity.py* and\n"
        "        *calculate_npz_spectrogram_entropy.py* are in the same "
        "directory as this script."
    )
    raise

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _json_ready(obj: Any):
    """
    Recursively convert every NumPy array to a vanilla Python list so
    json.dump writes the full data instead of a truncated string.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    return obj


def run_all(npz_path: Path | str, *, show_plots: bool = False):
    """Return (amp_per_data, entropy_data) for a single .npz file."""
    npz_path = Path(npz_path)

    amp_per = calculate_npz_spectrogram_amplitude_and_periodicity(
        str(npz_path),
        plot_amplitude_figure=show_plots,
        plot_periodicity_figure=show_plots,
    )
    ent = calculate_npz_spectrogram_entropy(
        str(npz_path),
        plot_shannon_figure=show_plots,
        plot_wiener_figure=show_plots,
    )
    return amp_per, ent


def _merge_metrics(amp_per, ent) -> Dict[str, Any]:
    """Merge two dataclass instances into a single dict."""
    return {**vars(amp_per), **vars(ent)}


def _output_path(folder: Path, out: Path | str | None) -> Path:
    """Return a JSON path saved *next to* *folder*."""
    if out is None:
        return folder.parent / f"{folder.name}_features.json"
    out = Path(out)
    return out if out.is_absolute() else folder.parent / out


def batch_extract(
    folder: Path | str,
    *,
    output_json: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """Process every *.npz* in *folder* and dump metrics to JSON."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    dest = _output_path(folder, output_json)
    results: List[Dict[str, Any]] = []

    for npz in sorted(folder.glob("*.npz")):
        try:
            amp_per, ent = run_all(npz, show_plots=show_plots)
            metrics = _json_ready(_merge_metrics(amp_per, ent))
            results.append(
                {
                    "file_name": npz.name,
                    "file_path": str(npz.resolve()),
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            print(f"[WARN] Skipping {npz.name}: {exc}")
            continue

    with open(dest, "w") as fh:              # ← no default=str !!
        json.dump(results, fh, indent=2)

    print(f"✅  Saved batch metrics ➜ {dest.resolve()}")
    return dest.resolve()

# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser(
        prog="npz_feature_extraction_pipeline.py",
        description="Extract amplitude/periodicity + entropy features "
                    "from .npz spectrograms (single file or batch).",
    )
    p.add_argument("target", type=Path,
                   help="Path to a .npz file OR a folder of .npz files.")
    p.add_argument("--plots", action="store_true",
                   help="Show diagnostic plots while processing.")
    p.add_argument("--batch", action="store_true",
                   help="Treat target as a folder and write a JSON summary.")
    p.add_argument("--out", type=Path, default=None,
                   help="Custom JSON name when using --batch.")
    args = p.parse_args()

    if args.batch:
        batch_extract(args.target, output_json=args.out, show_plots=args.plots)
    else:
        amp_per, ent = run_all(args.target, show_plots=args.plots)
        print("\n=== Amplitude / Periodicity ===")
        print("Amplitude-detected spans :", amp_per.amplitude_detected_song_times)
        print("Periodicity-detected spans:", amp_per.periodicity_detected_song_times)
        print("\n=== Entropy ===")
        print("Shannon-detected spans   :", ent.shannon_detected_song_times)
        print("Wiener-detected spans    :", ent.wiener_detected_song_times)


if __name__ == "__main__":
    # Strip Spyder’s extra CLI flags if present
    sys.argv = [sys.argv[0]] + [
        a for a in sys.argv[1:] if not a.startswith("--*spyder*")
    ]
    _cli()


#json_path = batch_extract(Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class/data_inputs/sample_GV_groundtruth_npzs"), show_plots=True)