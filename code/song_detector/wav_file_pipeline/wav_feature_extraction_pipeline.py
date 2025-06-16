#!/usr/bin/env python
"""
wav_feature_extraction_pipeline.py
==================================
Run BOTH acoustic-feature extractors on one or many *.wav* files and
(optionally) export their combined metrics to JSON.

The JSON contains the *full-length* NumPy arrays (converted to lists) so
nothing is truncated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np   # needed by _json_ready()

# ────────────────────────────────────────────────────────────────
# Local imports – your two wav-based extractors
# ────────────────────────────────────────────────────────────────
try:
    # amplitude + periodicity
    from calculate_wav_spectrogram_amp_and_periodicity import (
        calculate_wav_spectrogram_amplitude_and_periodicity,
    )
    # Shannon + Wiener entropy
    from calculate_wav_dual_entropy import (
        calculate_wav_spectrogram_entropy,
    )
except ModuleNotFoundError:
    print(
        "[ERROR] Missing extractor modules. Ensure\n"
        "        • calculate_wav_spectrogram_amp_and_periodicity.py and\n"
        "        • calculate_wav_dual_entropy.py\n"
        "        live in the same directory as this script."
    )
    raise

# ────────────────────────────────────────────────────────────────
# Helper – make everything JSON-serialisable
# ────────────────────────────────────────────────────────────────
def _json_ready(obj: Any):
    """Recursively turn any NumPy array into a vanilla list."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    return obj


# ────────────────────────────────────────────────────────────────
# Core wrappers
# ────────────────────────────────────────────────────────────────
def run_all_wav(wav_path: Path | str, *, show_plots: bool = False):
    """
    Run BOTH extractors on ONE .wav file and return
    ``(amp_per_data, entropy_data)``.
    """
    wav_path = Path(wav_path)

    amp_per = calculate_wav_spectrogram_amplitude_and_periodicity(
        wav_path,
        plot_amplitude_figure=show_plots,
        plot_periodicity_figure=show_plots,
    )
    ent = calculate_wav_spectrogram_entropy(
        wav_path,
        plot_shannon_figure=show_plots,
        plot_wiener_figure=show_plots,
    )
    return amp_per, ent


def _merge_metrics(amp_per, ent) -> Dict[str, Any]:
    """Flatten two dataclass instances into ONE dict."""
    return {**vars(amp_per), **vars(ent)}


def _output_path(folder: Path, out: Path | str | None) -> Path:
    """
    Default JSON path lives *next to* the processed folder:
        recordings/
        └─ features.json   ← default
    """
    if out is None:
        return folder.parent / f"{folder.name}_features.json"
    out = Path(out)
    return out if out.is_absolute() else folder.parent / out


def batch_extract_wav(
    folder: Path | str,
    *,
    output_json: Path | str | None = None,
    show_plots: bool = False,
) -> Path:
    """
    Process EVERY *.wav in *folder* and dump all metrics to a JSON file.
    Returns the absolute path of the JSON.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    dest = _output_path(folder, output_json)
    results: List[Dict[str, Any]] = []

    for wav in sorted(folder.glob("*.wav")):
        try:
            amp_per, ent = run_all_wav(wav, show_plots=show_plots)
            metrics = _json_ready(_merge_metrics(amp_per, ent))
            results.append(
                {
                    "file_name": wav.name,
                    "file_path": str(wav.resolve()),
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            # keep going if one file fails
            print(f"[WARN] Skipping {wav.name}: {exc}")
            continue

    with open(dest, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"✅  Saved batch metrics ➜ {dest.resolve()}")
    return dest.resolve()


# ────────────────────────────────────────────────────────────────
# Command-line interface
# ────────────────────────────────────────────────────────────────
def _cli():
    p = argparse.ArgumentParser(
        prog="wav_feature_extraction_pipeline.py",
        description="Extract amplitude/periodicity + entropy features "
                    "from .wav recordings (single file or batch).",
    )
    p.add_argument("target", type=Path,
                   help="Path to a .wav file OR a folder of .wav files.")
    p.add_argument("--plots", action="store_true",
                   help="Show diagnostic plots while processing.")
    p.add_argument("--batch", action="store_true",
                   help="Treat target as a folder and write a JSON summary.")
    p.add_argument("--out", type=Path, default=None,
                   help="Custom JSON name when using --batch.")
    args = p.parse_args()

    if args.batch:
        batch_extract_wav(args.target, output_json=args.out,
                          show_plots=args.plots)
    else:
        amp_per, ent = run_all_wav(args.target, show_plots=args.plots)

        print("\n=== Amplitude / Periodicity ===")
        print("Amplitude-detected spans :", amp_per.amplitude_detected_song_times)
        print("Periodicity-detected spans:", amp_per.periodicity_detected_song_times)

        print("\n=== Entropy ===")
        print("Shannon-detected spans   :", ent.shannon_detected_song_times)
        print("Wiener-detected spans    :", ent.wiener_detected_song_times)


if __name__ == "__main__":
    # Strip Spyder’s extra flags, if any
    sys.argv = [sys.argv[0]] + [
        a for a in sys.argv[1:] if not a.startswith("--*spyder*")
    ]
    _cli()


#from wav_feature_extraction_pipeline import run_all_wav, batch_extract_wav

#amp_per, ent = run_all_wav("bird1.wav", show_plots=True) #just generates for 1 wav file
#json_path = batch_extract_wav("recordings/", show_plots=False)