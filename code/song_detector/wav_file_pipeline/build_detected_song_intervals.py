#!/usr/bin/env python
#!/usr/bin/env python
"""
build_detected_song_intervals.py  ·  v2025‑07‑17b
================================================
Adds the *duration_seconds* field to every output record and
always echoes files that contain no qualifying song.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple

# ───────── tunables ─────────
MIN_PER_DUR = 0.75   # seconds
TOLERANCE   = 1.00   # seconds

# ───────── helpers ─────────
def _span_len(span: List[float]) -> float:
    return span[1] - span[0]

def _within_tol(a: Tuple[float, float],
                b: Tuple[float, float],
                tol: float) -> bool:
    return not (a[1] + tol < b[0] or b[1] + tol < a[0])

def _merge_overlaps(spans: List[Tuple[float, float]]
                    ) -> List[List[float]]:
    if not spans:
        return []
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        last = merged[-1]
        if s <= last[1] + 1e-6:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return merged

# ───────── core ─────────
def build_detected_song_intervals(
    features_json: str | Path,
    *,
    min_per_dur: float = MIN_PER_DUR,
    tol: float        = TOLERANCE,
) -> Path:
    features_json = Path(features_json).expanduser().resolve()
    if not features_json.is_file():
        raise FileNotFoundError(features_json)
    if not features_json.name.endswith("_features.json"):
        raise ValueError("Input must end with '_features.json'")

    out_path = features_json.with_name(
        features_json.stem.replace("_features", "_detected_song_intervals") + ".json"
    )

    with features_json.open() as f:
        feats = json.load(f)

    out_records = []
    for rec in feats:
        m          = rec["metrics"]
        duration   = rec.get("duration_seconds")  # ← NEW

        seeds = [tuple(p) for p in m["periodicity_detected_song_times"]
                 if _span_len(p) >= min_per_dur]  # threshold is ≥

        if not seeds:
            # still echo the file with duration_seconds
            out_records.append({
                "file_name":   rec["file_name"],
                "file_path":   rec["file_path"],
                "duration_seconds": duration,      # ← NEW
                "contains_song": False,
                "detected_song_times": [],
            })
            continue

        pool: List[Tuple[float, float]] = seeds.copy()

        # add nearby amp / Wiener spans (no extra length gating)
        for key in ("amplitude_detected_song_times",
                    "wiener_detected_song_times"):
            for span in m[key]:
                s = tuple(span)
                if any(_within_tol(s, seed, tol) for seed in seeds):
                    pool.append(s)

        pool.sort(key=lambda x: x[0])
        merged = _merge_overlaps(pool)

        out_records.append({
            "file_name":        rec["file_name"],
            "file_path":        rec["file_path"],
            "duration_seconds": duration,         # ← NEW
            "contains_song":    True,
            "detected_song_times": merged,
        })

    with out_path.open("w") as f:
        json.dump(out_records, f, indent=2)
    print(f"✅  Saved {out_path}")
    return out_path

# ───────── CLI ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert *_features.json ➜ *_detected_song_intervals.json "
                    "while retaining duration_seconds."
    )
    ap.add_argument("features_json", type=Path,
                    help="Path to *_features.json")
    ap.add_argument("--min_per", default=MIN_PER_DUR, type=float,
                    help="Minimum periodicity span (s)")
    ap.add_argument("--tol",     default=TOLERANCE, type=float,
                    help="Proximity window for amp/Wiener (s)")
    args = ap.parse_args()

    build_detected_song_intervals(
        args.features_json,
        min_per_dur=args.min_per,
        tol=args.tol,
    )


# =============================================================================
# from build_detected_song_intervals import build_detected_song_intervals
# json_path = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class_clean/data_inputs/USA5510_unsegmented_songs/55_subsample_features.json"
# build_detected_song_intervals(json_path, min_per_dur=0.5, tol=1.0)
# =============================================================================
