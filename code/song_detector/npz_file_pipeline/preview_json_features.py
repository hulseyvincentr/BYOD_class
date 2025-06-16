# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import textwrap

# ── point this at the JSON you just wrote ─────────────────────────
json_path = Path("/Users/mirandahulsey-vincent/Documents/allPythonCode/"
                 "BYOD_class/data_inputs/sample_GV_groundtruth_npzs_features.json")

# ── load & take the first record ─────────────────────────────────
first_rec = json.loads(json_path.read_text())[0]

print(f"\nFile name : {first_rec['file_name']}")
print(f"Abs. path : {first_rec['file_path']}")
print("─" * 60)

metrics = first_rec["metrics"]

for key, value in metrics.items():
    # 1-D numeric arrays arrive as lists  → convert to ndarray for stats
    if isinstance(value, list) and value and isinstance(value[0], (int, float, bool)):
        arr = np.array(value, dtype=float)
        print(f"{key:30}  n={arr.size:6d}   "
              f"min={arr.min(): .3g}   max={arr.max(): .3g}   mean={arr.mean(): .3g}")
    # Nested lists (e.g. 2-D envelope spectrogram) – print shape only
    elif isinstance(value, list) and value and isinstance(value[0], list):
        rows = len(value)
        cols = len(value[0])
        print(f"{key:30}  2-D list  shape=({rows}, {cols})")
    else:  # scalar, string, or something else – pretty-print
        wrapped = textwrap.shorten(str(value), width=50, placeholder="…")
        print(f"{key:30}  {wrapped}")
