import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

json_path = "/Users/mirandahulsey-vincent/Documents/allPythonCode/BYOD_class/data_inputs/sample_GV_groundtruth_npzs_features.json"
# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Look at the first entry's amplitude string
first_entry = data[0]
amp_raw = first_entry["metrics"]["z_log_amp_band_smoothed"]

print("Amplitude string preview:")
print(amp_raw[:500])  # first 500 characters


def safe_parse_array(raw_str):
    try:
        # Remove brackets and replace all whitespace with commas
        clean = raw_str.replace("[", "").replace("]", "").replace("\n", " ")
        clean = re.sub(r"[^\d\.\-eE]+", ",", clean)  # keep only numbers and decimal symbols
        nums = [float(val) for val in clean.split(",") if val]
        return np.array(nums)
    except Exception as e:
        print(f"Parsing error: {e}")
        return np.array([])


def plot_periodicity_vs_amplitude_from_json(json_path, 
                                            amp_key="z_log_amp_band_smoothed", 
                                            per_key="periodicity_bandpass", 
                                            title="Periodicity vs. Amplitude"):
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_amps = []
    all_pers = []

    for entry in data:
        metrics = entry.get("metrics", {})
        amp_raw = metrics.get(amp_key, "")
        per_raw = metrics.get(per_key, "")

        amp = safe_parse_array(amp_raw)
        per = safe_parse_array(per_raw)

        if len(amp) == 0 or len(per) == 0:
            print(f"Skipping {entry['file_name']} due to empty array(s).")
            continue

        if len(amp) != len(per):
            print(f"Length mismatch in {entry['file_name']}: amp={len(amp)}, per={len(per)} — truncating to shortest.")
            min_len = min(len(amp), len(per))
            amp = amp[:min_len]
            per = per[:min_len]

        all_amps.extend(amp)
        all_pers.extend(per)

    if not all_amps or not all_pers:
        print("No valid data to plot.")
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(all_amps, all_pers, alpha=0.4, s=5, color='slateblue')
    plt.xlabel("Z-scored log amplitude (smoothed)")
    plt.ylabel("Periodicity (bandpass)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
