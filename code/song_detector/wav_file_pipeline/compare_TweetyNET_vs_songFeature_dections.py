# -*- coding: utf-8 -*-

import json
from pathlib import Path

def compare_subset_song_detection_and_export(your_json_path,
                                             george_json_path,
                                             output_json_path):
    # Load your pipeline data
    with open(your_json_path, "r") as f:
        your_data = json.load(f)

    # Load George's pipeline data
    with open(george_json_path, "r") as f:
        george_data = json.load(f)

    # Build lookup dictionaries
    your_dict = {
        entry["file_name"]: {
            "contains_song": entry["contains_song"],
            "file_path": entry.get("file_path", "unknown")
        }
        for entry in your_data
    }

    george_dict = {
        entry["filename"]: entry["song_present"]
        for entry in george_data
    }

    # Only compare files that exist in *your* JSON
    mismatches = []
    for fname, yinfo in your_dict.items():
        if fname not in george_dict:
            continue  # Skip files not in George's data

        y_status = yinfo["contains_song"]
        g_status = george_dict[fname]

        if y_status != g_status:
            mismatches.append({
                "file_name": fname,
                "file_path": yinfo["file_path"],
                "your_pipeline": y_status,
                "george_pipeline": g_status
            })

    # Save mismatches
    output_path = Path(output_json_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mismatches, f, indent=2)

    # Summary
    print(f"✅ Compared {len(your_dict)} files from your dataset.")
    print(f"⚠️  Found {len(mismatches)} mismatches (George had different status).")
    print(f"💾 Mismatches saved to: {output_path}")

    return mismatches


"""
your_json_path = "/Volumes/my_own_SSD/USA5288/0/0_detected_song_intervals.json"
george_json_path = "/Volumes/my_own_SSD/song_detection.json"
mismatched_files = compare_song_detection(your_json_path, george_json_path)
"""