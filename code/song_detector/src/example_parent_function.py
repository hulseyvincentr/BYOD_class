# load_and_plot_GV_NPZ_file.py

from plot_spectrogram_with_song_highlight import plot_spectrogram_with_song_highlight

def analyze_npz_file(npz_path, plot=True):
    # Step 1: Load and (optionally) plot the spectrogram
    data = plot_NPZ_spectrogram_with_song_highlight(npz_path, plot_spectrogram=False)

    # Step 2: Perform additional analysis (example: total song duration)
    time_bin_width = (data.t[1] - data.t[0]) if len(data.t) > 1 else 0
    song_duration_sec = sum(data.ends - data.starts) * time_bin_width

    # Step 3: Return results or store them as needed
    print(f"Total song duration in file: {song_duration_sec:.2f} sec")

    return {
        "data": data,
        "song_duration_sec": song_duration_sec
    }

# ── Example usage ───────────────────────────────────────────────
if __name__ == "__main__":
    test_path = "your_npz_files/example_segment.npz"
    result = analyze_npz_file(test_path, plot=True)
