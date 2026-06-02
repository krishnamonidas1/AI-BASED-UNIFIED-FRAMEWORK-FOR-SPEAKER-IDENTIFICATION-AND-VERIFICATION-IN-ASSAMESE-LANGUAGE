import os
import librosa

DATASET_PATH = "Dataset"  # change only if your dataset folder name is different

total_duration_seconds = 0
speaker_durations = {}
total_files = 0

for speaker in sorted(os.listdir(DATASET_PATH)):
    speaker_path = os.path.join(DATASET_PATH, speaker)

    if os.path.isdir(speaker_path):
        speaker_total = 0
        speaker_files = 0

        for file in os.listdir(speaker_path):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(speaker_path, file)

                try:
                    duration = librosa.get_duration(path=file_path)
                    speaker_total += duration
                    total_duration_seconds += duration
                    speaker_files += 1
                    total_files += 1

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        speaker_durations[speaker] = {
            "duration": speaker_total,
            "files": speaker_files
        }

print("\nSpeaker-wise Duration")
print("-" * 50)

for speaker, info in speaker_durations.items():
    minutes = info["duration"] / 60
    hours = info["duration"] / 3600
    files = info["files"]

    print(f"{speaker}: {files} files | {minutes:.2f} minutes | {hours:.2f} hours")

print("\nTotal Dataset Duration")
print("-" * 50)
print(f"Total files: {total_files}")
print(f"Total seconds: {total_duration_seconds:.2f}")
print(f"Total minutes: {total_duration_seconds / 60:.2f}")
print(f"Total hours: {total_duration_seconds / 3600:.2f}")