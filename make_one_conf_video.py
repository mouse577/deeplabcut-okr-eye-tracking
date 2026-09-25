import pandas as pd
import cv2
from pathlib import Path

# === CONFIGURATION ===
video_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/videos/eye_recording_20250424_192113_0007_stack_export.mp4")
video_dir = video_path.parent
video_stem = video_path.stem
pcutoff = 0.0  # Confidence threshold

# === FIND MATCHING .h5 FILE ===
h5_candidates = sorted(video_dir.glob(f"{video_stem}*shuffle1*.h5"))
if not h5_candidates:
    raise FileNotFoundError(f"❌ No matching .h5 file found for video: {video_stem}")
h5_path = h5_candidates[0]
print(f"🎯 Using H5 file: {h5_path.name}")

# === OUTPUT PATH ===
output_path = video_path.with_name(video_stem + "_labeled_conf.mp4")

# === LOAD DATA ===
df = pd.read_hdf(h5_path)
scorer = df.columns.get_level_values(0)[0]
bodyparts = list(set(df.columns.get_level_values(1)))

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
print(f"🔧 Writing to {output_path.name}")

for i in range(frame_count):
    success, frame = cap.read()
    if not success:
        print(f"⚠️ Failed to read frame {i}")
        break

    for bp in bodyparts:
        x = df[scorer][bp]["x"].iloc[i]
        y = df[scorer][bp]["y"].iloc[i]
        p = df[scorer][bp]["likelihood"].iloc[i]

        if p < pcutoff or pd.isna(x) or pd.isna(y):
            continue

        # Color code
        color = (0, 0, 255) if p < 0.3 else (0, 255, 255) if p < 0.6 else (0, 255, 0)
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        cv2.putText(frame, f"{bp}:{p:.2f}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out.write(frame)

cap.release()
out.release()

# === CHECK OUTPUT ===
if output_path.exists() and output_path.stat().st_size > 0:
    print(f"✅ Video saved: {output_path}")
else:
    print(f"❌ Failed to save video: {output_path}")

