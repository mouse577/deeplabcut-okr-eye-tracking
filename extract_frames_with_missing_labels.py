import pandas as pd
import cv2
from pathlib import Path

# === CHANGE THIS to your actual project folder ===
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")
video_dir = project_path / "videos"
output_dir = project_path / "labeled-data" / "missed-points"
pred_suffix = "DLC_resnet101_OKR_MICROBEADS_BASELINEshuffle1.h5"
required_bodyparts = [
    "left_pupil_corner",
    "right_pupil_corner",
    "top_pupil",
    "bottom_pupil",
    "center_pupil",
]
likelihood_threshold = 0.6
frame_skip = 1

output_dir.mkdir(parents=True, exist_ok=True)

for pred_file in video_dir.glob(f"*{pred_suffix}"):
    print(f"🔍 Checking: {pred_file.name}")
    df = pd.read_hdf(pred_file)
    video_name = pred_file.name.replace(pred_suffix, ".mp4")
    video_path = video_dir / video_name

    if not video_path.exists():
        print(f"⚠️  Video not found: {video_path}")
        continue

    scorer = df.columns.get_level_values(0)[0]
    n_frames = len(df)

    bad_frames = []
    for idx in range(0, n_frames, frame_skip):
        missing = 0
        for bp in required_bodyparts:
            likelihood = df[scorer][bp]["likelihood"].iloc[idx]
            if likelihood < likelihood_threshold:
                missing += 1
        if missing > 0:
            bad_frames.append(idx)

    print(f"❌ Found {len(bad_frames)} problematic frames.")

    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in bad_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            outname = f"{video_path.stem}_frame_{frame_idx:05d}.png"
            outpath = output_dir / outname
            cv2.imwrite(str(outpath), frame)
    cap.release()

print(f"\n✅ Done. Extracted frames saved to: {output_dir}")
print("➡️  You can now label them using the DLC GUI.")
