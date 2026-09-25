import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")
video_dir = project_path / "videos"
pcutoff = 0.0  # Minimum likelihood to show a point

# === STEP 1: Get all video files ===
# === STEP 1: Get only original unprocessed .mp4 videos ===
video_list = sorted([
    v for v in video_dir.glob("*.mp4")
    if v.is_file() and "labeled" not in v.stem and "conf" not in v.stem
])


if not video_list:
    print("❌ No video files found.")
    exit()

# === STEP 2: Create confidence overlay videos ===
print("🎞️ Creating confidence overlay videos...")

for video_path in video_list:
    base_stem = video_path.stem
    # Adjust pattern to match the correct h5 file naming format
    h5_path = video_dir / f"{base_stem}DLC_Resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1_snapshot_070.h5"

    if not h5_path.exists():
        print(f"❌ Skipping, missing prediction: {h5_path.name}")
        continue

    try:
        df = pd.read_hdf(h5_path)
    except Exception as e:
        print(f"❌ Error reading {h5_path.name}: {e}")
        continue

    if df.isnull().all().all():
        print(f"⚠️ Empty predictions in {h5_path.name}")
        continue

    scorer = df.columns.get_level_values(0)[0]
    bodyparts = list(set(df.columns.get_level_values(1)))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = video_dir / (video_path.stem + "_labeled_conf.mp4")
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
    print(f"🔧 Writing: {out_path.name}")

    for i in tqdm(range(frame_count), desc=f"{video_path.stem}", unit="frame"):
        success, frame = cap.read()
        if not success:
            break

        for bp in bodyparts:
            x = df[scorer][bp]["x"].iloc[i]
            y = df[scorer][bp]["y"].iloc[i]
            p = df[scorer][bp]["likelihood"].iloc[i]

            if p < pcutoff or pd.isna(x) or pd.isna(y):
                continue

            color = (0, 0, 255) if p < 0.3 else (0, 255, 255) if p < 0.6 else (0, 255, 0)
            cv2.putText(frame, "*", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"✅ Saved: {out_path.name}")
    else:
        print(f"❌ Failed to save: {out_path.name}")
