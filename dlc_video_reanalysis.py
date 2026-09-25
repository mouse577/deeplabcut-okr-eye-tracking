import deeplabcut
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm  # ✅ Added tqdm

# === CONFIG ===
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"
project_path = Path(config_path).parent
video_dir = project_path / "videos"
pcutoff = 0.0

# === STEP 1: Get all valid .mp4 videos ===
video_list = sorted([v for v in video_dir.glob("*.mp4") if cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_WIDTH) > 0])

if not video_list:
    print("❌ No valid videos found.")
    exit()

# === STEP 2: Run DLC analysis on all videos ===
print("🎥 Analyzing videos...")
deeplabcut.analyze_videos(config_path, videos=[str(v) for v in video_list], shuffle=1, save_as_csv=True)

# === STEP 3: Rename h5 files to include snapshot suffix if needed ===
for h5file in video_dir.glob("*shuffle1.h5"):
    if "_snapshot" not in h5file.name:
        new_name = h5file.with_name(h5file.stem + "_snapshot_070.h5")
        h5file.rename(new_name)
        print(f"✅ Renamed {h5file.name} → {new_name.name}")

# === STEP 4: Generate labeled videos with skeleton overlay ===
print("🎞️ Creating labeled videos...")
deeplabcut.create_labeled_video(config_path, videos=[str(v) for v in video_list], shuffle=1, draw_skeleton=True, pcutoff=0.3)

# === STEP 5: Create confidence overlay videos ===
print("🎞️ Creating confidence overlay videos...")

for video_path in video_list:
    base_stem = video_path.stem
    h5_path = video_dir / f"{base_stem}DLC_Resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1_snapshot_070.h5"

    if not h5_path.exists():
        print(f"❌ Skipping, missing prediction: {base_stem}")
        continue

    try:
        df = pd.read_hdf(h5_path)
    except Exception as e:
        print(f"❌ Error reading {h5_path.name}: {e}")
        continue

    if df.isnull().all().all():
        print(f"⚠️ Empty predictions: {h5_path.name}")
        continue

    scorer = df.columns.get_level_values(0)[0]
    bodyparts = list(set(df.columns.get_level_values(1)))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = video_dir / f"{base_stem}_labeled_conf.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"🔧 Writing: {out_path.name}")

    for i in tqdm(range(frame_count), desc=f"{base_stem}", unit="frame"):  # ✅ Progress bar
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


