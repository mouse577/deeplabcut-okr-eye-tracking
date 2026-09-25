import deeplabcut
from pathlib import Path
import pandas as pd
import cv2
import yaml
import time
import math

# Step 1: Define config path
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"
project_path = Path(config_path).parent
pcutoff_overlay = 0.0  # For confidence overlay videos


def wait_for_and_patch_pytorch_config(project_path, epochs=200, save_epochs=100, batch_size=None, timeout=10):
    search_path = project_path / "dlc-models-pytorch" / "iteration-0"
    for _ in range(timeout * 2):  # Try for 10s (every 0.5s)
        config_files = list(search_path.glob("*/train/pytorch_config.yaml"))
        if config_files:
            break
        time.sleep(0.5)
    else:
        print("❌ pytorch_config.yaml not found after waiting.")
        return

    config_path = config_files[0]
    print(f"🛠️ Patching: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["train_settings"]["epochs"] = epochs
    if batch_size is not None:
        cfg["train_settings"]["batch_size"] = batch_size
    if "runner" in cfg and "snapshots" in cfg["runner"]:
        cfg["runner"]["snapshots"]["save_epochs"] = save_epochs

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"✅ Patched {config_path.name}: epochs={epochs}, save_epochs={save_epochs}"
          + (f", batch_size={batch_size}" if batch_size else ""))
    


def estimate_training_progress(project_path, batch_size=12):
    # Locate train folder
    train_dir = next(project_path.glob("dlc-models-pytorch/iteration-0/*train*"), None)
    if not train_dir:
        print("❌ Could not find training directory.")
        return

    # Load train set size
    label_file = next((project_path / "labeled-data").glob("*/CollectedData_*.csv"), None)
    if not label_file:
        print("❌ Could not find labeled CSV to estimate dataset size.")
        return

    df = pd.read_csv(label_file)
    num_frames = len(df)
    iterations_per_epoch = math.ceil(num_frames / batch_size)

    print(f"\n🔢 Estimated training progress:")
    print(f"  Labeled frames: {num_frames}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations per epoch: {iterations_per_epoch}")

    # Tail training log
    log_path = train_dir / "train.log"
    if not log_path.exists():
        print("⚠️ No train.log file found. Cannot track live progress.")
        return

    print(f"📡 Monitoring: {log_path} (press Ctrl+C to stop)\n")
    try:
        with open(log_path, "r") as f:
            f.seek(0, 2)  # Go to end of file
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                if "Epoch" in line and "train loss" in line:
                    # Extract epoch number
                    parts = line.split()
                    for part in parts:
                        if "Epoch" in part:
                            try:
                                epoch_str = part.split("/")[0].replace("Epoch", "")
                                epoch = int(epoch_str)
                                iter_total = epoch * iterations_per_epoch
                                print(f"🟢 Epoch {epoch}, estimated iteration: {iter_total}")
                            except:
                                continue
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")

    

# Step 1.5: Rebuild the training dataset (required if labels/config changed)
print("\n🛠️ Creating training dataset...")
deeplabcut.create_training_dataset(config_path)
wait_for_and_patch_pytorch_config(project_path, epochs=200, save_epochs=100, batch_size=8)


# Step 2: Train the network
print("\n🚀 Training the network...")
deeplabcut.train_network(
    config_path,
    shuffle=1,
    displayiters=100,
    saveiters=1000,
    maxiters=20000  # Set higher for full training
)

estimate_training_progress(project_path, batch_size=12)

# Step 3: Evaluate the network
print("\n📊 Evaluating the trained model...")
deeplabcut.evaluate_network(
    config_path,
    plotting=True,
    show_errors=True
)

# Step 4: Collect all .mp4 videos in the project's videos/ folder
video_dir = project_path / "videos"
video_list = list(video_dir.glob("*.mp4"))
video_list_str = [str(v) for v in video_list]

# Step 5: Analyze all videos
print("\n🎥 Analyzing videos...")
deeplabcut.analyze_videos(
    config_path,
    videos=video_list_str,
    shuffle=1,
    save_as_csv=True
)

# Step 6: Create labeled videos with overlaid skeletons
print("\n🎞️ Creating labeled videos...")
deeplabcut.create_labeled_video(
    config_path,
    videos=video_list_str,
    shuffle=1,
    draw_skeleton=True,
    pcutoff=0.3 
)

# Step 7: Create videos with confidence overlay
print("\n🎞️ Creating labeled videos with confidence overlay...")

skeleton = [
    ("pupil_left_edge", "pupil_center"),
    ("pupil_right_edge", "pupil_center"),
    ("pupil_top_edge", "pupil_center"),
    ("pupil_bottom_edge", "pupil_center"),
]

prediction_suffix = "DLC_resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1"
pcutoff_overlay = 0.0  # Make sure this is defined before use!

for pred_file in video_dir.glob(f"*{prediction_suffix}*.h5"):
    df = pd.read_hdf(pred_file)
    if df.isnull().all().all():
        print(f"❌ Prediction file {pred_file.name} contains no valid data.")
        continue
    scorer = df.columns.get_level_values(0)[0]
    bodyparts = list(set(df.columns.get_level_values(1)))
    
    base_name = pred_file.name.replace(prediction_suffix, "").split(".h5")[0]
    video_file = video_dir / f"{base_name}.mp4"

    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = video_file.with_name(video_file.stem + "_labeled_conf.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    print(f"▶️ Annotating with confidence: {video_file.name} -> {output_file.name}")

    for i in range(frame_count):  # FIXED INDENTATION — this loop must be indented
        success, frame = cap.read()
        if not success:
            break

        # 🔵 STEP 1: Draw keypoints with color-coded confidence
        for bp in bodyparts:
            x = df[scorer][bp]["x"].iloc[i]
            y = df[scorer][bp]["y"].iloc[i]
            p = df[scorer][bp]["likelihood"].iloc[i]

            if p < pcutoff_overlay:
                continue

            # Color based on confidence
            if p < 0.3:
                color = (0, 0, 255)       # 🔴 Red
            elif p < 0.6:
                color = (0, 255, 255)     # 🟡 Yellow
            else:
                color = (0, 255, 0)       # 🟢 Green

            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            cv2.putText(frame, f"{bp}:{p:.2f}", (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 🟡 STEP 2: Draw skeleton lines (after points)
        for a, b in skeleton:
            p1 = df[scorer][a]["likelihood"].iloc[i]
            p2 = df[scorer][b]["likelihood"].iloc[i]
            if p1 >= pcutoff_overlay and p2 >= pcutoff_overlay:
                x1 = int(df[scorer][a]["x"].iloc[i])
                y1 = int(df[scorer][a]["y"].iloc[i])
                x2 = int(df[scorer][b]["x"].iloc[i])
                y2 = int(df[scorer][b]["y"].iloc[i])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # 🟦 STEP 3: Add confidence legend
        legend_items = [
            ("≥ 0.6", (0, 255, 0), "High"),
            ("0.3–0.6", (0, 255, 255), "Moderate"),
            ("< 0.3", (0, 0, 255), "Low")
        ]
        for j, (text, color, label) in enumerate(legend_items):
            y_offset = 20 + j * 20
            cv2.circle(frame, (10, y_offset), 5, color, -1)
            cv2.putText(frame, f"{label} ({text})", (20, y_offset + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ⏱ STEP 4: Add frame number
        cv2.putText(frame, f"Frame {i+1}/{frame_count}", (width - 160, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


        # ✅ Write the annotated frame
        out.write(frame)

    cap.release()
    out.release()
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"✅ File saved successfully: {output_file}")
    else:
        print(f"❌ File not saved: {output_file}")
    print(f"✅ Saved: {output_file}")
