import deeplabcut
from pathlib import Path

# Step 1: Define config path
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"

# Step 2: Collect all .mp4 videos in the project's videos/ folder
project_path = Path(config_path).parent
video_dir = project_path / "videos"
video_list = list(video_dir.glob("*.mp4"))  # Change if you're using .avi

video_list_str = [str(v) for v in video_list]  # Convert Path objects to strings

# Step 3: Analyze all videos
deeplabcut.analyze_videos(
    config_path,
    videos=video_list_str,
    shuffle=1,
    save_as_csv=True
)

# Step 4: Create labeled videos with overlayed skeleton and points
deeplabcut.create_labeled_video(
    config_path,
    videos=video_list_str,
    shuffle=1,
    draw_skeleton=True
)