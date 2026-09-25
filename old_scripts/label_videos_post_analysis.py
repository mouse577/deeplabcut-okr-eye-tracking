import deeplabcut
from pathlib import Path

# Path to your config.yaml
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"

# Locate all .mp4 videos in the project’s videos/ folder
project_path = Path(config_path).parent
video_dir = project_path / "videos"
video_list = list(video_dir.glob("*.mp4"))  # Change to *.avi if needed

# Convert Path objects to strings
video_list_str = [str(video) for video in video_list]

# Create labeled videos (overlay predicted points and skeleton)
deeplabcut.create_labeled_video(
    config_path,
    videos=video_list_str,
    shuffle=1,
    draw_skeleton=True
)
