import deeplabcut
from pathlib import Path

# Path to your project config.yaml
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"

# Automatically collect all .mp4 videos from the 'videos/' folder
project_path = Path(config_path).parent
video_dir = project_path / "videos"
video_list = list(video_dir.glob("*.mp4"))  # Change to *.avi if needed

# Analyze videos
deeplabcut.analyze_videos(
    config_path,
    videos=[str(v) for v in video_list],  # convert Path objects to strings
    shuffle=1,
    save_as_csv=True
)

