import shutil
from pathlib import Path

# Path to your DLC project
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")

# Remove model + train/test sets
models_dir = project_path / "dlc-models-pytorch"
if models_dir.exists():
    shutil.rmtree(models_dir)

# Remove evaluation results
eval_dir = project_path / "evaluation-results"
if eval_dir.exists():
    shutil.rmtree(eval_dir)

# Remove labeled frames
labeled_data_dir = project_path / "labeled-data"
if labeled_data_dir.exists():
    shutil.rmtree(labeled_data_dir)

# Remove labeled videos only (keep originals)
video_dir = project_path / "videos"
for labeled_video in video_dir.glob("*_labeled.mp4"):
    labeled_video.unlink()

print("✅ Cleaned all training artifacts. You’re ready to restart with new labels.")
