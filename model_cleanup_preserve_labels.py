import shutil
from pathlib import Path

# Set your project directory here
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")

# Paths to delete
folders_to_delete = [
    project_path / "dlc-models",
    project_path / "dlc-models-pytorch",
    project_path / "training-datasets",
    project_path / "evaluation-results",
    project_path / "evaluation-results-pytorch"
]

# Delete entire folders
for folder in folders_to_delete:
    if folder.exists():
        print(f"🗑️  Deleting {folder}")
        shutil.rmtree(folder)

# Delete predicted outputs and labeled videos in videos/
video_dir = project_path / "videos"
for ext in ["*.h5", "*.csv", "*full.pickle", "*meta.pickle", "*_labeled.mp4"]:
    for f in video_dir.glob(ext):
        print(f"🗑️  Deleting {f}")
        f.unlink()

print("✅ Cleanup complete. Manual labels in 'labeled-data/' are preserved.")
