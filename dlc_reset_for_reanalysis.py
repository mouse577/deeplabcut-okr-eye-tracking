import shutil
from pathlib import Path

# ✅ Set your DLC project path
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")

# 🔹 Remove only analysis-related output files from videos/
video_dir = project_path / "videos"
patterns_to_delete = [
    "*_labeled.mp4",
    "*_labeled_conf.mp4",
    "*.h5",
    "*.csv",
    "*full.pickle",
    "*meta.pickle"
]

for pattern in patterns_to_delete:
    for f in video_dir.glob(pattern):
        if f.is_file():
            print(f"🗑️ Deleting file: {f}")
            f.unlink()

# 🔹 Clean labeled-data/*: keep CollectedData_*.h5/.csv and all .pngs
labeled_data_dir = project_path / "labeled-data"
if labeled_data_dir.exists():
    for subfolder in labeled_data_dir.glob("*"):
        if subfolder.is_dir():
            for f in subfolder.iterdir():
                if f.is_file():
                    keep = (
                        f.name.startswith("CollectedData_") and f.suffix in [".h5", ".csv"]
                    ) or f.suffix == ".png"
                    if not keep:
                        print(f"🗑️ Removing non-essential file: {f}")
                        f.unlink()

print("\n✅ Reanalysis cleanup complete. Training data and labels preserved.") 