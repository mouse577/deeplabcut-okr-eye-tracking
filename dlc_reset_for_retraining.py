import shutil
from pathlib import Path

# ✅ Set your DLC project path
project_path = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30")

# 🔹 Remove entire folders (training, models, evaluation)
folders_to_delete = [
    project_path / "dlc-models",
    project_path / "dlc-models-pytorch",
    project_path / "training-datasets",
    project_path / "evaluation-results",
    project_path / "evaluation-results-pytorch",
]

for folder in folders_to_delete:
    if folder.exists():
        print(f"🗑️ Deleting folder: {folder}")
        shutil.rmtree(folder)

# 🔹 Remove iteration-* subfolders under any model folder (just in case)
for model_root in project_path.glob("dlc-models*/iteration-*"):
    print(f"🗑️ Deleting model iteration: {model_root}")
    shutil.rmtree(model_root)

# 🔹 Remove labeled videos and analysis output files from videos/
video_dir = project_path / "videos"
for pattern in ["*_labeled.mp4", "*.h5", "*.csv", "*full.pickle", "*meta.pickle"]:
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

print("\n✅ Cleanup complete. Manual labels and PNG frames preserved in 'labeled-data/'.")
print("You’re now ready to recreate the training dataset and retrain the model.")

