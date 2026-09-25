import deeplabcut
from pathlib import Path
import yaml

# === CONFIGURATION ===
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"
missed_folder = "missed-points"
label_folder = Path(config_path).parent / "labeled-data" / missed_folder

# === Check for CollectedData files ===
csv_file = next(label_folder.glob("CollectedData_*.csv"), None)
h5_file = next(label_folder.glob("CollectedData_*.h5"), None)

if not csv_file or not h5_file:
    raise FileNotFoundError(f"Missing CollectedData_*.csv or .h5 in {label_folder}")

# === Load config.yaml ===
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# === Add the missed-points folder as a 'video set' entry if not present ===
video_set_key = str(label_folder)
if video_set_key not in cfg["video_sets"]:
    cfg["video_sets"][video_set_key] = {}  # No cropping applied

    # Save back the config.yaml
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"✅ Added '{missed_folder}' to video_sets in config.yaml.")

# === Merge new labels and recreate training dataset ===
print("\n🔀 Merging annotated datasets...")
deeplabcut.merge_datasets(config_path)

print("🛠️  Rebuilding training dataset...")
deeplabcut.create_training_dataset(config_path)

print("\n✅ New frames have been added to training. Ready to retrain!")
