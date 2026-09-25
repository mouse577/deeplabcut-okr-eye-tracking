import pandas as pd
from pathlib import Path

def list_dlc_keypoints():
    # ✅ Hardcoded full path to your known DLC .h5 file
    h5_file = Path("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/videos/eye_recording_20250424_102517_0001_stack_exportDLC_Resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1_snapshot_110.h5")

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    print(f"\n📂 File: {h5_file.resolve()}")
    df = pd.read_hdf(h5_file)

    if isinstance(df.columns, pd.MultiIndex):
        print("🧠 Column structure: MultiIndex")
        for col in df.columns.to_flat_index():
            print(" -", col)
    else:
        print("🧠 Column structure: Flat")
        for col in df.columns:
            print(" -", col)

if __name__ == "__main__":
    list_dlc_keypoints()
