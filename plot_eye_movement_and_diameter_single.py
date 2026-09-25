import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_eye_data(h5_path):
    h5_path = Path(h5_path)
    df = pd.read_hdf(h5_path)

    # Grab model prefix (first level of MultiIndex)
    model_name = df.columns.levels[0][0]

    # Helper to access columns
    def get_col(bodypart, coord):
        return df[(model_name, bodypart, coord)]

    # Get pupil center position
    pupil_x = get_col("pupil_center", "x")
    pupil_y = get_col("pupil_center", "y")

    # Compute displacement (from initial center)
    init_x, init_y = pupil_x.iloc[0], pupil_y.iloc[0]
    displacement = np.sqrt((pupil_x - init_x) ** 2 + (pupil_y - init_y) ** 2)

    # Compute pupil diameter (horizontal and vertical)
    left_x = get_col("pupil_left_edge", "x")
    right_x = get_col("pupil_right_edge", "x")
    top_y = get_col("pupil_top_edge", "y")
    bottom_y = get_col("pupil_bottom_edge", "y")

    diameter_h = np.abs(right_x - left_x)
    diameter_v = np.abs(bottom_y - top_y)
    diameter_avg = (diameter_h + diameter_v) / 2

    frames = np.arange(len(df))

    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Eye Movement & Pupil Size\n{h5_path.name}", fontsize=14)

    axs[0].plot(frames, pupil_x, label='Pupil X', color='blue')
    axs[0].plot(frames, pupil_y, label='Pupil Y', color='green')
    axs[0].set_ylabel("Center Position (px)")
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(frames, displacement, label="Displacement", color='purple')
    axs[1].set_ylabel("Movement (px)")
    axs[1].legend(); axs[1].grid(True)

    axs[2].plot(frames, diameter_h, label='Horizontal', color='orange')
    axs[2].plot(frames, diameter_v, label='Vertical', color='red')
    axs[2].set_ylabel("Diameter (px)")
    axs[2].legend(); axs[2].grid(True)

    axs[3].plot(frames, diameter_avg, label='Avg Diameter', color='black')
    axs[3].set_xlabel("Frame #")
    axs[3].set_ylabel("Avg Diameter (px)")
    axs[3].legend(); axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_eye_data("/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/videos/eye_recording_20250424_102517_0001_stack_exportDLC_Resnet101_OKR_MICROBEADS_BASELINEApr30shuffle1_snapshot_110.h5")


