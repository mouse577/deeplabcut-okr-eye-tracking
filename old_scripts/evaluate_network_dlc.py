import deeplabcut
from pathlib import Path

# Set the full path to your config.yaml
config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"

# Train the network
deeplabcut.train_network(
    config_path,
    shuffle=1,
    displayiters=100,
    saveiters=1000,
    maxiters=2000  # for quick test
)

# Evaluate the network after training
deeplabcut.evaluate_network(
    config_path,
    plotting=True,    # Generates box plots for errors
    show_errors=True  # Prints out train/test RMSE
)
