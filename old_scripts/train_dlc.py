import deeplabcut

config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"


deeplabcut.train_network(
    config_path,
    shuffle=1,
    displayiters=100,
    saveiters=1000,
    maxiters=2000
)
