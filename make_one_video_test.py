import deeplabcut

config_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/config.yaml"
video_path = "/home/jg/Desktop/DLC_torch_projects/OKR_MICROBEADS_BASELINE-JAG-2025-04-30/videos/eye_recording_20250424_192113_0007_stack_export.mp4"

deeplabcut.create_labeled_video(
    config_path,
    videos=[video_path],
    shuffle=1,
    draw_skeleton=True,
    pcutoff=0.3
)