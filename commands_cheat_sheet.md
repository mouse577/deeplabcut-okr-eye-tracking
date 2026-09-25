pip install --pre "deeplabcut[gui]"
python -c "import torch; print(torch.cuda.is_available())"
pip freeze > requirements.txt
watch -n 0.5 nvidia-smi
python -m deeplabcut


# to reset for retraining new model with same labels and videos
python dlc_reset_for_retraining


# find video frames with missing label predictions, extract frames and add to training data for labelling
python extract_frames_with_missing_labels.py
python -m deeplabcut


# label frames in DLC GUi
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(config_path, maxiters=...)

# add new labeled frames to training data
python add_missed_frames_to_training.py







