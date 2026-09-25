pip install --pre "deeplabcut[gui]"

python -c "import torch; print(torch.cuda.is_available())"

pip freeze > requirements.txt

watch -n 0.5 nvidia-smi

python -m deeplabcut


