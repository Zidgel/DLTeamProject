# DLTeamProject
Team Project for Deep Learning 7643

conda env create -f cs7643_teamproj_env.yaml
conda activate cs7643_teamproj

The cs7643_teamproj_env doesn't include pytorch / torchvision; to get those, you have to go to https://pytorch.org/ and input your OS/platform info
e.g. with cs7643_teamproj activated, (the specific url varies depending on your setup)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126