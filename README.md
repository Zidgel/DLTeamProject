# DLTeamProject
Team Project for Deep Learning - Georgia Tech CS7643

Team: Comic Sans (Benjamin Kozel, Don Pham, Hayden Schennum)

Re-implementation of the following style transfer techniques:

Neural Style Transfer: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

Fast Style Transfer: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf

Adaptive Instance Normalization: https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf


conda env create -f cs7643_teamproj_env.yaml
conda activate cs7643_teamproj

The cs7643_teamproj_env doesn't include pytorch / torchvision; to get those, you have to go to https://pytorch.org/ and input your OS/platform info
e.g. with cs7643_teamproj activated, (the specific url varies depending on your setup)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
