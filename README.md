# Hand segmentation models tests
## Prerequisites
Python packages needed:
 - tensorflow
 - keras
 - numpy
 - opencv-python

## Usage
The main script is started with command
`python run.py`
It will take video from the default webcam. To set source explicitly use flag `--source <source>`:
 - Webcam number N: `python run.py --source N`
 - File: `python run.py --source <path>/input.avi`
### Models
1. Simple HLS color filter with low Hue values
`python run.py --color-mask-light`
2. Simple HLS color filter with high Hue values
`python run.py --color-mask-dark`
3. Gaussian mixture background subtractor
`python run.py --gauss-mixture`
4. K-nearest neighbours background subtractor
`python run.py --knn`
5. HGR-Net segmentation model from https://arxiv.org/abs/1806.05653
`python run.py --hgr-net`
6. HGR-Net segmentation model from https://arxiv.org/abs/1806.05653 with dense ASPP from http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html
`python run.py --hgr-net-dense`

To save model output to file use flag `--save`. It will be saved to local directory `model_rec`

