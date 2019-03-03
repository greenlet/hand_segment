# Hand segmentation models tests
## Prerequisites
Python packages needed:
 - tensorflow
 - keras
 - numpy
 - opencv-python

## Usage
1. Simple HLS color filter
`python run.py --color-threshold`
2. Gaussian mixture background subtractor
`python run.py --gauss-mixture`
3. K-nearest neighbours background subtractor
`python run.py --knn`
4. HGR-Net segmentation model (https://arxiv.org/abs/1806.05653)
`python run.py --hgr-net`

