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
4. HGR-Net segmentation model from https://arxiv.org/abs/1806.05653
`python run.py --hgr-net`
5. HGR-Net segmentation model from https://arxiv.org/abs/1806.05653 with dense ASPP from http://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html
`python run.py --hgr-net-dense`

