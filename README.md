# Unsupervised Multi-View Object Segmentation Using Radiance Field Propagation
## [Project page](https://xinhangliu.com/nerf_seg) |  [Paper](https://arxiv.org/pdf/2210.00489.pdf)
This repository contains a PyTorch implementation for NeurIPS 2022 paper: [Unsupervised Multi-View Object Segmentation Using Radiance Field Propagation](https://arxiv.org/pdf/2210.00489.pdf). Our work present one of the first unsupervised approaches for tackling 3D real scene object segmentation for neural radiance field (NeRF) without any supervision, annotations, or other cues such as 3D bounding boxes and prior knowledge of object class.

<div>
<img src="https://xinhangliu.com/img/method.jpg" height="360"/>

</div>


## Installation

#### Tested on Ubuntu 22.04 + Pytorch 1.13.0

Install environment:
```
conda create -n nerf_seg python=3.8
conda activate nerf_seg
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
```

## Data Preparation
Please download the following datasets and remember to set correct data path in `configs`.
- [LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- [CO3D](https://github.com/facebookresearch/co3d)

Our method relies on unsupervised single image segmentation algorithms to get good initialization. Recommended methods are [iem](https://github.com/lolemacs/iem-code) and [ReDO](https://github.com/mickaelChen/ReDO). We provide a set of initial labels [here](https://drive.google.com/drive/folders/1kjRf-PSoP0uGWYql_Ty3TFjFRHt6R32W?usp=sharing). 

Please the data folder as follows:
```
<case_name>
|-- poses_bounds.npy         # camera parameters
|-- images_4
    |-- image000.png        # target image for each view
    |-- image001.png
    ...
|-- masks
    |-- image000.png        # target mask for each view
    |-- image001.png
    ...
```

## Quick Start
The training script is in `train.py`, to train our method on a scene in the LLFF dataset:

```
python train.py --config configs/fortress.txt
```

## TODO
- [ ] EM refinement for multi-object scenes
- [ ] Synthetic multi-object dataset
- [ ] Interface for [CO3D](https://github.com/facebookresearch/co3d)

## Acknowlegements
Thanks [TensoRF](https://github.com/apchenstu/TensoRF), [semantic_nerf](https://github.com/Harry-Zhi/semantic_nerf), and [st-nerf](https://github.com/DarlingHang/st-nerf) for providing awesome open source libraries.

## Citation
If you find our code or paper helps, please consider citing:
```
@inproceedings{liu2022unsupervised,
    title={Unsupervised Multi-View Object Segmentation Using Radiance Field Propagation},
    author={Liu, Xinhang and Chen, Jiaben and Yu, Huai and Tai, Yu-Wing and Tang, Chi-Keung},
    booktitle={Advances in Neural Information Processing Systems},
    volume = {35},
    url = {https://arxiv.org/pdf/2210.00489.pdf},
    year={2022}
    }
```
