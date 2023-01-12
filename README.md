# AIRLab khu Dataset Task Force 
- [Nuscene](https://www.nuscenes.org/)
- [Waymo](https://waymo.com/)
- [KITTI-360](https://github.com/autonomousvision/kitti360Scripts)

Welcome to the devkit of the [nuScenes](https://www.nuscenes.org/nuscenes) and [nuImages](https://www.nuscenes.org/nuimages) datasets.
![](https://www.nuscenes.org/public/images/road.jpg)


## Changelog
- Aug. 01, 2022: Add Nuscene folder and Camera_models

## Setup
We use a common devkit for nuScenes and nuImages.
The devkit is tested for Python 3.6 and Python 3.7.
To install Python, please check [here](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md#install-python).

Devkit is available and can be installed via [pip](https://pip.pypa.io/en/stable/installing/) :
```
pip install nuscenes-devkit
```

## Citation
Please use the following citation when referencing [nuScenes or nuImages](https://arxiv.org/abs/1903.11027):
```
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```

Please use the following citation when referencing
[Panoptic nuScenes or nuScenes-lidarseg](https://arxiv.org/abs/2109.03805):
```
@article{fong2021panoptic,
  title={Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
  author={Fong, Whye Kit and Mohan, Rohit and Hurtado, Juana Valeria and Zhou, Lubing and Caesar, Holger and
          Beijbom, Oscar and Valada, Abhinav},
  journal={arXiv preprint arXiv:2109.03805},
  year={2021}
}
```


