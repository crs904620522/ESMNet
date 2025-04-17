# ESMNet

###### *PyTorch implementation of EAAI paper: "Stereo Matching on Epipolar Plane Image for Light Field Depth Estimation via Oriented Structurer"*.

[Paper]([Wait])
#### Requirements
et
- python 3.6
- pytorch 1.8.0
- ubuntu 18.04

### Installation

First you have to make sure that you have all dependencies in place. 

You can create an anaconda environment called ESMNet using

```
conda env create -f ESMNet.yaml
conda activate ESMNet
```

##### Dataset: 

Light Field Dataset: We use [HCI 4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/) for training and test. Please first download light field dataset, and put them into corresponding folders in ***data/HCInew***.



## ESMNet

##### Model weights(https://drive.google.com/drive/folders/1PS0S5oX7ylb3ONV8wxxAyDLZj2SvxWoJ?usp=sharing).

##### To train, run:

```
python train.py --config configs/HCInew/ESMNet.yaml 
```

##### To generate, run:

```
python generate.py --config configs/pretrained/HCInew/ESMNet_pretrained.yaml 
```



**If you find our code or paper useful, please consider citing:**
```
@article{chen2025stereo,
  title={Stereo matching on epipolar plane image for light field depth estimation via oriented structure},
  author={Chen, Rongshan and Sheng, Hao and Cong, Ruixuan and Yang, Da and Cui, Zhenglong and Wang, Sizhe and Ke, Wei},
  journal={Engineering Applications of Artificial Intelligence},
  volume={151},
  pages={110608},
  year={2025},
  publisher={Elsevier}
}
```
