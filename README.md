## Introduction
This repository contains the Torch implementation for [ScaleNet: Guiding Object Proposal Generation in Supermarkets and Beyond](https://arxiv.org/abs/1704.06752) at ICCV 2017.
The code is built on [DeepMask and SharpMask](https://github.com/facebookresearch/deepmask).

### Citation
If you find ScaleNet useful in your research, please consider citing:
```
@inproceedings{ScaleNet,
   title = {ScaleNet: Guiding Object Proposal Generation in Supermarkets and Beyond},
   author = {Siyuan Qiao and Wei Shen and Weichao Qiu and Chenxi Liu and Alan Yuille},
   booktitle = {ICCV},
   year = {2017}
}
```

## Get Started
1. Install the following packages for [Torch](http://torch.ch): [COCO API](https://github.com/pdollar/coco), [image](https://github.com/torch/image), [tds](https://github.com/torch/tds), [cjson](https://github.com/clementfarabet/lua---json), [nnx](https://github.com/clementfarabet/lua---nnx), [optim](https://github.com/torch/optim), [inn](https://github.com/szagoruyko/imagine-nn), [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch)
2. Clone this repository
```bash
SCALENET=/desired/absolute/path/to/scalenet/ # set absolute path as desired
git clone https://github.com/joe-siyuan-qiao/ScaleNet.git $SCALENET
```
3. Prepare environment
```bash
cd $SCALENET
mkdir -p data intermediate pretrained/scalenet pretrained/sharpmask
```
Download the pretrained [ResNet-50](https://s3.amazonaws.com/deepmask/models/resnet-50.t7) to $SCALENET/pretrained if you want to train ScaleNet or SharpMask.
Move the downloaded MS COCO dataset to $SCALENET/data: $SCALENET/data/annotations, $SCALENET/data/train2014, $SCALENET/data/val2014.

## Training and Evaluation
```bash
th trainScaleNet.lua # For ScaleNet
th train.lua # For DeepMask and SharpMask. Please see their repo for the training details
```
The trained models will be found in $SCALENET/exps. Move the trained models for ScaleNet and SharpMask into the corresponding folders $SCALENET/pretrained/scalenet and $SCALENET/pretrained/sharpmask.
Our pretrained models can be found here: [ScaleNet](http://www.cs.jhu.edu/~syqiao/pretrained-model-scalenet.t7) and [SharpMask](http://www.cs.jhu.edu/~syqiao/pretrained-model-sharpmask.t7).
Next, we can evaluate the models on MS COCO.
```bash
th evalCocoBbox.lua
```
