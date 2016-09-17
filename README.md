This repo is forked (downloaded actually) from D. Pathak's [repo](https://github.com/pathak22/context-encoder) aiming at adapting it to video inpainting task.

The following is the original README.md, and the video inpainting method is under development. Unfortunately I can't promise that the original README still works....

Planned: 2D-CNN for pretraining and 3D-CNN for real inpainting network. 2D-CNN to catch the main spatial structure (and thus the resolution is lower) and 3D-CNN to catch the temporal and more delicate structure of the input (and thus the spatial resolution is higher).
2D-CNN receives 128x128 input and output 64x64 patch which is the center of the input.
3D-CNN receives 20x128x128 input and output 20x128x128 patch. The patches with context (20x256x256 resized to 20x128x128) go through 2D-CNN first and 

## Context Encoders: Feature Learning by Inpainting
[Project Website](http://cs.berkeley.edu/~pathak/context_encoder/)

This is the training code for our CVPR 2016 paper on Context Encoders for learning deep feature representation in an unsupervised manner by image inpainting. This code is adapted from an initial fork of [Soumith's DCGAN](https://github.com/soumith/dcgan.torch) implementation. Scroll down to try out a quick demo or train your own inpainting models!

If you find Context Encoders useful in your research, please cite:

    @inproceedings{pathakCVPR16context,
        Author = {Pathak, Deepak and Kr\"ahenb\"uhl, Philipp and Donahue, Jeff and Darrell, Trevor and Efros, Alexei},
        Title = {Context Encoders: Feature Learning by Inpainting},
        Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2016}
    }

### Contents
1. [Semantic Inpainting Demo](#1-semantic-inpainting-demo)
2. [Train Context Encoders](#2-train-context-encoders)
3. [Download Features Caffemodel](#3-download-features-caffemodel)
4. [TensorFlow Implementation](#4-tensorflow-implementation)

### 1) Semantic Inpainting Demo

Inpainting using context encoder trained jointly with reconstruction and adversarial loss. Currently, I have only released the demo for the center region inpainting only and will release the arbitrary region semantic inpainting models soon.

1. Install Torch:  http://torch.ch/docs/getting-started.html#_

2. Clone the repository
  ```Shell
  git clone https://github.com/pathak22/context-encoder.git
  ```
  
3. Demo
  ```Shell
  cd context-encoder
  bash ./models/scripts/download_inpaintCenter_models.sh
  # This will populate the `./models/` folder with trained models.

  net=models/inpaintCenter/paris_inpaintCenter.t7 name=paris_result imDir=images/paris overlapPred=4 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/imagenet_inpaintCenter.t7 name=imagenet_result imDir=images/imagenet overlapPred=0 manualSeed=222 batchSize=21 gpu=1 th demo.lua
  net=models/inpaintCenter/paris_inpaintCenter.t7 name=ucberkeley_result imDir=images/ucberkeley overlapPred=4 manualSeed=222 batchSize=4 gpu=1 th demo.lua
  # Note: If you are running on cpu, use gpu=0
  # Note: samples given in ./images/* are held-out images
  ```
  
Sample results on held-out images: 

![teaser](images/teaser.jpg "Sample inpainting results on held-out images")

### 2) Train Context Encoders

If you could successfully run the above demo, run following steps to train your own context encoder model for image inpainting.

0. [Optional] Install Display Package as follows. If you don't want to install it, then set `display=0` in `train.lua`.
  ```Shell
  luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
  cd ~
  th -ldisplay.start 8000
  # if working on server machine create tunnel: ssh -f -L 8000:localhost:8000 -N server_address.com
  # on client side, open in browser: http://localhost:8000/
  ```

1. Make the dataset folder.
  ```Shell
  mkdir -p /path_to_wherever_you_want/mydataset/train/images/
  # put all training images inside mydataset/train/images/
  mkdir -p /path_to_wherever_you_want/mydataset/val/images/
  # put all val images inside mydataset/val/images/
  cd context-encoder/
  ln -sf /path_to_wherever_you_want/mydataset dataset
  ```

2. Train the model
  ```Shell
  DATA_ROOT=dataset/train display_id=11 name=inpaintCenter overlapPred=4 wtl2=0.999 nBottleneck=4000 niter=500 loadSize=350 fineSize=128 gpu=1 th train.lua
  ```

3. Test the model
  ```Shell
  # you can either use demo.lua to display the result or use test.lua using following commands:
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_patch overlapPred=4 manualSeed=222 batchSize=30 loadSize=350 gpu=1 th test.lua
  DATA_ROOT=dataset/val net=checkpoints/inpaintCenter_500_net_G.t7 name=test_full overlapPred=4 manualSeed=222 batchSize=30 loadSize=129 gpu=1 th test.lua
  ```

### 3) Download Features Caffemodel

Features for context encoder trained with reconstruction loss.

- [Prototxt](http://www.cs.berkeley.edu/~pathak/context_encoder/resources/ce_features.prototxt)
- [Caffemodel](http://www.cs.berkeley.edu/~pathak/context_encoder/resources/ce_features.caffemodel)

### 4) TensorFlow Implementation

Checkout the cool TensorFlow implementation of our paper by Taeksoo [here](https://github.com/jazzsaxmafia/Inpainting).
