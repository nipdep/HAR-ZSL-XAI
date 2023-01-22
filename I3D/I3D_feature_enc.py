# Check Pytorch installation
import torch, torchvision
from mmaction.apis import init_recognizer
print(torch.__version__, torch.cuda.is_available())

import os
import os.path as osp
import re
import warnings
from operator import itemgetter

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_recognizer

# Check MMAction2 installation
import mmaction
print(mmaction.__version__)

# Check MMCV installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# Choose to use a config and initialize the recognizer
config = '../mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
# Initialize the recognizer
model = init_recognizer(config, checkpoint, device='cuda:0')