# -*- coding: utf-8 -*-
"""Toward DNN Deployment Cost Measurement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d9P5ovI2hVx7iR7MnjgjuxplLeVZgHRf

<div align="center">
<h1>Toward DNN Deployment Cost Measurements</h1>
Lanxiang Hu
</div>

---

# Reference
[1] Ma, Ningning, et al. **Shufflenet v2: Practical guidelines for efficient cnn architecture design**. Proceedings of the European conference on computer vision (ECCV). 2018. [[paper]](https://arxiv.org/abs/1807.11164v1).

[2] **THOP: PyTorch-OpCounter**. [[code]](https://github.com/Lyken17/pytorch-OpCounter).

[3] **Flops counter for convolutional networks in pytorch framework**. [[code]](https://github.com/sovrasov/flops-counter.pytorch).

[4] Chang, Jiho, et al. **Reducing MAC operation in convolutional neural network with sign prediction.** 2018 International Conference on Information and Communication Technology Convergence (ICTC). IEEE, 2018. [[paper]](https://junheecho.com/assets/papers/ictc18.pdf).

[5] Model optimization: **model FLOPs**. [[slides]](https://indico.cern.ch/event/917049/contributions/3856417/attachments/2034165/3405345/Quantized_CNN_LLP.pdf).

[6] Wang, Xin, et al. **Skipnet: Learning dynamic routing in convolutional networks.** Proceedings of the European Conference on Computer Vision (ECCV). 2018. [[paper]](https://arxiv.org/abs/1711.09485)[[code]](https://github.com/ucbdrive/skipnet).

[7] ICLR‘20 Once-for-All tutorial: **Train One Network and Specialize it for Efficient Deployment**. [[paper]](https://arxiv.org/pdf/1908.09791.pdf), [[code]](https://github.com/mit-han-lab/once-for-all/tree/master/tutorial), [[talk]](https://youtu.be/a_OeT8MXzWI).

# Load Libraries
"""

from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from torchsummary import summary

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

"""
Build Evaluator Class
"""

class CostEvaluator:
  def __init__(self, **kwargs):
    # hyper-parameters
    self.turn_on_log = kwargs.get("turn_on_log", 0)

  def memory_size_evaluator(self, nn_model):
    param_size = 0
    for param in nn_model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in nn_model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(total_size_mb))
    return total_size_mb

  def flops_evaluator(self, nn_model, in_channels, input_h, input_w):
    # summing all layers together
    total_flops = 0
    h_prev = 0
    w_prev = 0
    h = input_h
    w = input_w
    c_prev = in_channels

    counter = 0
    # unwrap nn model
    unwrapped_model = [module for module in nn_model.modules() if not isinstance(module, torch.nn.modules.container.Sequential) and not isinstance(module, torchvision.models.resnet.ResNet)]

    for module in unwrapped_model:
      if (self.turn_on_log):
        print('processing layer {}'.format(counter))
        print('layer type: {}'.format(type(module)))
      # there is a possible change in output size if the module is a convolutional layer or maxpool layer
      if (type(module) == torch.nn.modules.conv.Conv2d or type(module) == torch.nn.modules.pooling.MaxPool2d):
        if (type(module.kernel_size) == int):
          # handle int cases
          if (module.kernel_size > 1 and module.stride > 1):
            ratio = module.stride
            h_prev = h
            w_prev = w
            h = h / ratio
            w = w / ratio
            if (self.turn_on_log): print('output dimensions shrinking')
        else:
          # handle tuple cases
          if ((module.kernel_size[0] > 1 or module.kernel_size[1] > 1) and (module.stride[0] > 1 or module.stride[1] > 1)):
            ratio = module.stride[0]
            h_prev = h
            w_prev = w
            h = h / ratio
            w = w / ratio
            if (self.turn_on_log): print('output dimensions shrinking')
      elif (type(module) == torch.nn.modules.pooling.AdaptiveAvgPool2d):
        h_prev = h
        w_prev = w
        h = 1
        w = 1

      if (type(module) == torch.nn.Conv2d):
        # convolutional layer.
        if (self.turn_on_log): print('calculating FLOPs for Conv2d...')
        c_prev = module.out_channels
        total_flops += ((module.kernel_size[0] * module.kernel_size[1]) * module.in_channels  + 1) * (h * w) * module.out_channels
      elif (type(module) == torch.nn.MaxPool2d):
        # handle else case with maxpool
        if (self.turn_on_log): print('calculating FLOPs for MaxPool2d...')
        # number of filters
        n_1 = h_prev / module.stride
        n_2 = w_prev / module.stride
        n_tot = n_1 * n_2
        # note that number of channels should be held unchanged
        total_flops += (module.kernel_size * module.kernel_size + 1) * (h * w) * c_prev * n_tot
      elif (type(module) == torch.nn.ReLU):
        # handle else case with ReLU 
        # Assuming number of flops equal to length of input vector, ReLU takes 1 comparison and 1 multiplication
        if (self.turn_on_log): print('calculating FLOPs for ReLU...')
        total_flops += 2 * (h * w) * c_prev
      elif (type(module) == torch.nn.BatchNorm2d):
        # handle else case with BatchNorm
        if (self.turn_on_log): print('calculating FLOPs for BatchNorm2d...')
        mean_ops = (h * w) * module.num_features + 1
        std_ops = 2 * ((h * w) * module.num_features + 1)
        normalization_ops = 2 * (h * w) * module.num_features
        scale_and_shift_ops = 2 * (h * w) * module.num_features
        total_flops += mean_ops + std_ops + normalization_ops + scale_and_shift_ops

      counter += 1
      if (self.turn_on_log): print('------------------')

    total_flops_G = total_flops / 10**9
    print('total FLOPs: {:.3f} = {}G'.format(total_flops, total_flops_G))
    return total_flops


  def mac_evaluator(self, nn_model, input_h, input_w):
    # summing all layers together
    total_mac = 0
    h = input_h
    w = input_w

    # unwrap the nn model
    # the complete ResNet object is specifically excluded
    unwrapped_model = [module for module in nn_model.modules() if not isinstance(module, torch.nn.modules.container.Sequential) and not isinstance(module, torchvision.models.resnet.ResNet)]

    counter = 0
    for module in unwrapped_model:
      if (self.turn_on_log): print('processing layer {}'.format(counter))
      if (self.turn_on_log): print('layer type: {}'.format(type(module)))
      if (type(module) == torch.nn.modules.conv.Conv2d or type(module) == torch.nn.modules.pooling.MaxPool2d):
        if (type(module.kernel_size) == int):
          # handle int case
          if (module.kernel_size > 1 and module.stride > 1):
            ratio = module.stride
            h = h / ratio
            w = w / ratio
            if (self.turn_on_log): print('output dimensions shrinking')
        else:
          # handle tuple case
          if ((module.kernel_size[0] > 1 or module.kernel_size[1] > 1) and (module.stride[0] > 1 or module.stride[1] > 1)):
            ratio = module.stride[0]
            h = h / ratio
            w = w / ratio
            if (self.turn_on_log): print('output dimensions shrinking')
      elif (type(module) == torch.nn.modules.pooling.AdaptiveAvgPool2d):
        h_prev = h
        w_prev = w
        h = 1
        w = 1

      if (type(module) == torch.nn.Conv2d):
        # MAC operations are dominated by computations carried out in convolutional layer
        total_mac += (module.in_channels * module.out_channels) * (h * w) * (module.kernel_size[0] * module.kernel_size[1])
      counter += 1
      if (self.turn_on_log): print('------------------')
    total_mac_G = total_mac / 10**9
    print('total MACCs: {:.3f} = {}G'.format(total_mac, total_mac_G))
    return total_mac
  
  def inference_time_evaluator(self, device_flops, device_name, nn_model, in_channels, input_h, input_w):
    print('starting inference time analysis...')
    total_flop = self.flops_evaluator(nn_model, in_channels, input_h, input_w)
    inf_time = total_flop / device_flops
    print('total inference time: {:.3f}s with {}'.format(inf_time, device_name))
    return total_flop / device_flops

  def power_evaluator(self, device_power_spec, device_flops, nn_model, in_channels, input_h, input_w):
    print('starting inference power analysis...')
    total_flop = self.flops_evaluator(nn_model, in_channels, input_h, input_w)
    inf_time = total_flop / device_flops
    power = inf_time * device_power_spec
    print('total inference power consumption: {:.3f}s'.format(power))
    return power