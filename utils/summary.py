import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import matplotlib
import random

class TensorboardSummary(object):
        
    def create_summary(self, directory,args):
        writer = SummaryWriter(log_dir=os.path.join(directory))
        self.args = args
        self.mean = np.array(args.normal_mean).reshape(1,1,3)
        self.std = np.array(args.normal_std).reshape(1,1,3)
        return writer
