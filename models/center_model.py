import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.voice2face_model import Euclidean_distance, cosine_distance
import math

class center(BaseModel):
    def __init__(self, args):
        super(center, self).__init__(args)
        output_channel = self.args.output_channel
        num_classes = self.args.num_classes

        self.ratio = -1

        self.loss_mean = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.loss_mean, 14.0)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.weight, 1)

        self.set_metrics()

    def set_metrics(self, metrics=cosine_distance):
        self.metrics = metrics

    def forward(self, loss, y):
        self.loss_mean[y] = self.loss_mean[y] * 0.9 + loss * 0.1

    def update(self):
        if self.ratio < 0:
            nn.init.constant_(self.weight, 0)
            add_ratio = 0.5
        else:
            add_ratio = 0.024
        if self.ratio >= 0.90:
            return
        n = int(add_ratio * self.args.num_classes)
        self.loss_mean[self.weight > 0] += 1e6
        top = self.loss_mean.argsort()[:n]
        self.loss_mean[self.weight > 0] -= 1e6
        self.weight *= 0.99
        self.weight[top] = 1
        self.ratio = float((self.weight > 0).float().sum() / self.weight.shape[0])

    def get_weight(self, y):
        weight = self.weight * len(self.weight) / self.weight.sum()
        w = torch.index_select(weight, 0, y)
        if w.sum() < 1e-3:
            return w + 1
        # w *= len(y) / w.sum()
        return w
