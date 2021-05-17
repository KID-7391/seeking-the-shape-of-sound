import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_normal_, calculate_gain, xavier_normal_
from collections import OrderedDict
from abc import ABC, abstractmethod
from .nn.module import NormalizeConv, ConvModule, NoneLayer


class BaseModel(nn.Module,ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_channel = 3
        if args.get('normal_layer', False):
            self.normalize = NormalizeConv(3,normal_mean=args.normal_mean,normal_std=args.normal_std)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def freezeBN(self):
        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1 and not m.weight.requires_grad:
                m.eval()

    def param_groups(self, lr=None):
        params = list(filter(lambda x:x.requires_grad, self.parameters()))
        if len(params):
            if lr is not None:
                return [{'params': params, 'lr': lr}]
            else:
                return [{'params': params}]
        else:
            return []


class BasePGModel(BaseModel):
    def __init__(self, args):
        super(BasePGModel, self).__init__(args)
        self.flag_tanh = args.get('flag_tanh', False)
        if args.get('flag_gdrop', False):
            self.gdrop_param = {
                'mode': 'prop',
                'strength': 0.0,
                'axes': [0, 1],
                'normalize': False
            }
        else:
            self.gdrop_param = None

        self.model = None

        self.act = args.act
        self.bn = args.bn

        self.n_img = args.n_img
        self.n_latent = args.n_latent
        self.channels = args.channels
        self.input_channel = args.input_channel
        self.cur_index = 0

    def init_net(self):
        self.alpha = 1.0
        self.model = nn.ModuleList()
        self.model.append(self.first_block())
        self.trans_rgb = nn.ModuleList()

    def first_block(self):
        pass

    def grow_net(self, start_it, end_it):
        self.start_it = torch.nn.Parameter(torch.tensor([start_it]).float(), requires_grad=False)
        self.end_it = torch.nn.Parameter(torch.tensor([end_it]).float(), requires_grad=False)
        self.cur_index += 1

    def update_alpha(self, x, it=-1):
        if it < 0:
            self.alpha = torch.tensor([1.0]).cuda()
        else:
            if hasattr(self, 'start_it'):
                self.alpha = (it - self.start_it) / (self.end_it - self.start_it)
                self.alpha = self.alpha.to(x.device)
                self.alpha = torch.clamp(self.alpha, 0.0, 1.0)
