import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
from models.nn.module import NormalizeConv, GDropLayer, MultiLayerPerceptron
from models.gen_network import generate_net
import copy


def Euclidean_distance(x, y):
    return ((x - y)**2).sum(-1)

def cosine_distance(x, y):
    return (F.normalize(x) * F.normalize(y)).sum(-1)

class Voice2Face(BaseModel):
    """
    An implementation of "Seeking the Shape of Sound" (https://arxiv.org/abs/2103.07293).

    Args:
    args (easydict.EasyDict): The dict with model options including:
      - args.conponents.F (easydict.EasyDict): options for the face feature extractor (see backbones.py)
      - args.conponents.V (easydict.EasyDict): options for the voice feature extractor (see backbones.py)
      - args.conponents.center (easydict.EasyDict): options for class weights (see center_model.py)

    """
    def __init__(self, args):
        super(Voice2Face, self).__init__(args)

        if 'V' in self.args.conponents.keys():
            self.V = generate_net(self.args.conponents.V)

        if 'F' in self.args.conponents.keys():
            self.F = generate_net(self.args.conponents.F)

        if 'center' in self.args.conponents.keys():
            self.center = generate_net(self.args.conponents.center)

        for c in self.args.conponents.keys():
            model = getattr(self, c)
            if model.args.get('freeze', False):
                for p in model.parameters():
                    p.requires_grad = False

    def update_hard(self):
        """Update class weights."""
        if self.args.weight:
            self.center.update()

    def forward_match(self, sample, is_triple=False, V2F=True):
        """ Forward 
            
        """
        if is_triple:
            sample, sample_p, sample_n = sample
            if V2F:
                x = self.V(sample['audio'])
                x_p = self.F(self.normalize(sample_p['image']))
                x_n = self.F(self.normalize(sample_n['image']))
            else:
                x = self.F(self.normalize(sample['image']))
                x_p = self.V(sample_p['audio'])
                x_n = self.V(sample_n['audio'])    
            dist_pos = cosine_distance(x, x_p)
            dist_neg = cosine_distance(x, x_n)

            pred = dist_pos > dist_neg
            out = {
                'dist_pos': dist_pos,
                'dist_neg': dist_neg,
                'pred': pred.long(),
                'target': torch.ones_like(pred).long().to(pred.device)
            }
            return [out]

        x = self.normalize(sample['image'])
        v = sample['audio']
        rd = np.random.choice(range(int(2.5*16000), int(5.0*16000)))
        rd_start = np.random.choice(range(0, v.shape[1] - rd))
        v = v[:,rd_start:rd + rd_start]
        y = sample['ID']

        z_x = self.F(x)
        z_v = self.V(v)

        logit_x = self.F.cls(z_x, y)
        logit_v = self.F.cls(z_v, y)

        logit_cross = 0.5 * self.F.cls.cross_logit(z_x, z_v) + 0.5 * self.F.cls.cross_logit(z_v, z_x)
        loss_weight = F.cross_entropy(logit_x, y, reduction='none').detach() + F.cross_entropy(logit_v, y, reduction='none').detach()
        if self.args.weight:
            self.center(loss_weight, y)
            w = self.center.get_weight(y)
        else:
            w = None

        out_x = {
            'logit': logit_x,
            'target': y,
            'weight': w,
            'loss_type': 'cross_entropy_loss',
            'loss_name': 'face_ce_loss'
        }

        out_v = {
            'logit': logit_v,
            'target': y,
            'weight': w,
            'loss_type': 'cross_entropy_loss',
            'loss_name': 'voice_ce_loss'
        }
        outputs = [out_x, out_v]

        if self.args.with_cross:
            out_cross = {
                'loss': logit_cross,
                'weight': w,
                'loss_type': 'one_loss',
                'loss_name': 'crosssp_loss'
            }
            outputs.append(out_cross)

        return outputs

    def forward(self, samples, type, **kwargs):
        return getattr(self, 'forward_' + type)(samples, **kwargs)

    def param_groups(self):
        param = dict()
        for c in self.args.conponents.keys():
            param[c] = getattr(self, c).param_groups()
        return param
