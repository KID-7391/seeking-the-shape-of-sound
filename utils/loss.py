import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import math
import time


def focal_loss(sample, alpha=1.0, beta=2.0):
    logit, target = sample['logit'], sample['target']
    pred = torch.softmax(logit, dim=1)

    one_hot = torch.zeros(logit.size()).to(logit.device)
    one_hot.scatter_(1, target.view(-1, 1).long(), 1)
    weight = alpha * (1 - (one_hot * pred).sum(-1))**beta
    loss = weight * F.cross_entropy(logit, target)
    return loss.mean()

def cross_entropy_loss(sample):
    logit, target = sample['logit'], sample['target']
    loss = F.cross_entropy(logit, target)
    if 'weight' in sample.keys() and sample['weight'] is not None:
        loss = loss * sample['weight']
    loss = loss.mean()
    return loss

def binary_cross_entropy_loss(sample):
    logit, target = sample['logit'], sample['target']
    loss = F.binary_cross_entropy(torch.sigmoid(logit), target.float())
    loss = loss.mean()
    return loss

# def triplet_loss(sample, margin=1.0, p=2):
#     emb_ref, emb_pos, emb_neg = sample['emb_ref'], sample['emb_pos'], sample['emb_neg']
#     loss = F.triplet_margin_loss(emb_ref, emb_pos, emb_neg, margin, p)
#     loss = loss.mean()
#     return loss

def mean_square_error_loss(sample):
    logit, target = sample['logit'], sample['target']
    if isinstance(logit, torch.Tensor):
        loss = (logit - target.float()) ** 2
    elif isinstance(logit, (list, tuple)):
        loss = 0
        for l,t in zip(logit, target):
            loss += ((l - t.float()) ** 2).mean()
    loss = loss.mean()
    return loss

def l1_loss(sample):
    logit, target = sample['logit'], sample['target']
    if isinstance(logit, torch.Tensor): 
        loss = torch.abs(logit - target).mean()
    else:
        assert False
    loss = loss.mean()
    return loss

def grad_penalty_loss(sample):
    logit, target = sample['logit'], sample['target']
    logit = logit.view(logit.shape[0], -1)
    if isinstance(logit, torch.Tensor):
        loss = ((logit.norm(2, dim=1) - target) ** 2).mean() / (target ** 2)
    else:
        assert False
    loss = loss.mean()
    return loss

def wgan_eps_loss(sample):
    iwass_epsilon = 0.001
    logit, target = sample['logit'], sample['target']
    if isinstance(logit, torch.Tensor): 
        loss = (logit * (-2.0 * target.float() + 1) + iwass_epsilon * logit ** 2).mean()
    else:
        assert False
    loss = loss.mean()
    return loss

def wgan_gen_loss(sample):
    (logit_real, logit_fake), D = sample['logit'], sample['D']
    d_real = D(logit_real)
    d_fake = D(logit_fake)
    loss = -torch.mean(d_fake) + torch.mean(d_real)
    loss = loss.mean()
    return loss

def wgan_dis_loss(sample):
    lambda_drift = sample.get('lambda_drift', 0.001)
    lambda_gp = sample.get('lambda_gp', 0)

    (logit_real, logit_fake), D = sample['logit'], sample['D']
    d_real = D(logit_real)
    d_fake = D(logit_fake)
    loss = torch.mean(d_fake) - torch.mean(d_real) + lambda_drift * torch.mean(d_real ** 2)

    def __gradient_penalty(real, fake, D):
        N = real.shape[0]
        epsilon = torch.rand((N, 1)).to(real.device)

        merged = epsilon * real + (1.0 - epsilon) * fake
        merged.requires_grad_(True)

        out = D(merged)
        out = out.view(-1)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=merged,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad = grad.view(grad.shape[0], -1)
        return ((grad.norm(p=2, dim=1) * (merged.shape[-1] / 4) - 1) ** 2).mean()

    # gp = __gradient_penalty(logit_real, logit_fake, D)
    # loss += lambda_gp * gp
    loss = loss.mean()
    return loss

def triplet_loss(sample):
    weight = sample['weight']
    loss = weight[0] * sample['pos'] + \
           weight[1] * torch.max(torch.tensor([0.0]).to(sample['neg'].device), sample['margin'] - sample['neg'])
    return loss.mean()

def realnvp_loss(sample):
    prior_log_like = sample['prior_ll']
    # if sample['cnt'] % 100 == 0:
    #     print(prior_log_like.mean().item(), sample['log_det_jacob'].mean().item())
    ll = prior_log_like + sample['log_det_jacob']
    nll = -ll
    return nll.mean()

def one_loss(sample):
    loss = sample['loss']
    if 'weight' in sample.keys() and sample['weight'] is not None:
        loss = loss * sample['weight']
    return loss.mean()

class MMLoss(object):
    def __init__(self, args):
        self.args = args
        self.is_iou = False
        self.cnt = 0

    def __call__(self, samples, type):
        loss_dict = {
            'focal_loss': focal_loss,
            'cross_entropy_loss': cross_entropy_loss,
            'binary_cross_entropy_loss': binary_cross_entropy_loss,
            'triplet_loss': triplet_loss,
            'mean_square_error_loss': mean_square_error_loss,
            'grad_penalty_loss': grad_penalty_loss,
            'wgan_gen_loss': wgan_gen_loss,
            'wgan_dis_loss': wgan_dis_loss,
            'realnvp_loss': realnvp_loss,
            'one_loss': one_loss
        }

        loss_weight = self.args.optimizer[type].loss_weight
        losses = {'loss': 0.0}
        cnt = 0
        for sample in samples:
            sample['cnt'] = self.cnt
            self.cnt += 1
            if 'loss_type' not in sample.keys() or sample['loss_type'] is None:
                continue
            loss = loss_dict[sample['loss_type']](sample)
            losses[sample['loss_name']] = loss.item()
            losses['loss'] += loss * loss_weight[cnt]
            cnt += 1

        return losses
