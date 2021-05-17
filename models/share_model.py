import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.voice2face_model import Euclidean_distance
from models.nn.module import MultiLayerPerceptron, LinearModule


class share(BaseModel):
    def __init__(self, args):
        super(share, self).__init__(args)
        num_channels = self.args.num_channels
        num_classes = self.args.num_classes
        act = self.args.act
        bn = self.args.bn

        if self.args.get('shared_fc', False):
            self.fc = LinearModule(
                128,
                output_channel,
                act='none',
                bn='none',
                equaliaeed=False
            )

        # self.model = nn.ModuleList()
        # for  i in range(len(num_channels) - 1):
        #     self.model.append(ResidualBlock(num_channels[i], num_channels[i+1], act, bn))
        
        # self.cls = nn.ModuleList([ArcMarginDistb(i, num_classes) for i in num_channels])

    def forward(self, x):
        outs = [x]
        for l in self.model:
            x = l(x)
            outs.append(x)

        return outs
    
    def get_logit(self, x, y, margin=None):
        assert len(x) == len(self.cls)
        # return self.cls[0](x, y, margin)
        logit = []
        for i in range(len(self.cls)):
            logit.append(self.cls[i](x[i], y, margin))
        return logit


# class ResidualBlock(nn.Module):
#     def __init__(self, nin, nout, act, bn, in_bn_relu=True):
#         super(ResidualBlock, self).__init__()
#         self.bn = nn.BatchNorm1d(nin)
#         self.act = nn.ReLU(inplace=True)
#         self.model = MultiLayerPerceptron(
#             nin,
#             nout,
#             nout,
#             3,
#             act=act,
#             bn=bn,
#             equaliaeed=False,
#             bn_first=True
#         )

#     def forward(self, x):
#         return self.model(self.act(self.bn(x)))


# class ArcMarginDistb(nn.Module):
#     def __init__(self, nin, nout):
#         super(ArcMarginDistb, self).__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(nout, nin))
#         nn.init.xavier_uniform_(self.weight)

#     def dist(self, x):
#         dist = ((self.weight.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(2)
#         return dist

#     def arc_margin(self, x, y, margin):
#         dist = self.dist(x) # N x M
#         one_hot = torch.zeros(dist.size(), device='cuda')
#         one_hot.scatter_(1, y.view(-1, 1).long(), 1)
#         if margin is None:
#             logit = (one_hot * (dist)) + ((1.0 - one_hot) * torch.clamp(dist, 0, 5))
#         else:
#             logit = (one_hot * (dist + margin.unsqueeze(1))) + ((1.0 - one_hot) * torch.clamp(dist, 0, 5))
#         logit = -logit
#         return logit
    
#     def forward(self, x, y, margin=None):
#         logit = self.arc_margin(x, y, margin)
#         return logit
