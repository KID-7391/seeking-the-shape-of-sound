import os
import importlib
import torch
from torch import nn
from .base_model import BaseModel


def generate_net(args):
    if hasattr(args, 'model_warpper'):
        model_name = args.model_warpper
        from .voice2face_model import Voice2Face
        all_model = [Voice2Face]
    else:
        model_name = args.model
        from .center_model import center
        from .backbones import SEResNet50IR, ThinResNet34

        all_model = [center, SEResNet50IR, ThinResNet34]

    model = None
    for m in all_model:
        if m.__name__ == model_name:
            model = m
            break

    if model is None:
        raise NotImplementedError("there has no %s" % (model_name))

    return model(args)
