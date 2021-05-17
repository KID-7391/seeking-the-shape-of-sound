import os
import torch
from torch.utils.data import DataLoader
from models import generate_net
from dataloaders.mm_dataset import MMDataset
from utils.loss import MMLoss
from utils.metrics import Evaluator
import yaml
from easydict import EasyDict as edict
import copy
import sys
import numpy as np
from utils.utils import load_pretrained_model, load_checkpoint


def worker_init_fn_seed(worker_id):
    torch.initial_seed()
    np.random.seed()


class BaseContainer(object):
    def __init__(self):
        # init the args necessary
        fi = open(sys.argv[1],'r')
        args = yaml.load(fi, Loader=yaml.FullLoader)
        self.args = edict(args)
        self.Dataset_train = MMDataset
        self.Dataset_val = MMDataset
        self.args.training.cuda = not self.args.training.get('no_cuda',False)
        self.args.training.gpus = torch.cuda.device_count()

        torch.backends.cudnn.benchmark = True
        torch.manual_seed(1)
    
    def init_training_container(self):
        # Define dataset
        self.train_set = self.Dataset_train(self.args.dataset, split='train')    
        self.val_set = self.Dataset_val(self.args.dataset, split='val')
        self.test_set = self.Dataset_val(self.args.dataset, split='test')
    
        # Define network
        self.model = generate_net(self.args.models)
        self.model = self.model.cuda()

        start_it = 0
        stage = 0
        # # Resuming checkpoint
        if self.args.training.resume_train is not None and os.path.exists(self.args.training.resume_train):
            state_dict, optimizer, start_it, stage = load_checkpoint(checkpoint_path=self.args.training.resume_train)
            self.gen_optimizer(self.model.param_groups(), (stage + 1) // 2)

            if isinstance(self.args.training.batchsize, list):
                self.batchsize = self.args.training.batchsize[(stage + 1) // 2]
            else:
                self.batchsize = self.args.training.batchsize

            load_pretrained_model(self.model, state_dict)
            self.model = self.model.cuda()

            if not self.args.training.ft and optimizer is not None:
                for name in self.args.training.optimizer.keys():
                    if name in optimizer.keys():
                        self.optimizer[name].load_state_dict(optimizer[name])
            else:
                start_it = 0
        else:
            self.gen_optimizer(self.model.param_groups(), (stage + 1) // 2)
            self.start_epoch = 0

        self.start_it = start_it
        self.stage = stage

        # Define Criterion
        self.criterion = MMLoss(self.args.training)

    def init_evaluation_container(self):
        # Define network
        self.model = generate_net(self.args.models)

        # # Resuming checkpoint
        state_dict, _, _, _ = load_checkpoint(checkpoint_path=self.args.evaluation.resume_eval)
        load_pretrained_model(self.model, state_dict)
        self.model = self.model.cuda()

    def gen_optimizer(self, train_params, stage=0):
        args = self.args.training.optimizer
        self.optimizer = dict()
        for name in args.keys():
            item = args[name]
            params = []
            for i in item.train_params:
                params += train_params[i]
            if len(params) == 0:
                continue
            if item.optim_method == 'sgd':
                self.optimizer[name] = torch.optim.SGD(
                    params,
                    momentum=item.get('momentum', 0.0),
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                    nesterov=item.get('nesterov', False)
                )
            elif item.optim_method == 'adagrad':
                self.optimizer[name] = torch.optim.Adagrad(
                    params,
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                )
            elif item.optim_method == 'adam':
                self.optimizer[name] = torch.optim.Adam(
                    params,
                    lr=item.lr * item.get('lr_decay', 1) ** stage,
                    weight_decay=item.get('weight_decay', 0),
                    betas=item.get('betas', (0.9, 0.999))
                )
            else:
                raiseNotImplementedError(
                    "optimizer %s not implemented!"%item.optim_method)

    def reset_batchsize(self):
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batchsize,
            worker_init_fn=worker_init_fn_seed,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.training.get('num_workers', 4)
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=256, #self.batchsize,
            worker_init_fn=worker_init_fn_seed,
            num_workers=self.args.training.get('num_workers', 4)
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=256, #self.batchsize,
            worker_init_fn=worker_init_fn_seed,
            num_workers=self.args.training.get('num_workers', 4)
        )


    def training(self):
        pass

    def validation(self):
        pass
