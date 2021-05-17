import os
import numpy as np
import torch
import torch.nn.functional as F
import time

from base_container import BaseContainer
from utils.summary import TensorboardSummary
from utils.logger import logger, set_logger_path
from utils.utils import Saver
from utils.metrics import Evaluator


class Trainer(BaseContainer):
    def __init__(self):
        super().__init__()
        now_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        logger_path = os.path.join(
            self.args.training.save_dir,
            self.args.dataset.dataset_train,
            self.args.models.model_warpper,
            self.args.training.experiment_id,
            '%s.log' % now_time
        )
        set_logger_path(logger_path)
        logger.info(self.args)

        # Define Saver
        self.saver = Saver(self.args)

        # Define Tensorboard Summary
        self.summary = TensorboardSummary()
        self.writer = self.summary.create_summary(self.saver.experiment_dir, self.args.models)


        self.init_training_container()
        self.batchsize = self.args.training.batchsize
        self.reset_batchsize()
        self.evaluator = Evaluator()
        self.best = 0.0

        # show parameters to be trained
        logger.debug('\nTraining params:')
        for p in self.model.named_parameters():
            if p[1].requires_grad:
                logger.debug(p[0])
        logger.debug('\n')

        # Clear start epoch if fine-tuning
        logger.info('Starting iteration: %d' % self.start_it)
        logger.info('Total iterationes: %d' % self.args.training.max_iter)

    # main function for training
    def training(self):
        self.model.train()

        num_img_tr = len(self.train_loader)
        logger.info('\nTraining')

        max_iter = self.args.training.max_iter
        it = self.start_it

        # support multiple optimizers, but only one 
        # optimizer is used here, i.e., names = ['match']
        names = self.args.training.optimizer.keys()

        while it < max_iter:
            for samples in self.train_loader:
                samples = to_cuda(samples)

                # validation
                val_iter = self.args.training.get('val_iter', -1)
                if val_iter > 0 and it % val_iter == 0 and it >= self.args.training.get('start_eval_it', 15000):
                    self.validation(it, 'val')
                    self.model.train()

                if it % 100 == 0:
                    logger.info('\n===> Iteration  %d/%d' % (it, max_iter))
    
                # update class weights
                if it >= 500 and self.args.training.get('weight_update_iter', -1) > 0 and it % self.args.training.get('weight_update_iter', -1) == 0:
                    self.model.update_hard()
                    logger.info('\nUpdate hard ID: %.3f'%self.model.center.ratio)
                    self.writer.add_scalar('train/data_ratio', self.model.center.ratio, it)

                for name in names:
                    losses = dict()

                    self.optimizer[name].zero_grad()
                    outputs = self.model(samples, type=name)
                    losses = self.criterion(outputs, name)
                    loss = losses['loss']
                    loss.backward()
                    self.optimizer[name].step()

                    losses.update(losses)

                    # log training loss
                    if it % 100 == 0:
                        loss_log_str = '=>%s   loss: %.4f'%(name, loss.item())
                        for loss_name in losses.keys():
                            if loss_name != 'loss':
                                loss_log_str += '    %s: %.4f'%(loss_name, losses[loss_name])
                                self.writer.add_scalar('train/%s_iter'%loss_name, losses[loss_name], it)
                        logger.info(loss_log_str)
                        self.writer.add_scalar('train/total_loss_iter_%s'%name, loss.item(), it)

                    # adjust learning rate
                    lr_decay_iter = self.args.training.optimizer[name].get('lr_decay_iter', None)
                    if lr_decay_iter is not None:
                        for i in range(len(lr_decay_iter)):
                            if it == lr_decay_iter[i]:
                                lr = self.args.training.optimizer[name].lr * (self.args.training.optimizer[name].lr_decay ** (i+1))
                                logger.info('\nReduce lr to %.6f\n'%(lr))
                                for param_group in self.optimizer[name].param_groups:
                                    param_group["lr"] = lr 
                                break

                it += 1

                # save model and optimizer
                if it % self.args.training.save_iter == 0 or it == max_iter or it == 1:
                    logger.info('\nSaving checkpoint ......')
                    optimizer_to_save = dict()
                    for i in self.optimizer.keys():
                        optimizer_to_save[i] = self.optimizer[i].state_dict()
                    self.saver.save_checkpoint({
                        'start_it': it,
                        'stage': self.stage,
                        'state_dict': self.model.state_dict(),
                        'optimizer': optimizer_to_save,
                    }, filename='ckp_%06d.pth.tar'%it)
                    logger.info('Done.')

    # main function for validation
    def validation(self, it, split):
        logger.info('\nEvaluating %s...'%split)
        self.evaluator.reset()
        self.model.eval()

        data_loader = self.val_loader if split == 'val' else self.test_loader
        num_img_tr = len(data_loader)
        dist_pos = []
        dist_neg = []
        total_loss = []
        name = list(self.args.training.optimizer.keys())[0]
        for i, samples in enumerate(data_loader):
            samples = to_cuda(samples)

            with torch.no_grad():
                outputs = self.model(samples, type=name, is_triple=True)
                dist_pos.append(outputs[-1]['dist_pos'].mean().item())
                dist_neg.append(outputs[-1]['dist_neg'].mean().item())

            self.evaluator.add_batch(outputs[-1]['pred'], outputs[0]['target'])

        self.writer.add_scalar('%s/dist_pos'%split, np.array(dist_pos).mean(), it)
        self.writer.add_scalar('%s/dist_neg'%split, np.array(dist_neg).mean(), it)

        acc = self.evaluator.Accuracy()
        self.writer.add_scalar('%s/acc'%split, acc, it)
        if split == 'val':
            logger.info('=====>[Iteration: %d    %s/acc=%.4f    previous best=%.4f'%(it, split, acc, self.best))
        else:
            logger.info('=====>[Iteration: %d    %s/acc=%.4f'%(it, split, acc))

        # if split == 'val':
        #     self.validation(it, 'test')

        if split == 'val' and acc > self.best:
            self.best = acc
            logger.info('\nSaving checkpoint ......')
            optimizer_to_save = dict()
            for i in self.optimizer.keys():
                optimizer_to_save[i] = self.optimizer[i].state_dict()
            self.saver.save_checkpoint({
                'start_it': it,
                'stage': self.stage,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer_to_save,
            }, filename='best.pth.tar')

def to_cuda(sample):
    if isinstance(sample, list):
        return [to_cuda(i) for i in sample]
    elif isinstance(sample, dict):
        for key in sample.keys():
            sample[key] = to_cuda(sample[key])
        return sample
    elif isinstance(sample, torch.Tensor):
        return sample.cuda()
    else:
        return sample

def main():
    trainer = Trainer()
    trainer.training()
    trainer.writer.close()

if __name__ == "__main__":
    main()
