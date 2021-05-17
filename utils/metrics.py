import numpy as np


class Evaluator(object):
    def __init__(self):
        self.status = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    def Accuracy(self):
        acc = float(self.status['TP'] + self.status['TN']) / \
            float(self.status['TP'] + self.status['TN'] + self.status['FP'] + self.status['FN'] + 1e-6)
        return acc

    def add_batch(self, pred, gt):
        pred = pred.detach()
        gt = gt.detach()
        self.status['TP'] += ((pred == 1) & (pred == gt)).sum()
        self.status['TN'] += ((pred == 0) & (pred == gt)).sum()
        self.status['FP'] += ((pred == 1) & (pred != gt)).sum()
        self.status['FN'] += ((pred == 0) & (pred != gt)).sum()

    def reset(self):
        self.status = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
