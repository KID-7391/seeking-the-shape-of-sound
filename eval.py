import sys
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from sklearn import metrics

from base_container import BaseContainer


class NetworkTester(BaseContainer):
    def __init__(self):
        super().__init__()
        self.init_evaluation_container()

        self.loaders = []
        self.v2f = []

        self.test_set = self.Dataset_val(self.args.evaluation.dataset, split='test', sample_mode='id')
        self.num_id = len(self.test_set)
    
    def gen_feat(self, save_path):
        print('Generating features...')
        self.model.eval()
        name2feat = dict()
        name2id = dict()
        name2gender = dict()
        for i in tqdm(range(self.num_id)):
            sample_img, sample_aud = self.test_set.getitem_all(i)
            imgs, img_filenames, img_gender = array2tensor_img(sample_img)
            auds, aud_filenames, aud_gender = array2tensor_aud(sample_aud)
            with torch.no_grad():
                imgs = self.model.normalize(imgs.cuda())
                auds = auds.cuda()
                img_feats = self.model.F(imgs)
                aud_feats = self.model.V(auds)
                if 'wang' in save_path:
                    img_feats = self.model.F.shared_fc(img_feats)
                    aud_feats = self.model.F.shared_fc(aud_feats)
            for j in range(len(img_feats)):
                name2feat[img_filenames[j]] = img_feats[j]
                name2id[img_filenames[j]] = i
                name2gender[img_filenames[j]] = img_gender[j]

            for j in range(len(aud_feats)):
                name2feat[aud_filenames[j]] = aud_feats[j]
                name2id[aud_filenames[j]] = i
                name2gender[aud_filenames[j]] = aud_gender[j]

        save = {
            'name2feat': name2feat,
            'name2id': name2id,
            'name2gender': name2gender
        }
        torch.save(save, save_path)
        print('Done.')

    def load_feat(self):
        save_path = self.args.evaluation.save_feat
        # if not osp.exists(save_path):
        self.gen_feat(save_path)
        res = torch.load(save_path)
        self.name2feat = res['name2feat']
        self.name2id = res['name2id']
        self.name2gender = res['name2gender']

    def eval(self):
        args = self.args.evaluation.dataset
        self.load_feat()
        if not isinstance(args.eval_triplet_test_all, list):
            with open(osp.join(args.list_dir, args.eval_triplet_test_all + '.txt'), 'r') as f:
                args.eval_triplet_test_all = f.read().splitlines()

        all_res = []
        for l in args.eval_triplet_test_all:
            with open(osp.join(args.list_dir, l + '.txt'), 'r') as f:
                eval_list = f.read().splitlines()
            eval_list = [i.split(' ') for i in eval_list]
            if l.startswith('match'):
                res = self.eval_match(eval_list)
            elif l.startswith('verify'):
                res = self.eval_verify(eval_list)
            elif l.startswith('reterival'):
                res = self.eval_retrival(eval_list)
            print('%s: %.6f'%(l, res))
            all_res.append(res)

        with open(self.args.evaluation.save_result, 'w') as f:
            for i in range(len(all_res)):
                f.write(args.eval_triplet_test_all[i] + ' ' + str(all_res[i]) + '\n')

    def eval_match(self, eval_list):
        N = len(eval_list[0]) - 1
        cnt_pos = 0
        cnt_neg = 0
        for i in tqdm(range(len(eval_list))):
            cand = []
            for j in range(N):
                cand.append(self.name2feat[eval_list[i][j+1]])
            cand = torch.stack(cand)
            x = self.name2feat[eval_list[i][0]]
            score = cosine_distance(x.unsqueeze(0), cand)
            ID = self.name2id[eval_list[i][0]]

            if score.argmax() == 0:
                cnt_pos += 1
            else:
                cnt_neg += 1
        return cnt_pos / (cnt_pos + cnt_neg)

    def eval_verify(self, eval_list):
        pred = []
        target = []
        for i in tqdm(eval_list):
            x = self.name2feat[i[0]]
            cand = self.name2feat[i[1]]
            score = cosine_distance(x, cand)
            pred.append(float(score))
            target.append(int(self.name2id[i[0]] == self.name2id[i[1]]))
        fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
        return metrics.auc(fpr, tpr)
    
    def eval_retrival(self, eval_list):
        assert len(eval_list) == 2
        mAP = []
        x = torch.stack([self.name2feat[i] for i in eval_list[0]], 0)
        cand = torch.stack([self.name2feat[i] for i in eval_list[1]], 0)
        x_id = torch.tensor([self.name2id[i] for i in eval_list[0]])
        cand_id = torch.tensor([self.name2id[i] for i in eval_list[1]])
        for i in tqdm(range(len(x))):
            score = cosine_distance(x[i].unsqueeze(0), cand)
            r = torch.argsort(-score)
            ID = cand_id[r]
            ap = compute_ap((ID == x_id[i]))
            mAP.append(ap)
        return np.array(mAP).mean()

def array2tensor_img(sample):
    imgs = []
    filenames = []
    gender = []
    for i in range(len(sample)):
        imgs.append(sample[i]['image'])
        filenames.append(sample[i]['image_filename'])
        gender.append(sample[i]['gender'])

    imgs = torch.stack(imgs, 0)
    return imgs, filenames, gender

def array2tensor_aud(sample):
    auds = []
    filenames = []
    gender = []
    for i in range(len(sample)):
        auds.append(sample[i]['audio'])
        filenames.append(sample[i]['audio_filename'])
        gender.append(sample[i]['gender'])

    auds = torch.stack(auds, 0)
    return auds, filenames, gender

def Euclidean_distance(x, y):
    return ((x - y)**2).sum(-1)

def cosine_distance(x, y):
    return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)

def compute_ap(label):
    old_recall = 0.0
    ap = 0.0
    intersect_size = 0.0
    nz = label.nonzero().squeeze(1)
    n_true = len(nz)
    for i in nz:
        i = int(i)
        intersect_size += 1
        recall = intersect_size / n_true
        precision = intersect_size / (i+1)
        ap += (recall - old_recall) * precision
        old_recall = recall
    return ap

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
    tester = NetworkTester()
    tester.eval()

if __name__ == "__main__":
    main()
