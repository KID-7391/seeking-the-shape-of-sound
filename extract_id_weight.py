import sys
import os
import os.path as osp
import torch
import yaml
from easydict import EasyDict as edict

def extract_id_weight(src, dst):
    weight = torch.load(src)['state_dict']['center.weight']
    torch.save({'center.weight': weight}, dst)

def main():
    fi = open(sys.argv[1],'r')
    args = yaml.load(fi, Loader=yaml.FullLoader)
    args = edict(args)
    # args = args.training
    path = osp.join(
        args.training.save_dir,
        args.dataset.dataset_train,
        args.models.model_warpper,
        args.training.experiment_id
    )
    extract_id_weight(osp.join(path, 'ckp_002400.pth.tar'), osp.join(path, 'id_weight.pth'))

if __name__ == "__main__":
    main()
