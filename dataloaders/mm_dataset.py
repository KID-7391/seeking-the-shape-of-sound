from .base_dataset import BaseDataset
from .base_dataset import pil_loader
import numpy as np
import io
import os
import os.path as osp
import torch
from utils.logger import logger
import importlib
import random
import cv2
import time
import csv
from collections import namedtuple
import lmdb
from tqdm import tqdm
import pickle


class MMDataset(BaseDataset):
    def __init__(self, args, split='train', sample_mode='list', **kwargs):
        super().__init__(args)
        self.root_dir_image = args.image_data_dir
        self.root_dir_audio = args.audio_data_dir
        self.split = split
        if split == 'train':
            args.train_list = os.path.join(args.list_dir, args.dataset_train+'.csv')
            self.transform = self.transform_train()
            _data_list = args.train_list
        elif split == 'val':
            args.val_list = os.path.join(args.list_dir, args.dataset_val+'.csv')
            self.transform = self.transform_validation()
            _data_list = args.val_list
        elif split == 'test':
            args.test_list = os.path.join(args.list_dir, args.dataset_test+'.csv')
            self.transform = self.transform_validation()
            _data_list = args.test_list
        else:
            raise ValueError

        self.modal = self.args.get('modal', 'both')

        self.rank = 0
        self.split = split

        def get_all_files(dir, ext):
            for e in ext:
                if dir.endswith(e):
                    return [dir]

            file_list = os.listdir(dir)
            ret = []
            for i in file_list:
                ret += get_all_files(osp.join(dir, i), ext)
            return ret

        self.metas = []
        self.id_mapping = dict()
        len_image = 0
        len_audio = 0

        with open(_data_list) as f:
            lines = csv.DictReader(f, delimiter=',')
            for line in lines:
                if line['Set'] == 'none':
                    continue
                face_id = line['VGGFace1_ID']
                audio_id = line['VoxCeleb1_ID']

                line['image_list'] = get_all_files(osp.join(self.root_dir_image, face_id), ['.jpg'])
                line['audio_list'] = get_all_files(osp.join(self.root_dir_audio, audio_id), ['.npy', '.wav'])

                if not face_id in self.id_mapping.keys():
                    self.id_mapping[face_id] = len(self.id_mapping)
                
                len_image += len(line['image_list'])
                len_audio += len(line['audio_list'])
                self.metas.append(line)

        if sample_mode == 'list' and (split == 'val' or split == 'test'):
            self.triplet_list = []
            with open(osp.join(self.args.list_dir, self.args.get('eval_triplet_%s'%split) + '.txt'), 'r') as f:
                triplet_list = f.read().splitlines()

            filename2index = dict()
            for i in range(len(self.metas)):
                for j in range(len(self.metas[i]['audio_list'])):
                    filename = self.metas[i]['audio_list'][j]
                    filename = filename.replace(self.root_dir_audio + '/', '')
                    filename2index[filename] = (i, j)
                for j in range(len(self.metas[i]['image_list'])):
                    filename = self.metas[i]['image_list'][j]
                    filename = filename.replace(self.root_dir_image + '/', '')
                    filename2index[filename] = (i, j)

            self.triplet_list = []
            for i, item in enumerate(triplet_list):
                self.triplet_list.append([filename2index[ii] for ii in item.split(' ')])

            self.num = len(self.triplet_list)
        else:
            self.num = len(self.metas)
            self.triplet_list = None

        if self.rank == 0:
            logger.info('%s set has %d images, %d audios, %d samples per epoch' % (self.split, len_image, len_audio, self.__len__()))

        self.initialized = False
 
    def __len__(self):
        return self.num

    def __str__(self):
        return self.args.data_dir + '  split=' + str(self.split)
    
    def load_image(self, metas, image_id):
        image_filename = metas['image_list'][image_id]
        img = pil_loader(filename=image_filename)
        return img

    def load_audio(self, metas, audio_id):
        audio_filename = metas['audio_list'][audio_id]
        aud = pil_loader(filename=audio_filename)
        return aud

    def load(self, metas, image_id=-1, audio_id=-1, with_filename=False):
        if image_id < 0:
            image_id = np.random.choice(range(len(metas['image_list'])))

        if audio_id < 0:
            audio_id = np.random.choice(range(len(metas['audio_list'])))

        ID = metas['VGGFace1_ID']
        sample = {'ID': self.id_mapping[ID]}
        if self.modal == 'image' or self.modal == 'both':
            sample['image'] = self.load_image(metas, image_id)
            if with_filename:
                sample['image_filename'] = metas['image_list'][image_id].replace(self.root_dir_image + '/', '')
                sample['gender'] = metas['Gender']
        if self.modal == 'audio' or self.modal == 'both':
            sample['audio'] = self.load_audio(metas, audio_id)
            if with_filename:
                sample['audio_filename'] = metas['audio_list'][audio_id].replace(self.root_dir_audio + '/', '')
                sample['gender'] = metas['Gender']

        return sample

    def getitem(self, idx):
        if self.triplet_list is not None:
            return self.getitem_triple(idx)
        else:
            sample = self.load(self.metas[idx])
        sample = self.transform(sample)
        return sample
    
    def getitem_triple(self, idx):
        triplet = self.triplet_list[idx]
        sample = self.load(self.metas[triplet[0][0]], audio_id=triplet[0][1])
        sample_p = self.load(self.metas[triplet[1][0]], image_id=triplet[1][1])
        sample_n = self.load(self.metas[triplet[2][0]], image_id=triplet[2][1])
        sample = self.transform(sample)
        sample_p = self.transform(sample_p)
        sample_n = self.transform(sample_n)
        return sample, sample_p, sample_n

    def getitem_all(self, idx):
        samples_audio = []
        samples_image = []
        for j in range(len(self.metas[idx]['image_list'])):
            sample = self.load(self.metas[idx], image_id=j, with_filename=True)
            del sample['audio']
            del sample['audio_filename']
            sample = self.transform(sample)
            samples_image += [sample]
                        
        for j in range(len(self.metas[idx]['audio_list'])):
            sample = self.load(self.metas[idx], audio_id=j, with_filename=True)
            del sample['image']
            del sample['image_filename']
            sample = self.transform(sample)
            samples_audio += [sample]

        return samples_image, samples_audio

    def __getitem__(self, idx):
        return self.getitem(idx)
