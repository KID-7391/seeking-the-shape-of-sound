import torch
import random
import numpy as np
import cv2
import math


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = list(sample.keys())
        for key in key_list:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            if 'image' == key:
                img = sample[key]
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=0).copy()
                else:
                    img = img.transpose((2, 0, 1)).copy()
                sample[key] = torch.from_numpy(img).float()
            elif 'audio' == key:
                aud = sample[key]
                sample[key] = torch.from_numpy(aud).float()
            elif 'label' in key:
                mask = sample[key]
                mask = np.expand_dims(mask,axis=0).copy()
                sample[key] = torch.from_numpy(mask).long()
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            key_list = sample.keys()
            for key in key_list:
                if 'image' not in key:
                    continue
                image = sample[key]
                image_flip = np.flip(image, axis=1)
                sample[key] = image_flip
        return sample

class RandomRotate(object):
    """Randomly rotate image"""
    def __init__(self, angle_r,image_value=127,is_continuous=True):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
        self.IMAGE_VALUE = image_value

    def __call__(self, sample):
        if np.random.rand() < 0.5:
            return sample
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        PI = 3.141592653
        Hangle = rand_angle*PI / 180
        Hcos = math.cos(Hangle)
        Hsin = math.sin(Hangle)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            image = sample[key]
            imgsize = image.shape
            srcWidth = imgsize[1]
            srcHeight = imgsize[0]
            x = [0,0,0,0]
            y = [0,0,0,0]
            x1 = [0,0,0,0]
            y1 = [0,0,0,0]
            x[0] = -(srcWidth - 1) / 2
            x[1] = -x[0]
            x[2] = -x[0]
            x[3] = x[0]
            y[0] = -(srcHeight - 1) / 2
            y[1] = y[0]
            y[2] = -y[0]
            y[3] = -y[0]
            for i in range(4):
                x1[i] = int(x[i] * Hcos + y[i] * Hsin + 0.5)
                y1[i] = int(-x[i] * Hsin + y[i] * Hcos + 0.5)
            if (abs(y1[2] - y1[0]) > abs(y1[3] - y1[1])):
                Height = abs(y1[2] - y1[0])
                Width = abs(x1[3] - x1[1])
            else:
                Height = abs(y1[3] - y1[1])
                Width = abs(x1[2] - x1[0])
            row, col = image.shape[:2]
            m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
            new_image = cv2.warpAffine(image, m, (Width,Height), flags=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation, 
                borderValue=self.IMAGE_VALUE if 'image' in key else self.MASK_VALUE)
            sample[key] = new_image
        return sample

class Resize(object):
    def __init__(self,output_size,is_continuous=False,label_size=None):
        assert isinstance(output_size, (tuple,list))
        if len(output_size) == 1:
            self.output_size = (output_size[0],output_size[0])
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
    
    def __call__(self,sample):
        if not 'image' in sample.keys():
            return sample
        img = sample['image']
        image_shape = list(img.shape)
        if self.output_size[0] / float(image_shape[0]) <= \
             self.output_size[1] / float(image_shape[1]):
            out_h = self.output_size[0]
            out_w = int(self.output_size[0] * image_shape[1] / image_shape[0])
        else:
            out_h = int(self.output_size[1] * image_shape[0] / image_shape[1])
            out_w = self.output_size[1]

        key_list = sample.keys()
        for key in key_list:
            if key != 'image':
                continue
            img = sample[key]
            h, w = img.shape[:2]
            img = cv2.resize(img, dsize=(out_w,out_h), interpolation=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation)
            sample[key] = img
        
        return sample

class RandomCrop(object):
    def __init__(self,crop_size, image_value=127):
        assert isinstance(crop_size, (tuple,list))
        if len(crop_size) == 1:
            self.crop_size = (crop_size[0],crop_size[0])
        else:
            self.crop_size = crop_size
        self.IMAGE_VALUE = image_value

    def __call__(self,sample):
        rand_pad = random.uniform(0, 1)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            img = sample[key]
            h,w = img.shape[:2]
            new_h,new_w = self.crop_size
            pad_w = new_w - w
            pad_h = new_h - h
            w_begin = max(0,-pad_w)
            h_begin = max(0,-pad_h)
            pad_w = max(0,pad_w)
            pad_h = max(0,pad_h)
            w_begin = int(w_begin * rand_pad)
            h_begin = int(h_begin * rand_pad)
            w_end = w_begin + min(w,new_w)
            h_end = h_begin + min(h,new_h)
            shape = list(img.shape)
            shape[0] = new_h
            shape[1] = new_w
            new_img = np.zeros(shape,dtype=np.float)
            new_img.fill(self.IMAGE_VALUE)
            new_img[pad_h//2:min(h,new_h)+pad_h//2,pad_w//2:min(w,new_w)+pad_w//2] = img[h_begin:h_end,w_begin:w_end]
            sample[key] = new_img
        return sample
