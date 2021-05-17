import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from dataloaders import custom_transforms as tr
from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageFile
import io
import scipy.io.wavfile as wf
from utils.logger import logger
import time
import librosa

ImageFile.LOAD_TRUNCATED_IMAGES = True
def pil_loader(filename, label=False):
    ext = os.path.splitext(filename)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
        return img
    elif ext == '.wav':
        rate, data = wf.read(filename)
        if rate != 16000:
            raise RuntimeError('input wav must be sampled at 16,000 Hz, get %d Hz'%rate)
        if data.ndim > 1:
            # take the left channel
            data = data[:, 0]
        if data.shape[0] < 16000*10:
            # make the wav at least 10-second long
            data = np.tile(data, (16000*10 + data.shape[0] - 1) // data.shape[0])
        # take the first 10 seconds
        data = np.reshape(data[:16000*10], [-1]).astype(np.float32)
        return data
    elif ext == '.npy':
        data = np.load(filename, allow_pickle=True)
        return data.T.reshape((1, 64, -1))[:,:,:1000]
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

def wav2spec(wav):
    linear_spect = librosa.stft(wav, n_fft=512, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

class BaseDataset(Dataset,ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ignore_index = 255

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __str__(self):
        pass
    
    @staticmethod
    def modify_commandline_options(parser,istrain=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def transform_train(self):
        temp = []
        temp.append(tr.Resize(self.args.input_size))
        temp.append(tr.RandomHorizontalFlip())
        temp.append(tr.RandomRotate(15))
        temp.append(tr.RandomCrop(self.args.input_size))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms

    def transform_validation(self):
        temp = []
        temp.append(tr.Resize(self.args.input_size))
        # temp.append(tr.RandomCrop(self.args.input_size))
        temp.append(tr.ToTensor())
        composed_transforms = transforms.Compose(temp)
        return composed_transforms
