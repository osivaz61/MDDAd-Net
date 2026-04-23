import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train):

        self.is_train = is_train

        self.imgs_path = [f for f in glob.glob(data_path + "/train_*.bmp", recursive=True) if "anno" not in f]
        ## self.labels_path = glob.glob(os.path.join(data_path, 'train_*_anno.bmp'))

        self.lbl_imgs_path = self.imgs_path[:17]
        self.unlbl_imgs_path = self.imgs_path[17:]

        self.transform = A.Compose([A.Flip(p = 0.75), A.Transpose(p = 0.5), A.RandomRotate90(p = 1)])
        self.normalize = A.Compose([A.Normalize(mean, std), ToTensorV2()])
        self.crop_size = 256

        self.comLabels_2 = pd.DataFrame()
        self.comLabels_2['ID'] = np.concatenate((self.unlbl_imgs_path, self.lbl_imgs_path))
        self.comLabels_2['SSL'] = np.concatenate((np.zeros(len(self.unlbl_imgs_path)), np.ones(len(self.lbl_imgs_path))))

    def __getitem__(self, index):

        ssl = torch.tensor(self.comLabels_2['SSL'].values[index]).float()
        if ssl == 1:
            image_path = self.lbl_imgs_path[index % len(self.lbl_imgs_path)]
            label_path = image_path.replace('.bmp', '_anno.bmp')  
            image = Image.open(image_path)
            label = Image.open(label_path)
            label = label.convert('L')
            image = np.array(image)
            label = np.array(label)
            label[np.where(label > 0)] = 1
        else:
            eyeQ_path = self.unlbl_imgs_path[index]
            image = Image.open(eyeQ_path)
            image = np.array(image)

        if self.is_train == 1:

            if ssl == 1:
                augment = self.transform(image = image, mask = label)
                normalize = self.normalize(image = augment['image'], mask = augment['mask'])
                image = normalize['image']
                label = normalize['mask'].long()
            else:
                augment = self.transform(image = image)
                normalize = self.normalize(image = augment['image'])
                image = normalize['image']

            if np.random.random_sample() <= 1:
                # random.uniform(),从一个均匀分布[low,high)中随机采样
                w = int(random.uniform(0, image.shape[1] - self.crop_size))
                h = int(random.uniform(0, image.shape[2] - self.crop_size))
                if ssl == 1:
                    image = TF.crop(image, w, h, self.crop_size, self.crop_size)
                    label = TF.crop(label, w, h, self.crop_size, self.crop_size)
                else:
                    image = TF.crop(image, w, h, self.crop_size, self.crop_size)

        if ssl == 1:
            label = label.reshape(label.shape[0], label.shape[1])
            label = np.array(label)
        else:
            label = np.array(np.zeros((self.crop_size, self.crop_size)))

        image_down, label_down = self.down(image.permute(1,2,0), label)
        image_down = torch.tensor(image_down.transpose(2, 0, 1))

        return image, label, image_down, label_down

    def down(self, image, label = None):
        outsize = round(self.crop_size * 0.5 / 32) * 32
        image = zoom(image, (outsize / self.crop_size, outsize / self.crop_size, 1), order = 0)
        label = zoom(label, (outsize / self.crop_size, outsize / self.crop_size), order = 0)
        return image, label

    def __len__(self):

        return len(self.comLabels_2['SSL'].values)

class FundusSeg_Loader_Test(Dataset):
    def __init__(self, data_path):

        self.imgs_path = [f for f in glob.glob(data_path + "/test*.bmp", recursive=True) if "anno" not in f]

        self.transform = A.Compose([A.Resize(448, 448, p = 1)])
        self.normalize = A.Compose([A.Normalize(mean, std), ToTensorV2()])

    def __getitem__(self, index):
        
        image_path = self.imgs_path[index]
        label_path = image_path.replace('.bmp', '_anno.bmp') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')
        label = np.array(label)
        label[np.where(label > 0)] = 1

        augment = self.transform(image = np.array(image), mask = label)
        normalize = self.normalize(image = augment['image'], mask = augment['mask'])
        img = normalize['image']
        mask = normalize['mask'].long()

        return img, mask

    def __len__(self):
        return len(self.imgs_path)