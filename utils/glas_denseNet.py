import os
import torch
import glob
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

mean =  [0.787803, 0.512017, 0.784938]
std =  [0.428206, 0.507778, 0.426366]

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path):

        self.imgs_path = [f for f in glob.glob(data_path + "/train_*.bmp", recursive=True) if "anno" not in f]
        ## self.labels_path = glob.glob(os.path.join(data_path, 'train_*_anno.bmp'))

        self.lbl_imgs_path = self.imgs_path[:17]
        self.unlbl_imgs_path = self.imgs_path[17:]

        self.idrid_eyeQ_ratio = 4  
        self.bs_idrid_eyeQ_ratio = 2  

        self.transform = A.Compose([A.Resize(256, 256, p = 1), A.Flip(p = 0.75), 
                                      A.Transpose(p = 0.5), A.RandomRotate90(p = 1)])
        
        self.normalize = A.Compose([A.Normalize(mean, std), ToTensorV2()])


    def __getitem__(self, index):

        image_path = self.lbl_imgs_path[index % len(self.lbl_imgs_path)]
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

        indx1 = index * self.bs_idrid_eyeQ_ratio
        indx2 = index * self.bs_idrid_eyeQ_ratio + 1
        imgEyeQ_1 = Image.open(self.unlbl_imgs_path[indx1])
        imgEyeQ_2 = Image.open(self.unlbl_imgs_path[indx2])

        augment = self.transform(image = np.array(imgEyeQ_1))
        augment2 = self.transform(image = np.array(imgEyeQ_2))
        normalize = self.normalize(image = augment['image'])
        normalize2 = self.normalize(image = augment2['image'])
        img_unlbl = normalize['image']
        img_unlbl2 = normalize2['image']

        return img, mask, np.stack([img_unlbl, img_unlbl2], axis = 0)

    def __len__(self):

        return len(self.lbl_imgs_path) * int(self.idrid_eyeQ_ratio / self.bs_idrid_eyeQ_ratio)

class FundusSeg_Loader_Test(Dataset):
    def __init__(self, data_path):

        self.imgs_path = [f for f in glob.glob(data_path + "/train_*.bmp", recursive=True) if "anno" not in f]

        self.transform = A.Compose([A.Resize(256, 256, p = 1)])
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