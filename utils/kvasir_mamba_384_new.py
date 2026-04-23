import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from PIL import ImageEnhance
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF

mean =  [0.485, 0.456, 0.406]
std =  [0.229, 0.224, 0.225]

def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomCrop(image, label):
    border = 30
    image_width    = image.size[0]
    image_height   = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

def randomPeper_eg(img):
    
    img  = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY]  = 255
            
    return Image.fromarray(img)

def cv_random_flip_ji(img):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def randomCrop_ji(image):
    border = 30
    image_width    = image.size[0]
    image_height   = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region)


def randomRotation_ji(image):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
    return image


class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train):

        self.is_train = is_train

        self.imgs_path = [f for f in glob.glob(data_path + "images/*.jpg", recursive = True) if "anno" not in f]
        ## self.labels_path = glob.glob(os.path.join(data_path, 'train_*_anno.bmp'))

        self.lbl_imgs_path = self.imgs_path[:200]
        self.unlbl_imgs_path = self.imgs_path[200:800]
        self.crop_size = 384

        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([transforms.ToTensor()])

        self.comLabels_2 = pd.DataFrame()
        self.comLabels_2['ID'] = np.concatenate((self.unlbl_imgs_path, self.lbl_imgs_path))
        self.comLabels_2['SSL'] = np.concatenate((np.zeros(len(self.unlbl_imgs_path)), np.ones(len(self.lbl_imgs_path))))

    def __getitem__(self, index):

        ssl = torch.tensor(self.comLabels_2['SSL'].values[index]).float()
        if ssl == 1:
            image_path = self.lbl_imgs_path[index % len(self.lbl_imgs_path)]
            label_path = image_path.replace('images', 'masks')
            image = self.rgb_loader(image_path)
            label = self.binary_loader(label_path)
        else:
            eyeQ_path = self.unlbl_imgs_path[index]
            image = self.rgb_loader(eyeQ_path)

        if self.is_train == 1:

            if ssl == 1:

                image, label = cv_random_flip(image, label)
                image, label = randomCrop(image, label)
                image, label = randomRotation(image, label)
                image = colorEnhance(image)
                label = randomPeper_eg(label)

                image = self.img_transform(image)
                label = self.gt_transform(label)[0]
            else:

                image = cv_random_flip_ji(image)
                image = randomCrop_ji(image)
                image = randomRotation_ji(image)
                image = colorEnhance(image)
                image = randomPeper_eg(image)

                image = self.img_transform(image)

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

        image_down, label_down = self.down(image.permute(1, 2, 0), label)
        image_down = torch.tensor(image_down.transpose(2, 0, 1))

        label[np.where(label <= 0.5)] = 0
        label[np.where(label > 0.5)] = 1

        label_down[np.where(label_down <= 0.5)] = 0
        label_down[np.where(label_down > 0.5)] = 1

        return image, label, image_down, label_down
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def down(self, image, label = None):
        outsize = round(self.crop_size * 0.5 / 32) * 32
        image = zoom(image, (outsize / self.crop_size, outsize / self.crop_size, 1), order = 0)
        label = zoom(label, (outsize / self.crop_size, outsize / self.crop_size), order = 0)
        return image, label

    def __len__(self):

        return len(self.comLabels_2['SSL'].values)

class FundusSeg_Loader_Test(Dataset):
    def __init__(self, data_path):

        self.imgs_path = [f for f in glob.glob(data_path + "images/*.jpg", recursive = True) if "anno" not in f]
        self.imgs_path = self.imgs_path[800:900]
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
        print('bsivaz')

    def __getitem__(self, index):
        
        image_path = self.imgs_path[index % len(self.imgs_path)]
        label_path = image_path.replace('images', 'masks')
        image = self.rgb_loader(image_path)
        image = self.transform(image)
        label = self.binary_loader(label_path)
        label = self.gt_transform(label)[0]

        label[torch.where(label <= 0.5)] = 0
        label[torch.where(label > 0.5)] = 1

        return image, label

    def __len__(self):
        return len(self.imgs_path)
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')