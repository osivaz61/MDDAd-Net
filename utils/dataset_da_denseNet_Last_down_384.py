import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from scipy.ndimage import zoom
import pandas as pd

print('user')
class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, eyePacsTrainFiles, eyePacs_path):

        self.data_path = data_path
        self.eyePacsTrainFiles = eyePacsTrainFiles
        self.eyePacs_path = eyePacs_path

        ## trainEyeQ = pd.read_csv('Label_EyeQ_train.csv')
        ## testtEyeQ = pd.read_csv('Label_EyeQ_test.csv')

        ## trn_df = trainEyeQ.loc[np.where(trainEyeQ['quality'].values == 0)].reset_index(drop = True)
        ## val_df = testtEyeQ.loc[np.where(testtEyeQ['quality'].values == 0)].reset_index(drop = True)

        ## trainNonNormalImages = np.where(trn_df['DR_grade'].values != 0)[0]
        ## testtNonNormalImages = np.where(val_df['DR_grade'].values != 0)[0]
        ## trGood_nonNormal = trn_df.loc[trainNonNormalImages, :].reset_index(drop = True)
        ## vlGood_nonNormal = val_df.loc[testtNonNormalImages, :].reset_index(drop = True)

        ## self.eyeQFinalNames = [*trGood_nonNormal['image'].values, *vlGood_nonNormal['image'].values]
        self.eyeQFinalNames = glob.glob(os.path.join(eyePacs_path, '*.png'))
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        self.is_train = is_train

        self.comLabels_2 = pd.DataFrame()
        self.comLabels_2['ID'] = np.concatenate((self.eyeQFinalNames, self.imgs_path))
        self.comLabels_2['SSL'] = np.concatenate((np.zeros(len(self.eyeQFinalNames)), np.ones(len(self.imgs_path))))
        print('user')
        self.crop_size = 384

    def __getitem__(self, index):

        ssl = torch.tensor(self.comLabels_2['SSL'].values[index]).float()
        if ssl == 1:
            image_path = self.imgs_path[index % len(self.imgs_path)]
            label_path = image_path.replace('image', 'label')
            label_path = label_path.replace('.tif', '_ALL.tif') 
            image = Image.open(image_path)
            label = Image.open(label_path)
            label = label.convert('L')
        else:
            eyeQ_path = self.eyeQFinalNames[index]
            image = Image.open(eyeQ_path.split('.')[0] + '.png')

        if self.is_train == 1:
            if np.random.random_sample() <= 0.5:
                image = TF.adjust_brightness(image, brightness_factor = np.random.uniform(0.5, 1.5))
                image = TF.adjust_contrast(image, contrast_factor = np.random.uniform(0.5, 1.5))
                image = TF.adjust_saturation(image, saturation_factor = np.random.uniform(0, 1.5))

            if np.random.random_sample() <= 0.5:
                if ssl == 1:
                    image, label = self.randomRotation(image, label)
                else:
                    image, _ = self.randomRotation(image, image)

            if np.random.random_sample() <= 0.25:
                if ssl == 1:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    label = label.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random_sample() <= 0.25:
                if ssl == 1:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                    label = label.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)

            if np.random.random_sample() <= 1:
                # random.uniform(),从一个均匀分布[low,high)中随机采样
                w = random.uniform(0, image.size[0] - self.crop_size)
                h = random.uniform(0, image.size[1] - self.crop_size)
                if ssl == 1:
                    image = TF.crop(image, w, h, self.crop_size, self.crop_size)
                    label = TF.crop(label, w, h, self.crop_size, self.crop_size)
                else:
                    image = TF.crop(image, w, h, self.crop_size, self.crop_size)

        if ssl == 1:
            image = np.asarray(image)
            label = np.asarray(label)
        else:
            image = np.asarray(image)

        if ssl == 1:
            label = label.reshape(label.shape[0], label.shape[1])
            label = np.array(label)
        else:
            label = np.array(np.zeros((self.crop_size, self.crop_size)))

        image_down, label_down = self.down(image, label)

        image = image.transpose(2, 0, 1)
        image_down = image_down.transpose(2, 0, 1)

        return image, label, image_down, label_down

    def down(self, image, label = None):
        outsize = round(self.crop_size * 0.5 / 32) * 32
        image = zoom(image, (outsize / self.crop_size, outsize / self.crop_size, 1), order = 0)
        label = zoom(label, (outsize / self.crop_size, outsize / self.crop_size), order = 0)
        return image, label

    def __len__(self):

        return len(self.comLabels_2['SSL'].values)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        #random_angle = np.random.randint(1, 360)
        #return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)
        random_angle = np.random.randint(1, 4)
        return image.rotate(random_angle*90, mode), label.rotate(random_angle*90, Image.NEAREST)

    def padding_image(self,image, label, pad_to_h, pad_to_w):
        #新建长宽608像素，背景色为（0, 0, 0）的画布对象,即背景为黑，RGB是彩色三通道图像，P是8位图像
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        # 把新建的图像粘贴在原图上
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        return new_image, new_label

class FundusSeg_Loader_Test(Dataset):
    def __init__(self, data_path, is_train):

        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        label_path = label_path.replace('.tif', '_ALL.tif') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')

        image = np.asarray(image)
        label = np.asarray(label)

        image = image.transpose(2, 0, 1)
        label = label.reshape(label.shape[0], label.shape[1])
        label = np.array(label)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        return image, label, filename

    def __len__(self):
        return len(self.imgs_path)