import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import copy
import pandas as pd

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, eyePacsTrainFiles, eyePacs_path):

        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.eyePacsTrainFiles = eyePacsTrainFiles
        self.eyePacs_path = eyePacs_path

        trainEyeQ = pd.read_csv('Label_EyeQ_train.csv')
        testtEyeQ = pd.read_csv('Label_EyeQ_test.csv')

        trn_df = trainEyeQ.loc[np.where(trainEyeQ['quality'].values == 0)].reset_index(drop = True)
        val_df = testtEyeQ.loc[np.where(testtEyeQ['quality'].values == 0)].reset_index(drop = True)

        ## goodNormalImagesNum = 1800
        ## trainGoodNormalImagesIndx = np.where(trn_df['DR_grade'].values == 0)[0][:goodNormalImagesNum]
        ## testtGoodNormalImagesIndx = np.where(val_df['DR_grade'].values == 0)[0][:goodNormalImagesNum]

        trainNonNormalImages = np.where(trn_df['DR_grade'].values != 0)[0]
        testtNonNormalImages = np.where(val_df['DR_grade'].values != 0)[0]

        ## trGood_normal = trainEyeQ.loc[trainGoodNormalImagesIndx, :].reset_index(drop = True)
        ## vlGood_normal = testtEyeQ.loc[testtGoodNormalImagesIndx, :].reset_index(drop = True)

        trGood_nonNormal = trainEyeQ.loc[trainNonNormalImages, :].reset_index(drop = True)
        vlGood_nonNormal = testtEyeQ.loc[testtNonNormalImages, :].reset_index(drop = True)

        ## self.eyeQFinalNames = [*trGood_normal['image'].values, *vlGood_normal['image'].values, \
        ##                        *trGood_nonNormal['image'].values, *vlGood_nonNormal['image'].values]

        self.eyeQFinalNames = [*trGood_nonNormal['image'].values, *vlGood_nonNormal['image'].values]

        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))

        self.is_train = is_train
        self.idrid_eyeQ_ratio = 83  ## 4509 / 54 = 83.5
        self.bs_idrid_eyeQ_ratio = 2  ## idrid 8 eyeQ 16

    def __getitem__(self, index):

        image_path = self.imgs_path[index % len(self.imgs_path)]
        label_path = image_path.replace('image', 'label')
        label_path = label_path.replace('.tif', '_ALL.tif') 

        image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')

        indx1 = index * self.bs_idrid_eyeQ_ratio
        indx2 = index * self.bs_idrid_eyeQ_ratio + 1
        if self.eyeQFinalNames[indx1].split('.')[0] in self.eyePacsTrainFiles:
            eyeQ_path = self.eyePacs_path + self.eyeQFinalNames[indx1].split('.')[0] + '.jpg'
        else:
            eyeQ_path = self.eyePacs_path + self.eyeQFinalNames[indx1].split('.')[0] + '.jpg'
        imgEyeQ_1 = Image.open(eyeQ_path)
        
        if self.eyeQFinalNames[indx2].split('.')[0] in self.eyePacsTrainFiles:
            eyeQ_path = self.eyePacs_path + self.eyeQFinalNames[indx2].split('.')[0] + '.jpg'
        else:
            eyeQ_path = self.eyePacs_path + self.eyeQFinalNames[indx2].split('.')[0] + '.jpg'
        imgEyeQ_2 = Image.open(eyeQ_path)


        if self.is_train == 1:
            if np.random.random_sample() <= 0.5:
                image = TF.adjust_brightness(image, brightness_factor = np.random.uniform(0.5, 1.5))
                image = TF.adjust_contrast(image, contrast_factor = np.random.uniform(0.5, 1.5))
                image = TF.adjust_saturation(image, saturation_factor = np.random.uniform(0, 1.5))
            
                imgEyeQ_1 = TF.adjust_brightness(imgEyeQ_1, brightness_factor = np.random.uniform(0.5, 1.5))
                imgEyeQ_1 = TF.adjust_contrast(imgEyeQ_1, contrast_factor = np.random.uniform(0.5, 1.5))
                imgEyeQ_1 = TF.adjust_saturation(imgEyeQ_1, saturation_factor = np.random.uniform(0, 1.5))

                imgEyeQ_2 = TF.adjust_brightness(imgEyeQ_2, brightness_factor = np.random.uniform(0.5, 1.5))
                imgEyeQ_2 = TF.adjust_contrast(imgEyeQ_2, contrast_factor = np.random.uniform(0.5, 1.5))
                imgEyeQ_2 = TF.adjust_saturation(imgEyeQ_2, saturation_factor = np.random.uniform(0, 1.5))

            if np.random.random_sample() <= 0.5:
                image, label = self.randomRotation(image, label)

                imgEyeQ_1, _ = self.randomRotation(imgEyeQ_1, imgEyeQ_1)
                imgEyeQ_2, _ = self.randomRotation(imgEyeQ_2, imgEyeQ_2)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

                imgEyeQ_1 = imgEyeQ_1.transpose(Image.FLIP_LEFT_RIGHT)
                imgEyeQ_2 = imgEyeQ_2.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random_sample() <= 0.25:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

                imgEyeQ_1 = imgEyeQ_1.transpose(Image.FLIP_TOP_BOTTOM)
                imgEyeQ_2 = imgEyeQ_2.transpose(Image.FLIP_TOP_BOTTOM)

            if np.random.random_sample() <= 1:
                # random.uniform(),从一个均匀分布[low,high)中随机采样
                crop_size = 256
                w = random.uniform(0, image.size[0] - crop_size)
                h = random.uniform(0, image.size[1] - crop_size)
                image = TF.crop(image, w, h, crop_size, crop_size)
                label = TF.crop(label, w, h, crop_size, crop_size)

                imgEyeQ_1 = TF.crop(imgEyeQ_1, w, h, crop_size, crop_size)
                imgEyeQ_2 = TF.crop(imgEyeQ_2, w, h, crop_size, crop_size)

        image = np.asarray(image)
        label = np.asarray(label)

        imgEyeQ_1 = np.asarray(imgEyeQ_1)
        imgEyeQ_2 = np.asarray(imgEyeQ_2)

        image = image.transpose(2, 0, 1)
        label = label.reshape(label.shape[0], label.shape[1])
        label = np.array(label)

        imgEyeQ_1 = imgEyeQ_1.transpose(2, 0, 1)
        imgEyeQ_2 = imgEyeQ_2.transpose(2, 0, 1)

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        return image, label, np.stack([imgEyeQ_1, imgEyeQ_2], axis = 0), filename

    def __len__(self):

        # 返回训练集大小
        return len(self.imgs_path) * int(self.idrid_eyeQ_ratio / self.bs_idrid_eyeQ_ratio)

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