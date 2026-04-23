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

        self.imgs_path = [f for f in glob.glob(data_path + "images/*.png", recursive=True) if "anno" not in f]
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

trubaTestFiles = ['data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cjyzkmjy8evns070165gf9dmq.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5f26ebcuai0818xlwh6116.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju34aozyyy830993bn16u32n.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju13hp5rnbjx0835bf0jowgx.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8914beokbf0850isxpocrk.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju183od81ff608017ekzif89.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5h57xedz5h0755mjpc8694.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju33231uy4gi0993qc7b1jch.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7dvl5m2n4t0755hlnnjjet.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8ando2qqdo0818ck7i1be1.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5y4hgqmk0i08180rjhbwvp.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1ewnoh5z030855vpex9uzt.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3521y5d5mq0878t3ezsu4p.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju6yxyt0wh080871sqpepu47.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju302fqq9spc0878rrygyzzz.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5fi0yxd3ei0801v7u0yudn.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3xjqtpikx50817tppy6g84.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju17bz250pgd0799u1hqkj5u.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju35740hzm0g0993zl5ic246.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2ysg748ru80878sp6j0gm0.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/ck2395w2mb4vu07480otsu6tw.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju77k828z46w0871r0avuoo9.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2oq5570avm079959o20op1.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5jx7jzf7c90871c2i9aiov.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1g4nsb6ngy0799l4ezm8ab.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju14pxbaoksp0835qzorx6g6.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju358pwtdby20878cg7nm0np.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3xuj20ivgp0818mij8bjrd.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8c5223s8j80850b4kealt4.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3yb47cj1xq0817zfotbni4.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju89z6pqpqfx0817mfv8ixjc.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2wve9v7esz0878mxsdcy04.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2yi9tz8vky0801yqip0xyl.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju6z7e4bwgdd0987ogkzq9kt.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cjyzk8qieoboa0848ogj51wwm.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5wqonpm0e60801z88ewmy1.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2tvrvm53ws0801a0jfjdxg.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2yw4s7z7p20988lmf2gdgd.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8chdlqsu620755azjty1tj.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2syxa93yw40799x2iuwabz.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1gv7106qd008784gk603mg.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju30lncba3ny0878jwnous8n.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3tsh4lfsok0987w6x3a0v1.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2zp89k9q1g0855k1x0f1xa.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5hi52odyf90817prvcwg45.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8d2q30tfhs0801n7lx77xl.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2ulk385h170799rlklxob0.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju43jcqim2cp08172dvjvyui.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2pag1f0s4r0878h52uq83s.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju41kd7yl4nm0850gil5qqwh.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5c5xc7algd0817pb1ej5yo.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju88vx2uoocy075531lc63n3.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2iatlki5u309930zmgkv6h.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju887ftknop008177nnjt46y.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5x15djm7ae0755h8czf6nt.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2r7h21sj9608354gzks3ae.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7bf1lp1shi081835vs84lc.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5fw37edaae0801vkwvocn7.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju35i2e63uxr0835h7zgkg9k.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1brhsj3rls0855a1vgdlen.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3ykamdj9u208503pygyuc8.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju88cddensj00987788yotmg.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2zm0axztpe0988r8s9twjr.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju3xl264ingx0850rcf0rshj.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5hqz50e7o90850e0prlpa0.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5buy2bal250818ipl6fqwv.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2uokeg5jm20799xwgsyz89.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1cdxvz48hw0801i0fjwcnk.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju426tomlhll0818fc0i7nvh.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2nguelpmlj0835rojdn097.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8ca4geseia0850i2ru11hw.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2upu4evw7g08358guwozxv.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7amjna1ly40871ugiokehb.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8c3xs7sauj0801ieyzezr5.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7er4kc2opa0801anuxc0eb.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju336l68y7if0993wf092166.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju40taxlkrho0987smigg0x0.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7alcgr1lsr0871riqk84z7.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2uwz9f5yf1085506cfamfx.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju32gzs6xo8x0993r8tedbpb.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2zr3c3vwb00993jn06bbaz.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju6ywm40wdbo0987pbftsvtg.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju85omszllp30850b6rm9mi3.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju6x0yqbvxqt0755dhxislgb.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8c1a0ws7o208181c6lbsom.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2txjfzv60w098839dcimys.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8b0jr0r2oi0801jiquetd5.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju1d31sp4d4k0878r3fr02ul.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2raxlosl630988jdbfy9b0.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5wj0faly5008187n6530af.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju843yjskhq30818qre4rwm2.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju6v5ilsv8hk0850rb5sgh6o.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2z1nxlzaj40835wj81s1iy.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju7ekbo32pft0871fv7kzwb9.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju5fydrud94708507vo6oy21.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2hos57llxm08359g92p6jj.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju34uhepd3dd0799hs8782ad.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju8b4ja9r2s808509d45ma86.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju88nroho44508500129f1nh.jpg', 
'data/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG/images/cju2ricdv2iys0878sv1adh0u.jpg']

class FundusSeg_Loader_Test(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        ## self.imgs_path = [f for f in glob.glob(data_path + "images/*.png", recursive = True) if "anno" not in f]
        ## self.imgs_path = self.imgs_path[800:900]
        self.imgs_path = trubaTestFiles
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
        print('bsivaz')

    def __getitem__(self, index):
        
        image_path = self.data_path + self.imgs_path[index % len(self.imgs_path)]
        label_path = image_path.replace('images', 'masks')
        image = self.rgb_loader(image_path)
        image = self.transform(image)
        label = self.binary_loader(label_path)
        label = self.gt_transform(label)[0]

        label[torch.where(label <= 0.5)] = 0
        label[torch.where(label > 0.5)] = 1

        return image, label, image_path

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