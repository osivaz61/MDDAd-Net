import os
import torch
import pandas as pd
import torch.nn as nn
from torch import optim
from timm.utils import NativeScaler
import myCheckpointSaver as ckptSaver
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.dataset_da_denseNet_Last_down_384 import FundusSeg_Loader, FundusSeg_Loader_Test


dataset_name = "idrid" #

if dataset_name == "idrid":
    train_data_path = "data/ddr/train1024x1024/"
    valid_data_path = "data/ddr/test1024x1024/"
    N_epochs = 75
    lr_decay_step = [21, 42]
    lr_init = 0.00005
    batch_size = 8
    test_iter = 190

import torch.nn.functional as F
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_planes)

        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):

        x1 = self.avgpool(x) # b c 1 1

        x2 = self.fc1(x1) #b k 1 1

        x3 = nn.LeakyReLU(negative_slope = 0.2, inplace = False)(x2)
        x4 = self.fc2(x3).view(x3.size(0), -1) # b k
        return F.softmax(x4/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4,temperature=30, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.groups = min(in_planes,out_planes)
        self.groups = in_planes
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)
        self.finall = nn.BatchNorm2d(out_planes)
        self.rl = nn.LeakyReLU(negative_slope = 0.2, inplace = False)
        self.weight = nn.Parameter(torch.Tensor(K,in_planes, in_planes//self.groups, kernel_size, kernel_size), requires_grad=True)
        self.conv_1 = nn.Conv2d(in_planes,out_planes,1)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的#2 3 512 512
        softmax_attention = self.attention(x)

        batch_size, in_planes, height, width = x.size()
        #
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积

        weight = self.weight.view(self.K, -1) #4 432   # K output input 3 3

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size) #分组卷积将权重和特征图都进行分组

        output = output.view(batch_size, self.in_planes, output.size(-2), output.size(-1))
        output_f = self.conv_1(output)
        # output = self.finall(output)
        # output = self.rl(output)
        return output_f

class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=channel//16, kernel_size=1,stride=1,padding=0),
                nn.Conv2d(in_channels=channel//16, out_channels=1, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)

class DYBAC(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,stride, padding):
        super(DYBAC, self).__init__()
        self.dybac = nn.Sequential(
           sSE_Module(in_planes),
           Dynamic_conv2d(in_planes=in_planes, out_planes=out_planes,kernel_size=kernel_size,stride=stride,padding=padding),
           nn.BatchNorm2d(out_planes),
           ## nn.ReLU(inplace=False),
           nn.LeakyReLU(negative_slope = 0.2, inplace = False)
        )
    def forward(self, x):
        z = self.dybac(x)
        return  z

from mambavision.models.mamba_vision_my2 import MambaVision
class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64, n_channel = 3):
        super(Discriminator, self).__init__()
        self.vssm_encoder = MambaVision(depths = [1, 3, 8, 4],
                        num_heads = [2, 4, 8, 16],
                        window_size = [8, 8, 24, 12],
                        dim = 80,
                        in_dim = 32,
                        mlp_ratio = 4,
                        resolution = 384,
                        drop_path_rate = 0.5,
                        layer_scale = None,
                        layer_scale_conv = None)
        state_dict = torch.load("/arf/home/osivaz/preweights/mambavision_tiny_1k.pth.tar", weights_only = False)
        self.vssm_encoder.load_state_dict(state_dict['state_dict'], strict = False)
        self.in_features = self.vssm_encoder.head.in_features
        self.vssm_encoder.avgpool = nn.Identity()
        self.vssm_encoder.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Conv2d(self.in_features, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(32, 2, kernel_size = 1, stride = 1, padding = 0),
            )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, map, feature):
        
        x = self.vssm_encoder(feature, map)
        x = x.reshape(x.shape[0], self.in_features, 12, 12)
        x = self.classifier(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

from mambavision.models.mamba_vision_my2 import MambaVision
class Discriminator2(nn.Module):
    def __init__(self, num_classes, ndf = 64, n_channel = 3):
        super(Discriminator2, self).__init__()
        self.vssm_encoder = MambaVision(depths = [1, 3, 8, 4],
                        num_heads = [2, 4, 8, 16],
                        window_size = [8, 8, 24, 12],
                        dim = 80,
                        in_dim = 32,
                        mlp_ratio = 4,
                        resolution = 384,
                        drop_path_rate = 0.65,
                        layer_scale = None,
                        layer_scale_conv = None)
        state_dict = torch.load("/arf/home/osivaz/preweights/mambavision_tiny_1k.pth.tar", weights_only = False)
        self.vssm_encoder.load_state_dict(state_dict['state_dict'], strict = False)
        self.in_features = self.vssm_encoder.head.in_features
        self.vssm_encoder.avgpool = nn.Identity()
        self.vssm_encoder.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Conv2d(self.in_features, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(32, 1, kernel_size = 1, stride = 1, padding = 0),
            )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, map, feature):

        x = self.vssm_encoder(feature, map)
        ## print(x.shape) ## (8, 40960)
        x = x.reshape(x.shape[0], self.in_features, 12, 12)
        x = self.classifier(x)
        ## x = self.avgpool(x)
        ## return x.view(x.size(0), -1)
        return x

import itertools
from torch.utils.data.sampler import Sampler
## Dataloader
def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

lbl_bs = 5
unLbl_bs = 10
import numpy as np
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, 200.0)

def calculate_aucPr(predList, labelList, lesion_id):

    num_threshold = 30
    re_array_ex = np.zeros((num_threshold + 1, 1))
    pr_array_ex = np.zeros((num_threshold + 1, 1))

    kkk = -1
    for thr in range(num_threshold + 1):

        thres = thr / num_threshold
        sum_tp_ex, sum_tn_ex, sum_fp_ex, sum_fn_ex = 0, 0, 0, 0

        for pred_soft, label in zip(predList, labelList):

            tp_ex, tn_ex, fp_ex, fn_ex = 0, 0, 0, 0

            label_ex = np.zeros((label.shape[1], label.shape[2]))
            label_ex[np.where(label[0] == lesion_id)] = 1

            pred_soft_ex = pred_soft[0][lesion_id]
            pred_binary_ex = (pred_soft_ex >= thres) * 1

            tp = sum(sum(np.logical_and(pred_binary_ex, label_ex).astype(int)))
            fp = sum(sum(pred_binary_ex)) - tp
            fn = sum(sum(label_ex)) - tp

            sum_tp_ex = sum_tp_ex + tp
            sum_fp_ex = sum_fp_ex + fp
            sum_fn_ex = sum_fn_ex + fn

        if (sum_tp_ex + sum_fp_ex) == 0 or (sum_tp_ex + sum_fn_ex) == 0:
            ## print("CC_1 " + str(lesion_id))
            continue

        re_ex = (sum_tp_ex / (sum_tp_ex + sum_fn_ex))
        pr_ex = (sum_tp_ex / (sum_tp_ex + sum_fp_ex))

        if re_ex == 0 and pr_ex == 0:
            ## print("CC_2 " + str(lesion_id))
            continue

        kkk = kkk + 1
        re_array_ex[kkk] = re_ex
        pr_array_ex[kkk] = pr_ex

        f1 = 2 * re_ex * pr_ex / (re_ex + pr_ex)
        if thr == num_threshold / 2:
            f1_05_ex = f1
            iou_ex = sum_tp_ex / (sum_tp_ex + sum_fp_ex + sum_fn_ex)

    re_array_ex = re_array_ex[:kkk, :]
    pr_array_ex = pr_array_ex[:kkk, :]

    re_array_ex = np.transpose(np.fliplr(np.transpose(re_array_ex)))
    pr_array_ex = np.transpose(np.fliplr(np.transpose(pr_array_ex)))
    re_array_ex = np.vstack((np.array([[0]]), re_array_ex))
    pr_array_ex = np.vstack((np.array([[1]]), pr_array_ex))
    pr_ex = np.trapz(pr_array_ex.ravel(), x = re_array_ex.ravel())

    return f1_05_ex, iou_ex, pr_ex

from torch.optim import lr_scheduler
from timm.utils import AverageMeter
def train_net(net, discriminator, discriminator2, device, epochs = N_epochs, batch_size = batch_size, lr = lr_init):

    eyePacs_path = "data/ddr/ddr_grade_1024x1024/"
    ## eyePacs_path = "D:/datasets/EyePACS_512/"
    eyePacsTrainFiles = pd.read_csv("/arf/home/osivaz/data/EyePACS_384/trainLabels.csv")['image'].values

    len_lbl = len(glob.glob(os.path.join(train_data_path + "/image/", '*.tif')))
    len_unlbl = len(glob.glob(os.path.join(eyePacs_path, '*.png')))

    train_dataset = FundusSeg_Loader(train_data_path, 1, eyePacsTrainFiles, eyePacs_path)
    valid_dataset = FundusSeg_Loader_Test(valid_data_path, 0)
    batch_sampler = TwoStreamBatchSampler(list(range(len_unlbl)), list(range(len_unlbl, len_unlbl + len_lbl)), unLbl_bs, lbl_bs)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, num_workers = 8, batch_sampler = batch_sampler,
                                               multiprocessing_context = 'fork', persistent_workers = False, prefetch_factor = 1)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, persistent_workers = False)
    print('Traing images: %s' % len(train_loader))
    print('Valid  images: %s' % len(valid_loader))

    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = 1e-5)
    optimizer_dsc = optim.Adam(discriminator.parameters(), lr = lr / 2, weight_decay = 1e-5)
    optimizer_dsc2 = optim.Adam(discriminator2.parameters(), lr = lr / 4, weight_decay = 1e-5)
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    second_order_dsc = hasattr(optimizer_dsc, 'is_second_order') and optimizer_dsc.is_second_order
    second_order_dsc2 = hasattr(optimizer_dsc2, 'is_second_order') and optimizer_dsc2.is_second_order
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr = 0.00075, steps_per_epoch = len(train_loader), epochs = epochs, pct_start = 0.2)
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor([1., 2, 1, 1.5, 1.5]).to(device))
    ## criterion = nn.CrossEntropyLoss()
    criterionReducNone = nn.CrossEntropyLoss(reduction = 'none')
    ## best_loss = float('inf')
    best_loss = 0
    start_metric_score = 0
    train_loss_list = []
    val_loss_list = []
    net.train()
    discriminator.train()
    discriminator2.train()
    beforeSoftmax = True
    save_aucpr_metric_epoch = 12

    decreasing = False
    saver = ckptSaver.CheckpointSaver(
            model = net, optimizer = optimizer, args = None, model_ema = None, amp_scaler = loss_scaler,
            checkpoint_dir = 'ckpt_ddr_shuntedSeg384_add3_3mf_new/', 
            recovery_dir = 'ckpt_ddr_shuntedSeg384_add3_3mf_new/', decreasing = decreasing, max_history = 7)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        lossesDsc_lbl = AverageMeter()
        lossesDsc_unlbl = AverageMeter()
        losses_2dec_lbl = AverageMeter()
        losses_2dec_unlbl = AverageMeter()
        lossesTrain1 = AverageMeter()
        lossesTrain2 = AverageMeter()
        lossesAdv = AverageMeter()
        lossesAdv2 = AverageMeter()
        lossesDsc2 = AverageMeter()
        lossesDsc2_128 = AverageMeter()

        train_loss = 0
        for i, (image, label, image_down, label_down) in enumerate(train_loader):

            image = image.to(device = device, dtype = torch.float32)
            label = label.to(device = device, dtype = torch.float32)

            image_down = image_down.to(device = device, dtype = torch.float32)
            label_down = label_down.to(device = device, dtype = torch.float32)

            net.train()
            discriminator.eval()
            discriminator2.eval()
            optimizer.zero_grad()
            optimizer_dsc.zero_grad()
            optimizer_dsc2.zero_grad()
            with amp_autocast():

                pred, pred128 = net(image[unLbl_bs - lbl_bs:], image_down[unLbl_bs - lbl_bs:])
                pred_unl, pred_unl128 = net(image[:unLbl_bs - lbl_bs], image_down[:unLbl_bs - lbl_bs])

                ss_unlbl = discriminator(torch.softmax(pred_unl[0], dim = 1), image[:unLbl_bs - lbl_bs])
                lossAdv = F.cross_entropy(ss_unlbl, torch.tensor([1] * len(ss_unlbl)).to(device).long())

                pred_unl128_soft = torch.softmax(pred_unl128[0], dim = 1)
                pred_unl128_soft_intrplt = F.interpolate(pred_unl128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                image_down_intrplt = F.interpolate(image_down[:unLbl_bs - lbl_bs], size = (384, 384), mode = 'nearest')
                ss_unlbl_128 = discriminator(pred_unl128_soft_intrplt, image_down_intrplt)
                lossAdv += 0 * F.cross_entropy(ss_unlbl_128, torch.tensor([1] * len(ss_unlbl_128)).to(device).long())

                if beforeSoftmax == True:
                    ## pred128_soft = torch.softmax(pred128[0], dim = 1)
                    pred_unl128_soft = torch.softmax(pred_unl128[0], dim = 1)

                    ## pred128_soft_intrplt = F.interpolate(pred128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                    pred_unl128_soft_intrplt = F.interpolate(pred_unl128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                    ## image_down_intrplt = F.interpolate(image_down, size = (384, 384), mode = 'nearest')

                    ## ss_mix_intrplt = discriminator2(torch.cat((pred_unl128_soft_intrplt, pred128_soft_intrplt), dim = 0), image)
                    ss_mix_intrplt = discriminator2(pred_unl128_soft_intrplt, image[:unLbl_bs - lbl_bs])
                    lossAdv2 = nn.BCEWithLogitsLoss()(ss_mix_intrplt.squeeze(1), torch.ones_like(ss_mix_intrplt.squeeze(1)).to(device))
                else:
                    pred128_intrplt = F.interpolate(pred128[0], size = (384, 384), mode = 'bilinear', align_corners = True)
                    pred_unl128_intrplt = F.interpolate(pred_unl128[0], size = (384, 384), mode = 'bilinear', align_corners = True)
                    image_down_intrplt = F.interpolate(image_down, size = (384, 384), mode = 'bilinear', align_corners = True)
                    ss_mix_intrplt = discriminator2(torch.softmax(torch.cat((pred_unl128_intrplt, pred128_intrplt), dim = 0), dim = 1), image_down_intrplt)
                    lossAdv2 = F.cross_entropy(ss_mix_intrplt, torch.tensor([1] * len(image_down_intrplt)).to(device).long())

                loss_sp_256 = criterion(pred[0], label[unLbl_bs - lbl_bs:].long())
                loss_sp_128 = criterion(pred128[0], label_down[unLbl_bs - lbl_bs:].long())

                out1_soft_lbl = torch.softmax(pred[0], dim = 1)
                out2_soft_lbl = torch.softmax(pred128[0], dim = 1)
                loss_2dec_lbl = F.mse_loss(out1_soft_lbl, F.interpolate(out2_soft_lbl, size = (384, 384), mode = 'bilinear', align_corners = True), reduction = 'mean')

                out1_soft_unlbl = torch.softmax(pred_unl[0], dim = 1)
                out2_soft_unlbl = torch.softmax(pred_unl128[0], dim = 1)
                loss_2dec_unlbl = F.mse_loss(out1_soft_unlbl, F.interpolate(out2_soft_unlbl, size = (384, 384), mode = 'bilinear', align_corners = True), reduction = 'mean')

                consistency_weight = get_current_consistency_weight(epoch)
                if epoch < 10:
                    loss = loss_sp_256 + loss_sp_128 + 0.25 * loss_2dec_lbl + 0.25 * loss_2dec_unlbl + 0.002 * lossAdv + 0 * lossAdv2
                else:
                    loss = loss_sp_256 + loss_sp_128 + 0.25 * loss_2dec_lbl + 0.25 * loss_2dec_unlbl + 0.002 * lossAdv + 0.003 * lossAdv2


            train_loss = train_loss + loss.item()
            loss_scaler(loss, optimizer, clip_grad = None, parameters = net.parameters(), create_graph = second_order)
            lossesTrain1.update(loss_sp_256, len(pred))
            lossesTrain2.update(loss_sp_128, len(pred))
            lossesAdv.update(lossAdv, len(pred_unl))
            lossesAdv2.update(lossAdv2, len(image))
            losses_2dec_lbl.update(loss_2dec_lbl, len(pred))
            losses_2dec_unlbl.update(loss_2dec_unlbl, len(pred_unl))

            net.eval()
            discriminator.train()
            discriminator2.train()
            optimizer.zero_grad()
            optimizer_dsc.zero_grad()
            optimizer_dsc2.zero_grad()
            with amp_autocast():

                with torch.no_grad():
                    pred, pred128 = net(image[unLbl_bs - lbl_bs:], image_down[unLbl_bs - lbl_bs:])
                    pred_unl, pred_unl128 = net(image[:unLbl_bs - lbl_bs], image_down[:unLbl_bs - lbl_bs])

                ss_lbl = discriminator(torch.softmax(pred[0], dim = 1).detach(), image[unLbl_bs - lbl_bs:])
                ss_unlbl = discriminator(torch.softmax(pred_unl[0], dim = 1).detach(), image[:unLbl_bs - lbl_bs])

                pred128_soft = torch.softmax(pred128[0], dim = 1)
                pred128_soft_intrplt = F.interpolate(pred128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                pred_unl128_soft = torch.softmax(pred_unl128[0], dim = 1)
                pred_unl128_soft_intrplt = F.interpolate(pred_unl128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                image_down_intrplt = F.interpolate(image_down, size = (384, 384), mode = 'nearest')

                ss_lbl_128 = discriminator(pred128_soft_intrplt.detach(), image_down_intrplt[unLbl_bs - lbl_bs:])
                ss_unlbl_128 = discriminator(pred_unl128_soft_intrplt.detach(), image_down_intrplt[:unLbl_bs - lbl_bs])

                lossDsc1 = F.cross_entropy(ss_lbl, torch.tensor([1] * lbl_bs).to(device).long())
                lossDsc2 = F.cross_entropy(ss_unlbl, torch.tensor([0] * (unLbl_bs - lbl_bs)).to(device).long())
                lossDsc1 += 0 * F.cross_entropy(ss_lbl_128, torch.tensor([1] * lbl_bs).to(device).long())
                lossDsc2 += 0 * F.cross_entropy(ss_unlbl_128, torch.tensor([0] * (unLbl_bs - lbl_bs)).to(device).long())

            loss_scaler((lossDsc1 + lossDsc2) * 0.5, optimizer_dsc, clip_grad = None, parameters = discriminator.parameters(), create_graph = second_order_dsc)
            lossesDsc_lbl.update(lossDsc1, len(ss_lbl))
            lossesDsc_unlbl.update(lossDsc2, len(ss_unlbl))

            ## discriminator 2
            optimizer.zero_grad()
            optimizer_dsc.zero_grad()
            optimizer_dsc2.zero_grad()
            with amp_autocast():
                with torch.no_grad():
                    pred, pred128 = net(image[unLbl_bs - lbl_bs:], image_down[unLbl_bs - lbl_bs:])
                    pred_unl, pred_unl128 = net(image[:unLbl_bs - lbl_bs], image_down[:unLbl_bs - lbl_bs])

                if beforeSoftmax == True:

                    ## pred128_soft = torch.softmax(pred128[0], dim = 1)
                    pred_unl128_soft = torch.softmax(pred_unl128[0], dim = 1)

                    ## pred128_soft_intrplt = F.interpolate(pred128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                    pred_unl128_soft_intrplt = F.interpolate(pred_unl128_soft, size = (384, 384), mode = 'bilinear', align_corners = True)
                    ## image_down_intrplt = F.interpolate(image_down, size = (384, 384), mode = 'nearest')

                    ss_mix_intrplt = discriminator2(pred_unl128_soft_intrplt.detach(), image[:unLbl_bs - lbl_bs])
                    lossDsc2_1 = nn.BCEWithLogitsLoss()(ss_mix_intrplt.squeeze(1), torch.zeros_like(ss_mix_intrplt.squeeze(1)).to(device))

                    ss_mix = discriminator2(torch.softmax(pred_unl[0], dim = 1).detach(), image[:unLbl_bs - lbl_bs])
                    lossDsc2_2 = nn.BCEWithLogitsLoss()(ss_mix.squeeze(1), torch.ones_like(ss_mix.squeeze(1)).to(device))

                else:
                    pred128_intrplt = F.interpolate(pred128[0], size = (384, 384), mode = 'bilinear', align_corners = True)
                    pred_unl128_intrplt = F.interpolate(pred_unl128[0], size = (384, 384), mode = 'bilinear', align_corners = True)
                    image_down_intrplt = F.interpolate(image_down, size = (384, 384), mode = 'bilinear', align_corners = True)
                    ss_mix_intrplt = discriminator2(torch.softmax(torch.cat((pred_unl128_intrplt, pred128_intrplt), dim = 0), dim = 1).detach(), image_down_intrplt)
                    lossDsc2_1 = F.cross_entropy(ss_mix_intrplt, torch.tensor([0] * len(image_down_intrplt)).to(device).long())
                    ss_mix = discriminator2(torch.softmax(torch.cat((pred_unl[0], pred[0]), dim = 0), dim = 1).detach(), image)
                    lossDsc2_2 = F.cross_entropy(ss_mix, torch.tensor([1] * len(image)).to(device).long())

            if epoch < 10:
                loss_scaler((lossDsc2_1 + lossDsc2_2) * 0, optimizer_dsc2, clip_grad = None, parameters = discriminator2.parameters(), create_graph = second_order_dsc2)
            else:
                loss_scaler((lossDsc2_1 + lossDsc2_2) * 0.5, optimizer_dsc2, clip_grad = None, parameters = discriminator2.parameters(), create_graph = second_order_dsc2)
            lossesDsc2_128.update(lossDsc2_1, len(image))
            lossesDsc2.update(lossDsc2_2, len(image_down))
            # Validation
            iv = 0
            if ((i + 1) % test_iter == 0):

                epoch_add = (i + 1) // test_iter

                net.eval()
                val_loss = 0
                predList, labelList = [], []
                valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, persistent_workers = False)
                for iv, (image, label, filename) in enumerate(valid_loader):

                    image = image.to(device = device, dtype = torch.float32)
                    label = label.to(device = device, dtype = torch.float32)
                    ## print(net(torch.randn(2,3,768,1024).cuda())[0].shape)
                    with amp_autocast():
                        with torch.no_grad():
                            ## print(image.shape)
                            pred, _ = net(image, None)
                            ## pred = net(image)
                            loss = criterion(pred[0], label.long())
                    val_loss = val_loss + loss.item()
                    pred_soft = torch.softmax(pred[0], dim = 1)
                    predList.append(np.float32(pred_soft.detach().cpu().numpy()))
                    labelList.append(label.cpu().detach().numpy())

                ## if epoch <= save_aucpr_metric_epoch and val_loss < best_loss:
                ##     best_loss = val_loss
                ##     ##     torch.save(net.state_dict(), '/content/drive/My Drive/SEBNet-main/snapshot/idrid_da.pth')
                ##     ##     print('saving model............................................')
                ##     print('saving model.......................................................................................')
                ##     best_score = val_loss
                ##     best_metric, best_epoch = saver.save_checkpoint(epoch + 1, metric = best_score)

                if epoch > save_aucpr_metric_epoch:
                    f1_ex, IoU_ex, aucPR_ex = calculate_aucPr(predList, labelList, 1)
                    f1_he, IoU_he, aucPR_he = calculate_aucPr(predList, labelList, 2)
                    f1_ma, IoU_ma, aucPR_ma = calculate_aucPr(predList, labelList, 3)
                    f1_se, IoU_se, aucPR_se = calculate_aucPr(predList, labelList, 4)

                    best_mF1 = (f1_ex + f1_he + f1_ma + f1_se) / 4
                    print(str(f1_ex) + " " + str(f1_he) + " " + str(f1_ma) + " " + str(f1_se))

                    best_mIoU = (IoU_ex + IoU_he + IoU_ma + IoU_se) / 4
                    print(str(IoU_ex) + " " + str(IoU_he) + " " + str(IoU_ma) + " " + str(IoU_se))

                    best_mAUC_PR = (aucPR_ex + aucPR_he + aucPR_ma + aucPR_se) / 4
                    print(str(aucPR_ex) + " " + str(aucPR_he) + " " + str(aucPR_ma) + " " + str(aucPR_se))

                    if (best_mF1 + best_mIoU + best_mAUC_PR) / 3 > start_metric_score:
                        start_metric_score = (best_mF1 + best_mIoU + best_mAUC_PR) / 3
                        print('saving model.......................................................................................')

                    best_score = (best_mF1 + best_mIoU + best_mAUC_PR) / 3
                    best_metric, best_epoch = saver.save_checkpoint((epoch + 1) * 10 + epoch_add, metric = best_score)

                val_loss_list.append(val_loss / iv)
                print('Loss/valid', val_loss / iv)
                del valid_loader
                ## print("Train : ", lossesTrain.avg.item(), " Adv : ", lossesAdv.avg.item(), " AdvSemi : ", lossesAdvSemi.avg.item(), " Dsc_lbl : ", lossesDsc_lbl.avg.item(), " Dsc_unlbl : ", lossesDsc_unlbl.avg.item())
                print("Tr1 : ", lossesTrain1.avg.item(), "Tr2 : ", lossesTrain2.avg.item(),  "Adv : ", lossesAdv.avg.item(), "Adv2 : ", lossesAdv2.avg.item())
                print("2dec_lbl : ", losses_2dec_lbl.avg.item(), " 2dec_unlbl : ", losses_2dec_unlbl.avg.item())
                print("Dsc1_lbl : ", lossesDsc_lbl.avg.item(), " Dsc1_unlbl : ", lossesDsc_unlbl.avg.item())
                print("Dsc2 : ", lossesDsc2.avg.item(), " Dsc2_128 : ", lossesDsc2_128.avg.item())
        train_loss_list.append(train_loss / i)
        scheduler.step()

import random
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from shuntedSeg import HybridModel
if __name__ == "__main__":
    device = torch.device('cuda')
    set_seed(2148)
    ## net = SEBNet(n_channels = 3, n_classes = 5)
    net = HybridModel(num_classes = 5)
    ## net = TimmModel()
    dsc = Discriminator(num_classes = 5)
    dsc2 = Discriminator2(num_classes = 5)
    net.to(device = device)
    dsc.to(device = device)
    dsc2.to(device = device)
    train_net(net, dsc, dsc2, device)