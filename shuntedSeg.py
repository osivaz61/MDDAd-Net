import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C
                k2, v2 = kv2[0], kv2[1]
                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

                x = torch.cat([x1,x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H,W

class ShuntedTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, num_conv=0):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(num_conv)#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward_features(self, x):
        
        outs = []
        outs.append(x)
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1).contiguous()
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i == self.num_stages - 1:
                x = x.flatten(2).transpose(1, 2)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


from einops import rearrange, repeat
class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # B, C, H, W ==> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C//4)
        x = self.norm(x)
        x = x.reshape(B, H*2, W*2, C//4)

        return x

class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        # B, C, H, W ==> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H*self.dim_scale, W*self.dim_scale, self.output_dim)

        # x.permute(0, 3, 1, 2)
        return x

class ShuntedTransformerDecoder(nn.Module):
    def __init__(self, num_classes, drop_path_rate = 0.0):
        super().__init__()

        norms = [] 
        stages = []
        seg_layers = []
        expand_layers = []
        n_stages_encoder = 4
        concat_back_dim = []
        depths = [2, 2, 2, 2]
        encoder_output_channels = [64, 128, 256, 512]
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder-1)*2)]

        embed_dims = [512, 256, 128, 64] 
        num_heads = [16, 8, 4, 2] 
        mlp_ratios = [4, 4, 8, 8] 
        drop_rate = 0.
        qkv_bias = True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        sr_ratios = [1, 2, 4, 8]
        qk_scale = None
        attn_drop_rate = 0.

        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=2,
                norm_layer = nn.LayerNorm,
            ))
            stages.append(nn.ModuleList([Block(
                dim = embed_dims[s], num_heads = num_heads[s], mlp_ratio = mlp_ratios[s], qkv_bias = qkv_bias, qk_scale = qk_scale,
                drop = drop_rate, attn_drop = attn_drop_rate, drop_path = dpr[sum(depths[:s-1]):sum(depths[:s])][j], norm_layer = norm_layer,
                sr_ratio = sr_ratios[s])
                for j in range(depths[s])]))
            
            norms.append(norm_layer(embed_dims[s]))
            
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(2*input_features_skip, input_features_skip))

        expand_layers.append(FinalPatchExpand_X4(input_resolution = None, dim = encoder_output_channels[0], dim_scale = 4, norm_layer = nn.LayerNorm,))
        stages.append(nn.Identity())
        norms.append(norm_layer(embed_dims[-1]))
        seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias = True))

        self.norms = nn.ModuleList(norms)
        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, skips):

        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s+2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)

            H, W = x.shape[1], x.shape[2]
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], -1).contiguous()
            if not isinstance(self.stages[s], nn.Identity):
                for stageeee in self.stages[s]:
                    x = stageeee(x, H, W)
            else:
                x = self.stages[s](x)

            x = self.norms[s](x)

            if s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()))
            lres_input = x.reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        return seg_outputs


@register_model
def shunted_t(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 4, 1], sr_ratios=[8, 4, 2, 1], num_conv=0,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def shunted_s(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 4, 12, 1], sr_ratios=[8, 4, 2, 1], num_conv=1, **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def shunted_b(pretrained=False, **kwargs):
    model = ShuntedTransformer(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        **kwargs)
    model.default_cfg = _cfg()

    return model

from timm.models.helpers import load_checkpoint
class HybridModel(nn.Module):
    def __init__(self, num_classes = 24):
        super().__init__()

        self.encoder = shunted_s(drop_path_rate = 0.3)
        load_checkpoint(self.encoder, "/arf/home/osivaz/preweights/ckpt_S.pth", strict = True)
        self.encoder.head = nn.Identity()
        self.decoder = ShuntedTransformerDecoder(num_classes = num_classes, drop_path_rate = 0.3)
        self.decoder2 = ShuntedTransformerDecoder(num_classes = num_classes, drop_path_rate = 0.3)

    def forward(self, x1, x2):
        skips = self.encoder(x1)
        if x2 is not None:
            skips_128 = self.encoder(x2)
            return self.decoder(skips), self.decoder2(skips_128)
        return self.decoder(skips), None

## model = HybridModel(num_classes = 7).cuda()
## outputSegmodel = print(model(torch.randn(4, 3, 384, 384).cuda())[0].shape)

