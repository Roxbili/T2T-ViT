# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import sys
import math

import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
from .statistic import MemStatistic

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'Search_model': _cfg(),
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64, kernel_size=7, overlap_ratio=0.2):
        super().__init__()
        sp0_stride = round(kernel_size * (1 - overlap_ratio))

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=kernel_size, stride=sp0_stride, padding=(2, 2)) # 上下左右各添加2行/列
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * kernel_size * kernel_size, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        # self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately
        self.num_patches = self._unfold_shape(img_size, kernel_size, sp0_stride, 2)
        self.num_patches = self._unfold_shape(self.num_patches, 3, 2, 1)
        self.num_patches = self._unfold_shape(self.num_patches, 3, 2, 1)
        self.num_patches = self.num_patches ** 2

    def _unfold_shape(self, size: int, kernel_size: int, stride:int, padding: int):
        return math.floor((size - kernel_size + 2 * padding) / stride + 1)

    def forward(self, x):
        if MemStatistic.mem_eval:
            self.forward_mem_eval(x)
        # step0: soft split
        # print(x.shape) # torch.Size([32, 3, 224, 224])

        x = self.soft_split0(x).transpose(1, 2)     # floor((feature_len - kernel_size + 2 * padding) / stride + 1)
        # print(x.shape) # torch.Size([32, 3136, 147])

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        # print(x.shape)  # torch.Size([32, 3136, 64])
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))  # 这里其实就是把Token序列换源成了图片矩阵
        # x: [32, 64, 56, 56]
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2) # [32, 576, 784]

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x
    
    def forward_mem_eval(self, x):
        '''峰值内存估算'''
        x_sp0 = self.soft_split0(x).transpose(1, 2)     # floor((feature_len - kernel_size + 2 * padding) / stride + 1)
        MemStatistic.record([x.shape], [x_sp0.shape], 't2t_module_sp0')    # transpose暂时忽略

        x = self.attention1(x_sp0)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))  # 这里其实就是把Token序列换源成了图片矩阵
        # x: [32, 64, 56, 56]
        # iteration1: soft split
        x_sp1 = self.soft_split1(x).transpose(1, 2) # [32, 576, 784]
        MemStatistic.record([x.shape], [x_sp1.shape], 't2t_module_sp1')    # transpose暂时忽略

        x = self.attention2(x_sp1)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x_sp2 = self.soft_split2(x).transpose(1, 2)
        MemStatistic.record([x.shape], [x_sp2.shape], 't2t_module_sp2')    # transpose暂时忽略

        # final tokens
        x = self.project(x_sp2)
        MemStatistic.record([x_sp2.shape], [x.shape], 't2t_module_project')

        return x

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, kernel_size=7, overlap_ratio=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim,
                kernel_size=kernel_size, overlap_ratio=overlap_ratio)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x) # [batch_size, 196, embed_dim]
        # 为了和下一层维度对应起来，embed_dim要能够整除num_head

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # [32, 196, 384] -> [32, 197, 384]
        x = x + self.pos_embed  # 此处pos_embed是一个定值，直接计算得到
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        if MemStatistic.mem_eval:
            self.forward_mem_eval(x)

        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_mem_eval(self, x):
        '''用于评估推理过程中峰值内存'''

        # forward_features
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # [32, 196, 384] -> [32, 197, 384]
        x = x + self.pos_embed  # 此处pos_embed是一个定值，直接计算得到
        MemStatistic.record([x.shape], [x.shape], 't2t_vit_x+pos')
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        MemStatistic.record([x.shape], [x.shape], 't2t_vit_norm')
        x = self.norm(x)
        x = x[:, 0]

        # head
        x_head = self.head(x)
        MemStatistic.record([x.shape], [x_head.shape], 't2t_vit_head')
        assert len(MemStatistic.reserve_stack) == 0
        return x_head

@register_model
def search_model(pretrained=False, **kwargs):
    # print(kwargs)
    model = T2T_ViT(tokens_type='transformer', **kwargs)    # 所有参数均来源于create_model，否则就是默认值
    model.default_cfg = default_cfgs['Search_model']
    return model
