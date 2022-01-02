# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Take the standard Transformer as T2T Transformer
"""
import sys
import torch.nn as nn
from timm.models.layers import DropPath
from .transformer_block import Mlp
from .statistic import MemStatistic

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)    # 这里乘3是分别是q、k、v
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        if MemStatistic.mem_eval:
            self.forward_mem_eval(x)

        B, N, C = x.shape
        # print(x.shape)  # torch.Size([32, 3136, 147])

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)  # 这里就1个head，是手动设置的
        # self.qkv(x) -> [B, N, C] -> [32, 3136, 192]
        # reshape -> [32, 3136, 3, 1, 64]
        # permute -> torch.Size([3, 32, 1, 3136, 64])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

    def forward_mem_eval(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        MemStatistic.record([x.shape], [qkv.shape], 'token_transformer_qkv')
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)  # 这里就1个head，是手动设置的
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        MemStatistic.push_reserve([v.shape])    # v此时是需要保留的
        MemStatistic.record([q.shape, k.shape], [attn.shape], 'token_transformer_attn')
        MemStatistic.pop_reserve()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        MemStatistic.record([attn.shape, v.shape], [x.shape], 'token_transformer_attn*v')
        MemStatistic.push_reserve([v.shape])
        x_proj = self.proj(x)
        MemStatistic.record([x.shape], [x_proj.shape], 'token_transformer_proj')
        MemStatistic.pop_reserve()
        x = self.proj_drop(x_proj)

        # skip connection
        MemStatistic.record([x.shape, v.shape], [x.shape], 'token_transformer_x+v')
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if MemStatistic.mem_eval:
            self.forward_mem_eval(x)

        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_mem_eval(self, x):
        MemStatistic.record([x.shape], [x.shape], 'token_transformer_norm1')  # norm1
        x = self.attn(self.norm1(x))

        MemStatistic.push_reserve([x.shape])
        MemStatistic.record([x.shape], [x.shape], 'token_transformer_norm2')  # norm2
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.drop_path(x_mlp)
        MemStatistic.pop_reserve()
        MemStatistic.record([x.shape, x_mlp.shape], [x.shape], 'token_transformer_x+x_mlp')

        return x