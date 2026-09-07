import os
import torch
from torch.functional import Tensor
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
try:
    from model.position_encoding import build_position_encoding
except ModuleNotFoundError:
    # Fallback for direct execution from inside `model/`
    from position_encoding import build_position_encoding
import random
import numpy as np
import torch.nn.functional as F

import clip
import math

from typing import Dict, List

def upsample_pos_emb(emb, new_size):
    # upsample the pretrained embedding for higher resolution
    # emb size NxD
    first = emb[:1, :]
    emb = emb[1:, :]
    N, D = emb.size(0), emb.size(1)
    size = int(np.sqrt(N))
    assert size * size == N
    #new_size = size * self.upsample
    emb = emb.permute(1, 0)
    emb = emb.view(1, D, size, size).contiguous()
    emb = F.upsample(emb, size=new_size, mode='bilinear',)
    emb = emb.view(D, -1).contiguous()
    emb = emb.permute(1, 0)
    emb = torch.cat([first, emb], 0)
    emb = nn.parameter.Parameter(emb.half())
    return emb

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class Backbone(nn.Module):
    def __init__(self, name: str,
                 train_backbone: bool = True,
                 pretrained: bool = True):
        super().__init__()
        name = name.split('_')[0]
        if name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                pretrained=pretrained,
                norm_layer=torch.nn.BatchNorm2d)

            NCDICT = {
                'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048,
                'resnet101': 2048,
            }
            num_channels = NCDICT[name]
            # if not train_backbone:
            #     for name, parameter in backbone.named_parameters():
            #         parameter.requires_grad_(False)

            self.num_channels = num_channels
            return_layers = {'layer4': "final"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)


    def forward(self, input: Tensor):
        out = self.body(input)['final']
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args):
        super().__init__(backbone, position_embedding)
        self.backbone_name = args.backbone


    def forward(self, input: Tensor):
        if self.backbone_name.startswith('resnet'):
            out = self[0](input)
        else:
            NCDICT = {
                'ViT-B/16': 16,
                'ViT-B/32': 32,
                'ViT-L/14': 14
            }
            patch = NCDICT[self.backbone_name]
            out = self[0](input, patch)

        if out.dim() == 3:
            B, C, area = out.shape
            H = W = int(math.sqrt(area))
            out = out.view(B, C, H, W)
        pos = self[1](out).to(out.dtype)
        return out, pos

def ViT_forward(self, x: torch.Tensor, patch=14):
    self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (224 // patch, 224 // patch))
    x = self.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat(
        [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding_new.to(x.dtype)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = self.ln_post(x[:, 1:, :])

    if self.proj is not None:
        x = x @ self.proj

    return x.permute(0, 2, 1)


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if args.backbone.startswith('resnet'):
        backbone = Backbone(args.backbone, train_backbone, args.pretrained)
    else:
        backbone, _ = clip.load(args.backbone, device=args.device)
        backbone = backbone.visual
        backbone.requires_grad_(False)
        backbone.forward = ViT_forward.__get__(backbone, type(backbone))
        backbone.to(dtype=torch.float32)
        NCDICT = {
            'ViT-B/32': 512,
            'ViT-L/14': 768
        }
        num_channels = NCDICT[args.backbone]
        backbone.num_channels = num_channels

    num_channels = backbone.num_channels
    model = Joiner(backbone, position_embedding, args)
    model.num_channels = num_channels
    return model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='MLC model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.backbone = 'resnet50'
    args.device = 'cpu'
    args.img_size = 448
    args.hidden_dim = 768
    args.pretrained = True
    args.position_embedding = 'v2'
    args.seed = 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    backbone = build_backbone(args)
    print(backbone)

    x = torch.randn(1, 3, 224, 224)
    out = backbone(x)
    print(out[0].shape)
    print(out[1].shape)

