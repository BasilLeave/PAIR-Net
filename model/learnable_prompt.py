import math
import os
import time
import sys

import torch
import torch.nn as nn
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

import random
import numpy as np
try:
    from model.backbone import build_backbone
    from model.transformer import build_transformer
except ModuleNotFoundError:
    from backbone import build_backbone
    from transformer import build_transformer


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, clip_model, class_token_position, device, random_init=False):
        super(PromptLearner, self).__init__()
        n_cls = len(classnames)
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if random_init:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, device=device)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=device)

        # kaiming normal initialization
        nn.init.kaiming_uniform_(ctx_vectors)
        # nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # print('text encoder ln_final output shape: ', x.shape)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class WordEmbedding(nn.Module):
    def __init__(self, clip_name, device='cpu'):
        super(WordEmbedding, self).__init__()

        self.model, _ = clip.load(clip_name, device=device)
        self.model.eval()
        self.device = device

    def forward(self, classes):
        text_queries = [f'an aerial picture of {c}' for c in classes]
        with torch.no_grad():
            print(text_queries)
            text_token = clip.tokenize(text_queries)
            text_token = text_token.to(self.device)
            text_features = self.model.encode_text(text_token)

        return text_features


class Label2Visual(nn.Module):
    def __init__(self, backbone, transformer, args, classes=None):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        num_class = args.num_class
        self.backbone_name = args.backbone
        self.args = args

        hidden_dim = args.hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

        # learnable label embedding
        if args.learnable_label_embed:
            word2vec = WordEmbedding(args.clip_name, device=args.device).float()
            print(classes)
            self.prompt_learner = PromptLearner(n_ctx=args.n_ctx, classnames=classes, clip_model=word2vec.model,
                                                class_token_position=args.class_token_position,
                                                random_init=args.random_init, device=args.device)
            self.text_encoder = TextEncoder(word2vec.model)

            # 👇 【新增代码】：绝对冻结 Text Encoder 的所有参数！
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
            # 为了保险起见，确保 word2vec 模型本身也没有泄露梯度
            for param in word2vec.model.parameters():
                param.requires_grad = False

            # prompts = self.prompt_learner()
            # tokenized_prompts = self.prompt_learner.tokenized_prompts
            # text_encoder = TextEncoder(word2vec.model)
            # self.label_embed = text_encoder(prompts, tokenized_prompts)

        else:
            word2vec = WordEmbedding(args.clip_name, device=args.device).float()
            self.label_embed = word2vec(classes)
            print(self.label_embed.shape)
        
        # 👇 加上这块：平滑视觉特征的投影层初始化
        nn.init.normal_(self.input_proj.weight, std=0.01)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)

    def forward(self, input):
        if self.backbone_name.startswith('resnet'):
            src, pos = self.backbone(input)
        else:
            src, pos = self.backbone(input)
            pos = None
        # print('visual pos:', pos.shape)
        if self.args.learnable_label_embed:
            prompts = self.prompt_learner() 
            tokenized_prompts = self.prompt_learner.tokenized_prompts
            query_input = self.text_encoder(prompts, tokenized_prompts)
        else:
            query_input = self.label_embed.to(src.device)

        src = self.input_proj(src)

        hs, _, attn_maps, enc_attn_maps = self.transformer(src, query_input, pos)

        out = self.fc(hs)

        if isinstance(out, tuple):
            out = out[0]

        return out, attn_maps


def build_l2v(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Label2Visual(backbone, transformer, args, args.classes)

    return model