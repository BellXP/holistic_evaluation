# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import functools

import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding, ColumnParallelLinear
)
from ..peft import LoraColumnParallelLinear, LoraRowParallelLinear

from apex.normalization import FusedRMSNorm as RMSNorm

from transformers import Blip2Processor, Blip2Model

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    lora_rank: int = 16


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wk = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wv = LoraColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )
        self.wo = LoraRowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=True,
            input_is_parallel=True,
            init_method=default_linear_init,
            lora_rank=args.lora_rank
        )

        self.flash = True

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        if self.flash:
            output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        args: ModelArgs
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=True, gather_output=False, init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w2 = LoraRowParallelLinear(
            hidden_dim, dim, bias=True, input_is_parallel=True, init_method=default_linear_init, lora_rank=args.lora_rank
        )
        self.w3 = LoraColumnParallelLinear(
            dim, hidden_dim, bias=True, gather_output=False, init_method=default_linear_init, lora_rank=args.lora_rank
        )

    # @torch.compile
    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], prompt=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size = 336,
            patch_size = 14,
            in_chans = 3,
            embed_dim = 768,
            norm_layer = nn.LayerNorm,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = [patch_size, patch_size]
        self.img_size = [img_size, img_size]
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, with_visual=False):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=nn.init.normal_,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=default_linear_init,
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.image_words = 0
        if with_visual:
            print("build llama model with qformer")
            self.qformer = Blip2Model.from_pretrained("./blip2_opt2.7b/", torch_dtype=torch.float16)

            self.qformer.language_projection = None
            self.qformer.language_model = None

            self.qformer_proj = nn.Sequential(
                nn.Linear(768, params.dim),
                nn.LayerNorm(params.dim)
            )
            self.image_words = 32

        self.set_default_trainability()

    def get_trainable_params(self, phase='finetune'):
        trainable = {}
        for name, para in self.named_parameters():
            if not name.startswith("qformer."):
                if 'norm' in name or 'bias' in name or 'lora' in name:
                    trainable[name] = para

        return trainable

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False
            value.data = value.data.half()
        for key, value in self.get_trainable_params().items():
            value.data = value.data.float()
            value.requires_grad = True


    def encode_image(self, image):
        # [B, 32, 768]
        image_feats = self.qformer.get_qformer_features(pixel_values=image)
        image_feats = self.qformer_proj(image_feats.last_hidden_state)
        return image_feats

    def forward(self, examples, image=None):
        _bsz, seqlen = examples.shape
        h = self.tok_embeddings(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)
        start_pos = 0

        if image is not None:
            image_tokens = self.encode_image(image)
            h = torch.cat((image_tokens, h), dim=1)
            start_pos = image_tokens.shape[1]
            seqlen = h.shape[1]

        print(f"image: {start_pos}, text: {seqlen - start_pos}, seq_len: {seqlen}")

        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)
        mask[:, :, :, :start_pos] = 0
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, start_pos:, :])
        return output


    @torch.inference_mode()
    def forward_inference(self, tokens: torch.Tensor, start_pos: int, image=None):
        assert start_pos==0
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if image is not None:
            image_tokens = self.encode_image(image)
            h = torch.cat((image_tokens, h), dim=1)
            start_pos = start_pos + image_tokens.shape[1]
            seqlen = h.shape[1]

        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)
            mask[:, :, :, :start_pos] = 0


        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
