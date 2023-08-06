import logging
import math
from contextlib import contextmanager
from functools import partial, wraps
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from x_clip.mlm import MLM


# helper functions

def identity(t, *args, **kwargs):
    return t


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def max_neg_value(dtype):
    return -torch.finfo(dtype).max


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom


def log(t, eps=1e-20):
    return torch.log(t + eps)


def l2norm(t):
    return F.normalize(t, dim=-1)


def matrix_diag(t, labels):
    # 对角线上标签为1，其他为0，也就是构建标签矩阵
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    # 对角线为True的矩阵
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    for i in i_range:
        for j in j_range:
            if i == j:
                continue
            else:
                if labels[int(i)] == labels[int(j)] == 1:
                    diag_mask[i, j] = True
                elif labels[int(i)] == labels[int(j)] == 0 or labels[int(i)] != labels[int(j)]:
                    diag_mask[i, j] = False

    # diag_el = t.masked_select(diag_mask)
    diag_el = t.masked_fill(~diag_mask, 0.).sum(dim=-1)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el), diag_mask


# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner


# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all=False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device=device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device=device).topk(num_patches_keep, dim=-1).indices

        return x[batch_indices, patch_indices_keep]


# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head=64,
            heads=8,
            causal=False,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            checkpoint_during_training=False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim=dim, mult=ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb=None,
            mask=None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)


# text and vision transformers
class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb=None,
            causal=False,
            cls_token=True,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = cls_token

        self.transformer = Transformer(dim, dim_head=dim_head, causal=causal, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask=None):

        if x.ndim == 2:
            b, n, device = *x.shape, x.device

            x = self.token_emb(x)

        if x.ndim == 3:
            b, n, device = *x.shape[:-1], x.device

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device=device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device=device)

        if self.cls_token:
            cls_tokens = x.sum(dim=1) / math.sqrt(n)
            x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)

        out = self.transformer(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
        return out


def model_forward_with_context(
        *,
        fn,
        args,
        freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc


class CLIP(nn.Module):
    def __init__(
            self,
            *,
            dim_text,
            num_text_tokens,
            args,
            text_enc_depth=6,
            text_seq_len=512,
            text_heads=16,
            text_dim_head=64,
            text_pad_id=1,
            text_rotary_pos_emb=False,
            text_causal_mask=False,
            text_eos_id=None,
            checkpoint_during_training=False,
            **kwargs
    ):
        super().__init__()
        # store some parameters for access

        self.args = args

        self.dim_text = dim_text

        # instantiate text transformer
        self.text_pad_id = text_pad_id
        self.text_seq_len = text_seq_len

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(
            text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        self.text_transformer = TextTransformer(
            dim=dim_text,
            num_tokens=num_text_tokens + 1,
            max_seq_len=text_seq_len,
            depth=text_enc_depth,
            heads=text_heads,
            causal=text_causal_mask,
            dim_head=text_dim_head,
            rotary_pos_emb=text_rotary_pos_emb,
            checkpoint_during_training=checkpoint_during_training
        )

        mlm_kwargs, kwargs = groupby_prefix_and_trim('mlm_', kwargs)

        self.mlm = MLM(
            self.text_transformer,
            dim=dim_text,
            num_tokens=num_text_tokens,
            mask_prob=0.3,
            **mlm_kwargs
        )

        # temperature
        self.to_text_latent1 = nn.Linear(dim_text, dim_text, bias=False)

        self.to_text_latent2 = nn.Linear(dim_text, dim_text, bias=False)

        self.token_emb = nn.Embedding(num_text_tokens, dim_text)

        self.batch_norm = nn.BatchNorm1d(dim_text)

        self.fc = nn.Linear(2 * dim_text, 2)

    def forward(
            self,
            text1=None,
            text2=None,
            dropout=0.3,
            training_classifier=False,
            freeze_text_encoder=False,
    ):
        # derive text mask
        text_mask1 = text1 != self.text_pad_id
        text_mask2 = text2 != self.text_pad_id

        # ssl
        if training_classifier:
            text_args = (text1,)
            text_args = (*text_args, text_mask1)
            enc_text = model_forward_with_context(
                fn=self.text_transformer,
                args=text_args,
                freeze=freeze_text_encoder
            )

            CLS = enc_text[:, 0, :]
            CLS = l2norm(self.to_text_latent1(CLS))
            CLS = self.batch_norm(self.to_text_latent2(CLS))

            enc_text = enc_text[:, 1:, :].mean(dim=1)
            enc_text = torch.cat([enc_text, CLS], dim=-1)

            return self.fc(enc_text)

        text1_ssl_loss = self.mlm(text1, mask=text_mask1)
        text2_ssl_loss = self.mlm(text2, mask=text_mask2)
        text_args1 = (text1,)
        text_args2 = (text2,)

        text_args1 = (*text_args1, text_mask1)
        enc_text1 = model_forward_with_context(
            fn=self.text_transformer,
            args=text_args1,
            freeze=freeze_text_encoder
        )
        text_args2 = (*text_args2, text_mask2)

        enc_text2 = model_forward_with_context(
            fn=self.text_transformer,
            args=text_args2,
            freeze=freeze_text_encoder
        )

        CLS1 = enc_text1[:, 0, :]
        CLS2 = enc_text2[:, 0, :]

        CLS1 = l2norm(self.to_text_latent1(CLS1))
        CLS2 = l2norm(self.to_text_latent1(CLS2))

        CLS1 = self.batch_norm(self.to_text_latent2(CLS1))
        CLS2 = self.batch_norm(self.to_text_latent2(CLS2))

        return CLS1, CLS2, text1_ssl_loss + text2_ssl_loss
