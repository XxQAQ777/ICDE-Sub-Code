
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .filters import sinc_filter_1d, custom_upsample, custom_downsample

# This file adapts the original U-Net used in EDM [1] to time series data, adding custom resampling filters and positional embeddings.
# In addition, we add two alternative architectures: a ConvLSTM-based architecture and a simple attention-based architecture. 
# [1] https://github.com/NVlabs/edm/blob/main/training/networks.py

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            init_mode='kaiming_normal',
            init_weight=1,
            init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_features,
            fan_out=out_features)
        self.weight = nn.Parameter(weight_init(
            [out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(
            weight_init(
                [out_features],
                **init_kwargs) *
            init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

# ----------------------------------------------------------------------------
# Convolutional layer with optional (custom) up/downsampling.


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        alias_free=True,
        kaiser_size=6,
        kaiser_beta=3,
        init_mode='kaiming_normal',
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.alias_free = alias_free
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels *
            kernel *
            kernel,
            fan_out=out_channels *
            kernel *
            kernel)
        self.weight = nn.Parameter(
            weight_init([out_channels, in_channels, kernel],
                        **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(
            weight_init(
                [out_channels],
                **init_kwargs) *
            init_bias) if kernel and bias else None
        f = sinc_filter_1d(size=kaiser_size, beta=kaiser_beta)
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(
            x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0

        if self.up:
            x = custom_upsample(x, f)
        if self.down:
            x = custom_downsample(x, f)
        if w is not None:
            x = nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x

# ----------------------------------------------------------------------------
# Group & layer normalization.


class GroupNorm(nn.Module):
    def __init__(
            self,
            num_channels,
            num_groups=32,
            min_channels_per_group=4,
            eps=1e-5):
        super().__init__()
        self.num_groups = min(
            num_groups,
            num_channels //
            min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = nn.functional.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(
                x.dtype), bias=self.bias.to(
                x.dtype), eps=self.eps)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum(
            'ncq,nck->nqk',
            q.to(
                torch.float32),
            (k /
             np.sqrt(
                 k.shape[1])).to(
                torch.float32)).softmax(
                    dim=2).to(
                        q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(
                torch.float32), output=w.to(
                torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32),
                          db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32),
                          db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.


class UNetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_proj=False,
        adaptive_scale=True,
        rope_attn=True,
        alias_free=True,
        kaiser_size=6,
        kaiser_beta=3,
        init=dict(),
        init_zero=dict(
            init_weight=0),
        filter_kwargs=dict(),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        if not attention:
            self.num_heads = 0
        elif num_heads is not None:
            self.num_heads = num_heads
        else:
            self.num_heads = out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.rope_attn = rope_attn
        filter_kwargs = dict(
            alias_free=alias_free,
            kaiser_size=kaiser_size,
            kaiser_beta=kaiser_beta)

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            **init,
            **filter_kwargs)
        self.affine = Linear(
            in_features=emb_channels, out_features=out_channels *
            (2 if adaptive_scale else 1),
            **init)
        # self.affine = Linear(in_features=96, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            **init_zero,
            **filter_kwargs)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                **init,
                **filter_kwargs)

        if self.num_heads:
            if rope_attn:
                self.rope = RotaryPositionEmbedding(dim=out_channels)
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1d(in_channels=out_channels,
                              out_channels=out_channels * 3,
                              kernel=1,
                              **(init_attn if init_attn is not None else init),
                              **filter_kwargs)
            self.proj = Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
                **filter_kwargs)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.conv1(
            nn.functional.dropout(
                x,
                p=self.dropout,
                training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(
                self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            if self.rope_attn:
                q = self.rope(q)
                k = self.rope(k)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.base = base

    def forward(self, x, seq_len=None):
        x = x.transpose(-1, -2)  # (bs, seq_len, dim)
        dim = x.shape[-1]
        if seq_len is None:
            seq_len = x.shape[1]
        inv_freq = (
            1.0 /
            (self.base **
             (torch.arange(0, dim, 2).float().to(x.device) / dim)))
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        phase = torch.outer(t, inv_freq)
        phase = phase.reshape(1, seq_len, -1)
        phase = torch.cat((phase, phase), dim=-1)
        sin, cos = torch.sin(phase), torch.cos(phase)
        # Rotate for relative positional encoding
        x_embed = (x * cos) + (self._rotate_half(x) * sin)
        return x_embed.transpose(-1, -2)

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim=96, emb_dim=None, reduce=False):
        super().__init__()
        assert (dim % 2) == 0
        self.reduce = reduce
        half_emb_dim = dim // 2 if emb_dim is None else emb_dim // 2
        self.map = Linear(dim, half_emb_dim)

    def forward(self, x):
        freqs = 2 * np.pi * F.silu(self.map(x))
        fouriered = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        fouriered = fouriered.mean(1) if self.reduce else fouriered
        return fouriered


# EDM reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion.
# Here we reimplement for time series, adding custom resampling filters.

class UNet(nn.Module):
    def __init__(
        self,
        feat_size,
        # Image resolution at input/output.
        seq_len,
        past_k=2,             # Number of conditioning time steps
        cond_net=False,         # Use mapping for conditional window
        label_dim=0,             # Number of class labels, 0 = unconditional.
        # Base multiplier for the number of channels.
        model_channels=32,
        # Per-resolution multipliers for the number of channels.
        channel_mult=[1, 2, 3, 4, 5],
        # Multiplier for the dimensionality of the embedding vector.
        channel_mult_emb=4,
        num_blocks=2,             # Number of residual blocks per resolution.
        # List of resolutions with self-attention.
        attn_resolutions=[96, 48, 24, 12],
        rope_attn=True,          # Use rotatory positional embedding
        alias_free=True,          # Use alias-free resampling filters
        kaiser_size=6,             # Size of the Kaiser filter, if alias-free
        kaiser_beta=3,             # Beta of the Kaiser filter, if alias-free
        dropout=0.10,          # Dropout probability.
        # Dropout probability of class labels for classifier-free guidance.
        label_dropout=0,
        **kwargs,
    ):
        super().__init__()
        self.cond_net = cond_net
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode='kaiming_uniform',
            init_weight=np.sqrt(
                1 / 3),
            init_bias=np.sqrt(
                1 / 3))
        init_zero = dict(
            init_mode='kaiming_uniform',
            init_weight=0,
            init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero)
        filter_kwargs = dict(
            alias_free=alias_free,
            kaiser_size=kaiser_size,
            kaiser_beta=kaiser_beta)

        # Mapping.
        if kwargs['sigma_from_t']:
            pos_emb_layer = PositionalEmbedding(num_channels=model_channels)
        else:
            pos_emb_layer = LearnedSinusoidalPosEmb(
                dim=seq_len, emb_dim=model_channels, reduce=True)
        self.map_noise = pos_emb_layer
        self.map_layer0 = Linear(
            in_features=model_channels,
            out_features=emb_channels,
            **init)
        self.map_layer1 = Linear(
            in_features=emb_channels,
            out_features=emb_channels,
            **init)
        self.map_label = Linear(
            in_features=label_dim,
            out_features=emb_channels,
            bias=False,
            init_mode='kaiming_normal',
            init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = nn.ModuleDict()
        cout = 2 * feat_size  # Conditioning changes feat_size
        for level, mult in enumerate(channel_mult):
            res = seq_len >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{1}_conv'] = Conv1d(
                    in_channels=cin, out_channels=cout, kernel=3, **init, **filter_kwargs)
            else:
                self.enc[f'{res}x{1}_down'] = UNetBlock1D(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    rope_attn=rope_attn,
                    **block_kwargs,
                    **filter_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{1}_block{idx}'] = UNetBlock1D(
                    in_channels=cin, out_channels=cout, rope_attn=rope_attn, attention=(
                        res in attn_resolutions), **block_kwargs, **filter_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = seq_len >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{1}_in0'] = UNetBlock1D(
                    in_channels=cout,
                    out_channels=cout,
                    rope_attn=rope_attn,
                    attention=True,
                    **block_kwargs,
                    **filter_kwargs)
                self.dec[f'{res}x{1}_in1'] = UNetBlock1D(
                    in_channels=cout,
                    out_channels=cout,
                    rope_attn=rope_attn,
                    **block_kwargs,
                    **filter_kwargs)
            else:
                self.dec[f'{res}x{1}_up'] = UNetBlock1D(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    rope_attn=rope_attn,
                    **block_kwargs,
                    **filter_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{1}_block{idx}'] = UNetBlock1D(
                    in_channels=cin, out_channels=cout, rope_attn=rope_attn, attention=(
                        res in attn_resolutions), **block_kwargs, **filter_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv1d(
            in_channels=cout,
            out_channels=feat_size,
            kernel=3,
            **init_zero,
            **filter_kwargs)

    def forward(self, x, xc, noise_labels):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        emb = F.silu(emb)

        # Concatenate x and xc
        x = torch.cat((x, xc), dim=1)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock1D) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        x = F.silu(self.out_norm(x))
        x = self.out_conv(x)
        return x


class Unsqueeze(nn.Module):
    def __init__(self, dim, repeat=None):
        super().__init__()
        self.dim = dim
        self.repeat = repeat

    def forward(self, x):
        x = x.unsqueeze(self.dim)
        return x.repeat(1, self.repeat, 1)


class ConvLSTMNet(nn.Module):
    def __init__(
        self,
        feat_size,
        seq_len,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feat_size = feat_size
        self.past_k = 2

        self.define_with_cond_mapping()
        self.define_sigma_mapping(kwargs['sigma_from_t'])
        self._define_lstm()

    def define_sigma_mapping(self, sigma_from_t=False):
        if sigma_from_t:
            pos_emb_layer = nn.Sequential(
                PositionalEmbedding(num_channels=self.seq_len),
                Unsqueeze(1, repeat=self.feat_size)
            )
        else:
            pos_emb_layer = LearnedSinusoidalPosEmb(dim=self.seq_len)
        self.map_sigma_pos = pos_emb_layer
        self.map_sigma_layer0 = nn.Conv1d(
            self.feat_size, self.feat_size, kernel_size=3, padding=1)
        self.map_sigma_layer1 = nn.Conv1d(
            self.feat_size, self.feat_size, kernel_size=3, padding=1)
        # Adapt noisy signal
        self.map_shift_scale = nn.Conv1d(
            self.feat_size,
            2 * self.feat_size,
            kernel_size=3,
            padding=1)
        self.map_x_with_cond = nn.Conv1d(
            2 * self.feat_size,
            self.feat_size,
            kernel_size=3,
            padding=1)
        self.norm0 = LayerNorm(dim=2 * self.feat_size)
        self.norm1 = LayerNorm(dim=self.feat_size)

    def define_with_cond_mapping(self):
        self.map_cond_layer0 = Linear(self.seq_len, self.seq_len)
        self.map_cond_layer1 = Linear(self.seq_len, self.seq_len)

    def _define_lstm(self, hidden_dim=64):
        self.pre_lstm = nn.Conv1d(
            self.feat_size,
            self.feat_size,
            kernel_size=3,
            padding=1)
        self.lstm = nn.LSTM(
            self.feat_size,
            hidden_dim,
            batch_first=True,
            bidirectional=True)
        self.map_lstm = Linear(
            hidden_dim,
            self.feat_size,
            init_weight=0,
            init_bias=0)

    def forward(self, x, xc, sigma):
        # Map condition
        xc = F.silu(self.map_cond_layer0(xc))
        xc = F.silu(self.map_cond_layer1(xc))
        x = torch.cat([x, xc], dim=1)
        # Map noise
        emb_sigma = self.map_sigma_pos(sigma)
        emb_sigma = F.silu(self.map_sigma_layer0(emb_sigma))
        emb_sigma = F.silu(self.map_sigma_layer1(emb_sigma))
        # Map noisy signal with condition
        x = self.map_x_with_cond(F.silu(self.norm0(x)))
        params = self.map_shift_scale(emb_sigma)
        scale, shift = params.chunk(chunks=2, dim=1)
        x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        x = self.pre_lstm(F.dropout(x, p=0.1, training=self.training))
        # Bidirectional LSTM
        out, _ = self.lstm(x.transpose(2, 1))
        out = out[..., :out.size(-1) // 2]
        x = self.map_lstm(out).transpose(2, 1)
        return x


class AttnNet(nn.Module):
    def __init__(
        self,
        feat_size,
        seq_len,
        num_heads=1,
        **kwargs
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_size, num_heads=num_heads)

    def forward(self, x, xc, sigma):
        # x: (batch, features, seq_len) => (seq_len, batch, features)
        x = x.permute(2, 0, 1)
        xc = xc.permute(2, 0, 1)
        # Cross-attention: queries=xc, keys=x, values=x
        attn_output, _ = self.attn(xc, x, x)
        x = attn_output.permute(1, 2, 0)
        return x


class LinearNet(nn.Module):
    def __init__(
        self,
        feat_size,
        seq_len,
        **kwargs
    ):
        super().__init__()
        self.linear = Linear(seq_len, seq_len)

    def forward(self, x, xc, sigma):
        x = self.linear(x)
        return x
