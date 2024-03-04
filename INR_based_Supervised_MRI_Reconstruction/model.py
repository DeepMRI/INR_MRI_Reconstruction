import torch.nn as nn
import torch
import math
import numpy as np

############################# activation ##################################
def get_act(config):
    """Get activation functions from the config file."""

    if config.nonlinearity == 'elu':
        return nn.ELU()
    elif config.nonlinearity == 'relu':
        return nn.ReLU()
    elif config.nonlinearity == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.nonlinearity == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')

############################# PosEncoding ##################################
class PosEncoding(nn.Module):
    """reference: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"""
    def __init__(self, in_dim, num_frequencies, k, include_coord=False):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * k
        self.register_buffer("B", B)
        self.out_dim = num_frequencies * 2
        self.out_dim = self.out_dim + in_dim if include_coord else self.out_dim
        self.include_coord = include_coord

    def forward(self, x):
        x_proj = torch.matmul(2 * math.pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.include_coord:
            out = torch.cat([x, out], dim=-1)
        return out

############################# MLP ##################################
class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, act, skips=(4,), bias=True):

        super().__init__()
        self.act = act
        # layer with skip connection
        self.skips = skips

        # MLP
        model = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        for i in range(1, num_layer - 1):
            if i in skips:
                model.append(nn.Linear(hidden_dim + in_dim, hidden_dim, bias=bias))
            else:
                model.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.model = nn.ModuleList(model)
        self.out_linear = nn.Linear(hidden_dim, out_dim, bias=bias)

    def forward(self, x):

        h = x
        for i, layer in enumerate(self.model):
            h = self.act(self.model[i](h))
            if i+1 in self.skips:
                h = torch.cat([x, h], dim=-1)
        out = self.out_linear(h)
        return out

############################# ResEncoder ##################################
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.):

    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class ResBlock(nn.Module):

    def __init__(self, hidden_dim, act, embed_dim=None):
        super().__init__()

        self.act = act
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(hidden_dim // 4, 32), num_channels=hidden_dim, eps=1e-6)
        self.Conv_0 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2,))
        if embed_dim is not None:
            self.Dense_0 = nn.Linear(embed_dim, hidden_dim)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(hidden_dim // 4, 32), num_channels=hidden_dim, eps=1e-6)
        self.Conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2,))

    def forward(self, x, scale_embedding=None):

        h = self.Conv_0(self.act(self.GroupNorm_0(x)))
        if scale_embedding is not None:
            h += self.Dense_0(self.act(scale_embedding))[:, :, None, None]
        h = self.Conv_1(self.act(self.GroupNorm_1(h)))

        return x + h

class ResEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, act, block_num=5, embed_dim=None):
        super().__init__()

        model = [nn.ReflectionPad2d(2),
                 nn.Conv2d(in_dim, hidden_dim, (5, 5))]
        for _ in range(block_num):
            model.append(ResBlock(hidden_dim, act, embed_dim))
        model += [nn.ReflectionPad2d(2),
                  nn.Conv2d(hidden_dim, out_dim, (5, 5))]
        self.model = nn.ModuleList(model)
        self.block_num = block_num

    def forward(self, x, scale_embedding=None):

        h = self.model[0](x)
        h = self.model[1](h)
        for i in range(self.block_num):
            h = self.model[2 + i](h, scale_embedding)
        h = self.model[-2](h)
        h = self.model[-1](h)

        return h

############################# inr network ##################################
class network(nn.Module):

    def __init__(self, config, scale_embed=False):
        super().__init__()

        act = get_act(config)

        # positional encoding
        if config.pos_encoding:
            self.pos_encoding = PosEncoding(2,
                                            num_frequencies=config.pos_fre_num,
                                            k=config.pos_scale,
                                            include_coord=config.include_coord)
            in_dim = self.pos_encoding.out_dim
        else:
            self.pos_encoding = None
            in_dim = 2

        # scale embedding
        if scale_embed:
            self.embedding_projection = PosEncoding(in_dim=1,
                                                    num_frequencies=64,
                                                    k=config.pos_scale)
            embed_dim = self.embedding_projection.out_dim
        else:
            embed_dim = None

        # encoder
        if config.use_encoder:
            self.encoder = ResEncoder(in_dim=config.in_dim,
                                      hidden_dim=64,
                                      out_dim=128,
                                      act=act,
                                      embed_dim=embed_dim)
            in_dim += 128
        else:
            self.encoder = None

        # MLP
        self.network = MLP(in_dim=in_dim,
                           hidden_dim=config.hidden_dim,
                           out_dim=config.out_dim,
                           num_layer=config.num_layer,
                           act=act,
                           skips=config.skips)


    def forward(self, coords, prior_intensity=None, scale=None):

        # the scale project to a vector
        if scale is not None:
            scale_embed = self.embedding_projection(scale)
        else:
            scale_embed = None

        # encoding prior_intensity
        if self.encoder is not None:
            prior_intensity = self.encoder(prior_intensity, scale_embed)
            prior_intensity = prior_intensity.permute(0, 2, 3, 1)

        # position encoding
        if self.pos_encoding is not None:
            coords = self.pos_encoding(coords)

        # concat position and prior intensity
        if prior_intensity is not None:
            coords = torch.cat([coords, prior_intensity], dim=-1).type(coords.dtype)

        # MLP
        # TODO: Whether scale embedding is required for MLP
        out = self.network(coords)      # MLP

        return out
