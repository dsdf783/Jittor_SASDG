import math
import jittor
import jittor as jt
from jittor import nn
import numpy as np
from math import pi


# absolute positional embedding used for vanilla transformer sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = jt.zeros(max_len, d_model)
        position = jt.arange(0, max_len).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe=jt.array(pe)
        self.pe.stop_grad()

    def execute(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def execute(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jt.exp(jt.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jt.concat((emb.sin(), emb.cos()), dim=-1)
        return emb


# dropout mask
def prob_mask_like(shape, prob):
    if prob == 1:
        return jt.ones(shape, dtype=jt.bool)
    elif prob == 0:
        return jt.zeros(shape, dtype=jt.bool)
    else:
        return jt.zeros(shape).float().uniform_(0, 1) < prob



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = jt.gather(a, -1, jt.array(t))
    return jt.reshape(out,b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            jt.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=jt.float32
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            jt.arange(n_timestep + 1, dtype=jt.float32) / n_timestep + cosine_s 
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = jt.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)


    elif schedule == "sqrt_linear":
        betas = jt.linspace(
            linear_start, linear_end, n_timestep, dtype=jt.float32
        )
    elif schedule == "sqrt":
        betas = (
            jt.linspace(linear_start, linear_end, n_timestep, dtype=jt.float32)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    
    return jt.array(betas)