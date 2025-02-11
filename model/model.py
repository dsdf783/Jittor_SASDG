from typing import Any, Callable, List, Optional, Union

import jittor
import jittor as jt
jt.flags.use_cuda=1
import numpy as np
from jittor import nn
from jittor import attention
from model.blocks import LinearBlock
from jittor import einops 
from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    # print("adain_params.shape:", adain_params.shape) # adain_params.shape: torch.Size([128, 1024])
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[: , : m.num_features]
            std = adain_params[: , m.num_features: 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[: , 2 * m.num_features:]


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def execute(self, position):
        pos_encoding = self.block(position)
        pos_encoding = einops.rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[jt.Var], jt.Var]] = nn.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = attention.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def execute(
        self,
        src: jt.Var,
        src_mask: Optional[jt.Var] = None,
        src_key_padding_mask: Optional[jt.Var] = None,
    ) -> jt.Var:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    def _sa_block(self, x:jt.Var, attn_mask:Optional[jt.Var], key_padding_mask:Optional[jt.Var]
    ) -> jt.Var:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: jt.Var) -> jt.Var:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class MLP(nn.Module):
    def __init__(self, out_dim):
        super(MLP, self).__init__()
        dims = [512, 640, 896]
        n_blk = len(dims)
        norm = 'none'
        acti = 'lrelu'

        layers = []
        for i in range(n_blk - 1):
            layers += LinearBlock(dims[i], dims[i + 1], norm=norm, acti=acti)
        layers += LinearBlock(dims[-1], out_dim,
                                   norm='none', acti='none')
        self.model = nn.Sequential(*layers)

    def execute(self, x):
        x = jt.array(x)
        return self.model(x.view(x.size(0), -1)) 


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', jt.zeros(num_features))
        self.register_buffer('running_var', jt.ones(num_features))

    def execute(self, x, cond, t):

        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        x = x.permute(0, 2, 1)
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = nn.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            False, self.momentum, self.eps) 
        out = out.view(b, c, *x.size()[2:])
        out = out.permute(0, 2, 1)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = attention.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = attention.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def execute(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    # def forward(self, x, cond, t):
    def execute(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[jt.Var], jt.Var] = nn.gelu,
        use_rotary=True,
        **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        self.rotary = None
        self.abs_pos_encoding = nn.Identity()

        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim + 16, dropout, batch_first=True
            )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, (latent_dim + 16)),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, (latent_dim + 16) * 2),  # 2 time tokens
            lambda x: einops.rearrange(x, 'b (r d) -> b r d', r=2)
        )

        self.motionclip_features_mlp = MLP(out_dim = 2 * (512 + 16)) 
        self.null_cond_embed = jt.array(jt.randn(1, seq_len, latent_dim))
        self.null_cond_embed.requires_grad = True
        self.null_cond_hidden =jt.array(jt.randn(1, latent_dim + 16))
        self.null_cond_hidden.requires_grad = True

        self.norm_cond = nn.LayerNorm(latent_dim + 16)

        self.input_projection = nn.Linear(nfeats, latent_dim + 16)
        self.cond_encoder = nn.Sequential()


        beta_dim = 16
        self.beta_projection = nn.Linear(beta_dim, latent_dim)

        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim + 16),
            # nn.silu(),
            lambda x: nn.silu(x), 
            nn.Linear(latent_dim + 16, latent_dim + 16),
        )

        decoderstack = nn.ModuleList([])


        norm_dim = 512 + 16

        decoderstack.append(AdaptiveInstanceNorm1d(norm_dim))

        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim + 16,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim + 16, output_feats)

    def guided_forward(self, x, cond_embed, motionclip_features , beta ,times, guidance_weight):
        unc = self.execute(x, cond_embed, motionclip_features, beta, times, cond_drop_prob=1)
        conditioned = self.execute(x, cond_embed, motionclip_features, beta, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def execute(
        self, x: jt.Var,cond_embed: jt.Var,motionclip_features: jt.Var, beta: jt.Var, times: jt.Var, cond_drop_prob: float = 0.0
    ):
        adain_params = self.motionclip_features_mlp(motionclip_features)

        assign_adain_params(adain_params, self.seqTransDecoder)
        batch_size = x.shape[0]
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob)
        keep_mask_embed = einops.rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = einops.rearrange(keep_mask, "b -> b 1")


        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.astype(cond_tokens.dtype)
        cond_tokens = jt.where(keep_mask_embed, cond_tokens, null_cond_embed)


        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)
        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.astype(t.dtype)  
        cond_hidden = jt.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        cond_tokens = jt.concat((cond_tokens, beta), dim=-1)
        c = jt.concat((cond_tokens, t_tokens), dim=-2)
        c = c.to(jt.float32)
        cond_tokens = self.norm_cond(c)
        
        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)
        output = self.final_layer(output)
        return output