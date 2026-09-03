import pdb
import torch
import torch.nn as nn
import re
from .projector import FlattenProjector


class AdpPost(nn.Module):
    """Learned dim-reduce AFTER pack (order A): Linear(in -> D') on the packed token, then
    GELU + Linear(D' -> hidden). in = native * r^3. Full spatial x channel mixing."""
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.reduce = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(out_dim, hidden)

    def forward(self, x):  # x [B,T,H,W,in_dim] (grid) or [..., in_dim]
        if x.ndim == 5:
            x = x.flatten(1, 3)  # [B, T*H*W, in_dim] — same flatten as FlattenProjector
        return self.proj(self.act(self.reduce(x)))


class AdpPre(nn.Module):
    """Learned dim-reduce BEFORE pack (order B), realized on the packed vector by a SHARED
    per-block Linear(native -> d), then concat blocks -> D' = d*blocks, GELU, Linear->hidden.
    Packed dim is channel-major (c outer, block inner): reshape [.,native,blocks] -> transpose
    -> [.,blocks,native]. Channel-aligned reduction across spatial positions."""
    def __init__(self, native, blocks, d, hidden):
        super().__init__()
        self.native, self.blocks, self.d = native, blocks, d
        self.reduce = nn.Linear(native, d)        # shared across all blocks
        self.act = nn.GELU()
        self.proj = nn.Linear(d * blocks, hidden)

    def forward(self, x):  # x [B,T,H,W,native*blocks] (grid) or [..., native*blocks]
        if x.ndim == 5:
            x = x.flatten(1, 3)  # [B, T*H*W, native*blocks] — same flatten as FlattenProjector
        lead = x.shape[:-1]
        x = x.reshape(*lead, self.native, self.blocks).transpose(-1, -2)  # [...,blocks,native]
        x = self.reduce(x)                                                # [...,blocks,d]
        x = x.reshape(*lead, self.blocks * self.d)                        # [...,D']
        return self.proj(self.act(x))


class ResamplerProjector(nn.Module):
    """Perceiver-style cross-attention resampler: `num_latents` LEARNED queries cross-attend the
    full encoder grid -> num_latents output tokens at LLM hidden dim. Strictly generalizes
    avgpool/pack/adp (all = fixed-attention specials); the resampler = learned attention.
    Reduces token count INSIDE the projector (vs pack which reduces in the dataset); the LLaVA
    image placeholder expands to whatever count the projector emits, so this is safe.

    mode:
      'pure'   = fully learned attention (no external importance signal).
      'l2norm' = add per-token L2-norm (z-scored per sample) as an additive importance bias on
                 the attention logits (background air = low norm -> down-weighted). Mapless.
    in_dim   = projector input per-token dim (768 raw COLIPRI / 512 CT-CLIP; pack2 -> *8).
    latent_dim = bottleneck width D' (the count-axis budget, e.g. 1024).
    hidden   = LLM hidden size (4096 Llama-3.1-8B).
    """
    def __init__(self, in_dim, latent_dim, hidden, num_latents=216, depth=2, num_heads=8, mode="pure"):
        super().__init__()
        self.num_latents = int(num_latents)
        self.num_heads = int(num_heads)
        self.mode = mode
        self.latents = nn.Parameter(torch.randn(self.num_latents, latent_dim) * 0.02)
        self.kv = nn.Linear(in_dim, latent_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([
                nn.LayerNorm(latent_dim),
                nn.LayerNorm(latent_dim),
                nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                nn.LayerNorm(latent_dim),
                nn.Sequential(nn.Linear(latent_dim, latent_dim * 4), nn.GELU(),
                              nn.Linear(latent_dim * 4, latent_dim)),
            ]) for _ in range(depth)
        )
        self.out = nn.Linear(latent_dim, hidden)
        if mode == "l2norm":
            self.bias_scale = nn.Parameter(torch.tensor(0.0))  # start neutral (==pure) -> stable; learns importance weight

    def forward(self, x):  # x [B,t,h,w,in_dim] (grid) or [B,T,in_dim]
        if x.ndim == 5:
            x = x.flatten(1, 3)                               # [B, T, in_dim]
        B, T, _ = x.shape
        attn_mask = None
        if self.mode == "l2norm":
            n = x.float().norm(dim=-1)                        # [B, T] per-token L2 norm
            n = (n - n.mean(dim=1, keepdim=True)) / (n.std(dim=1, keepdim=True) + 1e-6)
            bias = (self.bias_scale * n)[:, None, :].expand(B, self.num_latents, T)  # [B,N,T]
            attn_mask = bias.repeat_interleave(self.num_heads, dim=0).to(x.dtype)    # [B*heads,N,T]
        kv = self.kv(x)                                       # [B,T,latent]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)       # [B,N,latent]
        for ln_q, ln_kv, attn, ln_ff, ff in self.layers:
            kvn = ln_kv(kv)
            a, _ = attn(ln_q(q), kvn, kvn, attn_mask=attn_mask, need_weights=False)
            q = q + a
            q = q + ff(ln_ff(q))
        return self.out(q)                                    # [B, num_latents, hidden]


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type.startswith('attn_pool'):
        mlp_projector = projector_type.split('+')[1]
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mlp_projector)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            projector = nn.Sequential(*modules)
        else:
            projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        # embed_dim / context_dim were arguments to the disabled attentional pooler; the projector
        # is the whole module. See projector.py.
        return FlattenProjector(projector)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type.startswith('resampler'):
        # resampler / resampler_pure / resampler_l2norm
        mode = projector_type.split('_', 1)[1] if '_' in projector_type else 'pure'
        return ResamplerProjector(config.mm_hidden_size, config.mm_adp_out, config.hidden_size,
                                  num_latents=getattr(config, 'mm_resampler_latents', 216),
                                  mode=mode)

    if projector_type == 'adp_post':
        return AdpPost(config.mm_hidden_size, config.mm_adp_out, config.hidden_size)

    if projector_type == 'adp_pre':
        blocks = config.mm_hidden_size // config.mm_adp_native  # r^3
        d = config.mm_adp_out // blocks
        return AdpPre(config.mm_adp_native, blocks, d, config.hidden_size)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')