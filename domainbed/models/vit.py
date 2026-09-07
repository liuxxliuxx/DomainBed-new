"""Torchvision ViT with explicit spatial insertion points for DomainBed.

No global attention/backend flags are changed. The local attention implementation
uses ordinary differentiable matmuls so meta/gradient algorithms can differentiate
through it repeatedly. Pretrained parameters are loaded before any wrapping.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


# One-based block numbers in the public mapping; zero-based indices internally.
STAGE_BLOCKS = {
    "layer1": (0, 1, 2),
    "layer2": (3, 4, 5),
    "layer3": (6, 7, 8),
    "layer4": (9, 10),
}
STAGE_ENDS = {name: blocks[-1] for name, blocks in STAGE_BLOCKS.items()}


class MathSelfAttention(nn.Module):
    """ViT's batch-first MHA, retaining the original parameter names/objects."""

    def __init__(self, original):
        super().__init__()
        if not original.batch_first or not original._qkv_same_embed_dim:
            raise ValueError("ViT requires batch-first, equal-dimension Q/K/V")
        if original.bias_k is not None or original.bias_v is not None or original.add_zero_attn:
            raise ValueError("Extra attention tokens are not supported")
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.dropout = original.dropout
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self.in_proj_weight = original.in_proj_weight
        self.in_proj_bias = original.in_proj_bias
        self.out_proj = original.out_proj
        self.train(original.training)

    def project_qkv(self, x):
        return F.linear(x, self.in_proj_weight, self.in_proj_bias)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        if query is not key or key is not value or query.ndim != 3:
            raise ValueError("This ViT attention supports batch-first self-attention only")
        batch, length, channels = query.shape
        q, k, v = self.project_qkv(query).chunk(3, dim=-1)
        q, k, v = [item.reshape(batch, length, self.num_heads, self.head_dim)
                   .transpose(1, 2) for item in (q, k, v)]
        scores = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
        if is_causal:
            causal = torch.ones(length, length, device=query.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.reshape(batch, self.num_heads, length, length)
            scores = (scores.masked_fill(attn_mask, float("-inf"))
                      if attn_mask.dtype == torch.bool else scores + attn_mask)
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]
            scores = (scores.masked_fill(mask, float("-inf"))
                      if mask.dtype == torch.bool else scores + mask)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = (weights @ v).transpose(1, 2).reshape(batch, length, channels)
        output = self.out_proj(output)
        if not need_weights:
            return output, None
        return output, weights.mean(dim=1) if average_attn_weights else weights


class TokenGridAdapter(nn.Module):
    """Apply an existing BCHW module to patch tokens; leave prefix tokens alone.

The wrapped module owns its train/eval behavior. In particular AWWSL and active
FQ/codebook modules must not be turned into identities by this adapter at eval.
"""

    def __init__(self, op, grid_size, num_prefix_tokens=1):
        super().__init__()
        self.op = op
        self.grid_size = tuple(grid_size)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected BNC tokens, got {tuple(x.shape)}")
        prefix = x[:, :self.num_prefix_tokens]
        patches = x[:, self.num_prefix_tokens:]
        batch, count, channels = patches.shape
        height, width = self.grid_size
        if count != height * width:
            raise ValueError(f"{count} patch tokens cannot form {height}x{width}")
        features = patches.transpose(1, 2).reshape(batch, channels, height, width)
        changed = self.op(features)
        if changed.shape != features.shape:
            raise ValueError("A token-grid operation must preserve the BCHW shape")
        return torch.cat((prefix, changed.flatten(2).transpose(1, 2)), dim=1)


class ViT(nn.Module):
    is_vit_backbone = True

    def __init__(self, input_shape, hparams):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        from torchvision.models.vision_transformer import interpolate_embeddings

        if (len(input_shape) != 3 or input_shape[0] < 1
                or input_shape[1] != input_shape[2] or input_shape[1] < 16
                or input_shape[1] % 16):
            raise ValueError(
                f"ViT-B/16 requires a positive channel count and square size divisible by 16; got {input_shape}")
        channels, size, _ = input_shape
        pretrained = hparams.get("pretrained", True)
        if isinstance(pretrained, str):
            pretrained = pretrained.lower() not in ("false", "0", "no", "off")
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is None:
            network = vit_b_16(weights=None, image_size=size)
        else:
            network = vit_b_16(weights=weights)
            if size != network.image_size:
                state = interpolate_embeddings(
                    image_size=size, patch_size=network.patch_size,
                    model_state=network.state_dict())
                network = vit_b_16(weights=None, image_size=size)
                network.load_state_dict(state, strict=True)

        if channels != 3:
            old = network.conv_proj
            conv = nn.Conv2d(
                channels, old.out_channels, old.kernel_size, stride=old.stride,
                padding=old.padding, dilation=old.dilation, bias=old.bias is not None)
            with torch.no_grad():
                for index in range(channels):
                    conv.weight[:, index].copy_(old.weight[:, index % 3])
                if old.bias is not None:
                    conv.bias.copy_(old.bias)
            network.conv_proj = conv

        for block in network.encoder.layers:
            block.self_attention = MathSelfAttention(block.self_attention)
        self.n_outputs = network.hidden_dim
        network.heads = nn.Identity()
        self.network = network
        self.grid_size = (size // network.patch_size,) * 2
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self._insertions = set()

    def forward(self, x):
        return self.dropout(self.network(x))

    def _claim(self, position, tag):
        key = (position, tag)
        if key in self._insertions:
            raise ValueError(f"ViT {position} already has {tag}")
        self._insertions.add(key)

    def add_block_op(self, index, op, tag):
        # Reserve the last block to mix perturbed patches into the CLS token.
        if not 0 <= index < len(self.network.encoder.layers) - 1:
            raise ValueError(f"Invalid ViT perturbation block index: {index}")
        self._claim(f"block{index + 1}", tag)
        layers = self.network.encoder.layers
        layers[index] = nn.Sequential(
            layers[index], TokenGridAdapter(op, self.grid_size))

    def add_stage_op(self, position, op, tag):
        if position in STAGE_ENDS:
            self.add_block_op(STAGE_ENDS[position], op, tag)
        elif position == "conv1":
            self._claim(position, tag)
            self.network.conv_proj = nn.Sequential(self.network.conv_proj, op)
        elif position == "maxpool":
            self._claim(position, tag)
            encoder = self.network.encoder
            encoder.dropout = nn.Sequential(
                encoder.dropout, TokenGridAdapter(op, self.grid_size))
        else:
            raise ValueError(f"Unknown ViT insertion position: {position}")

    def block_indices(self, stage):
        if stage not in STAGE_BLOCKS:
            raise ValueError(f"Unknown ViT block group: {stage}")
        return STAGE_BLOCKS[stage]
