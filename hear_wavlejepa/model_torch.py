"""
PyTorch implementation of WavLeJEPA inference model.

Mirrors the JAX/Equinox model structure to enable checkpoint loading.
Only the inference path is implemented (no training/predictor).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Wav2Vec 2.0 base config with last layer removed for 100Hz output
# Format: (out_channels, kernel_size, stride)
CONV_LAYERS = [
    (512, 10, 5),  # 16kHz -> 3200Hz
    (512, 3, 2),  # -> 1600Hz
    (512, 3, 2),  # -> 800Hz
    (512, 3, 2),  # -> 400Hz
    (512, 3, 2),  # -> 200Hz
    (512, 2, 2),  # -> 100Hz
]


@dataclass
class WavLeJEPAConfigTorch:
    """Configuration for the PyTorch WavLeJEPA model."""

    # Waveform Encoder
    waveform_embed_dim: int = 768
    waveform_num_groups: int = 32

    # Context Encoder
    context_embed_dim: int = 768
    context_num_heads: int = 12
    context_num_layers: int = 12
    context_ffn_dim: int = 3072
    context_dropout: float = 0.0
    context_top_k_layers: int = 8
    context_top_k_norm: str = "instance"

    # Sequence length
    max_seq_len: int = 1000


class ConvBlockTorch(nn.Module):
    """Single convolutional block with GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_groups: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x


class WaveformEncoderTorch(nn.Module):
    """
    Waveform encoder that converts raw audio to dense embeddings.

    Input: Raw waveform at 16kHz, shape (batch, time)
    Output: Embeddings at 100Hz, shape (batch, frames, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_groups: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Build conv blocks
        blocks = []
        in_channels = 1  # Raw audio is mono
        for out_channels, kernel_size, stride in CONV_LAYERS:
            block = ConvBlockTorch(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                num_groups=num_groups,
            )
            blocks.append(block)
            in_channels = out_channels

        self.conv_blocks = nn.ModuleList(blocks)

        # Project from conv output (512) to embed_dim (768)
        self.proj = nn.Linear(
            in_features=CONV_LAYERS[-1][0],  # 512
            out_features=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw waveform to embeddings.

        Args:
            x: Raw waveform, shape (batch, time) at 16kHz

        Returns:
            Embeddings, shape (batch, frames, embed_dim) at 100Hz
        """
        # Handle unbatched input
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Reshape to (batch, 1, time) for conv1d
        x = x.unsqueeze(1)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Transpose to (batch, frames, 512) for linear projection
        x = x.transpose(1, 2)

        # Project to embed_dim
        x = self.proj(x)  # (batch, frames, embed_dim)

        if squeeze_batch:
            x = x.squeeze(0)

        return x

    @property
    def total_stride(self) -> int:
        """Total downsampling factor (160 for 100Hz at 16kHz)."""
        stride = 1
        for _, _, s in CONV_LAYERS:
            stride *= s
        return stride

    @property
    def receptive_field(self) -> int:
        """Receptive field in samples."""
        rf = CONV_LAYERS[0][1]  # First kernel size
        cumulative_stride = CONV_LAYERS[0][2]
        for _, kernel_size, stride in CONV_LAYERS[1:]:
            rf += (kernel_size - 1) * cumulative_stride
            cumulative_stride *= stride
        return rf

    def output_length(self, input_length: int) -> int:
        """Compute output length for a given input length."""
        length = input_length
        for _, kernel_size, stride in CONV_LAYERS:
            length = (length - kernel_size) // stride + 1
        return length


class SinusoidalPositionalEncodingTorch(nn.Module):
    """Fixed sinusoidal positional encodings."""

    pe: torch.Tensor  # Type annotation for buffer

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim

        # Precompute positional encodings
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )  # [embed_dim/2]

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (seq_len, embed_dim) or (batch, seq_len, embed_dim)
        """
        if x.ndim == 2:
            seq_len = x.shape[0]
            return x + self.pe[:seq_len]
        else:
            seq_len = x.shape[1]
            return x + self.pe[:seq_len]


class TransformerEncoderLayerTorch(nn.Module):
    """Standard transformer encoder layer with Pre-LN architecture.

    Matches the JAX/Equinox structure with explicit Q/K/V projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate Q, K, V projections to match Equinox MultiheadAttention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # MLP: Linear -> GELU -> Linear (matching eqx.nn.MLP with depth=1)
        self.mlp_linear1 = nn.Linear(embed_dim, ffn_dim)
        self.mlp_linear2 = nn.Linear(ffn_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [seq_len, embed_dim]
            mask: Attention mask [seq_len, seq_len]. True = attend, False = mask out.

        Returns:
            Output embeddings [seq_len, embed_dim]
        """
        # Pre-LN Self-Attention
        residual = x
        x = self.norm1(x)

        # Compute Q, K, V
        seq_len = x.shape[0]
        q = self.query_proj(x)  # [seq_len, embed_dim]
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape for multi-head attention: [seq_len, num_heads, head_dim]
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_heads, self.head_dim)
        v = v.view(seq_len, self.num_heads, self.head_dim)

        # Compute attention scores: [num_heads, seq_len, seq_len]
        scale = self.head_dim**-0.5
        # q: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [num_heads, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            # mask: [seq_len, seq_len], True = attend
            # Convert to attention mask format (add -inf where False)
            attn_mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [num_heads, seq_len, head_dim]

        # Reshape back: [num_heads, seq_len, head_dim] -> [seq_len, embed_dim]
        attn_output = attn_output.permute(1, 0, 2).reshape(seq_len, self.embed_dim)

        # Output projection
        x = self.output_proj(attn_output)
        x = self.dropout1(x)
        x = residual + x

        # Pre-LN MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp_linear1(x)
        x = F.gelu(x)
        x = self.mlp_linear2(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class ContextEncoderTorch(nn.Module):
    """Transformer encoder for processing context blocks."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        max_seq_len: int = 1000,
        top_k_layers: int = 8,
        top_k_norm: str = "instance",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.top_k_layers = top_k_layers
        self.top_k_norm = top_k_norm

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncodingTorch(
            embed_dim=embed_dim,
            max_len=max_seq_len,
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerTorch(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: Input embeddings [seq_len, embed_dim]
            context_mask: Boolean mask [seq_len] indicating context positions.
            return_all_layers: If True, return outputs from all layers

        Returns:
            If return_all_layers:
                (output, layer_outputs): Final output and list of all layer outputs
            Else:
                output: Final output embeddings [seq_len, embed_dim]
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Build attention mask from context mask
        attn_mask = None
        if context_mask is not None:
            seq_len = x.shape[0]
            attn_mask = context_mask.unsqueeze(0).expand(seq_len, seq_len)

        # Process through transformer layers
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, mask=attn_mask)
            layer_outputs.append(x)

        # Final normalization
        x = self.final_norm(x)

        if return_all_layers:
            return x, layer_outputs
        return x

    def forward_with_top_k(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Top-K layer averaging.

        Returns the average of instance-normalized outputs from the top K layers.
        """
        result = self.forward(x, context_mask=context_mask, return_all_layers=True)
        assert isinstance(result, tuple)
        _, layer_outputs = result

        # Get top K layers
        k = self.top_k_layers
        top_k_outputs = layer_outputs[-k:]

        if self.top_k_norm == "instance":
            # Normalize each timestep across embedding dim
            def instance_norm(z: torch.Tensor) -> torch.Tensor:
                mean = z.mean(dim=-1, keepdim=True)
                var = z.var(dim=-1, keepdim=True, unbiased=False)
                return (z - mean) / torch.sqrt(var + 1e-6)

            normalized_outputs = [instance_norm(out) for out in top_k_outputs]
        elif self.top_k_norm == "layer":
            normalized_outputs = [self.final_norm(out) for out in top_k_outputs]
        else:
            normalized_outputs = top_k_outputs

        # Average across layers
        stacked = torch.stack(normalized_outputs, dim=0)  # [K, seq_len, embed_dim]
        averaged = stacked.mean(dim=0)  # [seq_len, embed_dim]

        return averaged


class WavLeJEPATorch(nn.Module):
    """PyTorch implementation of WavLeJEPA for inference.

    HEAR attributes:
    - sample_rate: 16000
    - timestamp_embedding_size: context_embed_dim (768)
    - scene_embedding_size: context_embed_dim (768)
    """

    def __init__(self, config: WavLeJEPAConfigTorch):
        super().__init__()
        self.config = config

        # HEAR-required attributes
        self.sample_rate = 16000
        self.timestamp_embedding_size = config.context_embed_dim
        self.scene_embedding_size = config.context_embed_dim

        self.waveform_encoder = WaveformEncoderTorch(
            embed_dim=config.waveform_embed_dim,
            num_groups=config.waveform_num_groups,
        )

        self.context_encoder = ContextEncoderTorch(
            embed_dim=config.context_embed_dim,
            num_heads=config.context_num_heads,
            num_layers=config.context_num_layers,
            ffn_dim=config.context_ffn_dim,
            dropout=config.context_dropout,
            max_seq_len=config.max_seq_len,
            top_k_layers=config.context_top_k_layers,
            top_k_norm=config.context_top_k_norm,
        )

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features for downstream tasks.

        Uses Top-K layer averaging on full sequence (no masking).

        Args:
            waveform: Raw audio waveform [T] or [B, T] at 16kHz

        Returns:
            Features [frames, embed_dim] or [B, frames, embed_dim]
        """
        # Handle batched input
        if waveform.ndim == 1:
            features = self.waveform_encoder(waveform)
            output = self.context_encoder.forward_with_top_k(features, context_mask=None)
            return output
        else:
            # Batched: process each sample
            batch_outputs = []
            for i in range(waveform.shape[0]):
                features = self.waveform_encoder(waveform[i])
                output = self.context_encoder.forward_with_top_k(features, context_mask=None)
                batch_outputs.append(output)
            # Pad to same length and stack
            max_len = max(o.shape[0] for o in batch_outputs)
            padded = []
            for o in batch_outputs:
                if o.shape[0] < max_len:
                    pad = torch.zeros(
                        max_len - o.shape[0], o.shape[1], device=o.device, dtype=o.dtype
                    )
                    o = torch.cat([o, pad], dim=0)
                padded.append(o)
            return torch.stack(padded, dim=0)

    @property
    def total_stride(self) -> int:
        """Total downsampling factor (160 for 100Hz at 16kHz)."""
        return self.waveform_encoder.total_stride

    @property
    def receptive_field(self) -> int:
        """Receptive field in samples."""
        return self.waveform_encoder.receptive_field
