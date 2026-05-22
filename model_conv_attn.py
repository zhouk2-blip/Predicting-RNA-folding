
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionBiasEncoderLayer(nn.Module):
    """Purpose: Transformer encoder layer that accepts pairwise attention bias.

    Input:
        d_model: Hidden feature size.
        n_heads: Number of attention heads.
        dim_feedforward: Feed-forward hidden size.
        dropout: Dropout probability.
    Output:
        Module that maps residue embeddings to updated residue embeddings.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        """Purpose: Initialize one biased self-attention encoder layer.

        Input:
            d_model: Hidden feature size.
            n_heads: Number of attention heads.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.n_heads = n_heads

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        """Purpose: Apply one self-attention block with optional SPOT bias.

        Input:
            src: Residue embeddings shaped (B, L, d_model).
            src_key_padding_mask: Boolean padding mask shaped (B, L).
            attn_bias: Optional pairwise bias shaped (B, L, L).
        Output:
            Updated residue embeddings shaped (B, L, d_model).
        """
        attn_mask = None
        key_padding_mask = src_key_padding_mask
        if attn_bias is not None:
            if src_key_padding_mask is not None:
                attn_bias = attn_bias.masked_fill(src_key_padding_mask.unsqueeze(1), float("-inf"))
                key_padding_mask = None
            # MultiheadAttention expects a per-head mask shaped (B * H, L, L).
            attn_mask = attn_bias.repeat_interleave(self.n_heads, dim=0)

        attn_out, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


class AttentionBiasEncoder(nn.Module):
    """Purpose: Stack multiple attention-bias encoder layers.

    Input:
        d_model: Hidden feature size.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dropout: Dropout probability.
    Output:
        Module that repeatedly updates residue embeddings.
    """

    def __init__(self, d_model, n_heads, n_layers, dropout):
        """Purpose: Initialize a stack of attention-bias encoder layers.

        Input:
            d_model: Hidden feature size.
            n_heads: Number of attention heads.
            n_layers: Number of layers to stack.
            dropout: Dropout probability.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionBiasEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        """Purpose: Run all encoder layers using shared masks and bias.

        Input:
            src: Residue embeddings shaped (B, L, d_model).
            src_key_padding_mask: Boolean padding mask shaped (B, L).
            attn_bias: Optional pairwise bias shaped (B, L, L).
        Output:
            Final residue embeddings shaped (B, L, d_model).
        """
        for layer in self.layers:
            src = layer(
                src,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
            )
        return src


class RNAmodel(nn.Module):
    """Purpose: Predict normalized 3D RNA coordinates from sequence features.

    Input:
        input_channels: Number of per-residue input channels.
        d_model: Hidden feature size.
        n_heads: Number of transformer attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
        max_len: Maximum supported sequence length for positional embeddings.
        spot_bias_scale: Multiplier for SPOT contact-map attention bias.
    Output:
        Module that maps (B, C, L) features to (B, L, 3) coordinates.
    """

    def __init__(
            self,
            input_channels = 8,
            d_model =128,
            n_heads = 8,
            n_layers = 6,
            dropout = 0.05,
            max_len = 400,
            spot_bias_scale = 1.0,
    ):
        """Purpose: Initialize convolutional, positional, attention, and output layers.

        Input:
            input_channels: Number of input feature channels.
            d_model: Hidden feature size.
            n_heads: Number of transformer attention heads.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
            max_len: Maximum supported sequence length.
            spot_bias_scale: Multiplier for SPOT contact-map attention bias.
        Output:
            Initialized PyTorch model.
        """
        super().__init__()
        self.d_model = d_model
        self.spot_bias_scale = spot_bias_scale
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=9, padding=4),# large structural motif
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2), # middle motif
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), # small motif
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.transformer = AttentionBiasEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, 3)


    def forward(self, x, mask, contact_map=None):
        """Purpose: Run convolutional encoding, biased attention, and coordinate output.

        Input:
            x: Residue features shaped (B, C, L).
            mask: Valid-residue mask shaped (B, L).
            contact_map: Optional SPOT contact map shaped (B, L, L).
        Output:
            Predicted normalized coordinates shaped (B, L, 3).
        """
        # x: (B, C, L)
        B,C,L = x.shape 
        h = self.conv_block(x)  # (B, d_model, L)
        h = h.permute(0,2,1)  # (B, L, d_model)
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B,L) 
        pos_emb = self.pos_embed(positions)  # (B, L, d_model)
        h = h + pos_emb
        if mask is not None:
            key_padding_mask = (mask == 0)  # (B, L) boolean
        else:
            key_padding_mask = None

        attn_bias = None
        if contact_map is not None and self.spot_bias_scale != 0:
            attn_bias = contact_map[:, :L, :L].to(dtype=h.dtype) * self.spot_bias_scale

        h  = self.transformer(
            h,
            src_key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
        )  # (B,L, d_model)
        coords = self.fc_out(h)  # (B, L, 3)
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)  # (B, L, 3)
        return coords
    
