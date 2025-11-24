
import torch
import torch.nn as nn
from torch.nn import functional as F


class RNAmodel(nn.Module):
    def __init__(
            self,
            input_channels = 6,
            d_model =128,
            n_heads = 8,
            n_layers = 6,
            dropout = 0.05,
            max_len = 400,
    ):
        super().__init__()
        self.d_model = d_model
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True,# so we can use (B,L,D)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
        )
        self.fc_out = nn.Linear(d_model, 3)


    def forward(self, x, mask):
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
        h  = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B,L, d_model)
        coords = self.fc_out(h)  # (B, L, 3)
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)  # (B, L, 3)
        return coords
    