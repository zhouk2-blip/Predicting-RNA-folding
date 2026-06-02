
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


class ResidualGraphMessageLayer(nn.Module):
    """Purpose: Pass residue features along backbone, local, and SPOT graph edges.

    Input:
        d_model: Hidden feature size.
        dropout: Dropout probability applied to graph messages.
    Output:
        Module that returns one residual graph-updated feature tensor.
    """

    def __init__(self, d_model, dropout):
        """Purpose: Initialize edge-type projections and residual normalization.

        Input:
            d_model: Hidden feature size.
            dropout: Dropout probability.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.backbone_proj = nn.Linear(d_model, d_model)
        self.local_proj = nn.Linear(d_model, d_model)
        self.spot_proj = nn.Linear(d_model, d_model)
        self.message_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def aggregate(self, h, adjacency, projection):
        """Purpose: Aggregate projected neighbor features with normalized edges.

        Input:
            h: Residue embeddings shaped (B, L, d_model).
            adjacency: Edge weights shaped (B, L, L), destination by source.
            projection: Linear layer applied to source-node features.
        Output:
            Aggregated message tensor shaped (B, L, d_model).
        """
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalized = adjacency / degree
        return torch.bmm(normalized, projection(h))

    def forward(self, h, backbone_adj, local_adj, spot_adj, graph_scale):
        """Purpose: Apply one residual graph message-passing step.

        Input:
            h: Residue embeddings shaped (B, L, d_model).
            backbone_adj: Backbone edge weights shaped (B, L, L).
            local_adj: Local sequence edge weights shaped (B, L, L).
            spot_adj: SPOT contact edge weights shaped (B, L, L).
            graph_scale: Scalar residual multiplier for graph messages.
        Output:
            Updated residue embeddings shaped (B, L, d_model).
        """
        message = (
            self.aggregate(h, backbone_adj, self.backbone_proj)
            + self.aggregate(h, local_adj, self.local_proj)
            + self.aggregate(h, spot_adj, self.spot_proj)
        )
        message = self.message_proj(F.relu(message))
        return self.norm(h + graph_scale * self.dropout(message))


class ResidualGraphMessagePassing(nn.Module):
    """Purpose: Stack residual graph message-passing layers for RNA features.

    Input:
        d_model: Hidden feature size.
        num_layers: Number of graph message-passing layers.
        dropout: Dropout probability.
        graph_scale: Residual graph-message scale.
        spot_edge_threshold: Minimum SPOT contact probability to create an edge.
        spot_top_k: Maximum high-confidence SPOT neighbors per residue.
        local_edge_max_sep: Maximum sequence separation for local graph edges.
    Output:
        Module that maps residue embeddings to graph-enhanced embeddings.
    """

    def __init__(
            self,
            d_model,
            num_layers,
            dropout,
            graph_scale=0.10,
            spot_edge_threshold=0.50,
            spot_top_k=8,
            local_edge_max_sep=4,
    ):
        """Purpose: Initialize graph layers and edge-construction settings.

        Input:
            d_model: Hidden feature size.
            num_layers: Number of graph message-passing layers.
            dropout: Dropout probability.
            graph_scale: Residual graph-message scale.
            spot_edge_threshold: Minimum SPOT contact probability for SPOT edges.
            spot_top_k: Maximum SPOT neighbors per residue.
            local_edge_max_sep: Maximum sequence separation for local edges.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualGraphMessageLayer(d_model=d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.graph_scale = graph_scale
        self.spot_edge_threshold = spot_edge_threshold
        self.spot_top_k = spot_top_k
        self.local_edge_max_sep = local_edge_max_sep

    def sequence_adjacency(self, L, mask, min_sep, max_sep, dtype, device):
        """Purpose: Build a batched sequence-separation adjacency matrix.

        Input:
            L: Sequence length in the current batch.
            mask: Valid-residue mask shaped (B, L), or None.
            min_sep: Minimum sequence separation for edges.
            max_sep: Maximum sequence separation for edges.
            dtype: Floating dtype for adjacency values.
            device: Torch device.
        Output:
            Adjacency tensor shaped (B, L, L).
        """
        B = mask.shape[0] if mask is not None else 1
        idx = torch.arange(L, device=device)
        sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        adjacency = ((sep >= min_sep) & (sep <= max_sep)).to(dtype=dtype).unsqueeze(0)
        adjacency = adjacency.expand(B, L, L).clone()
        if mask is not None:
            valid_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
            adjacency = adjacency * valid_pair.to(dtype=dtype)
        return adjacency

    def spot_adjacency(self, contact_map, mask, L, dtype, device):
        """Purpose: Build sparse SPOT contact adjacency from probabilities.

        Input:
            contact_map: SPOT contact map shaped (B, L, L), or None.
            mask: Valid-residue mask shaped (B, L), or None.
            L: Sequence length in the current batch.
            dtype: Floating dtype for adjacency values.
            device: Torch device.
        Output:
            Adjacency tensor shaped (B, L, L).
        """
        B = mask.shape[0] if mask is not None else contact_map.shape[0]
        if contact_map is None or self.spot_top_k <= 0:
            return torch.zeros((B, L, L), dtype=dtype, device=device)

        contact = contact_map[:, :L, :L].to(device=device, dtype=dtype)
        idx = torch.arange(L, device=device)
        seq_sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        allowed = (seq_sep >= 4).unsqueeze(0)
        spot = contact.masked_fill(~allowed, 0.0)
        spot = torch.where(spot >= self.spot_edge_threshold, spot, torch.zeros_like(spot))

        k = min(self.spot_top_k, L)
        if k < L:
            top_values, top_indices = torch.topk(spot, k=k, dim=-1)
            sparse = torch.zeros_like(spot)
            sparse.scatter_(-1, top_indices, top_values)
            spot = sparse

        if mask is not None:
            valid_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
            spot = spot * valid_pair.to(dtype=dtype)
        return spot

    def forward(self, h, mask=None, contact_map=None):
        """Purpose: Apply graph message passing using sequence and SPOT edges.

        Input:
            h: Residue embeddings shaped (B, L, d_model).
            mask: Valid-residue mask shaped (B, L).
            contact_map: Optional SPOT contact map shaped (B, L, L).
        Output:
            Graph-enhanced residue embeddings shaped (B, L, d_model).
        """
        if not self.layers:
            return h

        B, L, _ = h.shape
        dtype = h.dtype
        device = h.device
        if mask is None:
            mask = torch.ones((B, L), dtype=dtype, device=device)

        backbone_adj = self.sequence_adjacency(
            L,
            mask,
            min_sep=1,
            max_sep=1,
            dtype=dtype,
            device=device,
        )
        local_adj = self.sequence_adjacency(
            L,
            mask,
            min_sep=2,
            max_sep=self.local_edge_max_sep,
            dtype=dtype,
            device=device,
        )
        spot_adj = self.spot_adjacency(contact_map, mask, L, dtype, device)

        for layer in self.layers:
            h = layer(h, backbone_adj, local_adj, spot_adj, self.graph_scale)
        return h


class CoordinateRefinementBlock(nn.Module):
    """Purpose: Refine coordinates using embeddings and local coordinate context.

    Input:
        d_model: Residue embedding size.
        hidden_dim: Hidden size for the delta-coordinate MLP.
        dropout: Dropout probability inside the MLP.
        local_window: Maximum sequence separation used for local coordinate features.
        delta_scale: Scalar multiplier for each predicted coordinate update.
    Output:
        Module that maps embeddings and current coordinates to updated coordinates.
    """

    def __init__(self, d_model, hidden_dim, dropout, local_window, delta_scale):
        """Purpose: Initialize one coordinate refinement block.

        Input:
            d_model: Residue embedding size.
            hidden_dim: Hidden size for the refinement MLP.
            dropout: Dropout probability.
            local_window: Maximum local sequence separation.
            delta_scale: Scale for bounded coordinate updates.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.local_window = local_window
        self.delta_scale = delta_scale
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model + 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def local_coordinate_features(self, coords, mask):
        """Purpose: Summarize local coordinate differences around each residue.

        Input:
            coords: Current coordinates shaped (B, L, 3).
            mask: Valid-residue mask shaped (B, L), or None.
        Output:
            Local features shaped (B, L, 4): mean neighbor delta and mean distance.
        """
        B, L, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        idx = torch.arange(L, device=device)
        sep = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        neighbor = ((sep >= 1) & (sep <= self.local_window)).to(dtype=dtype)
        weights = neighbor.unsqueeze(0).expand(B, L, L)
        if mask is not None:
            valid_pair = mask.unsqueeze(1) * mask.unsqueeze(2)
            weights = weights * valid_pair.to(dtype=dtype)

        diff = coords.unsqueeze(1) - coords.unsqueeze(2)
        distances = torch.norm(diff, dim=-1, keepdim=True)
        degree = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_delta = (diff * weights.unsqueeze(-1)).sum(dim=2) / degree
        mean_distance = (distances * weights.unsqueeze(-1)).sum(dim=2) / degree
        return torch.cat([mean_delta, mean_distance], dim=-1)

    def forward(self, h, coords, mask=None):
        """Purpose: Predict and apply one bounded coordinate update.

        Input:
            h: Residue embeddings shaped (B, L, d_model).
            coords: Current coordinates shaped (B, L, 3).
            mask: Valid-residue mask shaped (B, L), or None.
        Output:
            Updated coordinates shaped (B, L, 3).
        """
        local_features = self.local_coordinate_features(coords, mask)
        delta_input = torch.cat([h, coords, local_features], dim=-1)
        delta = self.delta_scale * torch.tanh(self.delta_mlp(delta_input))
        coords = coords + delta
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)
        return coords


class CoordinateRefinementHead(nn.Module):
    """Purpose: Iteratively refine coarse coordinates from residue embeddings.

    Input:
        d_model: Residue embedding size.
        num_steps: Number of refinement blocks.
        hidden_dim: Hidden size for each block.
        dropout: Dropout probability.
        local_window: Maximum local sequence separation.
        delta_scale: Scale for each coordinate update.
    Output:
        Module that returns refined coordinates shaped (B, L, 3).
    """

    def __init__(self, d_model, num_steps, hidden_dim, dropout, local_window, delta_scale):
        """Purpose: Initialize the stack of coordinate refinement blocks.

        Input:
            d_model: Residue embedding size.
            num_steps: Number of update steps.
            hidden_dim: Hidden size in each update MLP.
            dropout: Dropout probability.
            local_window: Maximum local sequence separation.
            delta_scale: Scale for coordinate deltas.
        Output:
            Initialized PyTorch module.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            CoordinateRefinementBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                dropout=dropout,
                local_window=local_window,
                delta_scale=delta_scale,
            )
            for _ in range(num_steps)
        ])

    def forward(self, h, coords, mask=None):
        """Purpose: Apply all coordinate refinement blocks.

        Input:
            h: Residue embeddings shaped (B, L, d_model).
            coords: Coarse coordinates shaped (B, L, 3).
            mask: Valid-residue mask shaped (B, L), or None.
        Output:
            Refined coordinates shaped (B, L, 3).
        """
        for block in self.blocks:
            coords = block(h, coords, mask=mask)
        return coords


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
        use_graph: Whether to enable residual graph message passing.
        graph_layers: Number of graph message-passing layers.
        graph_scale: Residual graph-message scale.
        spot_edge_threshold: Minimum SPOT probability for graph contact edges.
        spot_top_k: Maximum SPOT graph neighbors per residue.
        local_edge_max_sep: Maximum sequence separation for local graph edges.
        coord_refine_steps: Number of coordinate refinement steps after coarse prediction.
        coord_refine_hidden: Hidden size for coordinate refinement MLPs.
        coord_refine_dropout: Dropout probability in coordinate refinement MLPs.
        coord_refine_local_window: Sequence window for local coordinate features.
        coord_refine_delta_scale: Scale applied to each coordinate update.
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
            use_graph = False,
            graph_layers = 0,
            graph_scale = 0.10,
            spot_edge_threshold = 0.50,
            spot_top_k = 8,
            local_edge_max_sep = 4,
            coord_refine_steps = 0,
            coord_refine_hidden = 128,
            coord_refine_dropout = 0.05,
            coord_refine_local_window = 4,
            coord_refine_delta_scale = 0.10,
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
            use_graph: Whether to enable residual graph message passing.
            graph_layers: Number of graph message-passing layers.
            graph_scale: Residual graph-message scale.
            spot_edge_threshold: Minimum SPOT probability for graph contact edges.
            spot_top_k: Maximum SPOT graph neighbors per residue.
            local_edge_max_sep: Maximum sequence separation for local graph edges.
            coord_refine_steps: Number of coordinate refinement steps after coarse prediction.
            coord_refine_hidden: Hidden size for coordinate refinement MLPs.
            coord_refine_dropout: Dropout probability in coordinate refinement MLPs.
            coord_refine_local_window: Sequence window for local coordinate features.
            coord_refine_delta_scale: Scale applied to each coordinate update.
        Output:
            Initialized PyTorch model.
        """
        super().__init__()
        self.d_model = d_model
        self.spot_bias_scale = spot_bias_scale
        self.use_graph = bool(use_graph and graph_layers > 0)
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
        self.graph_encoder = (
            ResidualGraphMessagePassing(
                d_model=d_model,
                num_layers=graph_layers,
                dropout=dropout,
                graph_scale=graph_scale,
                spot_edge_threshold=spot_edge_threshold,
                spot_top_k=spot_top_k,
                local_edge_max_sep=local_edge_max_sep,
            )
            if self.use_graph
            else None
        )
        self.transformer = AttentionBiasEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(d_model, 3)
        self.coord_refiner = (
            CoordinateRefinementHead(
                d_model=d_model,
                num_steps=coord_refine_steps,
                hidden_dim=coord_refine_hidden,
                dropout=coord_refine_dropout,
                local_window=coord_refine_local_window,
                delta_scale=coord_refine_delta_scale,
            )
            if coord_refine_steps > 0
            else None
        )


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
        if self.graph_encoder is not None:
            h = self.graph_encoder(h, mask=mask, contact_map=contact_map)
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
        if self.coord_refiner is not None:
            coords = self.coord_refiner(h, coords, mask=mask)
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)  # (B, L, 3)
        return coords
    
