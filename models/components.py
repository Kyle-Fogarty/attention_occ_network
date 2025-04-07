import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceEncoder(nn.Module):
    """
    Encoder for surface points using transformer self-attention.
    """
    def __init__(self, input_dim, d_model, n_head, num_layers):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4,
            batch_first=True, 
            dropout=0.1, 
            activation=F.gelu
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Surface features, shape (N, input_dim)
            
        Returns:
            torch.Tensor: Encoded features, shape (N, d_model)
        """
        embedded = self.embed(x)  # (N, d_model)
        # Add batch dimension for transformer
        embedded_batch = embedded.unsqueeze(0)  # (1, N, d_model)
        encoded = self.transformer_encoder(embedded_batch)  # (1, N, d_model)
        return encoded.squeeze(0)  # (N, d_model)


class QueryEncoder(nn.Module):
    """
    Simple encoder for query points.
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Query coordinates, shape (B, input_dim)
            
        Returns:
            torch.Tensor: Encoded features, shape (B, d_model)
        """
        return self.embed(x)


class POCOAggregation(nn.Module):
    """
    Point Convolution (POCO) style feature aggregation with multi-head attention.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Relative encoder MLP
        self.relative_encoder_mlp = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Multi-head attention weights
        self.attention_weight_layers = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(num_heads)
        ])
        
    def forward(self, query_embed, neighbor_features, rel_coords):
        """
        Args:
            query_embed (torch.Tensor): Encoded query features, shape (B, d_model)
            neighbor_features (torch.Tensor): Features of neighboring points, shape (B, K, d_model)
            rel_coords (torch.Tensor): Relative coordinates from query to neighbors, shape (B, K, 3)
            
        Returns:
            torch.Tensor: Aggregated features, shape (B, d_model)
        """
        # 1. Relative encoding
        relative_encoder_input = torch.cat([neighbor_features, rel_coords], dim=-1)  # (B, K, d_model + 3)
        zp_q_relative = self.relative_encoder_mlp(relative_encoder_input)  # (B, K, d_model)
        
        # 2. Compute attention weights for each head
        attention_logits_list = []
        for head_layer in self.attention_weight_layers:
            attn_logit = head_layer(zp_q_relative)  # (B, K, 1)
            attention_logits_list.append(attn_logit)
        attention_logits = torch.cat(attention_logits_list, dim=-1)  # (B, K, num_heads)
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, K, num_heads)
        
        # 3. Weighted sum (aggregation) for each head
        aggregated_features_per_head = []
        for h in range(self.num_heads):
            weights_h = attention_weights[:, :, h].unsqueeze(-1)  # (B, K, 1)
            agg_h = torch.sum(weights_h * zp_q_relative, dim=1)  # (B, d_model)
            aggregated_features_per_head.append(agg_h)
        
        # 4. Average across heads
        aggregated_feature = torch.stack(aggregated_features_per_head, dim=0).mean(dim=0)  # (B, d_model)
        
        return aggregated_feature