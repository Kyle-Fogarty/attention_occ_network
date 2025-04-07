import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import SurfaceEncoder, QueryEncoder, POCOAggregation


class AttentionOccupancyNetwork(nn.Module):
    def __init__(self, d_model, n_head_self_attn, num_encoder_layers, k_neighbors,
                 input_surf_dim, input_query_dim, num_attn_heads_agg):
        """
        Attention-based occupancy network using surface points as context.
        
        Args:
            d_model (int): Model dimension
            n_head_self_attn (int): Number of heads for self-attention
            num_encoder_layers (int): Number of transformer encoder layers
            k_neighbors (int): K neighbors for POCO aggregation
            input_surf_dim (int): Input dimension for surface points
            input_query_dim (int): Input dimension for query points
            num_attn_heads_agg (int): Number of attention heads for aggregation
        """
        super().__init__()
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        
        # Surface Context Encoder
        self.surface_encoder = SurfaceEncoder(
            input_dim=input_surf_dim,
            d_model=d_model,
            n_head=n_head_self_attn,
            num_layers=num_encoder_layers
        )
        
        # Query Encoder
        self.query_encoder = QueryEncoder(
            input_dim=input_query_dim,
            d_model=d_model
        )
        
        # POCO-style Aggregation
        self.poco_aggregation = POCOAggregation(
            d_model=d_model,
            num_heads=num_attn_heads_agg
        )
        
        # Prediction Head
        self.norm_agg = nn.LayerNorm(d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def compute_surface_context(self, features_surface_norm):
        """
        Computes refined surface features and coordinates.
        
        Args:
            features_surface_norm (torch.Tensor): Normalized surface features
            
        Returns:
            tuple: (context_features, context_coords)
        """
        context_coords = features_surface_norm[:, :3].float()  # (N_surf, 3)
        context_features = self.surface_encoder(features_surface_norm.float())  # (N_surf, d_model)
        return context_features, context_coords

    def forward(self, query_points_coords, context_coords, context_features, knn_index=None, use_faiss=False):
        """
        Predicts occupancy for query points using POCO-style aggregation.
        
        Args:
            query_points_coords (B_q, 3): Normalized coordinates of query points
            context_coords (N_surf, 3): Normalized coordinates of surface context points
            context_features (N_surf, d_model): Refined features of surface context points
            knn_index: Pre-built Faiss index for context_coords
            use_faiss (bool): Flag to indicate whether to use the Faiss index
            
        Returns:
            logits (B_q, 1): Occupancy logits for the query points
        """
        B_q = query_points_coords.shape[0]
        N_ctx = context_coords.shape[0]
        k_actual = min(self.k_neighbors, N_ctx)

        query_points_coords = query_points_coords.float()
        context_coords = context_coords.float()
        context_features = context_features.float()

        # 1. Find K Nearest Neighbors (Indices)
        if use_faiss and knn_index is not None:
            # --- Faiss KNN Search ---
            # Move query points to CPU and ensure contiguous
            query_coords_cpu = query_points_coords.detach().cpu().numpy()
            
            # Perform search based on index type
            if hasattr(knn_index, 'GpuIndex') and isinstance(knn_index, knn_index.GpuIndex):
                # For GPU index, we need to ensure the input is contiguous
                distances_np, indices_np = knn_index.search(query_coords_cpu, k_actual)
            else:  # CPU Index
                distances_np, indices_np = knn_index.search(query_coords_cpu, k_actual)

            # Convert indices back to tensor on the correct device
            nn_indices = torch.from_numpy(indices_np).to(query_points_coords.device)
            nn_indices = nn_indices.long()  # Ensure long type for gather

        else:
            # --- PyTorch KNN Search ---
            dist_sq = torch.cdist(query_points_coords, context_coords)  # (B_q, N_surf)
            _, nn_indices = torch.topk(dist_sq, k_actual, dim=1, largest=False)  # (B_q, K)

        # 2. Encode query points
        query_embed = self.query_encoder(query_points_coords)  # (B_q, d_model)

        # 3. Gather neighbor coordinates and features using indices
        idx_coords_expanded = nn_indices.unsqueeze(-1).expand(-1, -1, 3)
        ctx_coords_batched = context_coords.unsqueeze(0).expand(B_q, -1, -1)
        neighbor_coords = torch.gather(ctx_coords_batched, 1, idx_coords_expanded)  # (B_q, K, 3)

        idx_feat_expanded = nn_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        ctx_feat_batched = context_features.unsqueeze(0).expand(B_q, -1, -1)
        neighbor_features = torch.gather(ctx_feat_batched, 1, idx_feat_expanded)  # (B_q, K, d_model)

        # 4. Compute relative coordinates
        rel_coords = query_points_coords.unsqueeze(1) - neighbor_coords  # (B_q, K, 3)

        # 5. POCO-style aggregation
        aggregated_feature = self.poco_aggregation(
            query_embed=query_embed,
            neighbor_features=neighbor_features,
            rel_coords=rel_coords,
        )

        # 6. Add residual connection from query embedding
        aggregated_feature = aggregated_feature + query_embed

        # 7. Normalize aggregated feature
        aggregated_feature_norm = self.norm_agg(aggregated_feature)  # (B_q, d_model)

        # 8. Predict Occupancy
        logits = self.prediction_head(aggregated_feature_norm)  # (B_q, 1)

        return logits