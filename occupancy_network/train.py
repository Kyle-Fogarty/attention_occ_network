import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from config import Config
from data.dataset import prepare_surface_data, sample_query_points
from models.network import AttentionOccupancyNetwork
from utils.knn import initialize_faiss_index
from utils.visualization import visualize_reconstruction


def train(config=None):
    """
    Main training function for the Attention Occupancy Network.
    
    Args:
        config: Configuration object, if None uses the default Config class
    """
    if config is None:
        config = Config
    
    # Make sure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Print configuration
    config.print_config()
    
    # 1. Prepare Surface Data
    mesh, features_surface_norm, norm_params = prepare_surface_data(
        config.MESH_FILE, 
        config.N_SURF_POINTS,
        config.OUTPUT_DIR
    )
    if mesh is None:
        print("Failed to load mesh. Exiting.")
        return
        
    center, scale = norm_params
    surface_coords_norm = features_surface_norm[:, :3].float().contiguous()
    
    # 2. Initialize FAISS index for faster KNN if available
    knn_index = initialize_faiss_index(surface_coords_norm)
    
    # 3. Sample initial query points
    points_query_norm, occupancy_query = sample_query_points(
        mesh, config.N_QUERY_POINTS, config.BOUNDING_BOX_PADDING, center, scale
    )
    num_query_points = points_query_norm.shape[0]
    
    # 4. Initialize model, loss, optimizer
    model = AttentionOccupancyNetwork(
        d_model=config.D_MODEL,
        n_head_self_attn=config.N_HEADS_SELF_ATTN,
        num_encoder_layers=config.N_SELF_ATTN_LAYERS,
        k_neighbors=config.K_NEIGHBORS,
        input_surf_dim=config.INPUT_SURF_DIM,
        input_query_dim=config.INPUT_QUERY_DIM,
        num_attn_heads_agg=config.N_HEADS_AGG
    ).to(device).float()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=config.ETA_MIN
    )
    
    print(f"Model initialized. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Using CosineAnnealingLR scheduler: T_max={config.EPOCHS}, eta_min={config.ETA_MIN}")
    
    # Helper function for context computation
    def get_surface_context(surf_features_norm):
        with torch.set_grad_enabled(config.TRAIN_END_TO_END):
            ctx_features, ctx_coords = model.compute_surface_context(surf_features_norm)
        if not config.TRAIN_END_TO_END:
            ctx_features = ctx_features.detach()
            ctx_coords = ctx_coords.detach()
        return ctx_features, ctx_coords
    
    # Pre-compute context if not training end-to-end
    if not config.TRAIN_END_TO_END:
        print("Pre-computing fixed surface context features...")
        _surface_context_features, _surface_context_coords = get_surface_context(features_surface_norm)
        print(f"Fixed Context Features: {_surface_context_features.shape}, Coords: {_surface_context_coords.shape}")
    else:
        # Need the initial coords even if features are recomputed each batch
        _, _surface_context_coords = get_surface_context(features_surface_norm)
        _surface_context_coords = _surface_context_coords.detach()  # Coords don't require grad
    
    # 5. Training Loop
    start_time = time.time()
    print(f"\n--- Starting Training (End-to-End: {config.TRAIN_END_TO_END}, Faiss: {config.USE_FAISS}) ---")
    
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        # Resample query points periodically
        if epoch > 0 and epoch % config.RESAMPLE_EVERY == 0:
            points_query_norm, occupancy_query = sample_query_points(
                mesh, config.N_QUERY_POINTS, config.BOUNDING_BOX_PADDING, center, scale
            )
            num_query_points = points_query_norm.shape[0]
        
        # Track metrics
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        # Shuffle data for this epoch
        perm = torch.randperm(num_query_points, device=device)
        points_query_shuffled = points_query_norm[perm]
        occupancy_query_shuffled = occupancy_query[perm]
        
        # Batch processing
        for i in range(0, num_query_points, config.BATCH_SIZE):
            batch_indices = range(i, min(i + config.BATCH_SIZE, num_query_points))
            query_coords_batch = points_query_shuffled[batch_indices]
            target_occ_batch = occupancy_query_shuffled[batch_indices].unsqueeze(-1)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Compute context features if training end-to-end
            if config.TRAIN_END_TO_END:
                surface_context_features, _ = get_surface_context(features_surface_norm)
            else:
                surface_context_features = _surface_context_features  # Use pre-computed
            
            # Forward pass
            logits = model(
                query_coords_batch,
                _surface_context_coords,  # Use fixed initial coords for KNN
                surface_context_features,  # Use potentially updated features
                knn_index=knn_index,      # Pass the Faiss index
                use_faiss=config.USE_FAISS  # Enable Faiss usage
            )
            
            # Loss and backward
            loss = criterion(logits, target_occ_batch)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_acc += (preds == target_occ_batch).float().mean().item()
            num_batches += 1
        
        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log progress
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        epoch_time = time.time() - epoch_start_time
        
        if (epoch + 1) % config.PRINT_EVERY == 0 or epoch == 0 or epoch == config.EPOCHS - 1:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.5f}, Acc: {avg_acc:.4f}, "
                  f"LR: {current_lr:.2e}, Time: {epoch_time:.2f}s (E2E: {config.TRAIN_END_TO_END}, Faiss: {config.USE_FAISS})")
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.2f} seconds.")
    
    # Save model
    save_path = os.path.join(config.OUTPUT_DIR, 
                           f"occupancy_model_surf_ctx_{'e2e' if config.TRAIN_END_TO_END else 'fixed'}_" +
                           f"faiss_{config.USE_FAISS}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Run visualization
    print("\n--- Starting Visualization ---")
    model.eval()
    with torch.no_grad():
        # Get final context features for visualization
        if config.TRAIN_END_TO_END:
            vis_context_features, _ = get_surface_context(features_surface_norm)
        else:
            vis_context_features = _surface_context_features
        
        visualize_reconstruction(
            model=model,
            context_coords=_surface_context_coords,
            context_features=vis_context_features,
            center=center,
            scale=scale,
            knn_index=knn_index,
            use_faiss=config.USE_FAISS,
            output_dir=config.OUTPUT_DIR
        )
    
    print("Training and visualization complete.")
    return model


if __name__ == "__main__":
    train()