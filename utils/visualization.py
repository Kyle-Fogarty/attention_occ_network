import os
import torch
import numpy as np
import trimesh
import skimage.measure
from  config import Config


@torch.no_grad()
def visualize_reconstruction(model, context_coords, context_features, center, scale,
                             grid_resolution=None, knn_index=None, use_faiss=False,
                             output_dir=None):
    """ 
    Generates mesh from learned occupancy field using marching cubes.
    
    Args:
        model: The trained occupancy network model
        context_coords (torch.Tensor): Context point coordinates
        context_features (torch.Tensor): Context point features
        center (torch.Tensor): Center used for normalization
        scale (torch.Tensor): Scale used for normalization
        grid_resolution (int, optional): Grid resolution for marching cubes
        knn_index: FAISS index for KNN search
        use_faiss (bool): Whether to use FAISS for KNN search
        output_dir (str, optional): Directory to save output meshes
    """
    print("Visualizing reconstruction...")
    model.eval()  # Ensure model is in eval mode

    # Use default resolution from config if not specified
    if grid_resolution is None:
        grid_resolution = Config.VIS_GRID_RESOLUTION
        
    # Use default output directory from config if not specified
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR

    def predict_occupancy_grid(resolution):
        print(f"  Querying grid at resolution {resolution}x{resolution}x{resolution}...")
        grid_vals = torch.linspace(-1.0, 1.0, resolution, device=Config.DEVICE)
        x, y, z = torch.meshgrid(grid_vals, grid_vals, grid_vals, indexing='ij')
        query_grid = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).float()

        occ_list = []
        grid_batch_size = 8192  # Adjust based on GPU memory
        for i in range(0, query_grid.shape[0], grid_batch_size):
            query_batch = query_grid[i: i + grid_batch_size]
            # Pass Faiss index to model during visualization inference
            logits = model(query_batch, context_coords, context_features,
                           knn_index=knn_index, use_faiss=use_faiss)
            occ_list.append(torch.sigmoid(logits).squeeze(-1))

        predicted_occ = torch.cat(occ_list).cpu().numpy()

        if np.any(np.isnan(predicted_occ)) or np.any(np.isinf(predicted_occ)):
            print("Warning: NaN or Inf found in predicted occupancies. Clamping.")
            predicted_occ = np.nan_to_num(predicted_occ, nan=0.0, posinf=1.0, neginf=0.0)

        volume = predicted_occ.reshape(resolution, resolution, resolution)
        print(f"  Grid prediction complete. Occupancy range: [{volume.min():.3f}, {volume.max():.3f}]")
        return volume

    # Generate high-res for export
    volume_highres = predict_occupancy_grid(Config.VIS_HIGH_RES)
    try:
        print(f"  Running Marching Cubes ({Config.VIS_HIGH_RES}³)...")
        verts_hr, faces_hr, _, _ = skimage.measure.marching_cubes(volume_highres, level=0.5)
        if len(verts_hr) == 0:
            print("Warning: Marching Cubes produced no vertices at level 0.5.")
            return

        verts_hr_denorm = verts_hr / (Config.VIS_HIGH_RES - 1) * 2.0 - 1.0
        verts_hr_denorm = verts_hr_denorm * scale.cpu().numpy() + center.cpu().numpy()

        reconstructed_mesh_hr = trimesh.Trimesh(vertices=verts_hr_denorm, faces=faces_hr, process=False)
        print(f"  Generated high-res mesh with {len(verts_hr)} vertices, {len(faces_hr)} faces.")

        export_filename = os.path.join(output_dir, 
                                       f"reconstructed_mesh_{Config.VIS_HIGH_RES}_surf_ctx_" + 
                                       f"{'e2e' if Config.TRAIN_END_TO_END else 'fixed'}_" + 
                                       f"faiss_{use_faiss}.obj")
        print(f"Exporting high-resolution mesh to {export_filename}...")
        reconstructed_mesh_hr.export(export_filename)
        print("High-resolution mesh exported.")

    except ValueError as ve:
        print(f"Marching Cubes Error (High Res): {ve}. Check occupancy range and level=0.5.")
    except Exception as e:
        print(f"Mesh generation/export failed (High Res): {e}")
        import traceback
        traceback.print_exc()