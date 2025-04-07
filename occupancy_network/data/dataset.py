import os
import torch
import torch.nn.functional as F
import trimesh
import numpy as np
from config import Config


def prepare_surface_data(mesh_file, n_surf, output_dir):
    """ Prepares initial surface points (context), normalization constants,
        and saves point cloud data. 
    
    Args:
        mesh_file (str): Path to the input mesh file
        n_surf (int): Number of surface points to sample
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: (mesh, features_surface_norm, (center, scale))
    """
    print(f"Loading mesh: {mesh_file}...")
    try:
        mesh = trimesh.load(mesh_file, force='mesh', process=True)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None, None, None

    if not mesh.is_watertight:
        print("Warning: Mesh may not be watertight.")

    print(f"Sampling {n_surf} surface points and normals...")
    points_surface, face_indices = trimesh.sample.sample_surface(mesh, n_surf)
    normals_surface = mesh.face_normals[face_indices]

    # --- Save Original Sampled Point Cloud ---
    print(f"Saving original sampled point cloud data...")
    raw_pc_file = os.path.join(output_dir, "surface_points_raw.ply")
    # Create a Trimesh object for the point cloud with normals
    pc_trimesh = trimesh.PointCloud(vertices=points_surface, vertex_normals=normals_surface)
    pc_trimesh.export(raw_pc_file)
    print(f"Saved raw point cloud to {raw_pc_file}")

    points_surface_torch = torch.tensor(points_surface, dtype=torch.float32)
    normals_surface_torch = torch.tensor(normals_surface, dtype=torch.float32)

    # Combine coordinates and normals for surface points
    features_surface = torch.cat([points_surface_torch, normals_surface_torch], dim=1) # (N_surf, 6)

    # --- Determine Normalization based on Surface Points ---
    center = points_surface_torch.mean(dim=0)
    points_surface_centered = points_surface_torch - center
    # Ensure scale calculation avoids division by zero if mesh is tiny
    scale = torch.max(torch.sqrt(torch.sum(points_surface_centered**2, dim=1)))
    scale = torch.max(scale, torch.tensor(1e-6)) # Prevent scale=0

    # Normalize surface features
    points_surface_norm = (points_surface_torch - center) / scale
    normals_surface_norm = F.normalize(normals_surface_torch, p=2, dim=1) # Ensure unit normals
    features_surface_norm = torch.cat([points_surface_norm, normals_surface_norm], dim=1)

    print(f"Initial surface points/features shape: {features_surface_norm.shape}")
    print(f"Normalization Center: {center.numpy()}, Scale: {scale.item()}")

    # --- Save Normalized Point Cloud and Norm Params ---
    norm_pc_file = os.path.join(output_dir, "surface_points_normalized.npy")
    norm_params_file = os.path.join(output_dir, "normalization_params.npz")
    np.save(norm_pc_file, features_surface_norm.numpy())
    np.savez(norm_params_file, center=center.numpy(), scale=scale.numpy())
    print(f"Saved normalized point cloud features to {norm_pc_file}")
    print(f"Saved normalization parameters to {norm_params_file}")

    return mesh, features_surface_norm.to(Config.DEVICE), (center.to(Config.DEVICE), scale.to(Config.DEVICE))


def sample_query_points(mesh, n_query, padding, center, scale):
    """ Samples VOLUME points to be used as QUERIES and computes their occupancy 
    
    Args:
        mesh (trimesh.Trimesh): The mesh to sample from
        n_query (int): Number of query points to sample
        padding (float): Padding for the bounding box
        center (torch.Tensor): Center of the mesh
        scale (torch.Tensor): Scale of the mesh
        
    Returns:
        tuple: (points_volume_norm, occupancy)
    """
    min_bound, max_bound = np.array(mesh.bounds)
    # Add padding to bounds for sampling
    min_bound_pad = min_bound - padding * (max_bound - min_bound)
    max_bound_pad = max_bound + padding * (max_bound - min_bound)

    # Sample points uniformly in the padded bounding box (original space)
    points_volume = (torch.rand(n_query, 3, dtype=torch.float32, device=Config.DEVICE) *
                     (torch.tensor(max_bound_pad - min_bound_pad, device=Config.DEVICE)) +
                     torch.tensor(min_bound_pad, device=Config.DEVICE))

    # Check occupancy (requires points on CPU for trimesh.contains)
    occupancy = torch.tensor(mesh.contains(points_volume.cpu().numpy()),
                             dtype=torch.float32, device=Config.DEVICE)

    # Normalize coordinates using pre-calculated center and scale
    points_volume_norm = (points_volume - center) / scale

    return points_volume_norm, occupancy