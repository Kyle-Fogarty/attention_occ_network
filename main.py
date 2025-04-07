import os
import argparse
import torch
import trimesh
from config import Config
from train import train


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Attention-based Occupancy Network")
    
    # Basic configuration
    parser.add_argument("--mesh", type=str, default=None, help="Path to input mesh file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    
    # Sampling configuration
    parser.add_argument("--surf_points", type=int, default=None, help="Number of surface points")
    parser.add_argument("--query_points", type=int, default=None, help="Number of query points")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=None, help="Model dimension")
    parser.add_argument("--k_neighbors", type=int, default=None, help="Number of neighbors for POCO aggregation")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    
    # Special modes
    parser.add_argument("--no_e2e", action="store_true", help="Disable end-to-end training")
    parser.add_argument("--no_faiss", action="store_true", help="Disable Faiss usage for KNN")
    
    return parser.parse_args()


def create_dummy_mesh(mesh_file):
    """Create a dummy sphere mesh if no input mesh is provided."""
    print(f"Mesh file {mesh_file} not found. Creating a dummy sphere...")
    try:
        sphere = trimesh.primitives.Sphere()
        sphere.export(mesh_file)
        print(f"Dummy sphere saved to {mesh_file}")
        return True
    except Exception as e:
        print(f"Failed to create dummy mesh: {e}")
        return False


def update_config_from_args(args):
    """Update configuration based on command line arguments."""
    # Only update config if arguments are provided
    if args.mesh:
        Config.MESH_FILE = args.mesh
    if args.output_dir:
        Config.OUTPUT_DIR = args.output_dir
    if args.surf_points:
        Config.N_SURF_POINTS = args.surf_points
    if args.query_points:
        Config.N_QUERY_POINTS = args.query_points
    if args.d_model:
        Config.D_MODEL = args.d_model
    if args.k_neighbors:
        Config.K_NEIGHBORS = args.k_neighbors
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    if args.no_e2e:
        Config.TRAIN_END_TO_END = False
    if args.no_faiss:
        Config.USE_FAISS = False


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration from arguments
    update_config_from_args(args)
    
    # Make sure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Check if mesh file exists, create dummy if needed
    if not os.path.exists(Config.MESH_FILE):
        if not create_dummy_mesh(Config.MESH_FILE):
            print("Cannot proceed without a valid mesh file.")
            return
    
    # Display configuration banner
    print("=" * 80)
    print(f" Attention-based Occupancy Network (Surface Context)")
    print("=" * 80)
    
    # Run training
    model = train(Config)
    
    print("=" * 80)
    print(" Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()