import torch
from config import Config

# Check faiss availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("Warning: Faiss library not found. Falling back to torch KNN.")
    print("Install Faiss (CPU): pip install faiss-cpu")
    print("Install Faiss (GPU): pip install faiss-gpu")
    FAISS_AVAILABLE = False
    faiss = None

def initialize_faiss_index(surface_coords):
    """
    Initialize a FAISS index for the surface coordinates.
    
    Args:
        surface_coords (torch.Tensor): Surface point coordinates, shape (N, 3)
        
    Returns:
        faiss.Index or None: FAISS index if FAISS is available, None otherwise
    """
    # Update config based on FAISS availability
    Config.USE_FAISS = FAISS_AVAILABLE
    
    if not FAISS_AVAILABLE:
        print("Faiss not available. Using PyTorch KNN instead.")
        return None
        
    knn_index = None
    try:
        print("Building Faiss index...")
        d = surface_coords.shape[1]  # Dimension (should be 3)
        
        # Convert to CPU numpy array for faiss
        surface_coords_cpu = surface_coords.detach().cpu().numpy()
        
        if Config.DEVICE.type == 'cuda':
            try:
                res = faiss.StandardGpuResources()  # Create GPU resources
                # Flat L2 index: Brute-force search, exact results
                index_flat = faiss.IndexFlatL2(d)
                knn_index = faiss.index_cpu_to_gpu(res, Config.DEVICE.index or 0, index_flat)
                knn_index.add(surface_coords_cpu)
                print(f"Using Faiss GPU IndexFlatL2 on device {Config.DEVICE}")
            except Exception as e:
                print(f"Failed to create Faiss GPU index: {e}. Falling back to CPU.")
                knn_index = faiss.IndexFlatL2(d)
                knn_index.add(surface_coords_cpu)
                print("Using Faiss CPU IndexFlatL2.")
        else:  # CPU device
            knn_index = faiss.IndexFlatL2(d)
            knn_index.add(surface_coords_cpu)
            print("Using Faiss CPU IndexFlatL2.")
            
        print(f"Faiss index built. Contains {knn_index.ntotal} points.")
        return knn_index
        
    except Exception as e:
        print(f"Error initializing Faiss index: {e}")
        Config.USE_FAISS = False
        return None