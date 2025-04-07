import torch

class Config:
    # --- Input/Output Configuration ---
    MESH_FILE = 'test.ply'
    OUTPUT_DIR = "output"  # Directory to save point clouds and model

    # --- Sampling Configuration ---
    N_SURF_POINTS = 2048      # Number of points defining the surface context
    N_QUERY_POINTS = 16384    # Number of volume points sampled per epoch FOR TRAINING QUERIES
    BOUNDING_BOX_PADDING = 0.1

    # --- Model Configuration ---
    # Feature dimensions
    INPUT_SURF_DIM = 6        # 3 Coordinates + 3 Normals (for surface points)
    INPUT_QUERY_DIM = 3       # 3 Coordinates (for query points)
    D_MODEL = 128
    N_HEADS_SELF_ATTN = 4     # Heads for surface self-attention
    N_SELF_ATTN_LAYERS = 3
    K_NEIGHBORS = 32          # K neighbors for POCO aggregation (from surface context)
    N_HEADS_AGG = 8           # Heads for POCO aggregation attention

    # --- Training Configuration ---
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 512          # Batch size for QUERY points
    EPOCHS = 1000
    PRINT_EVERY = 20
    RESAMPLE_EVERY = 100      # Resample query volume points every N epochs
    
    # --- Scheduler Configuration ---
    ETA_MIN = 5e-5

    # --- Special Modes ---
    TRAIN_END_TO_END = True  # Set to True to enable gradient flow through context encoder

    # --- Device Configuration ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- KNN Configuration ---
    USE_FAISS = False  # Will be updated based on availability

    # --- Visualization Configuration ---
    VIS_GRID_RESOLUTION = 64
    VIS_HIGH_RES = 128

    @classmethod
    def print_config(cls):
        """Print configuration settings."""
        print(f"Using device: {cls.DEVICE}")
        print(f"Using Faiss: {cls.USE_FAISS}")
        print(f"Surface Points (Context): {cls.N_SURF_POINTS}")
        print(f"Volume Points (Queries): {cls.N_QUERY_POINTS}")
        print(f"End-to-End Training Enabled: {cls.TRAIN_END_TO_END}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")