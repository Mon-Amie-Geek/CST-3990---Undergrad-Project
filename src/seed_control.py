"""
seed_control.py
Full seed control for reproducibility
Sets torch, numpy and random seeds to 42
CUDA seed (controls GPU randomness) is only set if a GPU is available (guarded for CPU-only environments)
Call set_all_seeds() before any other code in every script and notebook.
"""
# Changed code for robustness and portability to prevent unexpected behaviour or silent inconsistencies.
import torch
import numpy as np
import random

SEED = 42

def set_all_seeds(seed=SEED):
    """
    Set all random seeds for full reproducibility across torch, numpy and random.
    """
    # Fixes randomness
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixes randomness on CPU computations
    np.random.seed(seed)
    # Python itself has a random generator
    random.seed(seed)
    # Forces deterministic GPU Algorithms
    torch.backends.cudnn.deterministic = True
    # Disable performance optimisation that introduces randomness
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"[seed_control] All seeds set to {seed}. "
          f"CUDA available: {torch.cuda.is_available()}.")
          


