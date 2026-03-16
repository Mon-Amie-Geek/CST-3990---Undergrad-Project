"""
seed_control.py
Full seed control for reproducibility
Must be imported and called BEFORE any other code in every script and notebook.
"""
import torch
import numpy as np
import random

SEED = 42

def set_all_seeds():
    """
    Set all random seeds for full reproducibility across torch, numpy and random.
    """
    # Fixes randomness
    torch.manual_seed(SEED)
    # Fixes randomness on GPU computations
    torch.cuda.manual_seed_all(SEED)
    # Fixes randomness in NumpPy
    np.random.seed(SEED)
    # Python itself has a random generator
    random.seed(SEED)
    # Forces deterministic GPU Algorithms
    torch.backends.cudnn.deterministic = True
    # Disable performance optimisation that introduces randomness
    torch.backends.cudnn.benchmark = False
    print(f"[seed_control] All seeds set to {SEED}. Deterministic mode ON.")

if __name__ == "__main__":
    set_all_seeds()