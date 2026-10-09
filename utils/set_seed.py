import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds to make runs fully reproducible."""
    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    # Also seed CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Additional determinism settings
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Environment variable for TensorFlow compatibility
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
