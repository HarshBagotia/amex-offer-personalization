from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every source of randomness we control so runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash the training data so a saved model can be traced back to the exact file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
