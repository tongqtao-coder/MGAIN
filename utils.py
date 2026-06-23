"""通用工具函数。"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定随机种子，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_name: str) -> torch.device:
    """选择 GPU/CPU。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
