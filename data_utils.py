"""数据下载、读取、划分与 DataLoader 构建。"""
import os
import urllib.request
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from missing_utils import generate_missing_mask, apply_missing
from utils import ensure_dir


class ImputationDataset(Dataset):
    """同时保存完整数据、缺失数据、mask 和分类标签。"""
    def __init__(self, x_full, x_miss, mask, y):
        self.x_full = torch.tensor(x_full, dtype=torch.float32)
        self.x_miss = torch.tensor(x_miss, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_full[idx], self.x_miss[idx], self.mask[idx], self.y[idx]


class FullDataset(Dataset):
    """用于预训练分类器的完整数据集。"""
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def download_letters(data_dir, data_url):
    """自动下载 UCI Letter Recognition 数据集。"""
    ensure_dir(data_dir)
    path = os.path.join(data_dir, "letter-recognition.data")
    if not os.path.exists(path):
        print(f"正在下载 UCI Letters 数据集到 {path} ...")
        urllib.request.urlretrieve(data_url, path)
    return path


def load_letters(path):
    """读取 UCI Letters：第一列是字母标签，其余 16 列是数值特征。"""
    raw = np.genfromtxt(path, delimiter=",", dtype=str)
    y_str = raw[:, 0]
    x = raw[:, 1:].astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str).astype(np.int64)
    return x, y


def prepare_dataloaders(cfg):
    """先划分数据，再分别对 main/meta/test 施加同模式缺失，并保留完整数据。"""
    data_path = download_letters(cfg.data_dir, cfg.data_url)
    x, y = load_letters(data_path)

    x_train_all, x_test, y_train_all, y_test = train_test_split(
        x, y, test_size=cfg.test_ratio, random_state=cfg.random_seed, stratify=y
    )
    x_main, x_meta, y_main, y_meta = train_test_split(
        x_train_all, y_train_all, test_size=cfg.meta_ratio_in_train,
        random_state=cfg.random_seed, stratify=y_train_all
    )

    # 只用训练总集拟合 scaler，避免测试集信息泄露。
    scaler = StandardScaler()
    scaler.fit(x_train_all)
    x_main = scaler.transform(x_main).astype(np.float32)
    x_meta = scaler.transform(x_meta).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    x_train_all_scaled = scaler.transform(x_train_all).astype(np.float32)

    # main/meta/test 使用同一种缺失机制和缺失率；各自保留完整矩阵用于 ARMSE。
    m_main = generate_missing_mask(x_main, cfg.missing_rate, cfg.missing_type, cfg.random_seed + 1)
    m_meta = generate_missing_mask(x_meta, cfg.missing_rate, cfg.missing_type, cfg.random_seed + 2)
    m_test = generate_missing_mask(x_test, cfg.missing_rate, cfg.missing_type, cfg.random_seed + 3)

    train_main = ImputationDataset(x_main, apply_missing(x_main, m_main), m_main, y_main)
    train_meta = ImputationDataset(x_meta, apply_missing(x_meta, m_meta), m_meta, y_meta)
    test_set = ImputationDataset(x_test, apply_missing(x_test, m_test), m_test, y_test)
    classifier_set = FullDataset(x_train_all_scaled, y_train_all)

    loaders = {
        "main": DataLoader(train_main, batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        "meta": DataLoader(train_meta, batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        "test": DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False),
        "classifier": DataLoader(classifier_set, batch_size=cfg.batch_size, shuffle=True),
    }
    return loaders, x.shape[1], cfg.num_classes, scaler
