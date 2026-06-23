"""项目配置文件：运行 main.py 时会读取这里的默认超参数。"""
from dataclasses import dataclass


@dataclass
class Config:
    # 数据集：自动下载 UCI Letter Recognition
    data_dir: str = "data"
    data_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
    random_seed: int = 42

    # 数据划分：先 train/test，再将 train 划分为主网络集/元网络集
    test_ratio: float = 0.2
    meta_ratio_in_train: float = 0.2

    # 缺失机制：MCAR / MAR / MNAR，三部分数据使用同一种缺失模式和缺失率
    missing_type: str = "MCAR"
    missing_rate: float = 0.2

    # 训练超参数
    batch_size: int = 128
    epochs: int = 100
    classifier_pretrain_epochs: int = 30
    lr_main: float = 0.01
    lr_discriminator: float = 0.01
    lr_meta: float = 0.001
    lr_classifier: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0

    # 模型结构
    hidden_dim: int = 100
    latent_dim: int = 32
    meta_hidden_dim: int = 64
    num_classes: int = 26
    hint_rate: float = 0.9

    # 损失稳定项
    eps: float = 1e-8

    # 设备；为 auto 时自动选择 cuda，否则可写 cuda/cpu
    device: str = "auto"
