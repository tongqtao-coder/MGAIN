"""缺失数据生成工具：M=1 表示观测，M=0 表示缺失。"""
import numpy as np


def generate_missing_mask(x, missing_rate=0.2, missing_type="MCAR", seed=None):
    """根据指定机制生成缺失 mask。

    参数：
        x: 完整特征矩阵，形状 [n_samples, n_features]
        missing_rate: 缺失率
        missing_type: MCAR / MAR / MNAR
        seed: 随机种子
    返回：
        mask: 与 x 同形状，1 表示观测，0 表示缺失
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n, d = x.shape
    missing_type = missing_type.upper()

    if missing_type == "MCAR":
        miss = rng.random((n, d)) < missing_rate

    elif missing_type == "MAR":
        # MAR：让缺失概率依赖于其他特征。这里用前半特征的排序分位数影响后半特征缺失概率。
        driver = x[:, : max(1, d // 2)].mean(axis=1)
        driver = (driver - driver.min()) / (driver.max() - driver.min() + 1e-8)
        prob_row = missing_rate * (0.5 + driver)  # 平均附近仍接近 missing_rate
        prob_row = np.clip(prob_row, 0.0, 0.95)
        miss = rng.random((n, d)) < prob_row[:, None]

    elif missing_type == "MNAR":
        # MNAR：让每个变量自身数值越大，缺失概率越高。
        x_norm = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)
        prob = missing_rate * (0.5 + x_norm)
        prob = np.clip(prob, 0.0, 0.95)
        miss = rng.random((n, d)) < prob

    else:
        raise ValueError(f"未知缺失机制: {missing_type}，请使用 MCAR/MAR/MNAR")

    mask = (~miss).astype(np.float32)
    # 防止某一行全部缺失，至少保留一个观测特征。
    all_missing_rows = np.where(mask.sum(axis=1) == 0)[0]
    for i in all_missing_rows:
        mask[i, rng.integers(0, d)] = 1.0
    return mask


def apply_missing(x, mask):
    """将缺失位置置为 0；标准化后的数据用 0 作为占位值较自然。"""
    return np.asarray(x, dtype=np.float32) * np.asarray(mask, dtype=np.float32)
