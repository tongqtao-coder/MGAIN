"""损失函数和 ARMSE 评价指标。"""
import torch
import torch.nn.functional as F


def reconstruction_loss(x_full, x_gen, mask, eps=1e-8):
    """观测位置重构损失，用于学习数据分布。"""
    return (((x_full - x_gen) ** 2) * mask).sum() / (mask.sum() + eps)


def kl_loss(mu, logvar):
    """VAE KL 散度。"""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def discriminator_loss(d_prob, mask, eps=1e-8):
    """判别器二元交叉熵损失。"""
    return F.binary_cross_entropy(d_prob.clamp(eps, 1 - eps), mask)


def generator_adversarial_loss(d_prob, mask, eps=1e-8):
    """生成器对抗损失：在缺失位置希望判别器认为其为观测值。"""
    missing = 1.0 - mask
    target = torch.ones_like(d_prob)
    bce = F.binary_cross_entropy(d_prob.clamp(eps, 1 - eps), target, reduction="none")
    return (bce * missing).sum() / (missing.sum() + eps)


def classification_loss(classifier, x_hat, y):
    """固定分类器提供的下游任务损失。"""
    return F.cross_entropy(classifier(x_hat), y)


def armse(x_true, x_pred, mask, eps=1e-8, strict_formula=True):
    """按截图公式计算 ARMSE。

    UCI Letters 的输入特征全是数值型，所以 F_c 为空，ARMSE 等于所有数值特征
    在缺失位置 RMSE 的平均值。mask=1 表示观测，mask=0 表示缺失。
    strict_formula=True 时严格除以总特征数 d。
    """
    missing = 1.0 - mask
    d = x_true.shape[1]
    values = []
    total = x_true.new_tensor(0.0)
    for j in range(d):
        miss_j = missing[:, j]
        denom = miss_j.sum()
        if denom > 0:
            rmse_j = torch.sqrt((((x_true[:, j] - x_pred[:, j]) ** 2) * miss_j).sum() / (denom + eps))
            total = total + rmse_j
            values.append(rmse_j)
    if strict_formula:
        return total / d
    if len(values) == 0:
        return x_true.new_tensor(0.0)
    return torch.stack(values).mean()


def make_hint(mask, hint_rate):
    """GAIN hint 矩阵：部分位置泄露真实 mask，其余位置置 0.5。"""
    hint_selector = (torch.rand_like(mask) < hint_rate).float()
    return hint_selector * mask + (1.0 - hint_selector) * 0.5


def loss_stats_tensor(rec, kl, adv, cls, epoch_ratio):
    """构造元网络输入统计量，并 detach 避免统计量本身引入不必要高阶图。"""
    return torch.stack([
        rec.detach(), kl.detach(), adv.detach(), cls.detach(),
        torch.as_tensor(epoch_ratio, device=rec.device, dtype=rec.dtype),
    ]).view(1, -1)
