"""测试集评价。"""
import torch
from losses import armse


@torch.no_grad()
def evaluate(generator, classifier, loader, device):
    generator.eval()
    classifier.eval()
    total_correct = 0
    total_num = 0
    armse_weighted = 0.0

    for x_full, x_miss, mask, y in loader:
        x_full, x_miss, mask, y = x_full.to(device), x_miss.to(device), mask.to(device), y.to(device)
        x_hat, _, _, _ = generator(x_miss, mask)
        batch_armse = armse(x_full, x_hat, mask, strict_formula=True)
        armse_weighted += batch_armse.item() * x_full.size(0)

        pred = classifier(x_hat).argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_num += y.numel()

    return armse_weighted / max(total_num, 1), total_correct / max(total_num, 1)
