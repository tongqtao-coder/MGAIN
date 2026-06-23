"""训练流程：分类器预训练 + VAE-GAIN 元学习训练。"""
import itertools
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.func import functional_call

from losses import (
    reconstruction_loss, kl_loss, discriminator_loss, generator_adversarial_loss,
    classification_loss, armse, make_hint, loss_stats_tensor,
)
from evaluate import evaluate


def pretrain_classifier(classifier, loader, cfg, device):
    """在完整训练数据上预训练 MLP 分类器，之后冻结参数。"""
    classifier.to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg.lr_classifier, momentum=cfg.momentum)
    for epoch in range(1, cfg.classifier_pretrain_epochs + 1):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = classifier(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"分类器预训练 Epoch [{epoch:03d}/{cfg.classifier_pretrain_epochs}] Loss={total_loss/total:.4f} Acc={correct/total:.4f}")

    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)


def compute_generator_losses(generator, discriminator, classifier, x_full, x_miss, mask, y, cfg, epoch_ratio):
    """计算主网络各部分损失。"""
    x_hat, x_gen, mu, logvar = generator(x_miss, mask)
    hint = make_hint(mask, cfg.hint_rate)
    d_prob_for_g = discriminator(x_hat, hint)
    rec = reconstruction_loss(x_full, x_gen, mask, cfg.eps)
    kl = kl_loss(mu, logvar)
    adv = generator_adversarial_loss(d_prob_for_g, mask, cfg.eps)
    cls = classification_loss(classifier, x_hat, y)
    stats = loss_stats_tensor(rec, kl, adv, cls, epoch_ratio)
    return x_hat, rec, kl, adv, cls, stats


def virtual_generator_forward(generator, params, x_miss, mask):
    """使用临时参数执行生成器前向传播，用于方案 A 的可微 inner-loop。"""
    noise = torch.rand_like(x_miss)
    return functional_call(generator, params, (x_miss, mask, noise))


def train_meta_gain(generator, discriminator, meta_net, classifier, loaders, cfg, device):
    """主训练循环。"""
    generator.to(device)
    discriminator.to(device)
    meta_net.to(device)

    opt_g = torch.optim.SGD(generator.parameters(), lr=cfg.lr_main, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_d = torch.optim.SGD(discriminator.parameters(), lr=cfg.lr_discriminator, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_meta = torch.optim.SGD(meta_net.parameters(), lr=cfg.lr_meta, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    meta_iter = itertools.cycle(loaders["meta"])

    for epoch in range(1, cfg.epochs + 1):
        generator.train()
        discriminator.train()
        meta_net.train()
        epoch_ratio = epoch / cfg.epochs
        total_main_loss = 0.0
        total_samples = 0
        last_weights = (0.0, 0.0, 0.0)

        for x_full, x_miss, mask, y in loaders["main"]:
            x_full, x_miss, mask, y = x_full.to(device), x_miss.to(device), mask.to(device), y.to(device)
            x_meta_full, x_meta_miss, m_meta, y_meta = next(meta_iter)
            x_meta_full = x_meta_full.to(device)
            x_meta_miss = x_meta_miss.to(device)
            m_meta = m_meta.to(device)
            y_meta = y_meta.to(device)

            # 1) 先更新判别器：判别真实观测位置和生成缺失位置。
            with torch.no_grad():
                x_hat_detached, _, _, _ = generator(x_miss, mask)
            hint = make_hint(mask, cfg.hint_rate)
            d_prob = discriminator(x_hat_detached.detach(), hint)
            d_loss = discriminator_loss(d_prob, mask, cfg.eps)
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip_norm)
            opt_d.step()

            # 2) 元网络根据主训练 batch 的损失统计输出 beta/gamma/eta。
            _, rec, kl, adv, cls, stats = compute_generator_losses(
                generator, discriminator, classifier, x_full, x_miss, mask, y, cfg, epoch_ratio
            )
            beta, gamma, eta = meta_net(stats)
            inner_loss = rec + beta * kl + gamma * cls + eta * adv

            # 3) 方案 A：可微虚拟更新主生成器参数 theta_hat = theta - lr * grad。
            params = OrderedDict(generator.named_parameters())
            grads = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=True, allow_unused=True)
            fast_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                fast_params[name] = param if grad is None else param - cfg.inner_lr * grad

            # 4) 用虚拟更新后的主网络填充元数据缺失部分，与元数据完整版计算 ARMSE，更新元网络。
            x_meta_hat, _, _, _ = virtual_generator_forward(generator, fast_params, x_meta_miss, m_meta)
            meta_loss = armse(x_meta_full, x_meta_hat, m_meta, strict_formula=True)
            opt_meta.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_net.parameters(), cfg.grad_clip_norm)
            opt_meta.step()

            # 5) 用更新后的元网络重新输出权重，正式更新主生成器。
            _, rec2, kl2, adv2, cls2, stats2 = compute_generator_losses(
                generator, discriminator, classifier, x_full, x_miss, mask, y, cfg, epoch_ratio
            )
            beta2, gamma2, eta2 = meta_net(stats2)
            main_loss = rec2 + beta2 * kl2 + gamma2 * cls2 + eta2 * adv2
            opt_g.zero_grad()
            main_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.grad_clip_norm)
            opt_g.step()

            total_main_loss += main_loss.item() * x_full.size(0)
            total_samples += x_full.size(0)
            last_weights = (beta2.item(), gamma2.item(), eta2.item())

        test_armse, test_acc = evaluate(generator, classifier, loaders["test"], device)
        avg_loss = total_main_loss / max(total_samples, 1)
        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}] "
            f"TrainLoss={avg_loss:.6f} TestARMSE={test_armse:.6f} TestAcc={test_acc:.4f} "
            f"beta={last_weights[0]:.4f} gamma={last_weights[1]:.4f} eta={last_weights[2]:.4f}"
        )
