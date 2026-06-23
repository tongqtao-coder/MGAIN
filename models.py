"""模型定义：VAE-GAIN 主网络、判别器、元网络和固定分类器。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_dim, hidden_dim, output_dim, hidden_layers=5, final_activation=None):
    layers = []
    last = input_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(last, hidden_dim), nn.ReLU()]
        last = hidden_dim
    layers.append(nn.Linear(last, output_dim))
    if final_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class VAEGenerator(nn.Module):
    """5 层编码器 + 5 层解码器的 VAE 填充生成器。"""
    def __init__(self, input_dim, hidden_dim=100, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        # 输入拼接：缺失后数据 x_miss、mask、随机噪声 z_noise
        self.encoder = make_mlp(input_dim * 3, hidden_dim, hidden_dim, hidden_layers=5)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = make_mlp(latent_dim, hidden_dim, input_dim, hidden_layers=5)

    def encode(self, x_miss, mask, noise):
        h = self.encoder(torch.cat([x_miss, mask, noise], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_miss, mask, noise=None):
        if noise is None:
            noise = torch.rand_like(x_miss)
        mu, logvar = self.encode(x_miss, mask, noise)
        z = self.reparameterize(mu, logvar)
        x_gen = self.decode(z)
        x_hat = mask * x_miss + (1.0 - mask) * x_gen
        return x_hat, x_gen, mu, logvar


class Discriminator(nn.Module):
    """GAIN 判别器：预测每个位置是观测值还是生成值。"""
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = make_mlp(input_dim * 2, hidden_dim, input_dim, hidden_layers=3)

    def forward(self, x_hat, hint):
        return torch.sigmoid(self.net(torch.cat([x_hat, hint], dim=1)))


class MetaNet(nn.Module):
    """元网络：根据当前损失统计输出 beta、gamma、eta 三个正权重。"""
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, stats):
        weights = F.softplus(self.net(stats)) + 1e-6
        beta, gamma, eta = weights[:, 0].mean(), weights[:, 1].mean(), weights[:, 2].mean()
        return beta, gamma, eta


class MLPClassifier(nn.Module):
    """适合 UCI Letters 数值表格数据的 MLP 分类器。"""
    def __init__(self, input_dim, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
