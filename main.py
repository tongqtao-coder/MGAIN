"""主入口：直接运行 python main.py 即可开始实验。"""
from config import Config
from utils import set_seed, get_device
from data_utils import prepare_dataloaders
from models import VAEGenerator, Discriminator, MetaNet, MLPClassifier
from train import pretrain_classifier, train_meta_gain


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = get_device(cfg.device)
    print(f"使用设备: {device}")
    print(f"缺失机制: {cfg.missing_type}, 缺失率: {cfg.missing_rate}")

    loaders, input_dim, num_classes, _ = prepare_dataloaders(cfg)
    print(f"输入特征维度: {input_dim}, 类别数: {num_classes}")

    classifier = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    generator = VAEGenerator(input_dim=input_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    discriminator = Discriminator(input_dim=input_dim, hidden_dim=cfg.hidden_dim)
    meta_net = MetaNet(input_dim=5, hidden_dim=cfg.meta_hidden_dim)

    print("开始预训练并冻结下游 MLP 分类器...")
    pretrain_classifier(classifier, loaders["classifier"], cfg, device)

    print("开始训练 VAE-GAIN + MetaNet 缺失填充模型...")
    train_meta_gain(generator, discriminator, meta_net, classifier, loaders, cfg, device)


if __name__ == "__main__":
    main()
