"""训练 NCSN 风格 score model。所有配置都直接放在文件顶部。"""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from diffusion import ScoreProcess
from model import TinyUNet
from visualize import save_loss_curve


# ===== 可以手动修改的配置（改等号右边即可） =====
DATA_DIR = "data"               # MNIST 下载和缓存目录
OUTPUT_DIR = "outputs/score"    # 权重与 loss 曲线保存目录
BATCH_SIZE = 256                 # 每次训练图片数；显存不足时改成 64 或 32
EPOCHS = 10                      # 训练轮数；只测试代码时先设为 1
LEARNING_RATE = 2e-4             # AdamW 学习率；震荡明显时可以减小
NUM_NOISE_LEVELS = 20            # sigma 等级数；越多越细，但 Langevin 采样越慢
SIGMA_MAX = 2.0                  # 最大高斯噪声标准差，应足以大幅遮盖原图片
SIGMA_MIN = 0.02                 # 最小标准差，决定最后阶段能精修到多细
BASE_CHANNELS = 32               # U-Net 宽度；16 更快，64 容量更大但更耗显存
NUM_WORKERS = 2                  # 数据加载子进程数；有多进程问题时设为 0
SEED = 42                        # 随机种子，用于复现实验
# ============================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    # 不绑定具体 GPU 编号，单卡和多卡机器都默认使用当前可见的第一张卡。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 和 DDPM 一样，把 MNIST 像素从 [0,1] 映射到 [-1,1]。
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = TinyUNet(base_channels=BASE_CHANNELS).to(device)
    process = ScoreProcess(
        num_noise_levels=NUM_NOISE_LEVELS,
        sigma_max=SIGMA_MAX,
        sigma_min=SIGMA_MIN,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            loader,
            desc=f"Score epoch {epoch + 1}/{EPOCHS}",
            dynamic_ncols=True,
        )
        for images, _ in progress:
            images = images.to(device)
            # 内部随机选择 sigma，构造 x_0 + sigma*z，并拟合 -z/sigma。
            loss = process.training_loss(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = loss.item()
            losses.append(value)
            running_loss += value
            progress.set_postfix(loss=f"{value:.4f}")

        tqdm.write(f"epoch {epoch + 1:02d} 平均 loss: {running_loss / len(loader):.4f}")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "num_noise_levels": NUM_NOISE_LEVELS,
                "sigma_max": SIGMA_MAX,
                "sigma_min": SIGMA_MIN,
                "base_channels": BASE_CHANNELS,
            },
            output_dir / "checkpoint.pt",
        )

    save_loss_curve(losses, str(output_dir / "loss_curve.png"), "Score model training loss")
    print("训练完成，权重保存在", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
