"""训练 DDPM。所有常用配置都直接放在文件顶部。"""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from diffusion import DDPM
from model import TinyUNet
from visualize import save_loss_curve


# ===== 可以手动修改的配置（改等号右边即可） =====
DATA_DIR = "data"               # MNIST 下载和缓存目录
OUTPUT_DIR = "outputs/ddpm"     # 权重与 loss 曲线保存目录
BATCH_SIZE = 256                 # 每次训练图片数；显存不足时改成 64 或 32
EPOCHS = 10                      # 完整遍历数据集的次数；测试流程时先设为 1
LEARNING_RATE = 2e-4             # 学习率；不稳定可降到 1e-4，通常不必频繁修改
NUM_DIFFUSION_STEPS = 256        # 扩散总步数 T；越大采样越慢，但离散过程更细
BETA_START = 1e-4                # 第一步加入的噪声量
BETA_END = 0.05                  # 最后一步噪声量；若 T 改为经典 1000，建议改为 0.02
BASE_CHANNELS = 32               # U-Net 宽度；可改 16 提速，或改 64 增加模型容量
NUM_WORKERS = 2                  # 数据加载子进程数；环境报多进程错误时改为 0
SEED = 42                        # 随机种子；相同配置和种子便于复现实验
# ============================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    # 不绑定具体 GPU 编号，单卡和多卡机器都默认使用当前可见的第一张卡。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # ToTensor 把像素变为 [0,1]，Normalize 再把它变为模型使用的 [-1,1]。
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
    process = DDPM(
        num_steps=NUM_DIFFUSION_STEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        # 每个 epoch 新建一个进度条；postfix 会实时显示当前 batch 的 loss。
        progress = tqdm(loader, desc=f"DDPM epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)
        for images, _ in progress:
            images = images.to(device)
            # training_loss 内部会随机选择 t、加噪并计算 epsilon 预测误差。
            loss = process.training_loss(model, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            value = loss.item()
            losses.append(value)
            running_loss += value
            progress.set_postfix(loss=f"{value:.4f}")

        # tqdm.write 不会破坏终端中的进度条布局。
        tqdm.write(f"epoch {epoch + 1:02d} 平均 loss: {running_loss / len(loader):.4f}")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "num_steps": NUM_DIFFUSION_STEPS,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
                "base_channels": BASE_CHANNELS,
            },
            output_dir / "checkpoint.pt",
        )

    save_loss_curve(losses, str(output_dir / "loss_curve.png"), "DDPM training loss")
    print("训练完成，权重保存在", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
