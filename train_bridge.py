"""训练指定 MNIST 类别之间的条件 diffusion bridge，默认学习 0→1。"""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from continuous import DiffusionBridgeProcess
from model import TinyUNet
from paired_mnist import PairedMNIST
from visualize import save_loss_curve


# ===== 可以手动修改的配置（例如把 0→1 改成 2→7） =====
DATA_DIR = "data"
SOURCE_DIGIT = 0
TARGET_DIGIT = 1
OUTPUT_DIR = f"outputs/bridge_{SOURCE_DIGIT}_to_{TARGET_DIGIT}"
BATCH_SIZE = 256
EPOCHS = 15                    # 条件类别变换通常比无条件 MNIST 多训练几轮
LEARNING_RATE = 2e-4
BRIDGE_SIGMA = 0.5             # 中间时刻的随机性；端点处始终为 0
TIME_EPSILON = 1e-3            # 避开 bridge 噪声导数在端点的奇点
BASE_CHANNELS = 32
NUM_WORKERS = 2
SEED = 42
# =========================================================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print(f"训练数字 bridge: {SOURCE_DIGIT} → {TARGET_DIGIT}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    dataset = PairedMNIST(mnist, SOURCE_DIGIT, TARGET_DIGIT, pair_seed=SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    # current x_t 与 source 各占一个灰度通道；输出仍是一通道速度场。
    model = TinyUNet(base_channels=BASE_CHANNELS, condition_channels=1).to(device)
    process = DiffusionBridgeProcess(BRIDGE_SIGMA, TIME_EPSILON)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            loader,
            desc=f"Bridge {SOURCE_DIGIT}→{TARGET_DIGIT} epoch {epoch + 1}/{EPOCHS}",
            dynamic_ncols=True,
        )
        for source_images, target_images in progress:
            source_images = source_images.to(device)
            target_images = target_images.to(device)
            loss = process.training_loss(model, source_images, target_images)
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
                "method": "diffusion_bridge",
                "source_digit": SOURCE_DIGIT,
                "target_digit": TARGET_DIGIT,
                "bridge_sigma": BRIDGE_SIGMA,
                "time_epsilon": TIME_EPSILON,
                "base_channels": BASE_CHANNELS,
                "epoch": epoch + 1,
            },
            output_dir / "checkpoint.pt",
        )

    save_loss_curve(
        losses,
        str(output_dir / "loss_curve.png"),
        f"Diffusion Bridge {SOURCE_DIGIT} to {TARGET_DIGIT}",
    )
    print("训练完成，权重保存在", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
