"""训练从高斯噪声到 MNIST 的直线 Flow Matching 模型。"""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from continuous import FlowMatchingProcess
from model import TinyUNet
from visualize import save_loss_curve


# ===== 可以手动修改的配置 =====
DATA_DIR = "data"
OUTPUT_DIR = "outputs/flow_matching"
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 2e-4
BASE_CHANNELS = 32
NUM_WORKERS = 2
SEED = 42
# ============================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

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
    process = FlowMatchingProcess()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses = []
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Flow Matching epoch {epoch + 1}/{EPOCHS}", dynamic_ncols=True)
        for images, _ in progress:
            images = images.to(device)
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
                "method": "flow_matching",
                "base_channels": BASE_CHANNELS,
                "epoch": epoch + 1,
            },
            output_dir / "checkpoint.pt",
        )

    save_loss_curve(losses, str(output_dir / "loss_curve.png"), "Flow Matching training loss")
    print("训练完成，权重保存在", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
