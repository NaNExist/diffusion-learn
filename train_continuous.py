"""训练连续时间 VP 扩散模型，并选择网络要预测的数学量。"""

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from continuous import ContinuousVPProcess
from model import TinyUNet
from visualize import save_loss_curve


# ===== 可以手动修改的配置（不使用命令行参数） =====
DATA_DIR = "data"
PREDICTION_TYPE = "v"           # 可选 epsilon、x0、v、score
OUTPUT_DIR = f"outputs/continuous_vp_{PREDICTION_TYPE}"
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 2e-4
BETA_MIN = 0.1                   # 连续 VP schedule 在 t=0 的 beta
BETA_MAX = 20.0                  # t=1 时应已非常接近标准高斯噪声
TIME_EPSILON = 1e-3              # 避开 sigma(0)=0 的数值奇点
BASE_CHANNELS = 32
NUM_WORKERS = 2
SEED = 42
# =================================================


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("预测目标:", PREDICTION_TYPE)

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
    process = ContinuousVPProcess(
        prediction_type=PREDICTION_TYPE,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        time_epsilon=TIME_EPSILON,
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
            desc=f"Continuous VP ({PREDICTION_TYPE}) epoch {epoch + 1}/{EPOCHS}",
            dynamic_ncols=True,
        )
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
                "method": "continuous_vp",
                "prediction_type": PREDICTION_TYPE,
                "beta_min": BETA_MIN,
                "beta_max": BETA_MAX,
                "time_epsilon": TIME_EPSILON,
                "base_channels": BASE_CHANNELS,
                "epoch": epoch + 1,
            },
            output_dir / "checkpoint.pt",
        )

    save_loss_curve(
        losses,
        str(output_dir / "loss_curve.png"),
        f"Continuous VP ({PREDICTION_TYPE}) training loss",
    )
    print("训练完成，权重保存在", output_dir / "checkpoint.pt")


if __name__ == "__main__":
    main()
