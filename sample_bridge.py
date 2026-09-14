"""从测试集源数字出发，沿条件 bridge 生成目标数字。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from continuous import DiffusionBridgeProcess
from model import TinyUNet
from paired_mnist import PairedMNIST
from visualize import save_comparison_rows, save_trajectory


# ===== 可以手动修改的配置 =====
DATA_DIR = "data"
CHECKPOINT = "outputs/bridge_0_to_1/checkpoint.pt"
OUTPUT_DIR = "outputs/bridge_0_to_1"
SOLVER = "heun"                 # 可选 euler、heun、rk4
NUM_ODE_STEPS = 30
NUM_IMAGES = 8
SNAPSHOT_COUNT = 8
SEED = 123
# ============================


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    test_mnist = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    paired = PairedMNIST(
        test_mnist,
        checkpoint["source_digit"],
        checkpoint["target_digit"],
        pair_seed=SEED,
    )
    source_images, paired_targets = next(iter(DataLoader(paired, batch_size=NUM_IMAGES)))
    source_images = source_images.to(device)

    model = TinyUNet(
        base_channels=checkpoint["base_channels"], condition_channels=1
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    process = DiffusionBridgeProcess(
        checkpoint["bridge_sigma"], checkpoint["time_epsilon"]
    )
    solution = process.sample(
        model,
        source_images,
        solver=SOLVER,
        num_steps=NUM_ODE_STEPS,
        snapshot_count=SNAPSHOT_COUNT,
    )

    output_dir = Path(OUTPUT_DIR)
    save_comparison_rows(
        [source_images, solution.final, paired_targets],
        [
            f"source {checkpoint['source_digit']}",
            f"generated {checkpoint['target_digit']}",
            f"paired {checkpoint['target_digit']}",
        ],
        str(output_dir / f"bridge_samples_{SOLVER}.png"),
        count=NUM_IMAGES,
    )
    save_trajectory(
        solution.snapshots,
        str(output_dir / f"bridge_trajectory_{SOLVER}.png"),
        title=(
            f"Bridge {checkpoint['source_digit']} to {checkpoint['target_digit']} "
            f"({SOLVER})"
        ),
    )
    print(
        f"Bridge 采样完成：solver={SOLVER}, steps={NUM_ODE_STEPS}, "
        f"NFE={solution.nfe}, 用时={solution.elapsed_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
