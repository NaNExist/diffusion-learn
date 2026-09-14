"""使用 Euler、Heun 或 RK4 对 Flow Matching 速度场积分。"""

from pathlib import Path

import torch

from continuous import FlowMatchingProcess
from model import TinyUNet
from visualize import save_image_grid, save_trajectory


# ===== 可以手动修改的配置 =====
CHECKPOINT = "outputs/flow_matching/checkpoint.pt"
OUTPUT_DIR = "outputs/flow_matching"
SOLVER = "heun"
NUM_ODE_STEPS = 20
NUM_IMAGES = 16
SNAPSHOT_COUNT = 8
SEED = 123
# ============================


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = TinyUNet(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])

    solution = FlowMatchingProcess().sample(
        model,
        num_images=NUM_IMAGES,
        solver=SOLVER,
        num_steps=NUM_ODE_STEPS,
        snapshot_count=SNAPSHOT_COUNT,
    )
    output_dir = Path(OUTPUT_DIR)
    save_image_grid(
        solution.final,
        str(output_dir / f"samples_{SOLVER}.png"),
        nrow=4,
        title=f"Flow Matching / {SOLVER}",
    )
    save_trajectory(
        solution.snapshots,
        str(output_dir / f"trajectory_{SOLVER}.png"),
        title=f"Flow Matching trajectory ({SOLVER})",
    )
    print(
        f"采样完成：solver={SOLVER}, steps={NUM_ODE_STEPS}, "
        f"NFE={solution.nfe}, 用时={solution.elapsed_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
