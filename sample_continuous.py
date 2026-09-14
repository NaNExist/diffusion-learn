"""用指定 ODE 求解器从连续 VP 模型生成 MNIST。"""

from pathlib import Path

import torch

from continuous import ContinuousVPProcess
from model import TinyUNet
from visualize import save_image_grid, save_trajectory


# ===== 可以手动修改的配置 =====
CHECKPOINT = "outputs/continuous_vp_v/checkpoint.pt"
OUTPUT_DIR = "outputs/continuous_vp_v"
SOLVER = "heun"                 # 可选 euler、heun、rk4
NUM_ODE_STEPS = 40              # 每步 NFE：Euler=1、Heun=2、RK4=4
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
    process = ContinuousVPProcess(
        prediction_type=checkpoint["prediction_type"],
        beta_min=checkpoint["beta_min"],
        beta_max=checkpoint["beta_max"],
        time_epsilon=checkpoint["time_epsilon"],
    )

    solution = process.sample(
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
        title=f"Continuous VP / {checkpoint['prediction_type']} / {SOLVER}",
    )
    save_trajectory(
        solution.snapshots,
        str(output_dir / f"trajectory_{SOLVER}.png"),
        title=f"Probability-flow ODE ({SOLVER})",
    )
    print(
        f"采样完成：solver={SOLVER}, steps={NUM_ODE_STEPS}, "
        f"NFE={solution.nfe}, 用时={solution.elapsed_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
