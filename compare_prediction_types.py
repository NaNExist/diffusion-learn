"""比较 epsilon、x0、v、score 四种连续 VP 预测目标。"""

from pathlib import Path

import torch

from continuous import ContinuousVPProcess
from model import TinyUNet
from visualize import save_comparison_rows


# ===== 可以手动修改的配置 =====
# 先分别修改 train_continuous.py 顶部的 PREDICTION_TYPE，训练四个 checkpoint。
CHECKPOINTS = {
    "epsilon": "outputs/continuous_vp_epsilon/checkpoint.pt",
    "x0": "outputs/continuous_vp_x0/checkpoint.pt",
    "v": "outputs/continuous_vp_v/checkpoint.pt",
    "score": "outputs/continuous_vp_score/checkpoint.pt",
}
OUTPUT_PATH = "outputs/prediction_type_comparison.png"
SOLVER = "heun"
NUM_ODE_STEPS = 40
NUM_IMAGES = 8
SEED = 123
# ============================


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_noise = torch.randn(NUM_IMAGES, 1, 28, 28, device=device)
    image_groups = []
    labels = []

    for expected_type, checkpoint_path in CHECKPOINTS.items():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        actual_type = checkpoint["prediction_type"]
        if actual_type != expected_type:
            raise ValueError(
                f"{checkpoint_path} 中是 {actual_type} 模型，配置却把它标成 {expected_type}"
            )
        model = TinyUNet(base_channels=checkpoint["base_channels"]).to(device)
        model.load_state_dict(checkpoint["model"])
        process = ContinuousVPProcess(
            prediction_type=actual_type,
            beta_min=checkpoint["beta_min"],
            beta_max=checkpoint["beta_max"],
            time_epsilon=checkpoint["time_epsilon"],
        )
        solution = process.sample(
            model,
            initial_noise=initial_noise,
            solver=SOLVER,
            num_steps=NUM_ODE_STEPS,
            snapshot_count=2,
        )
        image_groups.append(solution.final)
        labels.append(f"{actual_type} ({solution.nfe} NFE)")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_comparison_rows(image_groups, labels, OUTPUT_PATH, count=NUM_IMAGES)
    print("预测目标对比图保存在", OUTPUT_PATH)


if __name__ == "__main__":
    main()
