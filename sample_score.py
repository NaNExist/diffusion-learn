"""加载 score model，并用退火 Langevin 动力学生成图片。"""

from pathlib import Path

import torch

from diffusion import ScoreProcess
from model import TinyUNet
from visualize import save_image_grid, save_trajectory


# ===== 可以手动修改的配置（训练参数会自动从权重读取） =====
CHECKPOINT = "outputs/score/checkpoint.pt"  # train_score.py 产生的权重
OUTPUT_DIR = "outputs/score"                # 最终网格和 Langevin 轨迹保存目录
NUM_IMAGES = 16                             # 一次生成数量；默认排成 4x4
STEPS_PER_LEVEL = 30                        # 每个 sigma 更新次数；越大越慢、通常更充分
BASE_STEP_SIZE = 2e-5                       # 最小 sigma 对应步长；发散时调小
SEED = 123                                  # 改变初始噪声，从而生成另一批图片
# ============================


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = TinyUNet(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    process = ScoreProcess(
        num_noise_levels=checkpoint["num_noise_levels"],
        sigma_max=checkpoint["sigma_max"],
        sigma_min=checkpoint["sigma_min"],
        device=device,
    )

    # 采样总网络调用次数约为 NUM_NOISE_LEVELS * STEPS_PER_LEVEL。
    images, snapshots = process.sample(
        model,
        num_images=NUM_IMAGES,
        steps_per_level=STEPS_PER_LEVEL,
        base_step_size=BASE_STEP_SIZE,
    )
    output_dir = Path(OUTPUT_DIR)
    save_image_grid(images, str(output_dir / "samples.png"), nrow=4, title="Score samples")
    save_trajectory(snapshots, str(output_dir / "langevin_process.png"), title="Annealed Langevin")
    print("生成结果保存在", output_dir)


if __name__ == "__main__":
    main()
