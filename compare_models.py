"""使用相同初始噪声，对比已经训练好的 DDPM 和 score model。"""

from pathlib import Path

import torch

from diffusion import DDPM, ScoreProcess
from model import TinyUNet
from visualize import save_comparison


# ===== 可以手动修改的配置 =====
DDPM_CHECKPOINT = "outputs/ddpm/checkpoint.pt"    # 已训练的 DDPM 权重
SCORE_CHECKPOINT = "outputs/score/checkpoint.pt" # 已训练的 score model 权重
OUTPUT_PATH = "outputs/comparison.png"           # 两行对比图保存位置
NUM_IMAGES = 8                                    # 每种模型展示的图片数量
SCORE_STEPS_PER_LEVEL = 30                        # score 分支每个 sigma 的更新次数
SCORE_BASE_STEP_SIZE = 2e-5                       # score 分支的基础 Langevin 步长
SEED = 123                                        # 两种模型共享由此产生的初始噪声
# ============================


def load_model(checkpoint: dict, device: torch.device) -> TinyUNet:
    model = TinyUNet(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 公平起见，两种方法从完全相同的高斯噪声张量开始。
    initial_noise = torch.randn(NUM_IMAGES, 1, 28, 28, device=device)

    ddpm_checkpoint = torch.load(DDPM_CHECKPOINT, map_location=device, weights_only=True)
    ddpm_model = load_model(ddpm_checkpoint, device)
    ddpm = DDPM(
        num_steps=ddpm_checkpoint["num_steps"],
        beta_start=ddpm_checkpoint["beta_start"],
        beta_end=ddpm_checkpoint["beta_end"],
        device=device,
    )
    ddpm_images, _ = ddpm.sample(ddpm_model, initial_noise=initial_noise)

    score_checkpoint = torch.load(SCORE_CHECKPOINT, map_location=device, weights_only=True)
    score_model = load_model(score_checkpoint, device)
    score_process = ScoreProcess(
        num_noise_levels=score_checkpoint["num_noise_levels"],
        sigma_max=score_checkpoint["sigma_max"],
        sigma_min=score_checkpoint["sigma_min"],
        device=device,
    )
    score_images, _ = score_process.sample(
        score_model,
        initial_noise=initial_noise,
        steps_per_level=SCORE_STEPS_PER_LEVEL,
        base_step_size=SCORE_BASE_STEP_SIZE,
    )

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_comparison(ddpm_images, score_images, OUTPUT_PATH, count=NUM_IMAGES)
    print("对比图保存在", OUTPUT_PATH)


if __name__ == "__main__":
    main()
