"""加载 DDPM 权重并生成图片。配置直接在文件顶部修改。"""

from pathlib import Path

import torch

from diffusion import DDPM
from model import TinyUNet
from visualize import save_image_grid, save_trajectory


# ===== 可以手动修改的配置（训练参数会自动从权重读取） =====
CHECKPOINT = "outputs/ddpm/checkpoint.pt"  # train_ddpm.py 产生的权重
OUTPUT_DIR = "outputs/ddpm"                # 最终网格和去噪轨迹的保存目录
NUM_IMAGES = 16                            # 一次生成图片数；默认正好排成 4x4
SNAPSHOT_COUNT = 8                         # 轨迹中大致保留几帧；不影响采样结果
SEED = 123                                 # 改这个数字即可换一批初始随机噪声
# ============================


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = TinyUNet(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    # beta 等参数必须与训练时完全相同，所以从 checkpoint 读取而不是重复硬编码。
    process = DDPM(
        num_steps=checkpoint["num_steps"],
        beta_start=checkpoint["beta_start"],
        beta_end=checkpoint["beta_end"],
        device=device,
    )

    images, snapshots = process.sample(
        model, num_images=NUM_IMAGES, snapshot_count=SNAPSHOT_COUNT
    )
    output_dir = Path(OUTPUT_DIR)
    save_image_grid(images, str(output_dir / "samples.png"), nrow=4, title="DDPM samples")
    save_trajectory(snapshots, str(output_dir / "denoise_process.png"), title="DDPM denoising")
    print("生成结果保存在", output_dir)


if __name__ == "__main__":
    main()
