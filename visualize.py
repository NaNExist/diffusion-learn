"""独立的可视化工具。

所有画图和保存图片的逻辑都放在这里，model.py 与 diffusion.py 不依赖
matplotlib。输入统一是训练范围 [-1, 1] 内的图片张量。
"""

import os
from pathlib import Path
from typing import List, Sequence

# 某些服务器的用户配置目录只读；使用项目内缓存可避免 matplotlib 警告。
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib

# 服务器一般没有图形桌面，Agg 后端可以直接把图片写入文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    """训练图片在 [-1, 1]，画图前转换到 [0, 1]。"""
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0, 1)


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4, title: str = "") -> None:
    """把一个 batch 排成网格；nrow 表示每行放多少张。"""
    path_object = Path(path)
    path_object.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(_to_display_range(images), nrow=nrow, padding=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path_object, dpi=160, bbox_inches="tight")
    plt.close()


def save_trajectory(
    snapshots: Sequence[torch.Tensor],
    path: str,
    sample_index: int = 0,
    title: str = "Denoising trajectory",
) -> None:
    """从每个快照取同一张图片，横向显示生成过程。"""
    selected = torch.stack([snapshot[sample_index] for snapshot in snapshots])
    save_image_grid(selected, path, nrow=len(selected), title=title)


def save_loss_curve(losses: List[float], path: str, title: str) -> None:
    """保存逐 iteration 的训练损失曲线。"""
    path_object = Path(path)
    path_object.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_object, dpi=160)
    plt.close()


def save_comparison(
    ddpm_images: torch.Tensor,
    score_images: torch.Tensor,
    path: str,
    count: int = 8,
) -> None:
    """第一行显示 DDPM，第二行显示 score model。"""
    images = torch.cat([ddpm_images[:count], score_images[:count]], dim=0)
    save_image_grid(images, path, nrow=count, title="DDPM (top) / Score model (bottom)")


def save_comparison_rows(
    image_groups: Sequence[torch.Tensor],
    labels: Sequence[str],
    path: str,
    count: int = 8,
) -> None:
    """把任意数量的方法按行排列，适合 solver、预测目标和 bridge 对比。

    每个 group 是一批 [B,C,H,W] 图片。所有行必须至少有 ``count`` 张图；
    行名写在最左侧，避免仅靠文件名猜测每一行代表哪个实验。
    """
    if len(image_groups) != len(labels):
        raise ValueError("image_groups 与 labels 的长度必须相同")
    if not image_groups:
        raise ValueError("至少需要一组图片")

    path_object = Path(path)
    path_object.parent.mkdir(parents=True, exist_ok=True)
    rows = len(image_groups)
    figure, axes = plt.subplots(rows, count, figsize=(1.25 * count, 1.45 * rows), squeeze=False)
    for row, (images, label) in enumerate(zip(image_groups, labels)):
        display_images = _to_display_range(images[:count])
        for column in range(count):
            image = display_images[column].permute(1, 2, 0).squeeze()
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        # axis("off") 会连 ylabel 一起隐藏，因此把行名作为独立文本 artist
        # 放在第一张图左侧；bbox_inches="tight" 会保留画布外的这部分文字。
        axes[row, 0].text(
            -0.10,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
        )
    plt.tight_layout()
    plt.savefig(path_object, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_forward_noising(clean_image: torch.Tensor, ddpm, path: str, count: int = 8) -> None:
    """显示同一张真实图片在不同时间步的前向加噪结果。"""
    clean_batch = clean_image[:1].to(ddpm.device).repeat(count, 1, 1, 1)
    # 在 0 到 T-1 之间等间隔取 count 个时间步。
    timesteps = torch.linspace(0, ddpm.num_steps - 1, count, device=ddpm.device).long()
    # 复用同一个基础噪声，更容易看清噪声强度的连续变化。
    noise = torch.randn_like(clean_batch[:1]).repeat(count, 1, 1, 1)
    noisy_images, _ = ddpm.add_noise(clean_batch, timesteps, noise)
    save_image_grid(noisy_images, path, nrow=count, title="Forward noising: clean to noise")


def main() -> None:
    """直接运行本文件，生成一张 MNIST 前向加噪过程图。"""
    # ===== 此处是 visualize.py 独立运行时可修改的配置 =====
    data_dir = "data"                         # MNIST 所在目录
    output_path = "outputs/forward_noising.png" # 输出图片位置
    diffusion_steps = 256                     # 应和想观察的 DDPM 配置一致
    beta_start = 1e-4                         # DDPM 起始 beta
    beta_end = 0.05                           # 200 步配置对应的结束 beta
    frame_count = 8                           # 横向展示多少个噪声阶段
    sample_index = 0                          # 选择 MNIST 中第几张图片
    # =========================================================

    # 放在函数内部导入，保证其他脚本使用画图函数时不会额外加载数据集。
    from torchvision import datasets, transforms
    from diffusion import DDPM

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    clean_image, label = dataset[sample_index]
    process = DDPM(
        num_steps=diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    save_forward_noising(clean_image.unsqueeze(0), process, output_path, count=frame_count)
    print(f"数字标签为 {label}，前向加噪图保存在 {output_path}")


if __name__ == "__main__":
    main()
