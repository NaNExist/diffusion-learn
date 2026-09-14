"""在相同初始状态和近似相同 NFE 下比较 Euler、Heun、RK4。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from continuous import ContinuousVPProcess, DiffusionBridgeProcess, FlowMatchingProcess
from model import TinyUNet
from paired_mnist import PairedMNIST
from visualize import save_comparison_rows


# ===== 可以手动修改的配置 =====
CHECKPOINT = "outputs/flow_matching/checkpoint.pt"
OUTPUT_DIR = "outputs/solver_comparison"
DATA_DIR = "data"               # 只有比较 diffusion_bridge 时会读取测试集
NUM_IMAGES = 8
# 每个 solver 的单步 NFE 分别为 1、2、4，所以下面三项总 NFE 都是 40。
SOLVER_STEPS = {"euler": 40, "heun": 20, "rk4": 10}
SEED = 123
# ============================


def make_process(checkpoint: dict):
    """从 checkpoint 的 method 字段恢复正确的连续过程。"""
    if checkpoint["method"] == "flow_matching":
        return FlowMatchingProcess()
    if checkpoint["method"] == "continuous_vp":
        return ContinuousVPProcess(
            prediction_type=checkpoint["prediction_type"],
            beta_min=checkpoint["beta_min"],
            beta_max=checkpoint["beta_max"],
            time_epsilon=checkpoint["time_epsilon"],
        )
    if checkpoint["method"] == "diffusion_bridge":
        return DiffusionBridgeProcess(
            bridge_sigma=checkpoint["bridge_sigma"],
            time_epsilon=checkpoint["time_epsilon"],
        )
    raise ValueError("checkpoint 的 method 不是 continuous_vp/flow_matching/diffusion_bridge")


def load_bridge_endpoints(checkpoint: dict, device: torch.device):
    """solver 对比若选择 bridge checkpoint，就从测试集加载相同的源/目标端点。"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    paired = PairedMNIST(
        mnist,
        checkpoint["source_digit"],
        checkpoint["target_digit"],
        pair_seed=SEED,
    )
    source, target = next(iter(DataLoader(paired, batch_size=NUM_IMAGES)))
    return source.to(device), target


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    is_bridge = checkpoint["method"] == "diffusion_bridge"
    model = TinyUNet(
        base_channels=checkpoint["base_channels"],
        condition_channels=1 if is_bridge else 0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    process = make_process(checkpoint)

    # 三个 solver 必须从逐像素完全相同的初始状态出发，差异才只来自数值积分。
    if is_bridge:
        initial_state, paired_target = load_bridge_endpoints(checkpoint, device)
    else:
        initial_state = torch.randn(NUM_IMAGES, 1, 28, 28, device=device)
        paired_target = None

    # 第一次 CUDA 前向通常还包含 kernel 初始化开销。正式计时前预热一次，
    # 避免总是排在第一行的 Euler 无端承担这部分时间。
    model.eval()
    with torch.no_grad():
        warmup_t = torch.full((NUM_IMAGES,), 0.5, device=device)
        if is_bridge:
            model(initial_state, warmup_t, spatial_condition=initial_state)
        else:
            model(initial_state, warmup_t)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    images = []
    labels = []
    report_lines = [f"checkpoint: {CHECKPOINT}", f"method: {checkpoint['method']}"]
    solutions = {}

    for solver, steps in SOLVER_STEPS.items():
        if is_bridge:
            solution = process.sample(
                model,
                source_images=initial_state,
                solver=solver,
                num_steps=steps,
                snapshot_count=2,
            )
        else:
            solution = process.sample(
                model,
                initial_noise=initial_state,
                solver=solver,
                num_steps=steps,
                snapshot_count=2,
            )
        solutions[solver] = solution
        images.append(solution.final)
        labels.append(f"{solver} ({solution.nfe} NFE)")
        report_lines.append(
            f"{solver:5s} steps={steps:3d} NFE={solution.nfe:3d} "
            f"time={solution.elapsed_seconds:.4f}s "
            f"mean={solution.final.mean().item():+.4f} "
            f"std={solution.final.std().item():.4f}"
        )

    # RK4 不是“真实答案”，这里只把它当作同 NFE 下的数值参照，衡量输出
    # 对积分方法有多敏感；生成质量仍应以对比图或额外指标判断。
    if "rk4" in solutions:
        rk4_images = solutions["rk4"].final
        report_lines.append("MSE relative to equal-NFE RK4 output:")
        for solver, solution in solutions.items():
            mse = (solution.final - rk4_images).square().mean().item()
            report_lines.append(f"  {solver:5s}: {mse:.8f}")

    if is_bridge:
        # 在 solver 行之外保留相同的两个端点，方便观察 0→1 等类别转换。
        images = [initial_state, *images, paired_target]
        labels = [
            f"source {checkpoint['source_digit']}",
            *labels,
            f"paired {checkpoint['target_digit']}",
        ]

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_rows(
        images,
        labels,
        str(output_dir / "solver_comparison.png"),
        count=NUM_IMAGES,
    )
    report_path = output_dir / "solver_stats.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print("对比结果保存在", output_dir)


if __name__ == "__main__":
    main()
