"""供连续扩散、Flow Matching 和 Diffusion Bridge 共用的 ODE 求解器。

求解器只关心微分方程 ``dx/dt = f(x, t)``，完全不知道网络预测的是
epsilon、score 还是 velocity。把“向量场从哪里来”和“如何沿向量场前进”
分开后，同一个模型就可以公平地比较 Euler、Heun 与 RK4。
"""

from dataclasses import dataclass
import time
from typing import Callable, List

import torch
from tqdm.auto import tqdm


# rhs 接收当前状态 [B,C,H,W] 和每个样本的时间 [B]，返回同形状的 dx/dt。
ODEFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ODESolution:
    """一次 ODE 采样的结果以及用于公平比较的统计量。"""

    final: torch.Tensor
    snapshots: List[torch.Tensor]
    nfe: int  # number of function evaluations，即调用神经网络的总次数
    elapsed_seconds: float


def _batch_time(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """把标量时间扩成 [batch]，与 TinyUNet 的时间条件接口保持一致。"""
    return value.expand(x.shape[0])


@torch.no_grad()
def solve_ode(
    rhs: ODEFunction,
    initial_state: torch.Tensor,
    t_start: float,
    t_end: float,
    num_steps: int,
    method: str = "heun",
    snapshot_count: int = 8,
    description: str = "ODE sampling",
) -> ODESolution:
    """使用固定步长求解 ODE。

    ``t_end`` 可以小于 ``t_start``，所以同一实现既支持 Flow Matching 的
    0→1 积分，也支持 probability-flow ODE 的 1→0 反向积分。

    三种方法每一步的网络调用次数分别是 1、2、4。比较算法时应优先让
    NFE 相同，而不是让 num_steps 相同。
    """
    method = method.lower()
    if method not in {"euler", "heun", "rk4"}:
        raise ValueError(f"未知 ODE 求解器 {method!r}，可选 euler/heun/rk4")
    if num_steps < 1:
        raise ValueError("num_steps 必须至少为 1")

    x = initial_state.clone()
    times = torch.linspace(t_start, t_end, num_steps + 1, device=x.device, dtype=x.dtype)
    # linspace 会保证最后一个时间恰好等于 t_end；固定步长时每段 dt 相同。
    save_indices = {
        round(i * num_steps / max(snapshot_count - 1, 1))
        for i in range(max(snapshot_count, 2))
    }
    snapshots = [x.detach().cpu()]
    nfe = 0

    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = time.perf_counter()

    progress = tqdm(range(num_steps), desc=description, dynamic_ncols=True)
    for index in progress:
        t = times[index]
        next_t = times[index + 1]
        dt = next_t - t
        batch_t = _batch_time(x, t)

        if method == "euler":
            # 一阶显式 Euler：便宜，但大步长时离散误差较明显。
            x = x + dt * rhs(x, batch_t)
            nfe += 1
        elif method == "heun":
            # Heun（改进 Euler）先预测终点，再平均起点和终点的斜率。
            k1 = rhs(x, batch_t)
            predicted = x + dt * k1
            k2 = rhs(predicted, _batch_time(x, next_t))
            x = x + 0.5 * dt * (k1 + k2)
            nfe += 2
        else:
            # 经典四阶 Runge-Kutta。精度高，但每步需要四次网络前向。
            half_t = t + 0.5 * dt
            k1 = rhs(x, batch_t)
            k2 = rhs(x + 0.5 * dt * k1, _batch_time(x, half_t))
            k3 = rhs(x + 0.5 * dt * k2, _batch_time(x, half_t))
            k4 = rhs(x + dt * k3, _batch_time(x, next_t))
            x = x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            nfe += 4

        if index + 1 in save_indices:
            snapshots.append(x.detach().cpu())

    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    return ODESolution(x, snapshots, nfe, elapsed)
