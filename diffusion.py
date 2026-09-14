"""DDPM 与 score-based 模型各自的数学过程。"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """从长度为 T 的表中取出一个 batch 对应的值，并扩维以便和图片运算。"""
    result = values.gather(0, timesteps)
    return result.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class DDPM:
    """离散时间 DDPM：训练时预测噪声，采样时逐步反推。

    beta_t 决定第 t 步加入多少新噪声；alpha_t = 1 - beta_t；
    alpha_bar_t 是从第 0 步到第 t 步所有 alpha 的累积乘积。
    """

    def __init__(
        self,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_steps = num_steps
        self.device = device
        # 本项目只使用 200 步，因此 beta_end 默认比经典 1000 步 DDPM 稍大，
        # 使最后一步仍能接近纯噪声。若把 num_steps 改为 1000，可用 0.02。
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_previous = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        # q(x_{t-1} | x_t, x_0) 的方差；反向采样时用到。
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_previous) / (1.0 - self.alpha_bars)
        )

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """直接从 x_0 得到任意 x_t，不必真的逐步加噪。"""
        if noise is None:
            noise = torch.randn_like(clean_images)
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
        sqrt_alpha_bar = extract(self.alpha_bars.sqrt(), timesteps, clean_images.shape)
        sqrt_one_minus = extract((1.0 - self.alpha_bars).sqrt(), timesteps, clean_images.shape)
        noisy_images = sqrt_alpha_bar * clean_images + sqrt_one_minus * noise
        return noisy_images, noise

    def training_loss(self, model, clean_images: torch.Tensor) -> torch.Tensor:
        """随机选择 t、构造 x_t，让模型预测本次实际加入的 epsilon。"""
        batch_size = clean_images.shape[0]
        # 每张图片独立选择时间步，因此一个 batch 会同时包含多种噪声强度。
        timesteps = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        noisy_images, real_noise = self.add_noise(clean_images, timesteps)
        time_condition = timesteps.float() / (self.num_steps - 1)
        predicted_noise = model(noisy_images, time_condition)
        # DDPM 最常用的简化目标：预测噪声与真实噪声的均方误差。
        return F.mse_loss(predicted_noise, real_noise)

    @torch.no_grad()
    def sample(
        self,
        model,
        num_images: int = 16,
        image_size: int = 28,
        initial_noise: Optional[torch.Tensor] = None,
        snapshot_count: int = 8,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """从标准高斯噪声开始执行 DDPM 祖先采样。"""
        model.eval()
        x = (
            initial_noise.clone().to(self.device)
            if initial_noise is not None
            else torch.randn(num_images, 1, image_size, image_size, device=self.device)
        )
        # 快照立即移到 CPU，防止为了画图长期占用 GPU 显存。
        snapshots = [x.detach().cpu()]
        save_every = max(self.num_steps // max(snapshot_count - 1, 1), 1)

        # tqdm 的 total 需要显式给出，因为 reversed(range(...)) 本身没有长度信息。
        reverse_steps = tqdm(
            reversed(range(self.num_steps)),
            total=self.num_steps,
            desc="DDPM 逐步去噪",
            dynamic_ncols=True,
        )
        for step in reverse_steps:
            timesteps = torch.full((x.shape[0],), step, device=self.device, dtype=torch.long)
            condition = timesteps.float() / (self.num_steps - 1)
            predicted_noise = model(x, condition)

            beta = extract(self.betas, timesteps, x.shape)
            alpha = extract(self.alphas, timesteps, x.shape)
            alpha_bar = extract(self.alpha_bars, timesteps, x.shape)
            # mu_theta = 1/sqrt(alpha_t) *
            #            (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
            mean = (x - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()

            if step > 0:
                variance = extract(self.posterior_variance, timesteps, x.shape)
                # 除最后一步外还要按后验方差加入随机项，因而同一模型可生成不同图片。
                x = mean + variance.sqrt() * torch.randn_like(x)
            else:
                x = mean

            if step == 0 or step % save_every == 0:
                snapshots.append(x.detach().cpu())

        return x.clamp(-1, 1), snapshots

    def noise_to_score(self, predicted_noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """展示 DDPM 噪声预测与 score 之间的关系。"""
        scale = extract((1.0 - self.alpha_bars).sqrt(), timesteps, predicted_noise.shape)
        return -predicted_noise / scale.clamp_min(1e-8)


class ScoreProcess:
    """NCSN 风格的得分匹配与退火 Langevin 采样。

    score 定义为 grad_x log p_sigma(x)，指向当前噪声分布中概率密度增加
    最快的方向。这里使用有限个 sigma 等级，是连续 score/SDE 模型的简化入门版。
    """

    def __init__(
        self,
        num_noise_levels: int = 20,
        sigma_max: float = 1.0,
        sigma_min: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        # 几何序列比线性序列更均匀地覆盖不同数量级的噪声。
        self.sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(sigma_max)),
                torch.log(torch.tensor(sigma_min)),
                num_noise_levels,
                device=device,
            )
        )

    def condition_from_levels(self, levels: torch.Tensor) -> torch.Tensor:
        """把整数噪声等级缩放到 [0, 1]，作为 U-Net 的条件输入。"""
        return levels.float() / max(len(self.sigmas) - 1, 1)

    def training_loss(self, model, clean_images: torch.Tensor) -> torch.Tensor:
        """去噪得分匹配：目标 score 等于 -noise / sigma。"""
        batch_size = clean_images.shape[0]
        levels = torch.randint(0, len(self.sigmas), (batch_size,), device=self.device)
        sigma = extract(self.sigmas, levels, clean_images.shape)
        noise = torch.randn_like(clean_images)
        # score 模型常用 VE（variance exploding）形式：x_noisy = x_0 + sigma*z。
        noisy_images = clean_images + sigma * noise

        predicted_score = model(noisy_images, self.condition_from_levels(levels))
        # 对高斯扰动核求 log 概率关于 x_noisy 的梯度，可得 -z/sigma。
        target_score = -noise / sigma
        # 乘 sigma^2，避免小 sigma 的巨大目标值完全主导训练。
        per_example = ((predicted_score - target_score) ** 2).flatten(1).mean(1)
        return (per_example * sigma.flatten().square()).mean()

    @torch.no_grad()
    def sample(
        self,
        model,
        num_images: int = 16,
        image_size: int = 28,
        steps_per_level: int = 30,
        base_step_size: float = 2e-5,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """从大噪声到小噪声，逐级执行 Langevin 动力学。"""
        model.eval()
        x = (
            initial_noise.clone().to(self.device) * self.sigmas[0]
            if initial_noise is not None
            else torch.randn(num_images, 1, image_size, image_size, device=self.device)
            * self.sigmas[0]
        )
        snapshots = [x.detach().cpu()]
        sigma_min = self.sigmas[-1]

        # 外层进度表示当前 sigma 等级；每个等级内部还会更新 steps_per_level 次。
        noise_levels = tqdm(
            self.sigmas,
            total=len(self.sigmas),
            desc="Score Langevin 采样",
            dynamic_ncols=True,
        )
        for level, sigma in enumerate(noise_levels):
            levels = torch.full((x.shape[0],), level, device=self.device, dtype=torch.long)
            condition = self.condition_from_levels(levels)
            # 大 sigma 时允许迈大步，小 sigma 时用小步精修。
            step_size = base_step_size * (sigma / sigma_min) ** 2
            for _ in range(steps_per_level):
                score = model(x, condition)
                # Langevin 更新 = 沿 score 上升 + 少量随机扰动。
                x = x + step_size * score + (2.0 * step_size).sqrt() * torch.randn_like(x)
            snapshots.append(x.detach().cpu())

        return x.clamp(-1, 1), snapshots
