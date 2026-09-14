"""连续时间生成过程：VP probability-flow ODE、Flow Matching 与数字桥。

三个过程共用 :mod:`ode_solvers`，但训练目标不同：

* ContinuousVPProcess 学习 epsilon/x0/v/score 中的一种参数化；
* FlowMatchingProcess 学习从高斯噪声直达数据的速度；
* DiffusionBridgeProcess 学习两个指定数字类别之间的带噪桥速度。
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ode_solvers import ODESolution, solve_ode


def append_dims(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """把 [B] 标量扩为 [B,1,1,1]，从而能逐样本乘在图像上。"""
    return values.reshape(values.shape[0], *((1,) * (target.ndim - 1)))


class VPSchedule:
    """连续时间 variance-preserving (VP) 噪声日程。

    beta(t) 在 [0,1] 上线性变化。对应的边缘分布有解析形式：
    x_t = alpha(t) x_0 + sigma(t) epsilon，且 alpha² + sigma² = 1。
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        if not 0.0 < beta_min <= beta_max:
            raise ValueError("需要满足 0 < beta_min <= beta_max")
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # integral_0^t beta(s) ds = beta_min*t + (beta_max-beta_min)*t²/2。
        integrated_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()
        alpha = torch.exp(-0.5 * integrated_beta)
        sigma = (1.0 - alpha.square()).clamp_min(1e-12).sqrt()
        return alpha, sigma


class ContinuousVPProcess:
    """连续 VP 扩散，可用多种预测目标训练并通过 ODE 确定性采样。"""

    VALID_PREDICTIONS = {"epsilon", "x0", "v", "score"}

    def __init__(
        self,
        prediction_type: str = "epsilon",
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        time_epsilon: float = 1e-3,
    ):
        if prediction_type not in self.VALID_PREDICTIONS:
            raise ValueError(
                f"prediction_type={prediction_type!r} 不支持；"
                f"请选择 {sorted(self.VALID_PREDICTIONS)}"
            )
        if not 0.0 < time_epsilon < 1.0:
            raise ValueError("time_epsilon 必须在 (0, 1) 内")
        self.prediction_type = prediction_type
        self.schedule = VPSchedule(beta_min, beta_max)
        # t=0 时 sigma=0，epsilon/x0/score 之间的转换会出现除零；训练和采样
        # 都避开一个极小的端点区域，对 28x28 MNIST 几乎没有可见影响。
        self.time_epsilon = time_epsilon

    def training_loss(self, model, clean_images: torch.Tensor) -> torch.Tensor:
        batch_size = clean_images.shape[0]
        t = torch.rand(batch_size, device=clean_images.device)
        t = self.time_epsilon + (1.0 - self.time_epsilon) * t
        alpha, sigma = self.schedule.alpha_sigma(t)
        alpha_image = append_dims(alpha, clean_images)
        sigma_image = append_dims(sigma, clean_images)
        noise = torch.randn_like(clean_images)
        noisy = alpha_image * clean_images + sigma_image * noise

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = clean_images
        elif self.prediction_type == "v":
            # Imagen/Stable Diffusion 常见的 v 参数化：
            # v = alpha*epsilon - sigma*x0。
            target = alpha_image * noise - sigma_image * clean_images
        else:
            target = -noise / sigma_image

        prediction = model(noisy, t)
        per_example = (prediction - target).square().flatten(1).mean(1)
        if self.prediction_type == "score":
            # score 在小 sigma 时尺度约为 1/sigma；乘 sigma² 后，各噪声等级
            # 对 loss 的影响更均衡，也与 epsilon prediction 的权重接近。
            per_example = per_example * sigma.square()
        return per_example.mean()

    def output_to_score(
        self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """把任意训练参数化统一换成 probability-flow ODE 所需的 score。"""
        alpha, sigma = self.schedule.alpha_sigma(t)
        alpha = append_dims(alpha, x_t)
        sigma = append_dims(sigma, x_t).clamp_min(1e-6)

        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "x0":
            epsilon = (x_t - alpha * model_output) / sigma
        elif self.prediction_type == "v":
            # 由 [x_t, v] 到 [x0, epsilon] 是一个正交旋转的逆变换。
            epsilon = alpha * model_output + sigma * x_t
        else:
            return model_output
        return -epsilon / sigma

    def probability_flow_rhs(self, model, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """VP SDE 对应的 probability-flow ODE 向量场。

        正向公式为 dx/dt = -beta*x/2 - beta*score/2。生成时求解器从
        t=1 积分到 t≈0，负的 dt 会自动沿反方向去噪。
        """
        score = self.output_to_score(model(x, t), x, t)
        beta = append_dims(self.schedule.beta(t), x)
        return -0.5 * beta * (x + score)

    @torch.no_grad()
    def sample(
        self,
        model,
        num_images: int = 16,
        image_size: int = 28,
        initial_noise: Optional[torch.Tensor] = None,
        solver: str = "heun",
        num_steps: int = 40,
        snapshot_count: int = 8,
    ) -> ODESolution:
        model.eval()
        if initial_noise is None:
            device = next(model.parameters()).device
            initial_noise = torch.randn(num_images, 1, image_size, image_size, device=device)
        solution = solve_ode(
            lambda x, t: self.probability_flow_rhs(model, x, t),
            initial_noise,
            t_start=1.0,
            t_end=self.time_epsilon,
            num_steps=num_steps,
            method=solver,
            snapshot_count=snapshot_count,
            description=f"VP ODE ({solver})",
        )
        solution.final = solution.final.clamp(-1, 1)
        return solution


class FlowMatchingProcess:
    """最基础的 conditional flow matching：学习 noise→data 的直线路径。"""

    def training_loss(self, model, clean_images: torch.Tensor) -> torch.Tensor:
        batch_size = clean_images.shape[0]
        t = torch.rand(batch_size, device=clean_images.device)
        noise = torch.randn_like(clean_images)
        t_image = append_dims(t, clean_images)
        # t=0 是高斯噪声，t=1 是数据；条件速度沿一条直线且与 t 无关。
        x_t = (1.0 - t_image) * noise + t_image * clean_images
        target_velocity = clean_images - noise
        return F.mse_loss(model(x_t, t), target_velocity)

    @torch.no_grad()
    def sample(
        self,
        model,
        num_images: int = 16,
        image_size: int = 28,
        initial_noise: Optional[torch.Tensor] = None,
        solver: str = "heun",
        num_steps: int = 20,
        snapshot_count: int = 8,
    ) -> ODESolution:
        model.eval()
        if initial_noise is None:
            device = next(model.parameters()).device
            initial_noise = torch.randn(num_images, 1, image_size, image_size, device=device)
        solution = solve_ode(
            lambda x, t: model(x, t),
            initial_noise,
            t_start=0.0,
            t_end=1.0,
            num_steps=num_steps,
            method=solver,
            snapshot_count=snapshot_count,
            description=f"Flow Matching ({solver})",
        )
        solution.final = solution.final.clamp(-1, 1)
        return solution


class DiffusionBridgeProcess:
    """两个图像端点之间的教学版 stochastic-interpolant bridge。

    路径在 t=0/1 严格经过 source/target，中间加入
    sigma_bridge*sqrt(t(1-t))*epsilon。网络同时看到当前 x_t 与 source，
    学习条件速度场。它展示了 diffusion bridge 的端点约束思想，但不是需要
    迭代拟合两个势函数的完整 Schrödinger Bridge。
    """

    def __init__(self, bridge_sigma: float = 0.5, time_epsilon: float = 1e-3):
        if bridge_sigma < 0.0:
            raise ValueError("bridge_sigma 不能为负数")
        if not 0.0 < time_epsilon < 0.5:
            raise ValueError("bridge 的 time_epsilon 必须在 (0, 0.5) 内")
        self.bridge_sigma = bridge_sigma
        self.time_epsilon = time_epsilon

    def training_loss(
        self, model, source_images: torch.Tensor, target_images: torch.Tensor
    ) -> torch.Tensor:
        batch_size = source_images.shape[0]
        t = torch.rand(batch_size, device=source_images.device)
        t = self.time_epsilon + (1.0 - 2.0 * self.time_epsilon) * t
        t_image = append_dims(t, source_images)
        noise = torch.randn_like(source_images)

        root = (t * (1.0 - t)).clamp_min(1e-8).sqrt()
        bridge_std = append_dims(self.bridge_sigma * root, source_images)
        x_t = (1.0 - t_image) * source_images + t_image * target_images + bridge_std * noise

        # 对 x_t 关于 t 求导。随机项的导数在端点附近会变大，所以训练时间
        # 避开精确的 0 和 1；time_epsilon 控制这段很小的安全边界。
        bridge_std_derivative = self.bridge_sigma * (1.0 - 2.0 * t) / (2.0 * root)
        target_velocity = (
            target_images
            - source_images
            + append_dims(bridge_std_derivative, source_images) * noise
        )
        predicted_velocity = model(x_t, t, spatial_condition=source_images)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        model,
        source_images: torch.Tensor,
        solver: str = "heun",
        num_steps: int = 30,
        snapshot_count: int = 8,
    ) -> ODESolution:
        model.eval()
        solution = solve_ode(
            lambda x, t: model(x, t, spatial_condition=source_images),
            source_images,
            t_start=self.time_epsilon,
            t_end=1.0 - self.time_epsilon,
            num_steps=num_steps,
            method=solver,
            snapshot_count=snapshot_count,
            description=f"Diffusion Bridge ({solver})",
        )
        solution.final = solution.final.clamp(-1, 1)
        return solution
