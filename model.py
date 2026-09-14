"""两种生成方法共用的小型 U-Net。

网络只负责回答一个问题：给定带噪图片和噪声程度，应当输出什么？
在 DDPM 中输出是噪声 epsilon；在 score model 中输出是 score。
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """把一个标量条件（时间或噪声等级）编码成向量。

    卷积层不能直接理解“当前是第几个扩散步骤”。正弦编码用不同频率的
    sin/cos 把一个数字展开成向量，使网络能区分不同噪声阶段。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # value 的形状为 [batch]，通常已经缩放到 0~1。
        half_dim = self.dim // 2
        # frequencies 覆盖从高频到低频，让编码既能表示局部差异，也能表示大范围变化。
        frequencies = torch.exp(
            torch.arange(half_dim, device=value.device)
            * (-math.log(10000) / max(half_dim - 1, 1))
        )
        angles = value[:, None] * 1000.0 * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """带条件信息的残差卷积块。

    普通卷积块只看到图片；这里额外把 condition 投影成每个通道的偏置，
    所以同一张图片处于不同噪声等级时，网络可以采取不同的处理方式。
    """

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.condition_projection = nn.Linear(condition_dim, out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]；condition: [B, condition_dim]。
        h = F.silu(self.norm1(self.conv1(x)))
        # 每个通道加上对应的时间/噪声条件。
        h = h + self.condition_projection(condition)[:, :, None, None]
        h = self.norm2(self.conv2(F.silu(h)))
        # shortcut 是残差连接，有助于深层网络稳定训练。
        return F.silu(h + self.shortcut(x))


class TinyUNet(nn.Module):
    """用于 28x28 单通道 MNIST 的小型 U-Net。

    编码器把空间尺寸从 28 -> 14 -> 7，解码器再恢复到 28。
    skip connection 把编码器中的局部细节直接传给解码器，这对预测
    每个像素上的噪声/score 很重要。

    Args:
        image_channels: MNIST 是灰度图，所以为 1；RGB 数据应改为 3。
        base_channels: 网络基础宽度。越大容量越强，但显存和计算量也越大。
        condition_channels: 额外空间条件的通道数。普通生成设为 0；条件 bridge
            把源图像作为一个额外通道传入，因此设为 1。
    """

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 32,
        condition_channels: int = 0,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.condition_channels = condition_channels
        condition_dim = base_channels * 4
        self.condition_mlp = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # 普通扩散模型只输入当前状态 x_t。条件 bridge 还会把源图像沿通道维
        # 拼在 x_t 后面，因此输入通道数是二者之和；网络仍只输出 image_channels。
        self.input_conv = nn.Conv2d(
            image_channels + condition_channels, base_channels, 3, padding=1
        )
        self.block_28 = ResidualBlock(base_channels, base_channels, condition_dim)
        self.down_14 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.block_14 = ResidualBlock(base_channels * 2, base_channels * 2, condition_dim)
        self.down_7 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)

        self.middle_1 = ResidualBlock(base_channels * 4, base_channels * 4, condition_dim)
        self.middle_2 = ResidualBlock(base_channels * 4, base_channels * 4, condition_dim)

        self.up_14 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up_block_14 = ResidualBlock(base_channels * 4, base_channels * 2, condition_dim)
        self.up_28 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.up_block_28 = ResidualBlock(base_channels * 2, base_channels, condition_dim)
        self.output_conv = nn.Conv2d(base_channels, image_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        condition_value: torch.Tensor,
        spatial_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # condition_value: [B]。DDPM 传入归一化时间，score model 传入噪声等级。
        condition = self.condition_mlp(condition_value.float())

        if self.condition_channels:
            if spatial_condition is None:
                raise ValueError("这个模型需要 spatial_condition，但调用时没有提供")
            if spatial_condition.shape[1] != self.condition_channels:
                raise ValueError(
                    f"spatial_condition 应有 {self.condition_channels} 个通道，"
                    f"实际为 {spatial_condition.shape[1]}"
                )
            x = torch.cat([x, spatial_condition], dim=1)
        elif spatial_condition is not None:
            raise ValueError("这个模型创建时 condition_channels=0，不能接收空间条件")

        # 下采样路径，同时留下两个 skip feature。
        skip_28 = self.block_28(self.input_conv(x), condition)
        skip_14 = self.block_14(self.down_14(skip_28), condition)
        h = self.down_7(skip_14)
        h = self.middle_2(self.middle_1(h, condition), condition)

        # 上采样后与同尺寸的 skip feature 在通道维拼接。
        h = self.up_14(h)
        h = self.up_block_14(torch.cat([h, skip_14], dim=1), condition)
        h = self.up_28(h)
        h = self.up_block_28(torch.cat([h, skip_28], dim=1), condition)
        return self.output_conv(h)


if __name__ == "__main__":
    # 单独运行 model.py 可以快速检查输入输出形状。
    network = TinyUNet()
    test_output = network(torch.randn(4, 1, 28, 28), torch.rand(4))
    print("输入/输出形状:", (4, 1, 28, 28), tuple(test_output.shape))
