"""为数字类别 bridge 构造可复现的 MNIST 端点配对，支持每轮重配。"""

from typing import Tuple

import torch
from torch.utils.data import Dataset


class PairedMNIST(Dataset):
    """把一个指定类别的图片与另一个类别逐一配对。

    例如 source_digit=0、target_digit=1 时，每个样本都是一张 0 和一张 1。
    两类样本数通常不同，因此数据集长度取较小值，不会偷偷重复某一端点。
    ``pair_seed`` 只负责打乱配对关系，不改变训练的全局随机种子。
    默认保持固定配对；调用 ``reshuffle_pairs()`` 只重排已有 target。
    """

    def __init__(
        self,
        mnist_dataset,
        source_digit: int = 0,
        target_digit: int = 1,
        pair_seed: int = 42,
    ):
        if source_digit == target_digit:
            raise ValueError("source_digit 与 target_digit 应不同，才能观察跨类别 bridge")

        targets = torch.as_tensor(mnist_dataset.targets)
        source_indices = torch.where(targets == source_digit)[0]
        target_indices = torch.where(targets == target_digit)[0]
        pair_count = min(len(source_indices), len(target_indices))

        # 使用局部 Generator，保证改变配对不会影响模型初始化或训练噪声。
        generator = torch.Generator().manual_seed(pair_seed)
        source_order = torch.randperm(len(source_indices), generator=generator)[:pair_count]
        target_order = torch.randperm(len(target_indices), generator=generator)[:pair_count]
        self.source_indices = source_indices[source_order].tolist()
        self.target_indices = target_indices[target_order].tolist()
        self._pair_generator = generator
        self.mnist_dataset = mnist_dataset
        self.source_digit = source_digit
        self.target_digit = target_digit

    def reshuffle_pairs(self) -> None:
        """重新打乱 target 的配对顺序，保持两端样本池和 source 顺序不变。"""
        order = torch.randperm(len(self.target_indices), generator=self._pair_generator)
        self.target_indices = [self.target_indices[i] for i in order.tolist()]

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        source_image, _ = self.mnist_dataset[self.source_indices[index]]
        target_image, _ = self.mnist_dataset[self.target_indices[index]]
        return source_image, target_image
