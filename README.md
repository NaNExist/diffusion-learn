# 一个用于学习扩散与流模型的小框架

这个项目在 MNIST 和同一个小型 U-Net 上并排实现多种生成动力学。重点不是追求
最高生成指标，而是让训练目标、采样方程、数值求解器之间的关系可以直接阅读、
修改和可视化。

所有常用配置仍然硬编码在脚本顶部，不使用命令行参数。图片统一归一化到
`[-1, 1]`，训练和采样均自动选择 CUDA 或 CPU。

## 已有方法

### 离散扩散与 score model

- **DDPM**：预测噪声 epsilon，使用离散祖先采样。
- **Score model（NCSN 风格）**：预测 score，使用退火 Langevin 动力学。

### 连续时间 VP diffusion

连续模型使用

```text
x_t = alpha(t) * x_0 + sigma(t) * epsilon
alpha(t)^2 + sigma(t)^2 = 1
```

网络可以选择预测以下任意一种目标：

```text
epsilon : 实际加入的高斯噪声
x0      : 干净图片
v       : alpha * epsilon - sigma * x0
score   : -epsilon / sigma
```

采样时，四种输出都会转换成 score，再构造同一个 VP probability-flow ODE。
因此可以在相同初始噪声、相同 schedule 下比较“预测什么”造成的差异。

### Flow Matching

基础版 conditional flow matching 使用从噪声到数据的直线路径：

```text
x_t = (1-t) * z + t * x_data
target velocity = x_data - z
```

训练后从 `t=0` 积分到 `t=1`。它与连续 VP diffusion 共用完全相同的 ODE
求解代码。

### Diffusion Bridge

教学版 bridge 默认学习 MNIST `0 → 1`，也可以在训练脚本顶部改成任意两个
不同数字。固定配对的 source 和 target 之间使用带端点约束的随机插值：

```text
x_t = (1-t) * source + t * target
      + bridge_sigma * sqrt(t*(1-t)) * epsilon
```

网络输入当前状态 `x_t`、时间 `t` 和源图片，预测条件速度。噪声在两个端点
严格为零、中间最大。这个实现用于理解 stochastic-interpolant / diffusion
bridge 的端点约束思想，并不是完整的 Schrödinger Bridge 迭代算法。

## ODE 求解器

`ode_solvers.py` 提供三个固定步长求解器：

| 求解器 | 阶数 | 每步网络调用（NFE） | 特点 |
|---|---:|---:|---|
| Euler | 1 | 1 | 最简单，步数少时误差较明显 |
| Heun | 2 | 2 | 速度和精度通常较均衡 |
| RK4 | 4 | 4 | 单步较贵，但高精度 |

比较求解器时应统一 NFE，而不只是统一步数。例如 Euler 40 步、Heun 20 步、
RK4 10 步都需要 40 次网络前向。

## 文件结构

```text
model.py                       共用 TinyUNet，可选额外空间条件
diffusion.py                   原有 DDPM 与 NCSN score process
continuous.py                  连续 VP、Flow Matching、Diffusion Bridge
ode_solvers.py                 Euler、Heun、RK4 与 NFE/耗时统计
paired_mnist.py                构造 0→1 等固定类别配对

train_ddpm.py                  训练离散 DDPM
sample_ddpm.py                 DDPM 祖先采样
train_score.py                 训练 NCSN 风格 score model
sample_score.py                退火 Langevin 采样

train_continuous.py            训练 epsilon/x0/v/score 连续 VP 模型
sample_continuous.py           probability-flow ODE 采样
train_flow_matching.py         训练 Flow Matching
sample_flow_matching.py        Flow Matching ODE 采样
train_bridge.py                训练数字类别 bridge，默认 0→1
sample_bridge.py               展示 source / generated / paired target

compare_models.py              原有 DDPM 与 score model 对比
compare_ode_solvers.py         相同初始噪声、相同 NFE 的 solver 对比
compare_prediction_types.py    epsilon/x0/v/score 对比
visualize.py                   所有图片、轨迹和 loss 可视化
```

## 建议运行顺序

运行前请确保当前 Python 环境已安装 PyTorch、torchvision、matplotlib 和 tqdm：

```bash
# 1. Flow Matching：最快理解 velocity 与 ODE
python train_flow_matching.py
python sample_flow_matching.py
python compare_ode_solvers.py

# 2. 连续 VP：修改 PREDICTION_TYPE 分别训练四次
python train_continuous.py
python sample_continuous.py
python compare_prediction_types.py

# 3. 数字 bridge：默认 0→1
python train_bridge.py
python sample_bridge.py
```

若只想确认流程，先把训练脚本顶部的 `EPOCHS` 改成 `1`、`BASE_CHANNELS`
改成 `16`。推理脚本会从 checkpoint 自动读取模型宽度、预测目标和噪声日程，
避免训练与推理设置不一致。

## 推荐观察方式

1. 在 `compare_ode_solvers.py` 中保持总 NFE 相等，观察不同阶数求解器的输出。
2. 固定 `SEED`，只改变 `NUM_ODE_STEPS`，观察低 NFE 下谁先失真。
3. 分别训练 epsilon/x0/v/score，比较 loss 尺度与最终样本，不直接比较不同
   定义的 loss 数值大小。
4. 在 `train_bridge.py` 中改变 `SOURCE_DIGIT`、`TARGET_DIGIT` 和
   `BRIDGE_SIGMA`，观察类别跨度与路径随机性带来的变化。

所有结果保存在 `outputs/`。`visualize.py` 只负责画图，核心数学过程不依赖
matplotlib。
