# lightning\_indexer\_kl\_loss

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口说明

  - **teacher 侧**（target_score）：压缩段未归一化的原始主注意力分数（sum ≠ 1），用 `clamp_min` 防止 target_score=0 处 log(0) 导致 NaN。
  - **student 侧**（index_probs）：indexer softmax 后的概率分布，用 `+eps` 保住 index_probs→0 处的梯度。

- 计算公式：

  $$
  y = \text{target\_score}, \quad Y = \text{index\_probs}
  $$

  $$
  P = \frac{y}{\text{sum}(y, \text{dim}=-1, \text{keepdim=True}) + \varepsilon}
  $$

  $$
  \log\_P = \log(\text{clamp\_min}(\tilde{y}, \varepsilon))
  $$

  $$
  \log\_Y = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum((\log\_P - \log\_Y) \cdot \text{weight})
  $$

  其中 $\varepsilon$ 为 `eps` 参数，默认值 $10^{-9}$。

  weight 的选择由 `weight_type` 控制：

  - `'logits'`（默认）：weight = y，即原始未归一化分数
  - `'probs'`：weight = P，即归一化概率

## 函数原型

```python
from cann_ops_transformer import lightning_indexer_kl_loss

lightning_indexer_kl_loss(
    target_score,
    index_probs,
    eps=1e-9,
    weight_type='logits'
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| target_score | 必选输入 | teacher 侧压缩段原始未归一化主注意力分数，sum != 1。 | `float16`、`bfloat16`、`float32` | `(B, S, K)` 或 `(T, K)` |
| index_probs | 必选输入 | student 侧 indexer softmax 后的概率分布。 | 与 `target_score` 一致 | 与 `target_score` 一致 |
| eps | 可选属性 | 数值稳定常数，默认值为1e-9。 | `float` | - |
| weight_type | 可选属性 | 外层权重选择。`'logits'`（默认）用原始 target_score 作为外层权重；`'probs'` 用归一化概率 P = target_score / sum(target_score) 作为外层权重。 | `str` | - |

## 返回值说明

| 输出 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- |
| loss | KL 散度标量损失。 | 与输入一致 | `(1,), 标量`|

## 约束说明

- `target_score` 与 `index_probs` 的数据类型必须一致，且shape一致。
- 支持 shape 为 (B, S, K) 或 (T, K)，B的取值范围为1\~512，最后一维 K 的取值范围为 1\~8192。
- `eps` 必须大于 0。
- `weight_type` 必须为 `'logits'` 或 `'probs'`。
- `target_score` 与 `index_probs` 需在同一 NPU 设备上。

## 确定性计算

- 默认不支持确定性计算，可通过PyTorch开关（torch.use_deterministic_algorithms）支持。

## 调用示例

### 单算子调用（默认 logits 权重）

```python
import torch
import torch_npu
from cann_ops_transformer import lightning_indexer_kl_loss

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

B, H, D = 4, 10, 128

target_score = torch.rand(B, H, D, dtype=torch.float32, device=device)
index_probs = torch.softmax(torch.randn(B, H, D, dtype=torch.float32, device=device), dim=-1)

# 使用 logits 权重（默认），等价于 loss = sum((log_P - log_Y) * y)
loss = lightning_indexer_kl_loss(target_score, index_probs)
torch.npu.synchronize()
print(f"KL loss (logits): {loss.item()}")
```
