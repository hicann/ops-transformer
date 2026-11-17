# flash_attn_func（FA4 兼容接口）

## 产品支持情况

支持 Ascend 950PR 和 950DT。该接口位于 `cann_ops_transformer` 的实验算子包，调用 `flash_attn_exp` 实现非量化注意力前向计算。

## 功能说明

`flash_attn_func` 提供与 FA4 `flash_attn_func` 对应的参数入口。接口根据输入生成 `flash_attn_exp_metadata`，再调用 `flash_attn_exp`，返回注意力结果和可选的 softmax log-sum-exp（LSE）。不支持的功能在调用算子前抛出异常。

```text
scores = softmax_scale * (Q @ K.transpose(-1, -2))
out = softmax(scores) @ V
lse = logsumexp(scores, dim=-1)
```

## 函数原型

```python
flash_attn_func(
    q, k, v, qv=None, gather_kv_indices=None, softmax_scale=None,
    causal=False, window_size=(None, None), learnable_sink=None, softcap=0.0,
    num_splits=1, pack_gqa=None, deterministic=False, score_mod=None,
    score_mod_bwd=None, mask_mod=None, aux_tensors=None, aux_scalars=None,
    block_sparse_tensors=None, block_sparse_tensors_bwd=None, return_lse=False,
    gather_bwd_recompute_p=False, gather_bwd_token_chunk=None, config=None,
    scales=None,
) -> (Tensor, Optional[Tensor])
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `q` | 必选，Query，BSND 格式，形状 `(B, S_q, N_q, 64)`。 |
| `k`、`v` | 必选，Key 和 Value，BSND 格式，形状均为 `(B, S_kv, N_kv, 64)`，且必须连续。`N_q` 必须能被 `N_kv` 整除。 |
| `softmax_scale` | 可选，softmax 前的缩放系数；`None` 时为 `1 / sqrt(head_dim)`，当前为 `0.125`。 |
| `return_lse` | 可选，默认为 `False`；为 `True` 时返回 LSE。 |
| 其他参数 | 保留 FA4 接口的参数位置。目前仅支持原型中列出的默认值；传入其他值会抛出 `NotImplementedError`。 |

`q`、`k`、`v` 必须在同一 NPU 上，数据类型相同且为 `torch.float16` 或 `torch.bfloat16`。批大小必须满足 `0 < B < 65536`，序列长度和 head 数必须大于 0。

目前不支持 causal 和滑动窗口、sink、softcap、MLA、gather、分页或块稀疏、自定义 score/mask、量化 scales、调度配置，以及反向传播。`num_splits` 仅支持 `1`，`pack_gqa` 仅支持 `None`。启用梯度计算且任一输入需要梯度时，接口会抛出 `NotImplementedError`。

## 返回值

返回二元组 `(out, lse)`：

- `out`：与 `q` 数据类型相同，形状为 `(B, S_q, N_q, 64)`。
- `lse`：`return_lse=True` 时为 float32 张量，形状为 `(B, N_q, S_q)`；否则为 `None`。

## 与 flash_attn_exp 的对应关系

| FA4 参数 | flash_attn_exp 参数或行为 |
| --- | --- |
| `q`、`k`、`v` | 直接传入，输入和输出布局均为 `BSND`。 |
| `softmax_scale` | 显式传入 `softmax_scale`；FA4 入口为 `None` 时先计算 `1 / sqrt(head_dim)`。 |
| `return_lse` | 对应 `return_softmax_lse`；不请求时将后端的第二个输出转换为 `None`。 |
| `causal=False`、`window_size=(None, None)` | 使用无掩码模式 `mask_mode=0`。 |
| metadata | 从输入的 batch、序列长度、head 数和维度生成，并传入主算子。 |

注意：直接调用 `flash_attn_exp` 或主线 `flash_attn` 时，底层接口默认 `softmax_scale=1.0`；需要与 FA4 默认行为一致时，应显式传入 `1 / sqrt(head_dim)`。

## 编译与安装

从仓库根目录运行，`CANN_DIR` 为实际的 CANN 安装路径，`INSTALL_DIR` 为自定义算子安装路径：

```bash
source ${CANN_DIR}/set_env.sh
bash build.sh --pkg --ops=flash_attn_exp,flash_attn_exp_metadata --experimental --soc=ascend950
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=${INSTALL_DIR}
source ${INSTALL_DIR}/vendors/custom_transformer/bin/set_env.bash

# 构建并安装实验 Torch 扩展包
bash build.sh --torch_extension --experimental --ops=flash_attn_exp --soc=ascend950
python3 -m pip install build_out/cann_ops_transformer_custom-*.whl --no-deps
```

## 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer.ops import flash_attn_func

q = torch.randn(1, 64, 4, 64, dtype=torch.float16, device="npu")
k = torch.randn(1, 128, 2, 64, dtype=torch.float16, device="npu")
v = torch.randn_like(k)

out, lse = flash_attn_func(q, k, v, return_lse=True)
torch.npu.synchronize()
print(out.shape)  # torch.Size([1, 64, 4, 64])
print(lse.shape)  # torch.Size([1, 4, 64])
```

直接调用 `flash_attn_exp` 和 `flash_attn_exp_metadata` 的参数及约束见 [flash_attn_exp 说明](../README.md)。
