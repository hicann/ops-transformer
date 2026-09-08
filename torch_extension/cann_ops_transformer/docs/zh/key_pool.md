# key\_pool

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：`key_pool` 是推理场景下的 Key 压缩算子。算子对当前输入
  token 分别执行 K 投影和 Gate 投影，将连续的 `cmp_ratio` 个 token 分为一组，
  使用 Gate 和位置偏置计算组内权重，并将组内 K 加权池化为一个 Key。未完成
  压缩组所需的 K 和 Gate 状态通过 `state_cache` 跨调用保存。

- **计算公式**：

    1. 对每个输入 token 计算 K 和 Gate：

        $$
        K = hidden\_states \mathbin{@} wk^T,
        \qquad G = hidden\_states \mathbin{@} gate\_weight^T
        $$

    2. 当同时提供 `norm_weight` 和 `norm_bias` 时，沿每个 token 的最后一维
       执行 LayerNorm：

        $$
        K_{norm} = LayerNorm(K, \text{dim}=-1)
        $$

    3. 对每个压缩组计算 Gate 权重并池化：

        $$
        logits_{q,r} = G_{q,r} + ape_r
        $$

        $$
        weight_{q,:} = softmax(logits_{q,:})
        $$

        $$
        pooled\_key_{q} = \sum_{r=0}^{cmp\_ratio-1}
        weight_{q,r} K_{norm,q,r}
        $$

  历史 token 通过 `cache_block_table` 从 `state_cache` 读取，当前调用产生的
  投影结果用于更新对应的 cache 位置并参与池化。`state_cache` 的前 `D` 列保存
  K，后 `D` 列保存 Gate；当启用 LayerNorm 时，写入 cache 的 K 为 LayerNorm
  后的结果。`ape` 只参与池化计算，不写入 `state_cache`。

  `cos` 和 `sin` 参数用于预留 RoPE 入口。当前版本不执行 RoPE，二者必须同时
  为空。`seqused` 也为预留参数，当前版本必须为空。

## 函数原型

```python
cann_ops_transformer.key_pool(
    hidden_states: Tensor,
    wk: Tensor,
    gate_weight: Tensor,
    ape: Tensor,
    state_cache: Tensor,
    cache_block_table: Tensor,
    start_pos: Tensor,
    *,
    norm_weight: Tensor | None = None,
    norm_bias: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    seqused: Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
) -> Tensor
```

底层 Torch 算子注册为：

```python
torch.ops.cann_ops_transformer.key_pool(
    hidden_states: Tensor,
    wk: Tensor,
    gate_weight: Tensor,
    ape: Tensor,
    state_cache: Tensor,
    cache_block_table: Tensor,
    start_pos: Tensor,
    *,
    norm_weight: Tensor | None = None,
    norm_bias: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    seqused: Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
)
```

`state_cache` 是原地更新输入。算子返回一个 `pooled_key` Tensor。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- | ---- |
| hidden_states | Tensor | 必选 | 当前输入的 hidden states，对应公式中的 `hidden_states`。数据格式为 ND，不支持非连续 Tensor。支持 BSH 和 TH 两种布局。 | bfloat16、float16 | BSH：`[B,S,H]`；TH：`[T,H]` |
| wk | Tensor | 必选 | K 投影权重，对应公式中的 `wk`。数据格式为 ND，不支持非连续 Tensor。 | bfloat16、float16 | `[D,H]` |
| gate_weight | Tensor | 必选 | Gate 投影权重，对应公式中的 `gate_weight`。数据格式为 ND，不支持非连续 Tensor。 | bfloat16、float16 | `[D,H]` |
| ape | Tensor | 必选 | 位置偏置，对应公式中的 `ape`。数据格式为 ND，不支持非连续 Tensor。 | float32 | `[cmp_ratio,D]` |
| state_cache | Tensor | 必选 | 保存历史 K 和 Gate 的分页 cache，并由算子原地更新。数据格式为 ND；支持第 0 轴非连续，实际第 0 轴 stride 由底层接口传入。前 `D` 列保存 K，后 `D` 列保存 Gate。 | float32 | `[block_num,block_size,2*D]` |
| cache_block_table | Tensor | 必选 | 逻辑 block 到物理 cache block 的映射表。数据格式为 ND，不支持非连续 Tensor。 | int32 | `[B,L]` |
| start_pos | Tensor | 必选 | 每个 Batch 当前输入在逻辑序列中的起始位置。数据格式为 ND，不支持非连续 Tensor。 | int32 | `[B]` |
| norm_weight | Tensor | 可选 | LayerNorm 权重。数据格式为 ND，不支持非连续 Tensor。与 `norm_bias` 必须同时提供或同时为空。 | float32 | `[D]` |
| norm_bias | Tensor | 可选 | LayerNorm 偏置。数据格式为 ND，不支持非连续 Tensor。与 `norm_weight` 必须同时提供或同时为空。 | float32 | `[D]` |
| cos | Tensor | 可选 | RoPE 余弦参数。当前版本保留接口，必须为空。 | bfloat16 | 当前版本不适用 |
| sin | Tensor | 可选 | RoPE 正弦参数。当前版本保留接口，必须为空。 | bfloat16 | 当前版本不适用 |
| cu_seqlens | Tensor | 可选 | TH 场景中各 Batch 在 `hidden_states` 首轴上的累计 token 边界。数据格式为 ND，不支持非连续 Tensor；BSH 场景必须为空。 | int32 | TH：`[B+1]` |
| seqused | Tensor | 可选 | 预留的有效序列长度参数，当前版本必须为空。 | int32 | 预留 |
| cmp_ratio | int | 可选 | 压缩率，即每个池化组包含的 token 数。默认值为 `4`。 | - | - |
| norm_eps | float | 可选 | LayerNorm 的 epsilon，默认值为 `1e-6`，必须大于 0。 | - | - |
| rotary_mode | int | 可选 | RoPE 模式入口，默认值为 `1`。仅支持 `0/1`，当前版本不执行 RoPE。 | - | - |

## 返回值说明

| 返回值 | 返回值类型 | 描述 | 数据类型 | 维度（shape） |
| ---- | ---- | ---- | ---- | ---- |
| pooled_key | Tensor | 按 `cmp_ratio` 对 K 加权池化后的结果。 | bfloat16、float16 | `[B,Sr,D]` |

其中：

- `B` 表示 Batch Size；
- `S` 表示 BSH 场景中每个 Batch 的输入序列长度；
- `T` 表示 TH 场景中所有 Batch 输入 token 数的总和；
- `H` 表示 hidden size；
- `D` 表示 K 和 Gate 的 head dimension，由 `wk.size(0)` 决定；
- `L` 表示 `cache_block_table` 的逻辑 block 数；
- `Sr = ceil(L * block_size / cmp_ratio)`，表示输出的固定容量。

`BSH` 表示 Batch-Sequence-Hidden 布局，`hidden_states` 的 shape 为
`[B,S,H]`。`TH` 表示 Token-Hidden 布局，`hidden_states` 的 shape 为
`[T,H]`，并通过 shape 为 `[B+1]` 的 `cu_seqlens` 描述各 Batch 的 token
边界。

## 约束说明

- `hidden_states` 为 BSH 时，`cu_seqlens` 必须为空；为 TH 时，必须提供
  合法的 `[B+1]` 前缀和数组，首元素为 0，末元素为 `T`，且单调不减。
- `cmp_ratio` 仅支持 `2/4/8/16/32/64/128`，默认值为 `4`。
- `norm_eps` 必须大于 0，默认值为 `1e-6`。
- `rotary_mode` 仅支持 `0/1`，当前版本不执行 RoPE。
- `norm_weight` 和 `norm_bias` 必须同时提供或同时为空。
- `cos` 和 `sin` 必须同时为空；当前版本不支持 RoPE。
- `seqused` 必须为空，当前版本不通过该参数选择参与计算的 token。
- `state_cache`、`cache_block_table` 和 `start_pos` 必须提供；`state_cache`
  的第 0 轴 stride 支持由实际 Tensor stride 传递。
- 支持 BSH 场景的 `B=0` 或 `S=0`，以及 TH 场景的 `T=0`。
- 当前接口用于 NPU 推理场景，输入 Tensor 应位于 NPU 设备。

## 调用示例

### 单算子模式

```python
import torch
import torch_npu
import cann_ops_transformer

B = 1
S = 8
H = 4096
D = 128
cmp_ratio = 4
block_size = 16
device = "npu:0"

hidden_states = torch.randn((B, S, H), dtype=torch.bfloat16, device=device)
wk = torch.randn((D, H), dtype=torch.bfloat16, device=device)
gate_weight = torch.randn((D, H), dtype=torch.bfloat16, device=device)
ape = torch.randn((cmp_ratio, D), dtype=torch.float32, device=device)

block_num = (S + block_size - 1) // block_size
cache_block_table = torch.arange(
    1, B * block_num + 1, dtype=torch.int32, device=device
).reshape(B, block_num)
state_cache = torch.zeros(
    (B * block_num + 1, block_size, 2 * D),
    dtype=torch.float32,
    device=device,
)
start_pos = torch.zeros((B,), dtype=torch.int32, device=device)

pooled_key = cann_ops_transformer.key_pool(
    hidden_states,
    wk,
    gate_weight,
    ape,
    state_cache,
    cache_block_table,
    start_pos,
    cmp_ratio=cmp_ratio,
)
torch.npu.synchronize()
print(pooled_key.shape)
```

启用 LayerNorm 时，同时传入 `norm_weight` 和 `norm_bias`：

```python
norm_weight = torch.ones((D,), dtype=torch.float32, device=device)
norm_bias = torch.zeros((D,), dtype=torch.float32, device=device)

pooled_key = cann_ops_transformer.key_pool(
    hidden_states,
    wk,
    gate_weight,
    ape,
    state_cache,
    cache_block_table,
    start_pos,
    norm_weight=norm_weight,
    norm_bias=norm_bias,
    norm_eps=1e-6,
    rotary_mode=1,
    cmp_ratio=cmp_ratio,
)
```

### TorchAir 图模式调用（暂不支持）

当前 `key_pool` 尚未提供 `graph_convert_key_pool.py` 及对应的 TorchAir
Converter 注册，因此暂不支持通过 TorchAir 图模式调用。请使用上面的单算子
模式接口。
