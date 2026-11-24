# msa\_index\_score\_v2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：封装 `aclnnMsaIndexScoreV2`，计算 MSA Index Branch 的 KV 块分数。Prefill 和 Decode 使用同一接口。接口可返回完整分数、TopK 块号，或 TopK 块号及对应分数。
- **计算公式**：

$$
S_h = \operatorname{MaxPool}_{\mathrm{block}}\left(
\widehat Q_{idx,h}\widehat K_{idx}^{T}+\mathrm{atten\_mask}
\right)
$$

$S_h$ 是第 $h$ 个 query head 的块内最大分数。常规浮点输入直接计算点积，MXFP4 的 query 和 key 分别用 E8M0 系数按每 32 个元素缩放后再计算。$D$ 为逻辑 head dim。

- **主要计算过程**：

  1. 按 `sparse_mode` 确定可见范围。取 0 时不做因果截断，取 3 时按右下对齐的因果掩码屏蔽不可见位置。
  2. 在每个逻辑块内对可见 key 行取最大值。key 由调用方压缩。每个逻辑块覆盖 `pool_block_size` 个原始 token，对应 `pool_block_size / compress_factor` 行 key。例如块长 16、压缩倍数 4 时，每块最多归约 4 行。因果模式按每行覆盖的第一个 token 判断该行是否可见。
  3. 块长 128 按 head 输出分数，并对前缀块和局部块写入高分。块长 16 继续做跨 head 最大值归约，输出 $D^{-1/2}\log_2(e)\cdot\max_h S_h$，转换为 float16，前缀块和局部块保留真实分数。
  4. `return_choice` 为 1 或 2 时，对可见块做 TopK，同分优先选块号较小的块。块长 128 按 head 选取，按块号升序写出。块长 16 在跨 head 归约后选取，可见块不超过 128 时按块号升序全部写出，否则优先保留可见前缀和局部块，并将最大入选块号移至最后一个有效槽。

## 函数原型

```python
cann_ops_transformer.msa_index_score_v2(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    query_scale=None,
    key_scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_key="BBND",
    sparse_mode=3,
    init_blocks=0,
    local_blocks=1,
    layout_query="TND",
    pool_block_size=128,
    return_choice=0,
    topk=16,
    compress_factor=1,
)
```

## 参数说明

必选张量和满足必选条件的张量禁止传入 `None`。表中的默认值为 Python 接口默认值。数据格式均为 ND。

| 参数名 | 参数类型 | 描述 | dtype | shape |
| --- | --- | --- | --- | --- |
| query | 必选 | $Q_{idx}$。布局由 `layout_query` 指定，须连续。 | fp16 / bf16，Atlas A2/A3 支持 int8，950 的扩展类型见下文 | TND `[T1, N1, D]`；BSND `[B, S1, N1, D]`；BNSD `[B, N1, S1, D]` |
| key | 必选 | $K_{idx}$。布局由 `layout_key` 指定。分页允许物理页轴非连续，其余长度大于 1 的轴须为紧凑步长。非分页须连续。 | 与 query 同型 | TND `[T2, N2, D]`；BSND `[B, S2, N2, D]`；BNSD `[B, N2, S2, D]`；BBND `[NP, P, N2, D]`；BNBD `[NP, N2, P, D]` |
| start_loc | 必选 | 每条请求当前 query 所在的逻辑打分块号。token 位置129、块长128时传入1。块长 128 用它定位局部窗口。 | int32 | `[B]` |
| block_table | 分页必选 | 逻辑页到物理页的映射，值为 key 第 0 维的下标。非分页传 `None`。 | int32 | `[B, MB]` |
| query_scale | A8C8、MXFP4 必选 | A8C8 为逐 token、逐 head 的 Q 反量化系数，MXFP4 为每 32 个元素一组的缩放系数，其他类型传 `None`。 | A8C8 为 float32，MXFP4 为 `torch.float8_e8m0fnu` 或 `torch.uint8`（E8M0 编码） | A8C8 为 `[Tq,Hq]`；MXFP4 与 query 同布局，末维为 `D/32` |
| key_scale | A8C8、MXFP4 必选 | A8C8 为逐 K token 的反量化系数，MXFP4 为每 32 个元素一组的缩放系数，其他类型传 `None`。分页时按物理页紧凑存储。 | A8C8 为 float32，MXFP4 为 `torch.float8_e8m0fnu` 或 `torch.uint8`（E8M0 编码） | A8C8 分页为 `[NP,1,128]`，TND 为 `[Tk,1]` 或 `[Tk]`；MXFP4 与 key 同布局，末维为 `D/32` |
| atten_mask | `sparse_mode=3` 必选 | 因果模板，0 表示可见，1 表示屏蔽。取 `sparse_mode=0` 时传 `None`。 | int8 | `[2048, 2048]` |
| actual_seq_qlen | 必选 | query 有效长度的前缀和，从 0 开始，单调不减。 | int32 张量或 Python 整数列表、元组 | `[B+1]` |
| actual_seq_klen | 必选 | 按压缩前的 KV token 数计。分页传各请求的可见长度，非分页传从 0 开始且单调不减的前缀和。 | int32 张量或 Python 整数列表、元组 | 分页 `[B]`；非分页 `[B+1]` |
| layout_key | 可选 | 支持 `TND` / `BSND` / `BNSD` / `BBND` / `BNBD`，默认 `BBND`。 | str | - |
| sparse_mode | 可选 | 支持 0 或 3，默认 3。 | int | - |
| init_blocks | 可选 | 强制保留的前缀块数，默认 0。 | int | - |
| local_blocks | 可选 | 局部窗口的块数，默认 1。块长 128 时窗口右端为 `start_loc`，块长 16 时为当前 token 的最后一个可见块。 | int | - |
| layout_query | 可选 | 支持 `TND` / `BSND` / `BNSD`，默认 `TND`。 | str | - |
| pool_block_size | 可选 | 逻辑打分块长，单位为原始 token。支持 16 或 128，默认 128。 | int | - |
| return_choice | 可选 | 支持 0、1、2，默认 0。取 0 返回 score，取 1 返回块号，取 2 返回块号和分数。 | int | - |
| topk | 可选 | 默认 16。取 `return_choice=1` 或 2 时，块长 128 支持 1 到 128，块长 16 须显式传入 128。 | int | - |
| compress_factor | 可选 | 每行 key 覆盖的原始 token 数，默认 1。Ascend 950 的块长 16 支持 1、2、4、8、16，其余场景只支持 1。 | int | - |

`B` 为请求数，`N1` / `N2` 为 query / key 的 head 数，`T1` 为有效 query token 总数，`T2` 为 TND key 的总存储行数，`S1` / `S2` 分别为 query / key 的序列容量。`NP` 为物理页数，`P` 为每页存储行数，`MB` 为每条请求的页表宽度。

<!-- npu="A3,910b" id8 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
    - query 和 key 支持 fp16 / bf16，以及同型 int8 的 A8C8。query 仅支持 TND，逻辑块长与物理页长均只支持 128，`return_choice` 支持 0、1、2。A8C8 TopK 在打分阶段结束并同步后执行，按 head 选出逻辑块。
<!-- end id8 -->
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
    - query 和 key 还支持同型 `torch.float8_e4m3fn`、`torch.float8_e5m2` 或 `torch.float4_e2m1fn_x2`（MXFP4）。块长 16 和 128 均支持这些类型的 TopK。
    - 块长 16 支持 16 行或 128 行物理页。压缩时另支持 `128 / compress_factor` 行。块长 128 仅支持 128 行物理页。
<!-- end id7 -->

## 返回值说明

| 返回值 | dtype | shape |
| --- | --- | --- |
| score（`return_choice=0`） | 块长 128 为 float32，块长 16 为 float16 | `[H, T1, RoundUp(M, 16)]` |
| sparse_indices（`return_choice=1` 或 2） | int32 | `[H, T1, topk]` |
| sparse_values（`return_choice=2`） | float32 | `[H, T1, topk]` |

块长 128 时 `H=N1`，块长 16 时 `H=1`。输出的 `T1` 等于 `actual_seq_qlen[-1]`。`return_choice=2` 返回 `(sparse_indices, sparse_values)`，索引和分数按槽位一一对应。TopK 未填满的槽位用块号 `-1` 补齐。对应分数和 score 的补位值在块长 128 时为 float32 最小有限值，在块长 16 时为 `-inf`。

`M` 是对齐前的 score 列数，score 末维至少为 16：

- 分页按页表容量分配，`M = MB * (P / (pool_block_size / compress_factor))`。
- 非分页通常取批内各请求逻辑块数的最大值，即对各请求的 KV token 数除以逻辑块长并向上取整。块长 128 的 BSND / BNSD key 在 `actual_seq_klen` 位于 NPU 或输入为 MXFP4 时，按序列容量 `S2` 计算逻辑块数。

## 约束说明

- `actual_seq_qlen` 和 `actual_seq_klen` 可在 CPU 或 query 所在的 NPU 上。CPU 长度会复制到设备，调用返回前等待当前 NPU 流完成。NPU 图捕获须使用设备侧长度。其余输入张量须位于 query 所在的 NPU。
- TorchAir 构图时，若输出形状取决于长度值，请将对应长度作为 Python 整数列表或元组传入。长度值变化时可能重新编译。
- TND key 只支持 TND query。BSND / BNSD key 在块长 128 时搭配 TND query，在块长 16 时搭配同布局的 query。分页 key 可搭配平台支持的任一种 query 布局。
- 单条请求或整批的有效 query / KV 长度可以为 0。BSND / BNSD 的有效 query 长度不能超过 `S1`，每条请求的压缩 key 行数不能超过 `S2`。分页的物理页长须是 `pool_block_size / compress_factor` 的正整数倍。
- `N2` 只支持 1。query 和 key 的逻辑 `D` 必须相同，支持 16 到 128 之间的 16 倍数。MXFP4 只支持 `D=64` 或 128。块长 16 要求 `N1` 整除 128。
- `init_blocks` 和 `local_blocks` 须为非负整数，且各自不超过逻辑块上限。块长 16 的 TopK 中，如果有 token 可以看到超过 128 个逻辑块，两者之和不得超过 128。
- Atlas A2/A3 和 Ascend 950 都支持 `return_choice=1` 和 `return_choice=2`。TorchAir 构图支持模式 0 和 1，模式 2 用于单算子调用。

## 确定性计算

默认支持确定性计算。

## 调用示例

运行前加载 CANN 环境并安装 [torch_extension](../../../torch_extension/README.md)。只安装算子子包时，将导入名改为 `cann_ops_transformer_<vendor>`。下面的示例可在同一进程中依次运行。

- PageAttention BBND，块长 128。BNBD 将 key 换为 `key_bbnd.permute(0, 2, 1, 3).contiguous()`，并将 `layout_key` 改为 `"BNBD"`。

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)

T1, N1, N2, D, P = 32, 8, 1, 128, 128
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(2, P, N2, D, dtype=torch.float16).npu()
block_table = torch.tensor([[0, 1]], dtype=torch.int32).npu()
actual_seq_qlen = torch.tensor([0, T1], dtype=torch.int32).npu()
actual_seq_klen = torch.tensor([256], dtype=torch.int32).npu()
start_loc = torch.tensor([1], dtype=torch.int32).npu()
atten_mask = torch.zeros(2048, 2048, dtype=torch.int8).npu()
score = cann_ops_transformer.msa_index_score_v2(
    query, key_bbnd, start_loc,
    block_table=block_table,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_klen=actual_seq_klen,
    layout_key="BBND",
)
# score: [8, 32, 16], float32
```

- 非分页 TND / BSND / BNSD。块长 128 时复用 TND query，`actual_seq_klen` 传前缀和，省略 `block_table`。

```python
T2 = 256
key_tnd = torch.randn(T2, N2, D, dtype=torch.float16).npu()
key_bsnd = torch.randn(1, T2, N2, D, dtype=torch.float16).npu()
key_bnsd = torch.randn(1, N2, T2, D, dtype=torch.float16).npu()
klen_prefix = torch.tensor([0, T2], dtype=torch.int32).npu()
for layout_key, key in (("TND", key_tnd), ("BSND", key_bsnd), ("BNSD", key_bnsd)):
    score = cann_ops_transformer.msa_index_score_v2(
        query, key, start_loc,
        atten_mask=atten_mask,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=klen_prefix,
        layout_key=layout_key,
    )
```

- 块长 16，key 已按 4 倍压缩，返回 TopK 块号和分数。仅 Ascend 950 支持。64 个原始 KV token 对应 16 行 key，前 4 个块号有效。

```python
query16 = torch.randn(4, 16, D, dtype=torch.float16).npu()
key16 = torch.randn(16, N2, D, dtype=torch.float16).npu()
start16 = torch.zeros(1, dtype=torch.int32).npu()
qlen16 = torch.tensor([0, 4], dtype=torch.int32).npu()
klen16 = torch.tensor([0, 64], dtype=torch.int32).npu()

indices, values = cann_ops_transformer.msa_index_score_v2(
    query16, key16, start16,
    actual_seq_qlen=qlen16,
    actual_seq_klen=klen16,
    layout_key="TND",
    sparse_mode=0,
    init_blocks=0,
    local_blocks=0,
    pool_block_size=16,
    compress_factor=4,
    return_choice=2,
    topk=128,
)
# indices: [1, 4, 128], int32
# values:  [1, 4, 128], float32
```

### MXFP4（A4C4）

仅 Ascend 950 支持。query 和 key 均为 `torch.float4_e2m1fn_x2`，每个元素打包两个 FP4 值，张量末维为逻辑 `D/2`。分别传入 `query_scale` 和 `key_scale`，两者的 dtype 均为 `torch.float8_e8m0fnu`，shape 前缀与对应输入相同，末维为 `D/32`。

也支持将 Q、K、`query_scale` 和 `key_scale` 均以 `torch.uint8` 传入，各张量沿用上述形状。Q/K 每字节存两个 E2M1 值，scale 每字节存一个 E8M0 编码。
