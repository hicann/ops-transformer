# StemIndexer

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

StemIndexer是推理场景下稀疏Attention的前处理算子，承担块级打分与动态选块职责。对于每个Query Block，算子基于`qflat`与`kflat`的相关性，并叠加Value量值偏置`vbias`（OAM，Output-Aware Metric）进行打分；随后按Position-Decay动态TopK预算选出关键Key Block，输出`sparse_indices`与`sparse_seq_len`供下游Block Sparse Attention算子使用。

当前固定参数为：

```text
stem_block_size = 128
stem_stride = 16
R = stem_block_size / stem_stride = 8
D = 128
D_flat = stem_stride * D = 2048
```

上游预处理将一个包含128个Token的Q/K Block组织为`[R, stem_stride, D] = [8, 16, 128]`，沿R轴聚合后得到16个代表向量，再展平为长度2048的`qflat`或`kflat`。每个Q代表和K代表均聚合8个Token，因此一次代表向量点积展开后包含`8 * 8 = 64`个Token Pair贡献。

计算公式如下：

$$
\text{score} = \text{qflat} \cdot \text{kflat}^{T} \cdot \text{rSquare} + \text{vbias}
$$

$$
\text{rSquare}
= \frac{1}{(\text{stem\_block\_size}/\text{stem\_stride})^2}
= \frac{1}{(128/16)^2}
= \frac{1}{64}
$$

即Q/K代表的点积结果乘以`rSquare=1/64`，等价于除以64，对每组内部的64个Token Pair贡献进行归一化。完整的`qflat * kflat`包含16组代表向量的点积。

主要计算过程如下：

1. `qflat`与`kflat`做矩阵乘，得到每个Query Block与各Key Block的相关性分数。
2. 乘以`rSquare`完成聚合尺度归一化，并叠加`vbias`，避免纯QK相关性漏选对输出贡献较大的Key Block。
3. 根据`num_prompt_tokens`计算Position-Decay动态TopK预算，并按Query位置进行线性衰减。
4. 在动态TopK结果之外，固定保留开头的`initial_blocks`个Sink Block和末尾的`window_size`个Window Block。
5. 输出选中的Key Block逻辑索引及每个Query Block对应的有效索引数量。

## 接口说明

该算子提供底层ACLNN接口`aclnnStemIndexer`，并通过PyTorch扩展注册为`torch.ops.custom.npu_stem_indexer`。PyTorch入口负责创建输出Tensor并调用底层ACLNN接口。

### PyTorch接口原型

```python
torch.ops.custom.npu_stem_indexer(
    qflat: Tensor,
    kflat: Tensor,
    vbias: Tensor,
    q_seq_lens: Tensor,
    kv_seq_lens: Tensor,
    *,
    num_prompt_tokens: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    causal: bool = True,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    alpha: float = 1.0,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
    topk_score_precision: int = 1,
) -> Tuple[Tensor, Tensor]
```

## 参数说明

维度符号说明：

- `B`：Batch size。
- `N1`：Query Head数量。
- `N2`：KV Head数量。
- `D`：原始Q/K Head Dim，当前固定为128。
- `R`：每个代表向量聚合的Token数，`R = stem_block_size / stem_stride`，当前固定为8。
- `D_flat`：`qflat`和`kflat`的最后一维，`D_flat = stem_stride * D`，当前固定为2048。
- `max_Qb`：最大Query Block数，对应`qflat`第2维。
- `max_Kb`：最大KV Block数，对应`kflat`第2维。
- `actual_Qb`：单个Batch的实际Query Block数，`actual_Qb = ceil(q_seq_lens / stem_block_size)`。
- `actual_Kb`：单个Batch的实际KV Block数，`actual_Kb = ceil(kv_seq_lens / stem_block_size)`。
- `actual_visible_Kb`：考虑Causal语义后，当前Query Block实际可见的KV Block数。
- `metadata_size`：Metadata Tensor的INT32元素数量。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| `qflat` | 输入 | Q侧块级压缩表示。 | Kernel按连续布局处理；标准执行路径声明了AutoContiguous。 | BF16 | ND | `(B,N1,max_Qb,D_flat)` |
| `kflat` | 输入 | K侧块级压缩表示。 | 最后一维必须与`qflat`一致。Kernel按连续布局处理；标准执行路径声明了AutoContiguous。 | BF16 | ND | `(B,N2,max_Kb,D_flat)` |
| `vbias` | 输入 | Value量值偏置，即OAM项。 | Batch、KV Head和KV Block维度必须与`kflat`一致。 | FLOAT32 | ND | `(B,N2,max_Kb)` |
| `q_seq_lens` | 输入 | 每个Batch中Query的有效Token数。 | 每个元素按`stem_block_size`向上取整为实际Query Block数。 | INT32 | ND | `(B)` |
| `kv_seq_lens` | 输入 | 每个Batch中KV的有效Token数。 | 每个元素按`stem_block_size`向上取整为实际KV Block数。 | INT32 | ND | `(B)` |
| `num_prompt_tokens` | 可选输入 | 每个Batch的Prompt Token数。 | 用于Position-Decay动态TopK预算分档；未传入时复用`kv_seq_lens`。 | INT32 | ND | `(B)` |
| `metadata` | 可选输入 | 分核调度信息。 | 接口声明为可选输入，但当前计算必须使用有效Metadata；未传入时Tiling返回参数错误。 | INT32 | ND | `(metadata_size)` |
| `causal` | 输入属性 | 是否采用Right-down Causal语义。 | 默认值为`true`。 | BOOL | - | - |
| `stem_block_size` | 输入属性 | 一个Stem Block包含的原始Token数。 | 当前仅支持128，默认值为128。 | INT64 | - | - |
| `stem_stride` | 输入属性 | Stem Block内部的分组数/聚合Stride。 | 当前仅支持16，默认值为16。 | INT64 | - | - |
| `alpha` | 输入属性 | 控制动态TopK预算随Query位置的衰减程度。 | `k_start`表示序列前部Query Block的初始TopK块数；随着Query位置向后移动，TopK预算线性衰减，结束预算为`k_end = k_start * alpha`。取值范围为`(0,1]`，默认值为1.0；`alpha=1.0`表示不衰减，值越小表示衰减越强、序列后部选择的Key Block越少。 | FLOAT32 | - | - |
| `initial_blocks` | 输入属性 | 开头固定保留的Sink Block数。 | 当前仅支持4，默认值为4。 | INT64 | - | - |
| `window_size` | 输入属性 | 末尾固定保留的Window Block数。 | 当前仅支持4，默认值为4。 | INT64 | - | - |
| `k_block_num_rate_medium` | 输入属性 | 中等长度Prompt的TopK预算系数。 | 当前仅支持0.2，默认值为0.2。 | FLOAT32 | - | - |
| `k_block_num_bias_medium` | 输入属性 | 中等长度Prompt的TopK预算偏置。 | 当前仅支持30，默认值为30。 | INT64 | - | - |
| `k_block_num_rate_large` | 输入属性 | 长Prompt的TopK预算系数。 | 当前仅支持0.1，默认值为0.1。 | FLOAT32 | - | - |
| `k_block_num_bias_large` | 输入属性 | 长Prompt的TopK预算偏置。 | 当前仅支持30，默认值为30。 | INT64 | - | - |
| `topk_score_precision` | 输入属性 | TopK内部可排序Score的存储精度。 | 1表示UINT32，2表示UINT16，默认值为1；不改变输出Tensor的数据类型。 | INT64 | - | - |
| `sparse_indices` | 输出 | 选中的Key Block逻辑索引。 | 每行仅前`sparse_seq_len`项有效，尾部无效区填充为-1；调用者不应依赖有效索引的排列顺序。 | INT32 | ND | `(B,N1,max_Qb,max_Kb)` |
| `sparse_seq_len` | 输出 | 每个Query Block对应的有效Key Block数量。 | 无有效Query/KV任务时对应值为0。 | INT32 | ND | `(B,N1,max_Qb)` |

## 约束说明

### 单参数约束

- 该接口支持图模式。
- 当前仅适配Ascend 950PR/Ascend 950DT，不支持A3、A2及其他产品。
- `qflat`、`kflat`仅支持BF16、ND格式和4D Tensor。
- `vbias`仅支持FLOAT32、ND格式和3D Tensor。
- `q_seq_lens`、`kv_seq_lens`仅支持INT32、ND格式和1D Tensor；传入`num_prompt_tokens`或`metadata`时同样仅支持INT32、ND格式和1D Tensor。
- `sparse_indices`、`sparse_seq_len`的数据类型固定为INT32，数据格式为ND。
- `stem_block_size`固定为128，`stem_stride`固定为16，因此`R=8`、`D_flat=2048`。
- `initial_blocks`固定为4，`window_size`固定为4。
- `k_block_num_rate_medium`固定为0.2，`k_block_num_bias_medium`固定为30。
- `k_block_num_rate_large`固定为0.1，`k_block_num_bias_large`固定为30。
- `alpha`应为有限浮点数且满足`0 < alpha <= 1`。
- `topk_score_precision`仅支持1和2。

### 存在性约束

- `qflat`、`kflat`、`vbias`、`q_seq_lens`和`kv_seq_lens`必须传入有效Tensor。
- `num_prompt_tokens`可以不传；缺省时OpHost在TilingData中记录复用标志，Kernel使用`kv_seq_lens`作为`num_prompt_tokens`。
- `metadata`在接口中声明为可选输入，以支持统一的Optional调用形式；当前计算仍要求传入有效Tensor，缺省时Tiling明确返回参数错误。
- `metadata`需要由StemIndexerMetadata算子根据本次输入生成，不支持传入空Tensor或与本次输入无关的占位Tensor。

### 一致性约束

- `B`、`max_Qb`和`max_Kb`必须大于0。
- `N1`仅支持32或64，`N2`仅支持2、4或8，并满足`N1 % N2 == 0`。
- `qflat`和`kflat`的Batch维、最后一维必须一致，最后一维固定为2048。
- `vbias`的shape必须为`(B,N2,max_Kb)`，与`kflat`对应维度一致。
- `q_seq_lens`和`kv_seq_lens`的shape必须为`(B)`；传入`num_prompt_tokens`时，其shape也必须为`(B)`。
- `q_seq_lens[b]`应满足`0 <= q_seq_lens[b] <= max_Qb * stem_block_size`。
- `kv_seq_lens[b]`应满足`0 <= kv_seq_lens[b] <= max_Kb * stem_block_size`。
- 有效Prompt Token数应为非负值，并满足`effective_num_prompt_tokens[b] >= kv_seq_lens[b]`；其中传入`num_prompt_tokens`时取其值，未传入时`effective_num_prompt_tokens = kv_seq_lens`。
- `sparse_indices`的shape由输入推导为`(B,N1,max_Qb,max_Kb)`；`sparse_seq_len`的shape由输入推导为`(B,N1,max_Qb)`。
- `sparse_indices`有效前缀中的元素为Key Block逻辑索引，取值范围为`[0, actual_Kb - 1]`。
- `sparse_seq_len`每个元素的取值范围为`[0, actual_visible_Kb]`，并且不大于对应输出行的`max_Kb`。

### 特性交叉约束

- `metadata`必须由与主算子相同的`q_seq_lens`、`kv_seq_lens`、`N1`、`N2`、`causal`、`stem_block_size`、`D_flat`和`window_size`生成。
- `metadata`最大Section数为`B * N2`，容量按以下公式计算，单位为INT32元素：

  ```text
  max_section_num = B * N2
  required_elems = 16 + max_section_num * (36 + 72) * 16
  metadata_size = AlignUp(required_elems, 4096)
  ```

- `causal=false`时，每个有效Query Block可以在实际KV Block范围内选块。
- `causal=true`且不是Decode场景时，第`qb`个Query Block的可见KV Block数量为：

  ```text
  s2_valid = clamp(Kb - Qb + qb + 1, 0, Kb)
  ```

- Decode场景定义为`q_seq_lens[b] == 1`且`effective_num_prompt_tokens[b] >= kv_seq_lens[b]`，该场景使用完整的实际KV Block范围。
- Sink Block和Window Block不参与普通TopK候选，最终结果为固定块与动态TopK结果的并集，避免重复索引。
- `topk_score_precision`只影响内部Score排序精度；无论取1还是2，`sparse_indices`与`sparse_seq_len`均为INT32。
- Tensor元素值相关约束由调用者保证；接口的Shape/Dtype校验不能替代对输入Tensor内容的合法性检查。

## 动态TopK预算说明

首先将Prompt Token数转换为Prompt Block数：

```text
prompt_block_num = ceil(num_prompt_tokens / stem_block_size)
```

根据Prompt Block数计算初始预算`k_start`：

```text
prompt_block_num < 56:
    k_start = prompt_block_num

56 <= prompt_block_num < 160:
    k_start = floor(prompt_block_num * 0.2 + 30)

prompt_block_num >= 160:
    k_start = floor(prompt_block_num * 0.1 + 30)
```

位置衰减终点为：

```text
k_end = k_start * alpha
```

在衰减区间内，当前Query Block的动态预算按照`k_start`到`k_end`线性插值后向下取整，并限制在`[1,k_start]`范围内。动态TopK最多选择256个普通候选块，此外固定保留最多4个Sink Block和4个Window Block；最终有效索引数量不超过当前Query Block的实际可见KV Block数量。

## Ascend 950PR/Ascend 950DT调用示例

```python
import torch
import torch_npu
import custom_ops

batch_size = 4
q_heads = 64
kv_heads = 8
d = 128
stem_block_size = 128
stem_stride = 16
q_block_num = 16
k_block_num = 64
q_seq_len = q_block_num * stem_block_size
kv_seq_len = k_block_num * stem_block_size
prompt_token_num = kv_seq_len

torch.manual_seed(0)
qflat = torch.randn(
    batch_size,
    q_heads,
    q_block_num,
    stem_stride * d,
    dtype=torch.bfloat16,
).npu()
kflat = torch.randn(
    batch_size,
    kv_heads,
    k_block_num,
    stem_stride * d,
    dtype=torch.bfloat16,
).npu()
vbias = torch.randn(
    batch_size,
    kv_heads,
    k_block_num,
    dtype=torch.float32,
).npu()
q_seq_lens = torch.full((batch_size,), q_seq_len, dtype=torch.int32).npu()
kv_seq_lens = torch.full((batch_size,), kv_seq_len, dtype=torch.int32).npu()
num_prompt_tokens = torch.full(
    (batch_size,), prompt_token_num, dtype=torch.int32
).npu()

# 1. 生成与主算子输入匹配的分核调度Metadata。
metadata = torch.ops.custom.npu_stem_indexer_metadata(
    q_seq_lens,
    kv_seq_lens,
    q_heads,
    kv_heads,
    causal=True,
    stem_block_size=stem_block_size,
    dim_qkflat=stem_stride * d,
    window_size=4,
)

# 2. 执行块级打分与动态选块。
sparse_indices, sparse_seq_len = torch.ops.custom.npu_stem_indexer(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens=num_prompt_tokens,
    metadata=metadata,
    causal=True,
    stem_block_size=stem_block_size,
    stem_stride=stem_stride,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    topk_score_precision=1,
)
```

## 相关接口

- `npu_stem_indexer_metadata`：根据序列长度、Head配置和Causal属性生成StemIndexer分核调度信息。

更多测试与调用示例见[pytest说明](../tests/pytest/README.md)。
