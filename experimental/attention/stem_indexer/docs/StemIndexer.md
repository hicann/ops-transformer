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

令$B_s$表示`stem_block_size`，$T_s$表示`stem_stride`，$D$表示原始Q/K Head Dim，$D_f$表示`qflat`和`kflat`的最后一维。当前固定参数为：

$$
B_s=128,\quad T_s=16,\quad R=\frac{B_s}{T_s}=8,\quad D=128,\quad D_f=T_sD=2048
$$

上游预处理将一个包含128个Token的Q/K Block组织为$[R,T_s,D]=[8,16,128]$，沿$R$轴聚合后得到16个代表向量，再展平为长度2048的`qflat`或`kflat`。每个Q代表和K代表均聚合8个Token，因此一次代表向量点积展开后包含$8\times8=64$个Token Pair贡献。

计算公式如下：

$$
S_{i,j}=\frac{\langle Q_i,K_j\rangle}{R^2}+V_j
$$

$$
\frac{1}{R^2}=\frac{1}{(B_s/T_s)^2}=\frac{1}{(128/16)^2}=\frac{1}{64}
$$

其中，$S_{i,j}$表示第$i$个Query Block与第$j$个Key Block的分数，$Q_i$和$K_j$分别表示对应的块级压缩向量，$V_j$表示对应Key Block的`vbias`。Q/K代表的点积结果乘以$1/R^2=1/64$，对每组内部的64个Token Pair贡献进行归一化；完整的$Q_iK_j^T$包含16组代表向量的点积。

对第$i$个Query Block，令$\mathcal{V}_i$为当前可见Key Block集合，$\mathcal{S}_i$为固定保留的Sink Block集合，$\mathcal{W}_i$为固定保留的Window Block集合，则普通TopK候选集合为：

$$
\mathcal{C}_i=\mathcal{V}_i\setminus(\mathcal{S}_i\cup\mathcal{W}_i)
$$

根据Position-Decay得到当前Query Block的动态TopK预算$K_i$后，令$\widehat{K}_i=\min(K_i,256,|\mathcal{C}_i|)$表示实际普通TopK数量，则普通TopK结果及最终输出集合为：

$$
\mathcal{T}_i=\operatorname{TopK}_{\widehat{K}_i}\left\{S_{i,j}\mid j\in\mathcal{C}_i\right\}
$$

$$
\mathcal{I}_i=\mathcal{S}_i\mathbin{\Vert}\mathcal{T}_i\mathbin{\Vert}\mathcal{W}_i,\qquad L_i=\left|\mathcal{I}_i\right|
$$

其中，$\Vert$表示顺序拼接，$\mathcal{I}_i$写入`sparse_indices`的有效前缀，$L_i$写入`sparse_seq_len`。实现不会额外执行去重；普通TopK候选集合$\mathcal{C}_i$已预先排除Sink Block和Window Block。

主要计算过程如下：

1. `qflat`与`kflat`做矩阵乘，得到每个Query Block与各Key Block的相关性分数。
2. 乘以$1/R^2$完成聚合尺度归一化，并叠加`vbias`，避免纯QK相关性漏选对输出贡献较大的Key Block。
3. 根据`num_prompt_tokens`计算Position-Decay动态TopK预算$K_i$，并按Query位置进行线性衰减。
4. 从排除Sink Block和Window Block后的普通候选集合中选择分数最高的$K_i$个Key Block，再按Sink Block、普通TopK结果、Window Block的顺序拼接输出。
5. 输出选中的Key Block逻辑索引及每个Query Block对应的有效索引数量。

## 接口说明

该算子通过PyTorch扩展注册为`torch.ops.custom.npu_stem_indexer`。

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

- $B$：Batch size。
- $N_q$：Query Head数量。
- $N_k$：KV Head数量。
- $D$：原始Q/K Head Dim，当前固定为128。
- $R$：每个代表向量聚合的Token数，$R=B_s/T_s$，当前固定为8。
- $D_f$：`qflat`和`kflat`的最后一维，$D_f=T_sD$，当前固定为2048。
- $Q_{\max}$：最大Query Block数，对应`qflat`第2维。
- $K_{\max}$：最大KV Block数，对应`kflat`第2维。
- $Q_b$：第$b$个Batch的有效Query Block数，$Q_b=\lceil L_b^q/B_s\rceil$，其中$L_b^q$为`q_seq_lens[b]`。
- $K_b$：第$b$个Batch的有效KV Block数，$K_b=\lceil L_b^k/B_s\rceil$，其中$L_b^k$为`kv_seq_lens[b]`。
- $K_{b,i}^{\mathrm{vis}}$：考虑Causal语义后，第$b$个Batch中第$i$个Query Block实际可见的KV Block数。
- $M$：Metadata Tensor的INT32元素数量。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| `qflat` | 输入 | Q侧块级压缩表示。 | Kernel按连续布局处理；标准执行路径声明了AutoContiguous。 | BF16 | ND | $(B,N_q,Q_{\max},D_f)$ |
| `kflat` | 输入 | K侧块级压缩表示。 | 最后一维必须与`qflat`一致。Kernel按连续布局处理；标准执行路径声明了AutoContiguous。 | BF16 | ND | $(B,N_k,K_{\max},D_f)$ |
| `vbias` | 输入 | Value量值偏置，即OAM项。 | Batch、KV Head和KV Block维度必须与`kflat`一致。 | FLOAT32 | ND | $(B,N_k,K_{\max})$ |
| `q_seq_lens` | 输入 | 每个Batch中Query的有效Token数，不是Query Block数。 | 算子按$Q_b=\lceil\text{q\_seq\_lens}[b]/B_s\rceil$计算该Batch参与计算的有效Query Block数；不足一个完整Block的尾部Token计为一个有效Block。 | INT32 | ND | $(B)$ |
| `kv_seq_lens` | 输入 | 每个Batch中KV的有效Token数，不是KV Block数。 | 算子按$K_b=\lceil\text{kv\_seq\_lens}[b]/B_s\rceil$计算该Batch参与计算的有效KV Block数；不足一个完整Block的尾部Token计为一个有效Block。 | INT32 | ND | $(B)$ |
| `num_prompt_tokens` | 可选输入 | 每个Batch的Prompt Token数。 | 用于Position-Decay动态TopK预算分档；未传入时复用`kv_seq_lens`。 | INT32 | ND | $(B)$ |
| `metadata` | 可选输入 | 分核调度信息。 | 接口声明为可选输入，但当前计算必须使用有效Metadata；未传入时Tiling返回参数错误。 | INT32 | ND | $(M)$ |
| `causal` | 输入属性 | 是否采用Right-down Causal语义。 | 默认值为`true`。 | BOOL | - | - |
| `stem_block_size` | 输入属性 | 一个Stem Block包含的原始Token数。 | 当前仅支持128，默认值为128。 | INT64 | - | - |
| `stem_stride` | 输入属性 | Stem Block内部的分组数/聚合Stride。 | 当前仅支持16，默认值为16。 | INT64 | - | - |
| `alpha` | 输入属性 | 控制动态TopK预算随Query位置的衰减程度。 | $K_s$表示序列前部Query Block的初始TopK块数；随着Query位置向后移动，TopK预算线性衰减，结束预算为$K_e=K_s\alpha$。取值范围为$(0,1]$，默认值为1.0；$\alpha=1.0$表示不衰减，值越小表示衰减越强、序列后部选择的Key Block越少。 | FLOAT32 | - | - |
| `initial_blocks` | 输入属性 | 开头固定保留的Sink Block数。 | 当前仅支持4，默认值为4。 | INT64 | - | - |
| `window_size` | 输入属性 | 末尾固定保留的Window Block数。 | 当前仅支持4，默认值为4。 | INT64 | - | - |
| `k_block_num_rate_medium` | 输入属性 | 中等长度Prompt的TopK预算系数。 | 当前仅支持0.2，默认值为0.2。 | FLOAT32 | - | - |
| `k_block_num_bias_medium` | 输入属性 | 中等长度Prompt的TopK预算偏置。 | 当前仅支持30，默认值为30。 | INT64 | - | - |
| `k_block_num_rate_large` | 输入属性 | 长Prompt的TopK预算系数。 | 当前仅支持0.1，默认值为0.1。 | FLOAT32 | - | - |
| `k_block_num_bias_large` | 输入属性 | 长Prompt的TopK预算偏置。 | 当前仅支持30，默认值为30。 | INT64 | - | - |
| `topk_score_precision` | 输入属性 | TopK内部可排序Score的存储精度。 | 1表示UINT32，2表示UINT16，默认值为1；不改变输出Tensor的数据类型。 | INT64 | - | - |
| `sparse_indices` | 输出 | 选中的Key Block逻辑索引。 | 每行仅前`sparse_seq_len`项有效，尾部无效区填充为-1；调用者不应依赖有效索引的排列顺序。 | INT32 | ND | $(B,N_q,Q_{\max},K_{\max})$ |
| `sparse_seq_len` | 输出 | 每个Query Block对应的有效Key Block数量。 | 无有效Query/KV任务时对应值为0。 | INT32 | ND | $(B,N_q,Q_{\max})$ |

## 约束说明

### 单参数约束

- 该接口支持图模式。
- 当前仅适配Ascend 950PR/Ascend 950DT，不支持A3、A2及其他产品。
- `qflat`、`kflat`仅支持BF16、ND格式和4D Tensor。
- `vbias`仅支持FLOAT32、ND格式和3D Tensor。
- `q_seq_lens`、`kv_seq_lens`仅支持INT32、ND格式和1D Tensor；传入`num_prompt_tokens`或`metadata`时同样仅支持INT32、ND格式和1D Tensor。
- `sparse_indices`、`sparse_seq_len`的数据类型固定为INT32，数据格式为ND。
- `stem_block_size`固定为128，`stem_stride`固定为16，因此$R=8$、$D_f=2048$。
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

- $B$的取值范围为$[1,65536]$，$Q_{\max}$和$K_{\max}$必须大于0。
- $N_q$仅支持32或64，$N_k$仅支持2、4或8，并满足$N_q\bmod N_k=0$。
- `qflat`和`kflat`的Batch维、最后一维必须一致，最后一维固定为2048。
- `vbias`的shape必须为$(B,N_k,K_{\max})$，与`kflat`对应维度一致。
- `q_seq_lens`和`kv_seq_lens`的shape必须为$(B)$；传入`num_prompt_tokens`时，其shape也必须为$(B)$。
- `q_seq_lens[b]`应满足$0\leq\text{q\_seq\_lens}[b]\leq Q_{\max}B_s$。
- `kv_seq_lens[b]`应满足$0\leq\text{kv\_seq\_lens}[b]\leq K_{\max}B_s$。
- 有效Prompt Token数应为非负值，并满足`effective_num_prompt_tokens[b] >= kv_seq_lens[b]`；其中传入`num_prompt_tokens`时取其值，未传入时`effective_num_prompt_tokens = kv_seq_lens`。
- `sparse_indices`的shape由输入推导为$(B,N_q,Q_{\max},K_{\max})$；`sparse_seq_len`的shape由输入推导为$(B,N_q,Q_{\max})$。
- `sparse_indices`有效前缀中的元素为Key Block逻辑索引，第$b$个Batch中的取值范围为$[0,K_b-1]$。
- `sparse_seq_len`每个元素的取值范围为$[0,K_{b,i}^{\mathrm{vis}}]$，并且不大于对应输出行的$K_{\max}$。

### 特性交叉约束

- `metadata`必须由与主算子相同的`q_seq_lens`、`kv_seq_lens`、$N_q$、$N_k$、`causal`、`stem_block_size`、$D_f$和`window_size`生成。
- `metadata`最大Section数为$BN_k$。令$C_{\max}$表示最大Section数、$E$表示对齐前所需元素数，则容量$M$按以下公式计算，单位为INT32元素：

    $$
    C_{\max}=BN_k,\qquad E=16+C_{\max}(36+72)\times16,\qquad M=\operatorname{AlignUp}(E,4096)
    $$

- `causal=false`时，每个有效Query Block可以在实际KV Block范围内选块；`causal=true`时按Right-down Causal语义限制可见范围。
- Sink Block和Window Block不参与普通TopK候选，最终按Sink Block、普通TopK结果、Window Block的顺序拼接；实现不执行额外的去重操作。
- `topk_score_precision`只影响内部Score排序精度；无论取1还是2，`sparse_indices`与`sparse_seq_len`均为INT32。
- Tensor元素值相关约束由调用者保证；接口的Shape/Dtype校验不能替代对输入Tensor内容的合法性检查。

## 动态TopK预算说明

令$L_b^p$表示第$b$个Batch的有效Prompt Token数，首先将其转换为Prompt Block数：

$$
P_b=\left\lceil\frac{L_b^p}{B_s}\right\rceil
$$

根据$P_b$计算初始预算$K_s$：

$$
K_s=
\begin{cases}
P_b, & P_b<56,\\
\lfloor0.2P_b+30\rfloor, & 56\leq P_b<160,\\
\lfloor0.1P_b+30\rfloor, & P_b\geq160.
\end{cases}
$$

位置衰减终点为：

$$
K_e=\alpha K_s
$$

对于第$i$个Query Block，令$p_i=i+K_b-Q_b$表示其Right-down对齐位置，$\Delta_b=P_b-K_s$表示衰减区间长度。当$p_i<K_s$或$\Delta_b\leq1$时，$K_i=K_s$；否则：

$$
t_i=\frac{p_i-K_s}{\Delta_b-1}
$$

$$
K_i=\operatorname{clamp}\left(\left\lfloor K_s+t_i(K_e-K_s)\right\rfloor,1,K_s\right)
$$

实际普通TopK数量还会限制为不超过256。最终结果额外固定保留最多4个Sink Block和4个Window Block，并且有效索引数量不超过当前Query Block的实际可见KV Block数量。

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
