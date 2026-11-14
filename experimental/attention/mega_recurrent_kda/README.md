# mega_recurrent_kda

[English](README_en.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

`mega_recurrent_kda` 用于推理阶段的 KDA 递推。一次调用完成输入投影、因果 Conv1D、SiLU、KDA 递推、RMSNorm、输出门控和输出投影。算子支持等长的 BSH 输入和按 token 拼接的 TH 输入。

### 输入投影

以单个请求为例，令 $x_t$ 为本轮第 $t$ 个 token 的输入列向量（$t$ 从 1 开始）。$W_{\mathrm{qkv}}$ 对应 QKV 权重，$W_a$ 和 $W_b$ 对应两级 decay 权重，$W_\beta$ 和 $W_z$ 分别对应 beta 和输出门控权重。以下公式省略中间舍入。

$$
\begin{aligned}
p_t &= W_{\mathrm{qkv}}x_t,\\
a_t &= W_a x_t,\qquad f_t = W_b a_t,\\
b_t &= W_\beta x_t,\qquad z_t = W_z x_t.
\end{aligned}
$$

其中 $x_t$ 为长度 $H$ 的列向量；$p_t$ 为 QKV 投影输出（长度 $3P$），$a_t$ 为第一级 decay 投影输出（长度 $D$），$f_t$ 为第二级 decay 投影输出（长度 $P$），$b_t$ 为 beta 投影输出（长度 $N$，每个 head 对应一个标量），$z_t$ 为输出门控投影输出（长度 $P$）。

### 因果 Conv1D 与 SiLU

四个卷积权重向量 $w_0,\ldots,w_3$（各长度 $3P$）分别作用于从最早历史到当前 token 的四行。令 $s$ 为当前请求的物理块号，$o=\max(\mathrm{count}-1,0)$，其中 $\mathrm{count}$ 对应 `conv_num_accepted_tokens`。$p_{-2},p_{-1},p_0$ 分别取自 `conv_state` 中块 $s$ 的第 $o,o+1,o+2$ 行。$\odot$ 表示逐元素乘法，$\sigma$ 表示 sigmoid。

$$
\begin{aligned}
c_t &= \sum_{j=0}^{3} w_j \odot p_{t-3+j},\\
(\hat q_t,\hat k_t,v_t)
&= \operatorname{Split}_{Q,K,V}\!\left(c_t\odot\sigma(c_t)\right).
\end{aligned}
$$

其中 $p_{t-3+j}$ 为卷积窗口第 $j$ 行的输入向量（长度 $3P$）：下标 $\le 0$ 时取上述 `conv_state` 历史行，下标 $\ge 1$ 时取本轮对应 token 的原始 QKV 投影，包括当前 token。$c_t$ 为卷积输出向量（长度 $3P$），$c_t\odot\sigma(c_t)$ 为 SiLU 激活。$\operatorname{Split}_{Q,K,V}$ 将激活结果按通道顺序均分为三段，得到 Q、K、V 向量 $\hat q_t$、$\hat k_t$、$v_t$（各长度 $P$）。

按 Q、K、V 拆分后，每组向量再按 head 拆为 $N$ 个长度为 $D$ 的向量。

### KDA 递推

第 $h$ 个 head 的初始状态 $M_{0,h}$ 从同一物理块的 `checkpoint_state` 读取。$A_h$、$\delta_h$ 和 $\ell$ 分别对应 `A_log[h]`、`dt_bias[h]` 和 `lower_bound`。$f_{t,h}$ 是第 $h$ 个 head 的 decay 投影向量，$b_{t,h}$ 是该 head 的 beta 投影标量。

$$
\begin{aligned}
q_{t,h} &= \frac{\mathrm{scale}\,\hat q_{t,h}}{\sqrt{\lVert\hat q_{t,h}\rVert_2^2+10^{-6}}},\\
k_{t,h} &= \frac{\hat k_{t,h}}{\sqrt{\lVert\hat k_{t,h}\rVert_2^2+10^{-6}}},\\
d_{t,h} &= \exp\!\left(\ell\,\sigma\!\left(e^{A_h}(f_{t,h}+\delta_h)\right)\right),\\
G_{t,h} &= M_{t-1,h}\operatorname{diag}(d_{t,h}),\\
u_{t,h} &= \sigma(b_{t,h})\left(v_{t,h}-G_{t,h}k_{t,h}\right),\\
M_{t,h} &= G_{t,h}+u_{t,h}k_{t,h}^{\mathsf T},\\
r_{t,h} &= M_{t,h}q_{t,h}.
\end{aligned}
$$

其中 $\mathrm{scale}$ 为 query 缩放系数，对应参数 `scale`；$q_{t,h}$、$k_{t,h}$ 为归一化后的 query 和 key 向量（各长度 $D$），分母中的 $10^{-6}$ 为固定常量，与 `rms_norm_eps` 无关。$d_{t,h}$ 为逐通道衰减系数向量（长度 $D$）；$\operatorname{diag}(d_{t,h})$ 为以 $d_{t,h}$ 为对角线的 $D\times D$ 对角矩阵。$G_{t,h}$ 为衰减后的状态矩阵（$D\times D$）；$u_{t,h}$ 为 beta 门控的新息向量（长度 $D$）；$M_{t,h}$ 为递推状态矩阵（$D\times D$），初值 $M_{0,h}$ 来自 `checkpoint_state`；$r_{t,h}=M_{t,h}q_{t,h}$ 为该 head 的递推输出向量（长度 $D$）。

三个 replay 池分别记录 $u_{t,h}$、归一化的 $k_{t,h}$（以 FP16 存储）和指数化后的 $d_{t,h}$。

### 最终输出总公式

以更新后的状态 $M_{t,h}$ 计算各 head 的递推输出，再执行带权 RMSNorm、sigmoid 门控、head 拼接和输出投影。令 $\gamma$、$\epsilon$ 和 $W_o$ 分别对应 `output_norm_weight`、`rms_norm_eps` 和 `output_projection_weight`，则 `out` 在第 $t$ 个 token 处为

$$
\begin{aligned}
y_t &= W_o\,\operatorname{Concat}_{h=1}^{N}
\left[
  \frac{\gamma \odot (M_{t,h}q_{t,h})}
       {\sqrt{\lVert M_{t,h}q_{t,h}\rVert_2^2/D+\epsilon}}
  \odot \sigma\!\left((W_z x_t)_h\right)
\right].
\end{aligned}
$$

其中 $y_t$ 为第 $t$ 个 token 的输出列向量（长度 $H$），对应返回值 `out`；$\operatorname{Concat}_{h=1}^{N}$ 将 $N$ 个 head 的门控输出按 head 顺序拼接为长度 $P$ 的向量；$\lVert M_{t,h}q_{t,h}\rVert_2^2/D$ 表示递推输出 $r_{t,h}=M_{t,h}q_{t,h}$ 的元素平方和除以 $D$（取均方），整个分母为 RMSNorm 归一化项；$(W_z x_t)_h=z_{t,h}$ 是当前输入的输出门控投影在第 $h$ 个 head 上的切片。

状态按物理块存放。每个逻辑请求通过 `state_indices` 选择一个块，Conv 历史、checkpoint 和三个 replay 池共用这个块号。各池可以使用不同的第 0 轴步长。

| 状态 | 调用行为 |
|---|---|
| `checkpoint_state` | 读取请求的递推初始状态，调用期间保持只读。 |
| `conv_state` | 原地更新所选块的卷积历史和本轮 QKV 投影。 |
| `replay_u`、`replay_k`、`replay_decay` | 原地写入所选块的有效 token 前缀，保留未选块、记录尾部和块间间隙。 |

## 接口说明

调用 `torch.ops.cann_ops_transformer.mega_recurrent_kda`。

### PyTorch 接口原型

```python
def mega_recurrent_kda(
    hidden_states: torch.Tensor,
    qkv_projection_weight: torch.Tensor,
    decay_projection_a_weight: torch.Tensor,
    decay_projection_b_weight: torch.Tensor,
    beta_projection_weight: torch.Tensor,
    output_gate_projection_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv_state: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replay_u: torch.Tensor,
    replay_k: torch.Tensor,
    replay_decay: torch.Tensor,
    state_indices: torch.Tensor,
    conv_num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    rms_norm_eps: float = 1e-5,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ...
```

## 参数说明

本文按从 1 开始的方式描述张量的第几维。代码索引如 `shape[2]` 和 `stride(0)` 从 0 开始。

### 维度符号说明

- `B` 本次调用的逻辑请求数。BSH 取输入第 1 维，TH 取 `cu_seqlens.numel() - 1`。
- `seq_len` BSH 中每个请求的 token 数。
- `total_tokens` TH 中各请求 token 数之和。
- `H` 隐藏维度，固定为 7168。
- `N` 当前 TP 实例的 head 数。TP4、TP8、TP16 分别对应 24、12、6。
- `D` 每个 head 的维度，固定为 128。
- `P` 投影通道数，`P = N * D`。
- `block_num` 五个状态池共用的物理块数，对应各池第 1 维。
- `L` 每个 Conv 块的行数，对应 `conv_state` 第 2 维，至少为最长序列长度加 2。
- `S` 每个 replay 块中每个 head 的 token 容量，对应 replay 池第 3 维，至少覆盖最长序列。

### 参数表

| 参数名 | 输入/输出/属性 | 必选/可选 | 数据类型/格式 | 形状或取值 | 描述 |
| --- | --- | --- | --- | --- | --- |
| `hidden_states` | 输入 | 必选 | BF16 / ND | `[B,seq_len,H]` 或 `[total_tokens,H]` | BSH 或 TH 输入。 |
| `qkv_projection_weight` | 输入 | 必选 | BF16 / NZ | `[3*P,H]` | Q、K、V 投影权重。 |
| `decay_projection_a_weight` | 输入 | 必选 | BF16 / NZ | `[D,H]` | 第一级 decay 投影。 |
| `decay_projection_b_weight` | 输入 | 必选 | BF16 / NZ | `[P,D]` | 第二级 decay 投影。 |
| `beta_projection_weight` | 输入 | 必选 | BF16 / NZ | `[N,H]` | beta 投影。 |
| `output_gate_projection_weight` | 输入 | 必选 | BF16 / NZ | `[P,H]` | 输出门控投影。 |
| `output_norm_weight` | 输入 | 必选 | FP32 / ND | `[D]` | 输出 RMSNorm 权重。 |
| `output_projection_weight` | 输入 | 必选 | BF16 / NZ | `[H,P]` | 输出投影。 |
| `conv1d_weight` | 输入 | 必选 | BF16 / ND | `[4,3*P]`、`[3,N,4,D]` 或 `[3,4,D]` | Conv 权重，卷积核长度为 4。 |
| `conv_state` | 输入/输出 | 必选 | BF16 / ND | `[block_num,L,3*P]` | PA Conv 池，原地更新。 |
| `checkpoint_state` | 输入 | 必选 | FP32 / ND | `[block_num,N,D,D]` | PA checkpoint 池，只读。 |
| `replay_u` | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA 更新向量 U 记录池，原地更新。 |
| `replay_k` | 输入/输出 | 必选 | FP16 / ND | `[block_num,N,S,D]` | PA 归一化 K 记录池，原地更新。 |
| `replay_decay` | 输入/输出 | 必选 | FP32 / ND | `[block_num,N,S,D]` | PA 指数化衰减系数记录池，原地更新。 |
| `state_indices` | 输入 | 必选 | int32 / ND | `[B]` | 每个逻辑 batch 在五个池中共用的块号。 |
| `conv_num_accepted_tokens` | 输入 | 必选 | int32 / ND | `[B]` | 选择 Conv 历史窗口的计数。 |
| `A_log` | 输入 | 必选 | FP32 / ND | `[N]` | log decay 参数。 |
| `dt_bias` | 输入 | 必选 | FP32 / ND | `[N,D]` | decay bias。 |
| `scale` | 属性 | 必选 | FP32 scalar | — | 有限 query scale，常用 `D**-0.5`。 |
| `lower_bound` | 属性 | 必选 | FP32 scalar | — | decay gate 下界，在 `[-5,0]` 内。 |
| `rms_norm_eps` | 属性 | 可选（默认 `1e-5`） | FP32 scalar | — | 输出 RMSNorm epsilon。 |
| `cu_seqlens` | 输入 | 可选（TH 必传） | int32/int64 / ND | `[B+1]` 或 None | TH 的序列前缀和；BSH 必须为 None。 |
| `out` | 输出 | — | BF16 / ND | `[B,seq_len,H]` / `[total_tokens,H]` | 主输出，形状与输入一致。 |

## 返回值

返回 `(out, conv_state, replay_u, replay_k, replay_decay)`。`out` 是与 `hidden_states` 同形状的 BF16 张量，后四项依次返回对应的输入池并保留其存储布局。`checkpoint_state` 为只读输入。

## 约束说明

### 约束类型说明

- **单参数约束** 说明单个张量的形状、类型、格式和数值范围，或单个属性的取值。
- **存在性约束** 说明特定场景需要提供或省略的参数。
- **一致性约束** 说明同组参数的维度、类型或存储关系。
- **特性交叉约束** 说明不同组之间的维度、容量或执行条件。

### 特性参数组

分组按计算阶段和职责划分。可选字段以外的输入均须提供。四个 `INPUT/OUTPUT` 张量原地更新并作为返回值返回。

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| --- | --- | --- | --- |
| 公共 | `hidden_states` | INPUT | Tensor |
|  | `out` | OUTPUT | Tensor |
| QKV 投影 | `qkv_projection_weight` | INPUT | Tensor |
| 因果 Conv1D | `conv1d_weight` | INPUT | Tensor |
|  | `conv_state` | INPUT/OUTPUT | Tensor |
|  | `conv_num_accepted_tokens` | INPUT | Tensor |
| 衰减投影与门控 | `decay_projection_a_weight` | INPUT | Tensor |
|  | `decay_projection_b_weight` | INPUT | Tensor |
|  | `A_log` | INPUT | Tensor |
|  | `dt_bias` | INPUT | Tensor |
|  | `lower_bound` | ATTR | float |
| Beta 投影与查询缩放 | `beta_projection_weight` | INPUT | Tensor |
|  | `scale` | ATTR | float |
| 分页递推与回放 | `state_indices` | INPUT | Tensor |
|  | `checkpoint_state` | INPUT | Tensor |
|  | `replay_u` | INPUT/OUTPUT | Tensor |
|  | `replay_k` | INPUT/OUTPUT | Tensor |
|  | `replay_decay` | INPUT/OUTPUT | Tensor |
| 输出归一化、门控与投影 | `output_norm_weight` | INPUT | Tensor |
|  | `rms_norm_eps` | ATTR(OPTIONAL) | float |
|  | `output_gate_projection_weight` | INPUT | Tensor |
|  | `output_projection_weight` | INPUT | Tensor |
| 变长输入 | `cu_seqlens` | INPUT(OPTIONAL) | Tensor |

### 基准信息说明

| 命名 | 含义 |
| --- | --- |
| `max_seq_len` | BSH 取 `seq_len`，TH 取 `cu_seqlens` 的最大相邻差值。 |
| `slot` | 当前请求的物理块号，取 `state_indices[b]`。 |
| `pool.stride(0)` | 各状态池相邻物理块之间的元素间距。 |

### 参数组约束

各条约束按拦截行为标注，标注含义如下：

- 【已拦截】调用时同步校验，违反即抛出 `ValueError` 或 `TypeError`，且算子不改动任何状态池（原子拒绝）。校验覆盖形状、数据类型、格式、布局、标量取值范围和存储别名关系，不校验张量数值内容（如权重、`A_log`、`dt_bias` 和 checkpoint 初值的数值），数值正确性由调用方保证。
- 【已拦截（主机侧）】取值校验需要把设备张量读到主机执行；图捕获期间这类值校验自动跳过，重放前须由调用方写入合法值（见[图捕获](#图捕获)）。
- 【不拦截】算子不做校验或无需校验的情形，包括由调用方负责的内容与语义约定，以及由算子执行行为天然保证的结果。
- 同一条约束内的混合语句在句内单独注明拦截状态。

#### 公共参数组

- **单参数约束【已拦截】** `hidden_states` 为 BF16/ND，隐藏维度固定为 7168。BSH 的批量和序列长度为正，TH 的 token 数为正，批量满足 `1 <= B <= block_num`。两种布局的总 token 数均小于 `2**31`（主机侧值校验，图捕获期间跳过）。

- **存在性约束【已拦截】** 接口用于推理。除 `rms_norm_eps` 和 `cu_seqlens` 外，所有输入均须提供，缺失参数由接口 schema 直接拒绝。使用普通 Tensor，Parameter 先调用 `.detach()`，未 detach 的 Parameter 输入会被拒绝。

- **一致性约束【已拦截】** 所有张量位于同一 NPU，设备一致性逐张量校验并要求位于 NPU 设备。主输出与输入的形状、布局和数据类型一致（不拦截：输出由算子按输入形状和布局分配，天然成立）。

- **特性交叉约束【已拦截】** 六个投影权重须预先转换为 NZ，其余张量使用 ND，NZ/ND 格式逐张量校验。五个状态池以外的输入张量必须连续。

#### QKV 投影参数组

- **单参数约束【已拦截】** `qkv_projection_weight` 为 BF16/NZ，形状为 `[3*P,H]`，依次对应 Q、K、V。Q、K、V 通道的排列顺序属于数据内容约定（不拦截：算子按固定顺序切分，不检查权重内容排布）。

- **特性交叉约束【已拦截】** 输入宽度与 `hidden_states` 的 `H` 一致。输出的 `3*P` 通道与 Conv 权重及历史池匹配（Conv 权重形状枚举和 `conv_state` 第 3 维均按同一 `3*P` 校验）。

#### 因果 Conv1D参数组

- **单参数约束【已拦截】** `conv1d_weight` 支持 `[4,3*P]`、`[3,N,4,D]` 或 `[3,4,D]`，固定 4 个卷积 tap。`conv_state` 为 BF16/ND，形状为 `[block_num,L,3*P]`，`L >= 3` 且 `block_num` 为正。

- **一致性约束【已拦截（主机侧）】** `conv_num_accepted_tokens` 为 `[B]` 的 int32 张量，取值为 `[0,L-2]`。历史从 `max(count-1,0)` 开始读取三行、后两行写回块首、本轮未卷积的 QKV 投影紧随其后均为执行行为（不拦截）；计数值与真实历史长度是否语义一致由调用方保证（不拦截）。

- **特性交叉约束【已拦截】** 按 `state_indices` 选择历史块。若本次最长请求有 `max_seq_len` 个 token，须满足 `L >= max_seq_len + 2`（BSH 在调用入口校验；TH 依赖主机侧序列长度，图捕获期间跳过）。

#### 衰减投影与门控参数组

- **单参数约束【已拦截】** 两级权重均为 BF16/NZ，形状依次为 `[D,H]` 和 `[P,D]`。`A_log`、`dt_bias` 为 FP32/ND。`lower_bound` 为有限 FP32，取值为 `[-5,0]`。

- **一致性约束【已拦截】** 第一级的输出宽度等于第二级的输入宽度 `D`。第二级的输出按 `[N,D]` 分组，与 `A_log[N]` 和 `dt_bias[N,D]` 的 head 对齐。

#### Beta 投影与查询缩放参数组

- **单参数约束【已拦截】** `beta_projection_weight` 为 BF16/NZ，形状为 `[N,H]`。`scale` 须可表示为有限 FP32；其具体取值不做限制（不拦截：常用值为 `D**-0.5`，数值效果由调用方负责）。Q/K 归一化使用固定的 `1e-6`（编译期常量，无需拦截）。

- **特性交叉约束【已拦截】** `N` 取 24、12 或 6，分别对应 TP4、TP8、TP16。所有权重与状态池按同一 `N` 和 `D=128` 组织。

#### 分页递推与回放参数组

- **单参数约束【已拦截（主机侧）】** `state_indices` 为 `[B]` 的 int32 张量，块号在 `[0,block_num)` 内，同次调用互不重复（含块 0，块 0 是普通有效块）。块号取值与去重校验需读到主机，图捕获期间跳过。`checkpoint_state` 为只读输入属于行为承诺（不拦截：算子不写入该池，无需校验）。

- **一致性约束【已拦截】** 三个 replay 池的形状均为 `[block_num,N,S,D]`，分别存储 FP32 更新向量、FP16 归一化 key 和 FP32 指数化衰减系数，`S` 为正。只写所选块的有效 token 前缀，保留其他块、记录尾部及块间间隙均为执行行为（不拦截）。

- **特性交叉约束【已拦截】** Conv、checkpoint 和三个 replay 池共享 `block_num` 与每个请求的物理块号。各池的 `stride(0)` 可独立设置，但至少为一个块的元素数，其余轴必须连续。四个可写池须设置 `requires_grad=False`，且不得与其他输入的有效存储区重叠（别名重叠逐对校验）。回放容量须满足 `S >= max_seq_len`（BSH 在调用入口校验；TH 依赖主机侧序列长度，图捕获期间跳过）。

例如，`state_indices=[5,0]` 且两个请求分别有 1 和 3 个 token 时，算子分别写入物理块 5 的第 0 条 replay 记录和块 0 的前 3 条记录。两块剩余记录以及其他块保留原值。若第二个请求的 Conv 计数为 2，则从块 0 的第 1 行开始读取历史。

#### 输出归一化、门控与投影参数组

- **单参数约束【已拦截】** `output_norm_weight` 为 `[D]` 的 FP32/ND 张量，门控权重和输出投影权重分别为 `[P,H]` 和 `[H,P]` 的 BF16/NZ 张量。`rms_norm_eps` 必须为正的正规 FP32 数。

- **存在性约束【已拦截】** 省略 `rms_norm_eps` 时使用 `1e-5`（由接口默认值保证）。

- **特性交叉约束【已拦截】** RMSNorm 权重在各 head 间共享。门控投影与拼接后的 `P` 个通道对齐，最终投影恢复到隐藏宽度 `H`。

#### 变长输入参数组

- **单参数约束【已拦截】** `cu_seqlens` 为 `[B+1]` 的 int32 或 int64 连续 ND 张量。

- **存在性约束【已拦截】** TH 必须传入 `cu_seqlens`，BSH 使用 `None`。

- **一致性约束【已拦截（主机侧）】** 前缀和从 0 开始，严格递增，末项等于 `total_tokens`。每个相邻差值是对应请求的有效 token 数。值校验需读到主机，图捕获期间跳过。

- **特性交叉约束【已拦截（主机侧）】** 最长相邻差值决定 `max_seq_len`，同时用于 Conv 和 replay 容量检查。TH 需要将前缀和读到主机，因此不支持图捕获（属执行限制而非校验项，无法在算子内拦截，由调用方避免在 TH 输入上捕获）。

### 图捕获

BSH 支持图捕获。捕获前需预热算子与工作区，重放前须提供有效且互不重复的块号和合法的 Conv 计数。调用方负责共享工作区的生命周期与串行使用。TH 需要在主机读取序列边界，因此不支持捕获。

## 调用说明

安装仓库构建的算子包和 PyTorch 扩展后，导入扩展包以注册接口，再通过 `torch.ops.cann_ops_transformer.mega_recurrent_kda` 调用。扩展包名随构建配置而定。下面的示例使用默认包名。

### 等长 BSH

准备形状为 `[2,4,7168]` 的 BF16/ND `hidden_states`、六个 NZ 投影权重、`conv1d_weight`、`output_norm_weight`、`A_log` 和 `dt_bias`。下例创建状态池和请求索引，两个请求分别使用块 5 和块 0。切片展示各池可使用独立的 `stride(0)`。

```python
import torch
import cann_ops_transformer

block_num, N, D = 6, 12, 128
S, L = 8, 10
P = N * D
device = hidden_states.device

conv_state = torch.zeros(2*block_num, L, 3*P, dtype=torch.bfloat16, device=device)[::2]
checkpoint_state = torch.zeros(3*block_num, N, D, D, dtype=torch.float32, device=device)[::3]
replay_u = torch.zeros(2*block_num, N, S, D, dtype=torch.float32, device=device)[::2]
replay_k = torch.zeros(3*block_num, N, S, D, dtype=torch.float16, device=device)[::3]
replay_decay = torch.zeros(4*block_num, N, S, D, dtype=torch.float32, device=device)[::4]
state_indices = torch.tensor([5, 0], dtype=torch.int32, device=device)
conv_num_accepted_tokens = torch.zeros(2, dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
    rms_norm_eps=1e-5,
    cu_seqlens=None,
)
# 第一个请求的记录写入块 5，第二个请求的记录写入块 0。
```

### 变长 TH

使用同一组权重。将两个请求的输入依次拼成形状为 `[4,7168]` 的 `th_hidden_states`，两个请求分别含 1 和 3 个 token。用同一设备上的前缀和指定边界。下例重新初始化上面的四个可写池，作为一次独立调用。

```python
for pool in (conv_state, replay_u, replay_k, replay_decay):
    pool.zero_()
cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int32, device=device)

out, conv_state, replay_u, replay_k, replay_decay = torch.ops.cann_ops_transformer.mega_recurrent_kda(
    hidden_states=th_hidden_states,
    qkv_projection_weight=qkv_projection_weight,
    decay_projection_a_weight=decay_projection_a_weight,
    decay_projection_b_weight=decay_projection_b_weight,
    beta_projection_weight=beta_projection_weight,
    output_gate_projection_weight=output_gate_projection_weight,
    output_norm_weight=output_norm_weight,
    output_projection_weight=output_projection_weight,
    conv1d_weight=conv1d_weight,
    conv_state=conv_state,
    checkpoint_state=checkpoint_state,
    replay_u=replay_u,
    replay_k=replay_k,
    replay_decay=replay_decay,
    state_indices=state_indices,
    conv_num_accepted_tokens=conv_num_accepted_tokens,
    A_log=A_log,
    dt_bias=dt_bias,
    scale=D**-0.5,
    lower_bound=-5.0,
    rms_norm_eps=1e-5,
    cu_seqlens=cu_seqlens,
)
```

完整接口约束见 [PyTorch API 文档](docs/torchapi_mega_recurrent_kda.md)。

## 执行路径

| 源码目录 | 路径选择 |
|---|---|
| [arch35](op_dsl/arch35/) | 等长 4 或 8 token 且批量满足优化条件时使用 S4 或 S8 专用路径，其余合法输入使用通用路径。 |
| [arch92](op_dsl/arch92/) | BSH 和 TH 输入均使用通用路径。 |

支持 TP4、TP8 和 TP16，每个实例分别使用 24、12 和 6 个 head。设模型批量为 `b`，算子批量为 `B=b*TP`，优化路径覆盖 `b` 为 1～32 的合法等长输入。核数配额见 [执行路径与核数配额](docs/torchapi_mega_recurrent_kda.md#执行路径与核数配额)。
