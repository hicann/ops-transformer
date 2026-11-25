# MegaRecurrentKda

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

`mega_recurrent_kda`用于推理阶段的KDA递推。一次调用完成输入投影、因果Conv1D、SiLU、KDA递推、RMSNorm、输出门控和输出投影。算子支持等长的BSH输入和按token拼接的TH输入。

### 输入投影

以单个请求为例，令$x_t$为本轮第$t$个token的输入列向量（$t$从1开始）。$W_{\mathrm{qkv}}$对应QKV权重，$W_a$和$W_b$对应两级decay权重，$W_\beta$和$W_z$分别对应beta和输出门控权重。以下公式省略中间舍入。

$$
\begin{aligned}
p_t &= W_{\mathrm{qkv}}x_t,\\
a_t &= W_a x_t,\qquad f_t = W_b a_t,\\
b_t &= W_\beta x_t,\qquad z_t = W_z x_t.
\end{aligned}
$$

其中$x_t$为长度$H$的列向量；$p_t$为QKV投影输出（长度$3P$），$a_t$为第一级decay投影输出（长度$D$），$f_t$为第二级decay投影输出（长度$P$），$b_t$为beta投影输出（长度$N$，每个head对应一个标量），$z_t$为输出门控投影输出（长度$P$）。

### 因果Conv1D与SiLU

四个卷积权重向量$w_0,\ldots,w_3$（各长度$3P$）分别作用于从最早历史到当前token的四行。令$s$为当前请求的物理块号，$o=\max(\mathrm{count}-1,0)$，其中$\mathrm{count}$对应`conv_num_accepted_tokens`。$p_{-2},p_{-1},p_0$分别取自`conv_state`中块$s$的第$o,o+1,o+2$行。$\odot$表示逐元素乘法，$\sigma$表示sigmoid。

$$
\begin{aligned}
c_t &= \sum_{j=0}^{3} w_j \odot p_{t-3+j},\\
(\hat q_t,\hat k_t,v_t)
&= \operatorname{Split}_{Q,K,V}\!\left(c_t\odot\sigma(c_t)\right).
\end{aligned}
$$

其中$p_{t-3+j}$为卷积窗口第$j$行的输入向量（长度$3P$）：下标$\le 0$时取上述`conv_state`历史行，下标$\ge 1$时取本轮对应token的原始QKV投影，包括当前token。$c_t$为卷积输出向量（长度$3P$），$c_t\odot\sigma(c_t)$为SiLU激活。$\operatorname{Split}_{Q,K,V}$将激活结果按通道顺序均分为三段，得到Q、K、V向量$\hat q_t$、$\hat k_t$、$v_t$（各长度$P$）。

按Q、K、V拆分后，每组向量再按head拆为$N$个长度为$D$的向量。

### KDA递推

第$h$个head的初始状态$M_{0,h}$从同一物理块的`checkpoint_state`读取。$A_h$、$\delta_h$和$\ell$分别对应`A_log[h]`、`dt_bias[h]`和`lower_bound`。$f_{t,h}$是第$h$个head的decay投影向量，$b_{t,h}$是该head的beta投影标量。

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

其中$\mathrm{scale}$为query缩放系数，对应参数`scale`；$q_{t,h}$、$k_{t,h}$为归一化后的query和key向量（各长度$D$），分母中的$10^{-6}$为固定常量，与`rms_norm_eps`无关。$d_{t,h}$为逐通道衰减系数向量（长度$D$）；$\operatorname{diag}(d_{t,h})$为以$d_{t,h}$为对角线的$D\times D$对角矩阵。$G_{t,h}$为衰减后的状态矩阵（$D\times D$）；$u_{t,h}$为beta门控的新息向量（长度$D$）；$M_{t,h}$为递推状态矩阵（$D\times D$），初值$M_{0,h}$来自`checkpoint_state`；$r_{t,h}=M_{t,h}q_{t,h}$为该head的递推输出向量（长度$D$）。

三个replay池分别记录$u_{t,h}$、归一化的$k_{t,h}$（以FP16存储）和衰减系数$d_{t,h}$（已含指数）。

### 最终输出总公式

以更新后的状态$M_{t,h}$计算各head的递推输出，再执行带权RMSNorm、sigmoid门控、head拼接和输出投影。令$\gamma$、$\epsilon$和$W_o$分别对应`output_norm_weight`、`rms_norm_eps`和`output_projection_weight`，则`out`在第$t$个token处为

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

其中$y_t$为第$t$个token的输出列向量（长度$H$），对应返回值`out`；$\operatorname{Concat}_{h=1}^{N}$将$N$个head的门控输出按head顺序拼接为长度$P$的向量；$\lVert M_{t,h}q_{t,h}\rVert_2^2/D$表示递推输出$r_{t,h}=M_{t,h}q_{t,h}$的元素平方和除以$D$（取均方），整个分母为RMSNorm归一化项；$(W_z x_t)_h=z_{t,h}$是当前输入的输出门控投影在第$h$个head上的切片。

状态按物理块存放。每个逻辑请求通过`state_indices`选择一个块，Conv历史、checkpoint和三个replay池共用这个块号。各池可以使用不同的第0轴步长。

| 状态 | 调用行为 |
|---|---|
| `checkpoint_state` | 读取请求的递推初始状态，调用期间保持只读。 |
| `conv_state` | 原地更新所选块的卷积历史和本轮QKV投影。 |
| `replay_u`、`replay_k`、`replay_decay` | 原地写入所选块的有效token前缀，保留未选块、记录尾部和块间间隙。 |

## 参数说明

本文按从1开始的方式描述张量的第几维。代码索引如`shape[2]`和`stride(0)`从0开始。

### 维度符号说明

- `B`本次调用的逻辑请求数。BSH取输入第1维，TH取`cu_seqlens.numel() - 1`。
- `seq_len`BSH中每个请求的token数。
- `total_tokens`TH中各请求token数之和。
- `H`隐藏维度，固定为7168。
- `N`当前TP实例的head数。TP4、TP8、TP16分别对应24、12、6。
- `D`每个head的维度，固定为128。
- `P`投影通道数，`P = N * D`。
- `block_num`五个状态池共用的物理块数，对应各池第1维。
- `L`每个Conv块的行数，对应`conv_state`第2维，至少为最长序列长度加2。
- `S`每个replay块中每个head的token容量，对应replay池第3维，至少覆盖最长序列。

### 参数表

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `hidden_states` | 输入 | BSH或TH输入，形状为`[B,seq_len,H]`或`[total_tokens,H]`。 | BF16 | ND |
| `qkv_projection_weight` | 输入 | Q、K、V投影权重，形状为`[3*P,H]`。 | BF16 | NZ |
| `decay_projection_a_weight` | 输入 | 第一级decay投影，形状为`[D,H]`。 | BF16 | NZ |
| `decay_projection_b_weight` | 输入 | 第二级decay投影，形状为`[P,D]`。 | BF16 | NZ |
| `beta_projection_weight` | 输入 | beta投影，形状为`[N,H]`。 | BF16 | NZ |
| `output_gate_projection_weight` | 输入 | 输出门控投影，形状为`[P,H]`。 | BF16 | NZ |
| `output_norm_weight` | 输入 | 输出RMSNorm权重，形状为`[D]`。 | FP32 | ND |
| `output_projection_weight` | 输入 | 输出投影，形状为`[H,P]`。 | BF16 | NZ |
| `conv1d_weight` | 输入 | Conv权重，卷积核长度为4，形状为`[4,3*P]`、`[3,N,4,D]`或`[3,4,D]`。 | BF16 | ND |
| `conv_state` | 输入/输出 | PA Conv池，原地更新，形状为`[block_num,L,3*P]`。 | BF16 | ND |
| `checkpoint_state` | 输入 | PA checkpoint池，只读，形状为`[block_num,N,D,D]`。 | FP32 | ND |
| `replay_u` | 输入/输出 | PA更新向量U记录池，原地更新，形状为`[block_num,N,S,D]`。 | FP32 | ND |
| `replay_k` | 输入/输出 | PA归一化K记录池，原地更新，形状为`[block_num,N,S,D]`。 | FP16 | ND |
| `replay_decay` | 输入/输出 | PA指数化衰减系数记录池，原地更新，形状为`[block_num,N,S,D]`。 | FP32 | ND |
| `state_indices` | 输入 | 每个逻辑batch在五个池中共用的块号，形状为`[B]`。 | INT32 | ND |
| `conv_num_accepted_tokens` | 输入 | 选择Conv历史窗口的计数，形状为`[B]`。 | INT32 | ND |
| `A_log` | 输入 | log decay参数，形状为`[N]`。 | FP32 | ND |
| `dt_bias` | 输入 | decay bias，形状为`[N,D]`。 | FP32 | ND |
| `scale` | 属性 | 有限query scale，常用`D**-0.5`。 | FP32 | — |
| `lower_bound` | 属性 | decay gate下界，取值在`[-5,0]`内。 | FP32 | — |
| `rms_norm_eps` | 属性 | 可选，默认`1e-5`；输出RMSNorm epsilon，为正的正规FP32数。 | FP32 | — |
| `cu_seqlens` | 输入 | 可选，TH必传；TH的序列前缀和，形状为`[B+1]`；BSH必须为`None`。 | INT32、INT64 | ND |
| `out` | 输出 | 主输出，形状与输入一致，为`[B,seq_len,H]`或`[total_tokens,H]`。 | BF16 | ND |

五个状态池（`conv_state`、`checkpoint_state`、`replay_u`、`replay_k`、`replay_decay`）支持第0轴非连续的Tensor，其余张量不支持非连续。

## 约束说明

### 约束类型说明

- **单参数约束** 说明单个张量的形状、类型、格式和数值范围，或单个属性的取值。
- **存在性约束** 说明特定场景需要提供或省略的参数。
- **一致性约束** 说明同组参数的维度、类型或存储关系。
- **特性交叉约束** 说明不同组之间的维度、容量或执行条件。

### 特性参数组

分组按计算阶段和职责划分。可选字段以外的输入均须提供。四个`INPUT/OUTPUT`张量原地更新并作为返回值返回。

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| --- | --- | --- | --- |
| 公共 | `hidden_states` | INPUT | Tensor |
|  | `out` | OUTPUT | Tensor |
| QKV投影 | `qkv_projection_weight` | INPUT | Tensor |
| 因果Conv1D | `conv1d_weight` | INPUT | Tensor |
|  | `conv_state` | INPUT/OUTPUT | Tensor |
|  | `conv_num_accepted_tokens` | INPUT | Tensor |
| 衰减投影与门控 | `decay_projection_a_weight` | INPUT | Tensor |
|  | `decay_projection_b_weight` | INPUT | Tensor |
|  | `A_log` | INPUT | Tensor |
|  | `dt_bias` | INPUT | Tensor |
|  | `lower_bound` | ATTR | float |
| Beta投影与查询缩放 | `beta_projection_weight` | INPUT | Tensor |
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
| `max_seq_len` | BSH取`seq_len`，TH取`cu_seqlens`的最大相邻差值。 |
| `slot` | 当前请求的物理块号，取`state_indices[b]`。 |
| `pool.stride(0)` | 各状态池相邻物理块之间的元素间距。 |

### 参数组约束

#### 公共参数组

- **单参数约束** `hidden_states`为BF16/ND，隐藏维度固定为7168。BSH的批量和序列长度为正，TH的token数为正，批量满足`1 <= B <= block_num`，各维度均为正，不支持空Tensor。两种布局的总token数均小于`2**31`（主机侧值校验，aclgraph捕获期间跳过）。

- **存在性约束** 接口用于推理。除`rms_norm_eps`和`cu_seqlens`外，所有输入均须提供，缺失参数由接口schema直接拒绝。使用普通Tensor，Parameter先调用`.detach()`，未detach的Parameter输入会被拒绝。

- **一致性约束** 所有张量位于同一NPU，设备一致性逐张量校验并要求位于NPU设备。主输出与输入的形状、布局和数据类型一致（输出由算子按输入形状和布局分配，天然成立）。

- **特性交叉约束** 六个投影权重须预先转换为NZ，其余张量使用ND，NZ/ND格式逐张量校验。五个状态池以外的输入张量必须连续。

#### QKV投影参数组

- **单参数约束** `qkv_projection_weight`为BF16/NZ，形状为`[3*P,H]`，依次对应Q、K、V。Q、K、V通道的排列顺序属于数据内容约定（算子按固定顺序切分，不检查权重内容排布）。

- **特性交叉约束** 输入宽度与`hidden_states`的`H`一致。输出的`3*P`通道与Conv权重及历史池匹配（Conv权重形状枚举和`conv_state`第3维均按同一`3*P`校验）。

#### 因果Conv1D参数组

- **单参数约束** `conv1d_weight`支持`[4,3*P]`、`[3,N,4,D]`或`[3,4,D]`，固定4个卷积tap。`conv_state`为BF16/ND，形状为`[block_num,L,3*P]`，`L >= 3`且`block_num`为正。

- **一致性约束** `conv_num_accepted_tokens`为`[B]`的int32张量，取值为`[0,L-2]`。历史从`max(count-1,0)`开始读取三行、后两行写回块首、本轮未卷积的QKV投影紧随其后均为执行行为；计数值与真实历史长度是否语义一致由调用方保证。

- **特性交叉约束** 按`state_indices`选择历史块。若本次最长请求有`max_seq_len`个token，须满足`L >= max_seq_len + 2`（BSH在调用入口校验；TH依赖主机侧序列长度，aclgraph捕获期间跳过）。

#### 衰减投影与门控参数组

- **单参数约束** 两级权重均为BF16/NZ，形状依次为`[D,H]`和`[P,D]`。`A_log`、`dt_bias`为FP32/ND。`lower_bound`为有限FP32，取值为`[-5,0]`。

- **一致性约束** 第一级的输出宽度等于第二级的输入宽度`D`。第二级的输出按`[N,D]`分组，与`A_log[N]`和`dt_bias[N,D]`的head对齐。

#### Beta投影与查询缩放参数组

- **单参数约束** `beta_projection_weight`为BF16/NZ，形状为`[N,H]`。`scale`须可表示为有限FP32；其具体取值不做限制（常用值为`D**-0.5`，数值效果由调用方负责）。Q/K归一化使用固定的`1e-6`（编译期常量）。

- **特性交叉约束** `N`取24、12或6，分别对应TP4、TP8、TP16。所有权重与状态池按同一`N`和`D=128`组织。

#### 分页递推与回放参数组

- **单参数约束** `state_indices`为`[B]`的int32张量，块号在`[0,block_num)`内，同次调用互不重复（含块0，块0是普通有效块）。块号取值与去重校验需读到主机，aclgraph捕获期间跳过。`checkpoint_state`为只读输入属于行为承诺（算子不写入该池，无需校验）。

- **一致性约束** 三个replay池的形状均为`[block_num,N,S,D]`，分别存储FP32更新向量、FP16归一化key和FP32指数化衰减系数，`S`为正。只写所选块的有效token前缀，保留其他块、记录尾部及块间间隙均为执行行为。

- **特性交叉约束** Conv、checkpoint和三个replay池共享`block_num`与每个请求的物理块号。各池允许第0轴非连续，`stride(0)`可独立设置，但至少为一个块的元素数；其余轴必须连续。四个可写池须设置`requires_grad=False`，且不得与其他输入的有效存储区重叠（别名重叠逐对校验）。回放容量须满足`S >= max_seq_len`（BSH在调用入口校验；TH依赖主机侧序列长度，aclgraph捕获期间跳过）。

例如，`state_indices=[5,0]`且两个请求分别有1和3个token时，算子分别写入物理块5的第0条replay记录和块0的前3条记录。两块剩余记录以及其他块保留原值。若第二个请求的Conv计数为2，则从块0的第1行开始读取历史。

#### 输出归一化、门控与投影参数组

- **单参数约束** `output_norm_weight`为`[D]`的FP32/ND张量，门控权重和输出投影权重分别为`[P,H]`和`[H,P]`的BF16/NZ张量。`rms_norm_eps`必须为正的正规FP32数。

- **存在性约束** 省略`rms_norm_eps`时使用`1e-5`（由接口默认值保证）。

- **特性交叉约束** RMSNorm权重在各head间共享。门控投影与拼接后的`P`个通道对齐，最终投影恢复到隐藏宽度`H`。

#### 变长输入参数组

- **单参数约束** `cu_seqlens`为`[B+1]`的int32或int64连续ND张量。

- **存在性约束** TH必须传入`cu_seqlens`，BSH使用`None`。

- **一致性约束** 前缀和从0开始，严格递增，末项等于`total_tokens`。每个相邻差值是对应请求的有效token数。值校验需读到主机，aclgraph捕获期间跳过。

- **特性交叉约束** 最长相邻差值决定`max_seq_len`，同时用于Conv和replay容量检查。TH需要将前缀和读到主机，因此不支持aclgraph（属执行限制而非校验项，算子内无法校验，由调用方避免对TH输入做aclgraph捕获）。

### aclgraph图模式（ACL Graph）

BSH输入支持aclgraph图模式。捕获前需预热算子与工作区，重放前须提供有效且互不重复的块号和合法的Conv计数。调用方负责共享工作区的生命周期与串行使用。TH输入需要在主机读取序列边界，因此不支持aclgraph。

## 调用说明

安装仓库构建的算子包和PyTorch扩展后，导入扩展包以注册接口，再通过`torch.ops.cann_ops_transformer.mega_recurrent_kda`调用。扩展包名随构建配置而定。

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| PyTorch API | [调用示例](docs/torchapi_mega_recurrent_kda.md#调用示例) | 通过[torchapi_mega_recurrent_kda](docs/torchapi_mega_recurrent_kda.md)接口调用MegaRecurrentKda算子。 |

## 测试

在Ascend 950PR&950DT系列产品环境安装上述产物后，从仓库根目录运行：

```bash
python -m pytest experimental/attention/mega_recurrent_kda/tests/ut -q
```

测试通过已安装扩展调用Torch算子。
