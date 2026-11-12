# flash_kda_grad / flash_kda_grad_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term>（`dav-3510`） | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

当前实现固定使用 `C=64` 的 chunk 大小，支持 `Dk=Dv=128`。其他产品的支持情况未在本实现中声明。

## 功能说明

FlashKDAGrad 用于 Flash Kimi Delta Attention（KDA）的训练反向计算。算子接收前向输入、前向缓存、
输出梯度 `do` 和最终状态梯度 `dht`，计算 Q、K、V、初始状态、gate、beta 以及 gate 参数的梯度，
支持变长序列、GQA、Q/K L2 归一化、gate 激活和 beta sigmoid 融合。

对于一条序列的一个 value head，令 $\bar q_t$、$\bar k_t$ 为参与注意力计算的 Q/K，$\ell_t$ 为单步 log gate，
$b_t$ 为激活后的 beta，状态 $H_t$ 按 `[Dv,Dk]` 存储。对应前向计算语义为：

$$
\widetilde H_t = H_{t-1}\operatorname{diag}(\exp(\ell_t))
$$

$$
\Delta v_t = b_t\left(v_t-\widetilde H_t\bar k_t\right),\qquad
H_t = \widetilde H_t+\Delta v_t\bar k_t^T
$$

$$
o_t = scale\cdot H_t\bar q_t
$$

反向以 `do` 和 `dht` 分别作为 $o_t$ 和最终状态的上游梯度，返回：

$$
(dq,dk,dv,dh0,dg,dbeta,dA,dBias)
=
\left(
\frac{\partial L}{\partial q},
\frac{\partial L}{\partial k},
\frac{\partial L}{\partial v},
\frac{\partial L}{\partial H_0},
\frac{\partial L}{\partial g},
\frac{\partial L}{\partial beta},
\frac{\partial L}{\partial A_{log}},
\frac{\partial L}{\partial dt_{bias}}
\right)
$$

其中 `dA` 是对 `A_log` 的梯度。启用 Q/K L2 归一化时，`dq/dk` 是对归一化前 Q/K 的梯度。
上述公式描述计算语义；实现采用 chunk 分解和状态梯度逆序扫描，Cube 操作数使用 BF16，
状态缓存和状态梯度使用 FP32。

主机根据有效序列长度、head 数和设备核数自动选择 persistent 或 staged 调度，两种策略共用同一公开接口。
主反向下发一个混合 Cube/Vector 计算 kernel；默认使用前向保存的七项缓存执行反向。
显式设置 `disable_recompute=False` 时，在该 kernel 内先重算所需缓存，再执行反向。
调度信息由单独的 AICPU metadata 接口生成。

**重算实现**位于反向目录的 [flash_kda_grad_recompute.py](../op_dsl/flash_kda_grad_recompute.py)，
不调用或导入前向实现。以 STC 的 FP32 参考计算及 `cross_check L2` 为精度标准：原始 `g`、`beta` 和已保存的
beta 激活缓存保留输入精度；`W/U` 的乘法输入和 FP32 递推状态使用 BF16 主值、残差两组 Cube 乘积补偿，
最终存储的 token 缓存仍为 BF16。复用已保留的 `Akk`、归一化 Q/K 和 beta 激活缓存；padded 布局的无效
token 缓存统一清零。

**分阶段重建**：两种反向调度共用同一重建流程。先按 chunk 分核计算 `qg/kg/g_cumsum/W/U`，
全核同步并回收片上资源后，按 value 维分片递推 `chunk_states/v_new`。先读取已舍入的 BF16 `U`，
再用 FP32 减去补偿后的 `W @ H`，将 `v_new` 舍入到 BF16，并以 FP32 乘法、加法更新状态。
复用已有 metadata 和 `_staged_chunk_start` 等分核接口；重建结束后全核同步，再进入原有 persistent
或 staged backward。整个过程仍只下发一个计算 kernel。

带尾块的重算调用使用 staged backward，使重建和反向阶段之间的通道获取、释放路径保持确定；
保存缓存模式和无尾块的 persistent 重算调度不受影响。

重算与保存缓存的一致性以保留 FP32 中间计算的参考缓存为基础，并通过原生 TTK L2 验证最终梯度。
由于提高了中间计算精度，重算不再要求逐位复现真实前向的额外 BF16 操作数舍入。

## 接口说明

Torch 侧入口位于 [torch_extension/flash_kda_grad.py](../torch_extension/flash_kda_grad.py)，
注册为 `torch.ops.cann_ops_transformer.flash_kda_grad`；DSL 侧主入口和前向缓存适配入口位于
[op_dsl/flash_kda_grad.py](../op_dsl/flash_kda_grad.py)，metadata 入口位于
[op_dsl/flash_kda_grad_metadata.py](../op_dsl/flash_kda_grad_metadata.py)。
调用方先生成反向 metadata，再调用主入口或适配入口。

### PyTorch 接口原型

```python
cann_ops_transformer.flash_kda_grad(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    dht: Tensor,
    Aqk: Tensor,
    Akk: Tensor,
    metadata: Optional[Tensor] = None,
    *,
    initial_state: Optional[Tensor] = None,
    chunk_states: Optional[Tensor] = None,
    w: Optional[Tensor] = None,
    u: Optional[Tensor] = None,
    v_new: Optional[Tensor] = None,
    qg: Optional[Tensor] = None,
    kg: Optional[Tensor] = None,
    g_cumsum: Optional[Tensor] = None,
    q_rtsd: Optional[Tensor] = None,
    k_rstd: Optional[Tensor] = None,
    beta_sigmoid: Optional[Tensor] = None,
    A_log: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    seqused: Optional[Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
```

Torch 入口不做额外数值处理，直接以 PrivateUse1 调用 DSL 实现。`disable_recompute` 在 Torch 入口默认
`False`（重算、省显存），在 DSL 主入口默认 `True`（消费保存缓存），两者语义一致。

### DSL 主入口原型

```python
flash_kda_grad_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = True,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
)
```

`q_rtsd` 是当前接口的实际参数名，与 `k_rstd` 的拼写不同。主入口不接收独立的 `q_norm/k_norm` 参数；
启用归一化时，将前向保存的归一化结果按输入布局传入 `q/k`。

### Metadata 接口原型

```python
flash_kda_grad_metadata(
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    batch_size: int,
    n_heads_qk: int,
    n_heads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    layout: str = "TND",
) -> torch.Tensor
```

### 前向缓存适配入口原型

```python
flash_kda_grad_from_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    forward_outputs: tuple[torch.Tensor | None, ...],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: torch.Tensor,
    scale: Optional[float] = None,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    disable_recompute: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states_fp32: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
)
```

该入口接收原始 `q/k` 和训练前向的 16 项返回值。开启归一化时，适配入口从 `forward_outputs` 取出
`q_norm/k_norm`，转换为输入布局后交给主入口。`A_log`、`dt_bias` 在此入口中没有默认值，
不使用时需显式传 `None`。

## 参数说明

维度编号说明：本文中"第 N 维"从 1 开始计数；`shape[0]` 等代码表达式从 0 开始计数。

维度符号说明：

- `B`：逻辑序列数，`B >= 1`。
- `S`：BSND/BNSD 输入的物理序列长度。
- `T`：TND 输入的总 token 数，等于所有序列有效长度之和。
- `Nqk`：Q/K head 数。
- `Nv`：V head 数，要求 `Nv % Nqk == 0`。
- `R`：每个 Q/K head 对应的 value head 数，`R = Nv / Nqk`。
- `Dk`、`Dv`：Q/K 和 V 的 head dim，当前均为 128。
- `C`：chunk 大小，固定为 64。
- `Cmax`：padded 布局的 chunk 容量，`Cmax = ceil(S / C)`。
- `Nc`：TND 的 chunk 缓存容量，`Nc = ceil(T / C) + B`，包含预留槽位。

### 输入布局

以下形状均为接口实际接收的形状。`layout` 同时指定 `q/k/v/g/beta/do` 的布局。

| 参数 | TND | BNSD | BSND |
| :--- | :--- | :--- | :--- |
| `q`、`k` | `[T,Nqk,Dk]` | `[B,Nqk,S,Dk]` | `[B,S,Nqk,Dk]` |
| `v`、`do` | `[T,Nv,Dv]` | `[B,Nv,S,Dv]` | `[B,S,Nv,Dv]` |
| `g` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | `[B,S,Nv,Dk]` |
| `beta` | `[T,Nv]` | `[B,Nv,S]` | `[B,S,Nv]` |
| `initial_state`、`dht` | `[B,Nv,Dv,Dk]` | `[B,Nv,Dv,Dk]` | `[B,Nv,Dv,Dk]` |

### 主接口参数

表中"缓存布局"指下一节定义的前向缓存形状；FLOAT32、BFLOAT16 分别对应 `torch.float32`、`torch.bfloat16`。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `q`、`k` | 输入 | Query、Key。 | 必传；开启 L2Norm 时为前向归一化后的值，关闭时为原始值。 | BFLOAT16 | ND | 见输入布局 |
| `v` | 输入 | Value。 | 必传；与前向输入一致。 | BFLOAT16 | ND | 见输入布局 |
| `g` | 输入 | Gate 输入。 | 必传；融合时为原始 gate 输入，未融合时为单步 log gate。 | BFLOAT16 / FLOAT32 | ND | 见输入布局 |
| `beta` | 输入 | 每 token、每 value head 的更新系数输入。 | 必传；融合时为 sigmoid 前的输入，未融合时为已激活的系数。 | BFLOAT16 / FLOAT32 | ND | 见输入布局 |
| `do` | 输入 | 前向输出梯度。 | 必传；没有该路梯度时传零 Tensor，不能传 `None`。 | BFLOAT16 | ND | 与 `v` 相同 |
| `dht` | 输入 | 最终状态梯度。 | 必传；没有该路梯度时传零 Tensor，不能传 `None`。 | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `Aqk` | 输入 | 前向因果 QK 矩阵，已包含 `scale`。 | 两种重算模式均必传，不重算。 | BFLOAT16 | ND | 缓存布局 |
| `Akk` | 输入 | 前向 WY 变换的单位下三角矩阵逆。 | 两种重算模式均必传，不重算。 | BFLOAT16 | ND | 缓存布局 |
| `metadata` | 输入 | 反向任务调度信息。 | 必传；由 `flash_kda_grad_metadata` 生成，不能使用前向 metadata 替代。 | INT32 | ND | `[8+7*B*Nv]` |
| `initial_state` | 输入 | 前向初始状态。 | 重算时使用，`None` 表示零状态；保存缓存模式不使用。 | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `chunk_states` | 输入 | 各 chunk 入口状态。 | 保存缓存模式必传；重算模式忽略并重新生成。 | FLOAT32 | ND | 缓存布局 |
| `w` | 输入 | WY 变换后的 Key 缓存。 | 保存缓存模式必传；重算模式重新生成。 | BFLOAT16 | ND | 缓存布局 |
| `u` | 输入 | WY 变换后的 Value 缓存。 | 保存缓存模式仍须提供；当前反向数值计算不读取其保存值。 | BFLOAT16 | ND | 缓存布局 |
| `v_new` | 输入 | 结合 chunk 入口状态修正后的 Value。 | 保存缓存模式必传；重算模式重新生成。 | BFLOAT16 | ND | 缓存布局 |
| `qg`、`kg` | 输入 | 结合累计 gate 的 Q/K 缓存。 | 保存缓存模式必传；`qg` 不包含 `scale`。 | BFLOAT16 | ND | 缓存布局 |
| `g_cumsum` | 输入 | chunk 内累计 log gate。 | 保存缓存模式必传；不能用逐 token 的原始 `g` 替代。 | FLOAT32 | ND | 缓存布局 |
| `q_rtsd`、`k_rstd` | 输入 | 前向 Q/K L2Norm 的逆范数。 | 开启 Q/K 归一化时必传，不参与重算。 | FLOAT32 | ND | 缓存布局 |
| `beta_sigmoid` | 输入 | 前向激活后的 beta。 | 开启 beta sigmoid 融合时必传；启用负特征值时已包含系数 2。 | FLOAT32 | ND | 缓存布局 |
| `A_log` | 输入 | 每个 value head 的 gate 参数。 | `use_gate_in_kernel=True` 时必传。 | FLOAT32 | ND | `[Nv]` |
| `dt_bias` | 输入 | 每个 value head、每个 Key 通道的 gate 偏置。 | 可选；融合 gate 时 `None` 表示零偏置。 | FLOAT32 | ND | `[Nv,Dk]` |
| `cu_seqlens` | 输入 | TND 的 token 前缀和。 | 主反向可省略；显式传入时须与 metadata 一致。 | INT32 | ND | `[B+1]` |
| `seqused` | 输入 | BSND/BNSD 的有效序列长度。 | 主反向可省略；显式传入时须与 metadata 一致。 | INT32 | ND | `[B]` |
| `scale` | 输入属性 | 输出/Query 缩放系数。 | 默认 `None`，使用 `1/sqrt(Dk)`；显式值必须为有限数值，且须与前向一致。 | float / None | - | - |
| `use_qk_l2norm_in_kernel` | 输入属性 | 是否计算 Q/K 归一化反向。 | 默认 `True`。 | bool | - | - |
| `use_gate_in_kernel` | 输入属性 | 是否计算 gate 激活反向。 | 默认 `True`。 | bool | - | - |
| `use_beta_sigmoid_in_kernel` | 输入属性 | 是否计算 beta sigmoid 反向。 | 默认 `True`。 | bool | - | - |
| `allow_neg_eigval` | 输入属性 | 是否使用 `2*sigmoid(beta)`。 | 默认 `False`；为 `True` 时必须开启 beta sigmoid 融合。 | bool | - | - |
| `disable_recompute` | 输入属性 | 是否使用保存的七项缓存。 | DSL 入口默认 `True`，要求完整缓存；为 `False` 时启用重算。Torch 入口默认 `False`。 | bool | - | - |
| `lower_bound` | 输入属性 | 融合 gate 的下界及公式选择。 | 默认 `None`，使用 softplus；非空时须在 `[-5,0)`，使用有界 sigmoid。 | float / None | - | - |
| `layout` | 输入属性 | 输入和对应梯度的布局。 | 默认 `"TND"`；支持 `"TND"`、`"BNSD"`、`"BSND"`。 | str | - | - |
| `dq`、`dk`、`dv` | 输出 | Q/K/V 梯度。 | GQA 的 Q/K 梯度已沿对应 value heads 求和。 | BFLOAT16 | ND | 分别与 `q/k/v` 相同 |
| `dh0` | 输出 | 初始状态梯度。 | 始终返回 FP32 的 V-first 状态。 | FLOAT32 | ND | `[B,Nv,Dv,Dk]` |
| `dg`、`dbeta` | 输出 | Gate/beta 输入梯度。 | 分别保持 `g/beta` 的 dtype 和形状。 | BFLOAT16 / FLOAT32 | ND | 分别与 `g/beta` 相同 |
| `dA` | 输出 | `A_log` 梯度。 | 关闭 gate 融合时为 `None`。 | FLOAT32 | ND | `[Nv]` |
| `dBias` | 输出 | `dt_bias` 梯度。 | 关闭 gate 融合或未传 `dt_bias` 时为 `None`。 | FLOAT32 | ND | `[Nv,Dk]` |

### 前向缓存布局

BNSD 和 BSND 输入均使用 head 在前的前向缓存布局。缓存不能随 BSND 输入一起转换为 `[B,S,N,...]`；
由主入口内部创建计算所需的视图。

| 缓存 | TND | BNSD / BSND | 数据类型 |
| :--- | :--- | :--- | :--- |
| `q_norm`、`k_norm`（仅适配入口从 tuple 读取） | `[T,Nqk,Dk]` | `[B,Nqk,S,Dk]` | BFLOAT16 |
| `q_rtsd`、`k_rstd` | `[T,Nqk]` | `[B,Nqk,S]` | FLOAT32 |
| `beta_sigmoid` | `[T,Nv]` | `[B,Nv,S]` | FLOAT32 |
| `Aqk`、`Akk` | `[Nc,Nv,C,C]` | `[B,Nv,Cmax,C,C]` | BFLOAT16 |
| `w`、`qg`、`kg` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | BFLOAT16 |
| `u`、`v_new` | `[T,Nv,Dv]` | `[B,Nv,S,Dv]` | BFLOAT16 |
| `g_cumsum` | `[T,Nv,Dk]` | `[B,Nv,S,Dk]` | FLOAT32 |
| `chunk_states` | `[Nc,Nv,Dv,Dk]` | `[B,Nv,Cmax,Dv,Dk]` | FLOAT32 |

TND 中，第 `s` 条序列的第 `j` 个 chunk（均从 0 开始）使用槽位 `floor(cu_seqlens[s]/64) + s + j`。
`Nc` 不是各序列有效 chunk 数之和；预留槽位不参与计算。矩阵和状态缓存必须保留 chunk 轴，
入口不接受旧的紧凑缓存或展平后的矩阵缓存。

`forward_outputs` 必须保持训练前向的 16 项返回顺序：

```python
(
    out, final_state, q_norm, k_norm, q_rtsd, k_rstd, beta_sigmoid,
    Aqk, Akk, w, u, qg, kg, v_new, g_cumsum, saved_chunk_states,
)
```

适配入口不读取前两项的数值，`do/dht` 仍需显式提供。`chunk_states_fp32` 可在保存缓存模式覆盖
`saved_chunk_states`，其 dtype、shape 和存储顺序必须满足上表；重算模式不使用该覆盖值。

### Metadata 参数

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cu_seqlens` | 输入 | TND 序列边界。 | TND 必传；首项为 0、非递减、末项为 T。 | INT32 | ND | `[B+1]` |
| `seqused` | 输入 | Padded 输入的有效长度。 | BSND/BNSD 可选，每项在 `[0,S]`；省略时所有序列长度均为 S。 | INT32 | ND | `[B]` |
| `batch_size` | 输入属性 | 逻辑序列数。 | 必传正整数。 | int | - | - |
| `n_heads_qk` | 输入属性 | Q/K head 数。 | 必传正整数。 | int | - | - |
| `n_heads_v` | 输入属性 | V head 数。 | 必传；须为 `n_heads_qk` 的正整数倍。 | int | - | - |
| `seqlens` | 输入属性 | 物理存储长度。 | 必传非负整数；TND 为 T，BSND/BNSD 为 S。 | int | - | - |
| `headdim_qk`、`headdim_v` | 输入属性 | Q/K、V 的 head dim。 | 默认 128，当前仅支持 128。 | int | - | - |
| `layout` | 输入属性 | 输入布局。 | 默认 `"TND"`，可选值与主入口一致。 | str | - | - |
| 返回 Tensor | 输出 | 反向调度快照。 | 包含 8 个 header 字段和每个 `(sequence,value_head)` 的 7 个任务字段。 | INT32 | ND | `[8+7*B*Nv]` |

长度 Tensor 必须连续且位于 NPU。输出位于长度 Tensor 所在设备；BSND/BNSD 不提供长度 Tensor 时，
输出位于当前 NPU。

## 返回值

主入口、适配入口和 Torch 入口均返回 `(dq, dk, dv, dh0, dg, dbeta, dA, dBias)`，顺序固定：

- `dq/dk/dv` 为 BF16，分别保持 Q/K/V 的形状和逻辑布局；开启归一化时，`dq/dk` 已包含 L2Norm 反向。
- `dh0` 为 FP32 `[B,Nv,Dv,Dk]`；空序列对应的 `dh0` 等于 `dht`。
- `dg/dbeta` 分别保持原始 `g/beta` 的 dtype 和形状。关闭对应融合时，返回对已处理 gate/beta 的梯度。
- `dA` 仅在 gate 融合开启时返回；`dBias` 还要求传入 `dt_bias`，否则对应位置为 `None`。

BSND/BNSD 的无效 token 行在 `dq/dk/dv/dg/dbeta` 中写零。输出不保证连续存储，例如 `dbeta` 可能为转置视图，
`dA` 为内部存储的列视图；需要连续 Tensor 的下游应显式调用 `.contiguous()`。

## 约束说明

### 约束类型说明

| 约束类型 | 含义 |
| :--- | :--- |
| 单参数约束 | 单个 Tensor 的 shape、dtype、设备、连续性，以及属性的取值范围。 |
| 存在性约束 | 在指定布局、融合开关或重算模式下，参数必须提供或必须省略。 |
| 一致性约束 | 参数之间的形状、head 数、序列边界、布局及缓存语义一致。 |
| 特性交叉约束 | 融合、GQA、重算和前向缓存等多个参数组共同决定的使用要求。 |

### 特性参数组

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
| :--- | :--- | :--- | :--- |
| 公共参数组 | `q/k/v/do/dht` | INPUT | Tensor |
| 公共参数组 | `scale/layout` | ATTR | float / str |
| 公共参数组 | `dq/dk/dv/dh0` | OUTPUT | Tensor |
| 前向缓存与重算参数组 | `Aqk/Akk/chunk_states/w/u/v_new/qg/kg/g_cumsum/initial_state` | INPUT | Tensor / None |
| 前向缓存与重算参数组 | `disable_recompute` | ATTR | bool |
| Q/K 归一化参数组 | `q_rtsd/k_rstd` | INPUT | Tensor / None |
| Q/K 归一化参数组 | `use_qk_l2norm_in_kernel` | ATTR | bool |
| Gate 参数组 | `g/A_log/dt_bias` | INPUT | Tensor / None |
| Gate 参数组 | `use_gate_in_kernel/lower_bound` | ATTR | bool / float / None |
| Gate 参数组 | `dg/dA/dBias` | OUTPUT | Tensor / None |
| Beta 参数组 | `beta/beta_sigmoid` | INPUT | Tensor / None |
| Beta 参数组 | `use_beta_sigmoid_in_kernel/allow_neg_eigval` | ATTR | bool |
| Beta 参数组 | `dbeta` | OUTPUT | Tensor |
| 序列长度与 Metadata 参数组 | `cu_seqlens/seqused/metadata` | INPUT | Tensor / None |

### 基准信息说明

| 命名 | 含义 |
| :--- | :--- |
| TND | 将各序列有效 token 拼接，使用 `cu_seqlens` 描述序列边界。 |
| BNSD / BSND | 保留 batch 和物理序列长度 S，使用 `seqused` 描述有效长度，允许 padding。 |
| GQA | 每个 Q/K head 对应 R 个连续的 value heads；value head `h` 使用 Q/K head `h // R`。 |
| V-first | 状态最后两维依次为 Dv、Dk；状态缓存、`initial_state/dht/dh0` 均遵循此约定。 |
| 保存缓存模式 | `disable_recompute=True`，直接消费前向保存的七项缓存。 |
| 重算模式 | `disable_recompute=False`，在反向计算 kernel 内重建七项缓存。 |
| persistent / staged | 内部自动选择的两种反向执行策略，不是公开输入属性。 |

### 参数组约束

#### 公共参数组

- **单参数约束**：固定 `Dk=Dv=128`、`C=64`。Q/K/V、`do`、矩阵缓存及普通 token 缓存使用 BF16；
  状态和状态梯度使用 FP32。实际消费的 Tensor 必须位于同一 NPU。
- **存在性约束**：`do/dht` 均必传。即使只对输出或最终状态中的一路求导，也须为另一路提供形状、类型匹配的零 Tensor。
- **一致性约束**：Q/K 形状相同；Q/K/V 的 batch 和 token 维一致；`1 <= Nqk <= Nv` 且 `Nv % Nqk == 0`。
  `g` 与 `v` 形状相同，`beta` 的形状为 `v.shape[:-1]`。
- **特性交叉约束**：`scale`、布局、融合开关、`allow_neg_eigval` 和 `lower_bound` 必须与生成缓存的前向计算一致。
  GQA 下 Q/K 输入和梯度仍按 Nqk 个 head 存储，gate、beta、矩阵和状态缓存按 Nv 个 head 存储。

#### 前向缓存与重算参数组

| 参数或行为 | `disable_recompute=False` | `disable_recompute=True` |
| :--- | :--- | :--- |
| `w/u/v_new/qg/kg/g_cumsum/chunk_states` | 可省略或传 `None`；即使传入也会忽略并重新生成。 | 七项全部必传，满足前向缓存布局。 |
| `initial_state` | `None` 表示零状态；非零前向初始状态必须显式传入。 | 不使用。 |
| `Aqk/Akk` | 必传，不重算。 | 必传。 |
| `q_rtsd/k_rstd/beta_sigmoid` | 对应融合开启时必传，不重算。 | 对应融合开启时必传。 |

- **单参数约束**：`chunk_states` 为 FP32 的 chunk 入口状态。重算时的 `initial_state` 为
  FP32 `[B,Nv,Dv,Dk]`，最后一维 stride 必须为 1。
- **一致性约束**：缓存来自同一次前向，保留当前前向返回的形状和存储顺序。BSND 输入的缓存仍为 head 在前的布局；
  TND 必须使用包含预留槽位的 `Nc` 容量。
- **特性交叉约束**：当前训练前向将归一化和 beta 激活缓存的返回绑定到前向 `disable_recompute=True`。
  因此，反向开启重算不代表可以省略这些融合所需的缓存。保存状态缓存还需前向设置
  `return_intermediate_state=True`；反向重算所需的 `initial_state` 不在前向返回 tuple 中。

#### Q/K 归一化参数组

- **存在性约束**：`use_qk_l2norm_in_kernel=True` 时，主入口的 `q/k` 必须为前向保存的 BF16 归一化结果，
  并提供 FP32 `q_rtsd/k_rstd`。关闭时直接传原始 Q/K，不使用逆范数缓存。
- **一致性约束**：逆范数按 Nqk 个 head 保存。BSND 输入下，主入口 `q/k` 为 `[B,S,Nqk,Dk]`，
  逆范数仍为 `[B,Nqk,S]`。
- **特性交叉约束**：归一化开关控制归一化反向，不会在主入口重新归一化原始 Q/K。使用适配入口时可传原始 Q/K，
  由其从前向 tuple 取出归一化结果。

#### Gate 参数组

令 $x=g+dt_{bias}$（未提供偏置时取 0），$a=\exp(A_{log})$。融合 gate 的单步 log gate 为：

$$
\ell =
\begin{cases}
-a\cdot\operatorname{softplus}(x), & lower\_bound=\mathrm{None},\\
lower\_bound\cdot\sigma(a\cdot x), & -5\le lower\_bound<0.
\end{cases}
$$

- **单参数约束**：`g` 支持 BF16 或 FP32；`A_log` 为 FP32 `[Nv]`，`dt_bias` 为 FP32 `[Nv,Dk]`。
  `lower_bound` 只能为 `None` 或 `[-5,0)` 内的值，0 不被反向入口接受。
- **存在性约束**：融合开启时必须提供 `A_log`，`dt_bias` 可省略。关闭融合时 `g` 表示已处理的单步非正 log gate，
  不使用 gate 激活参数计算梯度。
- **一致性约束**：`g_cumsum` 是 chunk 内对单步 log gate 的累加。参数梯度沿全部有效 token 和序列归约。
- **特性交叉约束**：反向对 FP32 `g` 直接进行 FP32 Vector 运算；BF16 输入在核内提升为 FP32。
  `dg` 在最终写回时匹配 `g.dtype`。若前向生产者对 FP32 gate 做了预先量化，调用方须保证传入反向的
  gate 数值与缓存生成时一致。

#### Beta 参数组

- **单参数约束**：`beta` 独立支持 BF16 或 FP32，每个 token、每个 value head 一个值；`beta_sigmoid` 必须为 FP32。
- **存在性约束**：`use_beta_sigmoid_in_kernel=True` 时必须提供前向激活缓存；关闭时直接使用 `beta` 作为更新系数。
- **一致性约束**：`allow_neg_eigval=False` 时激活缓存为 `sigmoid(beta)`；为 `True` 时为 `2*sigmoid(beta)`，
  缓存中已含系数 2。
- **特性交叉约束**：`allow_neg_eigval=True` 要求启用 beta sigmoid 融合。`dbeta.dtype` 由原始 `beta.dtype` 决定，
  不由 FP32 激活缓存的 dtype 决定。

#### 序列长度与 Metadata 参数组

- **单参数约束**：`metadata` 为连续的一维 NPU INT32 Tensor，长度为 `8+7*B*Nv`。metadata 的整数维度属性
  须在 int32 范围内，容量也须能用 int32 表示。
- **存在性约束**：生成 TND metadata 时必须提供 `cu_seqlens`，不得提供 `seqused`；BSND/BNSD 仅可提供 `seqused`，
  省略时按满长 S 生成。两种长度 Tensor 互斥。主反向可只传已生成的 metadata。
- **一致性约束**：TND 前缀和从 0 开始、非递减并以 T 结束；padded 有效长度在 `[0,S]` 内。
  主反向显式传入长度时，会校验其与 metadata 的序列信息一致。
- **特性交叉约束**：允许空序列和不足 64 token 的尾块；空序列传递 `dht` 到 `dh0`，padding 梯度清零。
  前向和反向 metadata 的 ABI 不同，必须分别生成。改变维度、布局或有效长度后须重新调用反向 metadata 接口，
  不应手工修改任务表。

主反向首次解析 metadata 时会检查 magic、ABI 版本、状态码及输入形状，非法快照会抛出 `ValueError`，
读取设备数据可能引入同步。相同且未修改的 metadata 可复用；接口生成的 metadata 带有生产完成事件，
用于跨 stream 等待。

必需输入和当前路径使用的缓存必须是 Tensor；类型或维度不合法时抛出 `ValueError`。
显式 `scale` 的转换及有限值检查在 metadata 解析和重算缓存分配之前完成，非法值不会进入计算准备。

## 调用示例

以下示例使用 BNSD、GQA 和有界 gate，先运行训练前向生成缓存，再调用反向。
需提前加载 CANN 环境（包括 `ASCEND_HOME_PATH`），并安装匹配的 `cannbotdsl`、PyTorch 和 `torch_npu`；
首次调用可能触发 DSL/AICPU 编译。

```python
import torch
import torch_npu

import cann_ops_transformer as ops

torch.npu.set_device(0)
device = torch.device("npu:0")
B, S, Nqk, Nv, D = 1, 64, 1, 2, 128
layout = "BNSD"
scale = D ** -0.5
lower_bound = -0.06

q = torch.randn(B, Nqk, S, D, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, Nv, S, D, device=device, dtype=torch.bfloat16)
g = torch.randn_like(v)
beta = torch.randn(B, Nv, S, device=device, dtype=torch.bfloat16)
initial_state = torch.zeros(B, Nv, D, D, device=device, dtype=torch.float32)
A_log = torch.zeros(Nv, device=device, dtype=torch.float32)
dt_bias = torch.zeros(Nv, D, device=device, dtype=torch.float32)

# 前向和反向分别生成各自的 metadata。
fwd_outputs = forward_op(          # 训练前向，返回 16 项
    q, k, v, g, beta, scale=scale, initial_state=initial_state,
    A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
    layout=layout, return_intermediate_state=True,
)
out, final_state = fwd_outputs[:2]
do = torch.randn_like(out)
dht = torch.zeros_like(final_state)  # 本例只对 out 提供非零上游梯度。

bwd_metadata = ops.flash_kda_grad_metadata(
    batch_size=B, n_heads_qk=Nqk, n_heads_v=Nv, seqlens=S, layout=layout,
)

# 保存缓存模式：直接消费前向返回的七项缓存。
dq, dk, dv, dh0, dg, dbeta, dA, dBias = ops.flash_kda_grad(
    q, k, v, g, beta, do, dht,
    fwd_outputs[7], fwd_outputs[8],           # Aqk, Akk
    bwd_metadata,
    initial_state=initial_state,
    chunk_states=fwd_outputs[15],
    w=fwd_outputs[9], u=fwd_outputs[10], v_new=fwd_outputs[13],
    qg=fwd_outputs[11], kg=fwd_outputs[12], g_cumsum=fwd_outputs[14],
    q_rtsd=fwd_outputs[4], k_rstd=fwd_outputs[5],
    beta_sigmoid=fwd_outputs[6],
    A_log=A_log, dt_bias=dt_bias,
    scale=scale,
    disable_recompute=True,
    allow_neg_eigval=False,
    lower_bound=lower_bound,
    layout=layout,
)
```

需要使用重算模式时，把 `disable_recompute` 改为 `False`，传入 `initial_state` 并省略七项可重算缓存：

```python
dq, dk, dv, dh0, dg, dbeta, dA, dBias = ops.flash_kda_grad(
    q, k, v, g, beta, do, dht,
    fwd_outputs[7], fwd_outputs[8], bwd_metadata,
    initial_state=initial_state,
    q_rtsd=fwd_outputs[4], k_rstd=fwd_outputs[5], beta_sigmoid=fwd_outputs[6],
    A_log=A_log, dt_bias=dt_bias,
    scale=scale,
    disable_recompute=False,
    lower_bound=lower_bound,
    layout=layout,
)
```

前向返回的归一化和 beta 激活缓存仍需保留，七项可重算缓存会被忽略。

直接调用 DSL 主入口 `flash_kda_grad_npu` 时，按接口原型拆分并传入前向缓存；本例为 BNSD，
`q/k` 应替换为 `fwd_outputs[2:4]` 中的归一化结果。BSND 场景还需将这两项由 `[B,Nqk,S,Dk]`
转为 `[B,S,Nqk,Dk]`；其余缓存保持前向返回布局。

单元测试与参考公式位于 [tests/ut](../tests/ut)，TTK 精度与性能验收见
[tests/ttk/README.md](../tests/ttk/README.md)。
