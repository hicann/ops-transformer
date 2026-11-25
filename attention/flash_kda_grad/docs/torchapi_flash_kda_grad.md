# flash_kda_grad

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT系列产品</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2系列产品</term>：不支持

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

- 接口功能：计算Kimi Delta Attention（KDA）的训练反向，返回Q、K、V、初始状态、gate、beta及gate参数的梯度，支持变长序列、分组查询注意力（Grouped Query Attention，GQA）和融合激活的反向。
  - `flash_kda_grad_metadata`：按输入规格和有效序列长度生成反向调度信息。
  - `flash_kda_grad`：消费前向缓存和上游梯度，固定返回八个梯度位置。直接调用前先生成反向metadata；使用[FlashKDA自动求导](../../flash_kda/docs/torchapi_flash_kda.md)时，由前向接口管理缓存和反向调用。
- 计算公式：对一条序列的一个value head，令$\bar q_t$、$\bar k_t$为前向实际使用的Q/K；开启L2归一化时为归一化结果。$\sigma$为sigmoid函数，未提供`dt_bias`时偏置取0。单步log gate和状态递推如下：

$$
\ell_t=\begin{cases}
g_t, & \text{未融合gate} \\
-\exp(A_{log})\operatorname{softplus}(g_t+dt_{bias}), & lower\_bound=\mathrm{None} \\
lower\_bound\cdot\sigma\!\left(\exp(A_{log})(g_t+dt_{bias})\right), & -5\le lower\_bound<0
\end{cases}
$$

$$
\widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t)),\quad
\Delta v_t=b_t(v_t-\widetilde H_t\bar k_t),\quad
H_t=\widetilde H_t+\Delta v_t\bar k_t^T,\quad
o_t=scale\cdot H_t\bar q_t
$$

$$
L=\sum_t\langle do_t,o_t\rangle+\langle dht,H_{last}\rangle
$$

其中，未融合beta sigmoid时$b_t=beta_t$；融合时$b_t=\sigma(beta_t)$，同时开启`allow_neg_eigval`时$b_t=2\sigma(beta_t)$。状态$H_t$按`(Dv,Dk)`存储。

反向对上述损失计算`(dq,dk,dv,dh0,dg,dbeta,dA,dBias)`，其中`dA`是对`A_log`的梯度。开启Q/K归一化反向时，`dq/dk`对应归一化前的输入。`dht=None`表示损失不包含最终状态项。

实现固定按64个token分块，head维度为128。默认在反向kernel内重算七项缓存；设置`disable_recompute=True`后使用保存缓存。保存的`chunk_states`为bfloat16，重算状态缓冲和状态梯度为float32；主机自动选择persistent或staged流水，主反向只下发一个计算kernel。

## 函数原型

通过`import cann_ops_transformer`导入模块。先调用`flash_kda_grad_metadata`；从`batch_size`开始的参数必须通过关键字提供，四个必选维度没有默认值。

```python
cann_ops_transformer.flash_kda_grad_metadata(
    cu_seqlens=None,
    seqused=None,
    batch_size=batch_size,
    nheads_qk=nheads_qk,
    nheads_v=nheads_v,
    seqlens=seqlens,
    headdim_qk=128,
    headdim_v=128,
    layout='TND',
)
```

`flash_kda_grad`从`Aqk`开始的参数必须通过关键字传入。`Aqk`和`Akk`为必选Tensor，无默认值。`dht`为可选参数，默认`None`，表示零最终状态梯度。`metadata`默认`None`，但执行时必须提供`flash_kda_grad_metadata`生成的调度信息。

```python
cann_ops_transformer.flash_kda_grad(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht=None,
    Aqk=Aqk,
    Akk=Akk,
    metadata=None,
    initial_state=None,
    chunk_states=None,
    w=None,
    u=None,
    v_new=None,
    qg=None,
    kg=None,
    g_cumsum=None,
    q_rstd=None,
    k_rstd=None,
    beta_sigmoid=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    lower_bound=None,
    layout='TND',
    dbeta_dtype=None,
)
```

## 参数说明

`B`为逻辑序列数，`S`为BNSD/BSND的物理序列长度，`T`为TND的总token数。`Nqk`、`Nv`分别为Q/K和V的head数，满足`Nv % Nqk == 0`；每个Q/K head对应连续的`Nv/Nqk`个value heads。当前`Dk=Dv=128`，chunk大小固定为64，张量格式为ND。

### flash_kda_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| cu_seqlens | Tensor | 可选 | 序列token前缀和，默认`None`。TND布局必传，首项为0、非递减、末项等于T；其他布局传`None`。 | int32 | ND | `(B+1,)` |
| seqused | Tensor | 可选 | 各序列有效长度，默认`None`。BNSD/BSND布局每项在`[0,S]`；省略时各序列均按S处理。TND布局传`None`。 | int32 | ND | `(B,)` |
| batch_size | int | 必选 | 逻辑序列数B，无默认值，取int32范围内的正整数。 | - | - | - |
| nheads_qk | int | 必选 | Q/K的head数Nqk，无默认值，须为正整数。 | - | - | - |
| nheads_v | int | 必选 | V的head数Nv，无默认值，须为Nqk的正整数倍。 | - | - | - |
| seqlens | int | 必选 | 物理存储长度，无默认值，取int32范围内的非负整数。TND传T，BNSD/BSND传S。 | - | - | - |
| headdim_qk | int | 可选 | Q/K的head维度，默认128，当前仅支持128。 | - | - | - |
| headdim_v | int | 可选 | V的head维度，默认128，当前仅支持128。 | - | - | - |
| layout | str | 可选 | 输入布局，默认`"TND"`，支持`"TND"`、`"BNSD"`和`"BSND"`。 | - | - | - |

### flash_kda_grad

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| q | Tensor | 必选 | 查询张量。开启Q/K归一化反向时，传前向保存的`q_norm`；关闭时传原始Q。 | bfloat16 | ND | 见[输入布局](#输入布局)。 |
| k | Tensor | 必选 | 键张量，形状与q相同。开启Q/K归一化反向时传前向保存的`k_norm`。 | bfloat16 | ND | 与`q`相同。 |
| v | Tensor | 必选 | 前向使用的原始Value张量。 | bfloat16 | ND | 见[输入布局](#输入布局)。 |
| g | Tensor | 必选 | Gate输入。融合gate时为激活前输入，且必须为bfloat16；未融合时为已处理的单步log gate，且必须为float32。 | bfloat16、float32 | ND | 见[输入布局](#输入布局)。 |
| beta | Tensor | 必选 | 更新系数输入。融合sigmoid时为激活前输入，且必须为bfloat16；未融合时为已激活的系数，且必须为float32。 | bfloat16、float32 | ND | 见[输入布局](#输入布局)。 |
| do | Tensor | 必选 | 前向输出的上游梯度。必须提供Tensor；仅对最终状态求导时，传形状匹配的零Tensor。 | bfloat16 | ND | 见[输入布局](#输入布局)。 |
| dht | Tensor | 可选 | 最终状态的上游梯度，默认`None`，按零梯度处理。 | float32 | ND | `(B,Nv,128,128)` |
| Aqk | Tensor | 必选 | 前向因果QK矩阵缓存，包含scale。必传，无默认值，须通过关键字传入；两种重算模式均不重算该项。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| Akk | Tensor | 必选 | 前向WY变换的块内逆矩阵缓存。必传，无默认值，须通过关键字传入。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| metadata | Tensor | 可选 | 反向调度信息，默认`None`，但执行时必须提供`flash_kda_grad_metadata`的返回值。 | int32 | ND | `(8+7*B*Nv,)` |
| initial_state | Tensor | 可选 | 前向初始状态，默认`None`。重算时`None`表示零初态，非零初态必须显式传入；该参数是否为`None`还决定是否返回dh0。 | float32 | ND | `(B,Nv,128,128)` |
| chunk_states | Tensor | 可选 | 各chunk入口状态，默认`None`。保存缓存模式必传。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| w | Tensor | 可选 | WY变换后的Key缓存，默认`None`。保存缓存模式必传。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| u | Tensor | 可选 | WY变换后的Value缓存，默认`None`。保存缓存模式须提供，保持前向缓存协议完整。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| v_new | Tensor | 可选 | 结合chunk入口状态修正后的Value缓存，默认`None`。保存缓存模式必传。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| qg | Tensor | 可选 | 结合累计gate的Q缓存，不包含scale，默认`None`。保存缓存模式必传。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| kg | Tensor | 可选 | 结合累计gate的K缓存，默认`None`。保存缓存模式必传。 | bfloat16 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| g_cumsum | Tensor | 可选 | chunk内累计log gate，默认`None`。保存缓存模式必传，不能用原始g替代。 | float32 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| q_rstd | Tensor | 可选 | 前向Q归一化的逆范数，默认`None`。开启Q/K归一化反向时必传。 | float32 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| k_rstd | Tensor | 可选 | 前向K归一化的逆范数，默认`None`。开启Q/K归一化反向时必传。 | float32 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| beta_sigmoid | Tensor | 可选 | 前向激活后的beta，默认`None`。融合beta sigmoid时必传；负特征值模式下已包含系数2。 | float32 | ND | 见[前向缓存布局](#前向缓存布局)。 |
| A_log | Tensor | 可选 | 每个value head的gate参数，默认`None`。融合gate时必传。 | float32 | ND | `(Nv,)` |
| dt_bias | Tensor | 可选 | Gate偏置，默认`None`，按零偏置处理。 | float32 | ND | `(Nv,128) / (Nv*128,)` |
| cu_seqlens | Tensor | 可选 | TND序列边界，默认`None`。主反向可省略；显式提供时须与metadata描述的边界一致。 | int32 | ND | `(B+1,)` |
| cu_seqlens_cpu | List[int] | 可选 | 预留参数，默认`None`，当前仅允许`None`。 | - | - | - |
| seqused | Tensor | 可选 | BNSD/BSND有效长度，默认`None`。主反向可省略；显式提供时须与metadata一致。 | int32 | ND | `(B,)` |
| scale | float | 可选 | Q/输出缩放系数，默认`None`，使用`1/sqrt(128)`；显式值须为有限数值，并与前向一致。 | - | - | - |
| use_qk_l2norm_in_kernel | bool | 可选 | 是否计算Q/K的L2归一化反向，默认`True`；支持`True`、`False`。 | - | - | - |
| use_gate_in_kernel | bool | 可选 | 是否计算gate激活反向，默认`True`；支持`True`、`False`。 | - | - | - |
| use_beta_sigmoid_in_kernel | bool | 可选 | 是否计算beta sigmoid反向，默认`True`；支持`True`、`False`。 | - | - | - |
| allow_neg_eigval | bool | 可选 | 是否使用`2*sigmoid(beta)`，默认`False`；支持`True`、`False`，取`True`时要求融合beta sigmoid。 | - | - | - |
| disable_recompute | bool | 可选 | 是否关闭缓存重算，默认`False`。`False`时重建七项缓存，`True`时消费已保存缓存。 | - | - | - |
| lower_bound | float | 可选 | Gate公式选择及下界，默认`None`。非空值须在`[-5,0)`内；融合gate时，`None`选择softplus，非空值选择有界sigmoid。 | - | - | - |
| layout | str | 可选 | Q/K/V的逻辑布局，默认`"TND"`；支持`"TND"`、`"BNSD"`和`"BSND"`。g、beta和do的布局见[输入布局](#输入布局)。 | - | - | - |
| dbeta_dtype | torch.dtype | 可选 | dbeta的输出类型，默认`None`，使用`beta.dtype`。显式值仅支持`beta.dtype`或`torch.bfloat16`。 | - | - | - |

### 输入布局

`layout`描述Q/K/V的布局；定长场景的g、beta和do始终将序列维放在head维之前。

| layout | q、k | v | g | beta | do |
| --- | --- | --- | --- | --- | --- |
| BNSD | `(B,Nqk,S,128)` | `(B,Nv,S,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` | `(B,S,Nv,128)` |
| BSND | `(B,S,Nqk,128)` | `(B,S,Nv,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` | `(B,S,Nv,128)` |
| TND | `(T,Nqk,128)` | `(T,Nv,128)` | `(T,Nv,128)` | `(T,Nv)` | `(T,Nv,128)` |

### 前向缓存布局

两种定长布局共用相同的缓存布局。Q/K归一化后的`q_norm/k_norm`通过主接口的q/k传入，形状与对应的Q/K输入相同；其逆范数按head在前的顺序存储。

| 缓存 | TND | BNSD / BSND | 数据类型 |
| --- | --- | --- | --- |
| q_rstd, k_rstd | `(Nqk,T)` | `(B,Nqk,S)` | float32 |
| beta_sigmoid | `(Nv,T)` | `(B,Nv,S)` | float32 |
| Aqk, Akk | `(Nv,T,64)` | `(B,Nv,S,64)` | bfloat16 |
| w, qg, kg | `(Nv,T,128)` | `(B,Nv,S,128)` | bfloat16 |
| u, v_new | `(Nv,T,128)` | `(B,Nv,S,128)` | bfloat16 |
| g_cumsum | `(Nv,T,128)` | `(B,Nv,S,128)` | float32 |
| chunk_states | `(Nc,Nv,128,128)` | `(B,Cmax,Nv,128,128)` | bfloat16 |

`Cmax=ceil(S/64)`，`Nc=ceil(T/64)+B`。TND的状态缓存包含预留槽位：序列b的第j个chunk使用`floor(cu_seqlens[b]/64)+b+j`，仅消费该序列的`ceil(length_b/64)`个有效位置。Aqk/Akk按token行寻址，最后64列表示块内列，尾块在工作区补零。

### 缓存与重算模式

| 参数 | `disable_recompute=False`（默认） | `disable_recompute=True` |
| --- | --- | --- |
| chunk_states、w、u、v_new、qg、kg、g_cumsum | 可省略，反向内部重建。显式传入时仍须满足公开接口的形状检查。 | 七项全部提供，使用同一次前向保存的缓存。 |
| initial_state | 重建非零初始状态时必传；`None`表示零初态。 | 缓存求值不读取该项，但需要dh0时仍须提供对应的初始状态。 |
| Aqk、Akk | 必传，不重算。 | 必传。 |
| q_rstd、k_rstd | 开启Q/K归一化反向时必传。 | 开启Q/K归一化反向时必传。 |
| beta_sigmoid | 融合beta sigmoid时必传。 | 融合beta sigmoid时必传。 |

## 返回值说明

### flash_kda_grad_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| metadata | Tensor | 必选 | 连续的一维反向调度信息。将返回值原样传给flash_kda_grad。 | int32 | ND | `(8+7*B*Nv,)` |

### flash_kda_grad

固定返回`(dq,dk,dv,dh0,dg,dbeta,dA,dBias)`，可选输出为`None`时也保留其位置，元组长度始终为8。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| dq | Tensor | 必选 | Q梯度。开启归一化时已包含L2归一化反向；GQA下已沿对应value heads求和。 | bfloat16 | ND | 与`q`相同。 |
| dk | Tensor | 必选 | K梯度，归一化和GQA语义与dq一致。 | bfloat16 | ND | 与`k`相同。 |
| dv | Tensor | 必选 | V梯度，形状和逻辑布局与v相同。 | bfloat16 | ND | 与`v`相同。 |
| dh0 | Tensor | 可选 | 初始状态梯度。`initial_state=None`时返回`None`，否则返回float32 Tensor。 | float32 | ND | `(B,Nv,128,128)` |
| dg | Tensor | 必选 | Gate输入梯度，保持g的dtype和形状；未融合时为对单步log gate的梯度。 | g.dtype | ND | 与`g`相同。 |
| dbeta | Tensor | 必选 | Beta输入梯度，形状与beta相同，dtype由dbeta_dtype决定；未融合时为对激活后系数的梯度。 | beta.dtype / bfloat16 | ND | 与`beta`相同。 |
| dA | Tensor | 可选 | 对A_log的梯度。关闭gate融合时返回`None`。 | float32 | ND | `(Nv,)` |
| dBias | Tensor | 可选 | 对dt_bias的梯度，形状与dt_bias相同。关闭gate融合或`dt_bias=None`时返回`None`。 | float32 | ND | 与`dt_bias`相同。 |

## 约束说明

### 公共约束

- 该接口用于训练反向的单算子模式调用，实际计算仅在NPU上执行。Meta/FakeTensor用于输出规格推导，不能代替设备计算验证。
- `B>=1`、`S>=0`、`T>=0`，`1<=Nqk<=Nv`且`Nv % Nqk == 0`，head维度固定为128。metadata的整数维度及`8+7*B*Nv`容量须能用int32表示。与FlashKDA联合使用时，还需满足前向接口的规格。
- 实际使用的输入、缓存和metadata位于同一NPU；调用前设置当前设备。长度和metadata张量必须连续。Q/K/V及被读取缓存的最后一维须连续；保留前向返回的缓存视图，不能自行按输入布局转置所有缓存。do/dht的不满足传输要求的视图由主机按需整理。
- `do`必须为bfloat16 Tensor；`dht`可以为`None`。提供`initial_state`时，公开Torch接口返回float32的dh0 Tensor；未提供时返回`None`。
- 输出不保证连续存储，例如dA可为内部存储的列视图；下游要求连续Tensor时调用`.contiguous()`。BNSD/BSND的padding梯度行写零；空序列在返回dh0时传递dht，缺省dht则传递零梯度。

### 缓存与融合约束

- 所有缓存来自同一次前向，布局、scale、初始状态和融合属性与前向一致。主接口不会重新归一化原始Q/K；开启归一化反向时，必须传入前向保存的q_norm/k_norm及float32逆范数。
- 融合gate时g必须为bfloat16并提供float32的A_log；未融合时g必须为float32的单步log gate。融合beta sigmoid时beta必须为bfloat16并提供float32的beta_sigmoid；未融合时beta必须为float32的已激活系数。
- `allow_neg_eigval=True`要求开启beta sigmoid融合，beta_sigmoid中已包含系数2。关闭gate融合时不返回dA/dBias；没有dt_bias时不返回dBias。
- 重算可以省略的缓存仅限[缓存与重算模式](#缓存与重算模式)列出的七项，Aqk/Akk、所需的逆范数及激活缓存仍须保留。重算与保存缓存可能受不同的中间舍入影响，不要求两种模式逐位相等。
- `dbeta_dtype`不会改变beta的输入要求；beta为float32时可显式请求bfloat16的dbeta，beta为bfloat16时不能据此请求float32输出。

### 序列长度与metadata约束

- 反向metadata必须由`flash_kda_grad_metadata`生成。TND只接受cu_seqlens，BNSD/BSND只接受seqused；两者互斥。生成TND调度时cu_seqlens必传，允许相邻边界相等；定长布局省略seqused表示各序列均为S。
- 主反向可只传metadata，不重复提供长度Tensor。显式传入长度时须与metadata的序列边界、有效长度一致；改变维度、设备、布局或有效长度后重新生成metadata，不能手工改写任务信息。
- 前向metadata与反向metadata的协议不同。直接反向调用使用独立的反向metadata；自动求导使用`flash_kda_metadata(enable_grad=True)`生成的组合调度，由前向接口提取反向部分。
- 该接口不提供二阶梯度。需要自动求导时，通过FlashKDA的训练入口调用配套反向。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。同一设备、输入、属性、metadata及分核配置下，gate参数梯度先在各向量核独立的缓冲区中累加，再按固定顺序归约。经FlashKDA自动求导时，完整链路还受前向接口的确定性约束。
- Batch一致性说明：默认不支持Batch一致性。改变批量或序列分组可能改变分核和归约路径，不保证跨Batch组合或不同调度配置逐位一致。

## 调用示例

示例需要匹配的CANN环境、CANNBotDSL、PyTorch、torch_npu及已安装的cann_ops_transformer，并加载FlashKDA和FlashKDAGrad扩展。以下均为单算子模式；前向协议由`torch.is_inference_mode_enabled()`决定，训练缓存生成使用普通模式或`torch.no_grad()`。

### 通过FlashKDA自动求导

前向接口负责保存缓存并调用反向。此例默认重算；需要保存缓存时，将前向的`disable_recompute`改为`True`。

```python
import torch
import torch_npu
import cann_ops_transformer as ops

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 1, 2, 128

# 训练使用BSND输入，融合g和beta使用bfloat16。
q = torch.randn(batch, length, n_qk, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(batch, length, n_v, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
g = torch.randn_like(v, requires_grad=True)
beta = torch.randn(batch, length, n_v, device="npu",
                   dtype=torch.bfloat16, requires_grad=True)
initial_state = torch.zeros(batch, n_v, dim, dim, device="npu",
                            dtype=torch.float32, requires_grad=True)
A_log = torch.full((n_v,), -3.5, device="npu",
                   dtype=torch.float32, requires_grad=True)
dt_bias = torch.zeros(n_v, dim, device="npu",
                      dtype=torch.float32, requires_grad=True)

# 自动求导需要包含前向、反向调度的metadata。
metadata = ops.flash_kda_metadata(
    batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
    seqlens=length, enable_grad=True, layout="BSND",
)
out, final_state, chunk_states = ops.flash_kda(
    q, k, v, g, beta, metadata, initial_state=initial_state,
    A_log=A_log, dt_bias=dt_bias, output_final_state=True,
    disable_recompute=False, layout="BSND",
)

# FlashKDA内部保存缓存并调用FlashKDAGrad。
loss = out.float().square().mean() + final_state.square().mean()
loss.backward()
torch.npu.synchronize()
assert q.grad.shape == q.shape and v.grad.shape == v.shape
assert initial_state.grad.shape == initial_state.shape
assert A_log.grad.shape == A_log.shape
assert chunk_states is None
```

### 直接调用保存缓存和重算反向

此例通过已注册的`flash_kda_fwd`接口取得16项训练返回值。公开`flash_kda`仅返回前三项，普通训练优先使用上面的自动求导方式。

```python
import torch
import torch_npu
import cann_ops_transformer as ops

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 1, 2, 128
scale = dim ** -0.5
q = torch.randn(batch, n_qk, length, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, n_v, length, dim, device="npu", dtype=torch.bfloat16)

# Q/K/V为BNSD时，g和beta仍按BSND/BSN存储。
g = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.bfloat16)
beta = torch.randn(batch, length, n_v, device="npu", dtype=torch.bfloat16)
initial_state = torch.zeros(batch, n_v, dim, dim, device="npu", dtype=torch.float32)
A_log = torch.full((n_v,), -3.5, device="npu", dtype=torch.float32)
dt_bias = torch.zeros(n_v, dim, device="npu", dtype=torch.float32)

# 通过已注册的16项前向接口获取实际训练缓存。
# no_grad()保留训练缓存协议，此处不能进入inference_mode()。
with torch.no_grad():
    forward_metadata = ops.flash_kda_metadata(
        batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=length, enable_grad=False, layout="BNSD",
    )
    forward_outputs = torch.ops.cann_ops_transformer.flash_kda_fwd(
        q, k, v, g, beta, forward_metadata,
        initial_state=initial_state, A_log=A_log, dt_bias=dt_bias,
        scale=scale, output_final_state=True,
        disable_recompute=True, layout="BNSD",
    )

(out, final_state, chunk_states, Aqk, Akk, w, u, qg, kg, v_new,
 g_cumsum, q_norm, q_rstd, k_norm, k_rstd, beta_activated) = forward_outputs
do = torch.randn_like(out)

# 为直接反向调用生成独立的反向metadata。
backward_metadata = ops.flash_kda_grad_metadata(
    batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
    seqlens=length, layout="BNSD",
)
common = dict(
    Aqk=Aqk, Akk=Akk, metadata=backward_metadata,
    initial_state=initial_state, q_rstd=q_rstd, k_rstd=k_rstd,
    beta_sigmoid=beta_activated, A_log=A_log, dt_bias=dt_bias,
    scale=scale, layout="BNSD",
)

# 保存缓存模式传入七项缓存；省略dht表示零最终状态梯度。
saved = ops.flash_kda_grad(
    q_norm, k_norm, v, g, beta, do, **common,
    chunk_states=chunk_states, w=w, u=u, v_new=v_new,
    qg=qg, kg=kg, g_cumsum=g_cumsum, disable_recompute=True,
)

# 重算模式仍保留Aqk/Akk及归一化、激活缓存。
rebuilt = ops.flash_kda_grad(
    q_norm, k_norm, v, g, beta, do, **common,
    disable_recompute=False,
)
torch.npu.synchronize()
for gradients in (saved, rebuilt):
    dq, dk, dv, dh0, dg, dbeta, dA, dBias = gradients
    assert dq.shape == q.shape and dv.shape == v.shape
    assert dg.shape == g.shape and dbeta.shape == beta.shape
    assert dh0.shape == initial_state.shape
    assert dA.shape == A_log.shape and dBias.shape == dt_bias.shape
```

接口实现见[torch_extension/flash_kda_grad.py](../torch_extension/flash_kda_grad.py)，缓存适配与设备启动见[op_dsl/flash_kda_grad.py](../op_dsl/flash_kda_grad.py)。前向返回值协议见[FlashKDA接口文档](../../flash_kda/docs/torchapi_flash_kda.md)。
