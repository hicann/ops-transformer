# flash_kda

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

- 接口功能：实现Kimi Delta Attention（KDA）的前向计算，通过逐通道门控衰减和增量更新维护注意力状态，支持训练和推理场景、变长序列、分组查询注意力（Grouped Query Attention，GQA）及融合激活。
  - 调用前先通过`cann_ops_transformer.flash_kda_metadata()`按存储规格、有效序列长度和运行模式生成调度信息，再将结果传入`cann_ops_transformer.flash_kda()`。
  - `flash_kda`：返回前向输出、可选的最终状态和分块入口状态；训练时支持通过PyTorch自动求导调用配套的[FlashKDAGrad](../../flash_kda_grad/docs/torchapi_flash_kda_grad.md)。
- 计算公式：以一条序列的一个value head为例，$q_t$、$k_t$、$v_t$分别为查询、键和值。启用`use_qk_l2norm_in_kernel`时，$\bar q_t$、$\bar k_t$为沿最后一维进行L2归一化后的Q/K；否则使用原始Q/K。令$x_t=g_t+dt\_bias$，未提供偏置时取零，单步log gate为：

  $$
  \ell_t=\begin{cases}
  g_t, & \text{未融合gate} \\
  -\exp(A_{log})\operatorname{softplus}(x_t), & \text{融合gate且lower\_bound为None} \\
  lower\_bound\cdot\sigma(\exp(A_{log})x_t), & \text{融合gate且指定lower\_bound}
  \end{cases}
  $$

  其中，$\sigma$为sigmoid函数。更新系数$b_t$在未融合beta sigmoid时为`beta`，融合时为$\sigma(beta_t)$，同时启用`allow_neg_eigval`时为$2\sigma(beta_t)$。状态$H_t$按$[D_v,D_k]$存储：

  $$
  \widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t)),\qquad
  \Delta v_t=b_t(v_t-\widetilde H_t\bar k_t)
  $$

  $$
  H_t=\widetilde H_t+\Delta v_t\bar k_t^T,\qquad
  o_t=scale\cdot H_t\bar q_t
  $$

  `initial_state`提供$H_0$，未提供时取零；GQA将连续的`Nv/Nqk`个value head映射到一个Q/K head。实现按64个token分块计算，状态递推使用float32，矩阵计算使用bfloat16。

## 函数原型

以下原型和示例通过`import cann_ops_transformer`导入模块。调用`cann_ops_transformer.flash_kda`前先生成调度信息，并将其原样作为`metadata`传入；`A_log`保留软件定义中的大小写。

先调用`flash_kda_metadata`。从`batch_size`开始的参数必须通过关键字传入；`batch_size`、`nheads_qk`、`nheads_v`和`seqlens`没有默认值，必须显式提供。以下同名变量表示调用方提供的值。

```python
cann_ops_transformer.flash_kda_metadata(
    cu_seqlens=None,
    seqused=None,
    batch_size=batch_size,
    nheads_qk=nheads_qk,
    nheads_v=nheads_v,
    seqlens=seqlens,
    headdim_qk=128,
    headdim_v=128,
    enable_grad=False,
    layout="TND"
)
```

`q`、`k`、`v`、`g`、`beta`和`metadata`为必传参数；`scale`及其后的参数必须通过关键字传入。

```python
cann_ops_transformer.flash_kda(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    seqused=None,
    scale=0.08838834764831845,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    lower_bound=None,
    layout="TND"
)
```

## 参数说明

`B`为序列数，`S`为定长存储长度，`T`为紧凑存储总长度，`Nqk`和`Nv`分别为Q/K与V的head数，`Dk=Dv=128`。张量物理格式为ND。

### flash_kda_metadata

生成调度信息时，以下参数须与后续`flash_kda`调用一致。`batch_size`、`nheads_qk`、`nheads_v`和`seqlens`均须显式传入，不会从长度张量推导。整数参数须为int32范围内的正整数。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| cu_seqlens | Tensor | 可选 | 序列长度累积和，默认`None`。TND布局必传，首项为0、严格递增且末项等于`seqlens`；定长布局传入时，相邻差值须大于0且不超过`seqlens`。 | int32 | ND | `(B+1,)` |
| seqused | Tensor | 可选 | 每条序列的有效长度，默认`None`。未传入时使用`cu_seqlens`的相邻差值；两者均未传入时，定长布局各序列使用`seqlens`。与`cu_seqlens`同时传入时，须与其相邻差值逐项相等。 | int32 | ND | `(B,)` |
| batch_size | int | 必选 | 序列数`B`，无默认值。定长布局为Q/K/V的batch维大小；TND布局为`cu_seqlens.numel()-1`。 | - | - | - |
| nheads_qk | int | 必选 | Q/K的head数`Nqk`，无默认值，须满足`1 <= Nqk <= Nv`。 | - | - | - |
| nheads_v | int | 必选 | V的head数`Nv`，无默认值，须满足`Nv <= 96`且`Nv % Nqk == 0`。 | - | - | - |
| seqlens | int | 必选 | 存储长度，无默认值。BNSD、BSND布局传定长存储长度`S`，TND布局传紧凑存储总长度`T`；不能用最大有效序列长度替代存储长度。 | - | - | - |
| headdim_qk | int | 可选 | Q/K的head维度`Dk`，默认128，仅支持128。 | - | - | - |
| headdim_v | int | 可选 | V的head维度`Dv`，默认128，仅支持128。 | - | - | - |
| enable_grad | bool | 可选 | 是否生成训练所需的调度信息，默认`False`。推理使用`False`；训练使用`True`，返回值包含前向和反向调度信息。该参数不会自动切换PyTorch的梯度或推理模式。 | - | - | - |
| layout | str | 可选 | Q/K/V的逻辑布局，默认`"TND"`，支持`"BNSD"`、`"BSND"`和`"TND"`，须与主接口一致。 | - | - | - |

### flash_kda

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| q | Tensor | 必选 | 查询张量，对应$q_t$。 | bfloat16 | ND | 见输入布局表。 |
| k | Tensor | 必选 | 键张量，对应$k_t$，形状与`q`相同。 | bfloat16 | ND | 与`q`相同。 |
| v | Tensor | 必选 | 值张量，对应$v_t$。 | bfloat16 | ND | 见输入布局表。 |
| g | Tensor | 必选 | 融合gate时为激活前输入，否则为单步log gate $\ell_t$。 | bfloat16、float32 | ND | 见输入布局表。 |
| beta | Tensor | 必选 | 融合sigmoid时为激活前输入，否则为更新系数$b_t$。 | bfloat16、float32 | ND | 见输入布局表。 |
| metadata | Tensor | 必选 | `flash_kda_metadata`返回的调度信息，执行时不得为`None`。 | int32 | ND | 一维，长度由接口生成。 |
| initial_state | Tensor | 可选 | 初始状态$H_0$，默认`None`，按零状态计算。 | float32 | ND | `(B,Nv,128,128)` |
| A_log | Tensor | 可选 | 每个value head的gate参数，默认`None`；`use_gate_in_kernel=True`时必传。 | float32 | ND | `(Nv,)` |
| dt_bias | Tensor | 可选 | gate偏置，默认`None`，按零偏置处理。 | float32 | ND | `(Nv,128)`或`(Nv*128,)` |
| cu_seqlens | Tensor | 可选 | 序列长度累积和，默认`None`；TND布局必传，与metadata生成时一致。 | int32 | ND | `(B+1,)` |
| cu_seqlens_cpu | List[int] | 可选 | 保留参数，当前仅支持默认值`None`。 | - | - | - |
| seqused | Tensor | 可选 | 每条序列的有效长度，默认`None`。未传入时使用`cu_seqlens`的相邻差值；两者均未传入时使用`S`。 | int32 | ND | `(B,)` |
| scale | float | 可选 | 输出缩放系数，默认`0.08838834764831845`，即`128**(-0.5)`；支持有限数值或`None`，`None`使用默认缩放。 | - | - | - |
| output_final_state | bool | 可选 | 是否返回最终状态，默认`False`。 | - | - | - |
| return_intermediate_states | bool | 可选 | 推理时是否返回分块入口状态，默认`False`；训练时由`disable_recompute`决定。 | - | - | - |
| use_qk_l2norm_in_kernel | bool | 可选 | 是否融合Q/K的L2归一化，默认`True`。 | - | - | - |
| use_gate_in_kernel | bool | 可选 | 是否融合gate激活，默认`True`。 | - | - | - |
| use_beta_sigmoid_in_kernel | bool | 可选 | 是否融合beta sigmoid，默认`True`。 | - | - | - |
| allow_neg_eigval | bool | 可选 | 是否将融合后的sigmoid结果乘2，默认`False`；取`True`时要求`use_beta_sigmoid_in_kernel=True`。 | - | - | - |
| disable_recompute | bool | 可选 | 训练时是否保存反向缓存以避免重计算，默认`False`。 | - | - | - |
| lower_bound | float | 可选 | gate激活下界，默认`None`；融合gate时支持`[-5,0)`内有限数值。 | - | - | - |
| layout | str | 可选 | Q/K/V的逻辑布局，默认`"TND"`，支持`"BNSD"`、`"BSND"`和`"TND"`。 | - | - | - |

输入布局表如下。`g`和`beta`在定长场景始终按序列维在head维之前的顺序存储。

| layout | q、k | v | g | beta |
| --- | --- | --- | --- | --- |
| BNSD | `(B,Nqk,S,128)` | `(B,Nv,S,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| BSND | `(B,S,Nqk,128)` | `(B,S,Nv,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| TND | `(T,Nqk,128)` | `(T,Nv,128)` | `(T,Nv,128)` | `(T,Nv)` |

## 返回值说明

### flash_kda_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| metadata | Tensor | 必选 | 连续的一维调度信息张量。`enable_grad=False`时包含前向调度信息，`True`时还包含反向调度信息；将返回值原样传给`flash_kda`的`metadata`参数。 | int32 | ND | 一维，长度由接口根据输入规格和运行模式生成。 |

### flash_kda

固定返回`(out, final_state, chunk_states)`。可选输出未请求时返回`None`，元组长度始终为3。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- | --- |
| out | Tensor | 必选 | 前向结果$o_t$；BNSD输入也返回BSND布局。 | bfloat16 | ND | 定长：`(B,S,Nv,128)`；TND：`(T,Nv,128)`。 |
| final_state | Tensor | 可选 | `output_final_state=True`时返回最终状态。 | float32 | ND | `(B,Nv,128,128)` |
| chunk_states | Tensor | 可选 | 推理且`return_intermediate_states=True`，或训练且`disable_recompute=True`时返回分块入口状态；不参与自动求导。 | bfloat16 | ND | 定长：`(B,C,Nv,128,128)`；TND：`(Ctotal,Nv,128,128)`。 |

`C=ceil(S/64)`；`Ctotal=ceil(T/64)+B`，包含TND分块对齐空间。序列`b`的有效状态从`floor(cu_seqlens[b]/64)+b`开始，占用`ceil(length_b/64)`个位置，其中`length_b`为该序列有效长度；仅消费这些有效位置。

## 约束说明

- 该接口支持训练、推理场景下的单算子模式调用。
- `B`、`S`或`T`必须为正数；`1 <= Nqk <= Nv <= 96`且`Nv % Nqk == 0`，Q/K和V的head维度均为128。metadata接口的整数维度及有效序列长度之和不得超过int32范围。
- 非连续张量支持情况：所有参与计算的输入张量（包括`flash_kda_metadata`与`flash_kda`的张量输入）均不支持非连续张量，传入非连续张量会校验报错。
- 空张量支持情况：所有参与计算的输入张量（包括`flash_kda_metadata`与`flash_kda`的张量输入）均不支持空张量（任意维度为0或元素数为0），传入空张量会校验报错。未提供可选张量参数时应传入`None`。
- 在同一NPU上准备输入和metadata，调用前设置当前设备；Meta张量仅用于输出规格推导。
- TND布局中，`cu_seqlens`从0开始、严格递增，末项等于`T`。定长布局中若提供`cu_seqlens`，相邻差值必须大于0且不超过`S`。`seqused`必须大于0且不超过存储长度；与`cu_seqlens`同时提供时，每一项必须等于对应的相邻差值，不支持用较小的`seqused`截断该区间。
- metadata必须与本次调用的布局、存储维度、head数、有效长度、运行模式和设备一致。上述条件变化后重新生成；不得跨设备或跨实现版本复用。定长布局的`seqlens`始终传`S`，TND始终传`T`。
- 前向输出及缓存的训练/推理协议由`torch.is_inference_mode_enabled()`决定。推理应使用`torch.inference_mode()`并设置`enable_grad=False`；`torch.no_grad()`不会切换为推理协议。训练使用`enable_grad=True`，`disable_recompute=False`保存必要缓存并在反向时重计算，取`True`时保存额外缓存。
- 自动求导使用`flash_kda`：开启梯度记录且至少一个输入需要梯度时，其内部保存反向缓存并调用FlashKDAGrad；不支持二阶梯度。
- 未融合gate时，`g`直接表示单步log gate，`A_log`、`dt_bias`和`lower_bound`不参与计算。未融合beta sigmoid时，`beta`直接表示更新系数，此时`allow_neg_eigval`必须为`False`。
- 输出存储可能包含分块对齐空间；不保证返回张量连续，后续接口如要求连续输入，应显式调用`.contiguous()`。仅消费有效token及有效分块对应的结果，未使用的存储区域无有效值保证。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认不支持Batch一致性。修改批量或序列分组可能改变调度方式，不保证同一序列在不同批量组合下逐比特一致。

## 调用示例

以下示例均为单算子模式，通过`import cann_ops_transformer`导入已安装的包。前两个示例执行推理；第三个示例还要求加载配套的FlashKDAGrad扩展。

### 定长推理

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 65, 2, 4, 128

# BSND输入；g和beta始终将序列维放在head维之前。
q = torch.randn(batch, length, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(batch, length, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(batch, length, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

with torch.inference_mode():
    metadata = cann_ops_transformer.flash_kda_metadata(
        batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=length, enable_grad=False, layout="BSND",
    )
    out, final_state, chunk_states = cann_ops_transformer.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        layout="BSND", output_final_state=True,
        return_intermediate_states=True,
    )

torch.npu.synchronize()
assert out.shape == (batch, length, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states.shape == (batch, 2, n_v, dim, dim)
```

### TND变长推理

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
lengths = [65, 33]
batch, total, n_qk, n_v, dim = 2, sum(lengths), 2, 4, 128
cu_seqlens = torch.tensor([0, 65, 98], device="npu", dtype=torch.int32)
q = torch.randn(total, n_qk, dim, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(total, n_v, dim, device="npu", dtype=torch.bfloat16)
g = torch.randn(total, n_v, dim, device="npu", dtype=torch.float32)
beta = torch.randn(total, n_v, device="npu", dtype=torch.float32)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32)

# seqlens传入存储总长度T；前向复用同一份cu_seqlens。
with torch.inference_mode():
    metadata = cann_ops_transformer.flash_kda_metadata(
        cu_seqlens, batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
        seqlens=total, enable_grad=False, layout="TND",
    )
    out, final_state, chunk_states = cann_ops_transformer.flash_kda(
        q, k, v, g, beta, metadata, A_log=A_log,
        cu_seqlens=cu_seqlens, layout="TND", output_final_state=True,
    )

torch.npu.synchronize()
assert out.shape == (total, n_v, dim)
assert final_state.shape == (batch, n_v, dim, dim)
assert chunk_states is None
```

### 自动求导训练

```python
import torch
import torch_npu
import cann_ops_transformer

torch.npu.set_device(0)
batch, length, n_qk, n_v, dim = 1, 64, 1, 1, 128

# 训练输入需要梯度，且不进入torch.inference_mode()。
q = torch.randn(batch, length, n_qk, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(batch, length, n_v, dim, device="npu",
                dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(batch, length, n_v, dim, device="npu",
                dtype=torch.float32, requires_grad=True)
beta = torch.randn(batch, length, n_v, device="npu",
                   dtype=torch.float32, requires_grad=True)
A_log = torch.zeros(n_v, device="npu", dtype=torch.float32, requires_grad=True)

# enable_grad=True生成前向与反向调度，flash_kda内部管理反向缓存。
metadata = cann_ops_transformer.flash_kda_metadata(
    batch_size=batch, nheads_qk=n_qk, nheads_v=n_v,
    seqlens=length, enable_grad=True, layout="BSND",
)
out, final_state, chunk_states = cann_ops_transformer.flash_kda(
    q, k, v, g, beta, metadata, A_log=A_log, layout="BSND",
)
out.float().sum().backward()
torch.npu.synchronize()
assert q.grad is not None and q.grad.shape == q.shape
assert v.grad is not None and v.grad.shape == v.shape
assert final_state is None and chunk_states is None
```
