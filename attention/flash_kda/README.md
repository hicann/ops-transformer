# FlashKDA

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

- 算子功能：计算Kimi Delta Attention（KDA）的前向输出，以逐通道门控衰减和增量更新维护注意力状态，支持变长序列、分组查询注意力（GQA）、Q/K的L2归一化、gate激活和beta sigmoid融合。
- 计算公式：对于一条序列的一个value head，令$\bar q_t$、$\bar k_t$为参与计算的Q/K，$\ell_t$为单步log gate，$b_t$为激活后的beta。状态$H_t$按$[D_v,D_k]$存储：

  $$
  \widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t))
  $$

  $$
  \Delta v_t=b_t(v_t-\widetilde H_t\bar k_t),\qquad
  H_t=\widetilde H_t+\Delta v_t\bar k_t^T
  $$

  $$
  o_t=scale\cdot H_t\bar q_t
  $$

  未传入`initial_state`时，$H_0$为零。启用归一化时，Q/K按最后一维进行L2归一化；GQA按连续分组将value head映射到对应的Q/K head。
- 实现说明：以64个token为一个chunk组织计算，Host根据有效序列长度、head数和设备核数选择staged或persistent调度。状态递推使用FLOAT32，矩阵计算使用BFLOAT16。`flash_kda`返回前向输出、可选的最终状态和分块入口状态。

## 参数说明

本算子通过PyTorch接口调用，以下参数按`torch_extension/flash_kda.py`中的公开接口顺序列出。`B`为序列数，`S`为定长存储长度，`T`为紧凑存储总长度，`Nqk`和`Nv`分别为Q/K与V的head数，`Dk=Dv=128`。张量格式均为ND，逻辑布局由`layout`指定。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| q | 输入 | 查询张量，对应公式中的$\bar q_t$或归一化前的Q。形状见下表。 | BFLOAT16 | ND |
| k | 输入 | 键张量，与q形状相同，对应$\bar k_t$或归一化前的K。 | BFLOAT16 | ND |
| v | 输入 | 值张量，对应$v_t$。 | BFLOAT16 | ND |
| g | 输入 | 融合gate时为激活前输入，否则为单步log gate $\ell_t$。 | BFLOAT16、FLOAT32 | ND |
| beta | 输入 | 融合sigmoid时为激活前输入，否则为更新系数$b_t$。 | BFLOAT16、FLOAT32 | ND |
| metadata | 输入 | 由`flash_kda_metadata`生成的一维调度张量，执行时必传。 | INT32 | ND |
| initial_state | 可选输入 | 初始状态$H_0$，形状为`(B,Nv,128,128)`；默认`None`，按零状态计算。 | FLOAT32 | ND |
| A_log | 可选输入 | gate的每head参数，形状为`(Nv,)`；默认`None`，`use_gate_in_kernel=True`时必传。 | FLOAT32 | ND |
| dt_bias | 可选输入 | gate偏置，形状为`(Nv,128)`或`(Nv*128,)`；默认`None`，按零偏置处理。 | FLOAT32 | ND |
| cu_seqlens | 可选输入 | 序列长度的累积和，形状为`(B+1,)`；默认`None`，TND布局下必传。 | INT32 | ND |
| cu_seqlens_cpu | 可选输入 | 保留参数，当前仅支持默认值`None`。 | INT列表 | - |
| seqused | 可选输入 | 每条序列参与计算的有效长度，形状为`(B,)`；默认`None`，使用`cu_seqlens`指定的长度；两者均未传入时使用存储长度`S`。 | INT32 | ND |
| scale | 可选属性 | 输出缩放系数，默认`128**(-0.5)`；支持有限数值或`None`，`None`使用默认缩放。 | FLOAT | - |
| output_final_state | 可选属性 | 是否返回最终状态，取`True`或`False`，默认`False`。 | BOOL | - |
| return_intermediate_states | 可选属性 | 推理时是否返回chunk入口状态，取`True`或`False`，默认`False`。 | BOOL | - |
| use_qk_l2norm_in_kernel | 可选属性 | 是否融合Q/K的L2归一化，取`True`或`False`，默认`True`。 | BOOL | - |
| use_gate_in_kernel | 可选属性 | 是否融合gate激活，取`True`或`False`，默认`True`。 | BOOL | - |
| use_beta_sigmoid_in_kernel | 可选属性 | 是否融合beta sigmoid，取`True`或`False`，默认`True`。 | BOOL | - |
| allow_neg_eigval | 可选属性 | 是否将融合后的sigmoid结果乘2，取`True`或`False`，默认`False`。取`True`时要求启用beta sigmoid融合。 | BOOL | - |
| disable_recompute | 可选属性 | 训练时是否保存chunk入口状态及反向缓存，取`True`或`False`，默认`False`。 | BOOL | - |
| lower_bound | 可选属性 | gate激活下界，默认`None`；融合gate时支持`[-5,0)`内有限数值。 | FLOAT | - |
| layout | 可选属性 | Q/K/V逻辑布局，取`BNSD`、`BSND`或`TND`，默认`TND`。 | STRING | - |
| out | 输出 | 前向结果$o_t$。定长布局均输出`(B,S,Nv,128)`，TND输出`(T,Nv,128)`。 | BFLOAT16 | ND |
| final_state | 可选输出 | 最终状态，形状为`(B,Nv,128,128)`；未请求时为`None`。 | FLOAT32 | ND |
| chunk_states | 可选输出 | 每个chunk的入口状态，定长布局为`(B,C,Nv,128,128)`，TND为`(Ctotal,Nv,128,128)`；返回条件见接口文档。 | BFLOAT16 | ND |

输入形状与逻辑布局的对应关系如下。`g`和`beta`在定长场景始终按序列维在前的顺序存储。

| layout | q、k | v | g | beta |
| --- | --- | --- | --- | --- |
| BNSD | `(B,Nqk,S,128)` | `(B,Nv,S,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| BSND | `(B,S,Nqk,128)` | `(B,S,Nv,128)` | `(B,S,Nv,128)` | `(B,S,Nv)` |
| TND | `(T,Nqk,128)` | `(T,Nv,128)` | `(T,Nv,128)` | `(T,Nv)` |

## 约束说明

- `B`、`S`或`T`为正数；head数满足`1 <= Nqk <= Nv <= 96`且`Nv % Nqk == 0`，Q/K和V的head维度均为128。
- 输入张量须连续。实际计算在NPU上执行，Meta张量仅用于输出规格推导。
- TND布局中，`cu_seqlens`从0开始、严格递增，末项等于`T`。定长布局中若提供`cu_seqlens`，相邻差值必须大于0且不超过`S`。`seqused`必须大于0且不超过存储长度；与`cu_seqlens`同时提供时，须等于对应的相邻差值。
- metadata必须与本次调用的布局、存储维度、head数及有效长度一致；输入规格或长度改变后需要重新生成。
- 训练或推理由`torch.is_inference_mode_enabled()`决定。使用`torch.inference_mode()`进入推理；`torch.no_grad()`仍采用训练输出协议。`flash_kda`的返回条件见[PyTorch接口说明](docs/torchapi_flash_kda.md)。
- 未融合gate时，`g`直接表示单步log gate，`A_log`、`dt_bias`和`lower_bound`不参与计算。未融合beta sigmoid时，`beta`直接表示更新系数。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| PyTorch API | [推理示例](docs/torchapi_flash_kda.md#调用示例) | 通过[flash_kda](docs/torchapi_flash_kda.md)调用。 |

构建与安装方式参见[PyTorch扩展说明](../../torch_extension/README.md)。该算子的DSL实现还需要匹配的CANN、torch_npu和CANNBot DSL运行环境。

## 参考资源

- [测试用例](tests/ttk/kda_cases.py)
- [性能测量指南](tests/ttk/PERFORMANCE.md)
- [FlashKDAGrad反向算子](../flash_kda_grad/README.md)
