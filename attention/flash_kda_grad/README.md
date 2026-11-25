# FlashKDAGrad

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：计算[Kimi Delta Attention（KDA）](../flash_kda/README.md)的训练反向，根据前向缓存、输出梯度`do`和可选的最终状态梯度`dht`，返回Q、K、V、初始状态、gate、beta及gate参数的梯度。支持变长序列、分组查询注意力（Grouped Query Attention，GQA）、Q/K的L2归一化和融合激活的反向。
- 计算公式：对一条序列的一个value head，令$\bar q_t$、$\bar k_t$为前向实际使用的Q/K，$\ell_t$为单步log gate，$b_t$为激活后的beta，状态$H_t$按`(Dv,Dk)`存储，则前向递推和损失为：

$$
\widetilde H_t=H_{t-1}\operatorname{diag}(\exp(\ell_t)),\quad
\Delta v_t=b_t(v_t-\widetilde H_t\bar k_t),\quad
H_t=\widetilde H_t+\Delta v_t\bar k_t^T,\quad
o_t=scale\cdot H_t\bar q_t
$$

$$
L=\sum_t\langle do_t,o_t\rangle+\langle dht,H_{last}\rangle
$$

反向计算上述损失对输入及gate参数的梯度，固定返回`(dq,dk,dv,dh0,dg,dbeta,dA,dBias)`。开启Q/K归一化反向时，`dq/dk`对应归一化前的Q/K；`dht=None`表示损失不包含最终状态项。gate和beta的激活公式见[PyTorch接口文档](docs/torchapi_flash_kda_grad.md#功能说明)。

- 实现说明：固定按64个token分块，主机根据有效长度、head数和设备核数选择persistent或staged流水。默认`disable_recompute=False`，在反向kernel内重建七项缓存；设为`True`时使用前向保存的缓存。保存的`chunk_states`为BFLOAT16，重算状态缓冲和状态梯度为FLOAT32。

## 参数说明

本算子基于CANNBot DSL实现。下表按公开PyTorch接口顺序列出主反向参数和返回值；metadata生成接口的参数、缓存shape与完整条件约束见[PyTorch接口文档](docs/torchapi_flash_kda_grad.md)。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| q | 输入 | 查询张量。开启Q/K归一化反向时，传前向保存的`q_norm`；关闭时传原始Q。 | BFLOAT16 | ND |
| k | 输入 | 键张量，形状与q相同。开启Q/K归一化反向时传前向保存的`k_norm`。 | BFLOAT16 | ND |
| v | 输入 | 前向使用的原始Value张量。 | BFLOAT16 | ND |
| g | 输入 | Gate输入。融合gate时为激活前输入，且必须为bfloat16；未融合时为已处理的单步log gate，且必须为float32。 | BFLOAT16、FLOAT32 | ND |
| beta | 输入 | 更新系数输入。融合sigmoid时为激活前输入，且必须为bfloat16；未融合时为已激活的系数，且必须为float32。 | BFLOAT16、FLOAT32 | ND |
| do | 输入 | 前向输出的上游梯度。必须提供Tensor；仅对最终状态求导时，传形状匹配的零Tensor。 | BFLOAT16 | ND |
| dht | 可选输入 | 最终状态的上游梯度，默认`None`，按零梯度处理。 | FLOAT32 | ND |
| Aqk | 输入 | 前向因果QK矩阵缓存，包含scale。必传，无默认值，须通过关键字传入；两种重算模式均不重算该项。 | BFLOAT16 | ND |
| Akk | 输入 | 前向WY变换的块内逆矩阵缓存。必传，无默认值，须通过关键字传入。 | BFLOAT16 | ND |
| metadata | 可选输入 | 反向调度信息，默认`None`，但执行时必须提供`flash_kda_grad_metadata`的返回值。 | INT32 | ND |
| initial_state | 可选输入 | 前向初始状态，默认`None`。重算时`None`表示零初态，非零初态必须显式传入；该参数是否为`None`还决定是否返回dh0。 | FLOAT32 | ND |
| chunk_states | 可选输入 | 各chunk入口状态，默认`None`。保存缓存模式必传。 | BFLOAT16 | ND |
| w | 可选输入 | WY变换后的Key缓存，默认`None`。保存缓存模式必传。 | BFLOAT16 | ND |
| u | 可选输入 | WY变换后的Value缓存，默认`None`。保存缓存模式须提供，保持前向缓存协议完整。 | BFLOAT16 | ND |
| v_new | 可选输入 | 结合chunk入口状态修正后的Value缓存，默认`None`。保存缓存模式必传。 | BFLOAT16 | ND |
| qg | 可选输入 | 结合累计gate的Q缓存，不包含scale，默认`None`。保存缓存模式必传。 | BFLOAT16 | ND |
| kg | 可选输入 | 结合累计gate的K缓存，默认`None`。保存缓存模式必传。 | BFLOAT16 | ND |
| g_cumsum | 可选输入 | chunk内累计log gate，默认`None`。保存缓存模式必传，不能用原始g替代。 | FLOAT32 | ND |
| q_rstd | 可选输入 | 前向Q归一化的逆范数，默认`None`。开启Q/K归一化反向时必传。 | FLOAT32 | ND |
| k_rstd | 可选输入 | 前向K归一化的逆范数，默认`None`。开启Q/K归一化反向时必传。 | FLOAT32 | ND |
| beta_sigmoid | 可选输入 | 前向激活后的beta，默认`None`。融合beta sigmoid时必传；负特征值模式下已包含系数2。 | FLOAT32 | ND |
| A_log | 可选输入 | 每个value head的gate参数，默认`None`。融合gate时必传。 | FLOAT32 | ND |
| dt_bias | 可选输入 | Gate偏置，默认`None`，按零偏置处理。 | FLOAT32 | ND |
| cu_seqlens | 可选输入 | TND序列边界，默认`None`。主反向可省略；显式提供时须与metadata描述的边界一致。 | INT32 | ND |
| cu_seqlens_cpu | 可选属性 | <ul><li>预留参数。</li><li>默认`None`，当前仅允许`None`。</li></ul> | INT列表 | - |
| seqused | 可选输入 | BNSD/BSND有效长度，默认`None`。主反向可省略；显式提供时须与metadata一致。 | INT32 | ND |
| scale | 可选属性 | <ul><li>Q/输出缩放系数。</li><li>默认`None`，使用`1/sqrt(128)`；显式值须为有限数值，并与前向一致。</li></ul> | FLOAT | - |
| use_qk_l2norm_in_kernel | 可选属性 | <ul><li>是否计算Q/K的L2归一化反向。</li><li>默认`True`；支持`True`、`False`。</li></ul> | BOOL | - |
| use_gate_in_kernel | 可选属性 | <ul><li>是否计算gate激活反向。</li><li>默认`True`；支持`True`、`False`。</li></ul> | BOOL | - |
| use_beta_sigmoid_in_kernel | 可选属性 | <ul><li>是否计算beta sigmoid反向。</li><li>默认`True`；支持`True`、`False`。</li></ul> | BOOL | - |
| allow_neg_eigval | 可选属性 | <ul><li>是否使用`2*sigmoid(beta)`。</li><li>默认`False`；支持`True`、`False`，取`True`时要求融合beta sigmoid。</li></ul> | BOOL | - |
| disable_recompute | 可选属性 | <ul><li>是否关闭缓存重算。</li><li>默认`False`。`False`时重建七项缓存，`True`时消费已保存缓存。</li></ul> | BOOL | - |
| lower_bound | 可选属性 | <ul><li>Gate公式选择及下界。</li><li>默认`None`。非空值须在`[-5,0)`内；融合gate时，`None`选择softplus，非空值选择有界sigmoid。</li></ul> | FLOAT | - |
| layout | 可选属性 | <ul><li>Q/K/V的逻辑布局。</li><li>默认`"TND"`；支持`"TND"`、`"BNSD"`和`"BSND"`。g、beta和do的布局见[输入布局](docs/torchapi_flash_kda_grad.md#输入布局)。</li></ul> | STRING | - |
| dbeta_dtype | 可选属性 | <ul><li>dbeta的输出类型。</li><li>默认`None`，使用`beta.dtype`；显式值仅支持`beta.dtype`或`torch.bfloat16`。</li></ul> | torch.dtype | - |
| dq | 输出 | Q梯度。开启归一化时已包含L2归一化反向；GQA下已沿对应value heads求和。 | BFLOAT16 | ND |
| dk | 输出 | K梯度，归一化和GQA语义与dq一致。 | BFLOAT16 | ND |
| dv | 输出 | V梯度，形状和逻辑布局与v相同。 | BFLOAT16 | ND |
| dh0 | 可选输出 | 初始状态梯度。`initial_state=None`时返回`None`，否则返回float32 Tensor。 | FLOAT32 | ND |
| dg | 输出 | Gate输入梯度，保持g的dtype和形状；未融合时为对单步log gate的梯度。 | BFLOAT16、FLOAT32 | ND |
| dbeta | 输出 | Beta输入梯度，形状与beta相同，dtype由dbeta_dtype决定；未融合时为对激活后系数的梯度。 | BFLOAT16、FLOAT32 | ND |
| dA | 可选输出 | 对A_log的梯度。关闭gate融合时返回`None`。 | FLOAT32 | ND |
| dBias | 可选输出 | 对dt_bias的梯度，形状与dt_bias相同。关闭gate融合或`dt_bias=None`时返回`None`。 | FLOAT32 | ND |

梯度保持对应输入的逻辑布局。张量形状和存储顺序见[输入布局](docs/torchapi_flash_kda_grad.md#输入布局)和[前向缓存布局](docs/torchapi_flash_kda_grad.md#前向缓存布局)。

## 约束说明

- 当前固定`Dk=Dv=128`、chunk大小为64；`B >= 1`，`S/T >= 0`，head数满足`1 <= Nqk <= Nv`且`Nv % Nqk == 0`。反向支持空序列和不足64个token的尾块。
- `Aqk`、`Akk`、反向metadata及启用融合所需的辅助缓存均须来自匹配的前向规格。保存缓存模式还须提供七项完整缓存；修改布局、维度或有效长度后须重新生成metadata。
- 定长场景的g、beta和do始终将序列维放在head维之前；BSND输入的缓存仍按head在前存储，不能套用主输入布局。缓存须保留前向返回的shape和存储顺序。
- 所有参与计算的张量须位于同一NPU，长度张量和metadata须连续。融合开关、dtype、长度范围及输出连续性等接口约束见[PyTorch接口文档](docs/torchapi_flash_kda_grad.md#约束说明)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| PyTorch API | [自动求导、保存缓存与重算示例](docs/torchapi_flash_kda_grad.md#调用示例) | 使用`cann_ops_transformer.flash_kda_grad_metadata`生成反向metadata，再调用[flash_kda_grad](docs/torchapi_flash_kda_grad.md)；常规训练可通过[FlashKDA自动求导](../flash_kda/docs/torchapi_flash_kda.md)管理缓存和反向调用。 |
| DSL直接调用 | - | 通过[DSL实现](op_dsl/flash_kda_grad.py)中的`flash_kda_grad_npu`或`flash_kda_grad_from_forward`调用。 |

构建与安装方式参见[PyTorch扩展说明](../../torch_extension/README.md)。运行需要匹配的CANN、torch_npu和CANNBot DSL环境。

## 参考资源

- [单元测试与接口契约测试](tests/ut/test_flash_kda_grad_ut.py)
- [TTK精度与性能测试指南](tests/assets/README.md)
- [FlashKDA前向算子](../flash_kda/README.md)
- [English API reference](docs/torchapi_flash_kda_grad_en.md)
