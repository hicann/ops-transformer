# FlashKDAGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      √     |
|<term>Atlas A3训练系列产品</term>|      ×     |
|<term>Atlas A3推理系列产品</term>|      ×     |
|<term>Atlas A2训练系列产品</term>|      ×     |
|<term>Atlas A2推理系列产品</term>|      ×     |

## 功能说明

- 算子功能：训练场景下计算 Flash Kimi Delta Attention（KDA）的反向输出。算子接收前向输入、前向缓存、
  输出梯度 `do` 与最终状态梯度 `dht`，输出 Q、K、V、初始状态、gate、beta 以及 gate 参数的梯度，
  支持变长序列、GQA、Q/K L2 归一化、gate 激活和 beta sigmoid 融合。

- 计算公式：对一条序列的一个 value head，令 $\bar q_t$、$\bar k_t$ 为参与注意力计算的 Q/K，
  $\ell_t$ 为单步 log gate，$b_t$ 为激活后的 beta，状态 $H_t$ 按 `[Dv,Dk]` 存储，则前向为：

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

  反向以 `do`、`dht` 分别作为 $o_t$ 与最终状态的上游梯度，输出：

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

  启用 Q/K L2 归一化时，`dq/dk` 为归一化前 Q/K 的梯度。

- 实现说明：反向采用 chunk 分解与状态梯度逆序扫描，Cube 操作数使用 BF16，状态缓存与状态梯度使用 FP32。
  默认消费前向保存的七项缓存（`disable_recompute=True`）；显式设置 `disable_recompute=False` 时，
  在同一次 kernel 内先重算所需缓存再执行反向。Host 按有效序列长度、head 数和设备核数自动选择
  persistent 或 staged 调度。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
  <colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 460px">
  <col style="width: 220px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>q</td>
      <td>输入</td>
      <td>公式中的 Q。启用 L2 归一化时为前向归一化后的值，否则为原始值。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中的 K，取值要求同 q。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输入</td>
      <td>公式中的 V。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>g</td>
      <td>输入</td>
      <td>门控输入。融合 gate 时为原始 gate 输入，未融合时为单步 log gate。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>每 token、每 value head 的更新系数。融合时为 sigmoid 前的输入，未融合时为已激活的系数。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>do</td>
      <td>输入</td>
      <td>前向输出 o 的上游梯度。无该路梯度时传同形状零 Tensor，不能传 None。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dht</td>
      <td>输入</td>
      <td>前向最终状态的上游梯度。无该路梯度时传零 Tensor，不能传 None。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Aqk</td>
      <td>输入</td>
      <td>前向因果 QK 矩阵缓存，已包含 scale，两种模式均必传且不重算。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Akk</td>
      <td>输入</td>
      <td>前向 WY 变换的单位下三角矩阵逆缓存，两种模式均必传且不重算。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输入</td>
      <td>反向任务调度信息，由 flash_kda_grad_metadata 生成，不能使用前向 metadata 替代。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initial_state</td>
      <td>输入</td>
      <td>前向初始状态。重算时使用，None 表示零状态；保存缓存模式不使用。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>chunk_states</td>
      <td>输入</td>
      <td>各 chunk 入口状态。保存缓存模式必传；重算模式忽略并重新生成。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>WY 变换后的 Key 缓存。保存缓存模式必传；重算模式重新生成。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>u</td>
      <td>输入</td>
      <td>WY 变换后的 Value 缓存。保存缓存模式仍须提供；反向数值计算不读取其保存值。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v_new</td>
      <td>输入</td>
      <td>结合 chunk 入口状态修正后的 Value。保存缓存模式必传；重算模式重新生成。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>qg</td>
      <td>输入</td>
      <td>结合累计 gate 的 Q 缓存，不包含 scale。保存缓存模式必传。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kg</td>
      <td>输入</td>
      <td>结合累计 gate 的 K 缓存。保存缓存模式必传。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>g_cumsum</td>
      <td>输入</td>
      <td>chunk 内累计 log gate。保存缓存模式必传，不能用逐 token 的原始 g 替代。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>q_rtsd</td>
      <td>输入</td>
      <td>前向 Q L2Norm 的逆范数。开启 Q/K 归一化时必传，不参与重算。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k_rstd</td>
      <td>输入</td>
      <td>前向 K L2Norm 的逆范数，使用要求同 q_rtsd。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta_sigmoid</td>
      <td>输入</td>
      <td>前向激活后的 beta。开启 beta sigmoid 融合时必传；启用负特征值时已包含系数 2。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>A_log</td>
      <td>输入</td>
      <td>每个 value head 的 gate 参数。use_gate_in_kernel=True 时必传。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dt_bias</td>
      <td>输入</td>
      <td>每个 value head、每个 Key 通道的 gate 偏置。融合 gate 时 None 表示零偏置。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cu_seqlens</td>
      <td>输入</td>
      <td>TND 布局的 token 前缀和。主反向可省略；显式传入时须与 metadata 一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seqused</td>
      <td>输入</td>
      <td>BSND/BNSD 布局的有效序列长度。主反向可省略；显式传入时须与 metadata 一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>输出/Query 缩放系数。None 表示 1/sqrt(Dk)，显式值须为有限数值且与前向一致。</td>
      <td>float、None</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_qk_l2norm_in_kernel</td>
      <td>属性</td>
      <td>是否计算 Q/K 归一化反向，默认 True。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_gate_in_kernel</td>
      <td>属性</td>
      <td>是否计算 gate 激活反向，默认 True。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_beta_sigmoid_in_kernel</td>
      <td>属性</td>
      <td>是否计算 beta sigmoid 反向，默认 True。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>allow_neg_eigval</td>
      <td>属性</td>
      <td>是否使用 2*sigmoid(beta)。默认 False；为 True 时必须开启 beta sigmoid 融合。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>disable_recompute</td>
      <td>属性</td>
      <td>是否使用保存的七项缓存。默认 True，要求完整缓存；为 False 时启用重算。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lower_bound</td>
      <td>属性</td>
      <td>融合 gate 的下界及公式选择。None 使用 softplus；非空时须在 [-5,0)，使用有界 sigmoid。</td>
      <td>float、None</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>属性</td>
      <td>输入和对应梯度的布局，默认 "TND"，支持 "TND"、"BNSD"、"BSND"。</td>
      <td>str</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dq</td>
      <td>输出</td>
      <td>Q 梯度。GQA 下已沿对应 value heads 求和；开启归一化时包含 L2Norm 反向。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dk</td>
      <td>输出</td>
      <td>K 梯度，取值要求同 dq。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dv</td>
      <td>输出</td>
      <td>V 梯度。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dh0</td>
      <td>输出</td>
      <td>初始状态梯度，始终为 FP32 的 V-first 状态；空序列对应 dht。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dg</td>
      <td>输出</td>
      <td>gate 输入梯度，保持 g 的 dtype 和形状。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>输出</td>
      <td>beta 输入梯度，保持 beta 的 dtype 和形状。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dA</td>
      <td>输出</td>
      <td>A_log 梯度。关闭 gate 融合时为 None。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dBias</td>
      <td>输出</td>
      <td>dt_bias 梯度。关闭 gate 融合或未传 dt_bias 时为 None。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入 q、k、v、do 的数据类型必须一致（BFLOAT16）；状态、状态梯度、矩阵缓存的 dtype 按上表固定。
- 固定 `Dk=Dv=128`、`C=64`。chunk 缓存容量：TND 为 `Nc = ceil(T/C) + B`，padded 布局为 `Cmax = ceil(S/C)`。
- 关于数据 shape 的约束，以 layout 的 TND、BNSD、BSND 为例：
  - B：逻辑序列数，`B >= 1`。
  - S：BSND/BNSD 的物理序列长度。
  - T：TND 的总 token 数，等于所有序列有效长度之和。
  - Nqk：Q/K head 数；Nv：V head 数。
  - Dk、Dv：Q/K 与 V 的 head dim，当前均为 128。
- `1 <= Nqk <= Nv` 且 `Nv % Nqk == 0`（GQA）；GQA 下 Q/K 输入和梯度按 Nqk 个 head 存储，
  gate、beta、矩阵和状态缓存按 Nv 个 head 存储。
- `g` 与 `v` 形状相同，`beta` 的形状为 `v.shape[:-1]`；`dg`、`dbeta` 的形状和 dtype 跟随输入。
- 必须提供 `do` 与 `dht`；即使只对输出或最终状态中的一路求导，也须为另一路提供形状、类型匹配的零 Tensor。
- `Aqk`、`Akk`、`metadata` 在两种模式下都必须传入；`disable_recompute=True` 时还必须传入
  `chunk_states`、`w`、`u`、`v_new`、`qg`、`kg`、`g_cumsum` 七项缓存。
- 融合开关打开时必须提供对应缓存：`use_qk_l2norm_in_kernel=True` 需 `q_rtsd`/`k_rstd`；
  `use_beta_sigmoid_in_kernel=True` 需 `beta_sigmoid`；`use_gate_in_kernel=True` 需 `A_log`（`dt_bias` 可省略）。
- `allow_neg_eigval=True` 要求 `use_beta_sigmoid_in_kernel=True`。
- `lower_bound` 只能为 None 或 `[-5, 0)` 内的值，0 不被反向入口接受。
- TND 的 `cu_seqlens` 长度为 `B+1`、首项为 0、非递减、末项为 T；BSND/BNSD 的 `seqused` 长度为 `B`，
  每项在 `[0, S]` 内。两种长度 Tensor 互斥；主反向显式传入时须与 metadata 一致。
- `metadata` 为连续的一维 INT32 Tensor，长度为 `8 + 7*B*Nv`；前向和反向 metadata 的 ABI 不同，必须分别生成，
  改变维度、布局或有效长度后须重新生成。首次解析会校验 magic、ABI 版本、状态码与输入形状，非法快照抛 `ValueError`。
- 缓存必须来自同一次前向并保留当前前向返回的形状和存储顺序；BSND 输入的缓存仍为 head 在前的布局。
  TND 中第 `s` 条序列的第 `j` 个 chunk 使用槽位 `floor(cu_seqlens[s]/64) + s + j`。
- 允许空序列和不足 64 token 的尾块；空序列将 `dht` 传入 `dh0`，padded 布局的无效 token 在
  `dq/dk/dv/dg/dbeta` 中写零。
- 输出不保证连续存储（例如 `dbeta` 可能为转置视图、`dA` 为列视图），需要连续 Tensor 的下游应显式
  `.contiguous()`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---|:---|:---|
| Torch 接口调用 | [torchapi_flash_kda_grad.md](./docs/torchapi_flash_kda_grad.md) | 通过 `cann_ops_transformer.flash_kda_grad` 与 `cann_ops_transformer.flash_kda_grad_metadata` 调用，含完整接口原型、参数、返回值与调用示例。 |
| DSL 直接调用 | [op_dsl/flash_kda_grad.py](./op_dsl/flash_kda_grad.py) | 通过 `flash_kda_grad_npu`（主入口）与 `flash_kda_grad_from_forward`（前向缓存适配入口）直接调用 DSL 实现。 |
| 单元测试 | [tests/ut](./tests/ut) | 算子 UT、异常契约、性能与精度比较入口。 |
| TTK 精度与性能 | [tests/ttk/README.md](./tests/ttk/README.md) | STC 用例执行、dump/replay 模式与 `cross_check L2` 判据说明。 |

## 参考资源

- [Torch 接口文档（中文）](./docs/torchapi_flash_kda_grad.md)
- [Torch API document (English)](./docs/torchapi_flash_kda_grad_en.md)
- [TTK 精度与性能说明](./tests/ttk/README.md)
