# BlockAttnResUpdate

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：将当前增量原地累加到局部残差，计算归一化score，与历史在线Softmax状态合并得到注意力输出，并支持融合输出均方根归一化（Root Mean Square Normalization，RMSNorm）。

- 版本演进：[aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2.md)在[aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate.md)的基础上，新增`output_norm_weight`和`output_norm_eps`对应的可选输出RMSNorm能力。

- 计算公式：

  对每个token $t$，先更新局部残差并原地写回：

  $$
  p_t=P_t+\operatorname{FP32}(\Delta_t),\qquad
  P^{\mathrm{out}}_t=p_t
  $$

  通过RMSNorm和Query点积得到当前残差的score：

  $$
  C_t=\left\langle Q,\operatorname{RMSNorm}(p_t,\mathbf{1},\epsilon)\right\rangle
  $$

  将当前残差与历史在线Softmax状态合并：

  $$
  m_t=\max(M_t,C_t),\qquad
  \alpha_t=\exp(M_t-m_t),\qquad
  \beta_t=\exp(C_t-m_t)
  $$

  $$
  H^{(0)}_t=\operatorname{BF16}\left(
  \frac{\alpha_t O_t+\beta_t p_t}{\alpha_t L_t+\beta_t}
  \right)
  $$

  未传入`output_norm_weight`时，输出 $H=H^{(0)}$；否则对BF16混合结果执行输出RMSNorm：

  $$
  H_t=\operatorname{BF16}\left(
  \operatorname{RMSNorm}\left(\operatorname{FP32}(H^{(0)}_t),\operatorname{FP32}(W),\epsilon_{\mathrm{out}}\right)
  \right)
  $$

  其中：

  - $T$ 和 $D$ 分别表示token数和隐藏维度，$t\in[0,T)$；$P_t$、$p_t$、$\Delta_t$、$O_t$、$H_t$ 和 $Q$ 均为 $D$ 维向量。
  - $P$ 和 $P^{\mathrm{out}}$ 分别表示`partial_block`调用前后的值，$\Delta$ 表示`delta`，$Q$ 表示`pseudo_query`；$p$ 为FP32残差累加结果。
  - $O$、$M$、$L$ 分别表示历史状态`numerator`、`logit_max`、`exp_sum`，$H$ 表示输出`h`。$W$ 表示`output_norm_weight`。
  - $\epsilon=\operatorname{FP32}(\texttt{eps})$，$\epsilon_{\mathrm{out}}=\operatorname{FP32}(\texttt{output\_norm\_eps})$，两者分别用于打分和输出归一化。
  - RMSNorm沿 $D$ 维计算，打分时的缩放参数为全1向量；打分所需的缩放参数可预先与原始Query逐元素相乘，合并到 $Q$ 中。

  对于token $t$ 的非空历史集合 $\mathcal{H}_t$，设历史logit为 $z_{t,i}$、对应的 $D$ 维value为 $v_{t,i}$，输入状态满足：

  $$
  \begin{aligned}
  M_t&=\max_{i\in\mathcal{H}_t}z_{t,i} \\
  L_t&=\sum_{i\in\mathcal{H}_t}\exp(z_{t,i}-M_t)>0 \\
  O_t&=\sum_{i\in\mathcal{H}_t}\exp(z_{t,i}-M_t)v_{t,i}
  \end{aligned}
  $$

## 参数说明

<table style="table-layout: fixed; width: 1000px">
  <colgroup>
    <col style="width: 160px">
    <col style="width: 150px">
    <col style="width: 450px">
    <col style="width: 140px">
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
      <td style="white-space: nowrap;">partial_block</td>
      <td>输入</td>
      <td>待更新的局部残差，对应公式中的P，shape为<code>[T, D]</code>；与同名输出复用存储。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">delta</td>
      <td>输入</td>
      <td>局部残差的增量，对应公式中的Δ，shape为<code>[T, D]</code>。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">pseudo_query</td>
      <td>输入</td>
      <td>用于计算归一化score的有效Query，对应公式中的Q，shape为<code>[D]</code>。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">numerator</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态的加权分子，对应公式中的O，shape为<code>[T, D]</code>；本算子只读。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">logit_max</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态的最大logit，对应公式中的M，shape为<code>[T]</code>；本算子只读。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">exp_sum</td>
      <td>输入</td>
      <td>非空历史在线Softmax状态的指数和，对应公式中的L，shape为<code>[T]</code>；本算子只读。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">output_norm_weight</td>
      <td>可选输入</td>
      <td>输出RMSNorm的gamma，对应公式中的W，shape为<code>[D]</code>；默认不传入，表示禁用输出RMSNorm。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">partial_block</td>
      <td>输出</td>
      <td>FP32残差累加结果，对应公式中的P<sup>out</sup>，shape为<code>[T, D]</code>；与同名输入构成引用关系。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">h</td>
      <td>输出</td>
      <td>当前层输出，对应公式中的H，shape为<code>[T, D]</code>；传入权重时为输出RMSNorm结果。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">eps</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>残差RMS score计算的稳定项，必须为有限正数。</li>
          <li>默认值为<code>1e-6</code>。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">output_norm_eps</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>输出RMSNorm的独立稳定项，启用时转换为FLOAT32前必须在<code>[FLT_MIN, FLT_MAX]</code>范围内；禁用时忽略。</li>
          <li>默认值为<code>1e-6</code>。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 启用输出RMSNorm时，`output_norm_eps`须在`[FLT_MIN, FLT_MAX]`范围内；`FLT_MIN = 2^-126 ≈ 1.1754943508222875e-38`，`FLT_MAX ≈ 3.4028234663852886e38`。不支持次正规epsilon，以免默认Sqrt的FTZ模式将其清零。
- 非空计算时，`T`和`D`必须为已知正整数，且`1 <= D <= 8192`。空Tensor行为请参见[aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate.md#约束说明)、[aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2.md#约束说明)和[PyTorch接口](docs/torchapi_block_attn_res_update.md#约束说明)。
- `partial_block`、`delta`、`numerator`及输出`partial_block`、`h`的shape必须为`[T, D]`；`pseudo_query`和非空`output_norm_weight`为`[D]`；`logit_max`、`exp_sum`为`[T]`。
- 非空计算时，调用者须保证每个token的历史状态非空：`logit_max`和`numerator`中的元素有限，`exp_sum`中的元素有限且严格大于0，且三者来自同一份历史在线Softmax状态。Host不读取或校验这些运行时元素，不支持以`logit_max=-inf`、`exp_sum=0`和零`numerator`表示的空历史状态。
- `T * D`不能超出有符号64位元素偏移的表示范围；按核切分后的`T`大小不能超出`uint32_t`的表示范围。
- 完整D维数据、可选归一化权重及双缓冲区必须满足统一缓冲区（Unified Buffer，UB）预算，否则Tiling失败。
- 输出RMSNorm只作用于注意力混合结果；`partial_block`始终原地保存FP32残差累加值。输入中的零值、极值、NaN或Inf不做额外处理。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_block_attn_res_update](examples/arch35/test_aclnn_block_attn_res_update.cpp) | 通过[aclnnBlockAttnResUpdate](docs/aclnnBlockAttnResUpdate.md)接口方式调用BlockAttnResUpdate算子。 |
| aclnn API（V2，推荐） | [test_aclnn_block_attn_res_update_v2](examples/arch35/test_aclnn_block_attn_res_update_v2.cpp) | 通过[aclnnBlockAttnResUpdateV2](docs/aclnnBlockAttnResUpdateV2.md)接口方式调用BlockAttnResUpdate算子，支持可选输出RMSNorm。 |
| PyTorch API | [调用示例](docs/torchapi_block_attn_res_update.md#调用示例) | 通过[cann_ops_transformer.block_attn_res_update](docs/torchapi_block_attn_res_update.md)接口方式调用BlockAttnResUpdate算子。 |

在仓库根目录执行以下命令编译算子，详细步骤请参考[算子调用](../../docs/zh/invocation/quick_op_invocation.md)。

```bash
bash build.sh --soc=ascend950 --ops=block_attn_res_update
```
