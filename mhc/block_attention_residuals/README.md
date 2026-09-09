# BlockAttentionResiduals

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- **算子功能**：将 `partialBlock` 与 `blockRes` 按 block 维拼接后，完成 RMS 归一化、`normWeight ⊙ projWeight` 投影打分与 Softmax 加权融合，输出 `hiddenStates`。当 `needBackward` 为 true 时，同时输出反向所需的 `invNorm` 和 `probs`。

- **计算公式**：

  $$
  v_{t,i,h} =
  \begin{cases}
  block\_res_{t,i,h}, & i < N \\
  partial\_block_{t,h}, & i = N
  \end{cases}
  $$

  $$
  variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
  $$

  $$
  inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
  $$

  $$
  k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
  $$

  $$
  score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
  $$

  $$
  s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
  $$

  $$
  probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
  $$

  $$
  hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>partialBlock</td>
        <td>输入</td>
        <td>拼接后的第 N 行 value，shape 为 (T, H)。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>blockRes</td>
        <td>输入</td>
        <td>前 N 行 value，shape 为 (T, N, H)；dtype 须与 partialBlock 一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>projWeight</td>
        <td>输入</td>
        <td>投影权重，shape 为 (H) 或 (1, H)；dtype 须与 partialBlock 一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normWeight</td>
        <td>输入</td>
        <td>RMS 缩放权重，shape 为 (H)；dtype 须与 partialBlock 一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>validBlockNum</td>
        <td>可选属性</td>
        <td>有效 block 数量。-1（默认）表示使用 blockRes 的 N；当前仅支持 -1 或等于 N。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps</td>
        <td>可选属性</td>
        <td>RMS 归一化数值稳定性参数，须大于 0，默认 1e-6。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needBackward</td>
        <td>可选属性</td>
        <td>是否输出反向中间量 invNorm、probs。默认 false。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hiddenStates</td>
        <td>输出</td>
        <td>加权融合结果，shape 为 (T, H)；dtype 须与 partialBlock 一致。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invNorm</td>
        <td>输出</td>
        <td>需要反向时输出的逐行归一化系数。needBackward 为 false 时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>probs</td>
        <td>输出</td>
        <td>需要反向时输出的 Softmax 概率。needBackward 为 false 时，输出无效。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- T 大于等于 0，H 大于等于 1，N 取值范围为 1~100。
- partialBlock、blockRes、projWeight、normWeight、hiddenStates 的数据类型须一致。
- validBlockNum 须为 -1（默认，使用 N）或等于 blockRes 的 N。
- <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>计算语义一致。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_block_attention_residuals](examples/test_aclnn_block_attention_residuals.cpp) | 通过[aclnnBlockAttentionResiduals](docs/aclnnBlockAttentionResiduals.md)接口方式调用BlockAttentionResiduals算子。 |
| PyTorch API | - | 通过[cann_ops_transformer.block_attention_residuals](../../mhc/block_attention_residuals/docs/torchapi_block_attention_residuals.md)接口方式调用BlockAttentionResiduals算子。 |
