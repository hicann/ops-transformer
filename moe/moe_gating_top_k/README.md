# MoeGatingTopK

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>                  |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：MoE计算中，对输入x做Sigmoid、SoftMax或者SqrtSoftplus计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。支持两种模式：
  - **TopK模式**：对normValue进行TopK排序选择专家
  - **Hash模式**：根据inputIds从tid2eid映射表中获取预计算的专家索引，跳过排序步骤直接输出
- 计算公式：

  **TopK模式：**

  **Step 1: 归一化**

  根据normType对输入x做归一化：

$$
normOut =
\begin{cases}
    \text{SoftMax}(x),      & normType = 0 \\
    \text{Sigmoid}(x),      & normType = 1 \\
    \sqrt{\text{Softplus}(x)},     & normType = 2
\end{cases}
$$

  **Step 2: 加偏置**

  若bias不为空，加偏置得到用于选择的值：

  $$
  normValue = normOut + bias
  $$

  否则 $normValue = normOut$。

  **Step 3: 分组筛选**（仅groupCount > 1 时执行）

  将normValue按groupCount分组，根据groupSelectMode计算每组得分：

  $$
  groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
  $$

  $$
  groupScore = \begin{cases}
      ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
      ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
  \end{cases}
  $$

  选取得分最高的kGroup个组，将未选中组的对应位置置为 $-\infty$：

  $$
  groupIdx = TopK(groupScore,\ k=kGroup).indices
  $$

  $$
  normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
  $$

  **Step 4: Top-K专家选择**

  对normValue取Top-K得到专家索引，这里只需要expertIdxOut：

  $$
  y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
  $$

  **Step 5: Renorm与缩放**

  根据expertIdxOut从normOut中取出对应的k个专家得分：

  $$
  gathered = normOut[\text{expertIdxOut}]
  $$

  normType=1 or normType=2 时做归一化；normType=0 时，renorm参数生效，renorm=1 时做renorm：

  $$
  if\ (normType = 1\ or\ normType = 2)\ or\ (normType = 0\ and\ renorm = 1):
  $$

  $$
  \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
  $$

  否则 $yOut = gathered$

  最终输出：

  $$
  yOut = yOut \times routedScalingFactor
  $$

  **Step 6: 可选输出**

  若outFlag为True，第三个输出为normOut；否则为空。

  **Hash模式：**

  当提供inputIds和tid2eid时，启用Hash模式：

  **Step 1: 归一化**

  根据normType对输入x做归一化（与TopK模式相同）：

$$
normOut =
\begin{cases}
    SoftMax(x),      & normType = 0 \\
    Sigmoid(x),      & normType = 1 \\
    \sqrt{Softplus(x)},     & normType = 2
\end{cases}
$$

  **Step 2: Hash索引查找**

  根据inputIds从tid2eid映射表获取专家索引：

  $$
  expertIdxOut = tid2eid[inputIds, :]
  $$

  其中tid2eid的shape为[numKeys, k]，inputIds的shape为[batch]，每个inputIds值对应一行k个专家索引。

  **Step 3: Gather与缩放**

  根据expertIdxOut从normOut中取出对应的k个专家得分：

  $$
  gathered = normOut[expertIdxOut]
  $$

  normType=1 or normType=2 时做归一化；normType=0 时，renorm参数生效，renorm=1 时做renorm：

  $$
  if\ (normType = 1\ or\ normType = 2)\ or\ (normType = 0\ and\ renorm = 1):
  $$

  $$
  \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
  $$

  否则 $yOut = gathered$

  最终输出：

  $$
  yOut = yOut \times routedScalingFactor
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出/属性</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>待计算输入，对应公式中的`x`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>与输入x进行计算的bias值，对应公式中的`bias`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inputIdsOptional</td>
        <td>输入</td>
        <td>Hash模式的输入索引，用于从tid2eid中查找专家索引。对应公式中的`inputIds`。shape为[batch]，与x的第一维相等。仅在Hash模式时需要。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>tid2eidOptional</td>
        <td>输入</td>
        <td>Hash映射表，存储预计算的专家索引。对应公式中的`tid2eid`。shape为[numKeys, k]，其中numKeys为映射表行数，k与参数k相等。仅在Hash模式时需要。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>k</td>
        <td>输入</td>
        <td>topk的k值，对应公式中的`k`。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kGroup</td>
        <td>输入</td>
        <td>分组排序后取的group个数，对应公式中的`kGroup`。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupCount</td>
        <td>输入</td>
        <td>分组的总个数，对应公式中的`groupCount`。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>routedScalingFactor</td>
        <td>输入</td>
        <td>计算yOut使用的routedScalingFactor系数，对应公式中的`routedScalingFactor`。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>输入</td>
        <td>用于计算yOut使用的eps系数，对应公式中的`eps`。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>输出</td>
        <td>对x做norm、分组排序topk后计算的结果，对应公式中的`yOut`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>输出</td>
        <td>对x做norm、分组排序topk后的索引，对应公式中的`expertIdxOut`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>输出</td>
        <td>norm计算的输出结果，对应公式中的`normOut`。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupSelectMode</td>
        <td>输入</td>
        <td>分组排序方式。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>renorm</td>
        <td>输入</td>
        <td>renorm标记。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>输入</td>
        <td>norm函数类型。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outFlag</td>
        <td>输入</td>
        <td>表示是否输出norm操作结果。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

## 约束说明

  * 输入shape限制：
      * x最后一维（即专家数）要求不大于2048。
  * 输入值域限制：
      * 要求1 <= k <= x_shape[-1] / groupCount * kGroup。
      * 要求1 <= kGroup <= groupCount，并且kGroup * x_shape[-1] / groupCount的值要大于等于k。
      * 要求groupCount > 0，x_shape[-1]能够被groupCount整除且整除后的结果大于groupSelectMode，并且整除的结果按照32个数对齐后乘groupCount的结果不大于2048。
      * renorm支持0和1。normType=1或者normType=2时做归一化；normType=0 时，renorm参数生效，renorm=1 时做renorm。
  * 其他限制：
      * groupSelectMode取值0和1，0表示使用最大值对group进行排序，1表示使用topk2的sum值对group进行排序。
      * normType取值0、1和2，0表示使用Softmax函数，1表示使用Sigmoid函数，2表示使用SqrtSoftplus函数。
      * normType取值为1或2时，renorm参数无效；normType取值为0时，renorm参数生效，renorm取值为0和1，0表示不做renorm，1表示做renorm。
      * outFlag取值true和false，true表示输出，false表示不输出。
  * **Hash模式限制**：
      * Hash模式需同时提供inputIdsOptional和tid2eidOptional，否则为TopK模式。
      * Hash模式仅支持简化路径（kGroup == groupCount或groupCount == expertCount）。
      * Hash模式下k要求不大于64。
      * tid2eid的shape必须为[numKeys, k]，其中numKeys为映射表总行数，k与参数k相等。
      * inputIds的shape必须为[batch]，与x的第一维相等。
      * inputIds中的每个值应在[0, numKeys-1]范围内。
      * inputIdsOptional和tid2eidOptional的数据类型支持INT32和INT64的组合。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_gating_top_k](examples/test_aclnn_moe_gating_top_k.cpp) | 通过[aclnnMoeGatingTopK](docs/aclnnMoeGatingTopK.md)接口方式调用MoeGatingTopK算子。 |
| aclnn接口  | [test_aclnn_moe_gating_top_k_v2](examples/test_aclnn_moe_gating_top_k_v2.cpp) | 通过[aclnnMoeGatingTopKV2](docs/aclnnMoeGatingTopKV2.md)接口方式调用MoeGatingTopKV2算子（支持Hash模式）。 |
| 图模式 | [test_geir_moe_gating_top_k](examples/test_geir_moe_gating_top_k.cpp)  | 通过[算子IR](op_graph/moe_gating_top_k_proto.h)构图方式调用MoeGatingTopK算子。         |
