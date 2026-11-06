# MoeGatingTopK

English | [简体中文](./README.md)

## Product Support

|Product                                |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>                  |    √     |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

normType=3 is supported only on <term>Ascend 950PR&950DT products</term>, and renorm controls whether to normalize the scores of the selected experts. The normType of <term>Atlas A2 products</term> and <term>Atlas A3 products</term> supports only 0, 1, and 2, and does not support 3. Mode 3 in the following formulas applies only to Ascend 950PR&950DT products.

## Description

- Operator function: In MoE computation, this operator applies the Sigmoid, SoftMax, or SqrtSoftplus computation to the input x, sorts the computation results by group, and finally selects the top k experts based on the grouped sorting results. Two modes are supported:
  - **TopK mode**: Select experts by performing TopK sorting on normValue
  - **Hash mode**: Obtain pre-computed expert indexes from the tid2eid mapping table based on inputIds and output them directly, skipping the sorting step
- Formula:

  **TopK mode:**

  **Step 1: Normalization**

  Normalize the input x according to normType:

$$
normOut =
\begin{cases}
    \text{SoftMax}(x),      & normType = 0 \\
    \text{Sigmoid}(x),      & normType = 1\ or\ 3 \\
    \sqrt{\text{Softplus}(x)},     & normType = 2
\end{cases}
$$

  **Step 2: Adding Bias**

  If bias is not empty, add the bias to obtain the value used for selection:

  $$
  normValue = normOut + bias
  $$

  Otherwise, $normValue = normOut$.

  **Step 3: Group Filtering** (performed only when groupCount > 1)

  Group normValue by groupCount and calculate the score of each group according to groupSelectMode:

  $$
  groupedValue = Reshape(normValue,\ [batch,\ groupCount,\ -1])
  $$

  $$
  groupScore = \begin{cases}
      ReduceMax(groupedValue,\ dim=-1), & groupSelectMode = 0 \\
      ReduceSum(TopK(groupedValue,\ k=2,\ dim=-1),\ dim=-1), & groupSelectMode = 1
  \end{cases}
  $$

  Select the kGroup groups with the highest scores and set the corresponding positions of unselected groups to $-\infty$:

  $$
  groupIdx = TopK(groupScore,\ k=kGroup).indices
  $$

  $$
  normValue = Mask(groupedValue,\ groupIdx,\ fillValue=-\infty)
  $$

  **Step 4: Top-K Expert Selection**

  Apply Top-K to normValue to obtain expert indexes. Only expertIdxOut is needed here:

  $$
  y, expertIdxOut = TopK(normValue[groupIdx, :],\ k=k)
  $$

  **Step 5: Renorm and Scaling**

  Obtain the corresponding k expert scores from normOut based on expertIdxOut:

  $$
  gathered = normOut[\text{expertIdxOut}]
  $$

  When normType=1 or 2, normalization is always performed; when normType=0 or 3, whether to normalize is controlled by renorm:

  $$
  if\ (normType = 1\ or\ normType = 2)\ or\ ((normType = 0\ or\ normType = 3)\ and\ renorm = 1):
  $$

  $$
  \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
  $$

  Otherwise, $yOut = gathered$

  Final output:

  $$
  yOut = yOut \times routedScalingFactor
  $$

  **Step 6: Optional Output**

  If outFlag is True, the third output is normOut; otherwise, it is empty.

  **Hash mode:**

  When inputIds and tid2eid are provided, Hash mode is enabled:

  **Step 1: Normalization**

  Normalize the input x according to normType (the same as in TopK mode):

$$
normOut =
\begin{cases}
    SoftMax(x),      & normType = 0 \\
    Sigmoid(x),      & normType = 1\ or\ 3 \\
    \sqrt{Softplus(x)},     & normType = 2
\end{cases}
$$

  **Step 2: Hash Index Lookup**

  Obtain expert indexes from the tid2eid mapping table based on inputIds:

  $$
  expertIdxOut = tid2eid[inputIds, :]
  $$

  where the shape of tid2eid is [numKeys, k] and the shape of inputIds is [batch]. Each inputIds value corresponds to a row of k expert indexes.

  **Step 3: Gather and Scaling**

  Obtain the corresponding k expert scores from normOut based on expertIdxOut:

  $$
  gathered = normOut[expertIdxOut]
  $$

  When normType=1 or 2, normalization is always performed; when normType=0 or 3, whether to normalize is controlled by renorm:

  $$
  if\ (normType = 1\ or\ normType = 2)\ or\ ((normType = 0\ or\ normType = 3)\ and\ renorm = 1):
  $$

  $$
  \quad yOut = \frac{gathered}{ReduceSum(gathered,\ dim=-1) + eps}
  $$

  Otherwise, $yOut = gathered$

  Final output:

  $$
  yOut = yOut \times routedScalingFactor
  $$

## Parameter Description

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
    <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter Name</th>
        <th>Input/Output/Attribute</th>
        <th>Description</th>
        <th>Data Type</th>
        <th>Data Format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input to be computed, corresponding to `x` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Bias value computed with the input x, corresponding to `bias` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>inputIdsOptional</td>
        <td>Input</td>
        <td>Input indexes in Hash mode, used to search for expert indexes in tid2eid. Corresponds to `inputIds` in the formula. The shape is [batch], which is equal to the first dimension of x. Required only in Hash mode.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>tid2eidOptional</td>
        <td>Input</td>
        <td>Hash mapping table that stores pre-computed expert indexes. Corresponds to `tid2eid` in the formula. The shape is [numKeys, k], where numKeys is the number of rows in the mapping table and k is equal to the parameter k. Required only in Hash mode.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Input</td>
        <td>k value of topk, corresponding to `k` in the formula.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kGroup</td>
        <td>Input</td>
        <td>Number of groups selected after grouped sorting, corresponding to `kGroup` in the formula.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupCount</td>
        <td>Input</td>
        <td>Total number of groups, corresponding to `groupCount` in the formula.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>routedScalingFactor</td>
        <td>Input</td>
        <td>routedScalingFactor coefficient used to compute yOut, corresponding to `routedScalingFactor` in the formula.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>eps coefficient used to compute yOut, corresponding to `eps` in the formula.</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>Result computed after norm, grouped sorting, and topk on x, corresponding to `yOut` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>Output</td>
        <td>Indexes obtained after norm, grouped sorting, and topk on x, corresponding to `expertIdxOut` in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>Output</td>
        <td>Output result of the norm computation, corresponding to `normOut` in the formula.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupSelectMode</td>
        <td>Input</td>
        <td>Grouped sorting method.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>renorm</td>
        <td>Input</td>
        <td>renorm flag.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normType</td>
        <td>Input</td>
        <td>Type of the norm function.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outFlag</td>
        <td>Input</td>
        <td>Indicates whether to output the result of the norm operation.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

## Constraints

  * Input shape restrictions:
      * The last dimension of x (that is, the number of experts) must not be greater than 2048.
  * Input value range restrictions:
      * 1 <= k <= x_shape[-1] / groupCount * kGroup is required.
      * 1 <= kGroup <= groupCount is required, and the value of kGroup * x_shape[-1] / groupCount must be greater than or equal to k.
      * groupCount > 0 is required, x_shape[-1] must be divisible by groupCount and the quotient must be greater than groupSelectMode, and the result of aligning the quotient to 32 and multiplying it by groupCount must not be greater than 2048.
      * renorm supports 0 and 1. When normType=0 or normType=3, renorm=0 indicates that renorm is not performed, and renorm=1 indicates that renorm is performed; when normType=1 or normType=2, the renorm parameter is invalid, and renorm is always performed.
  * Other restrictions:
      * groupSelectMode can be 0 or 1. 0 indicates that groups are sorted by the maximum value, and 1 indicates that groups are sorted by the sum of the topk2 values.
      * When normType is 0, 1, or 2: 0 indicates the Softmax function, 1 indicates the Sigmoid function with renorm, and 2 indicates the SqrtSoftplus function with renorm.
      * Only Ascend 950PR&950DT products additionally support normType=3, which indicates the Sigmoid function with normalization controlled by the renorm parameter.
      * On Atlas A2/A3 products, normType supports only 0, 1, and 2; when the value is 2, groupCount must be 1 (Ascend 950PR&950DT products do not have this groupCount restriction).
      * When normType is 0 or 3, the renorm parameter takes effect. 0 indicates that renorm is not performed, and 1 indicates that renorm is performed; when normType is 1 or 2, the renorm parameter is invalid, and renorm is always performed.
      * outFlag can be true or false. true indicates output, and false indicates no output.
  * **Hash mode restrictions**:
      * Hash mode requires inputIdsOptional and tid2eidOptional to be provided at the same time; otherwise, TopK mode is used.
      * Hash mode supports only the simplified path (kGroup == groupCount or groupCount == expertCount).
      * In Hash mode, k must not be greater than 64.
      * The shape of tid2eid must be [numKeys, k], where numKeys is the total number of rows in the mapping table and k is equal to the parameter k.
      * The shape of inputIds must be [batch], which is equal to the first dimension of x.
      * Each value in inputIds must be in the range [0, numKeys-1].
      * The data types of inputIdsOptional and tid2eidOptional support the combination of INT32 and INT64.

## Calling Instructions

| Calling Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_moe_gating_top_k](examples/test_aclnn_moe_gating_top_k.cpp) | Call the MoeGatingTopK operator through the [aclnnMoeGatingTopK](docs/aclnnMoeGatingTopK_en.md) API. |
| aclnn API  | [test_aclnn_moe_gating_top_k_v2](examples/test_aclnn_moe_gating_top_k_v2.cpp) | Call the MoeGatingTopKV2 operator through the [aclnnMoeGatingTopKV2](docs/aclnnMoeGatingTopKV2_en.md) API (Hash mode supported). |
| Graph mode | [test_geir_moe_gating_top_k](examples/test_geir_moe_gating_top_k.cpp)  | Call the MoeGatingTopK operator by building a graph through [operator IR](op_graph/moe_gating_top_k_proto.h).         |
