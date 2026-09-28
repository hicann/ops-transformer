# MoeInitRoutingQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理。

  本接口针对[aclnnMoeInitRoutingQuant](../moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)做了如下功能变更，请根据实际情况选择合适的接口：

  - 新增drop模式，在该模式下输出内容会根据每个专家的expertCapacity处理，超过expertCapacity不做处理，不足的会补0。
  - 新增dropless模式下expertTokensCountOrCumsumOutOptional可选输出，drop场景下expertTokensBeforeCapacityOutOptional可选输出。
  - 删除rowIdx输入。
  - 增加动态quant计算模式。

- **计算公式**：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在dropless模式下，对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
    $$

  4.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacityOutOptional：

    $$
    expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  5.计算quant结果：
    - 静态quant：

        $$
        quantResult = round((x * scaleOptional) + offsetOptional)
        $$

    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

  6.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
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
      <td>x</td>
      <td>输入</td>
      <td>MOE的输入即token特征输入，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>输入</td>
      <td>aclnnMoeGatingTopKSoftmaxV2的输出每一行特征对应的K个处理专家，对应公式中的`expertIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>可选输入</td>
      <td>表示用于计算quant结果的参数，要求静态quant场景下必须输入，对应公式中的`scaleOptional`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>可选输入</td>
      <td>表示用于计算quant结果的偏移值，要求在静态quant场景下必须输入，对应公式中的`offsetOptional`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>属性</td>
      <td>表示是否为Active场景，该属性在dropPadMode为0时生效，值范围大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>属性</td>
      <td>表示每个专家能够处理的tokens数，值范围大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>属性</td>
      <td>表示专家数，值范围大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>属性</td>
      <td>表示是否为Drop/Pad场景，取值为0和1。0：表示非Drop/Pad场景，该场景下不校验expertCapacity。
1：表示Drop/Pad场景，需要校验expertNum和expertCapacity，对于每个专家处理的超过和不足expertCapacity的值会做相应的处理。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>属性</td>
      <td>0：表示不输出expertTokensCountOrCumsumOutOptional。1：表示输出的值为各个专家处理的token数量的累计值。2：表示输出的值为各个专家处理的token数量。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>属性</td>
      <td>取值为false和true。false：表示不输出expertTokensBeforeCapacityOutOptional。
true：表示输出的值为在drop之前各个专家处理的token数量。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>属性</td>
      <td>取值为0和1。0：表示静态quant场景；1：表示动态quant场景。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>输出</td>
      <td>根据expertIdx进行扩展过的特征，对应公式中的`expandedXOut`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>输出</td>
      <td>expandedX和x的映射关系，对应公式中的`expandedRowIdxOut`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOutOptional</td>
      <td>可选输出</td>
      <td>输出每个专家处理的token数量的统计结果及累加值，通过expertTokensCountOrCumsumFlag参数控制是否输出，该值仅在非Drop/Pad场景下输出，对应公式中的`expertTokensCountOrCumsumOutOptional`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOutOptional</td>
      <td>可选输出</td>
      <td>输出drop之前每个专家处理的token数量的统计结果，通过expertTokensBeforeCapacityFlag参数控制是否输出，该值仅在Drop/Pad场景下输出，对应公式中的`expertTokensBeforeCapacityOutOptional`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dynamicQuantScaleOutOptional</td>
      <td>可选输出</td>
      <td>输出动态quant计算过程中的中间值，该值仅在动态quant场景下输出，对应公式中的`dynamicQuantScaleOutOptional`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

  无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_init_routing_quant_v2](examples/test_aclnn_moe_init_routing_quant_v2.cpp) | 通过[aclnnMoeInitRoutingQuantV2](docs/aclnnMoeInitRoutingQuantV2.md)接口方式调用MoeInitRoutingQuantV2算子。 |
