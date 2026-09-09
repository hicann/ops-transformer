# MoeFinalizeRoutingV2Grad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                      |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：aclnnMoeFinalizeRoutingV2的反向传播。
- **计算公式**：
  R: batch * sequence

  H: hidden

  K: topk

  gradY: (R, H)

  expandedRowIdx: (R * K)

  expandedXOptional: (R * K, H) or (activeNum, H) or (expertNum, expertCapacity, H)

  scalesOptional: (R, K)

  expertIdxOptional: (R, K)

  biasOptional: (E, H)

  i : 0 ~ R * K - 1

  j : 0 ~ H

  (1) scalesOptional为空指针：

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
  $$

  (2) scalesOptional不为空指针， biasOptional为空指针：

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
  $$

  $$
  gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] * gradY[i / K][j])
  $$

  (3) scalesOptional不为空指针， biasOptional不为空指针：

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i]
  $$

  $$
  gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) * gradY[i / K][j])
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
        <td>gradY</td>
        <td>输入</td>
        <td>公式中的`gradY`，表示MoeFinalizeRoutingV2正向输出y的导数。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>输入</td>
        <td>公式中的`expandedRowIdx`，表示token按照专家序排序的索引。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedXOptional</td>
        <td>输入</td>
        <td>公式中的`expandedXOptional`，表示根据expertIdx进行扩展过的特征。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>公式中的`scalesOptional`，表示对特征进行的缩放。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOptional</td>
        <td>输入</td>
        <td>公式中的`expertIdxOptional`，表示每一个特征对应的处理专家索引。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>公式中的`biasOptional`，表示对特征进行的偏移。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>dropPadMode</td>
        <td>属性</td>
        <td>表示使用不同的场景。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>activeNum</td>
        <td>属性</td>
        <td>公式中的`activeNum`，表示gradExpandedXOut最大输出行数。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertNum</td>
        <td>属性</td>
        <td>公式中的`expertNum`，表示专家数。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>expertCapacity</td>
        <td>属性</td>
        <td>公式中的`expertCapacity`。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradExpandedXOut</td>
        <td>输出</td>
        <td>公式中的`gradExpandedXOut`，表示MoeFinalizeRoutingV2正向输入expandedX的导数。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradScalesOut</td>
        <td>输出</td>
        <td>公式中的`gradScalesOut`，表示MoeFinalizeRoutingV2正向输入scales的导数。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_finalize_routing_v2_grad.cpp](examples/test_aclnn_moe_finalize_routing_v2_grad.cpp) | 通过[aclnnMoeFinalizeRoutingV2Grad](docs/aclnnMoeFinalizeRoutingV2Grad.md)接口方式调用MoeFinalizeRoutingV2Grad算子。 |
| PyTorch API | - | 通过[moe_finalize_routing_grad](../../moe/moe_finalize_routing_v2_grad/docs/torchapi_moe_finalize_routing_grad.md)接口调用moe_finalize_routing_grad算子。 |
