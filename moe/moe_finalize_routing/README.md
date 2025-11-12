# MoeFinalizeRouting

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

## 功能说明

-   **算子功能**：MoE计算中，最后处理合并MoE FFN的输出结果。
-   **计算公式**：

    $$
    expertid=expandedExpertIdx[i,k]
    $$
    
    $$
    out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num\_rows},j}+bias_{expertid,j}))
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
      <td>expandedX</td>
      <td>输入</td>
      <td>公式中的`expandedX` ，MoE的FFN输出，shape支持[NUM_ROWS*K, H]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的`x1`，shape要求与out一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2Optional</td>
      <td>输入</td>
      <td>公式中的`x2Optional`，shape要求与out一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的`bias`，shape支持[E, H]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>公式中的`scales`，shape支持[NUM_ROWS, K]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>输入</td>
      <td>公式中的`expandedRowIdx`，shape支持[NUM_ROWS, K]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedExpertIdx</td>
      <td>输入</td>
      <td>公式中的`expandedExpertIdx`，shape支持[NUM_ROWS, K]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的`out`，shape支持[NUM_ROWS, h]。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_finalize_routing.cpp](examples/test_aclnn_moe_finalize_routing.cpp) | 通过[aclnnMoeFinalizeRouting](docs/aclnnMoeFinalizeRouting.md)接口方式调用MoeFinalizeRouting算子。 |