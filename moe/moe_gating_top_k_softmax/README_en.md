# MoeGatingTopKSoftmax

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: in MoE computation, this operator performs Softmax computation on the output of x and takes the topk operation. yOut is the topk result of softmax; expertIdxOut is the indices result of topk, that is, the corresponding expert numbers; rowIdxOut is the column-wise value result with the same shape as expertIdxOut. If the corresponding row has finishedOptional set to true, the expert number is directly filled with the num\_expert value (that is, the size of the last axis of x).
- Computation formula:

  $$
  softmaxOut=softmax(x,axis=-1)
  $$

  $$
  yOut,expertIdxOut=topK(softmaxOut,k=k)
  $$

  $$
  R=rank(expertIdxOut), M=\prod_{i=0}^{R-2}expertIdxOut.shape[i], K=expertIdxOut.shape[R-1]
  $$

  $$
  rowIdxRange=arange(M*K)
  rowIdxOut=rowIdxRange.reshape([K,M]).transpose(1,0).reshape(expertIdxOut.shape)
  $$

## Parameter Description

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">Parameter Name</th>
        <th style="white-space: nowrap">Input/Output/Attribute</th>
        <th style="white-space: nowrap">Description</th>
        <th style="white-space: nowrap">Data Type</th>
        <th style="white-space: nowrap">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>`x` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>finishedOptional</td>
        <td>Input</td>
        <td>Specifies that if the corresponding row has finishedOptional set to true, the expert number is directly filled with the num_expert value (that is, the size of the last axis of x).</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Attribute</td>
        <td>`k` in the formula, indicating the k value of topk.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>`yOut` in the formula, indicating the topk result of softmax.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>Output</td>
        <td>`expertIdxOut` in the formula, indicating the indices result of topk, that is, the corresponding expert numbers.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>rowIdxOut</td>
        <td>Output</td>
        <td>`rowIdxOut` in the formula, indicating the original row position corresponding to each position.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- 1 <= k <= the size of the last dimension of x, and k is not greater than 1024.
- The size of each dimension of x and finishedOptional must not be greater than the maximum value of int32, 2147483647.

## Calling Instructions

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_moe_gating_top_k_softmax.cpp](examples/test_aclnn_moe_gating_top_k_softmax.cpp) | Calls the MoeGatingTopKSoftmax operator through the [aclnnMoeGatingTopKSoftmax](docs/aclnnMoeGatingTopKSoftmax_en.md) API. |
