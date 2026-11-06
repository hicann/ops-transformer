# MoeGatingTopKSoftmaxV2

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- Operator function: in MoE computation, if renorm=0, this operator first performs Softmax computation on the output of x and then takes the TopK operation; if renorm=1, it first takes the topk operation on the output of x and then performs the Softmax operation. yOut is the topk result of softmax; expertIdxOut is the indices result of topk, that is, the corresponding expert numbers; if finished of the corresponding row is true, the expert number is directly filled with the num\_expert value (that is, the size of the last axis of x).

- Computation formula:

1. renorm = 0,

  $$
  softmaxResultOutOptional=softmax(x,axis=-1)
  $$

  $$
  yOut,expertIdxOut=topK(softmaxResultOutOptional,k=k)
  $$

2. renorm = 1

  $$
  topkOut,expertIdxOut=topK(x, k=k)
  $$

  $$
  yOut = softmax(topkOut,axis=-1)
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
        <td>Optional input</td>
        <td>You can pass nullptr when the finished row mask is not used; otherwise, when finished is true for a row, the expert number of that row is filled with num_expert (that is, the size of the last dimension of x).</td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>k</td>
        <td>Attribute</td>
        <td>`k` in the formula, indicating the k value of topk.</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>renorm</td>
        <td>Attribute</td>
        <td>`renorm` in the formula, indicating the renorm flag, with the value 0 or 1.</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputSoftmaxResultFlag</td>
        <td>Attribute</td>
        <td>Specifies whether to output the softmax result.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut</td>
        <td>Output</td>
        <td>`yOut` in the formula, indicating the topk values taken after softmax is applied to x.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expertIdxOut</td>
        <td>Output</td>
        <td>`expertIdxOut` in the formula. When renorm=0, it indicates the indices of the topk values taken after softmax is applied to x; when renorm=1, it indicates the indices obtained by directly performing the topk operation on x, that is, the expert numbers.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>softmaxResultOutOptional</td>
        <td>Output</td>
        <td>`softmaxResultOutOptional` in the formula, indicating the Softmax result in the computation process.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## Constraints

- The value range of k is `1 <= k <= min(1024, the size of the last dimension of x)`.
- renorm supports only the values 0 and 1.
- The size of each dimension of x and finishedOptional must not be greater than the maximum value of int32, 2147483647.

## Calling Instructions

| Calling Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_moe_gating_top_k_softmax_v2.cpp](examples/test_aclnn_moe_gating_top_k_softmax_v2.cpp) | Calls the MoeGatingTopKSoftmaxV2 operator through the [aclnnMoeGatingTopKSoftmaxV2](docs/aclnnMoeGatingTopKSoftmaxV2_en.md) API. |
