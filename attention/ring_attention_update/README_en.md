# aclnnRingAttentionUpdate

English | [简体中文](./README.md)

## Supported Product Models

| Product                                                         | Support Status |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 products</term>     |    √     |
| <term>Ascend 950PR&950DT products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Description

- Operator function: Updates the outputs of two FlashAttention computations based on their different softmax max and sum values.

- Computation formula:

    $$
    softmax\_max = max(prev\_softmax\_max, cur\_softmax\_max)
    $$

    $$
    softmax\_sum = prev\_softmax\_sum * exp(prev\_softmax\_max - softmax\_max) + exp(cur\_softmax\_max - softmax\_max)
    $$

    $$
    attn\_out = prev\_attn\_out * exp(prev\_softmax\_max - softmax\_max) / softmax\_sum + cur\_attn\_out * exp(cur\_softmax\_max - softmax\_max) / softmax\_sum
    $$

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>prevAttnOut</td>
      <td>Input</td>
      <td>prev_attn_out in the formula. Output of the first FlashAttention. The input shape must be consistent with the inputLayoutOptional attribute. When the input data layout inputLayoutOptional is TND, D is restricted to a multiple of 64.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prevSoftmaxMax</td>
      <td>Input</td>
      <td>prev_softmax_max in the formula, the max result of the softmax of the first FlashAttention. The input shape is (B,N,S,8) or (T,N,8); the 8 numbers in the last dimension are identical and must be positive. Here, B is the batch size, N is the head number, S is the sequence length, and T is the time.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>prevSoftmaxSum</td>
      <td>Input</td>
      <td>prev_softmax_sum in the formula, the sum result of the softmax of the first FlashAttention. The input shape is consistent with prevSoftmaxMax; the 8 numbers in the last dimension are identical and must be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curAttnOut</td>
      <td>Input</td>
      <td>cur_attn_out in the formula, the output of the second FlashAttention. The data type and input shape are consistent with prevAttnOut. When the input data layout inputLayoutOptional is TND, D is restricted to a multiple of 64.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curSoftmaxMax</td>
      <td>Input</td>
      <td>cur_softmax_max in the formula, the max result of the softmax of the second FlashAttention. The input shape is consistent with prevSoftmaxMax; the 8 numbers in the last dimension are identical and must be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>curSoftmaxSum</td>
      <td>Input</td>
      <td>cur_softmax_sum in the formula, the sum result of the softmax of the second FlashAttention. The input shape is consistent with prevSoftmaxMax; the 8 numbers in the last dimension are identical and must be positive.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>Optional Input</td>
      <td>Cumulative sequence length starting from 0. The data type supports INT64. When the data layout inputLayoutOptional is TND, this parameter is required. It is an integer aclTensor that increments from 0 to T.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputLayoutOptional</td>
      <td>Attribute</td>
      <td>Data layout of the inputs related to attn_out. Currently, "TND" and "SBH" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attnOutOut</td>
      <td>Output</td>
      <td>attn_out in the formula, the output updated with the two results. The data type supports FLOAT16, FLOAT, and BFLOAT16. The data type and output shape are consistent with prevAttnOut.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxMaxOut</td>
      <td>Output</td>
      <td>softmax_max in the formula, the max of the softmax updated with the two results. The data type supports FLOAT, and the output shape is consistent with prevSoftmaxMax.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxSumOut</td>
      <td>Output</td>
      <td>softmax_sum in the formula, the sum of the softmax updated with the two results. The data type supports FLOAT, and the output shape is consistent with prevSoftmaxMax.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

  - When inputLayoutOptional is "TND", the last dimension of prevAttnOut must be a multiple of 64.
  - When inputLayoutOptional is "TND", actualSeqQlenOptional is required.
  - When inputLayoutOptional is "TND", pay attention to the size of N*D, which is restricted to: (N \* D) rounded up to a multiple of 64 \* (size of the attention input data type \* 6 + 8) + (N \* 8) rounded up to a multiple of 64 \* 56 <= 192 \* 1024. Data type sizes: FLOAT is 4, and FLOAT16 and BFLOAT16 are 2. If the size exceeds the limit, a corresponding interception message appears.

## Calling Description

| Calling Method | Calling Sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_ring_attention_update](./examples/test_aclnn_ring_attention_update.cpp) | Call the RingAttentionUpdate operator through the [aclnnRingAttentionUpdate](docs/aclnnRingAttentionUpdate_en.md) API.    |
