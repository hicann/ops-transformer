# NsaSelectedAttentionInfer

English | [简体中文](./README.md)

# Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |
|<term>Kirin X90 processor products</term> | √ |
|<term>Kirin 9030 processor products</term> | √ |

# Function Description

- Operator function: Computes Selected Attention during Native Sparse Attention inference.
- Calculation formula:

  self-attention builds an attention model by using the relationships within the input sample itself. The principle is to assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be viewed as a token embedding. The sequence is transformed through 3 weight matrices to obtain 3 matrices of the dimension $n*d$.

  The computation of Selected Attention fuses data fetching by topk indices with the attention computation, plus fetching kvCache for paged attention. First, $key_{topk}$ is fetched from $key$ and $value_{topk}$ is fetched from $value$ using the $topkIndices$ indices. The self_attention computation formula is as follows:

  $$
  Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
  $$

  The product of $query$ and $key_{topk}^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, then normalized row by row with softmax, and multiplied by $value_{topk}$ to obtain a matrix of the dimension $n*d$.

# Parameter Description

<div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 902px; border-collapse: collapse; border: 1px solid #ccc;">
    <colgroup>
      <col style="width: 170px">  <!-- Parameter Name -->
      <col style="width: 120px">  <!-- Input/Output -->
      <col style="width: 300px">  <!-- Description -->
      <col style="width: 212px">  <!-- Data Type -->
      <col style="width: 100px">  <!-- Data Format -->
    </colgroup>
    <thead>
      <tr>
        <th style="font-weight: bold;">Parameter Name</th>
        <th style="font-weight: bold;">Input/Output/Attribute</th>
        <th style="font-weight: bold;">Description</th>
        <th style="font-weight: bold;">Data Type</th>
        <th style="font-weight: bold;">Data Format</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="white-space: nowrap;">query</td>
        <td>Input</td>
        <td>Input query in the formula.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">key</td>
        <td>Input</td>
        <td>Input key in the formula.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">value</td>
        <td>Input</td>
        <td>Input value in the formula.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">topkIndices</td>
        <td>Input</td>
        <td>topK indices in the formula.</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td style="white-space: nowrap;">output</td>
        <td>Output</td>
        <td>Output of attention in the formula.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>
</div>

- <term>Kirin X90 processor products</term> and <term>Kirin 9030 processor products</term>: BFLOAT16 is not supported.

# Constraints

- The N of query equals numHeads, the N of key and value equals numKeyValueHeads, and numHeads is a multiple of numKeyValueHeads.
- The D of query equals the D (H/numKeyValueHeads) of key, and the D (H/numKeyValueHeads) of value equals the D of output.
- The usage restrictions on the query, key, and value inputs are as follows:
  - The B axis supports values less than or equal to 3072.
  - The N axis of key/value supports values less than or equal to 256.
  - The lengths of actualQSeqLenOptional and actualKvSeqLenOptional must equal dimension 0 (B) of query (when the layout is BSH/BSND); passing in only 1 element is invalid. When the layout is TND, the lengths of the two must be consistent.
  - The ratio of the N axis of query to the N axis of key/value (H/D), that is, the group size in GQA, supports values less than or equal to 16.
  - The D axis of query and Key supports the value 192.
  - The D axis of value supports the value 128.
  - The blockSize of Key and Value supports 64 or 128.
  - In common scenarios, only an S axis of query equal to 1 is supported.
  - In the multi-token inference scenario, only an S axis of query with a maximum value of 4 is supported. In this case, actualQSeqLen <= actualSelKvSeqLen is required for each individual batch.
  - Only paged attention is supported.
  - Only a selectBlockSize that is a multiple of 16 is supported, with a maximum of 128.
  - The upper limit of selectBlockCount satisfies selectBlockCount * selectBlockSize <= MaxKvSeqlen, where MaxKvSeqlen = Max(actualSelKvSeqLenOptional).

# Calling Description

| Calling Method  | Sample Code                                                                | Description                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn API | [test_aclnn_nsa_selected_attention_infer](./examples/test_aclnn_nsa_selected_attention_infer.cpp) | Calls the NsaCompressAttentionInfer operator through the [`aclnnNsaSelectedAttentionInfer`](docs/aclnnNsaSelectedAttentionInfer_en.md) API. |
