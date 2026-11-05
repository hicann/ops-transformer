# IncreFlashAttention

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Description

- Operator function: For auto-regressive language models, the inference input length keeps increasing as new words are generated. On the basis of the original full inference, **incremental inference is implemented**: the S axis of query is fixed to 1, and key and value are stored in the kvCache, where the state information inferred previously is accumulated. The actual length of the S axis may differ across batches, and the input data is fixed-length data after padding. The following features are supported: **quantization, position encoding, page attention, kvCache dequantization, and KV left padding features.**

  Compared with the FlashAttention operator in the full scenario ([PromptFlashAttention](../prompt_flash_attention/README_en.md)), the incremental inference process is not completely equivalent to normal full inference, but the precision of incremental inference is not significantly degraded.

  **Note:**

    kvCache is a common technique for optimizing the inference performance of large models. During sampling, the Transformer model performs inference with the given prompt/context as the initial input (which can be processed in parallel), and then generates additional tokens one by one to continuously refine the generated sequence (which reflects the auto-regressive nature of the model). During sampling, the Transformer performs self-attention operations. To do so, it needs to extract key-value (KV) vectors for each item in the current sequence (whether a prompt/context or a generated token). These vectors are stored in a matrix, which is commonly referred to as the KV cache (kvCache).

- Calculation formula:

  Self-attention builds an attention model by using the relationships within the input sample itself. The principle is as follows: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Passing such a sequence through 3 weight matrices yields 3 matrices of dimension $n*d$.

    The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input sample. They are obtained by spatial transformation of the input sample and can be unified into one feature space.

  $$
     Attention(Q,K,V)=Score(Q,K)V
  $$

    In this operator, the Score function uses the Softmax function. The calculation formula of self-attention is:

  $$
   Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

    The product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, and softmax normalization is performed on each row. After being multiplied by $V$, an $n*d$ matrix is obtained.

## Parameter Description

<div style="overflow-x: auto;">
<table style="undefined;table-layout: fixed; width: 930px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 250px">
<col style="width: 230px">
<col style="width: 150px">
</colgroup>
<thead>
  <tr>
    <th style="text-align: center;">Parameter Name</th>
    <th style="text-align: center;">Input/Output/Attribute</th>
      <th style="text-align: center;">Description</th>
    <th style="text-align: center;">Data Type</th>
      <th style="text-align: center;">Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td >Input</td>
    <td >Input Q in the formula.</td>
    <td >FLOAT, FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td >Input</td>
    <td >Input K in the formula.</td>
    <td >FLOAT, INT8, FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td >Input</td>
    <td >Input V in the formula.</td>
    <td >FLOAT, INT8, FLOAT16</td>
    <td >ND</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td >Attribute</td>
    <td >The reciprocal of the square root of d in the formula.</td>
    <td >DOUBLE</td>
    <td >-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td >Output</td>
    <td >The output in the formula.</td>
    <td >FLOAT, INT8, FLOAT16</td>
    <td >ND</td>
  </tr>
</tbody>
</table></div>

## Constraints

- The shapes of query and attentionOut must be exactly the same, and the shapes of the corresponding tensors in key and value must be exactly the same.
- In the non-contiguous scenario, the number of tensors in the tensorlist of key and value equals the B of query (due to the tensorlist limitation, B must be less than or equal to 256 in the non-contiguous scenario). The shapes must be exactly the same except for S, and the batch can only be 1.
- The N in query equals numHeads, the N in key and value equals numKeyValueHeads, and numHeads is a multiple of numKeyValueHeads.
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:
  - The B axis is less than or equal to 65536, the N axis is less than or equal to 256, and the D axis is less than or equal to 512.
- Atlas inference products:
  - The B axis is less than or equal to 256, the N axis is less than or equal to 256, the D axis is less than or equal to 512, and the S axes of key and value are less than or equal to 65536.
  - The data types of query, key, value, and attentionOut support only FLOAT16.
  - When the data layout format is BNSD, the ratio of numHeads to numKeyValueHeads must not be greater than 8. In other cases, only the value 0 is supported.
- Comprehensive limitations on the number of INT8 quantization-related input parameters and the input/output data formats:
  - In the scenario where the inputs query, key, and value are FLOAT16 and the output is INT8: the input parameter quantScale2 is required, quantOffset2 is optional, and the parameters dequantScale1, quantScale1, and dequantScale2 must not be passed in (that is, they must be nullptr).

- The data type of pseShift must be the same as the data type of query.

- Parameter constraints of antiquantScale and antiquantOffset:
    - per-channel mode: the shapes of the two parameters support (2, N, 1, D), (2, N, D), and (2, H), where N is numKeyValueHeads. The parameter data type is the same as the data type of query.
    - per-tensor mode: the shapes of both parameters are (2), and the data type is the same as the data type of query.

- The input parameters quantScale2 and quantOffset2 support two formats (per-tensor/per-channel) and two data types (FLOAT32/BFLOAT16). If quantOffset2 is passed in, its type and shape information must be the same as those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must equal H; for other layouts, the product must equal N*D. (It is recommended that when the output layout is BSH, the shape [1,1,H] or [H] is passed in for quantScale2; when the output is BNSD, it is recommended to pass in [1,N,1,D] or [N,D]; when the output is BSND, it is recommended to pass in [1,1,N,D] or [N,D].)

- page attention scenario:

  - The prerequisite for enabling page attention is that blockTable exists and is valid, and key and value are arranged in a contiguous memory area according to the indices in blockTable. The dtypes of key and value support FLOAT16/BFLOAT16/INT8. In this scenario, the inputLayout parameters of key and value are invalid.
  - blockSize is a user-defined parameter whose value affects the performance of page attention. In the scenario where page attention is enabled, a non-0 value must be passed in for blockSize, and blocksize must not exceed 512. When the input types of key and value are FLOAT16/BFLOAT16, the value must be aligned to 16; when the input types of key and value are INT8, the value must be aligned to 32. A value of 128 is recommended. In general, page attention can improve throughput but causes performance degradation.
  - In the page attention scenario, when the inputLayout of query is BNSD, the kvCache layout supports two formats: (blocknum, blocksize, H) and (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the kvCache layout supports only the format (blocknum, blocksize, H). blocknum cannot be less than the sum of the number of blocks of each batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be the same.
  - In the page attention scenario, the performance with the kvCache layout (blocknum, KV_N, blocksize, D) is usually better than that with the kvCache layout (blocknum, blocksize, H). It is recommended to preferentially select the format (blocknum, KV_N, blocksize, D).
  - In the scenario where page attention is enabled, when the input kvCache layout format is (blocknum, blocksize, H) and numKvHeads * headDim exceeds 64k, the request is intercepted and an error is reported due to hardware instruction constraints. This problem can be solved by enabling GQA (reducing numKvHeads) or changing the kvCache layout format to (blocknum, numKvHeads, blocksize, D).
  - In the page attention scenario, the input actualSeqLengths must be passed in.
  - In the page attention scenario, blockTable must be two-dimensional. The length of the first dimension must equal B, and the length of the second dimension cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
  - In the scenario where page attention is enabled, the input S must be greater than or equal to maxBlockNumPerSeq * blockSize in the following scenarios:
  - Attention Mask is enabled, for example, the mask shape is (B, 1, 1, S).
  - pseShift is enabled, for example, the pseShift shape is (B, N, 1, S).
- kv left padding scenario:
    - The calculation formula of the start of the kvCache transfer is: Smax - kvPaddingSize - actualSeqLengths; the calculation formula of the end of the kvCache transfer is: Smax - kvPaddingSize. When the start or end of the kvCache transfer is less than 0, the returned data result is all 0.
    - kvPaddingSize is set to 0 when it is less than 0.
    - This feature must be enabled together with the actualSeqLengths parameter; otherwise, the kv right padding scenario is used by default.
    - When this feature is enabled together with the Attention Mask parameter, ensure that the meaning of Attention Mask is correct, that is, invalid data can be correctly masked. Otherwise, precision problems are introduced.

## Calling Instructions

| Calling Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_IncreFlashAttentionV4](./examples/test_aclnn_incre_flash_attention_v4.cpp) | Calls the IncreFlashAttentionV4 operator by using [aclnnIncreFlashAttentionV4](docs/aclnnIncreFlashAttentionV4_en.md). |
