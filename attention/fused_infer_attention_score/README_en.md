# FusedInferAttentionScore

English | [简体中文](./README.md)

## Product Support Status

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |
|<term>Kirin X90 processor products</term>|      √     |
|<term>Kirin 9030 processor products</term>|      √     |

## Function Description

- Operator function: The FlashAttention operator adapts to both incremental and full inference scenarios. It supports the full computation scenario ([PromptFlashAttention](../prompt_flash_attention/README_en.md)) and the incremental computation scenario ([IncreFlashAttention](../incre_flash_attention/README_en.md)).

- Calculation formula:

    self-attention builds an attention model by using the relationships within the input samples themselves. The principle is as follows: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. The sequence is transformed by 3 weight matrices to obtain 3 matrices of dimensions $n*d$.

    The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input samples, obtained from the input samples through spatial transformation, and can be unified into one feature space. "Attention" in the formula and the operator name is short for "self-attention".

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the Score function uses the Softmax function. The calculation formula of self-attention is as follows:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    The product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, normalized by softmax for each row, and multiplied by $V$ to obtain an $n*d$ matrix.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter Name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>query</td>
    <td>Input</td>
    <td>Input Q in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>Input</td>
    <td>Input K in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>Input V in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8, INT4</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>Output</td>
    <td>Output in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- When this API is used with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages.

- Handling of empty input parameters: The operator internally checks whether the parameter query is empty. If it is empty, the operator returns directly. If query is not an empty tensor but key and value are empty tensors (that is, S2 is 0), attentionOut is filled with all zeros. When attentionOut is an empty tensor, the AscendCLNN framework handles it. When other input parameters marked as "nullptr can be passed in" in the preceding parameter description are null pointers, they are not processed.

- The shapes of the corresponding tensors in key and value must be exactly the same. In non-contiguous scenarios, the batch in the tensorlist of key and value can only be 1, the number equals the B of query, and N and D must be equal. Due to tensorlist limitations, B cannot be greater than 256 in non-contiguous scenarios.

- When Q_S is greater than 1, the function usage restrictions for the query, key, and value inputs are as follows:
    - The B axis supports a value less than or equal to 65536.

  - If the input type is INT8 and the D axis is not 32-byte aligned, the maximum supported value of the B axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis also supports a maximum value of 128.

  - The N axis supports a value less than or equal to 256, and the D axis supports a value less than or equal to 512. When inputLayout is BSH or BSND, N*D is recommended to be less than 65535.

  - S supports a value less than or equal to 20971520 (20M). In some long-sequence scenarios, an excessive computation volume may cause the pfa operator execution to time out (an aicore error is reported, with errorStr: timeout or trap error). In this scenario, S splitting is recommended. Note: the computation volume here is affected by B, S, N, D, and other factors. The larger the values, the larger the computation volume. Typical long-sequence scenarios (that is, scenarios where the product of B, S, N, and D is large) that may time out include but are not limited to:

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 550px"><colgroup>
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 200px">
    <col style="width: 100px">
    <col style="width: 100px">
    <col style="width: 150px">
    </colgroup><thead>
    <tr>
    <th>B</th>
    <th>Q_N</th>
    <th>Q_S</th>
    <th>D</th>
    <th>KV_N</th>
    <th>KV_S</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>1</td>
    <td>20</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>2</td>
    <td>20971520</td>
    <td>256</td>
    <td>2</td>
    <td>20971520</td>
    </tr>
    <tr>
    <td>20</td>
    <td>1</td>
    <td>2097152</td>
    <td>256</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    <tr>
    <td>1</td>
    <td>10</td>
    <td>2097152</td>
    <td>512</td>
    <td>1</td>
    <td>2097152</td>
    </tr>
    </tbody>
    </table>
    </div>

  - D axis restrictions: When the types of query, key, value, or attentionOut include INT8, the D axis needs 32-byte alignment. When the types of query, key, value, or attentionOut include INT4, the D axis needs 64-byte alignment. When all types are FLOAT16 or BFLOAT16, the D axis needs 16-byte alignment.

- When Q_S equals 1, the function usage restrictions for the query, key, and value inputs are as follows:

  - The B axis supports a value less than or equal to 65536, the N axis supports a value less than or equal to 256, and the D axis supports a value less than or equal to 512.
  - The scenario where the input types of query, key, and value are all INT8 is not supported yet.
  - In the INT4 pseudo-quantization scenario, the aclnn single-operator invocation supports KV INT4 input or INT4 concatenated into INT32 input (it is recommended to generate data in INT4 format through dynamicQuant, because one dynamicQuant INT32 contains 8 INT4 values).
  - In the INT4 pseudo-quantization scenario, if KV INT4 is concatenated into INT32 input, the N, D, or H of KV is one eighth of the actual value (the same applies to prefix).
  - key and value have D axis restrictions under specific data types
    - When the input types of key and value are INT4 (INT32), the D axis needs 64-byte alignment (INT32 supports only 8-byte alignment of D).

- The data layout formats of query, key, and value support interpretation from multiple dimensions, where B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Head-Size) indicates the size of the hidden layer, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the size of the smallest unit of the hidden layer and satisfies D=H/N, and T indicates the accumulated sum of the sequence lengths of input samples of all batches.
- The Kirin X90/Kirin 9030 series has the following additional constraints:
    - Only GQA is supported.
    - query, key, value, and attentionOut support only the FLOAT16 data type, and the D axis must be greater than or equal to 16 and aligned to 16.
    - input_layout supports only BNSD, BSH, BSND, and TND, where TND supports only sparse_mode 0 (NoMask) and 9 (TreeMask).
    - sparse_mode does not support 4 or 5.
    - attention_massk supports only 3D [1, S1, S2] or 4D [1, 1, S1, S2] / [1, 1, maxS, maxS], with the data types UINT8, INT8, or BOOL.
    - BNSD/BSH/BSND does not support actual_seq_lenths.

## Calling Instructions

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:

  | Calling Method  | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn API | [test_aclnn_FusedInferAttentionScoreV4](./examples/test_aclnn_fused_infer_attention_score_v4_gqa_noquant.cpp) | Calls the FusedInferAttentionScore operator through [aclnnFusedInferAttentionScoreV4](docs/aclnnFusedInferAttentionScoreV4_en.md) |

- <term>Ascend 950PR&950DT products</term>:

  | Calling Method  | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | aclnn API | [test_aclnn_FusedInferAttentionScoreV5](./examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) | Calls the FusedInferAttentionScore operator through [aclnnFusedInferAttentionScoreV5](docs/aclnnFusedInferAttentionScoreV5_en.md) |

- General (GE graph mode, applicable to all supported chips):

  | Calling Method  | Sample Code                                                     | Description                                                         |
  | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | Graph mode invocation | [test_geir_fused_infer_attention_score](./examples/test_geir_fused_infer_attention_score.cpp) | Calls the FusedInferAttentionScore operator by building a graph with [Ascend IR](./docs/ascendir_FusedInferAttentionScore_en.md) |
