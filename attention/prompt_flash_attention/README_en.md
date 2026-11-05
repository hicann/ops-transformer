# PromptFlashAttention

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

- Operator function: FlashAttention operator for full inference scenarios. It supports sparse optimization, actualSeqLengthsKv optimization, and INT8 quantization, and supports selection between the high-precision mode and the high-performance mode.

- Calculation formula:

    self-attention builds an attention model by using the relationships within the input samples themselves. Its principle is as follows: assume an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. The sequence is transformed by three weight matrices to obtain three matrices of dimension $n*d$.

    The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input samples. They are obtained through spatial transformation of the input samples and can be unified into one feature space. "Attention" in the formula and the operator name is the abbreviation of "self-attention".

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the Score function adopts the Softmax function. The calculation formula of self-attention is:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    where the product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, normalized row by row through softmax, and then multiplied by $V$ to obtain an $n*d$ matrix.

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
    <th>Parameter</th>
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
    <td>FLOAT16, BFLOAT16, INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>Input V in the formula.</td>
    <td>FLOAT16, BFLOAT16, INT8</td>
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

- <term>Atlas A2 products</term> and <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT16, BFLOAT16, and INT8.
- Atlas inference products: Only FLOAT16 is supported.

## Constraints

- When this API is used with PyTorch, ensure that the versions of the CANN-related packages match those of the PyTorch-related packages.

- Handling of empty input parameters: the operator internally checks whether the parameter query is empty, and directly returns if it is empty. If the parameter query is not an empty tensor but the parameters key and value are empty tensors, attentionOut is filled with all zeros. When attentionOut is an empty tensor, the aclnn API handles it. When the input parameters marked as "a null pointer can be passed in" in the preceding parameter description are null pointers, no processing is performed.

- Restrictions on the use of the query, key, and value inputs:

  - <term>Atlas A2 products</term> and <term>Ascend 950PR&950DT products</term>:

    - The B axis supports values less than or equal to 65536 (64k). When the input type includes INT8 and the D axis is not aligned to 32, or the input type is FLOAT16 or BFLOAT16 and the D axis is not aligned to 16, the B axis only supports up to 128.

    - The N axis supports values less than or equal to 256.

    - S supports values less than or equal to 20971520 (20M). In some long-sequence scenarios, an excessively large computation workload may cause the pfa operator execution to time out (an AI Core error is reported, with errorStr: timeout or trap error). In this scenario, splitting S is recommended. Note: the computation workload is affected by B, S, N, D, and other values; the larger the values, the larger the computation workload. Typical long-sequence scenarios that may time out (that is, scenarios where the product of B, S, N, and D is large) include but are not limited to the following:
      <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
      <col style="width: 100px">
      <col style="width: 100px">
      <col style="width: 200px">
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

    - The D axis supports values less than or equal to 512. When inputLayout is BSH or BSND, N*D must be less than 65535.

  - <term>Atlas A2 products</term>: comprehensive restrictions on the query, key, and value inputs in TND scenarios:
    - T is less than or equal to 65536.
    - N equals 8/16/32/64/128, and Q_N, K_N, and V_N are equal.
    - Q_D and K_D equal 192, and V_D equals 128/192.
    - Only the BFLOAT16 data type is supported.
    - The sparse mode supports only sparse=0 without a mask, or sparse=3 with a mask.
    - When sparse=3, the actualSeqLengths of each batch must be less than actualSeqLengthsKv.

  - Atlas inference products:
    - When inputLayout is BSH, the B axis supports values less than or equal to 300; in other cases, the B axis supports values less than or equal to 128. The N axis supports values less than or equal to 256. The S axis supports values less than or equal to 65535 (64k). When Q_S or KV_S is not aligned to 128 and Q_S and KV_S have different lengths, configuring atten_mask is not supported. The D axis supports values less than or equal to 512.

- When inputLayout is BNSD_BSND, the shape of the input query is BNSD, and the shape of the output attentionOut is BSND; in other cases, the shape of attentionOut must be consistent with the shape of the input parameter query.

## Invocation Description

| Invocation Method  | Sample Code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn API | [test_aclnn_PromptFlashAttentionV3](./examples/test_aclnn_prompt_flash_attention_v3.cpp) | Invokes the PromptFlashAttentionV3 operator through [aclnnPromptFlashAttentionV3](docs/aclnnPromptFlashAttentionV3_en.md) |
