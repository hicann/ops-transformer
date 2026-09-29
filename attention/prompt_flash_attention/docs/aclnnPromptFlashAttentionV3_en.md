# aclnnPromptFlashAttentionV3

[📄 View Source Code](https://gitcode.com/cann/ops-transformer/tree/master/attention/prompt_flash_attention)

## Supported Products

|Product     | Supported|
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      ×     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>|      √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Description

- Function: FlashAttention operator in the full inference scenario. This API supports the following functions: sparse optimization, `actualSeqLengthsKv` optimization, INT8 quantization, and the `innerPrecise` parameter to select the high-precision or high-performance mode.

- Formula:

  Self-attention constructs an attention model by leveraging the relationships within the input samples. The principle assumes there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. Such a sequence is transformed by three weight matrices to obtain three $n*d$ matrices.

  The calculation formula for self-attention is generally defined as follows, where $Q$, $K$, and $V$ are key attribute elements of the input sample, obtained through spatial transformation and unified into a single feature space. "Attention" in the formula and operator name is an abbreviation for "self-attention."

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  The Score function in this operator employs the Softmax function. The self-attention calculation formula is as follows:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  The product of $Q$ and $K^T$ represents the attention to the input $x$. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. The result is then multiplied by $V$ to produce an $n*d$ matrix.

## Prototype

The operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnPromptFlashAttentionV3GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnPromptFlashAttentionV3` is called to perform computation.

```cpp
aclnnStatus aclnnPromptFlashAttentionV3GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor   *deqScale1,
    const aclTensor   *quantScale1,
    const aclTensor   *deqScale2,
    const aclTensor   *quantScale2,
    const aclTensor   *quantOffset2,
    int64_t            numHeads,
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char              *inputLayout,
    int64_t            numKeyValueHeads,
    int64_t            sparseMode,
    int64_t            innerPrecise,
    const aclTensor   *attentionOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttentionV3(
     void              *workspace,
     uint64_t           workspaceSize,
     aclOpExecutor     *executor,
     const aclrtStream  stream)
```

## aclnnPromptFlashAttentionV3GetWorkspaceSize

- **Parameters**
  
    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup> 
     <col style="width: 180px"> 
     <col style="width: 120px"> 
     <col style="width: 300px"> 
     <col style="width: 330px"> 
     <col style="width: 252px"> 
     <col style="width: 100px">  
     <col style="width: 140px">  
     <col style="width: 145px">  
     </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Precaution</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the formula.</td>
        <td>The data type must be the same as that of key and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>The data type must be the same as that of query and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>The data type must be the same as that of query and key.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Positional encoding.</td>
        <td><ul><li>If this parameter is not used, pass nullptr.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>If this parameter is not used, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Valid sequence lengths of queries in different batches.</td>
        <td><ul><li>If the sequence lengths are not specified, pass nullptr.</li>
            <li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKv</td>
        <td>Input</td>
        <td>Valid sequence lengths of key and value in different batches.</td>
        <td><ul><li>If the sequence lengths are not specified, pass nullptr.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale1</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
          <td><ul><li> Per-tensor configuration is supported.</li></ul>
              <ul><li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li> Per-tensor configuration is supported. </li></ul>
            <ul><li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li> Per-tensor configuration is supported. </li></ul>
            <ul><li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>Input</td>
        <td>Output quantization factor.</td>
        <td><ul><li>Per-tensor and per-channel configuration is supported. </li></ul>
            <ul><li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Output quantization offset.</td>
        <td><ul><li>Per-tensor and per-channel configuration is supported. </li></ul>
            <ul><li>If this parameter is not used, pass nullptr.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of query heads.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td><ul><li>Its data type must be compatible with that of query according to the type deduction rules. </li></ul>
            <ul><li>If no specific value is required, you are advised to set it to 1.0. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Number of preceding tokens to associate in attention computation.</td>
        <td>If no specific value is required, 2147483647 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Number of succeeding tokens to associate in attention computation.</td>
        <td>If no specific value is required, you are advised to set it to 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Layout of the input query, key, and value.</td>
        <td><ul><li>If no specific value is required, BSH is recommended.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>If no specific value is required, you are advised to set it to 0, indicating that key, value and query have the same number of heads.</li></ul>
            <ul><li>For details about the constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Choice between the high-precision and high-performance modes.</td>
        <td>For details about the constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT8</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
    </tbody></table>
    </div>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
   The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of query, key, value, pseShift, attenMask, or attentionOut is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An error occurred when the API calls the NPU runtime interface.</td>
    </tr>
  </tbody>
  </table>

## aclnnPromptFlashAttentionV3

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnPromptFlashAttentionV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing:
  - `aclnnPromptFlashAttentionV3` defaults to a deterministic implementation.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.

- Processing logic for a null input parameter: The operator checks whether `query` is a null pointer. If so, an error is reported. If `query` is not an empty tensor but `key` and `value` are empty tensors (that is, S2 is 0), `attentionOut` is filled with all zeros. When attentionOut is an empty tensor, the AscendCLNN framework will process it. For other input parameters which support the passing of null pointers as described in the preceding parameter description, no processing is performed when they are null pointers.

- The data format of `query`, `key`, and `value` can be interpreted from multiple dimensions. To be specific, **B (Batch)** indicates the size of an input sample batch, **S (Seq-Length)** indicates the length of the input sample sequence, **H (Head-Size)** indicates the size of the hidden layer, **N (Head-Num)** indicates the number of heads, and **D (Head-Dim)** indicates the minimum unit size of the hidden layer (**D** = **H**/**N**). **T** indicates the total length of all input sample sequences.

- Restrictions on `query`, `key`, and `value`:

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT:
  
    - The B axis must be less than or equal to 65536 (64K). If the input type is INT8 and the D axis is not 32-byte aligned, or the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-byte aligned, the B axis can be up to 128.

    - The N axis must be less than or equal to 256.

    - The S axis must be less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator execution may time out (AI Core error, errorStr: timeout or trap error). In this case, S axis splitting is recommended. Note: The computation load is affected by parameters such as **B**, **S**, **N**, and **D**. Larger values indicate larger computation loads. The following lists some typical scenarios with long sequences (that is, the product of **B**, **S**, **N**, and **D** is large).
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

    - The D axis must be less than or equal to 512. If `inputLayout` is BSH or BSND, N x D must be less than 65535.
    
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: Constraints on the input of `query`, `key`, and `value` in the TND scenario:
      - **T** is less than or equal to 65536.
      - **N** is 8, 16, 32, 64, or 128, and **Q_N**, **K_N**, and **V_N** are equal.
      - **Q_D** and **K_D** are 192, and **V_D** is 128 or 192.
      - The data type is BFLOAT16.
      - The sparse mode can only be 0 without a mask or 3 with a mask.
      - When the sparse mode is 3, `actualSeqLengths` must be less than `actualSeqLengthsKv` for each batch.
      
  - <term>Atlas inference accelerator cards</term>:
      - When inputLayout is BSH, the B axis must be less than or equal to 300, and the B axis must be less than or equal to 128 in other cases. The N axis must be less than or equal to 256. The S axis must be less than or equal to 65535 (64k). If Q_S or KV_S is not 128-aligned and the lengths of Q_S and KV_S are different, atten_mask cannot be configured. The D axis must be less than or equal to 512.
      
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be FLOAT16, BFLOAT16, or INT8.
  - <term>Atlas inference accelerator cards</term>: The data type can be FLOAT16.
  
- Restrictions on `pseShift`:
  
  - The input shape must be (B, N, Q_S, KV_S) or (1, N, Q_S, KV_S). **Q_S** is **S** in the shape of **query**, and **KV_S** is **S** in the shape of **key** and **value**.
  - **Q_S** must be greater than or equal to **S** of **query**, and **KV_S** must be greater than or equal to **S** of **key**.
  - In the scenario where **KV_S** of **pseShift** is not 32-byte aligned, it is recommended that it be padded to 32 bytes to improve the performance. There is no requirement on the padding value of the extra part. If this parameter is not used, pass **nullptr**.

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be FLOAT16 or BFLOAT16. When pseShift is of the FLOAT16 type, the query must be of the FLOAT16 or INT8 type. When pseShift is of the BFLOAT16 type, the query must be of the BFLOAT16 type. When the data type of `query`, `key`, and `value` is FLOAT16 and `pseShift` exists, the high-precision mode is used by default. The corresponding restrictions are the same as those of the high-precision mode.
  - <term>Atlas inference accelerator cards</term>: The value can only be `nullptr`.

- Restrictions on `attenMask`:
  
  - In the scenario where **KV_S** of `attenMask` is not 32-byte aligned, it is recommended that it be padded to 32 bytes to improve the performance, filling excess positions with ones.
  - Recommended shapes are **Q_S,KV_S**, **B,Q_S,KV_S**, **1,Q_S,KV_S**, **B,1,Q_S,KV_S**, and **1,1,Q_S,KV_S**. **Q_S** is **S** in the shape of `query`, and **KV_S** is **S** in the shape of `key` and `value`.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be BOOL, INT8, or UINT8.
  - <term>Atlas inference accelerator cards</term>: The value can only be BOOL type.

- Restrictions on the input of `actualSeqLengths` and `actualSeqLengthsKv`:
  
  - For `actualSeqLengths`, if the sequence length is not specified, `nullptr` can be passed, indicating that the valid sequence length is the same as **S** in the shape of the `query`. Note that the valid sequence length of each batch in this parameter cannot exceed the sequence length of the corresponding batch in `query`.
  - For `actualSeqLengthsKv`, if the sequence length is not specified, `nullptr` can be passed, indicating that the valid sequence length is the same as **S** in the shape of the `key` and `value`. Note that the valid sequence length of each batch in this parameter cannot exceed the sequence length of the corresponding batch in `key` and `value`.
  - The rules for the input length of `seqlen` are as follows: If the input length of `seqlen` is 1, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` is used. Other lengths are not supported.
  - When `inputLayout` of `query` is TND, this parameter must be passed, and the number of elements in this parameter is used as the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of the next element must be greater than or equal to the value of the previous element, and cannot be negative.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be INT64. The TND format is supported.
  - <term>Atlas inference accelerator cards</term>: The data type can be INT64.
  - Ascend 950PR/Ascend 950DT: The data type can be INT64.
  
- Restrictions on the input of `deqScale1` and `deqScale2`:
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be UINT64 or FLOAT32.
  - <term>Atlas inference accelerator cards</term>: The value can only be `nullptr`.
  
- Restrictions on the input of `quantScale1`:
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be FLOAT32.
  - <term>Atlas inference accelerator cards</term>: The value can only be `nullptr`.
  
- Restrictions on the input of `quantScale2` and `quantOffset2`:
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be FLOAT32 or BFLOAT16.
  - <term>Atlas inference accelerator cards</term>: The value can only be `nullptr`.
  
- Restrictions on the input of `preTokens`:
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be INT64.
  - <term>Atlas inference accelerator cards</term>: The value can only be 2147483647.
  
- Restrictions on the input of `nextTokens`:
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be INT64.
  - <term>Atlas inference accelerator cards</term>: The value can only be 0 or 2147483647.
  
- Restrictions on the input of `inputLayout`:
  
  - The input format can be BSH, BSND, BNSD, or BNSD_BSND. When the input format is BNSD, the output format is BSND. If no specific format is required, you are advised to set it to BSH.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: In addition to the preceding formats, TND is also supported (PSE, full quantization, and post-quantization are not supported).
  
- Restrictions on the input of `numKeyValueHeads`:
  
  - `numHeads` must be divisible by `numKeyValueHeads`, and in BSND, BNSD, BNSD_BSND scenarios, it must match the N axis shape value of `key` and `value` in `shape`. Otherwise an error is reported.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be INT64.
  - <term>Atlas inference accelerator cards</term>: The value can only be 0.
  
- Restrictions on the input of `sparseMode`:
  
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT:
    - When `sparseMode` is 0, the defaultMask mode is used. If `attenMask` is not passed, the mask operation is not performed, and `preTokens` and `nextTokens` are ignored (internally set to INT_MAX). If `attenMask` is passed, a complete `attenMask` matrix (S1 × S2) needs to be passed, indicating that the portion between `preTokens` and `nextTokens` needs to be calculated.
    - When `sparseMode` is 1, the allMask mode is used. A complete `attenMask` matrix (S1 × S2) must be passed.
    - When `sparseMode` is 2, the leftUpCausal mode is used. An optimized `attenMask` matrix (2048 × 2048) needs to be passed.
    - When `sparseMode` is 3, the rightDownCausal mode for lower-triangle scenarios with right vertex as the dividing line. An optimized `attenMask` matrix (2048 × 2048) needs to be passed.
    - When `sparseMode` is 4, the band mode is used. An optimized `attenMask` matrix (2048 × 2048) needs to be passed.
    - When `sparseMode` is 5, 6, 7, or 8, the prefix, global, dilated, and block_local modes are used respectively, which are **not supported currently**. If no specific value is required, you are advised to set it to 0.
  - <term>Atlas A2 training products/Atlas A2 inference products</term>: When `inputLayout` is of the TND format, `sparseMode` can only be 0 or 3.
  - <term>Atlas inference accelerator cards</term>: The value can only be 0.
  
- Restrictions on the input of `innerPrecise`:
  
  - There are four modes (0, 1, 2, and 3) in total, represented by 2-bit combinations. Bit 0 indicates whether to use the high-precision or high-performance mode, and bit 1 indicates whether to perform invalid row correction.
    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
      <col style="width: 200px">
    </colgroup><thead>
    <tr>
    <th>innerPrecise</th>
    <th>Mode</th>
    <th>Invalid Row Correction</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>High-precision</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>High-performance</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>High-precision</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>High-performance</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>
  
  - When **Q_S** is greater than 1, `sparseMode` is set to 0 or 1, and a user-defined mask is passed. It is recommended that the invalid row correction be enabled.
  
  - The high-precision and high-performance modes are applicable to both BFLOAT16 and INT8. Invalid row correction takes effect for FLOAT16, BFLOAT16, and INT8. The values 0 and 1 are reserved. If the masks involved in the computation are all ones, the precision may be affected. In this case, you can set this parameter to 2 or 3 to enable invalid row correction to improve the precision. However, this configuration deteriorates the performance. If the operator can determine that invalid rows exist, the invalid row correction is automatically enabled, such as in scenarios where `sparseMode` is set to 3 and Sq is greater than Skv.
  
- Restrictions on the input of `attentionOut`:
  
  - When `inputLayout` is set to BNSD_BSND, the shape of the input query is BNSD and the output shape is BSND. In other cases, the shape of this input parameter must be the same as that of the input parameter `query`.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and Ascend 950PR/Ascend 950DT: The data type can be FLOAT16, BFLOAT16, or INT8.
  - <term>Atlas inference accelerator cards</term>: The value can only be FLOAT16.
  
- Constraints on the number of input parameters and input and output data formats related to INT8 quantization:
  
  - If both the input and output are of the INT8 type, the input parameters `deqScale1`, `quantScale1`, `deqScale2`, and `quantScale2` must exist at the same time. `quantOffset2` is optional. If this parameter is not specified, the default value 0 is used.
  - If the input is of the INT8 type and the output is of the FLOAT16 type, the input parameters `deqScale1`, `quantScale1`, and `deqScale2` must exist at the same time. If the input parameter `quantOffset2` or `quantScale2` exists (not `nullptr`), an error is reported and returned.
  - When the input is of the FLOAT16 or BFLOAT16 type and the output is of the INT8 type, the input parameter `quantScale2` must exist, and `quantOffset2` is optional (0 is used if no value is passed). If the input parameter `deqScale1`, `quantScale1`, or `deqScale2` exists (not `nullptr`), an error is reported and returned.
  - The input parameters `quantScale2` and `quantOffset2` support both the per-tensor and per-channel formats and the FLOAT32 and BFLOAT16 data types. If `quantOffset2` is passed, ensure that its type and shape are consistent with those of `quantScale2`. When the input is of the BFLOAT16 type, both FLOAT32 and BFLOAT16 are supported. Otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of `quantScale2` must be equal to **H**. For other layouts, the product must be equal to **N** × **D**. (When the output layout is BSH, it is recommended that the shape of `quantScale2` be set to [1,1,H] or [H]. When the output layout is BNSD, it is recommended that the shape of `quantScale2` be set to [1,N,1,D] or [N,D]. When the output layout is BSND, it is recommended that the shape of `quantScale2` be set to [1,1,N,D] or [N,D].)
  - When the output is INT8 and `quantScale2` and `quantOffset2` are set to per-channel mode, the left padding, ring attention, or non-32-byte alignment of the D axis are not supported.
  - If the output is INT8, the sparse mode cannot be band, and `preTokens` and `nextTokens` cannot be negative numbers.

- When the output is INT8, the input parameter `quantOffset2` is a non-null pointer and a non-empty tensor, and `sparseMode`, `preTokens`, and `nextTokens` meet the following conditions, some rows of the matrix are not involved in computation. As a result, the computation result is inaccurate. The following scenarios trigger interception (solution: perform post-quantization outside the PFA interface):
    - If sparseMode is set to 0 and attenMask is a non-null pointer, the interception condition is met when actualSeqLengths - actualSeqLengthsKV - preTokens > 0 or nextTokens < 0 for each batch.
    - When `sparseMode` is 1 or 2, interception does not occur.
    - When `sparseMode` is 3, interception occurs if `actualSeqLengthsKV` - `actualSeqLengths` < 0 for any batch.
    - When `sparseMode` is 4, interception occurs if `preTokens` < 0 or `nextTokens` + `actualSeqLengthsKV` - `actualSeqLengths` < 0 for any batch.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"
#include "securec.h"
 
using namespace std;

namespace {
 
#define CHECK_RET(cond) ((cond) ? true :(false))
 
#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)
 
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}
 
int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); 
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); 
    return ret;
  }
  return 0;
}
 
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
    return ret;
  }
  
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) { 
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
    return ret;
  }
 
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
 
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 16};
    std::vector<int64_t> keyShape = {1, 2, 2, 16};
    std::vector<int64_t> valueShape = {1, 2, 2, 16};
    std::vector<int64_t> attenShape = {1, 1, 1, 2};
    std::vector<int64_t> outShape = {1, 2, 1, 16};
    
    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t attenShapeSize = GetShapeSize(attenShape);
    int64_t outShapeSize = GetShapeSize(outShape);
    
    std::vector<float> queryHostData(queryShapeSize, 1);
    std::vector<float> keyHostData(keyShapeSize, 1);
    std::vector<float> valueHostData(valueShapeSize, 1);
    std::vector<float> attenHostData(attenShapeSize, 1);
    std::vector<float> outHostData(outShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr, 
                             aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(attenHostData, attenShape, &resources.attenDeviceAddr, 
                         aclDataType::ACL_BOOL, &resources.attenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr, 
                         aclDataType::ACL_FLOAT16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    std::vector<int64_t> actualSeqlenVector = {2};
    resources.actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), 
                                                  actualSeqlenVector.size());

    return ACL_SUCCESS;
}

int ExecutePromptFlashAttention(TensorResources& resources, aclrtStream stream, 
                               void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1 / sqrt(2);
    int64_t preTokens = 65535;
    int64_t nextTokens = 65535;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    
    constexpr const char LAYER_OUT_STR[] = "BNSD";
    constexpr size_t LAYER_OUT_LEN = sizeof(LAYER_OUT_STR);  
    char layerOut[LAYER_OUT_LEN];
    memcpy(layerOut, LAYER_OUT_STR, LAYER_OUT_LEN);

    aclOpExecutor* executor;
    int ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.valueTensor, 
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
        numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, 
        sparseMode, innerPrecise, resources.outTensor, workspaceSize, &executor);
        
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnPromptFlashAttentionV3(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int ProcessResults(TensorResources& resources, const std::vector<int64_t>& outShape) {
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);
    
    int ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), 
                         resources.outDeviceAddr, size * sizeof(resultData[0]), 
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    
    for (int64_t i = 0; i < size; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }
    
    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr, 
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.attenTensor) {
      aclDestroyTensor(resources.attenTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }
    if (resources.actualSeqLengths) {
      aclDestroyIntArray(resources.actualSeqLengths);
    }
    
    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.attenDeviceAddr) {
      aclrtFree(resources.attenDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }
    
    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace
 
int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 2, 1, 16};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecutePromptFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    ret = ProcessResults(resources, outShape);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
