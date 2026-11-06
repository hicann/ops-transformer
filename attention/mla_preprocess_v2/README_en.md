# MlaPreprocessV2

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      ×     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      ×     |
|<term>Atlas training products</term>|      ×     |

## Function Description

- **Operator function**: Computes the preprocessing of Multi-Head Latent Attention in inference scenarios. The main computation process is as follows:
    - First, the input $x$ is processed by RmsNormQuant and then multiplied by $W^{DQKV}$ for downsampling, and the result is split into path 1 and path 2.
    - Path 1 is processed by RmsNormQuant, multiplied by $W^{UQ}$, and then split into path 3 and path 4.
    - Path 3 is multiplied by $W^{uk}$ and outputs $q^N$.
    - Path 4 passes through the rotary position embedding and outputs $q^R$.
    - Path 2 is split into path 5 and path 6.
    - Path 5 passes through RmsNorm and is written into the cache to obtain $k^N$.
    - Path 6 passes through the rotary position embedding and is written into another cache to obtain $k^R$.
- **Computation formulas**:

    RmsNormQuant formula

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    RmsNormQuant(x) = ({RmsNorm}(x) + bias) * deqScale
    $$

    Query computation formulas, including the W^{DQKV} matrix multiplication, W^{UK} matrix multiplication, RmsNormQuant, and ROPE rotary position embedding processing

    $$
    q^N =  RmsNormQuant(x) \cdot W^{DQKV} \cdot W^{UK}
    $$

    $$
    q^R = ROPE(x^Q)
    $$

    Key computation formulas, including RmsNorm and ROPE. The computation results are stored in the cache.

    $$
    k^N = Cache({RmsNorm}(RmsNormQuant(x)))
    $$

    $$
    k^R = Cache(ROPE(RmsNormQuant(x)))
    $$

    When both `cos` and `sin` of the ACLNN interface are passed as null pointers, the Q-RoPE and K-RoPE mathematical operations are skipped, and $q^R$ and $k^R$ store the unrotated original Q-rope and K-rope components respectively. When both are non-null tensors, the original RoPE behavior is retained. Passing only one null pointer is prohibited. Other computations, output shapes, the `qDownOut` semantics, and cache layouts remain unchanged.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1427px"><colgroup>
<col style="width: 194px">
<col style="width: 146px">
<col style="width: 721px">
<col style="width: 230px">
<col style="width: 136px">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
  </tr></thead>
<tbody>
  <tr>
    <td>input</td>
    <td>Input</td>
    <td>Device-side aclTensor, the x used to compute the Query and Key, with shape [tokenNum,hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the γ parameter in the first RmsNorm computation, with shape [hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the β parameter in the first RmsNorm computation, with shape [hiddenSize]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the quantization scale parameter in the first RmsNorm formula, with shape [1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the quantization offset parameter in the first RmsNorm formula, with shape [1]</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wdqkv</td>
    <td>Input</td>
    <td>Device-side aclTensor, the dimension reduction matrix that first performs matrix multiplication with the input, with shape [qLoraDim + keyTotalDim,hiddenSize]</td>
    <td>INT8, FLOAT16, BFLOAT16</td>
    <td>NZ</td>
  </tr>
  <tr>
    <td>deScale0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient in the dimension reduction matrix that first performs matrix multiplication with the input, with shape [qLoraDim + keyTotalDim]. INT64 is supported when the dtype of input is FLOAT16, and FLOAT is supported when the input is BFLOAT16</td>
    <td>INT64, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias0</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient in the dimension reduction matrix that first performs matrix multiplication with the input, with shape [qLoraDim + keyTotalDim]. Passing an empty tensor is supported; do not pass this parameter when quantMode is 1 or 3</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the γ parameter in the second RmsNorm computation, with shape [qLoraDim]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the β parameter in the second RmsNorm computation, with shape [qLoraDim]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantScale1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the quantization scale parameter in the second RmsNorm formula, with shape [1]. Pass this parameter only when quantMode is 0</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>quantOffset1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the quantization offset parameter in the second RmsNorm formula, with shape [1]. Pass this parameter only when quantMode is 0</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wuq</td>
    <td>Input</td>
    <td>Device-side aclTensor, the weight matrix, with shape [headNum * (qNoRopeDim + qRopeDim),qLoraDim]</td>
    <td>INT8, FLOAT16, BFLOAT16</td>
    <td>NZ</td>
  </tr>
  <tr>
    <td>deScale1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient involved in the wuq matrix multiplication, with shape [headNum * (qNoRopeDim + qRopeDim)]. INT64 is supported when the dtype of input is FLOAT16, and FLOAT is supported when the input is BFLOAT16</td>
    <td>INT64, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias1</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient involved in the wuq matrix multiplication, with shape [headNum * (qNoRopeDim + qRopeDim)]. Do not pass this parameter when quantMode is 1 or 3</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gamma2</td>
    <td>Input</td>
    <td>Device-side aclTensor, the γ parameter involved in the RmsNormAndreshapeAndCache computation, with shape [512].</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>Input</td>
    <td>Device-side aclTensor, the cosine parameter matrix used to compute the rotary position embedding. When RoPE is enabled, the shape is [tokenNum,64], where tokenNum equals dim 0 of input; RoPE is disabled when both cos and sin are passed as null pointers.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>Input</td>
    <td>Device-side aclTensor, the sine parameter matrix used to compute the rotary position embedding. When RoPE is enabled, the shape is [tokenNum,64], where tokenNum equals dim 0 of input; RoPE is disabled when both cos and sin are passed as null pointers.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wuk</td>
    <td>Input</td>
    <td>Device-side aclTensor, the upsampling weight used to compute the Key, with shape [headNum,qNoRopeDim,512].</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kvCache</td>
    <td>Input</td>
    <td>Device-side aclTensor, the same tensor as the output kvCacheOut. The input format varies with cacheMode.<br><br>When cacheMode is 0: shape [blockNum,blockSize,1,576]<br>When cacheMode is 1: shape [blockNum,blockSize,1,512]<br>When cacheMode is 2: shape [blockNum,headNum*512/32,block_size,32]<br>When cacheMode is 3: shape [blockNum,headNum*512/16,block_size,16]</td>
    <td>When cacheMode is 0: same as input<br>When cacheMode is 1: same as input<br>When cacheMode is 2: INT8<br>When cacheMode is 3: same as input</td>
    <td>ND<br>ND<br>NZ<br>NZ</td>
  </tr>
  <tr>
    <td>kvCacheRope</td>
    <td>Input</td>
    <td>Device-side aclTensor, optional parameter; passing a null pointer is supported. This is the same tensor as the output krCacheOut. The input format varies with cacheMode.<br><br>When cacheMode is 0: not passed.<br>When cacheMode is 1: shape [blockNum,blockSize,1,64]<br>When cacheMode is 2 or 3: shape [blockNum, headNum*64 / 16 ,block_size, 16]</td>
    <td>Same as input</td>
    <td><br>ND<br>NZ</td>
  </tr>
  <tr>
    <td>slotmapping</td>
    <td>Input</td>
    <td>Device-side aclTensor, the indices used to store kv_cache and kr_cache, with shape [tokenNum]</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ctkvScale</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient involved in the output quantization processing. Pass this parameter only when cacheMode is 2, with shape [1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>qNopeScale</td>
    <td>Input</td>
    <td>Device-side aclTensor, the coefficient involved in the output quantization processing. Pass this parameter only when cacheMode is 2, with shape [1]</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>wdqDim</td>
    <td>Input</td>
    <td>Indicates the dim size split after the matmul. Reserved parameter; currently only 1536 is supported</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qRopeDim</td>
    <td>Input</td>
    <td>Indicates the dim size of q passed to rope. Reserved parameter; currently only 64 is supported.</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>kRopeDim</td>
    <td>Input</td>
    <td>Indicates the dim size of k passed to rope. Reserved parameter; currently only 64 is supported.</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Indicates the value added to the denominator to prevent division by 0</td>
    <td>float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qRotaryCoeff</td>
    <td>Input</td>
    <td>Indicates the q rotation coefficient. Reserved parameter; currently only 2 is supported</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>kRotaryCoeff</td>
    <td>Input</td>
    <td>Indicates the k rotation coefficient. Reserved parameter; currently only 2 is supported</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWdq</td>
    <td>Input</td>
    <td>Indicates whether wdq is transposed. Reserved parameter; currently only true is supported</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWuq</td>
    <td>Input</td>
    <td>Indicates whether wuq is transposed. Reserved parameter; currently only true is supported</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>transposeWuk</td>
    <td>Input</td>
    <td>Indicates whether wuk is transposed. Reserved parameter; currently only true is supported</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cacheMode</td>
    <td>Input</td>
    <td>Indicates the specified cache type. Value range: [0, 3]<br><br>0: kcache and q are output after concatenation<br>1: The output kvCacheOut is split into kvCacheOut and krCacheOut, and qOut is split into qOut and qRopeOut<br>2: krope and ctkv are converted to the NZ format for output, and ctkv and qnope are quantized to the int8 type through per_head static symmetric quantization<br>3: krope and ctkv are converted to the NZ format for output</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Indicates the specified RmsNorm quantization type. Value range: [0, 3]<br>0: per_tensor static asymmetric quantization, the default quantization type<br>1: per_token dynamic symmetric quantization, not implemented<br>2: per_token dynamic asymmetric quantization, not implemented<br>3: No quantization, floating-point output, not implemented</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>doRmsNorm</td>
    <td>Input</td>
    <td>Indicates whether to perform the RmsNormQuant operation on the input. false indicates that no operation is performed, and true indicates that the operation is performed. Reserved parameter; currently only true is supported</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>wdkvSplitCount</td>
    <td>Input</td>
    <td>Indicates the number of wdkv splits. [1-3] is supported, indicating no split, a split into 2 dimension reduction matrices, and a split into 3 dimension reduction matrices respectively. Reserved parameter; currently only 1 is supported</td>
    <td>int64_t</td>
    <td>-</td>
  </tr>
  <tr>
    <td>qOut</td>
    <td>Output</td>
    <td>Indicates the output tensor of the Query, corresponding to the output on the right side of the computation flow graph after NOPE and matrix multiplication. The shape and dtype vary with cacheMode<br><br>When cacheMode is 0: shape [tokenNum, headNum, 576]<br>When cacheMode is 1 or 3: shape [tokenNum, headNum, 512]<br>When cacheMode is 2: shape [tokenNum, headNum, 512]</td>
    <td>When cacheMode is 0: same as input<br>When cacheMode is 1 or 3: same as input<br>When cacheMode is 2: INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kvCacheOut</td>
    <td>Output</td>
    <td>Indicates the output of the Key after ReshapeAndCache. The shape and dtype vary with cacheMode<br><br>When cacheMode is 0: shape [blockNum, blockSize, 1, 576]<br>When cacheMode is 1: shape [blockNum, blockSize, 1, 512]<br>When cacheMode is 2: shape [blockNum, headNum*512/32, block_size, 32]<br>When cacheMode is 3: shape [blockNum, headNum*512/16, block_size, 16]</td>
    <td>When cacheMode is 0: same as input<br>When cacheMode is 1: same as input<br>When cacheMode is 2: INT8<br>When cacheMode is 3: same as input</td>
    <td>ND<br>ND<br>NZ<br>NZ</td>
  </tr>
  <tr>
    <td>qRopeOut</td>
    <td>Output</td>
    <td>Indicates the output of the Query after rotary position embedding. The shape and dtype vary with cacheMode<br><br>When cacheMode is 0: no output<br>When cacheMode is 1 or 3: shape [tokenNum, headNum, 64]<br>When cacheMode is 2: shape [tokenNum, headNum, 64]</td>
    <td><br>When cacheMode is 1 or 3: same as input<br>When cacheMode is 2: same as input</td>
    <td><br>ND<br>ND</td>
  </tr>
  <tr>
    <td>krCacheOut</td>
    <td>Output</td>
    <td>Indicates the output of the Key after ROPE and ReshapeAndCache. The shape and dtype vary with cacheMode,<br><br>When cacheMode is 0: no output<br>When cacheMode is 1: shape [blockNum, blockSize, 1, 64]<br>When cacheMode is 2 or 3: shape [blockNum, headNum*64 / 16 ,block_size, 16]</td>
    <td><br>When cacheMode is 1: same as input<br>When cacheMode is 2 or 3: same as input</td>
    <td><br>ND<br>NZ</td>
  </tr>
</tbody></table>

## Constraint Description

- Meanings and constraints of shape format fields
    - tokenNum: tokenNum indicates the batch size of input samples. Value range: 0~256
    - hiddenSize: hiddenSize indicates the size of the hidden layer. Value range: 2048~10240, a multiple of 256
    - headNum: indicates the number of heads. Value range: 1~128
    - blockNum: the number of blocks in the PagedAttention scenario. The value is fixed to 192
    - blockSize: the block size in the PagedAttention scenario. The value is fixed to 128
    - qloraDim: indicates the LoRA input dimension of the Q matrix. Value range: 32~4096, a multiple of 32
    - keyTotalDim: indicates the total dimension of the Key part. The value is fixed to 576 (512 main dimension + 64 rope dimension)
    - qRopeDim: indicates the dimension of the rotary embedding part in the Q matrix. The value is fixed to 64
    - qNoRopeDim: indicates the dimension of the non-rotary embedding part in the Q matrix. Value range: 16~256, a multiple of 16
- RoPE mode constraint
    - The Rotary Position Embedding (RoPE) operation in the mla_preprocess_v2 operator uses the half mode. The interleave mode is not yet supported.

## Invocation Description

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">Invocation Method</th>
    <th class="tg-0pky">Sample Code</th>
    <th class="tg-0pky">Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8">aclnn API</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_preprocess_v2.cpp">test_aclnn_mla_preprocess_v2
    </a>
    </td>
    <td class="tg-lboi">
    Invokes the MlaPreprocessV2 operator through the
    <a href="./docs/aclnnMlaPreprocessV2_en.md">aclnnMlaPreprocessV2
    </a>
    interface
    </td>
  </tr>
</tbody></table>
