# BlockSparseAttentionGrad

English | [简体中文](./README.md)

## Product Support

|Product      | Supported |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 training products</term>|      √     |
|<term>Atlas A3 inference products</term>|      ×     |
|<term>Atlas A2 training products</term>|      √     |
|<term>Atlas A2 inference products</term>|      ×     |
|<term>Atlas 200I/500 A2 inference products</term>                                         |    ×    |
|<term>Atlas inference products</term>                                                 |    ×    |
|<term>Atlas training products</term>                                                 |    ×    |

## Function Description

* Operator function: aclnnBlockSparseAttention sparse attention backward computation. This operator supports flexible block-level sparsity patterns and uses BlockSparseMask to specify the KV blocks selected by each Q block, implementing efficient sparse attention computation.
* Computation formula:

  The sparse block size is $blockShapeX×blockShapeY$, and BlockSparseMask specifies the sparsity pattern.

  blockShapeX and blockShapeY must be positive. On Ascend 950, both must also be multiples of 16. BlockSparseMask defines the sparsity pattern at the logical block granularity. Each mask element controls whether a contiguous block of blockShapeX × blockShapeY participates in the computation.
  The forward computation formula is:

  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$

  For ease of expression, the computation formulas are expressed using the variables $S$ and $P$:

  $$
  S = Mask(scale⋅query⋅key_{sparse}^{T},atten\_mask)
  $$

  $$
  P = SoftMax(S)
  $$

  $$
  V = value_{sparse}
  $$

  $$
  Out = PV
  $$

  The backward computation formulas are:

  $$
  softmax\_grad = softmaxGrad(dOut, attentionOut)
  $$

  $$
  dP=dOut * V^T
  $$

  $$
  dS = P * (dP-softmax\_grad)
  $$

  $$
  dV=P^T * dOut
  $$

  $$
  dQ=(dS*K)*scale
  $$

  $$
  dK=(dS^T*Q)*scale
  $$

BlockSparseAttentionGrad supports multiple interpretations of the data layout of the input dout, query, key, value, and attentionOut, which can be passed in through qInputLayout and kvInputLayout. To facilitate understanding of the specific layout formats supported later (such as BNSD and TND), this section first provides a unified description of the dimension meaning of each abbreviation letter in the layout formats:

* B: Batch size of the input samples (Batch)
* T: Length when B and S are tightly arranged on the same axis (Total tokens)
* S: Sequence length of the input samples (Seq-Length)
* H: Hidden layer size (Head-Size)
* N: Number of heads (Head-Num)
* D: Size of the smallest unit of the hidden layer, which must satisfy D=H/N (Head-Dim)

qInputLayout and kvInputLayout must be the same. Supported layouts by product:

- <term>Atlas A2 series</term> and <term>Atlas A3 series</term>: BNSD and TND.
- <term>Ascend 950PR/950DT series</term>: BNSD, TND, and BSND.

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1550px">
<colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 145px">
</colgroup>
<thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Usage Instructions</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimensions (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>dout (aclTensor*)</td>
    <td>Input</td>
    <td>Backward output gradient, representing the gradient information of the final output with respect to this operator.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query (aclTensor*)</td>
    <td>Input</td>
    <td>Query vector in the attention computation, that is, query in the formula.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key (aclTensor*)</td>
    <td>Input</td>
    <td>Key vector in the attention computation, that is, key in the formula.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalKTokens, numKeyValueHeads, headDim].</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value (aclTensor*)</td>
    <td>Input</td>
    <td>Value vector in the attention computation, that is, value in the formula.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalVTokens, numKeyValueHeads, headDim].</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut (aclTensor*)</td>
    <td>Input</td>
    <td>Output result of the forward BlockSparseAttention computation, that is, attentionOut in the formula.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse (aclTensor*)</td>
    <td>Input</td>
    <td>Log-sum-exp intermediate result of the Softmax computation. It is used for the logarithm and exponential back-derivation of gradients in the backward computation.</td>
    <td>Empty tensors are not supported.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, 1].</li><li>BNSD: [batch, headNum, maxQSeqLength, 1].</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Block sparse mask, indicating the actual sparsity pattern. Determines which blocks actually participate in the attention computation.</td>
    <td>Empty tensors are not supported.<br>Optional input (required in the current version): <ul><li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)].</li><li>Indicates which blocks participate in the computation (value 1) and which blocks do not (value 0) after the data is divided by blocks.</li><li>If nullptr is passed in, block sparse computation is considered disabled, that is, the attention scores between all tokens are computed.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Attention mask, that is, atten_mask in the formula. It is used to mask out specific tokens that should not participate in the computation.</td>
    <td>Empty tensors are supported.<br>Required when maskType=1. The shape must be exactly [batch, headNum, maxBlockNum, 2], and the dtype must be INT32: <ul><li>maxBlockNum = max(ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)).</li><li>[..., 0] is actualSizeX along the Q direction, and [..., 1] is actualSizeY along the KV direction. The constraint 0 ≤ actualSize ≤ corresponding blockShape must be satisfied.</li><li>Within a block, [0, actualSize) validly participates in the computation, and [actualSize, blockShape) is considered padding. actualSize=0 indicates that the block does not participate in the computation.</li></ul>When maskType=0, this input is ignored, and nullptr must be passed in.</td>
    <td>BOOL, INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockShapeOptional (aclIntArray*)</td>
    <td>Input</td>
    <td>Sparse block sizes in the Q and KV directions, respectively.</td>
    <td>Required. Pass at least two elements [blockShapeX, blockShapeY]. Both values must be positive and may differ. On Ascend 950, both must also be multiples of 16.</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsOptional (aclIntArray*)</td>
    <td>Input</td>
    <td>Array of actual query sequence lengths.<br>It describes the number of actually valid query tokens in each batch in variable-length sequence scenarios (that is, scenarios containing Padding data).</td>
    <td> Variable-length sequence scenario (when qInputLayout is "TND"): this input must be configured. Because the TND format is a one-dimensional contiguous layout, the operator relies on this array to accurately split and delimit the real boundaries of each sequence.</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKvOptional (aclIntArray*)</td>
    <td>Input</td>
    <td>Array of actual key/value sequence lengths.<br>It describes the number of actually valid key/value tokens in each batch in variable-length sequence scenarios (that is, scenarios containing Padding data).</td>
    <td> Variable-length sequence scenario (when kvInputLayout is "TND"): this input must be configured. Because the TND format is a one-dimensional contiguous layout, the operator relies on this array to accurately split and delimit the real boundaries of each sequence.</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout (char*)</td>
    <td>Input</td>
    <td>Data layout format of query. It indicates the specific arrangement of the input tensors in memory (such as contiguous or concatenated arrangement).</td>
    <td>Must match the other layout parameter. Atlas A2/A3 supports BNSD and TND. Ascend 950 also supports BSND.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout (char*)</td>
    <td>Input</td>
    <td>Data layout format of key and value. It indicates the specific arrangement of the input tensors in memory.</td>
    <td>Must match the other layout parameter. Atlas A2/A3 supports BNSD and TND. Ascend 950 also supports BSND.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads (int64_t)</td>
    <td>Input</td>
    <td>Number of attention heads of key/value. It is used to support the head number ratio mapping under the GQA (grouped query attention) mechanism.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType (int64_t)</td>
    <td>Input</td>
    <td>Mask type in the attention computation. It specifies which preset mask logic is used.</td>
    <td>Currently 0 and 1 are supported: <ul><li>0: No mask is applied. attenMaskOptional is ignored.</li><li>1: PerBlockSizeMask scenario. The computation is performed based on the block-level variable-length mask, and attenMaskOptional must be configured.</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue (double)</td>
    <td>Input</td>
    <td>Scaling factor, that is, scale in the formula. It is used to normalize the attention scores.</td>
    <td>Generally set to D^-0.5.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens (int64_t)</td>
    <td>Input</td>
    <td>Number of preceding tokens included in the sliding window. It limits the number of historical tokens before the current token that participate in the attention computation.</td>
    <td>Used for sliding window attention scenarios. Sliding window attention is currently not supported, so only 2147483647 can be passed in.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens (int64_t)</td>
    <td>Input</td>
    <td>Number of following tokens included in the sliding window. It limits the number of future tokens after the current token that participate in the attention computation.</td>
    <td>Used for sliding window attention scenarios. Sliding window attention is currently not supported, so only 2147483647 can be passed in.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq (aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output of query, that is, dq in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input query.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk (aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output of key, that is, dk in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input key.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv (aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output of value, that is, dv in the formula.</td>
    <td>Empty tensors are not supported.<br>The data type and shape are the same as those of the input value.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size that needs to be requested on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor (aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the operator computation process.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

<ul>
- <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>:
FLOAT8_E5M2 and FLOAT8_E4M3FN are not supported.
</ul>

<ul>
- <term>Ascend 950PR&950DT products</term>:
maskType does not support the value 1; the attenMaskOptional input is ignored, and nullptr should be passed in for this input.
</ul>

## Constraints

* When this API is used together with PyTorch, ensure that the versions of the CANN packages match those of the PyTorch packages.
* actualSeqLengthsOptional is required when qInputLayout is "TND"; actualSeqLengthsKvOptional is required when kvInputLayout is "TND".
* The layout of softmaxLse must be consistent with the layout of query. If the layout of query is "BSND", the layout "BNS1" should be passed in for softmaxLse.
* HeadDim must be 128 on Ascend 950.
* According to the input layouts supported by the operator, the size of the corresponding head dimension in the query tensor shape is denoted as N1, and the size of the corresponding head dimension in the key and value tensor shapes is denoted as N2. N1 % N2 == 0 must be satisfied. MQA, MHA, and GQA scenarios are supported.
* Constraints related to actualSeqLengthsOptional and actualSeqLengthsKvOptional: when qInputLayout or kvInputLayout is not "TND", these two input parameters are ignored.

## Calling Instructions

| Calling Method           | Calling Sample                                                                                                          | Description                                                                                                                  |
|----------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| aclnn calling | [test_aclnn_block_sparse_attention_grad](examples/test_aclnn_block_sparse_attention_grad.cpp)                     | Calls the BlockSparseAttentionGrad operator through the [aclnnBlockSparseAttentionGrad](docs/aclnnBlockSparseAttentionGrad_en.md) API.                   |
| PyTorch API | - | Calls the block_sparse_attention_grad operator through the [torchapi_block_sparse_attention_grad](docs/torchapi_block_sparse_attention_grad.md) API. |
