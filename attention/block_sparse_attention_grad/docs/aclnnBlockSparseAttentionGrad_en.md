# BlockSparseAttentionGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:19:44.459Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Not supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="310b" id6 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id7 -->
<!-- npu="910" id8 -->
- <term>Atlas training products</term>: Not supported
<!-- end id8 -->

## Function

- Description: Computes the backward pass of block-sparse attention. It supports flexible block-level sparsity patterns and uses BlockSparseMask to specify the KV blocks selected for each Q block.
- Formula:

  The sparse block size is $blockShapeX×blockShapeY$, where blockShapeX indicates the sparse granularity in the Q direction and blockShapeY indicates the sparse granularity in the KV direction. The granularities of the two directions are independent of each other and can be set to different values.

  blockShapeX and blockShapeY must be positive. On Ascend 950, both must be multiples of 16. Supported granularities below 128 include 16, 32, 48, 64, 80, 96, and 112. For example, [32, 64] indicates that one block is divided for every 32 tokens in the Q direction and one block for every 64 tokens in the KV direction.

  BlockSparseMask defines the sparse pattern at the logical block granularity. Each mask element controls whether a contiguous block of $blockShapeX×blockShapeY$ participates in the computation.
  The forward computation formula is as follows:

  $$
  attentionOut=Softmax(Mask(scale⋅query⋅key_{sparse}^{T},  atten\_mask))⋅value_{sparse}
  $$

  For ease of expression, variables $S$ and $P$ are used to represent the formulas:

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

  The backward computation formula is:

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

The data layouts of the BlockSparseAttentionGrad inputs dout, query, key, value, and attentionOut can be interpreted from multiple dimension layouts and passed in through qInputLayout and kvInputLayout. To facilitate understanding of the specific supported layouts (such as BNSD, TND, etc.), the dimension meanings represented by each abbreviation letter in the layouts are uniformly explained as follows:

- B: Indicates the batch size of the input samples (Batch).
- T: Indicates the length of B and S concatenated along the axis and tightly packed (Total tokens).
- S: Indicates the sequence length of the input samples (Seq-Length).
- H: Indicates the size of the hidden layer (Head-Size).
- N: Number of heads (Head-Num).
- D: Minimum unit size of the hidden layer, satisfying D = H/N (Head-Dim).

qInputLayout and kvInputLayout must be the same. Supported layouts by product:

- <term>Atlas A2 series</term> and <term>Atlas A3 series</term>: BNSD and TND.
- <term>Ascend 950PR/950DT series</term>: BNSD, TND, and BSND.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBlockSparseAttentionGradGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation process. Then call the "aclnnBlockSparseAttentionGrad" API to perform the computation.

```Cpp
aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(
  const aclTensor   *dout,
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *attentionOut,
  const aclTensor   *softmaxLse,
  const aclTensor   *blockSparseMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *blockShapeOptional,
  const aclIntArray *actualSeqLengthsOptional,
  const aclIntArray *actualSeqLengthsKvOptional,
  char              *qInputLayout,
  char              *kvInputLayout,
  int64_t            numKeyValueHeads,
  int64_t            maskType,
  double             scaleValue,
  int64_t            preTokens,
  int64_t            nextTokens,
  aclTensor         *dq,
  aclTensor         *dk,
  aclTensor         *dv,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnBlockSparseAttentionGrad(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnBlockSparseAttentionGradGetWorkspaceSize

## Parameters

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
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (shape)</th>
    <th>Non-contiguous Tensor</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>dout (aclTensor*)</td>
    <td>Input</td>
    <td>Backward output gradient, representing the gradient information of the final output with respect to the current operator.</td>
    <td>Does not support empty tensors.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>query (aclTensor*)</td>
    <td>Input</td>
    <td>Query vector in the attention computation, that is, the query in the formula.</td>
    <td>Does not support empty tensors.<br>Supported shapes are: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>key (aclTensor*)</td>
    <td>Input</td>
    <td>Key vector in the attention computation, that is, key in the formula.</td>
    <td>Does not support empty tensors.<br>Supported shapes are: <ul><li>TND: [totalKTokens, numKeyValueHeads, headDim].</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>value (aclTensor*)</td>
    <td>Input</td>
    <td>Value vector in the attention computation, that is, value in the formula.</td>
    <td>Does not support empty tensors.<br>Supported shapes are:<ul><li>TND: [totalVTokens, numKeyValueHeads, headDim].</li><li>BNSD: [batch, numKeyValueHeads, maxKvSeqLength, headDim].</li><li>BSND: [batch, maxKvSeqLength, numKeyValueHeads, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attentionOut (aclTensor*)</td>
    <td>Input</td>
    <td>Output result of the forward BlockSparseAttention computation, that is, attentionOut in the formula.</td>
    <td>Does not support empty tensors.<br>Supported shapes: <ul><li>TND: [totalQTokens, headNum, headDim].</li><li>BNSD: [batch, headNum, maxQSeqLength, headDim].</li><li>BSND: [batch, maxQSeqLength, headNum, headDim].</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>softmaxLse (aclTensor*)</td>
    <td>Input</td>
    <td>Intermediate log-sum-exp result of the Softmax computation. Used for the log and exp inverse derivation in backward gradient computation.</td>
    <td>Does not support empty tensors.<br>The supported shapes are:<ul><li>TND: [totalQTokens, headNum, 1].</li><li>BNSD: [batch, headNum, maxQSeqLength, 1].</li><li>BNS1: [batch, maxQSeqLength, headNum, 1].</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>blockSparseMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Block sparse mask that indicates the actual sparse pattern. Determines which blocks actually participate in the attention computation.</td>
    <td>Does not support empty tensors.<br>Optional input (required in the current version):<ul><li>The shape is [batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]. The 3rd dimension is calculated independently based on the blockShapeX granularity in the Q direction, and the 4th dimension is calculated independently based on the blockShapeY granularity in the KV direction. The two values can be different.</li><li>Indicates, after block division, which blocks need to participate in the computation (value 1) and which blocks do not (value 0).</li><li>When the sequence length is not an integer multiple of the blockShape in the corresponding direction, the last logical block covers the remaining valid tokens in that direction.</li><li>If nullptr is passed in, block sparse computation is disabled, which means the attention scores between all tokens are computed.</li></ul></td>
    <td>UINT8</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
    </tr>
    <tr>
    <td>attenMaskOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Attention mask, that is, atten_mask in the formula. Used to mask the specific tokens that should not participate in the computation.</td>
    <td>For maskType=0, pass nullptr. For maskType=1 (A2/A3 only), this input is required and must be an INT32 tensor with shape [batch, headNum, maxBlockNum, 2], where maxBlockNum = max(ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)). The last dimension stores the actual Q and KV lengths in each block; each length must be between 0 and the corresponding blockShape value.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>4</td>
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
    <td>Array of actual sequence lengths of query.<br>Describes the quantity of valid query tokens in each Batch in a variable-length sequence use case (that is, a use case containing Padding data).</td>
    <td> Variable-length sequence use case (when qInputLayout is "TND"): this input must be configured. Because the TND format is a one-dimensional contiguous layout, the operator relies on this array to accurately split and delimit the actual boundaries of each sequence.</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>actualSeqLengthsKvOptional(aclIntArray*)</td>
    <td>Input</td>
    <td>Array of actual sequence lengths of key/value.<br>Describes the quantity of actually valid key/value tokens in each Batch in a variable-length sequence use case (that is, a use case containing Padding data).</td>
    <td> Variable-length sequence use case (when kvInputLayout is "TND"): this input must be configured. The TND format is a one-dimensional contiguous data layout, so the operator relies on this array to accurately split and define the actual boundaries of each sequence.</td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
    </tr>
    <tr>
    <td>qInputLayout(char*)</td>
    <td>Input</td>
    <td>Data layout format of query. Indicates the specific layout of the input tensor in memory (such as contiguous or axis-fused arrangement).</td>
    <td>Must match the other layout parameter. Atlas A2/A3 supports BNSD and TND. Ascend 950 also supports BSND.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>kvInputLayout (char*)</td>
    <td>Input</td>
    <td>Data layout format of key and value. Indicates the specific layout of the input tensor in memory.</td>
    <td>Must match the other layout parameter. Atlas A2/A3 supports BNSD and TND. Ascend 950 also supports BSND.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numKeyValueHeads (int64_t)</td>
    <td>Input</td>
    <td>Number of attention heads of key/value. Used to support the head quantity ratio mapping under the GQA (grouped query attention) mechanism.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>maskType (int64_t)</td>
    <td>Input</td>
    <td>Mask type in the attention computation. Specifies the preset masking rule to use.</td>
    <td>Supports 0 and 1:<ul><li>0: no attention mask; attenMaskOptional must be nullptr.</li><li>1: PerBlockSizeMask on A2/A3; attenMaskOptional is required.</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scaleValue (double)</td>
    <td>Input</td>
    <td>Scaling factor, which is scale in the formula. Used to normalize attention scores.</td>
    <td>Generally set to D^-0.5.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>preTokens (int64_t)</td>
    <td>Input</td>
    <td>Number of tokens included in the forward sliding window. Limits attention computation of the current token to only a specified number of preceding historical tokens.</td>
    <td>Used for the sliding-window attention use case. Sliding-window attention is currently not supported; only 2147483647 can be passed in.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>nextTokens (int64_t)</td>
    <td>Input</td>
    <td>Number of tokens included in the backward sliding window. Limits attention computation of the current token to only a specified number of following future tokens.</td>
    <td>Used for sliding window attention use cases. Sliding window attention is currently not supported; only 2147483647 can be passed in.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dq (aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output result of query, that is, dq in the formula.</td>
    <td>Does not support empty tensors.<br>Data type and shape remain consistent with the input query.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dk (aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output result of key, that is, dk in the formula.</td>
    <td>Does not support empty tensors.<br>The data type and shape remain consistent with the input key.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>dv(aclTensor*)</td>
    <td>Output</td>
    <td>Gradient output of value, that is, dv in the formula.</td>
    <td>Does not support empty tensors.<br>The data type and shape remain consistent with the input value.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>√</td>
    </tr>
    <tr>
    <td>workspaceSize(uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size that needs to be applied on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor (aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation flow of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
</tbody>
</table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required tensor input or blockShapeOptional is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of dout, query, key, or value is outside the supported range.</td>
    </tr>
    <tr>
      <td>TND layout requires both actualSeqLengthsOptional and actualSeqLengthsKvOptional.</td>
    </tr>
    <tr>
      <td>qInputLayout or kvInputLayout is invalid, the layouts differ, or BSND is used on Atlas A2/A3.</td>
    </tr>
    <tr>
      <td>blockShapeOptional has fewer than two elements, either block size is nonpositive, or either block size is not a multiple of 16 on Ascend 950.</td>
    </tr>
  </tbody></table>

## aclnnBlockSparseAttentionGrad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnBlockSparseAttentionGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the AscendCL stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- actualSeqLengthsOptional is mandatory when qInputLayout is "TND"; actualSeqLengthsKvOptional is mandatory when kvInputLayout is "TND".
- The layout of softmaxLse must remain consistent with the layout of query. If the layout of query is "BSND", pass "BNS1" as the layout of softmaxLse.
- HeadDim must be 128 on Ascend 950.
- blockShapeX and blockShapeY in blockShapeOptional indicate the sparse granularity in the Q direction and the KV direction, respectively. Both values must be positive and may differ. On Ascend 950, both must also be multiples of 16. The number of Q blocks and KV blocks in BlockSparseMask should be calculated based on the sparse granularity in the corresponding direction, respectively.
- Based on the input layouts supported by the operator, the head dimension size in the query tensor shape is denoted as N1, and the head dimension sizes in the key and value tensor shapes are denoted as N2. N1 % N2 == 0 must be satisfied.
  - <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: Only MHA is currently supported, that is, N1 equals N2.

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: MQA, MHA, and GQA use cases are supported.
  <!-- end id9 -->

- Constraints related to actualSeqLengthsOptional and actualSeqLengthsKvOptional:

  - <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>: When qInputLayout or kvInputLayout is "BNSD", if this parameter is configured, the operator processes data based on the specified valid lengths and ignores the padding part to improve performance; if not configured (nullptr is passed), the operator by default uses the S dimension in the query shape as the valid length and processes the full data.

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: When qInputLayout or kvInputLayout is not "TND", these two input parameters are ignored.

  <!-- end id10 -->

- Deterministic computation use cases are not supported.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_block_sparse_attention_grad.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate: AscendCL initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    // Check whether the shape is valid.
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // Check whether the hostData size matches.
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // Call aclrtMalloc to allocate memory on the device side.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. Initialize the device/stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Set the core parameters (taking BNSD Layout as an example).
    int32_t batch = 1;
    int32_t numHeads = 1;
    int32_t numKvHeads = 1;
    int32_t qSeqlen = 161;
    int32_t kvSeqlen = 161;
    int32_t headDim = 128;
    // The Q/KV directions use different sparse granularities, both less than 128 and aligned to 16.
    int32_t blockShapeX = 48;
    int32_t blockShapeY = 80;

    // Calculate the quantity of blocks.
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilKv = (kvSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. Construct the tensor Shape.
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kvShape = {batch, numKvHeads, kvSeqlen, headDim};
    std::vector<int64_t> lseShape = {batch, numHeads, qSeqlen}; // LSE usually has no trailing 1-dim, to prevent GE squeeze.
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilKv};

    // 4. Allocate and initialize Host data.
    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kvShape);

    // Initialize Q, K, V to small values such as 0.1f.
    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kvSize, 0.1f);
    std::vector<op::fp16_t> vData(kvSize, 0.1f);

    // The initial gradient value can be set to a small positive number.
    std::vector<op::fp16_t> doutData(qSize, 0.01f);
    std::vector<op::fp16_t> outData(qSize, 0.1f);

    // Set LSE to a reasonable positive value, such as 5.0f, so that exp(S - LSE) is a very safe negative exponent and will never overflow.
    std::vector<float> lseData(GetShapeSize(lseShape), 5.0f);
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 1);

    // Create all forward input/output aclTensors.
    void *qAddr = nullptr, *kAddr = nullptr, *vAddr = nullptr;
    void *doutAddr = nullptr, *outAddr = nullptr;
    void *lseAddr = nullptr, *maskAddr = nullptr;

    aclTensor *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *doutTensor = nullptr, *outTensor = nullptr;
    aclTensor *lseTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kvShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(vData, kvShape, &vAddr, aclDataType::ACL_FLOAT16, &vTensor);
    CreateAclTensor(doutData, qShape, &doutAddr, aclDataType::ACL_FLOAT16, &doutTensor);
    CreateAclTensor(outData, qShape, &outAddr, aclDataType::ACL_FLOAT16, &outTensor);

    CreateAclTensor(lseData, lseShape, &lseAddr, aclDataType::ACL_FLOAT, &lseTensor);     // Strictly use FP32.
    CreateAclTensor(maskData, maskShape, &maskAddr, aclDataType::ACL_UINT8, &maskTensor); // Strictly use UINT8.

    // 5. Create backward output gradients (dq, dk, dv).
    std::vector<op::fp16_t> dqData(qSize, 0.0f);
    std::vector<op::fp16_t> dkData(kvSize, 0.0f);
    std::vector<op::fp16_t> dvData(kvSize, 0.0f);

    void *dqAddr = nullptr, *dkAddr = nullptr, *dvAddr = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;

    CreateAclTensor(dqData, qShape, &dqAddr, aclDataType::ACL_FLOAT16, &dqTensor);
    CreateAclTensor(dkData, kvShape, &dkAddr, aclDataType::ACL_FLOAT16, &dkTensor);
    CreateAclTensor(dvData, kvShape, &dvAddr, aclDataType::ACL_FLOAT16, &dvTensor);

    // 6. Create aclIntArray attribute parameters (BlockShape & ActualSeqLengths).
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    std::vector<int64_t> qSeqLenVec(batch, static_cast<int64_t>(qSeqlen));
    std::vector<int64_t> kvSeqLenVec(batch, static_cast<int64_t>(kvSeqlen));
    aclIntArray *qSeqLenArr = aclCreateIntArray(qSeqLenVec.data(), batch);
    aclIntArray *kvSeqLenArr = aclCreateIntArray(kvSeqLenVec.data(), batch);

    // 7. Scalar and string parameter configuration
    char qLayoutBuffer[16] = "BNSD";
    char kvLayoutBuffer[16] = "BNSD";
    int64_t maskType = 0;
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    // Forcibly specify the maximum sliding window value
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 2147483647;

    // 8. Call the first-phase API: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBlockSparseAttentionGradGetWorkspaceSize...\n");
    ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
        doutTensor,
        qTensor,
        kTensor,
        vTensor,
        outTensor,
        lseTensor,
        maskTensor,                 // blockSparseMaskOptional
        nullptr,                    // attenMaskOptional must be empty
        blockShapeArr,
        qSeqLenArr,
        kvSeqLenArr,
        qLayoutBuffer,
        kvLayoutBuffer,
        static_cast<int64_t>(numKvHeads),
        maskType,
        scaleValue,
        preTokens,
        nextTokens,
        dqTensor,
        dkTensor,
        dvTensor,
        &workspaceSize,
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 9. Allocate the workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 10. Call the second-phase API: execute the computation
    LOG_PRINT("Calling aclnnBlockSparseAttentionGrad...\n");
    ret = aclnnBlockSparseAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockSparseAttentionGrad failed. ERROR: %d\n", ret); return ret);

    // 11. Synchronize the Stream and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 12. Copy the result back to the Host side and print it
    ret = aclrtMemcpy(dqData.data(), qSize * sizeof(op::fp16_t), dqAddr, qSize * sizeof(op::fp16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! Output results (first 10 elements of dQ):\n");
    for (uint64_t i = 0; i < 10 && i < dqData.size(); i++) {
        LOG_PRINT("  dQ index %lu: %f\n", i, static_cast<float>(dqData[i]));
    }

    // 13. Release all resources
    LOG_PRINT("Cleaning up resources...\n");
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(doutAddr); aclrtFree(outAddr);
    aclrtFree(lseAddr); aclrtFree(maskAddr);
    aclrtFree(dqAddr); aclrtFree(dkAddr); aclrtFree(dvAddr);

    aclDestroyTensor(qTensor); aclDestroyTensor(kTensor); aclDestroyTensor(vTensor);
    aclDestroyTensor(doutTensor); aclDestroyTensor(outTensor);
    aclDestroyTensor(lseTensor); aclDestroyTensor(maskTensor);
    aclDestroyTensor(dqTensor); aclDestroyTensor(dkTensor); aclDestroyTensor(dvTensor);

    aclDestroyIntArray(blockShapeArr);
    aclDestroyIntArray(qSeqLenArr);
    aclDestroyIntArray(kvSeqLenArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BlockSparseAttentionGrad Test completed successfully!\n");
    return 0;
}
```
