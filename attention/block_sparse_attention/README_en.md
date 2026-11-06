# BlockSparseAttention Operator

English | [简体中文](./README.md)

## Overview

BlockSparseAttention is a high-performance sparse attention operator implemented based on the CATLASS template library. It supports flexible block-level sparse patterns.

## Features

- **Flexible sparse block pattern**: Supports custom x*y block-level sparse patterns, specified by the blockShape parameter
- **Multiple KV Cache layouts**: Supports the TND [T, N, D] and BNSD [B, N, S, D] KV cache layouts
- **Multiple Q input layouts**: Supports two Query input layouts: TND [T, N, D] and BNSD [B, N, S, D]
- **High-performance computing**: Based on the CATLASS template library, making full use of the Ascend A2 hardware features
- **Multi-precision support**: Supports FP16 and BF16 precisions
- **Unaligned sequences**: Supports scenarios where the sequence length is not divisible by the block size, and automatically handles boundary cases

## Interface Definition

### aclnnBlockSparseAttentionGetWorkspaceSize

```cpp
aclnnStatus aclnnBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query,                         // Query input
    const aclTensor *key,                           // Key input
    const aclTensor *value,                         // Value input
    const aclTensor   *blockSparseMaskOptional,     // Sparse mask
    const aclTensor   *attenMaskOptional,           // Attention mask (not supported currently)
    const aclIntArray *blockShapeOptional,          // Sparse block shape array
    const aclIntArray *actualSeqLengthsOptional,    // Actual Q sequence lengths (required for the TND layout)
    const aclIntArray *actualSeqLengthsKvOptional,  // Actual KV sequence lengths (required for the TND layout)
    const aclTensor   *blockTableOptional,          // Block table (used for PagedAttention, not supported currently)
    char *qInputLayout,                             // Query data layout (TND/BNSD)
    char *kvInputLayout,                            // Key/Value data layout (TND/BNSD)
    int64_t numKeyValueHeads,                       // Number of KV heads
    int64_t maskType,                               // Mask type
    double scaleValue,                              // Scale factor
    int64_t innerPrecise,                           // Precision level of the Softmax computation
    int64_t blockSize,                              // Block size (used for PagedAttention, not supported currently)
    int64_t preTokens,                              // Sliding window parameter (not supported currently)
    int64_t nextTokens,                             // Sliding window parameter (not supported currently)
    int64_t softmaxLseFlag,                         // Whether to output LSE
    aclTensor *attentionOut,                        // Output tensor
    aclTensor *softmaxLseOptional,                  // Softmax LSE output (optional)
    uint64_t *workspaceSize,                        // Returns the workspace size
    aclOpExecutor **executor);                      // Returns the executor
```

### aclnnBlockSparseAttention

```cpp
aclnnStatus aclnnBlockSparseAttention(
    void *workspace,                           // Workspace address
    uint64_t workspaceSize,                    // Workspace size
    aclOpExecutor *executor,                   // executor
    aclrtStream stream);                       // ACL stream
```

## Parameter Description

### Input Parameters

- **query**: Query tensor, supports the following layouts
  - TND: [total_tokens, num_heads, head_dim]
  - BNSD: [batch, num_heads, seqlen, head_dim]

- **key**: Key tensor, supports the following layouts
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - BNSD: [batch, num_kv_heads, kv_seqlen, head_dim]

- **value**: Value tensor, supports the following layouts
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - BNSD: [batch, num_kv_heads, kv_seqlen, head_dim]

- **blockSparseMaskOptional**:
  - Sparse mask: [batch, headNum, qBlockNum, kvBlockNum]. For TND, the block numbers are computed from the maximum values of actualSeqLengths/actualSeqLengthsKv; for BNSD/BSND, the block numbers can be computed from S in the Q/K shapes, or from their maximum values when actualSeqLengths/actualSeqLengthsKv are passed in.

- **blockShapeOptional**:
  - blockShapeX: Block size in the Q direction
  - blockShapeY: Block size in the KV direction

- **innerPrecise**: Controls the data types used in the online softmax and rescale computation stages. The value range is limited by the product and the input data type. For details, see the Precautions section.
  - 0: Both online softmax and rescale use FLOAT32.
  - 1: Both online softmax and rescale use FLOAT16; only FLOAT16 input is supported.
  - 4: Online softmax uses FLOAT16/BFLOAT16, and rescale uses FLOAT32.

### Output Parameters

- **attentionOut**: Attention output, with the same shape as query
- **softmaxLse**: Softmax log-sum-exp output (optional)

## Usage Example

```cpp
// 1. Prepare the input data
int64_t batch = 1;
int64_t qSeqlen = 256;
int64_t kvSeqlen = 512;
int64_t numHeads = 8;
int64_t numKvHeads = 8;
int64_t headDim = 128;
int64_t blockShapeX = 128;
int64_t blockShapeY = 128;

// 2. Create tensors (omitted)

// 3. Set blockShape
int64_t blockShapeData[2] = {blockShapeX, blockShapeY};
aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);

// 4. Call the operator
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnBlockSparseAttentionGetWorkspaceSize(
    query, key, value, blockSparseMask, nullptr, blockShape, nullptr,
    nullptr, nullptr, "BNSD", "BNSD", numKvHeads, 0, scaleValue, 0, 128,
    2147483647, 2147483647, 0, attentionOut, nullptr, &workspaceSize, &executor);

if (ret == ACLNN_SUCCESS) {
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnBlockSparseAttention(workspace, workspaceSize, executor, stream);

    aclrtFree(workspace);
}
```

## Sparse Pattern Description

### BlockSparseMask Sparse Pattern

BlockSparseAttention uses the blockSparseMask sparse pattern and does not need the selectIdx index format or the traditional sBlockIdx computation:

- **blockSparseMask**: Sparse mask with the shape `[batch, numHeads, qBlockNum, kvBlockNum]`
- **blockShapeX**: Basic block size in the Q direction
- **blockShapeY**: Basic block size in the KV direction

### Example

For qSeqlen=512, kvSeqlen=1024, blockShapeX=128, and blockShapeY=128:

```
Q direction: ceil(512/128)=4 blocks [0, 1, 2, 3]
KV direction: ceil(1024/128)=8 blocks [0, 1, 2, 3, 4, 5, 6, 7]

Each Q block can select any combination of KV blocks:
- KV blocks selected by Q block 0: [0, 2, 5]
- KV blocks selected by Q block 1: [1, 3, 6]
- KV blocks selected by Q block 2: [0, 4, 7]
- KV blocks selected by Q block 3: [2, 5, 6]
```

## Performance Characteristics

- **Memory efficiency**: Significantly reduces memory usage compared with dense attention
- **Computing efficiency**: Computes only the specified sparse blocks, reducing the computation workload
- **Flexible configuration**: Supports sparse block configurations with any blockShape
- **Hardware optimization**: Makes full use of the AI Core and Vector Core of Ascend A2

## Calling Description

| Invocation Method | Sample Code | Description                                          |
| ---------- | -------- | --------------------------------------------- |
| PyTorch API | - | Calls the block_sparse_attention operator through the [block_sparse_attention](docs/torchapi_block_sparse_attention_en.md) API. |

## Precautions

1. qInputLayout currently supports only "TND" and "BNSD".
2. kvInputLayout currently supports only "TND" and "BNSD".
3. If blockShapeOptional is passed in, it must contain at least two elements [blockShapeX, blockShapeY], and the values must be greater than 0. blockShapeY must be a multiple of 16 on <term>Ascend 950PR&950DT products</term> and a multiple of 128 on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>.
4. qSeqlen and kvSeqlen do not need to be divisible by blockShape. Unaligned scenarios are supported, and the actual number of blocks is computed by rounding up.
5. blockSparseMaskOptional must be passed in currently, with the shape [batch, headNum, qBlockNum, kvBlockNum]. For TND, qBlockNum = ceilDiv(max(actualSeqLengths), blockShapeX), and kvBlockNum = ceilDiv(max(actualSeqLengthsKv), blockShapeY). For BNSD/BSND, qBlockNum = ceilDiv(S in the Q shape, blockShapeX), and kvBlockNum = ceilDiv(S in the K shape, blockShapeY); if actualSeqLengths and actualSeqLengthsKv are passed in, the block numbers can also be computed from the actual lengths: qBlockNum = ceilDiv(max(actualSeqLengths), blockShapeX), and kvBlockNum = ceilDiv(max(actualSeqLengthsKv), blockShapeY).
6. attenMaskOptional currently supports only nullptr.
7. maskType currently supports only 0, which means no mask is added.
8. actualSeqLengthsOptional is required when qInputLayout is "TND"; actualSeqLengthsKvOptional is required when kvInputLayout is "TND".
9. PagedAttention is not supported currently, and blockTableOptional currently supports only nullptr.
10. On <term>Ascend 950PR&950DT products</term>, FLOAT16/BFLOAT16 non-quantized input supports innerPrecise set to 0 or 4, but not 1; on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, FLOAT16 input supports 0 or 1, and BFLOAT16 input supports only 0.

## Compilation Description

Operator compilation is integrated into the unified compilation process of ops-transformer:

```bash
bash build.sh
```

## Version History

- v1.0.0: Initial version
