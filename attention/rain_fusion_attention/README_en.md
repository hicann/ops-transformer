# RainFusionAttention Operator

English | [简体中文](./README.md)

## Overview

RainFusionAttention is a high-performance sparse attention operator implemented based on the CATLASS template library, and supports flexible block-level sparse patterns. The operator is migrated from the 32_sparse_attention_infer example of catlass and adapted for the ops-transformer-dev framework.

## Features

- **Flexible sparse block pattern**: Supports custom x*y block-level sparse patterns, specified by the blockShape parameter
- **TND format KV Cache**: Supports the KV cache layout in TND [T, N, D] format (note: the BNSD format is not supported yet)
- **Multiple Q input layouts**: Supports two Query input layouts, BSH and TND (note: the BNSD format is not supported yet)
- **High-performance computing**: Built on the CATLASS template library and makes full use of the hardware features of Ascend A2
- **Multiple precision support**: Supports both FP16 and BF16
- **Unaligned sequences**: Supports scenarios where the sequence length is not divisible by the block size, and automatically handles boundary cases

## Interface Definition

### aclnnRainFusionAttentionGetWorkspaceSize

```cpp
aclnnStatus aclnnRainFusionAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *selectIdx,
    const aclTensor *selectNumIdx,
    const aclIntArray *blockShape,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional,
    char *qInputLayout,
    char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t innerPrecise,
    int64_t blockSize,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);                 // Returns executor
```

### aclnnRainFusionAttention

```cpp
aclnnStatus aclnnRainFusionAttention(
    void *workspace,                           // Workspace address
    uint64_t workspaceSize,                    // Workspace size
    aclOpExecutor *executor,                   // executor
    const aclrtStream stream);                 // ACL stream
```

## Parameter Description

### Input Parameters

- **query**: Query tensor, which supports the following layouts
  - BSH: [batch, seqlen, num_heads * head_dim]
  - TND: [total_tokens, num_heads, head_dim]

- **key/value**: KV cache tensor, which supports only the TND layout
  - TND: [total_kv_tokens, num_kv_heads, head_dim]
  - Note: The BNSD format is not supported yet. Convert the data to the TND format if needed.

- **selectIdx**: Sparse block index array [T, headNum, maxKvBlockNum]
  - T: Total number of Q-direction blocks across all batches
  - headNum: Number of heads
  - maxKvBlockNum: Maximum number of KV blocks
  - Stores the indices of the KV blocks selected by each Q block

- **selectNumIdx**: KV block count array [T, headNum]
  - Stores the number of KV blocks actually selected by each Q block

- **blockShape**: Sparse block shape [blockShapeX, blockShapeY]
  - blockShapeX: Block size in the Q direction
  - blockShapeY: Block size in the KV direction

### Output Parameters

- **attentionOut**: Attention output, whose shape is the same as that of query
- **softmaxLse**: Softmax log-sum-exp output (optional)

## Usage Example

```cpp
#include <cmath>

// 1. Prepare the input data
int64_t batch = 1;
int64_t qSeqlen = 256;
int64_t kvSeqlen = 512;
int64_t numHeads = 8;
int64_t numKvHeads = 8;
int64_t headDim = 128;
int64_t blockShapeX = 64;
int64_t blockShapeY = 64;

// 2. Create tensors (omitted)

// 3. Set blockShape
int64_t blockShapeData[2] = {blockShapeX, blockShapeY};
aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);

// 4. Call the operator
char qInputLayout[] = "BSH";
char kvInputLayout[] = "TND";
double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnRainFusionAttentionGetWorkspaceSize(
    query, key, value, selectIdx, selectNumIdx, blockShape,
    nullptr, nullptr, nullptr, nullptr, qInputLayout, kvInputLayout,
    numKvHeads, 0, scaleValue, 0, 128,
    attentionOut, nullptr, &workspaceSize, &executor);

if (ret == ACLNN_SUCCESS) {
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ret = aclnnRainFusionAttention(workspace, workspaceSize, executor, stream);

    aclrtFree(workspace);
}
```

## KV Cache Layout Description

### TND Format [T, N, D] (Currently Supported)

- T: Total number of tokens
- N: Number of KV heads
- D: Head dimension
- Applicable to variable-length sequence scenarios
- **This is the only format currently supported**

### BNSD Format [B, N, S, D] (Not Supported Yet)

- The BNSD format is not implemented yet and is planned to be supported in a future version.
- To use a KV cache in BNSD format, convert it to the TND format first.
- Conversion method: Reshape BNSD [B, N, S, D] to TND [B*S, N, D]

## Sparse Pattern Description

### SelectIdx Index Format

RainFusionAttention uses a new selectIdx index format and does not require the traditional sBlockIdx computation:

- **selectIdx**: Sparse block index array with the shape `[T, headNum, maxKvblockNum]`
- **selectNumIdx**: Sparse block count array with the shape `[T, headNum]`
- **T**: Total number of Q-direction blocks across all batches (calculated by blockShapeX)
- **blockShapeX**: Basic block size in the Q direction
- **blockShapeY**: Basic block size in the KV direction

### Example

For qSeqlen=256, kvSeqlen=512, blockShapeX=64, blockShapeY=64:

```
Q direction: ceil(256/64)=4 blocks [0, 1, 2, 3]
KV direction: ceil(512/64)=8 blocks [0, 1, 2, 3, 4, 5, 6, 7]

Each Q block can select any combination of KV blocks:
- KV blocks selected by Q block 0: [0, 2, 5]
- KV blocks selected by Q block 1: [1, 3, 6]
- KV blocks selected by Q block 2: [0, 4, 7]
- KV blocks selected by Q block 3: [2, 5, 6]
```

## Performance Features

- **Memory efficiency**: Significantly reduces memory usage compared with dense attention
- **Computation efficiency**: Computes only the specified sparse blocks, reducing the computation amount
- **Flexible configuration**: Supports sparse block configurations with any blockShape
- **Hardware optimization**: Makes full use of the AI Core and Vector Core of Ascend A2

## Precautions

1. **KV cache format**: Currently only the TND format is supported, and the BNSD format is not supported yet.
2. **Q input format**: The BSH and TND formats are supported, and the BNSD format is not supported yet.
3. qSeqlen and kvSeqlen do not need to be divisible by blockShape, and unaligned scenarios are supported.
4. The actual number of blocks is calculated by rounding up: `ceil(qSeqlen/blockShapeX)` and `ceil(kvSeqlen/blockShapeY)`.
5. The sparse block indices must be within the valid range.
6. The last block may be incomplete, and the system automatically handles boundary cases.
7. It is recommended to avoid a sparsity value that is too low, so as to maintain computation efficiency.
8. The kvInputLayout parameter must be set to TND.

## Compilation Instructions

The operator compilation is integrated into the unified compilation process of ops-transformer-dev:

```bash
bash build.sh --soc=ascend910b
```

## Version History

- v1.0.0: Initial version, migrated from catlass 32_sparse_attention_infer
