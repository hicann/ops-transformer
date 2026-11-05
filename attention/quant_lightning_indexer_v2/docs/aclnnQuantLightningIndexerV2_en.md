# aclnnQuantLightningIndexerV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:40:37.068Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: `QuantLightningIndexerV2` performs the pre-processing computation for sparse attention during inference. It selects the key sparse tokens, and quantizes the input query and key to achieve int8 storage and int8 computation, maximizing performance gains.

- Version evolution: Built on QuantLightningIndexer, it adds capabilities such as the compressed key use case, core-split metadata computation, and sparse value output.

- Formula:

$$
out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
$$

The main computation process is as follows:

1. Multiply the input parameter `query` of a token ($Q_{index}^{Quant}\in\R^{g\times d}$) by the given context `key` ($K_{index}^{Quant}\in\R^{S_{k}\times d}$) to obtain the relevance.
2. Multiply the relevance result by the dequantization coefficients `query_dequant_scale` ($Scale_Q$) and `key_dequant_scale` ($Scale_K^T$) corresponding to `query` and `key`, and filter invalid negative relevance signals through the activation function $ReLU$ to obtain the relevance score vector between the current token and all preceding tokens.
3. Multiply it by the weight coefficient `weights` ($W$), and select the first $Top-k$ index values along the g direction to obtain the output $out$, which serves as the input to Attention.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQuantLightningIndexerV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnQuantLightningIndexerV2" API to execute the computation.

```Cpp
aclnnStatus aclnnQuantLightningIndexerV2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *cuSeqLensQOptional,
    const aclTensor *cuSeqLensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *outputIdxOffsetOptional,
    const aclTensor *metadataOptional,
    int64_t          topk,
    int64_t          quantMode,
    int64_t          maxSeqlenQOptional,
    char            *layoutQOptional,
    char            *layoutKOptional,
    int64_t          maskModeOptional,
    int64_t          cmpRatioOptional,
    int64_t          returnValueOptional,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantLightningIndexerV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuantLightningIndexerV2GetWorkspaceSize

- **Parameters:**

> [!NOTE]
>
> - Parameter dimensions: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the hidden layer size, N (Head Num) indicates the number of heads, and D (Head Dim) indicates the size of the smallest unit in the hidden layer, where D=H/N. T indicates the sum of sequence lengths of input samples across all batches.
> - S1 indicates S in the query shape, S2 indicates S in the key shape, T1 indicates T in the query shape, T2 indicates T in the key shape, N1 indicates N in the query shape, and N2 indicates N in the key shape.
> - maxBlockNumPerSeq indicates the number of blocks corresponding to the maximum sequsedKOptional in each batch, and S2_MAX indicates the maximum value in sequsedKOptional.

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>Quantized query in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT8, FLOAT8_e4m3fn, HIFLOAT8, FLOAT4_e2m1</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layoutQOptional is BSND, the shape is (B,S1,N1,D).</li>
                <li>When layoutQOptional is TND, the shape is (T1,N1,D).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>The quantized Key in the formula.</td>
      <td>          <ul>
                <li>Empty tensors are not supported.</li>
                <li>block_num is the total number of blocks in PageAttention, and block_size is the number of tokens in a block.</li>
          </ul>
      </td>
      <td>INT8, FLOAT8_e4m3fn, HIFLOAT8, FLOAT4_e2m1</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutKOptional is PA_BBND, the shape is (block_num, block_size, N2, D); if BSND, the shape is (B,S2,N2,D); if TND, the shape is (T2,N2,D).</li>
          </ul>
      </td>
      <td>Supports non-contiguous tensors on axis 0</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>Input</td>
      <td>The weight coefficient W in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>          <ul>
                <li>When layoutQOptional is BSND, the shape is (B,S1,N1).</li>
                <li>When layoutQOptional is TND, the shape is (T1,N1).</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>queryDequantScale</td>
      <td>Input</td>
      <td>Dequantization scale of Query in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT32, FLOAT8_e8m0</td>
      <td>ND</td>
      <td>          <ul>
                <li>When quantMode is 3/5, the shape is (B,S1,N1,D/64,2) when layoutQOptional is BSND and (T1,N1,D/64,2) when layoutQOptional is TND.</li>
                <li>When quantMode is 4, the shape is (1,).</li>
                <li>In other use cases, the shape is the same as weights.</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>keyDequantScale</td>
      <td>Input</td>
      <td>The Key dequantization coefficient in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT32, FLOAT8_e8m0</td>
      <td>ND</td>
      <td>          <ul>
                <li>When quantMode is 3/5, the shapes corresponding to layoutKOptional PA_BBND, BSND, and TND are (block_num,block_size,N2,D/64,2), (B,S2,N2,D/64,2), and (T2,N2,D/64,2), respectively.</li>
                <li>When quantMode is 4, the shape is (1,).</li>
                <li>In other use cases, the shapes corresponding to layoutKOptional PA_BBND, BSND, and TND are (block_num,block_size,N2), (B,S2,N2), and (T2,N2), respectively.</li>
          </ul>
      </td>
      <td>Supports non-contiguity on the 0th axis.</td>
    </tr>
    <tr>
      <td>cuSeqLensQOptional</td>
      <td>Input</td>
      <td>Number of valid Query tokens in each batch (the cu_seqlens format is used in TND use cases).</td>
      <td>          <ul>
                <li>When layoutQOptional is TND, this parameter is required. The number of elements is B+1, and the value of each element indicates the total number of tokens in the current batch and all preceding batches (that is, the prefix sum).</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional</td>
      <td>Input</td>
      <td>Number of valid tokens of Key in each Batch (the cu_seqlens format is used in the TND use case).</td>
      <td>          <ul>
                <li>This parameter is required when layoutKOptional is TND.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>Input</td>
      <td>Number of valid tokens of Query in each Batch.</td>
      <td>The number of valid tokens in each Batch of this parameter must not exceed the size of dimension S in query and must not be less than 0.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>Input</td>
      <td>Number of valid tokens of Key in each batch.</td>
      <td>          <ul>
                <li>The number of valid tokens of each batch in this parameter does not exceed the size of dimension S in key and is not less than 0.</li>
                <li>This parameter is required when layoutKOptional is PA_BBND.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>Input</td>
      <td>Residual length of Key in compression use cases.</td>
      <td>Must satisfy 0 <= cmpResidualKOptional[i] < cmpRatioOptional.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Indicates the block mapping table used for KV storage in PageAttention.</td>
      <td>          <ul>
                <li>Empty tensors are not supported.</li>
                <li>In PageAttention use cases, block_table must be two-dimensional. The first dimension length must equal B, and the second dimension length must not be less than maxBlockNumPerSeq.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S2_MAX/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>outputIdxOffsetOptional</td>
      <td>Input</td>
      <td>Offset of the output index.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Shape is (B,S1,N2) when layoutQOptional is BSND and (T1,N2) when layoutQOptional is TND.</td>
      <td>x</td>
    </tr>
    <tr>
      <td>metadataOptional</td>
      <td>Input</td>
      <td>Core splitting information passed in by the QuantLightningIndexerV2Metadata operator.</td>
      <td>          <ul>
                <li>Includes the number of used cores, block size, and the start point of data processed by each core.</li>
                <li>The shape is [8 * (1 + aic_num + aiv_num) + 256], where aic_num and aiv_num are obtained from the current device. The final 256 int32 elements form a shared extension area. Passing an empty value is currently not supported.</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(8 * (1 + aic_num + aiv_num) + 256,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>Input</td>
      <td>Number of key token indexes to retain in the topK stage.</td>
      <td>Supported range: [1, 8192].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Input</td>
      <td>Quantization mode.</td>
      <td>          <ul>
                <li>Values 1 (FLOAT8_e4m3fn quantization), 2 (Per-Token-Head quantization), 3 (MXFP8 quantization), 4 (HIFLOAT8 quantization), and 5 (MXFP4 quantization) are supported.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQOptional</td>
      <td>Input</td>
      <td>Maximum sequence length of Query.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>Input</td>
      <td>Identifies the data layout of input Query.</td>
      <td>          <ul>
                <li>Supports BSND and TND.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>Input</td>
      <td>Identifies the data layout of input Key.</td>
      <td>          <ul>
                <li>Supports PA_BBND, BSND, and TND.</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskModeOptional</td>
      <td>Input</td>
      <td>Specifies the sparse mode.</td>
      <td>          <ul>
                <li>0: defaultMask mode.</li>
                <li>3: mask of the rightDownCausal mode, corresponding to the lower-triangle use case with the right vertex as the division point.</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatioOptional</td>
      <td>Input</td>
      <td>Compression ratio of key.</td>
      <td>          <ul>
                <li>A positive integer within (0, 128].</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnValueOptional</td>
      <td>Input</td>
      <td>Indicates whether to output sparseValuesOut.</td>
      <td>          <ul>
                <li>1: output; 0: no output</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseIndicesOut</td>
      <td>Output</td>
      <td>Indices output in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>          <ul>
                <li>If layoutQOptional is "BSND", the output shape is [B, S1, N2, topk].</li>
                <li>If layoutQOptional is "TND", the output shape is [T1, N2, topk].</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>Output</td>
      <td>Values corresponding to the Indices output in the formula.</td>
      <td>          <ul>
                <li>If returnValueOptional is 1, valid values are output and invalid parts are filled with negative infinity in bf16; if returnValueOptional is 0, an empty tensor with shape (0,) is output.</li>
          </ul>
      </td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>If returnValueOptional is 1, the shape is the same as that of sparseIndicesOut; if returnValueOptional is 0, the shape is (0,).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:
  - `layoutKOptional` additionally supports BSND and TND; PA_BBND, BSND, and TND are supported.
  - `quantMode` supports 1 (FLOAT8_e4m3fn quantization), 2 (INT8 quantization), 3 (MXFP8 quantization), 4 (HIFLOAT8 quantization), and 5 (MXFP4 quantization).
  - `cmpRatioOptional` supports any positive integer in (0, 128].
  - `returnValueOptional` is supported.
  - query and key: FLOAT8_e4m3fn is supported when `quantMode` is 1/3, INT8 is supported when `quantMode` is 2, HIFLOAT8 is supported when `quantMode` is 4, and FLOAT4_e2m1 is supported when `quantMode` is 5.
  - queryDequantScale and keyDequantScale: FLOAT32 is supported when `quantMode` is 1/4, FLOAT16 is supported when `quantMode` is 2, and FLOAT8_e8m0 is supported when `quantMode` is 3/5.
  - weights: FLOAT16 is supported when `quantMode` is 2, and FLOAT32 is supported when `quantMode` is 1/3/4/5.
  - query: N1 supports [1, 64]; key: N2 supports only 1.
  - cuSeqLensQOptional cannot be passed in when layoutQOptional is BSND; cuSeqLensKOptional cannot be passed in when layoutKOptional is BSND or PA_BBND.
  - When layoutKOptional is PA_BBND, blockTableOptional must be passed in; otherwise, it must not be passed in.
<!-- end id10 -->
<!-- npu="A3,910b" id11 -->
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - `layoutKOptional` supports only PA_BBND.
  - `quantMode` supports only 2 (Per-Token-Head quantization).
  - `cmpRatioOptional` supports only powers of 2 in the range of [1, 128], namely 1/2/4/8/16/32/64/128.
  - `outputIdxOffsetOptional` is not supported.
  - `returnValueOptional` is not supported.
  - query and key: INT8 is supported, while FLOAT8_e4m3fn, HIFLOAT8, and FLOAT4_e2m1 are not supported.
  - queryDequantScale and keyDequantScale: FLOAT16 is supported, while FLOAT32 and FLOAT8_e8m0 are not supported.
  - weights: FLOAT16 is supported, while FLOAT32 is not supported.
  - query N1 supports only 64.
  - topk supports only [1, 2048].
<!-- end id11 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>If a passed-in parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data types and data formats of query, key, weights, queryDequantScale, keyDequantScale, cuSeqLensQOptional, cuSeqLensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, blockTableOptional, metadataOptional, layoutQOptional, layoutKOptional, topk, quantMode, maskModeOptional, cmpRatioOptional, returnValueOptional, sparseIndicesOut, and sparseValuesOut are not within the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnQuantLightningIndexerV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnQuantLightningIndexerV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- headdim supports 128.
- block_size must be a multiple of 16, with a maximum supported value of 1024.
- When `layoutKOptional` is not PA_BBND, `layoutQOptional` and `layoutKOptional` must be consistent.
- When `quantMode` is 3/5, `queryDequantScale` and `keyDequantScale` have 1 more dimension than `query` and `key` respectively, with consistent prefix dimensions and the last two dimensions of (D/64, 2); D must be a multiple of 64, and each scale corresponds to 32 consecutive logical elements along the D dimension.
- When the input parameter layoutQOptional is TND, cuSeqLensQOptional must be passed in. If sequsedQOptional is also passed in, ensure that the query length of each batch passed in by sequsedQOptional does not exceed the q sequence length of each batch calculated based on cuSeqLensQOptional. When the q sequence length seqlen1 passed in by sequsedQOptional for a batch is less than the query length seqlen2 calculated based on cuSeqLensQOptional, the TND Padding feature is enabled, and the sparseIndices and sparseValues of the query output for the difference between seqlen2 and seqlen1 in that batch are all set to invalid values. In some long-sequence use cases, if too much invalid data needs to be padded, aicore execution may time out due to hardware limitations. Calculate the amount of data to be padded using (seqlen2 - seqlen1) * topk. It is recommended to keep this amount within 400 million.
- **Deterministic computation:** aclnnQuantLightningIndexerV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_lightning_indexer_v2.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_lightning_indexer_v2.h"
#include "aclnn/opdev/platform.h"

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
    void* weightsDeviceAddr = nullptr;
    void* qScaleDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* metadataDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* qScaleTensor = nullptr;
    aclTensor* kScaleTensor = nullptr;
    aclTensor* metadataTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t S2 = 8;
    int64_t N1 = 64;
    int64_t N2 = 1;
    int64_t D = 128;
    int64_t topk = 512;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> qScaleShape = {B, S1, N1};
    std::vector<int64_t> kScaleShape = {B, S2, N2};
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_CUBE_CORE, &aicCoreNum) != ACL_SUCCESS) {
        LOG_PRINT("Failed to query AIC count\n");
        return ACL_ERROR_INVALID_PARAM;
    }
    if (aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreNum) != ACL_SUCCESS) {
        LOG_PRINT("Failed to query AIV count\n");
        return ACL_ERROR_INVALID_PARAM;
    }
    std::vector<int64_t> metadataShape = {8LL * (1 + aicCoreNum + aivCoreNum) + 256LL};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t qScaleShapeSize = GetShapeSize(qScaleShape);
    int64_t kScaleShapeSize = GetShapeSize(kScaleShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint8_t> queryHostData(queryShapeSize, 0x38);
    std::vector<uint8_t> keyHostData(keyShapeSize, 0x38);
    std::vector<float> weightsHostData(weightsShapeSize, 0.01f);
    std::vector<float> qScaleHostData(qScaleShapeSize, 1.0f);
    std::vector<float> kScaleHostData(kScaleShapeSize, 1.0f);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<uint16_t> sparseValuesHostData(sparseValuesShapeSize, 0);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT8_E4M3FN, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT8_E4M3FN, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(qScaleHostData, qScaleShape, &resources.qScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.qScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(kScaleHostData, kScaleShape, &resources.kScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.kScaleTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(metadataHostData, metadataShape, &resources.metadataDeviceAddr,
                          aclDataType::ACL_INT32, &resources.metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_BF16, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    return ACL_SUCCESS;
}

int ExecuteQuantLightningIndexerV2(TensorResources& resources, aclrtStream stream,
                                   void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 512;
    int64_t quantMode = 1;
    int64_t maskMode = 0;
    int64_t cmpRatio = 1;
    int64_t returnValue = 1;
    constexpr const char layoutQStr[] = "BSND";
    constexpr const char layoutKStr[] = "BSND";
    constexpr size_t layoutQLen = sizeof(layoutQStr);
    constexpr size_t layoutKLen = sizeof(layoutKStr);
    char layoutQ[layoutQLen];
    char layoutK[layoutKLen];
    errno_t memcpyRet = memcpy_s(layoutQ, sizeof(layoutQ), layoutQStr, layoutQLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQ failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutK, sizeof(layoutK), layoutKStr, layoutKLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutK failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnQuantLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        resources.qScaleTensor, resources.kScaleTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        resources.metadataTensor,
        topk, quantMode, -1, layoutQ, layoutK, maskMode, cmpRatio, returnValue,
        resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnQuantLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return ret;
  }
  LOG_PRINT("sparse_indices result (first 10 elements):\n");
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("  [%ld] = %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) { aclDestroyTensor(resources.queryTensor); }
    if (resources.keyTensor) { aclDestroyTensor(resources.keyTensor); }
    if (resources.weightsTensor) { aclDestroyTensor(resources.weightsTensor); }
    if (resources.qScaleTensor) { aclDestroyTensor(resources.qScaleTensor); }
    if (resources.kScaleTensor) { aclDestroyTensor(resources.kScaleTensor); }
    if (resources.metadataTensor) { aclDestroyTensor(resources.metadataTensor); }
    if (resources.sparseIndicesTensor) { aclDestroyTensor(resources.sparseIndicesTensor); }
    if (resources.sparseValuesTensor) { aclDestroyTensor(resources.sparseValuesTensor); }

    if (resources.queryDeviceAddr) { aclrtFree(resources.queryDeviceAddr); }
    if (resources.keyDeviceAddr) { aclrtFree(resources.keyDeviceAddr); }
    if (resources.weightsDeviceAddr) { aclrtFree(resources.weightsDeviceAddr); }
    if (resources.qScaleDeviceAddr) { aclrtFree(resources.qScaleDeviceAddr); }
    if (resources.kScaleDeviceAddr) { aclrtFree(resources.kScaleDeviceAddr); }
    if (resources.metadataDeviceAddr) { aclrtFree(resources.metadataDeviceAddr); }
    if (resources.sparseIndicesDeviceAddr) { aclrtFree(resources.sparseIndicesDeviceAddr); }
    if (resources.sparseValuesDeviceAddr) { aclrtFree(resources.sparseValuesDeviceAddr); }

    if (workspaceAddr) { aclrtFree(workspaceAddr); }
    if (stream) { aclrtDestroyStream(stream); }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        return 0;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 4;
    int64_t N2 = 1;
    int64_t topk = 512;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteQuantLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteQuantLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
