/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_TILING_H
#define BLOCK_SPARSE_ATTENTION_GRAD_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(BlockSparseAttentionGradTilingDataArch35)
TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, batchNum);
TILING_DATA_FIELD_DEF(uint64_t, qSeqLen);
TILING_DATA_FIELD_DEF(uint64_t, kvSeqLen);
TILING_DATA_FIELD_DEF(uint64_t, qGroup);
TILING_DATA_FIELD_DEF(uint64_t, qHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, headDim);
TILING_DATA_FIELD_DEF(uint64_t, baseM);
TILING_DATA_FIELD_DEF(uint64_t, baseN);
TILING_DATA_FIELD_DEF(uint64_t, singleM);
TILING_DATA_FIELD_DEF(uint64_t, dqSize);
TILING_DATA_FIELD_DEF(uint64_t, dkSize);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dvWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, sftgWorkspaceOffset);
TILING_DATA_FIELD_DEF(float, softmaxScale);
TILING_DATA_FIELD_DEF(uint32_t, sftgTmpSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, BlockX);
TILING_DATA_FIELD_DEF(uint32_t, BlockY);
TILING_DATA_FIELD_DEF(uint8_t, hasPerBlockSizeMask); // maskType=1 时为 1，否则为 0
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNum);        // max(ceilDiv(maxQSeq, blockX), ceilDiv(maxKvSeq, blockY))
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradFrontTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1000, BlockSparseAttentionGradTilingDataArch35)
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1001, BlockSparseAttentionGradTilingDataArch35)
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1002, BlockSparseAttentionGradTilingDataArch35)
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1003, BlockSparseAttentionGradTilingDataArch35)
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1004, BlockSparseAttentionGradTilingDataArch35)
REGISTER_TILING_DATA_CLASS(BlockSparseAttentionGrad_1005, BlockSparseAttentionGradTilingDataArch35)
// BlockSparseAttentionGrad编译信息
struct BlockSparseAttentionGradCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;

    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

} // namespace optiling

#endif // BLOCK_SPARSE_ATTENTION_GRAD_TILING_H
