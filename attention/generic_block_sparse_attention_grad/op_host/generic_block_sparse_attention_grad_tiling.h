/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_SPARSE_ATTENTION_GRAD_TILING_H
#define GENERIC_SPARSE_ATTENTION_GRAD_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GenericBlockSparseAttentionGradTilingDataArch35)
TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, batchNum);
TILING_DATA_FIELD_DEF(uint64_t, qSeqLen);  // TND: T1 total tokens
TILING_DATA_FIELD_DEF(uint64_t, kvSeqLen); // TND: T2 total tokens
TILING_DATA_FIELD_DEF(uint64_t, qGroup);
TILING_DATA_FIELD_DEF(uint64_t, qHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, kvHeadNum);
TILING_DATA_FIELD_DEF(uint64_t, headDim);
TILING_DATA_FIELD_DEF(uint64_t, baseM);
TILING_DATA_FIELD_DEF(uint64_t, baseN);
TILING_DATA_FIELD_DEF(uint64_t, maxS1);
TILING_DATA_FIELD_DEF(uint64_t, numJ);
TILING_DATA_FIELD_DEF(uint64_t, dqSize);
TILING_DATA_FIELD_DEF(uint64_t, dkSize);
TILING_DATA_FIELD_DEF(uint64_t, dqWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dkWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dvWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, sftgWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dqSelWorkspaceOffset);
TILING_DATA_FIELD_DEF(float, softmaxScale);
TILING_DATA_FIELD_DEF(uint32_t, sftgTmpSpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, BlockX);
TILING_DATA_FIELD_DEF(uint32_t, BlockY);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(uint32_t, isPackedGqa);
TILING_DATA_FIELD_DEF(int32_t, windowSizeLeft);
TILING_DATA_FIELD_DEF(int32_t, windowSizeRight);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradFrontTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GenericBlockSparseAttentionGrad, GenericBlockSparseAttentionGradTilingDataArch35)

struct GenericBlockSparseAttentionGradCompileInfo {
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

enum class GsagInputLayout : uint32_t {
    TND = 0,
    BNSD = 1,
    BSND = 2
};

} // namespace optiling

#endif // GENERIC_SPARSE_ATTENTION_GRAD_TILING_H
