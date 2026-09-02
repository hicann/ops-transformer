/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_compress_attention_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "op_host/data_copy_transpose_tiling_def.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(NsaInputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, alignedS2);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(int32_t, selectBlockCount);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint8_t, layoutType); // 1: BSH/BSND, 2: SBH, 3: BNSD
TILING_DATA_FIELD_DEF(uint8_t, sparseType);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaInputParamsOp, NsaInputParams)

BEGIN_TILING_DATA_DEF(NsaMultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, reserve);
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaMultiCoreParamsOp, NsaMultiCoreParams)

BEGIN_TILING_DATA_DEF(NsaCoreParams)
TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int32_t, d2Size);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NsaCoreParamsOp, NsaCoreParams)

BEGIN_TILING_DATA_DEF(ImportanceScoreParams)
TILING_DATA_FIELD_DEF(uint64_t, isM);
TILING_DATA_FIELD_DEF(uint64_t, isN);
TILING_DATA_FIELD_DEF(uint64_t, outerLoop);      // s2方向上一次l'切分循环的次数
TILING_DATA_FIELD_DEF(uint64_t, lastOuterLoop);  // s2方向上一次l'切分循环的次数
TILING_DATA_FIELD_DEF(uint64_t, innerLoop);      // s2方向上一次l 切分循环的次数
TILING_DATA_FIELD_DEF(uint64_t, s2ScoreLoopLen); // s2方向一次加载的长度
TILING_DATA_FIELD_DEF(uint64_t, lastLoopLen);    // s2方向加载的尾块大小
TILING_DATA_FIELD_DEF(uint64_t, s2Loop);         // s2方向上循环次数
TILING_DATA_FIELD_DEF(uint64_t, tmpUbSize);
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData);      // api tiling
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingData2);     // api tiling
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingDataLast);  // api tiling
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, confusionTransposeTilingDataLast2); // api tiling
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ImportanceScoreParamsOp, ImportanceScoreParams)

BEGIN_TILING_DATA_DEF(NsaCompressAttentionGeneralTilingData)
TILING_DATA_FIELD_DEF_STRUCT(NsaInputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaMultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(NsaCoreParams, coreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);      // 28 * int
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);      // 28 * int
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData); // 16 * int
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);       // 28 * int32
TILING_DATA_FIELD_DEF_STRUCT(ImportanceScoreParams, importanceScoreParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NsaCompressAttention, NsaCompressAttentionGeneralTilingData)

} // namespace optiling
