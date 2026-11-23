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
 * \file fia_arch38_tiling_common.h
 * \brief FIA arch38 本地化的 tiling 常量/枚举/中间数据结构（从 PFA 解耦，自持定义，不共用 common 头）
 */

#ifndef FIA_ARCH38_TILING_COMMON_H
#define FIA_ARCH38_TILING_COMMON_H

#include <cstddef>
#include <cstdint>
#include "op_host/data_copy_transpose_tiling_def.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
namespace arch38 {

constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t UINT8SIZE = 1;
constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t FLOAT8SIZE = 1;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t FLOAT32SIZE = 4;
constexpr uint32_t BOOLSIZE = 1;

constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t FIRST_DIM = 0;
constexpr uint32_t SECOND_DIM = 1;
constexpr uint32_t THIRD_DIM = 2;
constexpr uint32_t FOURTH_DIM = 3;
constexpr uint32_t FIFTH_DIM = 4;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr size_t DIM_NUM_1 = 1;
constexpr size_t DIM_NUM_3 = 3;
constexpr size_t DIM_NUM_4 = 4;
constexpr size_t DIM_NUM_5 = 5;
constexpr uint32_t N_SIZE_2 = 2;
constexpr uint32_t N_SIZE_4 = 4;
constexpr uint32_t N_SIZE_8 = 8;
constexpr uint32_t N_SIZE_16 = 16;
constexpr uint32_t N_SIZE_32 = 32;
constexpr uint32_t N_SIZE_64 = 64;
constexpr uint32_t N_SIZE_128 = 128;
constexpr uint32_t D_SIZE_192 = 192;
constexpr uint32_t D_SIZE_128 = 128;
constexpr uint32_t D_SIZE_64 = 64;
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;
constexpr int APPROXIMATE_COMPUTATION = 4;
constexpr uint32_t MSD_HIGH_PERFORMANCE_EXPEND_NUM = 2;
constexpr uint32_t MSD_HIGH_PRECISION_EXPEND_NUM = 3;

constexpr uint32_t MAX_BATCH = 256U;

constexpr int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
constexpr int64_t BALANCE_LOAD_LIST_SIZE = 8L;
constexpr int64_t COF[BALANCE_LOAD_LIST_SIZE] = {256, 384, 512, 640, 768, 896, 960, 1024};

enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    BNSD_BSND,
    TND,
    NTD,
    NTD_TND,
    NZ,
    BBH,
    BNBD,
    NONE,
};

enum class TilingMod {
    CVSAME = 0,
    CVDIFF,
    CVDIFF_BASE_API,
    CVDIFF_MLA,
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
    SPLIT_S1OUT_CUBE,
};

BEGIN_TILING_DATA_DEF(FiaArch38AttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint64_t, headNumSize);
TILING_DATA_FIELD_DEF(uint32_t, seqSize);
TILING_DATA_FIELD_DEF(uint64_t, headSize);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int32_t, preTokens);
TILING_DATA_FIELD_DEF(int32_t, nextTokens);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
TILING_DATA_FIELD_DEF(int32_t, PABlockNumSum);
TILING_DATA_FIELD_DEF(uint32_t, dimNumOfseq);
TILING_DATA_FIELD_DEF(uint32_t, typeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, seqInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, usePseShift);
TILING_DATA_FIELD_DEF(uint32_t, useMask);
TILING_DATA_FIELD_DEF(uint64_t, headNumRatio);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskElemType);
TILING_DATA_FIELD_DEF(uint32_t, pseShiftTypeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, pseMaskMaxSize);
TILING_DATA_FIELD_DEF(uint32_t, maskTypeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, outputTypeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, softmaxTypeByteNum);
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, alignedHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, splitS2);
TILING_DATA_FIELD_DEF(uint32_t, splitD);
TILING_DATA_FIELD_DEF(uint32_t, layoutType);
TILING_DATA_FIELD_DEF(uint32_t, PAlayoutType);
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1Size);
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2Size);
TILING_DATA_FIELD_DEF(uint32_t, maskKVsSize);
TILING_DATA_FIELD_DEF(uint32_t, maskQsSize);
TILING_DATA_FIELD_DEF(uint32_t, isLayoutSH);
TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsNull);
TILING_DATA_FIELD_DEF(uint32_t, isActualSeqLengthsKVNull);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
TILING_DATA_FIELD_DEF(uint32_t, deqScaleFlag);
TILING_DATA_FIELD_DEF(uint32_t, deqScale2Flag);
TILING_DATA_FIELD_DEF(uint32_t, isAntiPerchannel);
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid);
TILING_DATA_FIELD_DEF(uint32_t, softmaxOuterSize);
TILING_DATA_FIELD_DEF(uint32_t, isQuant2Perchannel);
TILING_DATA_FIELD_DEF(uint32_t, isQuant2BF16);
TILING_DATA_FIELD_DEF(uint32_t, isKvContinuous);
TILING_DATA_FIELD_DEF(uint32_t, fromFused);
TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut);
TILING_DATA_FIELD_DEF(uint32_t, isIFA);
TILING_DATA_FIELD_DEF(uint32_t, isSoftMaxLseEnable);
TILING_DATA_FIELD_DEF(uint32_t, isActualSharedPrefixLenNull);
TILING_DATA_FIELD_DEF(uint32_t, isQHasLeftPadding);
TILING_DATA_FIELD_DEF(uint32_t, isKVHasLeftPadding);
TILING_DATA_FIELD_DEF(int64_t, keyAntiquantMode);
TILING_DATA_FIELD_DEF(int64_t, valueAntiquantMode);
TILING_DATA_FIELD_DEF(uint32_t, hasKeyAntiquantOffset);
TILING_DATA_FIELD_DEF(uint32_t, isMsd);
TILING_DATA_FIELD_DEF(uint32_t, isQuant2FP16);
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, qkHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, vHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, gOfMla);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FiaArch38AttentionSeqParams)
// Temporary reuse
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, CoreHeadNumTail);       // coreNStart
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualS1);              // coreNEnd
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, actualCoreNums);        // coreSidStart
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, singleCoreHeadNumSize); // coreSidEnd
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosStart);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, coreSeqPosEnd);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FiaArch38AttentionSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSOuterSize);
TILING_DATA_FIELD_DEF(uint32_t, multiSmaxsInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, actualCoreNums);
TILING_DATA_FIELD_DEF(uint32_t, pseShiftBatch);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch);
TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantSInnerSize);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FiaArch38AttentionSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, pseShiftUbSize);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskUbSize);
TILING_DATA_FIELD_DEF(uint32_t, maskSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxSumSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxExpSize);
TILING_DATA_FIELD_DEF(uint32_t, softmaxValueSize);
TILING_DATA_FIELD_DEF(uint32_t, spmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, scmTmpSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpMMResBmm2PreUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpSoftmaxBmm2UbSize);
TILING_DATA_FIELD_DEF(uint32_t, selectSpaceUbSize);
TILING_DATA_FIELD_DEF(uint32_t, tmpSoftMaxV2Size);
TILING_DATA_FIELD_DEF(uint32_t, mm1TmpUbSize);
TILING_DATA_FIELD_DEF(uint32_t, mm2TmpUbSize);
TILING_DATA_FIELD_DEF(uint32_t, kvAntiquantUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbMsdSize);
TILING_DATA_FIELD_DEF(uint32_t, tempBmm2QueueMsdSize);
TILING_DATA_FIELD_DEF(uint32_t, msdInQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, msdQRowSumBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdAMaxTmpBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdAMaxResBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxResAmaxBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdSoftmaxRowSumScaleBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdScaleBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdOffsetBuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm1BuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdTmpMm2BuffSize);
TILING_DATA_FIELD_DEF(uint32_t, msdOutQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, msdComputeLines);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FiaArch38AttentionInitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
TILING_DATA_FIELD_DEF(uint32_t, needInit);
TILING_DATA_FIELD_DEF(uint32_t, isOneN);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(FiaArch38FlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingDataRect);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingDataRect);

TILING_DATA_FIELD_DEF_STRUCT(FiaArch38AttentionBaseParams, fiaArch38AttentionBaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FiaArch38AttentionSeqParams, fiaArch38AttentionSeqParams);
TILING_DATA_FIELD_DEF_STRUCT(FiaArch38AttentionSingleCoreParams, fiaArch38AttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(FiaArch38AttentionSingleCoreTensorSize, fiaArch38AttentionTensorSizeRect);
TILING_DATA_FIELD_DEF_STRUCT(FiaArch38AttentionInitOutputParams, fiaArch38AttentionInitOutputParams);

TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingDataRect);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingDataRect);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataRect);
END_TILING_DATA_DEF;

} // namespace arch38
} // namespace optiling

#endif // FIA_ARCH38_TILING_COMMON_H
