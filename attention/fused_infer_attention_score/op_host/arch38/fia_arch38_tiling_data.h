/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FIA_ARCH38_TILING_DATA_H
#define FIA_ARCH38_TILING_DATA_H

#include "register/tilingdata_base.h"

namespace optiling {

struct PFAShapeInfo {
    uint32_t b = 0;
    uint64_t n = 0;
    uint64_t s = 0;
    uint32_t d = 0;
    uint64_t h = 0;
    uint32_t t = 0;
};

BEGIN_TILING_DATA_DEF(InputParamsRegbase)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, alignedS2);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(int64_t, dSizeV);
TILING_DATA_FIELD_DEF(int64_t, dSizeRope);
TILING_DATA_FIELD_DEF(float, keepProb);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);
// in pse encoding scenes, s1 and s2 might not equal with s1, s2 in Q, K
TILING_DATA_FIELD_DEF(int64_t, pseS1Size);
TILING_DATA_FIELD_DEF(int64_t, pseS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseBSize);
TILING_DATA_FIELD_DEF(uint32_t, bandIndex);

// 1: BSH/BSND, 2: SBH, 3: BNSD
TILING_DATA_FIELD_DEF(uint8_t, layoutType);
// 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
TILING_DATA_FIELD_DEF(uint8_t, pseShapeType);
// 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
// 0: fp16, 1: bool(uint8)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
// ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
// 0: high precise, 1: high performance, 2: invalid line high precise
TILING_DATA_FIELD_DEF(uint8_t, implMode);
TILING_DATA_FIELD_DEF(uint8_t, sparseType);
TILING_DATA_FIELD_DEF(uint8_t, needDropMaskOp);
TILING_DATA_FIELD_DEF(uint8_t, dropMaskOuter);
TILING_DATA_FIELD_DEF(uint8_t, pseEncodeType);
TILING_DATA_FIELD_DEF(uint16_t, remain);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, pseType);
TILING_DATA_FIELD_DEF(uint32_t, rsv1);
TILING_DATA_FIELD_DEF(int64_t, qStartIdx);
TILING_DATA_FIELD_DEF(int64_t, kvStartIdx);
TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, seed);
TILING_DATA_FIELD_DEF(int64_t, offset);
TILING_DATA_FIELD_DEF(int64_t, keepProbUint8);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS1);
TILING_DATA_FIELD_DEF(int64_t, pseAlibiBaseS2);

// PFA
TILING_DATA_FIELD_DEF(uint8_t, deqScaleFlag);
TILING_DATA_FIELD_DEF(uint8_t, deqScale2Flag);
TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsNull);
TILING_DATA_FIELD_DEF(uint8_t, isActualSeqLengthsKVNull);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
TILING_DATA_FIELD_DEF(uint8_t, isKvContinuous);
TILING_DATA_FIELD_DEF(uint8_t, fromFused);
TILING_DATA_FIELD_DEF(uint8_t, isBSNDOut);
TILING_DATA_FIELD_DEF(uint8_t, isGqa);
TILING_DATA_FIELD_DEF(uint8_t, isSoftMaxLseEnable);
TILING_DATA_FIELD_DEF(uint8_t, isActualSharedPrefixLenNull);
TILING_DATA_FIELD_DEF(uint8_t, isQHasLeftPadding);
TILING_DATA_FIELD_DEF(uint8_t, isKVHasLeftPadding);
TILING_DATA_FIELD_DEF(uint32_t, ropeHeadSize);
TILING_DATA_FIELD_DEF(uint32_t, prefixSeqInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, headNumRatio);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, blockTableDim2);
TILING_DATA_FIELD_DEF(int32_t, paBlockNumSum);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS1Size);
TILING_DATA_FIELD_DEF(uint32_t, kvSplitPart);
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize);
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize);

TILING_DATA_FIELD_DEF(uint8_t, paLayoutType);
TILING_DATA_FIELD_DEF(uint8_t, isRowInvalid);
TILING_DATA_FIELD_DEF(uint8_t, isPostQuantPerChnl);
TILING_DATA_FIELD_DEF(uint8_t, isPostQuantBF16);
TILING_DATA_FIELD_DEF(uint16_t, antiquantPerTensorFlag);
TILING_DATA_FIELD_DEF(uint16_t, antiquantPerHeadFlag);
TILING_DATA_FIELD_DEF(uint32_t, antiquantParaSeqSize);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(MultiCoreParamsRegbase)
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 48, bnStartIdx);
TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(DropmaskParamsRegbase)
TILING_DATA_FIELD_DEF(int32_t, multiCoreFactorSize);
TILING_DATA_FIELD_DEF(int32_t, baseUbCalSize);
TILING_DATA_FIELD_DEF(int64_t, multiCoreTotalSize);
TILING_DATA_FIELD_DEF(int64_t, shapeTotalSize);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(InitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize);
TILING_DATA_FIELD_DEF(uint8_t, needInit);
TILING_DATA_FIELD_DEF(uint8_t, isOneN);
TILING_DATA_FIELD_DEF_ARR(uint8_t, 2, rsvd);
TILING_DATA_FIELD_DEF(int64_t, totalOutputSize);
TILING_DATA_FIELD_DEF(int64_t, totalSoftMaxLseOutputSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InputParamsRegbaseOp, InputParamsRegbase)
REGISTER_TILING_DATA_CLASS(MultiCoreParamsRegbaseOp, MultiCoreParamsRegbase)
REGISTER_TILING_DATA_CLASS(DropmaskParamsRegbaseOp, DropmaskParamsRegbase)
REGISTER_TILING_DATA_CLASS(InitOutputParamsOp, InitOutputParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreSimplifiedTilingData)
TILING_DATA_FIELD_DEF_STRUCT(InputParamsRegbase, inputParamsRegbase);
TILING_DATA_FIELD_DEF_STRUCT(MultiCoreParamsRegbase, multiCoreParamsRegbase);
TILING_DATA_FIELD_DEF_STRUCT(DropmaskParamsRegbase, dropmaskParamsRegbase);
TILING_DATA_FIELD_DEF_STRUCT(InitOutputParams, initOutputParams);
END_TILING_DATA_DEF;

} // namespace optiling

#endif // FIA_ARCH38_TILING_DATA_H
