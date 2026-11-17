/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QFMLA_IFA_TILING_DATA_MLA_GEN_H_
#define QFMLA_IFA_TILING_DATA_MLA_GEN_H_
class IncreFlashAttentionBaseParamsMla {
public:
    uint32_t batchSize = 0;
    uint32_t seqSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    float scaleValue = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t actualLenQDims = 0;
    uint32_t actualLenDims = 0;
    uint32_t antiquantMode = 0;
    uint32_t attenMaskFlag = 0;
    uint32_t attenMaskSize = 0;
    uint32_t sparseMode = 0;
    uint32_t outputLayout = 0;
    uint32_t softmaxLseFlag = 0;
    uint32_t tSeqSize = 0;
};

class IncreFlashAttentionSplitKVParamsMla {
public:
    uint32_t s2 = 0;
    uint32_t sInnerLoopSize = 0;
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

class IncreFlashAttentionTNDSplitCoreParamsMla {
public:
    uint32_t tndFDCoreArrLen = 0;
    uint32_t reserve = 0;
    uint32_t balanceFDCoreBArr[26] = {};
    uint32_t balanceFDCoreS1Arr[26] = {};
    uint32_t balanceFDCoreKVSplitArr[26] = {};
    uint32_t balanceFDCoreStartKVSplitNum[26] = {};
    uint32_t gS1IdxEndOfFdHead[52] = {};
    uint32_t gS1IdxEndOfFdHeadSplit[52] = {};
    uint32_t usedVecNumOfFd = 0;
    uint32_t FdBalanceFlag = 0;
    uint32_t gS1SplitNumOfFdHeadMla[52] = {};
    uint32_t gS1LastPartSizeOfFdHead[52] = {};
};

class IncreFlashAttentionCoreParamsMla {
public:
    uint32_t coreBEnd[26] = {};
    uint32_t coreSidxEnd[26] = {};
    uint32_t coreS1OuterEnd[26] = {};
    uint32_t coreS2End[26] = {};
};

class IncreFlashAttentionSingleCoreParamsMla {
public:
    uint32_t singleProcessSInnerSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t groupSplitSize = 0;
    uint32_t s1SplitSize = 0;
};

class IncreFlashAttentionSingleCoreTensorSizeMla {
public:
    uint32_t mmResUbSize = 0;
    uint32_t bmm2ResUbSize = 0;
};

class IncreFlashAttentionTilingDataMla {
public:
    IncreFlashAttentionBaseParamsMla baseParams;
    IncreFlashAttentionSplitKVParamsMla splitKVParams;
    IncreFlashAttentionTNDSplitCoreParamsMla tndSplitCoreParams;
    IncreFlashAttentionCoreParamsMla increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParamsMla increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSizeMla increFlashAttentionSingleCoreTensorSize;
} __attribute__((__may_alias__));

#endif
