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
 * \file incre_flash_attention_tiling.h
 * \brief

 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_TILINGDATA_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_TILINGDATA_H_

#include <cstdint>
#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {

class IncreFlashAttentionInitOutputParams {
public:
    uint32_t isPerChnOut;
    uint32_t isOutQuantTypeBf16;
    uint32_t singleCoreSize;
    uint32_t singleCoreLseSize;
    int64_t totalOutputSize;
    int64_t totalLseOutputSize;
    uint32_t needInit;
    uint32_t isBSNDOut;

public:
    // isPerChnOut
    void set_isPerChnOut(uint32_t value)
    {
        isPerChnOut = value;
    }
    uint32_t get_isPerChnOut() const
    {
        return isPerChnOut;
    }
    // isOutQuantTypeBf16
    void set_isOutQuantTypeBf16(uint32_t value)
    {
        isOutQuantTypeBf16 = value;
    }
    uint32_t get_isOutQuantTypeBf16() const
    {
        return isOutQuantTypeBf16;
    }
    // singleCoreSize
    void set_singleCoreSize(uint32_t value)
    {
        singleCoreSize = value;
    }
    uint32_t get_singleCoreSize() const
    {
        return singleCoreSize;
    }
    // singleCoreLseSize
    void set_singleCoreLseSize(uint32_t value)
    {
        singleCoreLseSize = value;
    }
    uint32_t get_singleCoreLseSize() const
    {
        return singleCoreLseSize;
    }
    // totalOutputSize
    void set_totalOutputSize(int64_t value)
    {
        totalOutputSize = value;
    }
    int64_t get_totalOutputSize() const
    {
        return totalOutputSize;
    }
    // totalLseOutputSize
    void set_totalLseOutputSize(int64_t value)
    {
        totalLseOutputSize = value;
    }
    int64_t get_totalLseOutputSize() const
    {
        return totalLseOutputSize;
    }
    // needInit
    void set_needInit(uint32_t value)
    {
        needInit = value;
    }
    uint32_t get_needInit() const
    {
        return needInit;
    }
    // isBSNDOut
    void set_isBSNDOut(uint32_t value)
    {
        isBSNDOut = value;
    }
    uint32_t get_isBSNDOut() const
    {
        return isBSNDOut;
    }
};

class IncreFlashAttentionBaseParams {
public:
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t headSize;
    uint32_t headSizeV;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t maxBlockNumPerSeq;
    float scaleValue;
    uint32_t kvHeadNum;
    uint32_t headNumRatio;
    uint32_t qHeadNum;
    uint32_t nNumOfQInOneGroup;
    uint32_t batchContinuousFlag;
    uint32_t pseShiftFlag;
    uint32_t pseShiftB;
    uint32_t pseShiftS;
    uint32_t pseShiftS0;
    uint32_t selectWithByteMaskTmpMinSize;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
    uint32_t msdIterNum;
    uint32_t l2CacheOffFlag;
    uint32_t antiquantPerTensorFlag;
    uint32_t antiquantPerHeadFlag;
    uint32_t antiquantParamsInPagedAttentionFlag;
    uint32_t attenMaskFlag;
    uint32_t attenMaskBatch;
    uint32_t attenMaskQSize;
    uint32_t attenMaskSize;
    uint32_t softmaxLseFlag;
    uint32_t totalBlockNum;
    uint32_t paKvShapeType;
    uint32_t antiqSeqSize;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t sparseMode;
    uint32_t slidingFlag;
    int64_t windowSize;

public:
    // batchSize
    void set_batchSize(uint32_t value)
    {
        batchSize = value;
    }
    uint32_t get_batchSize() const
    {
        return batchSize;
    }
    // seqSize
    void set_seqSize(uint32_t value)
    {
        seqSize = value;
    }
    uint32_t get_seqSize() const
    {
        return seqSize;
    }
    // qSeqSize
    void set_qSeqSize(uint32_t value)
    {
        qSeqSize = value;
    }
    uint32_t get_qSeqSize() const
    {
        return qSeqSize;
    }
    // headSize
    void set_headSize(uint32_t value)
    {
        headSize = value;
    }
    uint32_t get_headSize() const
    {
        return headSize;
    }
    // headSizeV
    void set_headSizeV(uint32_t value)
    {
        headSizeV = value;
    }
    uint32_t get_headSizeV() const
    {
        return headSizeV;
    }
    // blockSize
    void set_blockSize(uint32_t value)
    {
        blockSize = value;
    }
    uint32_t get_blockSize() const
    {
        return blockSize;
    }
    // maxBlockNumPerBatch
    void set_maxBlockNumPerBatch(uint32_t value)
    {
        maxBlockNumPerBatch = value;
    }
    uint32_t get_maxBlockNumPerBatch() const
    {
        return maxBlockNumPerBatch;
    }
    // maxBlockNumPerSeq
    void set_maxBlockNumPerSeq(uint32_t value)
    {
        maxBlockNumPerSeq = value;
    }
    uint32_t get_maxBlockNumPerSeq() const
    {
        return maxBlockNumPerSeq;
    }
    // scaleValue
    void set_scaleValue(float value)
    {
        scaleValue = value;
    }
    float get_scaleValue() const
    {
        return scaleValue;
    }
    // kvHeadNum
    void set_kvHeadNum(uint32_t value)
    {
        kvHeadNum = value;
    }
    uint32_t get_kvHeadNum() const
    {
        return kvHeadNum;
    }
    // headNumRatio
    void set_headNumRatio(uint32_t value)
    {
        headNumRatio = value;
    }
    uint32_t get_headNumRatio() const
    {
        return headNumRatio;
    }
    // qHeadNum
    void set_qHeadNum(uint32_t value)
    {
        qHeadNum = value;
    }
    uint32_t get_qHeadNum() const
    {
        return qHeadNum;
    }
    // nNumOfQInOneGroup
    void set_nNumOfQInOneGroup(uint32_t value)
    {
        nNumOfQInOneGroup = value;
    }
    uint32_t get_nNumOfQInOneGroup() const
    {
        return nNumOfQInOneGroup;
    }
    // batchContinuousFlag
    void set_batchContinuousFlag(uint32_t value)
    {
        batchContinuousFlag = value;
    }
    uint32_t get_batchContinuousFlag() const
    {
        return batchContinuousFlag;
    }
    // pseShiftFlag
    void set_pseShiftFlag(uint32_t value)
    {
        pseShiftFlag = value;
    }
    uint32_t get_pseShiftFlag() const
    {
        return pseShiftFlag;
    }
    // pseShiftB
    void set_pseShiftB(uint32_t value)
    {
        pseShiftB = value;
    }
    uint32_t get_pseShiftB() const
    {
        return pseShiftB;
    }
    // pseShiftS
    void set_pseShiftS(uint32_t value)
    {
        pseShiftS = value;
    }
    uint32_t get_pseShiftS() const
    {
        return pseShiftS;
    }
    // pseShiftS0
    void set_pseShiftS0(uint32_t value)
    {
        pseShiftS0 = value;
    }
    uint32_t get_pseShiftS0() const
    {
        return pseShiftS0;
    }
    // selectWithByteMaskTmpMinSize
    void set_selectWithByteMaskTmpMinSize(uint32_t value)
    {
        selectWithByteMaskTmpMinSize = value;
    }
    uint32_t get_selectWithByteMaskTmpMinSize() const
    {
        return selectWithByteMaskTmpMinSize;
    }
    // actualLenQDims
    void set_actualLenQDims(uint32_t value)
    {
        actualLenQDims = value;
    }
    uint32_t get_actualLenQDims() const
    {
        return actualLenQDims;
    }
    // actualLenDims
    void set_actualLenDims(uint32_t value)
    {
        actualLenDims = value;
    }
    uint32_t get_actualLenDims() const
    {
        return actualLenDims;
    }
    // qPaddingFlag
    void set_qPaddingFlag(uint32_t value)
    {
        qPaddingFlag = value;
    }
    uint32_t get_qPaddingFlag() const
    {
        return qPaddingFlag;
    }
    // kvPaddingFlag
    void set_kvPaddingFlag(uint32_t value)
    {
        kvPaddingFlag = value;
    }
    uint32_t get_kvPaddingFlag() const
    {
        return kvPaddingFlag;
    }
    // msdIterNum
    void set_msdIterNum(uint32_t value)
    {
        msdIterNum = value;
    }
    uint32_t get_msdIterNum() const
    {
        return msdIterNum;
    }
    // l2CacheOffFlag
    void set_l2CacheOffFlag(uint32_t value)
    {
        l2CacheOffFlag = value;
    }
    uint32_t get_l2CacheOffFlag() const
    {
        return l2CacheOffFlag;
    }
    // antiquantPerTensorFlag
    void set_antiquantPerTensorFlag(uint32_t value)
    {
        antiquantPerTensorFlag = value;
    }
    uint32_t get_antiquantPerTensorFlag() const
    {
        return antiquantPerTensorFlag;
    }
    // antiquantPerHeadFlag
    void set_antiquantPerHeadFlag(uint32_t value)
    {
        antiquantPerHeadFlag = value;
    }
    uint32_t get_antiquantPerHeadFlag() const
    {
        return antiquantPerHeadFlag;
    }
    // antiquantParamsInPagedAttentionFlag
    void set_antiquantParamsInPagedAttentionFlag(uint32_t value)
    {
        antiquantParamsInPagedAttentionFlag = value;
    }
    uint32_t get_antiquantParamsInPagedAttentionFlag() const
    {
        return antiquantParamsInPagedAttentionFlag;
    }
    // attenMaskFlag
    void set_attenMaskFlag(uint32_t value)
    {
        attenMaskFlag = value;
    }
    uint32_t get_attenMaskFlag() const
    {
        return attenMaskFlag;
    }
    // attenMaskBatch
    void set_attenMaskBatch(uint32_t value)
    {
        attenMaskBatch = value;
    }
    uint32_t get_attenMaskBatch() const
    {
        return attenMaskBatch;
    }
    // attenMaskQSize
    void set_attenMaskQSize(uint32_t value)
    {
        attenMaskQSize = value;
    }
    uint32_t get_attenMaskQSize() const
    {
        return attenMaskQSize;
    }
    // attenMaskSize
    void set_attenMaskSize(uint32_t value)
    {
        attenMaskSize = value;
    }
    uint32_t get_attenMaskSize() const
    {
        return attenMaskSize;
    }
    // softmaxLseFlag
    void set_softmaxLseFlag(uint32_t value)
    {
        softmaxLseFlag = value;
    }
    uint32_t get_softmaxLseFlag() const
    {
        return softmaxLseFlag;
    }
    // totalBlockNum
    void set_totalBlockNum(uint32_t value)
    {
        totalBlockNum = value;
    }
    uint32_t get_totalBlockNum() const
    {
        return totalBlockNum;
    }
    // paKvShapeType
    void set_paKvShapeType(uint32_t value)
    {
        paKvShapeType = value;
    }
    uint32_t get_paKvShapeType() const
    {
        return paKvShapeType;
    }
    // antiqSeqSize
    void set_antiqSeqSize(uint32_t value)
    {
        antiqSeqSize = value;
    }
    uint32_t get_antiqSeqSize() const
    {
        return antiqSeqSize;
    }
    // preToken
    void set_preToken(int32_t value)
    {
        preToken = value;
    }
    int32_t get_preToken() const
    {
        return preToken;
    }
    // nextToken
    void set_nextToken(int32_t value)
    {
        nextToken = value;
    }
    int32_t get_nextToken() const
    {
        return nextToken;
    }
    // isRowInvalid
    void set_isRowInvalid(uint32_t value)
    {
        isRowInvalid = value;
    }
    uint32_t get_isRowInvalid() const
    {
        return isRowInvalid;
    }
    // sparseMode
    void set_sparseMode(uint32_t value)
    {
        sparseMode = value;
    }
    uint32_t get_sparseMode() const
    {
        return sparseMode;
    }
    // slidingFlag
    void set_slidingFlag(uint32_t value)
    {
        slidingFlag = value;
    }
    uint32_t get_slidingFlag() const
    {
        return slidingFlag;
    }
    // windowSize
    void set_windowSize(int64_t value)
    {
        windowSize = value;
    }
    int64_t get_windowSize() const
    {
        return windowSize;
    }
};

class IncreFlashAttentionCoreParams {
public:
    uint32_t coreSidxEnd[50];
    uint32_t coreSidxEndRegbase[66];
    uint32_t coreSposStartRegbase[66];

public:
    uint32_t* get_coreSidxEnd()
    {
        return coreSidxEnd;
    }
    uint32_t* get_coreSidxEndRegbase()
    {
        return coreSidxEndRegbase;
    }
    uint32_t* get_coreSposStartRegbase()
    {
        return coreSposStartRegbase;
    }
};

class IncreFlashAttentionSplitCoreParams {
public:
    uint32_t headSplit;
    uint32_t maskHeadStride;
    uint32_t maskBatchStride;
    uint32_t qTokens;
    uint32_t isTriu;
    uint32_t maxSeqlen;
    uint32_t totalQBlockNum;
    uint32_t seqStepQ;
    uint32_t seqStepKv;
    uint32_t startBlk[50];
    uint32_t endBlk[50];
    uint32_t startBatch[50];
    uint32_t endBatch[50];

public:
    // headSplit
    void set_headSplit(uint32_t value)
    {
        headSplit = value;
    }
    uint32_t get_headSplit() const
    {
        return headSplit;
    }
    // maskHeadStride
    void set_maskHeadStride(uint32_t value)
    {
        maskHeadStride = value;
    }
    uint32_t get_maskHeadStride() const
    {
        return maskHeadStride;
    }
    // maskBatchStride
    void set_maskBatchStride(uint32_t value)
    {
        maskBatchStride = value;
    }
    uint32_t get_maskBatchStride() const
    {
        return maskBatchStride;
    }
    // qTokens
    void set_qTokens(uint32_t value)
    {
        qTokens = value;
    }
    uint32_t get_qTokens() const
    {
        return qTokens;
    }
    // isTriu
    void set_isTriu(uint32_t value)
    {
        isTriu = value;
    }
    uint32_t get_isTriu() const
    {
        return isTriu;
    }
    // maxSeqlen
    void set_maxSeqlen(uint32_t value)
    {
        maxSeqlen = value;
    }
    uint32_t get_maxSeqlen() const
    {
        return maxSeqlen;
    }
    // totalQBlockNum
    void set_totalQBlockNum(uint32_t value)
    {
        totalQBlockNum = value;
    }
    uint32_t get_totalQBlockNum() const
    {
        return totalQBlockNum;
    }
    // seqStepQ
    void set_seqStepQ(uint32_t value)
    {
        seqStepQ = value;
    }
    uint32_t get_seqStepQ() const
    {
        return seqStepQ;
    }
    // seqStepKv
    void set_seqStepKv(uint32_t value)
    {
        seqStepKv = value;
    }
    uint32_t get_seqStepKv() const
    {
        return seqStepKv;
    }

    const uint32_t* get_startBlk() const
    {
        return startBlk;
    }
    const uint32_t* get_endBlk() const
    {
        return endBlk;
    }
    const uint32_t* get_startBatch() const
    {
        return startBatch;
    }
    const uint32_t* get_endBatch() const
    {
        return endBatch;
    }
};

class IncreFlashAttentionSingleCoreParams {
public:
    uint32_t sInnerLoopTimes;
    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t usedCoreNum;
    uint32_t formerCoreNum;
    uint32_t blockSplitBn2Range;
    uint32_t tailSplitedBatchRange;
    uint32_t groupSplitSize;
    uint32_t s1SplitSize;

public:
    // sInnerLoopTimes
    void set_sInnerLoopTimes(uint32_t value)
    {
        sInnerLoopTimes = value;
    }
    uint32_t get_sInnerLoopTimes() const
    {
        return sInnerLoopTimes;
    }
    // singleProcessSInnerSize
    void set_singleProcessSInnerSize(uint32_t value)
    {
        singleProcessSInnerSize = value;
    }
    uint32_t get_singleProcessSInnerSize() const
    {
        return singleProcessSInnerSize;
    }
    // singleProcessSInnerSizeTail
    void set_singleProcessSInnerSizeTail(uint32_t value)
    {
        singleProcessSInnerSizeTail = value;
    }
    uint32_t get_singleProcessSInnerSizeTail() const
    {
        return singleProcessSInnerSizeTail;
    }
    // usedCoreNum
    void set_usedCoreNum(uint32_t value)
    {
        usedCoreNum = value;
    }
    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    // formerCoreNum
    void set_formerCoreNum(uint32_t value)
    {
        formerCoreNum = value;
    }
    uint32_t get_formerCoreNum() const
    {
        return formerCoreNum;
    }
    // blockSplitBn2Range
    void set_blockSplitBn2Range(uint32_t value)
    {
        blockSplitBn2Range = value;
    }
    uint32_t get_blockSplitBn2Range() const
    {
        return blockSplitBn2Range;
    }
    // tailSplitedBatchRange
    void set_tailSplitedBatchRange(uint32_t value)
    {
        tailSplitedBatchRange = value;
    }
    uint32_t get_tailSplitedBatchRange() const
    {
        return tailSplitedBatchRange;
    }
    // groupSplitSize
    void set_groupSplitSize(uint32_t value)
    {
        groupSplitSize = value;
    }
    uint32_t get_groupSplitSize() const
    {
        return groupSplitSize;
    }
    // s1SplitSize
    void set_s1SplitSize(uint32_t value)
    {
        s1SplitSize = value;
    }
    uint32_t get_s1SplitSize() const
    {
        return s1SplitSize;
    }
};

class IncreFlashAttentionSingleCoreTensorSize {
public:
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;

public:
    // mmResUbSize
    void set_mmResUbSize(uint32_t value)
    {
        mmResUbSize = value;
    }
    uint32_t get_mmResUbSize() const
    {
        return mmResUbSize;
    }
    // bmm2ResUbSize
    void set_bmm2ResUbSize(uint32_t value)
    {
        bmm2ResUbSize = value;
    }
    uint32_t get_bmm2ResUbSize() const
    {
        return bmm2ResUbSize;
    }
};

class IncreFlashAttentionSplitKVParams {
public:
    uint32_t s2;
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

public:
    // s2
    void set_s2(uint32_t value)
    {
        s2 = value;
    }
    uint32_t get_s2() const
    {
        return s2;
    }
    // sInnerLoopSize
    void set_sInnerLoopSize(uint32_t value)
    {
        sInnerLoopSize = value;
    }
    uint32_t get_sInnerLoopSize() const
    {
        return sInnerLoopSize;
    }
    // accumOutSize
    void set_accumOutSize(uint32_t value)
    {
        accumOutSize = value;
    }
    uint32_t get_accumOutSize() const
    {
        return accumOutSize;
    }
    // logSumExpSize
    void set_logSumExpSize(uint32_t value)
    {
        logSumExpSize = value;
    }
    uint32_t get_logSumExpSize() const
    {
        return logSumExpSize;
    }
};

class IncreFlashAttentionTilingData {
public:
    IncreFlashAttentionBaseParams baseParams;
    IncreFlashAttentionSplitKVParams splitKVParams;
    IncreFlashAttentionCoreParams increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParams increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSize increFlashAttentionSingleCoreTensorSize;
    IncreFlashAttentionInitOutputParams outputParams;
};

class IncreFlashAttentionTilingDataPrefix {
public:
    IncreFlashAttentionTilingData base;
    uint64_t prefixAttenOutOffset;
    uint64_t userPromptAttenOutOffset;
    uint64_t tmpLseOffset;
    uint64_t prefixLen;
    uint32_t formerCoreNum;
    uint32_t blockSplitBn2Range;
    uint32_t tailSplitedBatchRange;
    uint32_t usedCoreNum;
    uint32_t batchSizeQ;

public:
    // prefixAttenOutOffset
    void set_prefixAttenOutOffset(uint64_t value)
    {
        prefixAttenOutOffset = value;
    }
    uint64_t get_prefixAttenOutOffset() const
    {
        return prefixAttenOutOffset;
    }
    // userPromptAttenOutOffset
    void set_userPromptAttenOutOffset(uint64_t value)
    {
        userPromptAttenOutOffset = value;
    }
    uint64_t get_userPromptAttenOutOffset() const
    {
        return userPromptAttenOutOffset;
    }
    // tmpLseOffset
    void set_tmpLseOffset(uint64_t value)
    {
        tmpLseOffset = value;
    }
    uint64_t get_tmpLseOffset() const
    {
        return tmpLseOffset;
    }
    // prefixLen
    void set_prefixLen(uint64_t value)
    {
        prefixLen = value;
    }
    uint64_t get_prefixLen() const
    {
        return prefixLen;
    }
    // formerCoreNum
    void set_formerCoreNum(uint32_t value)
    {
        formerCoreNum = value;
    }
    uint32_t get_formerCoreNum() const
    {
        return formerCoreNum;
    }
    // blockSplitBn2Range
    void set_blockSplitBn2Range(uint32_t value)
    {
        blockSplitBn2Range = value;
    }
    uint32_t get_blockSplitBn2Range() const
    {
        return blockSplitBn2Range;
    }
    // tailSplitedBatchRange
    void set_tailSplitedBatchRange(uint32_t value)
    {
        tailSplitedBatchRange = value;
    }
    uint32_t get_tailSplitedBatchRange() const
    {
        return tailSplitedBatchRange;
    }
    // usedCoreNum
    void set_usedCoreNum(uint32_t value)
    {
        usedCoreNum = value;
    }
    uint32_t get_usedCoreNum() const
    {
        return usedCoreNum;
    }
    // batchSizeQ
    void set_batchSizeQ(uint32_t value)
    {
        batchSizeQ = value;
    }
    uint32_t get_batchSizeQ() const
    {
        return batchSizeQ;
    }
};

class IncreFlashAttentionTilingDataV2 {
public:
    IncreFlashAttentionTilingData tilingBase;
};

class IncreFlashAttentionTilingAtbDataV2 {
public:
    IncreFlashAttentionBaseParams tilingBase;
    IncreFlashAttentionSplitCoreParams tilingPerCore;
};

class IncreFlashAttentionEmptyInputTilingData {
public:
    IncreFlashAttentionInitOutputParams outputParams;
};

} // namespace optiling
#endif
