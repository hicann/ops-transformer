/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling.h
 * \brief

 */
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

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

    uint32_t get_batchSize() { return batchSize; }
    uint32_t get_seqSize() { return seqSize; }
    uint32_t get_qSeqSize() { return qSeqSize; }
    uint32_t get_headSize() { return headSize; }
    uint32_t get_headSizeV() { return headSizeV; }
    uint32_t get_blockSize() { return blockSize; }
    uint32_t get_maxBlockNumPerBatch() { return maxBlockNumPerBatch; }
    uint32_t get_maxBlockNumPerSeq() { return maxBlockNumPerSeq; }
    float get_scaleValue() { return scaleValue; }
    uint32_t get_kvHeadNum() { return kvHeadNum; }
    uint32_t get_headNumRatio() { return headNumRatio; }
    uint32_t get_qHeadNum() { return qHeadNum; }
    uint32_t get_nNumOfQInOneGroup() { return nNumOfQInOneGroup; }
    uint32_t get_batchContinuousFlag() { return batchContinuousFlag; }
    uint32_t get_pseShiftFlag() { return pseShiftFlag; }
    uint32_t get_pseShiftB() { return pseShiftB; }
    uint32_t get_pseShiftS() { return pseShiftS; }
    uint32_t get_pseShiftS0() { return pseShiftS0; }
    uint32_t get_selectWithByteMaskTmpMinSize() { return selectWithByteMaskTmpMinSize; }
    uint32_t get_actualLenQDims() { return actualLenQDims; }
    uint32_t get_actualLenDims() { return actualLenDims; }
    uint32_t get_qPaddingFlag() { return qPaddingFlag; }
    uint32_t get_kvPaddingFlag() { return kvPaddingFlag; }
    uint32_t get_msdIterNum() { return msdIterNum; }
    uint32_t get_l2CacheOffFlag() { return l2CacheOffFlag; }
    uint32_t get_antiquantPerTensorFlag() { return antiquantPerTensorFlag; }
    uint32_t get_antiquantPerHeadFlag() { return antiquantPerHeadFlag; }
    uint32_t get_antiquantParamsInPagedAttentionFlag() { return antiquantParamsInPagedAttentionFlag; }
    uint32_t get_attenMaskFlag() { return attenMaskFlag; }
    uint32_t get_attenMaskBatch() { return attenMaskBatch; }
    uint32_t get_attenMaskQSize() { return attenMaskQSize; }
    uint32_t get_attenMaskSize() { return attenMaskSize; }
    uint32_t get_softmaxLseFlag() { return softmaxLseFlag; }
    uint32_t get_totalBlockNum() { return totalBlockNum; }
    uint32_t get_paKvShapeType() { return paKvShapeType; }
    uint32_t get_antiqSeqSize() { return antiqSeqSize; }
    int32_t get_preToken() { return preToken; }
    int32_t get_nextToken() { return nextToken; }
    uint32_t get_isRowInvalid() { return isRowInvalid; }
    uint32_t get_sparseMode() { return sparseMode; }
    uint32_t get_slidingFlag() { return slidingFlag; }
    int64_t get_windowSize() { return windowSize; }

    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }
    void set_seqSize(uint32_t seqSize) { this->seqSize = seqSize; }
    void set_qSeqSize(uint32_t qSeqSize) { this->qSeqSize = qSeqSize; }
    void set_headSize(uint32_t headSize) { this->headSize = headSize; }
    void set_headSizeV(uint32_t headSizeV) { this->headSizeV = headSizeV; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatch) { this->maxBlockNumPerBatch = maxBlockNumPerBatch; }
    void set_maxBlockNumPerSeq(uint32_t maxBlockNumPerSeq) { this->maxBlockNumPerSeq = maxBlockNumPerSeq; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }
    void set_kvHeadNum(uint32_t kvHeadNum) { this->kvHeadNum = kvHeadNum; }
    void set_headNumRatio(uint32_t headNumRatio) { this->headNumRatio = headNumRatio; }
    void set_qHeadNum(uint32_t qHeadNum) { this->qHeadNum = qHeadNum; }
    void set_nNumOfQInOneGroup(uint32_t nNumOfQInOneGroup) { this->nNumOfQInOneGroup = nNumOfQInOneGroup; }
    void set_batchContinuousFlag(uint32_t batchContinuousFlag) { this->batchContinuousFlag = batchContinuousFlag; }
    void set_pseShiftFlag(uint32_t pseShiftFlag) { this->pseShiftFlag = pseShiftFlag; }
    void set_pseShiftB(uint32_t pseShiftB) { this->pseShiftB = pseShiftB; }
    void set_pseShiftS(uint32_t pseShiftS) { this->pseShiftS = pseShiftS; }
    void set_pseShiftS0(uint32_t pseShiftS0) { this->pseShiftS0 = pseShiftS0; }
    void set_selectWithByteMaskTmpMinSize(uint32_t selectWithByteMaskTmpMinSize) { this->selectWithByteMaskTmpMinSize = selectWithByteMaskTmpMinSize; }
    void set_actualLenQDims(uint32_t actualLenQDims) { this->actualLenQDims = actualLenQDims; }
    void set_actualLenDims(uint32_t actualLenDims) { this->actualLenDims = actualLenDims; }
    void set_qPaddingFlag(uint32_t qPaddingFlag) { this->qPaddingFlag = qPaddingFlag; }
    void set_kvPaddingFlag(uint32_t kvPaddingFlag) { this->kvPaddingFlag = kvPaddingFlag; }
    void set_msdIterNum(uint32_t msdIterNum) { this->msdIterNum = msdIterNum; }
    void set_l2CacheOffFlag(uint32_t l2CacheOffFlag) { this->l2CacheOffFlag = l2CacheOffFlag; }
    void set_antiquantPerTensorFlag(uint32_t antiquantPerTensorFlag) { this->antiquantPerTensorFlag = antiquantPerTensorFlag; }
    void set_antiquantPerHeadFlag(uint32_t antiquantPerHeadFlag) { this->antiquantPerHeadFlag = antiquantPerHeadFlag; }
    void set_antiquantParamsInPagedAttentionFlag(uint32_t antiquantParamsInPagedAttentionFlag) { this->antiquantParamsInPagedAttentionFlag = antiquantParamsInPagedAttentionFlag; }
    void set_attenMaskFlag(uint32_t attenMaskFlag) { this->attenMaskFlag = attenMaskFlag; }
    void set_attenMaskBatch(uint32_t attenMaskBatch) { this->attenMaskBatch = attenMaskBatch; }
    void set_attenMaskQSize(uint32_t attenMaskQSize) { this->attenMaskQSize = attenMaskQSize; }
    void set_attenMaskSize(uint32_t attenMaskSize) { this->attenMaskSize = attenMaskSize; }
    void set_softmaxLseFlag(uint32_t softmaxLseFlag) { this->softmaxLseFlag = softmaxLseFlag; }
    void set_totalBlockNum(uint32_t totalBlockNum) { this->totalBlockNum = totalBlockNum; }
    void set_paKvShapeType(uint32_t paKvShapeType) { this->paKvShapeType = paKvShapeType; }
    void set_antiqSeqSize(uint32_t antiqSeqSize) { this->antiqSeqSize = antiqSeqSize; }
    void set_preToken(int32_t preToken) { this->preToken = preToken; }
    void set_nextToken(int32_t nextToken) { this->nextToken = nextToken; }
    void set_isRowInvalid(uint32_t isRowInvalid) { this->isRowInvalid = isRowInvalid; }
    void set_sparseMode(uint32_t sparseMode) { this->sparseMode = sparseMode; }
    void set_slidingFlag(uint32_t slidingFlag) { this->slidingFlag = slidingFlag; }
    void set_windowSize(uint64_t windowSize) { this->windowSize = windowSize; }
};

class IncreFlashAttentionCoreParams {
public:
    uint32_t coreSidxEnd[50]; // 50:MAX_CORE_NUM of 910b coreSidxEnd数组首地址要保证8字节对齐
    uint32_t coreSidxEndRegbase[66]; // 66:MAX_CORE_NUM of 910_95
    uint32_t coreSposStartRegbase[66]; // 66:MAX_CORE_NUM of 910_95

    uint32_t* get_coreSidxEnd() { return coreSidxEnd; }
    uint32_t* get_coreSidxEndRegbase() { return coreSidxEndRegbase; }
    uint32_t* get_coreSposStartRegbase() { return coreSposStartRegbase; }
    void set_coreSidxEnd(const uint32_t* values) { 
        for (int i = 0; i < 50; i++) {
            this->coreSidxEnd[i] = values[i];
        }
    }
    void set_coreSidxEndRegbase(const uint32_t* values) { 
        for (int i = 0; i < 66; i++) {
            this->coreSidxEndRegbase[i] = values[i];
        }
     }
    void set_coreSposStartRegbase(const uint32_t* values) { 
        for (int i = 0; i < 66; i++) {
            this->coreSposStartRegbase[i] = values[i];
        } 
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

    uint32_t get_headSplit() { return headSplit; }
    uint32_t get_maskHeadStride() { return maskHeadStride; }
    uint32_t get_maskBatchStride() { return maskBatchStride; }
    uint32_t get_qTokens() { return qTokens; }
    uint32_t get_isTriu() { return isTriu; }
    uint32_t get_maxSeqlen() { return maxSeqlen; }
    uint32_t get_totalQBlockNum() { return totalQBlockNum; }
    uint32_t get_seqStepQ() { return seqStepQ; }
    uint32_t get_seqStepKv() { return seqStepKv; }
    const uint32_t* get_startBlk() { return startBlk; }
    const uint32_t* get_endBlk() { return endBlk; }
    const uint32_t* get_startBatch() { return startBatch; }
    const uint32_t* get_endBatch() { return endBatch; }

    void set_headSplit(uint32_t headSplit) { this->headSplit = headSplit; }
    void set_maskHeadStride(uint32_t maskHeadStride) { this->maskHeadStride = maskHeadStride; }
    void set_maskBatchStride(uint32_t maskBatchStride) { this->maskBatchStride = maskBatchStride; }
    void set_qTokens(uint32_t qTokens) { this->qTokens = qTokens; }
    void set_isTriu(uint32_t isTriu) { this->isTriu = isTriu; }
    void set_maxSeqlen(uint32_t maxSeqlen) { this->maxSeqlen = maxSeqlen; }
    void set_totalQBlockNum(uint32_t totalQBlockNum) { this->totalQBlockNum = totalQBlockNum; }
    void set_seqStepQ(uint32_t seqStepQ) { this->seqStepQ = seqStepQ; }
    void set_seqStepKv(uint32_t seqStepKv) { this->seqStepKv = seqStepKv; }
    void set_startBlk(const uint32_t* values) { 
        for (int i = 0; i < 50; i++) {
            this->startBlk[i] = values[i];
        }
    }
    void set_endBlk(const uint32_t* values) { 
        for (int i = 0; i < 50; i++) {
            this->endBlk[i] = values[i];
        }
    }  
    void set_startBatch(const uint32_t* values) {
        for (int i = 0; i < 50; i++) {
            this->startBatch[i] = values[i];
        }
    }
    void set_endBatch(const uint32_t* values) {
        for (int i = 0; i < 50; i++) {
            this->endBatch[i] = values[i];
        }
    }
};

class IncreFlashAttentionSingleCoreParams{
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

    uint32_t get_sInnerLoopTimes() { return sInnerLoopTimes; }
    uint32_t get_singleProcessSInnerSize() { return singleProcessSInnerSize; }
    uint32_t get_singleProcessSInnerSizeTail() { return singleProcessSInnerSizeTail; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_formerCoreNum() { return formerCoreNum; }
    uint32_t get_blockSplitBn2Range() { return blockSplitBn2Range; }
    uint32_t get_tailSplitedBatchRange() { return tailSplitedBatchRange; }
    uint32_t get_groupSplitSize() { return groupSplitSize; }
    uint32_t get_s1SplitSize() { return s1SplitSize; }

    void set_sInnerLoopTimes(uint32_t sInnerLoopTimes) { this->sInnerLoopTimes = sInnerLoopTimes; }
    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSize) { this->singleProcessSInnerSize = singleProcessSInnerSize; }
    void set_singleProcessSInnerSizeTail(uint32_t singleProcessSInnerSizeTail) { this->singleProcessSInnerSizeTail = singleProcessSInnerSizeTail; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_formerCoreNum(uint32_t formerCoreNum) { this->formerCoreNum = formerCoreNum; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2Range) { this->blockSplitBn2Range = blockSplitBn2Range; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRange) { this->tailSplitedBatchRange = tailSplitedBatchRange; }
    void set_groupSplitSize(uint32_t groupSplitSize) { this->groupSplitSize = groupSplitSize; }
    void set_s1SplitSize(uint32_t s1SplitSize) { this->s1SplitSize = s1SplitSize; }
};

class IncreFlashAttentionSingleCoreTensorSize {
public:
    uint32_t mmResUbSize; 
    uint32_t bmm2ResUbSize;

    uint32_t get_mmResUbSize() { return mmResUbSize; }
    uint32_t get_bmm2ResUbSize() { return bmm2ResUbSize; }

    void set_mmResUbSize(uint32_t mmResUbSize) { this->mmResUbSize = mmResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }
};

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

    uint32_t get_isPerChnOut() { return isPerChnOut; }
    uint32_t get_isOutQuantTypeBf16() { return isOutQuantTypeBf16; }
    uint32_t get_singleCoreSize() { return singleCoreSize; }
    uint32_t get_singleCoreLseSize() { return singleCoreLseSize; }
    int64_t get_totalOutputSize() { return totalOutputSize; }
    int64_t get_totalLseOutputSize() { return totalLseOutputSize; }
    uint32_t get_needInit() { return needInit; }
    uint32_t get_isBSNDOut() { return isBSNDOut; }

    void set_isPerChnOut(uint32_t isPerChnOut) { this->isPerChnOut = isPerChnOut; }
    void set_isOutQuantTypeBf16(uint32_t isOutQuantTypeBf16) { this->isOutQuantTypeBf16 = isOutQuantTypeBf16; }
    void set_singleCoreSize(uint32_t singleCoreSize) { this->singleCoreSize = singleCoreSize; }
    void set_singleCoreLseSize(uint32_t singleCoreLseSize) { this->singleCoreLseSize = singleCoreLseSize; }
    void set_totalOutputSize(int64_t totalOutputSize) { this->totalOutputSize = totalOutputSize; }
    void set_totalLseOutputSize(int64_t totalLseOutputSize) { this->totalLseOutputSize = totalLseOutputSize; }
    void set_needInit(uint32_t needInit) { this->needInit = needInit; }
    void set_isBSNDOut(uint32_t isBSNDOut) { this->isBSNDOut = isBSNDOut; }
};

class IncreFlashAttentionSplitKVParams {
public:
    uint32_t s2; 
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize; 
    uint32_t logSumExpSize;

    uint32_t get_s2() { return s2; }
    uint32_t get_sInnerLoopSize() { return sInnerLoopSize; }
    uint32_t get_accumOutSize() { return accumOutSize; }
    uint32_t get_logSumExpSize() { return logSumExpSize; }
    
    void set_s2(uint32_t s2) { this->s2 = s2; }
    void set_sInnerLoopSize(uint32_t sInnerLoopSize) { this->sInnerLoopSize = sInnerLoopSize; }
    void set_accumOutSize(uint32_t accumOutSize) { this->accumOutSize = accumOutSize; }
    void set_logSumExpSize(uint32_t logSumExpSize) { this->logSumExpSize = logSumExpSize; }
};

class IncreFlashAttentionTilingData {
public:
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    IncreFlashAttentionBaseParams baseParams;
    IncreFlashAttentionSplitKVParams splitKVParams;
    IncreFlashAttentionCoreParams increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParams increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSize increFlashAttentionSingleCoreTensorSize;
    SoftMaxTiling softmaxFlashTilingData;
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

    uint64_t get_prefixAttenOutOffset() { return prefixAttenOutOffset; }
    uint64_t get_userPromptAttenOutOffset() { return userPromptAttenOutOffset; }
    uint64_t get_tmpLseOffset() { return tmpLseOffset; }
    uint64_t get_prefixLen() { return prefixLen; }
    uint32_t get_formerCoreNum() { return formerCoreNum; }
    uint32_t get_blockSplitBn2Range() { return blockSplitBn2Range; }
    uint32_t get_tailSplitedBatchRange() { return tailSplitedBatchRange; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_batchSizeQ() { return batchSizeQ; }

    void set_prefixAttenOutOffset(uint64_t prefixAttenOutOffset) { this->prefixAttenOutOffset = prefixAttenOutOffset; }
    void set_userPromptAttenOutOffset(uint64_t userPromptAttenOutOffset) { this->userPromptAttenOutOffset = userPromptAttenOutOffset; }
    void set_tmpLseOffset(uint64_t tmpLseOffset) { this->tmpLseOffset = tmpLseOffset; }
    void set_prefixLen(uint64_t prefixLen) { this->prefixLen = prefixLen; }
    void set_formerCoreNum(uint32_t formerCoreNum) { this->formerCoreNum = formerCoreNum; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2Range) { this->blockSplitBn2Range = blockSplitBn2Range; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRange) { this->tailSplitedBatchRange = tailSplitedBatchRange; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_batchSizeQ(uint32_t batchSizeQ) { this->batchSizeQ = batchSizeQ; }
};

class IncreFlashAttentionTilingDataV2 {
public:
    IncreFlashAttentionTilingData tilingBase;
    IncreFlashAttentionTilingDataPrefix tilingPrefix;
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

const uint32_t MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

class IncreFlashAttentionBaseParamsMla{
public:
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenQDims;
    uint32_t actualLenDims;
    uint32_t antiquantMode;
    uint32_t attenMaskFlag;
    uint32_t attenMaskSize;
    uint32_t outputLayout;

    uint32_t get_batchSize() { return batchSize; }
    uint32_t get_seqSize() { return seqSize; }
    uint32_t get_qSeqSize() { return qSeqSize; }
    uint32_t get_blockSize() { return blockSize; }
    uint32_t get_maxBlockNumPerBatch() { return maxBlockNumPerBatch; }
    float get_scaleValue() { return scaleValue; }
    uint32_t get_nNumOfQInOneGroup() { return nNumOfQInOneGroup; }
    uint32_t get_actualLenQDims() { return actualLenQDims; }
    uint32_t get_actualLenDims() { return actualLenDims; }
    uint32_t get_antiquantMode() { return antiquantMode; }
    uint32_t get_attenMaskFlag() { return attenMaskFlag; }
    uint32_t get_attenMaskSize() { return attenMaskSize; }
    uint32_t get_outputLayout() { return outputLayout; }

    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }
    void set_seqSize(uint32_t seqSize) { this->seqSize = seqSize; }
    void set_qSeqSize(uint32_t qSeqSize) { this->qSeqSize = qSeqSize; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatch) { this->maxBlockNumPerBatch = maxBlockNumPerBatch; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }
    void set_nNumOfQInOneGroup(uint32_t nNumOfQInOneGroup) { this->nNumOfQInOneGroup = nNumOfQInOneGroup; }
    void set_actualLenQDims(uint32_t actualLenQDims) { this->actualLenQDims = actualLenQDims; }
    void set_actualLenDims(uint32_t actualLenDims) { this->actualLenDims = actualLenDims; }
    void set_antiquantMode(uint32_t antiquantMode) { this->antiquantMode = antiquantMode; }
    void set_attenMaskFlag(uint32_t attenMaskFlag) { this->attenMaskFlag = attenMaskFlag; }
    void set_attenMaskSize(uint32_t attenMaskSize) { this->attenMaskSize = attenMaskSize; }
    void set_outputLayout(uint32_t outputLayout) { this->outputLayout = outputLayout; }
};

class IncreFlashAttentionCoreParamsMla {
public:
    uint32_t coreBEnd[MAX_AIC_CORE_NUM]; 
    uint32_t coreSidxEnd[MAX_AIC_CORE_NUM];
    uint32_t coreS1OuterEnd[MAX_AIC_CORE_NUM]; 
    uint32_t coreS2End[MAX_AIC_CORE_NUM]; 

    uint32_t* get_coreBEnd() { return coreBEnd; }
    uint32_t* get_coreSidxEnd() { return coreSidxEnd; }
    uint32_t* get_coreS1OuterEnd() { return coreS1OuterEnd; }
    uint32_t* get_coreS2End() { return coreS2End; }
    void set_coreBEnd(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->coreBEnd[i] = values[i];
        }
    }
    void set_coreSidxEnd(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->coreSidxEnd[i] = values[i];
        }
    }
    void set_coreS1OuterEnd(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->coreS1OuterEnd[i] = values[i];
        }
    }
    void set_coreS2End(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->coreS2End[i] = values[i];
        }
    }
};

class IncreFlashAttentionSingleCoreParamsMla {
public:
    uint32_t singleProcessSInnerSize; 
    uint32_t usedCoreNum;
    uint32_t groupSplitSize; 
    uint32_t s1SplitSize; 

    uint32_t get_singleProcessSInnerSize() { return singleProcessSInnerSize; }
    uint32_t get_usedCoreNum() { return usedCoreNum; }
    uint32_t get_groupSplitSize() { return groupSplitSize; }
    uint32_t get_s1SplitSize() { return s1SplitSize; }

    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSize) { this->singleProcessSInnerSize = singleProcessSInnerSize; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_groupSplitSize(uint32_t groupSplitSize) { this->groupSplitSize = groupSplitSize; }
    void set_s1SplitSize(uint32_t s1SplitSize) { this->s1SplitSize = s1SplitSize; }
};

class IncreFlashAttentionSingleCoreTensorSizeMla {
public:
    uint32_t mmResUbSize; 
    uint32_t bmm2ResUbSize;

    uint32_t get_mmResUbSize() { return mmResUbSize; }
    uint32_t get_bmm2ResUbSize() { return bmm2ResUbSize; }

    void set_mmResUbSize(uint32_t mmResUbSize) { this->mmResUbSize = mmResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }
};

class IncreFlashAttentionSplitKVParamsMla {
public:
    uint32_t s2; 
    uint32_t sInnerLoopSize;
    uint32_t accumOutSize; 
    uint32_t logSumExpSize;

    uint32_t get_s2() { return s2; }
    uint32_t get_sInnerLoopSize() { return sInnerLoopSize; }
    uint32_t get_accumOutSize() { return accumOutSize; }
    uint32_t get_logSumExpSize() { return logSumExpSize; }
    
    void set_s2(uint32_t s2) { this->s2 = s2; }
    void set_sInnerLoopSize(uint32_t sInnerLoopSize) { this->sInnerLoopSize = sInnerLoopSize; }
    void set_accumOutSize(uint32_t accumOutSize) { this->accumOutSize = accumOutSize; }
    void set_logSumExpSize(uint32_t logSumExpSize) { this->logSumExpSize = logSumExpSize; }
};

class IncreFlashAttentionTNDSplitCoreParamsMla {
public:
    uint32_t tndFDCoreArrLen; 
    uint32_t reserve;
    uint32_t balanceFDCoreBArr[MAX_AIC_CORE_NUM]; 
    uint32_t balanceFDCoreS1Arr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreKVSplitArr[MAX_AIC_CORE_NUM];
    uint32_t balanceFDCoreStartKVSplitNum[MAX_AIC_CORE_NUM];

    uint32_t get_tndFDCoreArrLen() { return tndFDCoreArrLen; }
    uint32_t get_reserve() { return reserve; }
    uint32_t* get_balanceFDCoreBArr() { return balanceFDCoreBArr; }
    uint32_t* get_balanceFDCoreS1Arr() { return balanceFDCoreS1Arr; }
    uint32_t* get_balanceFDCoreKVSplitArr() { return balanceFDCoreKVSplitArr; }
    uint32_t* get_balanceFDCoreStartKVSplitNum() { return balanceFDCoreStartKVSplitNum; }
    
    void set_tndFDCoreArrLen(uint32_t tndFDCoreArrLen) { this->tndFDCoreArrLen = tndFDCoreArrLen; }
    void set_reserve(uint32_t reserve) { this->reserve = reserve; }
    void set_balanceFDCoreBArr(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->balanceFDCoreBArr[i] = values[i];
        }
    }
    void set_balanceFDCoreS1Arr(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->balanceFDCoreS1Arr[i] = values[i];
        }
    }
    void set_balanceFDCoreKVSplitArr(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->balanceFDCoreKVSplitArr[i] = values[i];
        }
    }
    void set_balanceFDCoreStartKVSplitNum(
        const uint32_t* values) {
        for (int i = 0; i < MAX_AIC_CORE_NUM; i++) {
            this->balanceFDCoreStartKVSplitNum[i] = values[i];
        }
    }
};

class IncreFlashAttentionTilingDataMla {
public:
    IncreFlashAttentionBaseParamsMla baseParams;
    IncreFlashAttentionSplitKVParamsMla splitKVParams;
    IncreFlashAttentionTNDSplitCoreParamsMla tndSplitCoreParams;
    IncreFlashAttentionCoreParamsMla increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParamsMla increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSizeMla increFlashAttentionSingleCoreTensorSize;
};
// #endif

// } // namespace optiling
// #endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_H_