/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_v3.h
 * \brief
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TILINGDATA_V3
#define FUSED_INFER_ATTENTION_SCORE_TILINGDATA_V3

#include <cstdint>

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif
namespace optiling {
constexpr uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

// 基础参数
class FiaBaseTilingParam {
public:
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t actualSeqS1Dims;
    uint32_t actualSeqS2Dims;
    float scaleValue;
    uint32_t usedCoreNum;
    uint32_t outputLayout;
    uint32_t softmaxLseFlag;
    bool slidingFlag;
    bool needInit;

    uint32_t get_bSize() const { return bSize; }
    uint32_t get_n2Size() const { return n2Size; }
    uint32_t get_gSize() const { return gSize; }
    uint32_t get_s1Size() const { return s1Size; }
    uint32_t get_s2Size() const { return s2Size; }
    uint32_t get_headDim() const { return headDim; }
    uint32_t get_headDimRope() const { return headDimRope; }
    uint32_t get_actualSeqS1Dims() const { return actualSeqS1Dims; }
    uint32_t get_actualSeqS2Dims() const { return actualSeqS2Dims; }
    float get_scaleValue() const { return scaleValue; }
    uint32_t get_usedCoreNum() const { return usedCoreNum; }
    uint32_t get_outputLayout() const { return outputLayout; }
    uint32_t get_softmaxLseFlag() const { return softmaxLseFlag; }
    bool get_slidingFlag() const { return slidingFlag; }
    bool get_needInit() const { return needInit; }

    void set_bSize(uint32_t bSize) { this->bSize = bSize; }
    void set_n2Size(uint32_t n2Size) { this->n2Size = n2Size; }
    void set_gSize(uint32_t gSize) { this->gSize = gSize; }
    void set_s1Size(uint32_t s1Size) { this->s1Size = s1Size; }
    void set_s2Size(uint32_t s2Size) { this->s2Size = s2Size; }
    void set_headDim(uint32_t headDim) { this->headDim = headDim; }
    void set_headDimRope(uint32_t headDimRope) { this->headDimRope = headDimRope; }
    void set_actualSeqS1Dims(uint32_t actualSeqS1Dims) { this->actualSeqS1Dims = actualSeqS1Dims; }
    void set_actualSeqS2Dims(uint32_t actualSeqS2Dims) { this->actualSeqS2Dims = actualSeqS2Dims; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }
    void set_usedCoreNum(uint32_t usedCoreNum) { this->usedCoreNum = usedCoreNum; }
    void set_outputLayout(uint32_t outputLayout) { this->outputLayout = outputLayout; }
    void set_softmaxLseFlag(uint32_t softmaxLseFlag) { this->softmaxLseFlag = softmaxLseFlag; }
    void set_slidingFlag(uint32_t slidingFlag) { this->slidingFlag = slidingFlag; }
    void set_needInit(uint32_t needInit) { this->needInit = needInit; }
};

// PageAttention 参数
class FiaPageAttentionTilingParam {
public:
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t get_blockSize() { return blockSize; }
    uint32_t get_maxBlockNumPerBatch() { return maxBlockNumPerBatch; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatch) { this->maxBlockNumPerBatch = maxBlockNumPerBatch; }
};

// AttenMask 参数
class FiaAttentionMaskTilingParam {
public:
    uint32_t attenMaskFlag;
    uint32_t attenMaskSize;
    int32_t preToken;
    int32_t nextToken;
    uint32_t isRowInvalid;
    uint32_t sparseMode;

    uint32_t get_attenMaskFlag() { return attenMaskFlag; }
    uint32_t get_attenMaskSize() { return attenMaskSize; }
    int32_t get_preToken() { return preToken; }
    int32_t get_nextToken() { return nextToken; }
    uint32_t get_isRowInvalid() { return isRowInvalid; }
    uint32_t get_sparseMode() { return sparseMode; }

    void set_attenMaskFlag(uint32_t attenMaskFlag) { this->attenMaskFlag = attenMaskFlag; }
    void set_attenMaskSize(uint32_t attenMaskSize) { this->attenMaskSize = attenMaskSize; }
    void set_preToken(int32_t preToken) { this->preToken = preToken; }
    void set_nextToken(int32_t nextToken) { this->nextToken = nextToken; }
    void set_isRowInvalid(uint32_t isRowInvalid) { this->isRowInvalid = isRowInvalid; }
    void set_sparseMode(uint32_t sparseMode) { this->sparseMode = sparseMode; }
};

// 内切基本块参数
class FiaInnerSplitTilingParam {
public:
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    uint32_t get_mBaseSize() { return mBaseSize; }
    uint32_t get_s2BaseSize() { return s2BaseSize; }
    void set_mBaseSize(uint32_t mBaseSize) { this->mBaseSize = mBaseSize; }
    void set_s2BaseSize(uint32_t s2BaseSize) { this->s2BaseSize = s2BaseSize; }
};

// workspace参数
class FiaWorkspaceTilingParam {
public:
    uint32_t mm1ResSize;
    uint32_t mm2ResSize;
    uint32_t fdAccumOutSize;
    uint32_t fdLogSumExpSize;

    uint32_t get_mm1ResSize() { return mm1ResSize; }
    uint32_t get_mm2ResSize() { return mm2ResSize; }
    uint32_t get_fdAccumOutSize() { return fdAccumOutSize; }
    uint32_t get_fdLogSumExpSize() { return fdLogSumExpSize; }

    void set_mm1ResSize(uint32_t mm1ResSize) { this->mm1ResSize = mm1ResSize; }
    void set_mm2ResSize(uint32_t mm2ResSize) { this->mm2ResSize = mm2ResSize; }
    void set_fdAccumOutSize(uint32_t fdAccumOutSize) { this->fdAccumOutSize = fdAccumOutSize; }
    void set_fdLogSumExpSize(uint32_t fdLogSumExpSize) { this->fdLogSumExpSize = fdLogSumExpSize; }
};

// 外切分核参数
class FiaOuterSplitTilingParam {
public:
    uint32_t bN2End[FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1End[FIA_MAX_AIC_CORE_NUM];
    uint32_t s2End[FIA_MAX_AIC_CORE_NUM];

    uint32_t *get_bN2End() { return bN2End; }
    uint32_t *get_gS1End() { return gS1End; }
    uint32_t *get_s2End() { return s2End; }
};

// FlashDecode规约参数
class FiaFlashDecodeTilingParam {
public:
    uint32_t numOfFdHead;
    uint32_t reserved;
    uint32_t gS1BaseSizeOfFd;
    uint32_t usedVecNumOfFd;
    uint32_t bN2IdxOfFdHead[FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxOfFdHead[FIA_MAX_AIC_CORE_NUM];
    uint32_t s2SplitNumOfFdHead[FIA_MAX_AIC_CORE_NUM];
    uint32_t s2SplitStartIdxOfCore[FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1SplitNumOfFdHead[FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1LastPartSizeOfFdHead[FIA_MAX_AIC_CORE_NUM];
    uint32_t gS1IdxEndOfFdHead[FIA_MAX_AIC_CORE_NUM * 2];
    uint32_t gS1IdxEndOfFdHeadSplit[FIA_MAX_AIC_CORE_NUM * 2];

    uint32_t get_numOfFdHead() { return numOfFdHead; }
    uint32_t get_reserved() { return reserved; }
    uint32_t get_gS1BaseSizeOfFd() { return gS1BaseSizeOfFd; }
    uint32_t get_usedVecNumOfFd() { return usedVecNumOfFd; }
    uint32_t *get_bN2IdxOfFdHead() { return bN2IdxOfFdHead; }
    uint32_t *get_gS1IdxOfFdHead() { return gS1IdxOfFdHead; }
    uint32_t *get_s2SplitNumOfFdHead() { return s2SplitNumOfFdHead; }
    uint32_t *get_s2SplitStartIdxOfCore() { return s2SplitStartIdxOfCore; }
    uint32_t *get_gS1SplitNumOfFdHead() { return gS1SplitNumOfFdHead; }
    uint32_t *get_gS1LastPartSizeOfFdHead() { return gS1LastPartSizeOfFdHead; }
    uint32_t *get_gS1IdxEndOfFdHead() { return gS1IdxEndOfFdHead; }
    uint32_t *get_gS1IdxEndOfFdHeadSplit() { return gS1IdxEndOfFdHeadSplit; }

    void set_numOfFdHead(uint32_t numOfFdHead) { this->numOfFdHead = numOfFdHead; }
    void set_reserved(uint32_t reserved) { this->reserved = reserved; }
    void set_gS1BaseSizeOfFd(uint32_t gS1BaseSizeOfFd) { this->gS1BaseSizeOfFd = gS1BaseSizeOfFd; }
    void set_usedVecNumOfFd(uint32_t usedVecNumOfFd) { this->usedVecNumOfFd = usedVecNumOfFd; }
};

// Left Padding 参数
class FiaLeftPaddingTilingParam {
public:
    uint32_t qPaddingFlag;
    uint32_t kvPaddingFlag;
    uint32_t get_qPaddingFlag() { return qPaddingFlag; }
    uint32_t get_kvPaddingFlag() { return kvPaddingFlag; }
    void set_qPaddingFlag(uint32_t qPaddingFlag) { this->qPaddingFlag = qPaddingFlag; }
    void set_kvPaddingFlag(uint32_t kvPaddingFlag) { this->kvPaddingFlag = kvPaddingFlag; }
};

// Pse 参数
class FiaPseShiftTilingParam {
public:
    uint32_t pseShiftFlag;
    uint32_t pseShiftB;
    uint32_t pseShiftS;

    uint32_t get_pseShiftFlag() { return pseShiftFlag; }
    uint32_t get_pseShiftB() { return pseShiftB; }
    uint32_t get_pseShiftS() { return pseShiftS; }

    void set_pseShiftFlag(uint32_t pseShiftFlag) { this->pseShiftFlag = pseShiftFlag; }
    void set_pseShiftB(uint32_t pseShiftB) { this->pseShiftB = pseShiftB; }
    void set_pseShiftS(uint32_t pseShiftS) { this->pseShiftS = pseShiftS; }
};

// 后量化 参数
class FiaPostQuantTilingParam {
public:
    uint32_t isPerChnOut;
    uint32_t isOutQuantTypeBf16;
    uint32_t get_isPerChnOut() { return isPerChnOut; }
    uint32_t get_isOutQuantTypeBf16() { return isOutQuantTypeBf16; }
    void set_isPerChnOut(uint32_t isPerChnOut) { this->isPerChnOut = isPerChnOut; }
    void set_isOutQuantTypeBf16(uint32_t isOutQuantTypeBf16) { this->isOutQuantTypeBf16 = isOutQuantTypeBf16; }
};

// 公共前缀 
class FiaPrefixTilingParam {
public:
    uint64_t prefixAttenOutOffset;
    uint64_t userPromptAttenOutOffset;
    uint64_t tmpLseOffset;
    uint64_t prefixLen;
    uint32_t formerCoreNum;
    uint32_t blockSplitBn2Range;
    uint32_t tailSplitedBatchRange;
    uint32_t batchSizeQ;

    uint64_t get_prefixAttenOutOffset() { return prefixAttenOutOffset; }
    uint64_t get_userPromptAttenOutOffset() { return userPromptAttenOutOffset; }
    uint64_t get_tmpLseOffset() { return tmpLseOffset; }
    uint64_t get_prefixLen() { return prefixLen; }
    uint32_t get_formerCoreNum() { return formerCoreNum; }
    uint32_t get_blockSplitBn2Range() { return blockSplitBn2Range; }
    uint32_t get_tailSplitedBatchRange() { return tailSplitedBatchRange; }
    uint32_t get_batchSizeQ() { return batchSizeQ; }

    void set_prefixAttenOutOffset(uint64_t prefixAttenOutOffset) { this->prefixAttenOutOffset = prefixAttenOutOffset; }
    void set_userPromptAttenOutOffset(uint64_t userPromptAttenOutOffset) { this->userPromptAttenOutOffset = userPromptAttenOutOffset; }
    void set_tmpLseOffset(uint64_t tmpLseOffset) { this->tmpLseOffset = tmpLseOffset; }
    void set_prefixLen(uint64_t prefixLen) { this->prefixLen = prefixLen; }
    void set_formerCoreNum(uint32_t formerCoreNum) { this->formerCoreNum = formerCoreNum; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2Range) { this->blockSplitBn2Range = blockSplitBn2Range; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRange) { this->tailSplitedBatchRange = tailSplitedBatchRange; }
    void set_batchSizeQ(uint32_t batchSizeQ) { this->batchSizeQ = batchSizeQ; }
};

// L2 Cache 参数
class FiaL2CacheTilingParam {
public:
    uint32_t l2CacheOffFlag;
    uint32_t get_l2CacheOffFlag() { return l2CacheOffFlag; }
    void set_l2CacheOffFlag(uint32_t l2CacheOffFlag) { this->l2CacheOffFlag = l2CacheOffFlag; }
};

// MSD 参数
class FiaMsdTilingParam {
public:
    uint32_t msdIterNum;
    uint32_t get_msdIterNum() { return msdIterNum; }
    void set_msdIterNum(uint32_t msdIterNum) { this->msdIterNum = msdIterNum; }
};

// 伪量化 参数
class FiaAntiquantTilingParam {
public:
    uint32_t antiqSeqSize;
    uint32_t get_antiqSeqSize() { return antiqSeqSize; }
    void set_antiqSeqSize(uint32_t antiqSeqSize) { this->antiqSeqSize = antiqSeqSize; }
};


//MLA非量化模板TilingData
class FusedInferAttentionScoreTilingData {
public:
    FiaBaseTilingParam baseParams;
    FiaPageAttentionTilingParam pageAttenParams;
    FiaAttentionMaskTilingParam maskParams;
    FiaWorkspaceTilingParam workspaceParams;
    FiaInnerSplitTilingParam innerSplitParams;
    FiaOuterSplitTilingParam outerSplitParams;
    FiaFlashDecodeTilingParam fdParams;
};
} // namespace optiling
#endif // FUSED_INFER_ATTENTION_SCORE_TILINGDATA_V3