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
 * \file prompt_flash_attention_tiling_data.h
 * \brief
 */
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

class PromptAttentionBaseParams {
public:
    uint32_t batchSize = 0;
    uint32_t headNumSize = 0;
    uint32_t seqSize = 0;
    uint32_t headSize = 0;
    float scaleValue = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    int32_t blockSize = 0;
    int32_t blockTableDim2 = 0;
    int32_t PABlockNumSum = 0;
    uint32_t dimNumOfseq = 0;
    uint32_t typeByteNum = 0;
    uint32_t seqInnerSize = 0;
    uint32_t prefixSeqInnerSize = 0;
    uint32_t usePseShift = 0;
    uint32_t useMask = 0;
    uint32_t headNumRatio = 0;
    uint32_t attenMaskElemType = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseMaskMaxSize = 0;
    uint32_t maskTypeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t sparseMode = 0;
    uint32_t alignedHeadSize = 0;
    uint32_t splitS2 = 0;
    uint32_t splitD = 0;
    uint32_t layoutType = 0;
    uint32_t PAlayoutType = 0;
    uint32_t pseShiftS1Size = 0;
    uint32_t pseShiftS2Size = 0;
    uint32_t maskKVsSize = 0;
    uint32_t maskQsSize = 0;
    uint32_t isLayoutSH = 0;
    uint32_t isActualSeqLengthsNull = 0;
    uint32_t isActualSeqLengthsKVNull = 0;
    uint32_t actualSeqLengthsSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    uint32_t deqScaleFlag = 0;
    uint32_t deqScale2Flag = 0;
    uint32_t isAntiPerchannel = 0;
    uint32_t isRowInvalid = 0;
    uint32_t softmaxOuterSize = 0;
    uint32_t isQuant2Perchannel = 0;
    uint32_t isQuant2BF16 = 0;
    uint32_t isKvContinuous = 0;
    uint32_t fromFused = 0;
    uint32_t isBSNDOut = 0;
    uint32_t isIFA = 0;
    uint32_t isSoftMaxLseEnable = 0;
    uint32_t isActualSharedPrefixLenNull = 0;
    uint32_t isQHasLeftPadding = 0;
    uint32_t isKVHasLeftPadding = 0;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    uint32_t hasKeyAntiquantOffset = 0;
    uint32_t isMsd = 0;
    uint32_t isQuant2FP16 = 0;
    uint32_t ropeHeadSize = 0;
    uint32_t qkHeadSize = 0;
    uint32_t vHeadSize = 0;
    uint32_t gOfMla = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_batchSize() const { return batchSize; }
    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }

    uint32_t get_headNumSize() const { return headNumSize; }
    void set_headNumSize(uint32_t headNumSize) { this->headNumSize = headNumSize; }

    uint32_t get_seqSize() const { return seqSize; }
    void set_seqSize(uint32_t seqSize) { this->seqSize = seqSize; }

    uint32_t get_headSize() const { return headSize; }
    void set_headSize(uint32_t headSize) { this->headSize = headSize; }

    float get_scaleValue() const { return scaleValue; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }

    int32_t get_preTokens() const { return preTokens; }
    void set_preTokens(int32_t preTokens) { this->preTokens = preTokens; }

    int32_t get_nextTokens() const { return nextTokens; }
    void set_nextTokens(int32_t nextTokens) { this->nextTokens = nextTokens; }

    int32_t get_blockSize() const { return blockSize; }
    void set_blockSize(int32_t blockSize) { this->blockSize = blockSize; }

    int32_t get_blockTableDim2() const { return blockTableDim2; }
    void set_blockTableDim2(int32_t blockTableDim2) { this->blockTableDim2 = blockTableDim2; }

    int32_t get_PABlockNumSum() const { return PABlockNumSum; }
    void set_PABlockNumSum(int32_t PABlockNumSum) { this->PABlockNumSum = PABlockNumSum; }

    uint32_t get_dimNumOfseq() const { return dimNumOfseq; }
    void set_dimNumOfseq(uint32_t dimNumOfseq) { this->dimNumOfseq = dimNumOfseq; }

    uint32_t get_typeByteNum() const { return typeByteNum; }
    void set_typeByteNum(uint32_t typeByteNum) { this->typeByteNum = typeByteNum; }

    uint32_t get_seqInnerSize() const { return seqInnerSize; }
    void set_seqInnerSize(uint32_t seqInnerSize) { this->seqInnerSize = seqInnerSize; }

    uint32_t get_prefixSeqInnerSize() const { return prefixSeqInnerSize; }
    void set_prefixSeqInnerSize(uint32_t prefixSeqInnerSize) { this->prefixSeqInnerSize = prefixSeqInnerSize; }

    uint32_t get_usePseShift() const { return usePseShift; }
    void set_usePseShift(uint32_t usePseShift) { this->usePseShift = usePseShift; }

    uint32_t get_useMask() const { return useMask; }
    void set_useMask(uint32_t useMask) { this->useMask = useMask; }

    uint32_t get_headNumRatio() const { return headNumRatio; }
    void set_headNumRatio(uint32_t headNumRatio) { this->headNumRatio = headNumRatio; }

    uint32_t get_attenMaskElemType() const { return attenMaskElemType; }
    void set_attenMaskElemType(uint32_t attenMaskElemType) { this->attenMaskElemType = attenMaskElemType; }

    uint32_t get_pseShiftTypeByteNum() const { return pseShiftTypeByteNum; }
    void set_pseShiftTypeByteNum(uint32_t pseShiftTypeByteNum) { this->pseShiftTypeByteNum = pseShiftTypeByteNum; }

    uint32_t get_pseMaskMaxSize() const { return pseMaskMaxSize; }
    void set_pseMaskMaxSize(uint32_t pseMaskMaxSize) { this->pseMaskMaxSize = pseMaskMaxSize; }

    uint32_t get_maskTypeByteNum() const { return maskTypeByteNum; }
    void set_maskTypeByteNum(uint32_t maskTypeByteNum) { this->maskTypeByteNum = maskTypeByteNum; }

    uint32_t get_outputTypeByteNum() const { return outputTypeByteNum; }
    void set_outputTypeByteNum(uint32_t outputTypeByteNum) { this->outputTypeByteNum = outputTypeByteNum; }

    uint32_t get_softmaxTypeByteNum() const { return softmaxTypeByteNum; }
    void set_softmaxTypeByteNum(uint32_t softmaxTypeByteNum) { this->softmaxTypeByteNum = softmaxTypeByteNum; }

    uint32_t get_sparseMode() const { return sparseMode; }
    void set_sparseMode(uint32_t sparseMode) { this->sparseMode = sparseMode; }

    uint32_t get_alignedHeadSize() const { return alignedHeadSize; }
    void set_alignedHeadSize(uint32_t alignedHeadSize) { this->alignedHeadSize = alignedHeadSize; }

    uint32_t get_splitS2() const { return splitS2; }
    void set_splitS2(uint32_t splitS2) { this->splitS2 = splitS2; }

    uint32_t get_splitD() const { return splitD; }
    void set_splitD(uint32_t splitD) { this->splitD = splitD; }

    uint32_t get_layoutType() const { return layoutType; }
    void set_layoutType(uint32_t layoutType) { this->layoutType = layoutType; }

    uint32_t get_PAlayoutType() const { return PAlayoutType; }
    void set_PAlayoutType(uint32_t PAlayoutType) { this->PAlayoutType = PAlayoutType; }

    uint32_t get_pseShiftS1Size() const { return pseShiftS1Size; }
    void set_pseShiftS1Size(uint32_t pseShiftS1Size) { this->pseShiftS1Size = pseShiftS1Size; }

    uint32_t get_pseShiftS2Size() const { return pseShiftS2Size; }
    void set_pseShiftS2Size(uint32_t pseShiftS2Size) { this->pseShiftS2Size = pseShiftS2Size; }

    uint32_t get_maskKVsSize() const { return maskKVsSize; }
    void set_maskKVsSize(uint32_t maskKVsSize) { this->maskKVsSize = maskKVsSize; }

    uint32_t get_maskQsSize() const { return maskQsSize; }
    void set_maskQsSize(uint32_t maskQsSize) { this->maskQsSize = maskQsSize; }

    uint32_t get_isLayoutSH() const { return isLayoutSH; }
    void set_isLayoutSH(uint32_t isLayoutSH) { this->isLayoutSH = isLayoutSH; }

    uint32_t get_isActualSeqLengthsNull() const { return isActualSeqLengthsNull; }
    void set_isActualSeqLengthsNull(uint32_t isActualSeqLengthsNull) { this->isActualSeqLengthsNull = isActualSeqLengthsNull; }

    uint32_t get_isActualSeqLengthsKVNull() const { return isActualSeqLengthsKVNull; }
    void set_isActualSeqLengthsKVNull(uint32_t isActualSeqLengthsKVNull) { this->isActualSeqLengthsKVNull = isActualSeqLengthsKVNull; }

    uint32_t get_actualSeqLengthsSize() const { return actualSeqLengthsSize; }
    void set_actualSeqLengthsSize(uint32_t actualSeqLengthsSize) { this->actualSeqLengthsSize = actualSeqLengthsSize; }

    uint32_t get_actualSeqLengthsKVSize() const { return actualSeqLengthsKVSize; }
    void set_actualSeqLengthsKVSize(uint32_t actualSeqLengthsKVSize) { this->actualSeqLengthsKVSize = actualSeqLengthsKVSize; }
    
    uint32_t get_deqScaleFlag() const { return deqScaleFlag; }
    void set_deqScaleFlag(uint32_t deqScaleFlag) { this->deqScaleFlag = deqScaleFlag; }

    uint32_t get_deqScale2Flag() const { return deqScale2Flag; }
    void set_deqScale2Flag(uint32_t deqScale2Flag) { this->deqScale2Flag = deqScale2Flag; }

    uint32_t get_isAntiPerchannel() const { return isAntiPerchannel; }
    void set_isAntiPerchannel(uint32_t isAntiPerchannel) { this->isAntiPerchannel = isAntiPerchannel; }

    uint32_t get_isRowInvalid() const { return isRowInvalid; }
    void set_isRowInvalid(uint32_t isRowInvalid) { this->isRowInvalid = isRowInvalid; }
    
    uint32_t get_softmaxOuterSize() const { return softmaxOuterSize; }
    void set_softmaxOuterSize(uint32_t softmaxOuterSize) { this->softmaxOuterSize = softmaxOuterSize; }

    uint32_t get_isQuant2Perchannel() const { return isQuant2Perchannel; }
    void set_isQuant2Perchannel(uint32_t isQuant2Perchannel) { this->isQuant2Perchannel = isQuant2Perchannel; }

    uint32_t get_isQuant2BF16() const { return isQuant2BF16; }
    void set_isQuant2BF16(uint32_t isQuant2BF16) { this->isQuant2BF16 = isQuant2BF16; }

    uint32_t get_isKvContinuous() const { return isKvContinuous; }
    void set_isKvContinuous(uint32_t isKvContinuous) { this->isKvContinuous = isKvContinuous; }

    uint32_t get_fromFused() const { return fromFused; }
    void set_fromFused(uint32_t fromFused) { this->fromFused = fromFused; }

    uint32_t get_isBSNDOut() const { return isBSNDOut; }
    void set_isBSNDOut(uint32_t isBSNDOut) { this->isBSNDOut = isBSNDOut; }

    uint32_t get_isIFA() const { return isIFA; }
    void set_isIFA(uint32_t isIFA) { this->isIFA = isIFA; }

    uint32_t get_isSoftMaxLseEnable() const { return isSoftMaxLseEnable; }
    void set_isSoftMaxLseEnable(uint32_t isSoftMaxLseEnable) { this->isSoftMaxLseEnable = isSoftMaxLseEnable; }

    uint32_t get_isActualSharedPrefixLenNull() const { return isActualSharedPrefixLenNull; }
    void set_isActualSharedPrefixLenNull(uint32_t isActualSharedPrefixLenNull) { this->isActualSharedPrefixLenNull = isActualSharedPrefixLenNull; }

    uint32_t get_isQHasLeftPadding() const { return isQHasLeftPadding; }
    void set_isQHasLeftPadding(uint32_t isQHasLeftPadding) { this->isQHasLeftPadding = isQHasLeftPadding; }

    uint32_t get_isKVHasLeftPadding() const { return isKVHasLeftPadding; }
    void set_isKVHasLeftPadding(uint32_t isKVHasLeftPadding) { this->isKVHasLeftPadding = isKVHasLeftPadding; }

    int64_t get_keyAntiquantMode() const { return keyAntiquantMode; }
    void set_keyAntiquantMode(int64_t keyAntiquantMode) { this->keyAntiquantMode = keyAntiquantMode; }

    int64_t get_valueAntiquantMode() const { return valueAntiquantMode; }
    void set_valueAntiquantMode(int64_t valueAntiquantMode) { this->valueAntiquantMode = valueAntiquantMode; }

    uint32_t get_hasKeyAntiquantOffset() const { return hasKeyAntiquantOffset; }
    void set_hasKeyAntiquantOffset(uint32_t hasKeyAntiquantOffset) { this->hasKeyAntiquantOffset = hasKeyAntiquantOffset; }

    uint32_t get_isMsd() const { return isMsd; }
    void set_isMsd(uint32_t isMsd) { this->isMsd = isMsd; }

    uint32_t get_isQuant2FP16() const { return isQuant2FP16; }
    void set_isQuant2FP16(uint32_t isQuant2FP16) { this->isQuant2FP16 = isQuant2FP16; }

    uint32_t get_ropeHeadSize() const { return ropeHeadSize; }
    void set_ropeHeadSize(uint32_t ropeHeadSize) { this->ropeHeadSize = ropeHeadSize; }

    uint32_t get_qkHeadSize() const { return qkHeadSize; }
    void set_qkHeadSize(uint32_t qkHeadSize) { this->qkHeadSize = qkHeadSize; }

    uint32_t get_vHeadSize() const { return vHeadSize; }
    void set_vHeadSize(uint32_t vHeadSize) { this->vHeadSize = vHeadSize; }

    uint32_t get_gOfMla() const { return gOfMla; }
    void set_gOfMla(uint32_t gOfMla) { this->gOfMla = gOfMla; }
};

class PromptAttentionBaseApiBaseParams {
public:
    uint32_t batchSize = 0;
    uint32_t headNumSize = 0;
    uint32_t headSize = 0;
    uint32_t maskTypeByteNum = 0;
  
    uint32_t inputLayoutType = 0;
    uint32_t kvHeadNumSize = 0;
    uint32_t maxSeqLen = 0;
    uint32_t maxKvSeqLen = 0;
    uint32_t totalQBlkNum = 0;
    uint32_t embeddingSizeV = 0;
    uint32_t quantType = 0;
    uint32_t dataShapeType = 0;
    uint32_t scaleType = 0;
    uint64_t workSize = 0;
    float tor = 0;
    uint32_t headStride = 0;
    uint32_t maskStride = 0;
    uint32_t isTriuMask = 0;
    uint32_t isClamp = 0;
    uint32_t clampMin = 0;
    uint32_t clampMax = 0;
    uint32_t tilingHeadSize = 0;
    uint32_t tilingParaSize = 0;
    uint32_t isLongSeq = 0;
    uint32_t isAlibiMaskSqrt = 0;
    uint32_t maskType = 0;
    uint32_t alibiCompressOffset = 0;
    uint32_t alibiLeftAlign = 0;
    uint32_t ppMScalar = 0;
    uint32_t ppNScalar = 0;
    uint32_t totalQBlkNumFirst = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_batchSize() const { return batchSize; }
    void set_batchSize(uint32_t batchSize) { this->batchSize = batchSize; }

    uint32_t get_headNumSize() const { return headNumSize; }
    void set_headNumSize(uint32_t headNumSize) { this->headNumSize = headNumSize; }

    uint32_t get_headSize() const { return headSize; }
    void set_headSize(uint32_t headSize) { this->headSize = headSize; }

    uint32_t get_maskTypeByteNum() const { return maskTypeByteNum; }
    void set_maskTypeByteNum(uint32_t maskTypeByteNum) { this->maskTypeByteNum = maskTypeByteNum; }

    uint32_t get_inputLayoutType() const { return inputLayoutType; }
    void set_inputLayoutType(uint32_t inputLayoutType) { this->inputLayoutType = inputLayoutType; }

    uint32_t get_kvHeadNumSize() const { return kvHeadNumSize; }
    void set_kvHeadNumSize(uint32_t kvHeadNumSize) { this->kvHeadNumSize = kvHeadNumSize; }

    uint32_t get_maxSeqLen() const { return maxSeqLen; }
    void set_maxSeqLen(uint32_t maxSeqLen) { this->maxSeqLen = maxSeqLen; }

    uint32_t get_maxKvSeqLen() const { return maxKvSeqLen; }
    void set_maxKvSeqLen(uint32_t maxKvSeqLen) { this->maxKvSeqLen = maxKvSeqLen; }

    uint32_t get_totalQBlkNum() const { return totalQBlkNum; }
    void set_totalQBlkNum(uint32_t totalQBlkNum) { this->totalQBlkNum = totalQBlkNum; }

    uint32_t get_embeddingSizeV() const { return embeddingSizeV; }
    void set_embeddingSizeV(uint32_t embeddingSizeV) { this->embeddingSizeV = embeddingSizeV; }

    uint32_t get_quantType() const { return quantType; }
    void set_quantType(uint32_t quantType) { this->quantType = quantType; }

    uint32_t get_dataShapeType() const { return dataShapeType; }
    void set_dataShapeType(uint32_t dataShapeType) { this->dataShapeType = dataShapeType; }

    uint32_t get_scaleType() const { return scaleType; }
    void set_scaleType(uint32_t scaleType) { this->scaleType = scaleType; }

    uint64_t get_workSize() const { return workSize; }
    void set_workSize(uint64_t workSize) { this->workSize = workSize; }

    float get_tor() const { return tor; }
    void set_tor(float tor) { this->tor = tor; }

    uint32_t get_headStride() const { return headStride; }
    void set_headStride(uint32_t headStride) { this->headStride = headStride; }

    uint32_t get_maskStride() const { return maskStride; }
    void set_maskStride(uint32_t maskStride) { this->maskStride = maskStride; }

    uint32_t get_isTriuMask() const { return isTriuMask; }
    void set_isTriuMask(uint32_t isTriuMask) { this->isTriuMask = isTriuMask; }

    uint32_t get_isClamp() const { return isClamp; }
    void set_isClamp(uint32_t isClamp) { this->isClamp = isClamp; }

    uint32_t get_clampMin() const { return clampMin; }
    void set_clampMin(uint32_t clampMin) { this->clampMin = clampMin; }

    uint32_t get_clampMax() const { return clampMax; }
    void set_clampMax(uint32_t clampMax) { this->clampMax = clampMax; }

    uint32_t get_tilingHeadSize() const { return tilingHeadSize; }
    void set_tilingHeadSize(uint32_t tilingHeadSize) { this->tilingHeadSize = tilingHeadSize; }

    uint32_t get_tilingParaSize() const { return tilingParaSize; }
    void set_tilingParaSize(uint32_t tilingParaSize) { this->tilingParaSize = tilingParaSize; }

    uint32_t get_isLongSeq() const { return isLongSeq; }
    void set_isLongSeq(uint32_t isLongSeq) { this->isLongSeq = isLongSeq; }

    uint32_t get_isAlibiMaskSqrt() const { return isAlibiMaskSqrt; }
    void set_isAlibiMaskSqrt(uint32_t isAlibiMaskSqrt) { this->isAlibiMaskSqrt = isAlibiMaskSqrt; }

    uint32_t get_maskType() const { return maskType; }
    void set_maskType(uint32_t maskType) { this->maskType = maskType; }

    uint32_t get_alibiCompressOffset() const { return alibiCompressOffset; }
    void set_alibiCompressOffset(uint32_t alibiCompressOffset) { this->alibiCompressOffset = alibiCompressOffset; }

    uint32_t get_alibiLeftAlign() const { return alibiLeftAlign; }
    void set_alibiLeftAlign(uint32_t alibiLeftAlign) { this->alibiLeftAlign = alibiLeftAlign; }

    uint32_t get_ppMScalar() const { return ppMScalar; }
    void set_ppMScalar(uint32_t ppMScalar) { this->ppMScalar = ppMScalar; }

    uint32_t get_ppNScalar() const { return ppNScalar; }
    void set_ppNScalar(uint32_t ppNScalar) { this->ppNScalar = ppNScalar; }

    uint32_t get_totalQBlkNumFirst() const { return totalQBlkNumFirst; }
    void set_totalQBlkNumFirst(uint32_t totalQBlkNumFirst) { this->totalQBlkNumFirst = totalQBlkNumFirst; }
};

class PromptAttentionSeqParams {
public:
  // Temporary reuse
   uint32_t CoreHeadNumTail[64] = {0};       // coreNStart
   uint32_t actualS1[64] = {0};             // coreNEnd
   uint32_t actualCoreNums[64] = {0};       // coreSidStart
   uint32_t singleCoreHeadNumSize[64] = {0};// coreSidEnd
   uint32_t coreSeqPosStart[64] = {0};
   uint32_t coreSeqPosEnd[64] = {0};

  // ========================
  // Getter & Setter 方法
  // ========================

  const uint32_t* get_CoreHeadNumTail() const { return CoreHeadNumTail; }
  void set_CoreHeadNumTail(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        CoreHeadNumTail[i] = values[i];
    }
  }

  
  const uint32_t* get_actualS1() const { return actualS1; }
  void set_actualS1(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        actualS1[i] = values[i];
    }
  }

  
  const uint32_t* get_actualCoreNums() const { return actualCoreNums; }
  void set_actualCoreNums(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        actualCoreNums[i] = values[i];
    }
  }

  
  const uint32_t* get_singleCoreHeadNumSize() const { return singleCoreHeadNumSize; }
  void set_singleCoreHeadNumSize(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        singleCoreHeadNumSize[i] = values[i];
    }
  }

  
  const uint32_t* get_coreSeqPosStart() const { return coreSeqPosStart; }
  void set_coreSeqPosStart(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        coreSeqPosStart[i] = values[i];
    }
  }

  
  const uint32_t* get_coreSeqPosEnd() const { return coreSeqPosEnd; }
  void set_coreSeqPosEnd(const uint32_t* values) { 
    for (int i = 0; i < 64; ++i) {
        coreSeqPosEnd[i] = values[i];
    }
  }
};

class PromptAttentionSplitCoreParams {
public:
   uint32_t startBlkArray[50] = {0};
   uint32_t endBlkArray[50] = {0};

  // ========================
  // Getter & Setter 方法
  // ========================

  const uint32_t* get_startBlkArray() const { return startBlkArray; }
  void set_startBlkArray(const uint32_t* values) { 
    for (int i = 0; i < 50; ++i) {
        startBlkArray[i] = values[i];
    }
  }
  
  const uint32_t* get_endBlkArray() const { return endBlkArray; }
  void set_endBlkArray(const uint32_t* values) { 
    for (int i = 0; i < 50; ++i) {
        endBlkArray[i] = values[i];
    }
  }
};

class PromptAttentionSingleCoreParams {
public:
    uint32_t singleProcessSInnerSize = 0;
    uint32_t singleProcessSOuterSize = 0;
    uint32_t multiSmaxsInnerLoopTimes = 0;
    uint32_t actualCoreNums = 0;
    uint32_t pseShiftBatch = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t kvAntiquantSInnerSize = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_singleProcessSInnerSize() const { return singleProcessSInnerSize; }
    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSize) { this->singleProcessSInnerSize = singleProcessSInnerSize; }

    uint32_t get_singleProcessSOuterSize() const { return singleProcessSOuterSize; }
    void set_singleProcessSOuterSize(uint32_t singleProcessSOuterSize) { this->singleProcessSOuterSize = singleProcessSOuterSize; }

    uint32_t get_multiSmaxsInnerLoopTimes() const { return multiSmaxsInnerLoopTimes; }
    void set_multiSmaxsInnerLoopTimes(uint32_t multiSmaxsInnerLoopTimes) { this->multiSmaxsInnerLoopTimes = multiSmaxsInnerLoopTimes; }

    uint32_t get_actualCoreNums() const { return actualCoreNums; }
    void set_actualCoreNums(uint32_t actualCoreNums) { this->actualCoreNums = actualCoreNums; }

    uint32_t get_pseShiftBatch() const { return pseShiftBatch; }
    void set_pseShiftBatch(uint32_t pseShiftBatch) { this->pseShiftBatch = pseShiftBatch; }

    uint32_t get_attenMaskBatch() const { return attenMaskBatch; }
    void set_attenMaskBatch(uint32_t attenMaskBatch) { this->attenMaskBatch = attenMaskBatch; }

    uint32_t get_kvAntiquantSInnerSize() const { return kvAntiquantSInnerSize; }
    void set_kvAntiquantSInnerSize(uint32_t kvAntiquantSInnerSize) { this->kvAntiquantSInnerSize = kvAntiquantSInnerSize; }
};

class PromptAttentionSingleCoreTensorSize {
  public:
    uint32_t mmResUbSize = 0;
    uint32_t pseShiftUbSize = 0;
    uint32_t attenMaskUbSize = 0;
    uint32_t maskSize = 0;
    uint32_t softmaxMaxSize = 0;
    uint32_t softmaxSumSize = 0;
    uint32_t softmaxExpSize = 0;
    uint32_t softmaxValueSize = 0;
    uint32_t spmTmpSize = 0;
    uint32_t scmTmpSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t tmpMMResBmm2PreUbSize = 0;
    uint32_t tmpSoftmaxBmm2UbSize = 0;
    uint32_t selectSpaceUbSize = 0;
    uint32_t tmpSoftMaxV2Size = 0;
    uint32_t mm1TmpUbSize = 0;
    uint32_t mm2TmpUbSize = 0;
    uint32_t kvAntiquantUbSize = 0;
    uint32_t bmm2ResUbMsdSize = 0;
    uint32_t tempBmm2QueueMsdSize = 0;
    uint32_t msdInQueueSize = 0;
    uint32_t msdQRowSumBuffSize = 0;
    uint32_t msdAMaxTmpBuffSize = 0;
    uint32_t msdAMaxResBuffSize = 0;
    uint32_t msdSoftmaxResAmaxBuffSize = 0;
    uint32_t msdSoftmaxRowSumScaleBuffSize = 0;
    uint32_t msdScaleBuffSize = 0;
    uint32_t msdOffsetBuffSize = 0;
    uint32_t msdTmpMm1BuffSize = 0;
    uint32_t msdTmpMm2BuffSize = 0;
    uint32_t msdOutQueueSize = 0;
    uint32_t msdComputeLines = 0;

    // ========================
    // Getter & Setter 方法
    // ========================

    uint32_t get_mmResUbSize() const { return mmResUbSize; }
    void set_mmResUbSize(uint32_t mmResUbSize) { this->mmResUbSize = mmResUbSize; }

    uint32_t get_pseShiftUbSize() const { return pseShiftUbSize; }
    void set_pseShiftUbSize(uint32_t pseShiftUbSize) { this->pseShiftUbSize = pseShiftUbSize; }

    uint32_t get_attenMaskUbSize() const { return attenMaskUbSize; }
    void set_attenMaskUbSize(uint32_t attenMaskUbSize) { this->attenMaskUbSize = attenMaskUbSize; }

    uint32_t get_maskSize() const { return maskSize; }
    void set_maskSize(uint32_t maskSize) { this->maskSize = maskSize; }

    uint32_t get_softmaxMaxSize() const { return softmaxMaxSize; }
    void set_softmaxMaxSize(uint32_t softmaxMaxSize) { this->softmaxMaxSize = softmaxMaxSize; }

    uint32_t get_softmaxSumSize() const { return softmaxSumSize; }
    void set_softmaxSumSize(uint32_t softmaxSumSize) { this->softmaxSumSize = softmaxSumSize; }

    uint32_t get_softmaxExpSize() const { return softmaxExpSize; }
    void set_softmaxExpSize(uint32_t softmaxExpSize) { this->softmaxExpSize = softmaxExpSize; }

    uint32_t get_softmaxValueSize() const { return softmaxValueSize; }
    void set_softmaxValueSize(uint32_t softmaxValueSize) { this->softmaxValueSize = softmaxValueSize; }

    uint32_t get_spmTmpSize() const { return spmTmpSize; }
    void set_spmTmpSize(uint32_t spmTmpSize) { this->spmTmpSize = spmTmpSize; }

    uint32_t get_scmTmpSize() const { return scmTmpSize; }
    void set_scmTmpSize(uint32_t scmTmpSize) { this->scmTmpSize = scmTmpSize; }

    uint32_t get_bmm2ResUbSize() const { return bmm2ResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }

    uint32_t get_tmpMMResBmm2PreUbSize() const { return tmpMMResBmm2PreUbSize; }
    void set_tmpMMResBmm2PreUbSize(uint32_t tmpMMResBmm2PreUbSize) { this->tmpMMResBmm2PreUbSize = tmpMMResBmm2PreUbSize; }

    uint32_t get_tmpSoftmaxBmm2UbSize() const { return tmpSoftmaxBmm2UbSize; }
    void set_tmpSoftmaxBmm2UbSize(uint32_t tmpSoftmaxBmm2UbSize) { this->tmpSoftmaxBmm2UbSize = tmpSoftmaxBmm2UbSize; }

    uint32_t get_selectSpaceUbSize() const { return selectSpaceUbSize; }
    void set_selectSpaceUbSize(uint32_t selectSpaceUbSize) { this->selectSpaceUbSize = selectSpaceUbSize; }

    uint32_t get_tmpSoftMaxV2Size() const { return tmpSoftMaxV2Size; }
    void set_tmpSoftMaxV2Size(uint32_t tmpSoftMaxV2Size) { this->tmpSoftMaxV2Size = tmpSoftMaxV2Size; }

    uint32_t get_mm1TmpUbSize() const { return mm1TmpUbSize; }
    void set_mm1TmpUbSize(uint32_t mm1TmpUbSize) { this->mm1TmpUbSize = mm1TmpUbSize; }

    uint32_t get_mm2TmpUbSize() const { return mm2TmpUbSize; }
    void set_mm2TmpUbSize(uint32_t mm2TmpUbSize) { this->mm2TmpUbSize = mm2TmpUbSize; }

    uint32_t get_kvAntiquantUbSize() const { return kvAntiquantUbSize; }
    void set_kvAntiquantUbSize(uint32_t kvAntiquantUbSize) { this->kvAntiquantUbSize = kvAntiquantUbSize; }

    uint32_t get_bmm2ResUbMsdSize() const { return bmm2ResUbMsdSize; }
    void set_bmm2ResUbMsdSize(uint32_t bmm2ResUbMsdSize) { this->bmm2ResUbMsdSize = bmm2ResUbMsdSize; }

    uint32_t get_tempBmm2QueueMsdSize() const { return tempBmm2QueueMsdSize; }
    void set_tempBmm2QueueMsdSize(uint32_t tempBmm2QueueMsdSize) { this->tempBmm2QueueMsdSize = tempBmm2QueueMsdSize; }

    uint32_t get_msdInQueueSize() const { return msdInQueueSize; }
    void set_msdInQueueSize(uint32_t msdInQueueSize) { this->msdInQueueSize = msdInQueueSize; }

    uint32_t get_msdQRowSumBuffSize() const { return msdQRowSumBuffSize; }
    void set_msdQRowSumBuffSize(uint32_t msdQRowSumBuffSize) { this->msdQRowSumBuffSize = msdQRowSumBuffSize; }

    uint32_t get_msdAMaxTmpBuffSize() const { return msdAMaxTmpBuffSize; }
    void set_msdAMaxTmpBuffSize(uint32_t msdAMaxTmpBuffSize) { this->msdAMaxTmpBuffSize = msdAMaxTmpBuffSize; }

    uint32_t get_msdAMaxResBuffSize() const { return msdAMaxResBuffSize; }
    void set_msdAMaxResBuffSize(uint32_t msdAMaxResBuffSize) { this->msdAMaxResBuffSize = msdAMaxResBuffSize; } 

    uint32_t get_msdSoftmaxResAmaxBuffSize() const { return msdSoftmaxResAmaxBuffSize; }
    void set_msdSoftmaxResAmaxBuffSize(uint32_t msdSoftmaxResAmaxBuffSize) { this->msdSoftmaxResAmaxBuffSize = msdSoftmaxResAmaxBuffSize; }

    uint32_t get_msdSoftmaxRowSumScaleBuffSize() const { return msdSoftmaxRowSumScaleBuffSize; }
    void set_msdSoftmaxRowSumScaleBuffSize(uint32_t msdSoftmaxRowSumScaleBuffSize) { this->msdSoftmaxRowSumScaleBuffSize = msdSoftmaxRowSumScaleBuffSize; }

    uint32_t get_msdScaleBuffSize() const { return msdScaleBuffSize; }
    void set_msdScaleBuffSize(uint32_t msdScaleBuffSize) { this->msdScaleBuffSize = msdScaleBuffSize; }

    uint32_t get_msdOffsetBuffSize() const { return msdOffsetBuffSize; }
    void set_msdOffsetBuffSize(uint32_t msdOffsetBuffSize) { this->msdOffsetBuffSize = msdOffsetBuffSize; }

    uint32_t get_msdTmpMm1BuffSize() const { return msdTmpMm1BuffSize; }
    void set_msdTmpMm1BuffSize(uint32_t msdTmpMm1BuffSize) { this->msdTmpMm1BuffSize = msdTmpMm1BuffSize; }

    uint32_t get_msdTmpMm2BuffSize() const { return msdTmpMm2BuffSize; }
    void set_msdTmpMm2BuffSize(uint32_t msdTmpMm2BuffSize) { this->msdTmpMm2BuffSize = msdTmpMm2BuffSize; }

    uint32_t get_msdOutQueueSize() const { return msdOutQueueSize; }
    void set_msdOutQueueSize(uint32_t msdOutQueueSize) { this->msdOutQueueSize = msdOutQueueSize; }

    uint32_t get_msdComputeLines() const { return msdComputeLines; }
    void set_msdComputeLines(uint32_t msdComputeLines) { this->msdComputeLines = msdComputeLines; }
};

class CopyTransposeTiling{
public:
    uint32_t dstShapeB = 0;
    uint32_t dstShapeN = 0;
    uint32_t dstShapeS = 0;
    uint32_t dstShapeHN = 0;
    uint32_t dstShapeH = 0;
    uint32_t srcShapeB = 0;
    uint32_t srcShapeN = 0;
    uint32_t srcShapeS = 0;
    uint32_t srcShapeHN = 0;
    uint32_t originalShapeNLen = 0;
    uint32_t shapeSHValue = 0;
    uint32_t shapeNsValue = 0;
    uint32_t shapeNsnValue = 0;
    uint32_t invalidParamCopyTransposeTiling = 0;
    uint32_t shapeBHValue = 0;
    uint32_t paramsAlign = 0;

    uint32_t get_dstShapeB() const { return dstShapeB; }
    void set_dstShapeB(uint32_t dstShapeB) { this->dstShapeB = dstShapeB; }

    uint32_t get_dstShapeN() const { return dstShapeN; }
    void set_dstShapeN(uint32_t dstShapeN) { this->dstShapeN = dstShapeN; }

    uint32_t get_dstShapeS() const { return dstShapeS; }
    void set_dstShapeS(uint32_t dstShapeS) { this->dstShapeS = dstShapeS; }

    uint32_t get_dstShapeHN() const { return dstShapeHN; }
    void set_dstShapeHN(uint32_t dstShapeHN) { this->dstShapeHN = dstShapeHN; }

    uint32_t get_dstShapeH() const { return dstShapeH; }
    void set_dstShapeH(uint32_t dstShapeH) { this->dstShapeH = dstShapeH; }

    uint32_t get_srcShapeB() const { return srcShapeB; }
    void set_srcShapeB(uint32_t srcShapeB) { this->srcShapeB = srcShapeB; }

    uint32_t get_srcShapeN() const { return srcShapeN; }
    void set_srcShapeN(uint32_t srcShapeN) { this->srcShapeN = srcShapeN; }

    uint32_t get_srcShapeS() const { return srcShapeS; }
    void set_srcShapeS(uint32_t srcShapeS) { this->srcShapeS = srcShapeS; }

    uint32_t get_srcShapeHN() const { return srcShapeHN; }
    void set_srcShapeHN(uint32_t srcShapeHN) { this->srcShapeHN = srcShapeHN; }

    uint32_t get_originalShapeNLen() const { return originalShapeNLen; }
    void set_originalShapeNLen(uint32_t originalShapeNLen) { this->originalShapeNLen = originalShapeNLen; }

    uint32_t get_shapeSHValue() const { return shapeSHValue; }
    void set_shapeSHValue(uint32_t shapeSHValue) { this->shapeSHValue = shapeSHValue; }

    uint32_t get_shapeNsValue() const { return shapeNsValue; }
    void set_shapeNsValue(uint32_t shapeNsValue) { this->shapeNsValue = shapeNsValue; }

    uint32_t get_shapeNsnValue() const { return shapeNsnValue; }
    void set_shapeNsnValue(uint32_t shapeNsnValue) { this->shapeNsnValue = shapeNsnValue; }

    uint32_t get_invalidParamCopyTransposeTiling() const { return invalidParamCopyTransposeTiling; }
    void set_invalidParamCopyTransposeTiling(uint32_t invalidParamCopyTransposeTiling) { this->invalidParamCopyTransposeTiling = invalidParamCopyTransposeTiling; }

    uint32_t get_shapeBHValue() const { return shapeBHValue; }
    void set_shapeBHValue(uint32_t shapeBHValue) { this->shapeBHValue = shapeBHValue; }

    uint32_t get_paramsAlign() const { return paramsAlign; }
    void set_paramsAlign(uint32_t paramsAlign) { this->paramsAlign = paramsAlign; }

    void GetDataCopyTransposeTiling(const int64_t dst0, const int64_t dst1, const int64_t dst2, const int64_t dst3,
                                    const int64_t src0, const int64_t src1, const int64_t src2, const int64_t src3,
                                    const uint32_t typeSize)
    {
        this->set_dstShapeB(dst0);
        this->set_dstShapeN(dst1);
        this->set_dstShapeS(dst2);
        this->set_dstShapeH(dst3);
        this->set_dstShapeHN(this->get_dstShapeH() / this->get_dstShapeN());

        this->set_srcShapeB(src0);
        this->set_srcShapeN(src1);
        this->set_srcShapeS(src2);
        this->set_srcShapeHN(src3);
        this->set_originalShapeNLen(this->get_srcShapeHN() * typeSize);
        this->set_shapeSHValue(this->get_dstShapeS() * this->get_dstShapeH());
        this->set_shapeNsValue(this->get_dstShapeN() * this->get_dstShapeS());
        this->set_shapeNsnValue(this->get_dstShapeN() * this->get_srcShapeS() * this->get_srcShapeN());
        this->set_shapeBHValue(this->get_dstShapeB() * this->get_dstShapeH());
    }

    void reset()
    {
        set_dstShapeB(0);
        set_dstShapeN(0);
        set_dstShapeS(0);
        set_dstShapeHN(0);
        set_dstShapeH(0);
        set_srcShapeB(0);
        set_srcShapeN(0);
        set_srcShapeS(0);
        set_srcShapeHN(0);
        set_originalShapeNLen(0);
        set_shapeSHValue(0);
        set_shapeNsValue(0);
        set_shapeNsnValue(0);
        set_invalidParamCopyTransposeTiling(0);
        set_shapeBHValue(0);
        set_paramsAlign(0);
    }
};

class PromptAttentionInitOutputParams {
public:
    uint32_t singleCoreSize = 0;
    int64_t totalOutputSize = 0;
    int64_t totalSoftMaxLseOutputSize = 0;
    uint32_t needInit = 0;
    uint32_t isOneN = 0;

    uint32_t get_singleCoreSize() const { return singleCoreSize; }
    void set_singleCoreSize(uint32_t singleCoreSize) { this->singleCoreSize = singleCoreSize; }

    int64_t get_totalOutputSize() const { return totalOutputSize; }
    void set_totalOutputSize(int64_t totalOutputSize) { this->totalOutputSize = totalOutputSize; }

    int64_t get_totalSoftMaxLseOutputSize() const { return totalSoftMaxLseOutputSize; }
    void set_totalSoftMaxLseOutputSize(int64_t totalSoftMaxLseOutputSize) { this->totalSoftMaxLseOutputSize = totalSoftMaxLseOutputSize; }

    uint32_t get_needInit() const { return needInit; }
    void set_needInit(uint32_t needInit) { this->needInit = needInit; }

    uint32_t get_isOneN() const { return isOneN; }
    void set_isOneN(uint32_t isOneN) { this->isOneN = isOneN; }
};

class PromptFlashAttentionTilingData {
public:
    AscendC::tiling::TCubeTiling bmm1TilingDataRect;
    AscendC::tiling::TCubeTiling bmm2TilingDataRect;

    PromptAttentionBaseParams promptAttentionBaseParams;
    PromptAttentionSeqParams promptAttentionSeqParams;
    PromptAttentionSingleCoreParams promptAttentionSingleCoreParams;
    PromptAttentionSingleCoreTensorSize promptAttentionTensorSizeRect;
    PromptAttentionInitOutputParams promptAttentionInitOutputParams;

    SoftMaxTiling softmaxTilingDataRect;
    SoftMaxTiling softmaxFlashTilingDataRect;
    CopyTransposeTiling transposeTilingDataRect;
};

class PFAInputParams {
public:
    int64_t bSize = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t alignedS2 = 0;
    int64_t dSize = 0;
    int64_t valueDSize = 0;
    float keepProb = 0;
    float scaleValue = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
  // in pse encoding scenes s1 and s2 might not equal with s1 s2 in Q K
    int64_t pseS1Size = 0;
    int64_t pseS2Size = 0;
    uint32_t pseBSize = 0;
    uint32_t bandIndex = 0;
    uint32_t blockSize = 0;
    uint32_t blockTableDim2 = 0;

  // 1: BSH/BSND 2: SBH 3: BNSD
    uint8_t layoutType = 0;
  // Paged Attention kvcache layout 0: BBH 1: BNBD 2: NZ
    uint32_t paCacheLayoutType = 0;
  // 0: (B,N2,G,S1,S2 { 1: (B,N2,G,1,S2 {
    uint8_t pseShapeType = 0;
  // 0: (B,N2,G,S1,S2 { 1: (B,1,1,S1,S2 { 2: (1,1,1,S1,S2 {
    uint8_t attenMaskShapeType = 0;
  // 0: fp16 1: bool(uint8 {
    uint8_t attenMaskDataType = 0;
  // ALL: 0 NONE: 1 ANY: 2 CAUSAL: 3 BAND: 4 } = 0;
    uint8_t attenMaskCompressMode = 0;
  // 0: high precise 1: high performance 2: invalid line high precise
    uint8_t implMode = 0;
    uint8_t sparseType = 0;
    uint8_t fromFused = 0;
    uint8_t pseEncodeType = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint16_t remain = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t pseType = 0;
    uint32_t rsv1 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    uint32_t hasLearnableSink = 0;

    int64_t get_bSize() const { return bSize; }
    void set_bSize(int64_t bSize) { this->bSize = bSize; }

    int64_t get_n2Size() const { return n2Size; }
    void set_n2Size(int64_t n2Size) { this->n2Size = n2Size; }

    int64_t get_gSize() const { return gSize; }
    void set_gSize(int64_t gSize) { this->gSize = gSize; }


    int64_t get_s1Size() const { return s1Size; }
    void set_s1Size(int64_t s1Size) { this->s1Size = s1Size; }


    int64_t get_s2Size() const { return s2Size; }
    void set_s2Size(int64_t s2Size) { this->s2Size = s2Size; }

    int64_t get_alignedS2() const { return alignedS2; }
    void set_alignedS2(int64_t alignedS2) { this->alignedS2 = alignedS2; }


    int64_t get_dSize() const { return dSize; }
    void set_dSize(int64_t dSize) { this->dSize = dSize; }

    int64_t get_valueDSize() const { return valueDSize; }
    void set_valueDSize(int64_t valueDSize) { this->valueDSize = valueDSize; }

    float get_keepProb() const { return keepProb; }
    void set_keepProb(float keepProb) { this->keepProb = keepProb; }

    float get_scaleValue() const { return scaleValue; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }

    int64_t get_preTokens() const { return preTokens; }
    void set_preTokens(int64_t preTokens) { this->preTokens = preTokens; }

    int64_t get_nextTokens() const { return nextTokens; }
    void set_nextTokens(int64_t nextTokens) { this->nextTokens = nextTokens; }

    int64_t get_pseS1Size() const { return pseS1Size; }
    void set_pseS1Size(int64_t pseS1Size) { this->pseS1Size = pseS1Size; }

    int64_t get_pseS2Size() const { return pseS2Size; }
    void set_pseS2Size(int64_t pseS2Size) { this->pseS2Size = pseS2Size; }

    uint32_t get_pseBSize() const { return pseBSize; }
    void set_pseBSize(uint32_t pseBSize) { this->pseBSize = pseBSize; }

    uint32_t get_bandIndex() const { return bandIndex; }
    void set_bandIndex(uint32_t bandIndex) { this->bandIndex = bandIndex; }

    uint32_t get_blockSize() const { return blockSize; }
    void set_blockSize(uint32_t blockSize) { this->blockSize = blockSize; }

    uint32_t get_blockTableDim2() const { return blockTableDim2; }
    void set_blockTableDim2(uint32_t blockTableDim2) { this->blockTableDim2 = blockTableDim2; }

    uint8_t get_layoutType() const { return layoutType; }
    void set_layoutType(uint8_t layoutType) { this->layoutType = layoutType; }

    uint32_t get_paCacheLayoutType() const { return paCacheLayoutType; }
    void set_paCacheLayoutType(uint32_t paCacheLayoutType) { this->paCacheLayoutType = paCacheLayoutType; }

    uint8_t get_pseShapeType() const { return pseShapeType; }
    void set_pseShapeType(uint8_t pseShapeType) { this->pseShapeType = pseShapeType; }

    uint8_t get_attenMaskShapeType() const { return attenMaskShapeType; }
    void set_attenMaskShapeType(uint8_t attenMaskShapeType) { this->attenMaskShapeType = attenMaskShapeType; }


    uint8_t get_attenMaskDataType() const { return attenMaskDataType; }
    void set_attenMaskDataType(uint8_t attenMaskDataType) { this->attenMaskDataType = attenMaskDataType; }

    uint8_t get_attenMaskCompressMode() const { return attenMaskCompressMode; }
    void set_attenMaskCompressMode(uint8_t attenMaskCompressMode) { this->attenMaskCompressMode = attenMaskCompressMode; }

    uint8_t get_implMode() const { return implMode; }
    void set_implMode(uint8_t implMode) { this->implMode = implMode; }

    uint8_t get_sparseType() const { return sparseType; }
    void set_sparseType(uint8_t sparseType) { this->sparseType = sparseType; }

    uint8_t get_fromFused() const { return fromFused; }
    void set_fromFused(uint8_t fromFused) { this->fromFused = fromFused; }

    uint8_t get_pseEncodeType() const { return pseEncodeType; }
    void set_pseEncodeType(uint8_t pseEncodeType) { this->pseEncodeType = pseEncodeType; }

    uint8_t get_isSoftMaxLseEnable() const { return isSoftMaxLseEnable; }
    void set_isSoftMaxLseEnable(uint8_t isSoftMaxLseEnable) { this->isSoftMaxLseEnable = isSoftMaxLseEnable; }

    uint16_t get_remain() const { return remain; }
    void set_remain(uint16_t remain) { this->remain = remain; }

    uint32_t get_attenMaskS2Size() const { return attenMaskS2Size; }
    void set_attenMaskS2Size(uint32_t attenMaskS2Size) { this->attenMaskS2Size = attenMaskS2Size; }

    uint32_t get_pseType() const { return pseType; }
    void set_pseType(uint32_t pseType) { this->pseType = pseType; }

    uint32_t get_rsv1() const { return rsv1; }
    void set_rsv1(uint32_t rsv1) { this->rsv1 = rsv1; }

    int64_t get_qStartIdx() const { return qStartIdx; }
    void set_qStartIdx(int64_t qStartIdx) { this->qStartIdx = qStartIdx; }

    int64_t get_kvStartIdx() const { return kvStartIdx; }
    void set_kvStartIdx(int64_t kvStartIdx) { this->kvStartIdx = kvStartIdx; }

    uint32_t get_hasLearnableSink() const { return hasLearnableSink; }
    void set_hasLearnableSink(uint32_t hasLearnableSink) { this->hasLearnableSink = hasLearnableSink; }
};

class PFAMultiCoreParams {
public:
  int32_t coreNum = 0;
  int32_t reserve = 0;
  // BN2GS1.o
  int64_t totalSize = 0;
  // BN2GS1.o / core_num
  int64_t splitFactorSize = 0;
  int64_t splitFactorTailSize = 0;
  int64_t sparseStartIdx[48] = {0};


  int32_t get_coreNum() const { return coreNum; }
  void set_coreNum(int32_t coreNum) { this->coreNum = coreNum; }

  int32_t get_reserve() const { return reserve; }
  void set_reserve(int32_t reserve) { this->reserve = reserve; }

  int64_t get_totalSize() const { return totalSize; }
  void set_totalSize(int64_t totalSize) { this->totalSize = totalSize; }

  int64_t get_splitFactorSize() const { return splitFactorSize; }
  void set_splitFactorSize(int64_t splitFactorSize) { this->splitFactorSize = splitFactorSize; }

  int64_t get_splitFactorTailSize() const { return splitFactorTailSize; }
  void set_splitFactorTailSize(int64_t splitFactorTailSize) { this->splitFactorTailSize = splitFactorTailSize; }

  int64_t* get_sparseStartIdx() { return sparseStartIdx; }
  void set_sparseStartIdx(const int64_t* values) { 
    for (int i = 0; i < 48; ++i) {
        sparseStartIdx[i] = values[i];
    }
  }
};

class PFACoreParams {
public:
    int32_t s1BaseSize = 0;
    int32_t s1BaseTailSize = 0;
    int64_t s1OuterSize = 0;
    int32_t s1Vec2BaseSize = 0;
    int32_t s1Vec2BaseTailSize = 0;
    int64_t s1Vec2OuterSize = 0;
    int32_t s2BaseSize = 0;
    int32_t s2BaseTailSize = 0;
    int64_t s2OuterSize = 0;
    int32_t dBaseSize = 0;
    int32_t dBaseTailSize = 0;
    int64_t dOuterSize = 0;
    int32_t bBaseSize = 0;
    int32_t bBaseTailSize = 0;
    int64_t bOuterSize = 0;
    int32_t n2BaseSize = 0;
    int32_t n2BaseTailSize = 0;
    int64_t n2OuterSize = 0;
    int32_t gBaseSize = 0;
    int32_t gBaseTailSize = 0;
    int64_t gOuterSize = 0;
    int32_t nRatio = 0;
    int32_t rsvd = 0;
    int64_t s1SparseValidSize = 0;
    int64_t s2SparseValidSize = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;

    int32_t get_s1BaseSize() const { return s1BaseSize; }
    void set_s1BaseSize(int32_t s1BaseSize) { this->s1BaseSize = s1BaseSize; }

    int32_t get_s1BaseTailSize() const { return s1BaseTailSize; }
    void set_s1BaseTailSize(int32_t s1BaseTailSize) { this->s1BaseTailSize = s1BaseTailSize; }

    int64_t get_s1OuterSize() const { return s1OuterSize; }
    void set_s1OuterSize(int64_t s1OuterSize) { this->s1OuterSize = s1OuterSize; }

    int32_t get_s1Vec2BaseSize() const { return s1Vec2BaseSize; }
    void set_s1Vec2BaseSize(int32_t s1Vec2BaseSize) { this->s1Vec2BaseSize = s1Vec2BaseSize; }

    int32_t get_s1Vec2BaseTailSize() const { return s1Vec2BaseTailSize; }
    void set_s1Vec2BaseTailSize(int32_t s1Vec2BaseTailSize) { this->s1Vec2BaseTailSize = s1Vec2BaseTailSize; }

    int64_t get_s1Vec2OuterSize() const { return s1Vec2OuterSize; }
    void set_s1Vec2OuterSize(int64_t s1Vec2OuterSize) { this->s1Vec2OuterSize = s1Vec2OuterSize; }

    int32_t get_s2BaseSize() const { return s2BaseSize; }
    void set_s2BaseSize(int32_t s2BaseSize) { this->s2BaseSize = s2BaseSize; }

    int32_t get_s2BaseTailSize() const { return s2BaseTailSize; }
    void set_s2BaseTailSize(int32_t s2BaseTailSize) { this->s2BaseTailSize = s2BaseTailSize; }

    int64_t get_s2OuterSize() const { return s2OuterSize; }
    void set_s2OuterSize(int64_t s2OuterSize) { this->s2OuterSize = s2OuterSize; }

    int32_t get_dBaseSize() const { return dBaseSize; }
    void set_dBaseSize(int32_t dBaseSize) { this->dBaseSize = dBaseSize; }

    int32_t get_dBaseTailSize() const { return dBaseTailSize; }
    void set_dBaseTailSize(int32_t dBaseTailSize) { this->dBaseTailSize = dBaseTailSize; }

    int64_t get_dOuterSize() const { return dOuterSize; }
    void set_dOuterSize(int64_t dOuterSize) { this->dOuterSize = dOuterSize; }

    int32_t get_bBaseSize() const { return bBaseSize; }
    void set_bBaseSize(int32_t bBaseSize) { this->bBaseSize = bBaseSize; }

    int32_t get_bBaseTailSize() const { return bBaseTailSize; }
    void set_bBaseTailSize(int32_t bBaseTailSize) { this->bBaseTailSize = bBaseTailSize; }

    int64_t get_bOuterSize() const { return bOuterSize; }
    void set_bOuterSize(int64_t bOuterSize) { this->bOuterSize = bOuterSize; }

    int32_t get_n2BaseSize() const { return n2BaseSize; }
    void set_n2BaseSize(int32_t n2BaseSize) { this->n2BaseSize = n2BaseSize; }

    int32_t get_n2BaseTailSize() const { return n2BaseTailSize; }
    void set_n2BaseTailSize(int32_t n2BaseTailSize) { this->n2BaseTailSize = n2BaseTailSize; }

    int64_t get_n2OuterSize() const { return n2OuterSize; }
    void set_n2OuterSize(int64_t n2OuterSize) { this->n2OuterSize = n2OuterSize; }

    int32_t get_gBaseSize() const { return gBaseSize; }
    void set_gBaseSize(int32_t gBaseSize) { this->gBaseSize = gBaseSize; }

    int32_t get_gBaseTailSize() const { return gBaseTailSize; }
    void set_gBaseTailSize(int32_t gBaseTailSize) { this->gBaseTailSize = gBaseTailSize; }

    int64_t get_gOuterSize() const { return gOuterSize; }
    void set_gOuterSize(int64_t gOuterSize) { this->gOuterSize = gOuterSize; }

    int32_t get_nRatio() const { return nRatio; }
    void set_nRatio(int32_t nRatio) { this->nRatio = nRatio; }
    
    int32_t get_rsvd() const { return rsvd; }
    void set_rsvd(int32_t rsvd) { this->rsvd = rsvd; }

    int64_t get_s1SparseValidSize() const { return s1SparseValidSize; }
    void set_s1SparseValidSize(int64_t s1SparseValidSize) { this->s1SparseValidSize = s1SparseValidSize; }

    int64_t get_s2SparseValidSize() const { return s2SparseValidSize; }
    void set_s2SparseValidSize(int64_t s2SparseValidSize) { this->s2SparseValidSize = s2SparseValidSize; }

    int64_t get_pseAlibiBaseS1() const { return pseAlibiBaseS1; }
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1) { this->pseAlibiBaseS1 = pseAlibiBaseS1; }

    int64_t get_pseAlibiBaseS2() const { return pseAlibiBaseS2; }
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2) { this->pseAlibiBaseS2 = pseAlibiBaseS2; }
};

class PFATensorSizeParams {
public:
    int32_t bmm1ResUbSize = 0;
    int32_t attenMaskUbSize = 0;
    int32_t pseUbSize = 0;
    int32_t dropMaskUbSize = 0;
    int32_t castUbSize = 0;
    int32_t softmaxMaxUbSize = 0;
    int32_t softmaxSumUbSize = 0;
    int32_t softmaxExpUbSize = 0;
    int32_t apiTmpBufferBytes = 0;
    int32_t bmm2ResUbSize = 0;
    int32_t inputQueBytes = 0;
    int32_t outputQueBytes = 0;
  // API buffer use remain space of ub
    int32_t tmpBufBytes = 0;
    int32_t softmaxMaxOffsetBytes = 0;
    int32_t softmaxSumOffsetBytes = 0;
    int32_t maxSumApiOffsetBytes = 0;
    int32_t customSoftmaxApiOffsetBytes = 0;
    int32_t pseTbufOffsetBytes = 0;
    int32_t dropoutApiOffsetBytes = 0;
    int32_t maxSumApiSize = 0;
    int32_t customSoftmaxApiSize = 0;
    int32_t dropoutApiSize = 0;
    int32_t attenMaskApiSize = 0;
    int32_t attenMaskApiOffsetBytes = 0;
    int32_t bmm1ProcessTInStage2Size = 0;
    int32_t bmm1ProcessTInStage2OffsetBytes = 0;
  // workspace
    int32_t wkspSection1OffsetBytes = 0;
    int32_t wkspSection2OffsetBytes = 0;

    int32_t get_bmm1ResUbSize() const { return bmm1ResUbSize; }
    void set_bmm1ResUbSize(int32_t bmm1ResUbSize) { this->bmm1ResUbSize = bmm1ResUbSize; }

    int32_t get_attenMaskUbSize() const { return attenMaskUbSize; }
    void set_attenMaskUbSize(int32_t attenMaskUbSize) { this->attenMaskUbSize = attenMaskUbSize; }

    int32_t get_pseUbSize() const { return pseUbSize; }
    void set_pseUbSize(int32_t pseUbSize) { this->pseUbSize = pseUbSize; }

    int32_t get_dropMaskUbSize() const { return dropMaskUbSize; }
    void set_dropMaskUbSize(int32_t dropMaskUbSize) { this->dropMaskUbSize = dropMaskUbSize; }

    int32_t get_castUbSize() const { return castUbSize; }
    void set_castUbSize(int32_t castUbSize) { this->castUbSize = castUbSize; }

    int32_t get_softmaxMaxUbSize() const { return softmaxMaxUbSize; }
    void set_softmaxMaxUbSize(int32_t softmaxMaxUbSize) { this->softmaxMaxUbSize = softmaxMaxUbSize; }

    int32_t get_softmaxSumUbSize() const { return softmaxSumUbSize; }
    void set_softmaxSumUbSize(int32_t softmaxSumUbSize) { this->softmaxSumUbSize = softmaxSumUbSize; }

    int32_t get_softmaxExpUbSize() const { return softmaxExpUbSize; }
    void set_softmaxExpUbSize(int32_t softmaxExpUbSize) { this->softmaxExpUbSize = softmaxExpUbSize; }

    int32_t get_apiTmpBufferBytes() const { return apiTmpBufferBytes; }
    void set_apiTmpBufferBytes(int32_t apiTmpBufferBytes) { this->apiTmpBufferBytes = apiTmpBufferBytes; }

    int32_t get_bmm2ResUbSize() const { return bmm2ResUbSize; }
    void set_bmm2ResUbSize(int32_t bmm2ResUbSize) { this->bmm2ResUbSize = bmm2ResUbSize; }

    int32_t get_inputQueBytes() const { return inputQueBytes; }
    void set_inputQueBytes(int32_t inputQueBytes) { this->inputQueBytes = inputQueBytes; }

    int32_t get_outputQueBytes() const { return outputQueBytes; }
    void set_outputQueBytes(int32_t outputQueBytes) { this->outputQueBytes = outputQueBytes; }

    int32_t get_tmpBufBytes() const { return tmpBufBytes; }
    void set_tmpBufBytes(int32_t tmpBufBytes) { this->tmpBufBytes = tmpBufBytes; }

    int32_t get_softmaxMaxOffsetBytes() const { return softmaxMaxOffsetBytes; }
    void set_softmaxMaxOffsetBytes(int32_t softmaxMaxOffsetBytes) { this->softmaxMaxOffsetBytes = softmaxMaxOffsetBytes; }

    int32_t get_softmaxSumOffsetBytes() const { return softmaxSumOffsetBytes; }
    void set_softmaxSumOffsetBytes(int32_t softmaxSumOffsetBytes) { this->softmaxSumOffsetBytes = softmaxSumOffsetBytes; }

    int32_t get_maxSumApiOffsetBytes() const { return maxSumApiOffsetBytes; }
    void set_maxSumApiOffsetBytes(int32_t maxSumApiOffsetBytes) { this->maxSumApiOffsetBytes = maxSumApiOffsetBytes; }

    int32_t get_customSoftmaxApiOffsetBytes() const { return customSoftmaxApiOffsetBytes; }
    void set_customSoftmaxApiOffsetBytes(int32_t s2BaseSize) { this->customSoftmaxApiOffsetBytes = customSoftmaxApiOffsetBytes; }

    int32_t get_pseTbufOffsetBytes() const { return pseTbufOffsetBytes; }
    void set_pseTbufOffsetBytes(int32_t pseTbufOffsetBytes) { this->pseTbufOffsetBytes = pseTbufOffsetBytes; }

    int32_t get_dropoutApiOffsetBytes() const { return dropoutApiOffsetBytes; }
    void set_dropoutApiOffsetBytes(int32_t dropoutApiOffsetBytes) { this->dropoutApiOffsetBytes = dropoutApiOffsetBytes; }

    int32_t get_maxSumApiSize() const { return maxSumApiSize; }
    void set_maxSumApiSize(int32_t maxSumApiSize) { this->maxSumApiSize = maxSumApiSize; }

    int32_t get_customSoftmaxApiSize() const { return customSoftmaxApiSize; }
    void set_customSoftmaxApiSize(int32_t customSoftmaxApiSize) { this->customSoftmaxApiSize = customSoftmaxApiSize; }

    int32_t get_dropoutApiSize() const { return dropoutApiSize; }
    void set_dropoutApiSize(int32_t dropoutApiSize) { this->dropoutApiSize = dropoutApiSize; }

    int32_t get_attenMaskApiSize() const { return attenMaskApiSize; }
    void set_attenMaskApiSize(int32_t attenMaskApiSize) { this->attenMaskApiSize = attenMaskApiSize; }

    int32_t get_attenMaskApiOffsetBytes() const { return attenMaskApiOffsetBytes; }
    void set_attenMaskApiOffsetBytes(int32_t attenMaskApiOffsetBytes) { this->attenMaskApiOffsetBytes = attenMaskApiOffsetBytes; }

    int32_t get_bmm1ProcessTInStage2Size() const { return bmm1ProcessTInStage2Size; }
    void set_bmm1ProcessTInStage2Size(int32_t bmm1ProcessTInStage2Size) { this->bmm1ProcessTInStage2Size = bmm1ProcessTInStage2Size; }

    int32_t get_bmm1ProcessTInStage2OffsetBytes() const { return bmm1ProcessTInStage2OffsetBytes; }
    void set_bmm1ProcessTInStage2OffsetBytes(int32_t bmm1ProcessTInStage2OffsetBytes) { this->bmm1ProcessTInStage2OffsetBytes = bmm1ProcessTInStage2OffsetBytes; }

    int32_t get_wkspSection1OffsetBytes() const { return wkspSection1OffsetBytes; }
    void set_wkspSection1OffsetBytes(int32_t wkspSection1OffsetBytes) { this->wkspSection1OffsetBytes = wkspSection1OffsetBytes; }

    int32_t get_wkspSection2OffsetBytes() const { return wkspSection2OffsetBytes; }
    void set_wkspSection2OffsetBytes(int32_t wkspSection2OffsetBytes) { this->wkspSection2OffsetBytes = wkspSection2OffsetBytes; }
};

class MLAGeneralTilingData {
public:
    PFAInputParams PFAinputParams;
    PFAMultiCoreParams PFAmultiCoreParams;
    PFACoreParams PFAcoreParams;
    PFATensorSizeParams PFAtensorSizeParams;
    AscendC::tiling::TCubeTiling bmm1TilingData;
    AscendC::tiling::TCubeTiling bmm2TilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    CopyTransposeTiling transposeTilingDataTailCore;
};

class PromptFlashAttentionBaseApiTilingData {
public:
    PromptAttentionBaseApiBaseParams promptAttentionBaseApiBaseParams;
    PromptAttentionSplitCoreParams promptAttentionSplitCoreParams;
};

class InputParamsRegbase {
public:
    int64_t bSize = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t alignedS2 = 0;
    int64_t dSize = 0;
    int64_t dSizeV = 0;
    int64_t dSizeRope = 0;
    float keepProb = 0;
    float scaleValue = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
  // in pse encoding scenes s1 and s2 might not equal with s1 s2 in Q K
    int64_t pseS1Size = 0;
    int64_t pseS2Size = 0;
    uint32_t pseBSize = 0;
    uint32_t bandIndex = 0;

  // 1: BSH/BSND 2: SBH 3: BNSD
    uint8_t layoutType = 0;
  // 0: (B,N2,G,S1,S2 { 1: (B,N2,G,1,S2 {
    uint8_t pseShapeType = 0;
  // 0: (B,N2,G,S1,S2 { 1: (B,1,1,S1,S2 { 2: (1,1,1,S1,S2 {
    uint8_t attenMaskShapeType = 0;
  // 0: fp16 1: bool(uint8 {
    uint8_t attenMaskDataType = 0;
  // ALL: 0 NONE: 1 ANY: 2 CAUSAL: 3 BAND: 4 } = 0;
    uint8_t attenMaskCompressMode = 0;
  // 0: high precise 1: high performance 2: invalid line high precise
    uint8_t implMode = 0;
    uint8_t sparseType = 0;
    uint8_t needDropMaskOp = 0;
    uint8_t dropMaskOuter = 0;
    uint8_t pseEncodeType = 0;
    uint16_t remain = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t pseType = 0;
    uint32_t rsv1 = 0;
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    int64_t s1SparseValidSize = 0;
    int64_t s2SparseValidSize = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t keepProbUint8 = 0;
    int64_t pseAlibiBaseS1 = 0;
    int64_t pseAlibiBaseS2 = 0;

  // PFA
    uint8_t deqScaleFlag = 0;
    uint8_t deqScale2Flag = 0;
    uint8_t isActualSeqLengthsNull = 0;
    uint8_t isActualSeqLengthsKVNull = 0;
    uint32_t actualSeqLengthsSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    uint8_t isKvContinuous = 0;
    uint8_t fromFused = 0;
    uint8_t isBSNDOut = 0;
    uint8_t isGqa = 0;
    uint8_t isSoftMaxLseEnable = 0;
    uint8_t isActualSharedPrefixLenNull = 0;
    uint8_t isQHasLeftPadding = 0;
    uint8_t isKVHasLeftPadding = 0;
    uint32_t ropeHeadSize = 0;
    uint32_t prefixSeqInnerSize = 0;
    uint32_t headNumRatio = 0;
    int32_t blockSize = 0;
    int32_t blockTableDim2 = 0;
    int32_t paBlockNumSum = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t kvSplitPart = 0;
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;

    uint8_t paLayoutType = 0;
    uint8_t isRowInvalid = 0;
    uint8_t isPostQuantPerChnl = 0;
    uint8_t isPostQuantBF16 = 0;
    uint16_t antiquantPerTensorFlag = 0;
    uint16_t antiquantPerHeadFlag = 0;
    uint32_t antiquantParaSeqSize = 0;

    int64_t get_bSize() const { return bSize; }
    void set_bSize(int64_t bSize) { this->bSize = bSize; }

    int64_t get_n2Size() const { return n2Size; }
    void set_n2Size(int64_t n2Size) { this->n2Size = n2Size; }

    int64_t get_gSize() const { return gSize; }
    void set_gSize(int64_t gSize) { this->gSize = gSize; }

    int64_t get_s1Size() const { return s1Size; }
    void set_s1Size(int64_t s1Size) { this->s1Size = s1Size; }

    int64_t get_s2Size() const { return s2Size; }
    void set_s2Size(int64_t s2Size) { this->s2Size = s2Size; }

    int64_t get_alignedS2() const { return alignedS2; }
    void set_alignedS2(int64_t alignedS2) { this->alignedS2 = alignedS2; }

    int64_t get_dSize() const { return dSize; }
    void set_dSize(int64_t dSize) { this->dSize = dSize; }

    int64_t get_dSizeV() const { return dSizeV; }
    void set_dSizeV(int64_t dSizeV) { this->dSizeV = dSizeV; }

    int64_t get_dSizeRope() const { return dSizeRope; }
    void set_dSizeRope(int64_t dSizeRope) { this->dSizeRope = dSizeRope; }

    float get_keepProb() const { return keepProb; }
    void set_keepProb(float keepProb) { this->keepProb = keepProb; }

    float get_scaleValue() const { return scaleValue; }
    void set_scaleValue(float scaleValue) { this->scaleValue = scaleValue; }

    int64_t get_preTokens() const { return preTokens; }
    void set_preTokens(int64_t preTokens) { this->preTokens = preTokens; }

    int64_t get_nextTokens() const { return nextTokens; }
    void set_nextTokens(int64_t nextTokens) { this->nextTokens = nextTokens; }

    int64_t get_pseS1Size() const { return pseS1Size; }
    void set_pseS1Size(int64_t pseS1Size) { this->pseS1Size = pseS1Size; }

    int64_t get_pseS2Size() const { return pseS2Size; }
    void set_pseS2Size(int64_t pseS2Size) { this->pseS2Size = pseS2Size; }

    uint32_t get_pseBSize() const { return pseBSize; }
    void set_pseBSize(uint32_t pseBSize) { this->pseBSize = pseBSize; }

    uint32_t get_bandIndex() const { return bandIndex; }
    void set_bandIndex(uint32_t bandIndex) { this->bandIndex = bandIndex; }

    uint8_t get_layoutType() const { return layoutType; }
    void set_layoutType(uint8_t layoutType) { this->layoutType = layoutType; }
    
    uint8_t get_pseShapeType() const { return pseShapeType; }
    void set_pseShapeType(uint8_t pseShapeType) { this->pseShapeType = pseShapeType; }

    uint8_t get_attenMaskShapeType() const { return attenMaskShapeType; }
    void set_attenMaskShapeType(uint8_t attenMaskShapeType) { this->attenMaskShapeType = attenMaskShapeType; }

    uint8_t get_attenMaskDataType() const { return attenMaskDataType; }
    void set_attenMaskDataType(uint8_t attenMaskDataType) { this->attenMaskDataType = attenMaskDataType; }

    uint8_t get_attenMaskCompressMode() const { return attenMaskCompressMode; }
    void set_attenMaskCompressMode(uint8_t attenMaskCompressMode) { this->attenMaskCompressMode = attenMaskCompressMode; }

    uint8_t get_implMode() const { return implMode; }
    void set_implMode(uint8_t implMode) { this->implMode = implMode; }

    uint8_t get_sparseType() const { return sparseType; }
    void set_sparseType(uint8_t sparseType) { this->sparseType = sparseType; }

    uint8_t get_needDropMaskOp() const { return needDropMaskOp; }
    void set_needDropMaskOp(uint8_t needDropMaskOp) { this->needDropMaskOp = needDropMaskOp; }

    uint8_t get_dropMaskOuter() const { return dropMaskOuter; }
    void set_dropMaskOuter(uint8_t dropMaskOuter) { this->dropMaskOuter = dropMaskOuter; }

    uint8_t get_pseEncodeType() const { return pseEncodeType; }
    void set_pseEncodeType(uint8_t pseEncodeType) { this->pseEncodeType = pseEncodeType; }

    uint16_t get_remain() const { return remain; }
    void set_remain(uint16_t remain) { this->remain = remain; }

    uint32_t get_attenMaskS2Size() const { return attenMaskS2Size; }
    void set_attenMaskS2Size(uint32_t attenMaskS2Size) { this->attenMaskS2Size = attenMaskS2Size; }

    uint32_t get_pseType() const { return pseType; }
    void set_pseType(uint32_t pseType) { this->pseType = pseType; }

    uint32_t get_rsv1() const { return rsv1; }
    void set_rsv1(uint32_t rsv1) { this->rsv1 = rsv1; }

    int64_t get_qStartIdx() const { return qStartIdx; }
    void set_qStartIdx(int64_t qStartIdx) { this->qStartIdx = qStartIdx; }

    int64_t get_kvStartIdx() const { return kvStartIdx; }
    void set_kvStartIdx(int64_t kvStartIdx) { this->kvStartIdx = kvStartIdx; }

    int64_t get_s1SparseValidSize() const { return s1SparseValidSize; }
    void set_s1SparseValidSize(int64_t s1SparseValidSize) { this->s1SparseValidSize = s1SparseValidSize; }

    int64_t get_s2SparseValidSize() const { return s2SparseValidSize; }
    void set_s2SparseValidSize(int64_t s2SparseValidSize) { this->s2SparseValidSize = s2SparseValidSize; }

    int64_t get_seed() const { return seed; }
    void set_seed(int64_t seed) { this->seed = seed; }

    int64_t get_offset() const { return offset; }
    void set_offset(int64_t offset) { this->offset = offset; }

    int64_t get_keepProbUint8() const { return keepProbUint8; }
    void set_keepProbUint8(int64_t keepProbUint8) { this->keepProbUint8 = keepProbUint8; }

    int64_t get_pseAlibiBaseS1() const { return pseAlibiBaseS1; }
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1) { this->pseAlibiBaseS1 = pseAlibiBaseS1; }

    int64_t get_pseAlibiBaseS2() const { return pseAlibiBaseS2; }
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2) { this->pseAlibiBaseS2 = pseAlibiBaseS2; }

    uint8_t get_deqScaleFlag() const { return deqScaleFlag; }
    void set_deqScaleFlag(uint8_t deqScaleFlag) { this->deqScaleFlag = deqScaleFlag; }

    uint8_t get_deqScale2Flag() const { return deqScale2Flag; }
    void set_deqScale2Flag(uint8_t deqScale2Flag) { this->deqScale2Flag = deqScale2Flag; }

    uint8_t get_isActualSeqLengthsNull() const { return isActualSeqLengthsNull; }
    void set_isActualSeqLengthsNull(uint8_t isActualSeqLengthsNull) { this->isActualSeqLengthsNull = isActualSeqLengthsNull; }

    uint8_t get_isActualSeqLengthsKVNull() const { return isActualSeqLengthsKVNull; }
    void set_isActualSeqLengthsKVNull(uint8_t isActualSeqLengthsKVNull) { this->isActualSeqLengthsKVNull = isActualSeqLengthsKVNull; }

    uint32_t get_actualSeqLengthsSize() const { return actualSeqLengthsSize; }
    void set_actualSeqLengthsSize(uint32_t actualSeqLengthsSize) { this->actualSeqLengthsSize = actualSeqLengthsSize; }

    uint32_t get_actualSeqLengthsKVSize() const { return actualSeqLengthsKVSize; }
    void set_actualSeqLengthsKVSize(uint32_t actualSeqLengthsKVSize) { this->actualSeqLengthsKVSize = actualSeqLengthsKVSize; }

    uint8_t get_isKvContinuous() const { return isKvContinuous; }
    void set_isKvContinuous(uint8_t isKvContinuous) { this->isKvContinuous = isKvContinuous; }

    uint8_t get_fromFused() const { return fromFused; }
    void set_fromFused(uint8_t fromFused) { this->fromFused = fromFused; }

    uint8_t get_isBSNDOut() const { return isBSNDOut; }
    void set_isBSNDOut(uint8_t isBSNDOut) { this->isBSNDOut = isBSNDOut; }

    uint8_t get_isGqa() const { return isGqa; }
    void set_isGqa(uint8_t isGqa) { this->isGqa = isGqa; }

    uint8_t get_isSoftMaxLseEnable() const { return isSoftMaxLseEnable; }
    void set_isSoftMaxLseEnable(uint8_t isSoftMaxLseEnable) { this->isSoftMaxLseEnable = isSoftMaxLseEnable; }

    uint8_t get_isActualSharedPrefixLenNull() const { return isActualSharedPrefixLenNull; }
    void set_isActualSharedPrefixLenNull(uint8_t isActualSharedPrefixLenNull) { this->isActualSharedPrefixLenNull = isActualSharedPrefixLenNull; }

    uint8_t get_isQHasLeftPadding() const { return isQHasLeftPadding; }
    void set_isQHasLeftPadding(uint8_t isQHasLeftPadding) { this->isQHasLeftPadding = isQHasLeftPadding; }

    uint8_t get_isKVHasLeftPadding() const { return isKVHasLeftPadding; }
    void set_isKVHasLeftPadding(uint8_t isKVHasLeftPadding) { this->isKVHasLeftPadding = isKVHasLeftPadding; }

    uint32_t get_ropeHeadSize() const { return ropeHeadSize; }
    void set_ropeHeadSize(uint32_t ropeHeadSize) { this->ropeHeadSize = ropeHeadSize; }

    uint32_t get_prefixSeqInnerSize() const { return prefixSeqInnerSize; }
    void set_prefixSeqInnerSize(uint32_t prefixSeqInnerSize) { this->prefixSeqInnerSize = prefixSeqInnerSize; }

    uint32_t get_headNumRatio() const { return headNumRatio; }
    void set_headNumRatio(uint32_t headNumRatio) { this->headNumRatio = headNumRatio; }

    int32_t get_blockSize() const { return blockSize; }
    void set_blockSize(int32_t blockSize) { this->blockSize = blockSize; }

    int32_t get_blockTableDim2() const { return blockTableDim2; }
    void set_blockTableDim2(int32_t blockTableDim2) { this->blockTableDim2 = blockTableDim2; }

    int32_t get_paBlockNumSum() const { return paBlockNumSum; }
    void set_paBlockNumSum(int32_t paBlockNumSum) { this->paBlockNumSum = paBlockNumSum; }

    uint32_t get_attenMaskS1Size() const { return attenMaskS1Size; }
    void set_attenMaskS1Size(uint32_t attenMaskS1Size) { this->attenMaskS1Size = attenMaskS1Size; }

    uint32_t get_kvSplitPart() const { return kvSplitPart; }
    void set_kvSplitPart(uint32_t kvSplitPart) { this->kvSplitPart = kvSplitPart; }

    uint32_t get_accumOutSize() const { return accumOutSize; }
    void set_accumOutSize(uint32_t accumOutSize) { this->accumOutSize = accumOutSize; }

    uint32_t get_logSumExpSize() const { return logSumExpSize; }
    void set_logSumExpSize(uint32_t logSumExpSize) { this->logSumExpSize = logSumExpSize; }

    uint8_t get_paLayoutType() const { return paLayoutType; }
    void set_paLayoutType(uint8_t paLayoutType) { this->paLayoutType = paLayoutType; }

    uint8_t get_isRowInvalid() const { return isRowInvalid; }
    void set_isRowInvalid(uint8_t isRowInvalid) { this->isRowInvalid = isRowInvalid; }

    uint8_t get_isPostQuantPerChnl() const { return isPostQuantPerChnl; }
    void set_isPostQuantPerChnl(uint8_t isPostQuantPerChnl) { this->isPostQuantPerChnl = isPostQuantPerChnl; }
    
    uint8_t get_isPostQuantBF16() const { return isPostQuantBF16; }
    void set_isPostQuantBF16(uint8_t isPostQuantBF16) { this->isPostQuantBF16 = isPostQuantBF16; }

    uint16_t get_antiquantPerTensorFlag () const { return antiquantPerTensorFlag ; }
    void set_antiquantPerTensorFlag (uint16_t antiquantPerTensorFlag ) { this->antiquantPerTensorFlag  = antiquantPerTensorFlag ; }

    uint16_t get_antiquantPerHeadFlag() const { return antiquantPerHeadFlag; }
    void set_antiquantPerHeadFlag(uint16_t antiquantPerHeadFlag) { this->antiquantPerHeadFlag = antiquantPerHeadFlag; }

    uint32_t get_antiquantParaSeqSize() const { return antiquantParaSeqSize; }
    void set_antiquantParaSeqSize(uint32_t antiquantParaSeqSize) { this->antiquantParaSeqSize = antiquantParaSeqSize; }
};

class MultiCoreParamsRegbase {
public:
  int32_t coreNum = 0;
  int64_t totalSize = 0;
  int64_t s1OuterSize = 0;
  int64_t splitFactorSize = 0;
  int64_t splitFactorTailSize = 0;
  uint32_t bnStartIdx[48] = {0};
  int64_t sparseStartIdx[48] = {0};

  int32_t get_coreNum() const { return coreNum; }
  void set_coreNum(int32_t coreNum) { this->coreNum = coreNum; }

  int64_t get_totalSize() const { return totalSize; }
  void set_totalSize(int64_t totalSize) { this->totalSize = totalSize; }

  int64_t get_s1OuterSize() const { return s1OuterSize; }
  void set_s1OuterSize(int64_t s1OuterSize) { this->s1OuterSize = s1OuterSize; }

  int64_t get_splitFactorSize() const { return splitFactorSize; }
  void set_splitFactorSize(int64_t splitFactorSize) { this->splitFactorSize = splitFactorSize; }

  int64_t get_splitFactorTailSize() const { return splitFactorTailSize; }
  void set_splitFactorTailSize(int64_t splitFactorTailSize) { this->splitFactorTailSize = splitFactorTailSize; }

  const uint32_t* get_bnStartIdx() const { return bnStartIdx; }
  void set_bnStartIdx(const uint32_t* values) { 
    for (int i = 0; i < 48; ++i) {
        bnStartIdx[i] = values[i];
    }
  }

  const int64_t* get_sparseStartIdx() const { return sparseStartIdx; }
  void set_sparseStartIdx(const int64_t* values) { 
    for (int i = 0; i < 48; ++i) {
        sparseStartIdx[i] = values[i];
    }
  }
};

class DropmaskParamsRegbase {
public:
    int32_t multiCoreFactorSize = 0;
    int32_t baseUbCalSize = 0;
    int64_t multiCoreTotalSize = 0;
    int64_t shapeTotalSize = 0;

    int32_t get_multiCoreFactorSize() const { return multiCoreFactorSize; }
    void set_multiCoreFactorSize(int32_t multiCoreFactorSize) { this->multiCoreFactorSize = multiCoreFactorSize; }


    int32_t get_baseUbCalSize() const { return baseUbCalSize; }
    void set_baseUbCalSize(int32_t baseUbCalSize) { this->baseUbCalSize = baseUbCalSize; }


    int64_t get_multiCoreTotalSize() const { return multiCoreTotalSize; }
    void set_multiCoreTotalSize(int64_t multiCoreTotalSize) { this->multiCoreTotalSize = multiCoreTotalSize; }


    int64_t get_shapeTotalSize() const { return shapeTotalSize; }
    void set_shapeTotalSize(int64_t shapeTotalSize) { this->shapeTotalSize = shapeTotalSize; }
};

class InitOutputParams {
public:
    uint32_t singleCoreSize = 0;
    uint8_t needInit = 0;
    uint8_t isOneN = 0;
    uint8_t rsvd[2] = {0};
    int64_t totalOutputSize = 0;
    int64_t totalSoftMaxLseOutputSize = 0;

    uint32_t get_singleCoreSize() const { return singleCoreSize; }
    void set_singleCoreSize(uint32_t singleCoreSize) { this->singleCoreSize = singleCoreSize; }

    uint8_t get_needInit() const { return needInit; }
    void set_needInit(uint8_t needInit) { this->needInit = needInit; }
    
    uint8_t get_isOneN() const { return isOneN; }
    void set_isOneN(uint8_t isOneN) { this->isOneN = isOneN; }

    const uint8_t* get_rsvd() const { return rsvd; }
    void set_rsvd(const uint8_t* values) { 
      for (int i = 0; i < 2; ++i) {
          rsvd[i] = values[i];
        }
    }

    int64_t get_totalOutputSize() const { return totalOutputSize; }
    void set_totalOutputSize(int64_t totalOutputSize) { this->totalOutputSize = totalOutputSize; }

    int64_t get_totalSoftMaxLseOutputSize() const { return totalSoftMaxLseOutputSize; }
    void set_totalSoftMaxLseOutputSize(int64_t totalSoftMaxLseOutputSize) { this->totalSoftMaxLseOutputSize = totalSoftMaxLseOutputSize; }

};

class FlashAttentionScoreSimplifiedTilingData {
public:
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
    DropmaskParamsRegbase dropmaskParamsRegbase;
    InitOutputParams initOutputParams;
};