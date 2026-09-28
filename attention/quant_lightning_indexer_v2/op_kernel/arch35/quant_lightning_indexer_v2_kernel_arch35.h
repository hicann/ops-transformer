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
 * \file quant_lightning_indexer_v2_kernel_arch35.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
#define QUANT_LIGHTNING_INDEXER_V2_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch35.h"
#include "quant_lightning_indexer_v2_service_vector_arch35.h"
#include "quant_lightning_indexer_v2_service_cube_arch35.h"
#include "../quant_lightning_indexer_v2_metadata.h"

#include "../../../lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_kernel_base_arch35.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace matmul;
using namespace optiling;
using namespace optiling::detail;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息
// 同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL; // 压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
    bool isNeedLD = false; // 该基本块是否需要LD
};

template <typename QLIV2T>
class QLIV2Preload {
public:
    __aicore__ inline QLIV2Preload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *qliV2SequsedK,
                                __gm__ uint8_t *qliV2CmpResidualK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
                                const QLIV2TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;
    using OUT_T = typename QLIV2T::outputType;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    using W_T = typename QLIV2T::weightType;
    using SCORE_T = typename QLIV2T::scoreType;
    using SCALE_T = typename QLIV2T::scaleType;
    using WEIGHT_T = typename QLIV2T::weightType;
    static constexpr bool IS_MX = QLIV2T::isMx;

    QLIV2Matmul<QLIV2T> matmulService;
    QLIV2Vector<QLIV2T> qliV2VectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t M_BASE_SIZE_SMALL = 128;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t TOPK_6K = 6144;
    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t qScaleCoreOffset = 0ULL; // MX场景qScale偏移
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t valueOutCoreOffset = 0ULL;
    uint64_t outputIdxCoreOffset = 0ULL;
    bool isUsedCoreEqZero = false;
    bool isOutputIdxOffsetValid = false;
    bool hasCuSeqlensQ = false;
    bool hasCuSeqlensK = false;
    bool hasSequsedQ = false;
    bool hasSequsedK = false;
    bool hasCmpResidualK = false;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<WEIGHT_T> weightsGm;
    GlobalTensor<bfloat16_t> mxQueryScaleGmBf16;
    GlobalTensor<bfloat16_t> mxKeyScaleGmBf16;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<uint32_t> metadataGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<bfloat16_t> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;
    GlobalTensor<uint32_t> cuSeqlensQGm;
    GlobalTensor<uint32_t> cuSeqlensKGm;
    GlobalTensor<uint32_t> sequsedQGm;
    GlobalTensor<uint32_t> sequsedKGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t qliV2BlockIndex = 0U;
    uint32_t qliV2AiCoreIndex = 0U;
    uint32_t usedCoreNum = 0U;

    QLIV2Common::ConstInfo qliV2KernelConstInfo{};
    TempLoopInfo qliV2LoopInfo{};
    QLIV2Common::SplitCoreInfo qliV2SplitInfo{};
    QLIV2Common::LdSplitCoreInfo qliV2LoadInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *qliV2SequsedK,
                                            __gm__ uint8_t *qliV2CmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                            GlobalTensor<uint32_t> &metadataGm);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, QLIV2Common::RunInfo qliV2KernelRunInfo,
                                            uint32_t qScaleLoop, uint32_t kScaleLoop);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &cuSeqlensQGm, GlobalTensor<uint32_t> &sequsedQGm,
                                               uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, uint32_t cmpResiduaKLenDims,
                                                  bool isAccumSeq, GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                  GlobalTensor<uint32_t> &sequsedKGm,
                                                  GlobalTensor<uint32_t> &cmpResidualKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QLIV2Common::RunInfo &qliV2KernelRunInfo,
                                       uint32_t qScaleLoop, uint32_t kScaleLoop);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitTilingData(const QLIV2TilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    qliV2KernelConstInfo.batchSize = tilingData->bSize;
    qliV2KernelConstInfo.qHeadNum = qliV2KernelConstInfo.gSize = tilingData->gSize;
    qliV2KernelConstInfo.kSeqSize = tilingData->s2Size;
    qliV2KernelConstInfo.qSeqSize = tilingData->s1Size;
    qliV2KernelConstInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    qliV2KernelConstInfo.kCacheBlockSize = tilingData->blockSize;
    qliV2KernelConstInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    qliV2KernelConstInfo.sparseCount = tilingData->sparseCount;
    qliV2KernelConstInfo.cmpRatio = tilingData->cmpRatio;
    qliV2KernelConstInfo.keyStride0 = tilingData->keyStride0;
    qliV2KernelConstInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    qliV2KernelConstInfo.maxSeqlenQ = tilingData->maxSeqlenQ;
    qliV2KernelConstInfo.quantMode = tilingData->quantMode;
    qliV2KernelConstInfo.outputLayout = Q_LAYOUT_T; // 输出和输入形状一致
    if (Q_LAYOUT_T == LI_LAYOUT::TND) {
        qliV2KernelConstInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        qliV2KernelConstInfo.isAccumSeqS2 = true;
    }

    qliV2KernelConstInfo.kHeadNum = K_HEAD_NUM;
    qliV2KernelConstInfo.headDim = HEAD_DIM;
    if (qliV2KernelConstInfo.sparseCount > TOPK_6K) {
        qliV2KernelConstInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
        qliV2KernelConstInfo.mBaseSizeMax = M_BASE_SIZE_SMALL;
    } else {
        qliV2KernelConstInfo.s1BaseSize = S1_BASE_SIZE;
        qliV2KernelConstInfo.mBaseSizeMax = M_BASE_SIZE;
    }
    qliV2KernelConstInfo.mBaseSize = qliV2KernelConstInfo.s1BaseSize * qliV2KernelConstInfo.gSize;
    qliV2KernelConstInfo.s2BaseSize = S2_BASE_SIZE;
    qliV2KernelConstInfo.returnValue = tilingData->returnValue;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitBuffers()
{
    LIV2Common::InitBuffers(qliV2VectorService, matmulService, pipe);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                              __gm__ uint8_t *sequsedQ, __gm__ uint8_t *qliV2SequsedK,
                                                              __gm__ uint8_t *qliV2CmpResidualK)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensK != nullptr) {
        cuSeqlensKGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensK);
        hasCuSeqlensK = true;
    }
    if (sequsedQ != nullptr) {
        sequsedQGm.SetGlobalBuffer((__gm__ uint32_t *)sequsedQ);
        hasSequsedQ = true;
    }
    if (qliV2SequsedK != nullptr) {
        sequsedKGm.SetGlobalBuffer((__gm__ uint32_t *)qliV2SequsedK);
        hasSequsedK = true;
    }
    if (qliV2CmpResidualK != nullptr) {
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t *)qliV2CmpResidualK);
        hasCmpResidualK = true;
    }
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                                 GlobalTensor<uint32_t> &cuSeqlensQGm,
                                                                 GlobalTensor<uint32_t> &sequsedQGm,
                                                                 uint32_t defaultSeqLen)
{
    return LIV2Common::GetActualSeqLen(bIdx, hasCuSeqlensQ, hasSequsedQ, cuSeqlensQGm, sequsedQGm, defaultSeqLen);
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                    uint32_t cmpResiduaKLenDims, bool isAccumSeq,
                                                                    GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                                    GlobalTensor<uint32_t> &sequsedKGm,
                                                                    GlobalTensor<uint32_t> &cmpResidualKGm,
                                                                    uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    uint32_t residual = hasCmpResidualK ? cmpResidualKGm.GetValue(bIdx) : 0;
    if (hasSequsedK) {
        return sequsedKGm.GetValue(bIdx) * cmpRatio + residual;
    } else if (hasCuSeqlensK) {
        return (cuSeqlensKGm.GetValue(bIdx + 1) - cuSeqlensKGm.GetValue(bIdx)) * cmpRatio + residual;
    } else {
        return defaultSeqLen * cmpRatio + residual;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                 uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, qliV2KernelConstInfo.actualLenQDims, qliV2KernelConstInfo.isAccumSeqS1,
                                cuSeqlensQGm, sequsedQGm, qliV2KernelConstInfo.qSeqSize);
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, qliV2KernelConstInfo.actualLenDims, qliV2KernelConstInfo.cmpResiduaKLenDims,
                           qliV2KernelConstInfo.isAccumSeqS2, cuSeqlensKGm, sequsedKGm, cmpResidualKGm,
                           qliV2KernelConstInfo.kSeqSize, qliV2KernelConstInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / qliV2KernelConstInfo.cmpRatio; // 真实使用的压缩后S2长度
}

template <typename QLIV2T>
__aicore__ inline uint32_t QLIV2Preload<QLIV2T>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                         uint32_t actS2SizeOrig)
{
    return LIV2Common::GetMaskedS2BaseBlockNum(s1gIdx, actS1Size, actS2SizeOrig, qliV2KernelConstInfo.s1BaseSize,
                                               qliV2KernelConstInfo.cmpRatio, qliV2KernelConstInfo.s2BaseSize,
                                               &qliV2LoopInfo.validS2Len);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                                              GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t qliV2CoreEnableIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_CORE_ENABLE_INDEX);
    uint32_t qliV2BN2StartIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_BN2_START_INDEX);
    uint32_t qliV2MStartIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_M_START_INDEX);
    uint32_t qliV2S2StartIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_S2_START_INDEX);
    uint32_t qliV2BN2EndIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_BN2_END_INDEX);
    uint32_t qliV2MEndIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_M_END_INDEX);
    uint32_t qliV2S2EndIndex = GetAttrAbsIndex(cubeCoreIdx, QLI_V2_S2_END_INDEX);

    uint32_t qliV2ZeroCoreEnableIndex = GetAttrAbsIndex(0, QLI_V2_CORE_ENABLE_INDEX);
    if (metadataGm.GetValue(qliV2ZeroCoreEnableIndex) == 0) {
        isUsedCoreEqZero = true;
    }
    if (metadataGm.GetValue(qliV2CoreEnableIndex) == 0) {
        qliV2SplitInfo.isCoreEnable = false;
        return;
    } else {
        qliV2SplitInfo.isCoreEnable = true;
    }

    qliV2SplitInfo.bN2Start = metadataGm.GetValue(qliV2BN2StartIndex);
    qliV2SplitInfo.gS1Start = metadataGm.GetValue(qliV2MStartIndex);
    qliV2SplitInfo.s2Start = metadataGm.GetValue(qliV2S2StartIndex);
    qliV2SplitInfo.bN2End = metadataGm.GetValue(qliV2BN2EndIndex);
    qliV2SplitInfo.gS1End = metadataGm.GetValue(qliV2MEndIndex);
    qliV2SplitInfo.s2End = metadataGm.GetValue(qliV2S2EndIndex);

    if (qliV2SplitInfo.s2End != 0) {
        // 此时只需要s2End往前退一格，bN2End和gS1End都不变
        qliV2SplitInfo.s2End = qliV2SplitInfo.s2End - 1;
    } else {
        // splitCoreInfo.gS1End != 0 splitCoreInfo.s2End == 0 时，gS1End需要往前退一格, bN2End不变
        if (qliV2SplitInfo.gS1End != 0) {
            // 此时需要使用bIdx获取实际Actal S2来计算出 s2End
            qliV2SplitInfo.gS1End = qliV2SplitInfo.gS1End - 1;
            // 需要获取当前的Actaul S2
            uint32_t qliV2BIdx = qliV2SplitInfo.bN2End / qliV2KernelConstInfo.kHeadNum;
            uint32_t qliV2ActS1Size, qliV2ActS2Size, qliV2ActS2SizeOrig;
            GetS1S2ActualSeqLen(qliV2BIdx, qliV2ActS1Size, qliV2ActS2Size, qliV2ActS2SizeOrig);
            // s2的切块数量
            uint32_t qliV2S2BaseNum;
            if (qliV2KernelConstInfo.attenMaskFlag) {
                qliV2S2BaseNum = GetS2BaseBlockNumOnMask(qliV2SplitInfo.gS1End, qliV2ActS1Size, qliV2ActS2SizeOrig);
            } else {
                qliV2S2BaseNum = CeilDiv(qliV2ActS2Size, qliV2KernelConstInfo.s2BaseSize);
            }
            qliV2SplitInfo.s2End = qliV2S2BaseNum - 1;
        } else {
            // splitCoreInfo.gS1End == 0 splitCoreInfo.s2End == 0 时，bN2End需要往前退一格
            // 此时需要使用bIdx获取实际Actal S1和S2来计算出 gS1End 和 s2End
            qliV2SplitInfo.bN2End = qliV2SplitInfo.bN2End - 1;

            // 需要获取当前的Actaul S1 S2
            uint32_t qliV2BIdx = qliV2SplitInfo.bN2End / qliV2KernelConstInfo.kHeadNum;
            uint32_t qliV2ActS1Size, qliV2ActS2Size, qliV2ActS2SizeOrig;
            GetS1S2ActualSeqLen(qliV2BIdx, qliV2ActS1Size, qliV2ActS2Size, qliV2ActS2SizeOrig);

            // s1的切块数量
            uint32_t qliV2S1GBaseNum = CeilDiv(qliV2ActS1Size, qliV2KernelConstInfo.s1BaseSize);
            qliV2SplitInfo.gS1End = qliV2S1GBaseNum - 1;

            // s2的切块数量
            uint32_t qliV2S2BaseNum;
            if (qliV2KernelConstInfo.attenMaskFlag) {
                qliV2S2BaseNum = GetS2BaseBlockNumOnMask(qliV2SplitInfo.gS1End, qliV2ActS1Size, qliV2ActS2SizeOrig);
            } else {
                qliV2S2BaseNum = CeilDiv(qliV2ActS2Size, qliV2KernelConstInfo.s2BaseSize);
            }
            qliV2SplitInfo.s2End = qliV2S2BaseNum - 1;
        }
    }
    uint32_t ldFirstWorkSpaceIndex =
        GetAttrAbsIndex(cubeCoreIdx, QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX, false); // LD 第一个workspace的索引
    qliV2LoadInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_CORE_ENABLE_INDEX, true);
        qliV2LoadInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!qliV2LoadInfo.isLdCoreEnable) {
            return;
        }

        // LD 参数信息
        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumINDEX = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_NUM_INDEX, true);

        qliV2LoadInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        qliV2LoadInfo.bIdx = qliV2LoadInfo.bn2Idx / qliV2KernelConstInfo.kHeadNum;
        qliV2LoadInfo.n2Idx = qliV2LoadInfo.bn2Idx % qliV2KernelConstInfo.kHeadNum;
        qliV2LoadInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        qliV2LoadInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        qliV2LoadInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumINDEX);
        qliV2LoadInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        qliV2LoadInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(qliV2LoadInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (qliV2LoadInfo.bIdx <= 0) ?
                                      0 :
                                      static_cast<uint64_t>(qliV2LoadInfo.bIdx) * qliV2KernelConstInfo.qSeqSize;
        }
        qliV2LoadInfo.indiceOutCoreOffset =
            actualSeqQPrefixSum * qliV2KernelConstInfo.kHeadNum * qliV2KernelConstInfo.sparseCount +
            static_cast<uint64_t>(qliV2LoadInfo.n2Idx) * qliV2KernelConstInfo.sparseCount +
            static_cast<uint64_t>(qliV2LoadInfo.mIdx) * qliV2KernelConstInfo.s1BaseSize *
                qliV2KernelConstInfo.kHeadNum * qliV2KernelConstInfo.sparseCount; // 搬出Topk的初始偏移地址
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    uint32_t tBase = 0U;
    uint32_t s1Count = qliV2LoopInfo.actS1Size;
    if (qliV2KernelConstInfo.outputLayout == LI_LAYOUT::TND) {
        tBase = cuSeqlensQGm.GetValue(bIdx);
        s1Count = cuSeqlensQGm.GetValue(bIdx + 1) - tBase;
    }
    LIV2Common::DealActSeqLenIsZero<LI_LAYOUT::TND, LI_LAYOUT::BSND>(bIdx, n2Idx, s1Start, tBase, s1Count,
                                                                     qliV2KernelConstInfo.sparseCount,
                                                                     qliV2KernelConstInfo, qliV2VectorService);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *queryScale,
    __gm__ uint8_t *keyScale, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *qliV2SequsedK, __gm__ uint8_t *qliV2CmpResidualK, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace, const QLIV2TilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        qliV2BlockIndex = GetBlockIdx(); // vec:0-47
        qliV2AiCoreIndex = qliV2BlockIndex / 2;
    } else {
        qliV2BlockIndex = GetBlockIdx(); // cube:0-23
        qliV2AiCoreIndex = qliV2BlockIndex;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, qliV2SequsedK, qliV2CmpResidualK);

    // 获取分核信息
    metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
    SplitCoreByAICPU(qliV2AiCoreIndex, qliV2BlockIndex, metadataGm);

    pipe = tPipe;

    uint64_t offset = 0;
    uint32_t topkCountAlign16_ =
        QLIV2Common::Align(qliV2KernelConstInfo.sparseCount, (uint64_t)16); // topkCount对齐到16
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize =
        qliV2KernelConstInfo.s1BaseSize *
        QLIV2Common::Align((uint64_t)qliV2KernelConstInfo.kSeqSize, (uint64_t)qliV2KernelConstInfo.s2BaseSize) *
        sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + qliV2AiCoreIndex * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;
    // vec 存储需要LD的s1对应的s2的score与index，
    // 大小为s1BaseSize * sparseCount * 2，一个核内最多有两个s1BaseSize需要LD
    GlobalTensor<SCORE_T> ldScoreGm; // 存放进行LD的s2 score
    ldScoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + offset));
    offset += static_cast<uint64_t>(GetBlockNum()) * qliV2KernelConstInfo.s1BaseSize * topkCountAlign16_ * 2 *
              sizeof(SCORE_T);
    GlobalTensor<int32_t> ldIndexGm; // 存放进行LD的s2 Index
    ldIndexGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset += static_cast<uint64_t>(GetBlockNum()) * qliV2KernelConstInfo.s1BaseSize * topkCountAlign16_ * 2 *
              sizeof(int32_t);

    if ASCEND_IS_AIV {
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ WEIGHT_T *)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        valueOutGm.SetGlobalBuffer((__gm__ bfloat16_t *)sparseValues);
        if (outputIdxOffset != nullptr) {
            isOutputIdxOffsetValid = true;
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t *)outputIdxOffset);
        }
        if constexpr (IS_MX) {
            qliV2VectorService.InitVecInputTensor(weightsGm, indiceOutGm, blockTableGm, valueOutGm, outputIdxOffsetGm);
        } else {
            GlobalTensor<SCALE_T> qScaleGmForVec;
            GlobalTensor<SCALE_T> kScaleGmForVec;
            qScaleGmForVec.SetGlobalBuffer((__gm__ SCALE_T *)queryScale);
            kScaleGmForVec.SetGlobalBuffer((__gm__ SCALE_T *)keyScale);
            qliV2VectorService.InitVecInputTensor(weightsGm, indiceOutGm, blockTableGm, valueOutGm, outputIdxOffsetGm,
                                                  qScaleGmForVec, kScaleGmForVec);
        }
        qliV2VectorService.InitVecWorkspaceTensor(scoreGm, ldScoreGm, ldIndexGm);
        qliV2VectorService.InitParams(qliV2KernelConstInfo, qliV2LoadInfo, tiling);
    } else {
        matmulService.InitParams(qliV2KernelConstInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        if constexpr (IS_MX) {
            mxQueryScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)queryScale);
            mxKeyScaleGmBf16.SetGlobalBuffer((__gm__ bfloat16_t *)keyScale);
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mxKeyScaleGmBf16, mxQueryScaleGmBf16);
        } else {
            matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
        }
    }
    InitBuffers();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::GetBN2Idx(uint32_t bN2Idx)
{
    LIV2Common::GetBN2Idx(qliV2LoopInfo, qliV2KernelConstInfo, bN2Idx);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    if (!qliV2KernelConstInfo.attenMaskFlag) {
        qliV2LoopInfo.validS2Len = qliV2LoopInfo.actS2Size;
    }
    LIV2Common::CalcS2LoopParams<true>(qliV2LoopInfo, qliV2KernelConstInfo, qliV2SplitInfo, bN2LoopIdx, gS1LoopIdx,
                                       qliV2KernelConstInfo.cmpRatio, &qliV2LoopInfo.validS2Len);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(qliV2LoopInfo.bIdx, qliV2LoopInfo.actS1Size, qliV2LoopInfo.actS2Size,
                        qliV2LoopInfo.actS2SizeOrig);
    LIV2Common::CalcGS1LoopParams<Q_LAYOUT_T == LI_LAYOUT::BSND>(qliV2LoopInfo, qliV2KernelConstInfo, qliV2SplitInfo,
                                                                 bN2LoopIdx);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                         QLIV2Common::RunInfo &qliV2KernelRunInfo, uint32_t qScaleLoop,
                                                         uint32_t kScaleLoop)
{
    qliV2KernelRunInfo.validS2Len = qliV2LoopInfo.validS2Len;
    qliV2KernelRunInfo.qScaleLoop = qScaleLoop;
    qliV2KernelRunInfo.kScaleLoop = kScaleLoop;
    if (!LIV2Common::InitRunInfo(loop, s2LoopIdx, qliV2KernelRunInfo, qliV2LoopInfo, qliV2KernelConstInfo,
                                 qliV2SplitInfo, qliV2LoadInfo, isOutputIdxOffsetValid)) {
        return; // 需要验证， v1 时候需要runInfo
    }
    uint64_t qkHeadDim = qliV2KernelConstInfo.headDim;
    if constexpr (QLIV2T::isMxFp4) {
        qkHeadDim = qliV2KernelConstInfo.headDim / FP4_PACK_NUM;
    }
    if (qliV2KernelRunInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(qliV2KernelRunInfo.bIdx);
            if (hasSequsedQ) {
                uint32_t curSequsedQ = sequsedQGm.GetValue(qliV2KernelRunInfo.bIdx);
                uint32_t nextPrefixSum = cuSeqlensQGm.GetValue(qliV2KernelRunInfo.bIdx + 1);
                uint32_t curCuLensQ = nextPrefixSum - actualSeqQPrefixSum;
                if (curSequsedQ < curCuLensQ) {
                    qliV2KernelRunInfo.needTndPadding = true;
                    qliV2KernelRunInfo.curCuSeqlensQ = curCuLensQ;
                    qliV2KernelRunInfo.curSequsedQ = curSequsedQ;
                }
            }
        } else { // BSND
            actualSeqQPrefixSum = (qliV2KernelRunInfo.bIdx <= 0) ?
                                      0 :
                                      static_cast<uint64_t>(qliV2KernelRunInfo.bIdx) * qliV2KernelConstInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * qliV2KernelConstInfo.qHeadNum * qkHeadDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + qliV2KernelRunInfo.gS1Idx * qliV2KernelConstInfo.mBaseSize * qkHeadDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset =
            actualSeqQPrefixSum * qliV2KernelConstInfo.qHeadNum + qliV2KernelRunInfo.n2Idx * qliV2KernelConstInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * qliV2KernelConstInfo.kHeadNum * qliV2KernelConstInfo.sparseCount +
                              qliV2KernelRunInfo.n2Idx * qliV2KernelConstInfo.sparseCount;
        valueOutCoreOffset = actualSeqQPrefixSum * qliV2KernelConstInfo.kHeadNum * qliV2KernelConstInfo.sparseCount +
                             qliV2KernelRunInfo.n2Idx * qliV2KernelConstInfo.sparseCount;
        outputIdxCoreOffset = (actualSeqQPrefixSum + qliV2KernelRunInfo.gS1Idx * qliV2KernelConstInfo.s1BaseSize) *
                                  qliV2KernelConstInfo.kHeadNum +
                              qliV2KernelRunInfo.n2Idx;
        if constexpr (IS_MX) {
            // MX: qScale offset, shape [B, S1, N1, D/64, 2]
            uint64_t qScalePrefixSum = actualSeqQPrefixSum * qliV2KernelConstInfo.qHeadNum *
                                       (qliV2KernelConstInfo.headDim / MX_SCALE_GROUP_SIZE);
            uint64_t qScaleS1Offset = static_cast<uint64_t>(qliV2KernelRunInfo.gS1Idx) *
                                      qliV2KernelConstInfo.mBaseSize *
                                      (qliV2KernelConstInfo.headDim / MX_SCALE_GROUP_SIZE);
            qScaleCoreOffset = qScalePrefixSum + qScaleS1Offset;
        }
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D, cu_seqlens_k
        actualSeqKPrefixSum = cuSeqlensKGm.GetValue(qliV2KernelRunInfo.bIdx);
    } else {
        actualSeqKPrefixSum =
            (qliV2KernelRunInfo.bIdx <= 0) ? 0 : qliV2KernelRunInfo.bIdx * qliV2KernelConstInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * qliV2KernelConstInfo.kHeadNum * qkHeadDim;
    keyCoreOffset = tndBIdxOffsetForK + qliV2KernelRunInfo.s2Idx * qliV2KernelConstInfo.s2BaseSize *
                                            qliV2KernelConstInfo.kHeadNum * qkHeadDim;
    uint64_t keyScaleS2Offset =
        actualSeqKPrefixSum + static_cast<uint64_t>(qliV2KernelRunInfo.s2Idx) * qliV2KernelConstInfo.s2BaseSize;
    if constexpr (IS_MX) {
        // MX: kScale offset, shape [B, S2, N2, D/64, 2]
        keyScaleCoreOffset =
            keyScaleS2Offset * qliV2KernelConstInfo.kHeadNum * (qliV2KernelConstInfo.headDim / MX_SCALE_GROUP_SIZE);
    } else {
        keyScaleCoreOffset = keyScaleS2Offset * qliV2KernelConstInfo.kHeadNum;
    }
    qliV2KernelRunInfo.tensorQueryOffset = queryCoreOffset;
    qliV2KernelRunInfo.tensorKeyOffset = keyCoreOffset;
    qliV2KernelRunInfo.tensorQScaleOffset = qScaleCoreOffset;
    qliV2KernelRunInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    qliV2KernelRunInfo.tensorWeightsOffset = weightsCoreOffset;
    qliV2KernelRunInfo.indiceOutOffset = indiceOutCoreOffset;
    qliV2KernelRunInfo.valueOutOffset = valueOutCoreOffset;
    qliV2KernelRunInfo.outputIdxCoreOffset = outputIdxCoreOffset;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::Process()
{
    if (isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();

    ProcessDecode();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize = static_cast<uint64_t>(qliV2KernelConstInfo.batchSize) *
                                   qliV2KernelConstInfo.qSeqSize * qliV2KernelConstInfo.kHeadNum *
                                   qliV2KernelConstInfo.sparseCount;
        uint64_t singleCoreSize =
            QLIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = qliV2BlockIndex * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, qliV2KernelConstInfo.INVALID_IDX);
            if (qliV2KernelConstInfo.returnValue) {
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint16_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint16_t> valueOut = valueOutGmTmp[baseSize];

                AscendC::InitGlobalMemory(valueOut, dealSize, qliV2KernelConstInfo.NEG_INF_BFLOAT);
            }
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessMain()
{
    if (!qliV2SplitInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        qliV2VectorService.AllocEventID();
        CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(QLIV2Common::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(QLIV2Common::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    QLIV2Common::RunInfo qliV2KernelRunInfo;
    uint32_t gloop = 0;
    uint32_t qScaleLoop = 0;
    for (uint32_t bN2LoopIdx = qliV2SplitInfo.bN2Start; bN2LoopIdx <= qliV2SplitInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (qliV2LoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(qliV2LoopInfo.bIdx, qliV2LoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = qliV2SplitInfo.gS1Start; gS1LoopIdx <= qliV2LoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            uint32_t kScaleLoop = 0;
            qliV2KernelRunInfo.s2Start = qliV2SplitInfo.s2Start;
            qliV2KernelRunInfo.s2LoopEnd = qliV2LoopInfo.s2LoopEnd;
            for (int s2LoopIdx = qliV2SplitInfo.s2Start; s2LoopIdx <= qliV2LoopInfo.s2LoopEnd; s2LoopIdx++) {
                if ((s2LoopIdx - qliV2SplitInfo.s2Start) % 16 == 0) {
                    ++kScaleLoop;
                }
                ProcessBaseBlock(gloop, s2LoopIdx, qliV2KernelRunInfo, qScaleLoop, kScaleLoop);
                ++gloop;
            }
            ++qScaleLoop;
            qliV2SplitInfo.s2Start = 0;
        }
        if (qliV2LoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(qliV2LoopInfo.bIdx, qliV2LoopInfo.n2Idx, qliV2LoopInfo.actS1Size);
        }
        qliV2SplitInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        qliV2VectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                              0);
        CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(QLIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                              1);
        CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(
            QLIV2Common::ConstInfo::CROSS_VC_EVENT + 0 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_FIX>(
            QLIV2Common::ConstInfo::CROSS_VC_EVENT + 1 + QLIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                              QLIV2Common::RunInfo qliV2KernelRunInfo,
                                                              uint32_t qScaleLoop, uint32_t kScaleLoop)
{
    CalcRunInfo(loop, s2LoopIdx, qliV2KernelRunInfo, qScaleLoop, kScaleLoop);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(qliV2KernelRunInfo);
    } else {
        if (qliV2KernelRunInfo.needTndPadding) {
            qliV2VectorService.DoTndPadding(qliV2KernelRunInfo);
        }
        qliV2VectorService.ProcessVec1(qliV2KernelRunInfo);
        if (qliV2KernelRunInfo.isLastS2InnerLoop) { // 本核s2last
            qliV2VectorService.ProcessTopK(qliV2KernelRunInfo);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Preload<QLIV2T>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        qliV2VectorService.InitLDBuffers(pipe, qliV2LoadInfo);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (qliV2LoadInfo.isLdCoreEnable) {
            qliV2VectorService.ProcessLD();
        }
    }
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_KERNEL_H
