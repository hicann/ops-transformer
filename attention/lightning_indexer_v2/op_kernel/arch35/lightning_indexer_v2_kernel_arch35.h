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
 * \file lightning_indexer_v2_kernel_arch35.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H
#define LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"
#include "lightning_indexer_v2_service_vector_arch35.h"
#include "lightning_indexer_v2_service_cube_arch35.h"
#include "../lightning_indexer_v2_metadata.h"

#include "common/lightning_indexer_v2_kernel_base_arch35.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace matmul;
using namespace optiling;
using namespace optiling::detail;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
constexpr uint64_t Align_16_Bytes = 16UL;
// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U; // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;  // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
    bool isNeedLD = false;                // 该基本块是否需要LD
};

template <typename LIT>
class LightningIndexerV2Kernel {
public:
    __aicore__ inline LightningIndexerV2Kernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *liV2SequsedK,
                                __gm__ uint8_t *liV2CmpResidualK, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
                                const LIV2TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = true;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using SCORE_T = typename LIT::scoreType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    using W_T = float;

    LightningIndexerV2ServiceCube<LIT> matmulService;
    LightningIndexerV2ServiceVector<LIT> liV2VectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
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
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<uint32_t> metadataGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;

    GlobalTensor<uint32_t> cuSeqlensQGm;
    GlobalTensor<uint32_t> cuSeqlensKGm;
    GlobalTensor<uint32_t> sequsedQGm;
    GlobalTensor<uint32_t> sequsedKGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t liV2BlockIndex = 0U;
    uint32_t liV2AiCoreIndex = 0U;
    uint32_t usedCoreNum = 0U;

    LIV2Common::ConstInfo liV2KernelConstInfo{};
    TempLoopInfo liV2LoopInfo{};
    LIV2Common::SplitCoreInfo liV2SplitInfo{};
    LIV2Common::LdSplitCoreInfo liV2LoadInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *liV2SequsedK,
                                            __gm__ uint8_t *liV2CmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LIV2Common::SplitCoreInfo &info);
    __aicore__ inline void SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                            GlobalTensor<uint32_t> &metadataGm);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, LIV2Common::RunInfo liV2KernelRunInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &cuSeqlensQGm, GlobalTensor<uint32_t> &sequsedQGm,
                                               uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                  GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                  GlobalTensor<uint32_t> &sequsedKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio, GlobalTensor<uint32_t> &cmpResidualKGm);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIV2Common::RunInfo &liV2KernelRunInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitTilingData(const LIV2TilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    liV2KernelConstInfo.batchSize = tilingData->bSize;
    liV2KernelConstInfo.qHeadNum = liV2KernelConstInfo.gSize = tilingData->gSize;
    liV2KernelConstInfo.kSeqSize = tilingData->s2Size;
    liV2KernelConstInfo.qSeqSize = tilingData->s1Size;
    liV2KernelConstInfo.attenMaskFlag = (tilingData->maskMode == 3);
    liV2KernelConstInfo.kCacheBlockSize = tilingData->blockSize;
    liV2KernelConstInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    liV2KernelConstInfo.topk = tilingData->topk;
    liV2KernelConstInfo.cmpRatio = tilingData->cmpRatio;
    liV2KernelConstInfo.batchSupperFlag = tilingData->batchSupperFlag;
    liV2KernelConstInfo.keyStride0 = tilingData->keyStride0;
    liV2KernelConstInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LI_V2_LAYOUT::TND) {
        liV2KernelConstInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_V2_LAYOUT::TND) {
        liV2KernelConstInfo.isAccumSeqS2 = true;
    }

    liV2KernelConstInfo.kHeadNum = K_HEAD_NUM;
    liV2KernelConstInfo.headDim = HEAD_DIM;

    if (liV2KernelConstInfo.gSize > 32 || liV2KernelConstInfo.topk > 2048) {
        liV2KernelConstInfo.mBaseSize = S1_BASE_SIZE_SMALL * liV2KernelConstInfo.gSize;
        liV2KernelConstInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
    } else {
        liV2KernelConstInfo.mBaseSize = S1_BASE_SIZE * liV2KernelConstInfo.gSize;
        liV2KernelConstInfo.s1BaseSize = S1_BASE_SIZE;
    }
    liV2KernelConstInfo.s2BaseSize = S2_BASE_SIZE;
    liV2KernelConstInfo.returnValueFlag = tilingData->returnValue;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitBuffers()
{
    LIV2Common::InitBuffers(liV2VectorService, matmulService, pipe);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitActualSeqLen(__gm__ uint8_t *liV2CuSeqlensQ,
                                                                       __gm__ uint8_t *cuSeqlensK,
                                                                       __gm__ uint8_t *liV2SequsedQ,
                                                                       __gm__ uint8_t *liV2SequsedK,
                                                                       __gm__ uint8_t *liV2CmpResidualK)
{
    if (liV2CuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ uint32_t *)liV2CuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensK != nullptr) {
        cuSeqlensKGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensK);
        hasCuSeqlensK = true;
    }
    if (liV2SequsedQ != nullptr) {
        sequsedQGm.SetGlobalBuffer((__gm__ uint32_t *)liV2SequsedQ);
        hasSequsedQ = true;
    }
    if (liV2SequsedK != nullptr) {
        sequsedKGm.SetGlobalBuffer((__gm__ uint32_t *)liV2SequsedK);
        hasSequsedK = true;
    }
    if (liV2CmpResidualK != nullptr) {
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t *)liV2CmpResidualK);
        hasCmpResidualK = true;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                                          bool isAccumSeq,
                                                                          GlobalTensor<uint32_t> &cuSeqlensQGm,
                                                                          GlobalTensor<uint32_t> &sequsedQGm,
                                                                          uint32_t defaultSeqLen)
{
    return LIV2Common::GetActualSeqLen(bIdx, hasCuSeqlensQ, hasSequsedQ, cuSeqlensQGm, sequsedQGm, defaultSeqLen);
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                             bool isAccumSeq,
                                                                             GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                                             GlobalTensor<uint32_t> &sequsedKGm,
                                                                             uint32_t defaultSeqLen, uint32_t cmpRatio,
                                                                             GlobalTensor<uint32_t> &cmpResidualKGm)
{
    uint32_t liV2Residual = hasCmpResidualK ? cmpResidualKGm.GetValue(bIdx) : 0;
    if (hasSequsedK) {
        return sequsedKGm.GetValue(bIdx) * cmpRatio + liV2Residual;
    } else if (hasCuSeqlensK) {
        return (cuSeqlensKGm.GetValue(bIdx + 1) - cuSeqlensKGm.GetValue(bIdx)) * cmpRatio + liV2Residual;
    } else {
        return defaultSeqLen * cmpRatio + liV2Residual;
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size,
                                                                          uint32_t &actS2Size, uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, liV2KernelConstInfo.actualLenQDims, liV2KernelConstInfo.isAccumSeqS1,
                                cuSeqlensQGm, sequsedQGm, liV2KernelConstInfo.qSeqSize);
    // 压缩前的actS2Size
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, liV2KernelConstInfo.actualLenDims, liV2KernelConstInfo.isAccumSeqS2, cuSeqlensKGm,
                           sequsedKGm, liV2KernelConstInfo.kSeqSize, liV2KernelConstInfo.cmpRatio, cmpResidualKGm);
    // 真实使用的压缩后S2长度
    actS2Size = actS2SizeOrig / liV2KernelConstInfo.cmpRatio;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                                                       GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t liV2CoreEnableIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_CORE_ENABLE_INDEX);
    uint32_t liV2BN2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_BN2_START_INDEX);
    uint32_t liV2MStartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_M_START_INDEX);
    uint32_t liV2S2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_S2_START_INDEX);
    uint32_t liV2BN2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_BN2_END_INDEX);
    uint32_t liV2MEndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_M_END_INDEX);
    uint32_t liV2S2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_S2_END_INDEX);

    uint32_t liV2ZeroCoreEnableIndex = GetAttrAbsIndex(0, LI_V2_CORE_ENABLE_INDEX);
    if (metadataGm.GetValue(liV2ZeroCoreEnableIndex) == 0) {
        isUsedCoreEqZero = true;
    }
    if (metadataGm.GetValue(liV2CoreEnableIndex) == 0) {
        liV2SplitInfo.isCoreEnable = false;
        return;
    } else {
        liV2SplitInfo.isCoreEnable = true;
    }

    liV2SplitInfo.bN2Start = metadataGm.GetValue(liV2BN2StartIndex);
    liV2SplitInfo.gS1Start = metadataGm.GetValue(liV2MStartIndex);
    liV2SplitInfo.s2Start = metadataGm.GetValue(liV2S2StartIndex);
    liV2SplitInfo.bN2End = metadataGm.GetValue(liV2BN2EndIndex);
    liV2SplitInfo.gS1End = metadataGm.GetValue(liV2MEndIndex);
    liV2SplitInfo.s2End = metadataGm.GetValue(liV2S2EndIndex);

    if (liV2SplitInfo.s2End != 0) {
        // 此时只需要s2End往前退一格，bN2End和gS1End都不变
        liV2SplitInfo.s2End = liV2SplitInfo.s2End - 1;
    } else {
        // splitCoreInfo.gS1End != 0 splitCoreInfo.s2End == 0 时，gS1End需要往前退一格, bN2End不变
        if (liV2SplitInfo.gS1End != 0) {
            // 此时需要使用bIdx获取实际Actal S2来计算出 s2End
            liV2SplitInfo.gS1End = liV2SplitInfo.gS1End - 1;
            // 需要获取当前的Actaul S2
            uint32_t liV2BIdx = liV2SplitInfo.bN2End / liV2KernelConstInfo.kHeadNum;
            uint32_t liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig;
            GetS1S2ActualSeqLen(liV2BIdx, liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig);
            // s2的切块数量
            uint32_t liV2S2BaseNum;
            if (liV2KernelConstInfo.attenMaskFlag) {
                liV2S2BaseNum = GetS2BaseBlockNumOnMask(liV2SplitInfo.gS1End, liV2ActS1Size, liV2ActS2SizeOrig);
            } else {
                liV2S2BaseNum = CeilDiv(liV2ActS2Size, liV2KernelConstInfo.s2BaseSize);
            }
            liV2SplitInfo.s2End = liV2S2BaseNum - 1;
        } else {
            // splitCoreInfo.gS1End == 0 splitCoreInfo.s2End == 0 时，bN2End需要往前退一格
            // 此时需要使用bIdx获取实际Actal S1和S2来计算出 gS1End 和 s2End
            liV2SplitInfo.bN2End = liV2SplitInfo.bN2End - 1;

            // 需要获取当前的Actaul S1 S2
            uint32_t liV2BIdx = liV2SplitInfo.bN2End / liV2KernelConstInfo.kHeadNum;
            uint32_t liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig;
            GetS1S2ActualSeqLen(liV2BIdx, liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig);

            // s1的切块数量
            uint32_t liV2S1GBaseNum = CeilDiv(liV2ActS1Size, liV2KernelConstInfo.s1BaseSize);
            liV2SplitInfo.gS1End = liV2S1GBaseNum - 1;

            // s2的切块数量
            uint32_t liV2S2BaseNum;
            if (liV2KernelConstInfo.attenMaskFlag) {
                liV2S2BaseNum = GetS2BaseBlockNumOnMask(liV2SplitInfo.gS1End, liV2ActS1Size, liV2ActS2SizeOrig);
            } else {
                liV2S2BaseNum = CeilDiv(liV2ActS2Size, liV2KernelConstInfo.s2BaseSize);
            }
            liV2SplitInfo.s2End = liV2S2BaseNum - 1;
        }
    }

    uint32_t ldFirstWorkSpaceIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX, false);
    liV2LoadInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_CORE_ENABLE_INDEX, true);
        liV2LoadInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!liV2LoadInfo.isLdCoreEnable) {
            return;
        }

        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_NUM_INDEX, true);

        liV2LoadInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        liV2LoadInfo.bIdx = liV2LoadInfo.bn2Idx / liV2KernelConstInfo.kHeadNum;
        liV2LoadInfo.n2Idx = liV2LoadInfo.bn2Idx % liV2KernelConstInfo.kHeadNum;
        liV2LoadInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        liV2LoadInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        liV2LoadInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumIndex);
        liV2LoadInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        liV2LoadInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(liV2LoadInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum =
                (liV2LoadInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(liV2LoadInfo.bIdx) * liV2KernelConstInfo.qSeqSize;
        }
        liV2LoadInfo.indiceOutCoreOffset =
            static_cast<uint64_t>(actualSeqQPrefixSum) * liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.topk +
            liV2LoadInfo.n2Idx * liV2KernelConstInfo.topk +
            static_cast<uint64_t>(liV2LoadInfo.mIdx) * liV2KernelConstInfo.s1BaseSize * liV2KernelConstInfo.kHeadNum *
                liV2KernelConstInfo.topk;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                                  uint32_t actS2SizeOrig)
{
    return LIV2Common::GetMaskedS2BaseBlockNum(s1gIdx, actS1Size, actS2SizeOrig, liV2KernelConstInfo.s1BaseSize,
                                               liV2KernelConstInfo.cmpRatio, liV2KernelConstInfo.s2BaseSize);
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < liV2KernelConstInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, liV2KernelConstInfo.s1BaseSize);
        if (!liV2KernelConstInfo.attenMaskFlag) {
            s2BaseNum = liV2KernelConstInfo.isLDOpen ? CeilDiv(actS2Size, liV2KernelConstInfo.s2BaseSize) :
                                                       (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s1GBaseNum * s2BaseNum * liV2KernelConstInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = liV2KernelConstInfo.isLDOpen ? GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) :
                                                       (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s2BaseNum * liV2KernelConstInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIT>
__aicore__ void inline LightningIndexerV2Kernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                                LIV2Common::SplitCoreInfo &liV2Info)
{
    uint32_t liV2TotalBlockNum = GetTotalBaseBlockNum();
    uint32_t liV2MinBlockPerCore = liV2TotalBlockNum / coreNum;
    uint32_t liV2Deal1MoreBlockCoreNum = liV2TotalBlockNum % coreNum;
    uint32_t liV2CoreIdx = 0;
    uint32_t liV2LastGS1RemainBlockCnt = 0;
    uint32_t liV2CoreDealBlockCnt =
        liV2CoreIdx < liV2Deal1MoreBlockCoreNum ? liV2MinBlockPerCore + 1 : liV2MinBlockPerCore;
    coreNum = liV2MinBlockPerCore == 0 ? liV2Deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        liV2SplitInfo.isCoreEnable = true;
    } else {
        liV2SplitInfo.isCoreEnable = false;
        return;
    }

    bool liV2FindLastCoreEnd = true;
    uint32_t liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig;
    uint32_t liV2S1GBaseNum, liV2S2BaseNum, liV2S2Loop;
    for (uint32_t liV2BN2Idx = 0; liV2BN2Idx < liV2KernelConstInfo.batchSize * liV2KernelConstInfo.kHeadNum;
         liV2BN2Idx++) {
        uint32_t liV2BIdx = liV2BN2Idx / liV2KernelConstInfo.kHeadNum;
        if (liV2BN2Idx % liV2KernelConstInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(liV2BIdx, liV2ActS1Size, liV2ActS2Size, liV2ActS2SizeOrig);
            liV2S1GBaseNum = CeilDiv(liV2ActS1Size, liV2KernelConstInfo.s1BaseSize);
            liV2S2BaseNum = CeilDiv(liV2ActS2Size, liV2KernelConstInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::BSND) {
            if (liV2FindLastCoreEnd && (liV2S1GBaseNum == 0U || liV2S2BaseNum == 0U)) {
                liV2Info.bN2Start = liV2BN2Idx;
                liV2Info.gS1Start = 0;
                liV2Info.s2Start = 0;
                liV2FindLastCoreEnd = false;
            }
        }
        for (uint32_t liV2GS1Idx = 0; liV2GS1Idx < liV2S1GBaseNum; liV2GS1Idx++) {
            if (liV2KernelConstInfo.attenMaskFlag) {
                liV2S2BaseNum = GetS2BaseBlockNumOnMask(liV2GS1Idx, liV2ActS1Size, liV2ActS2SizeOrig);
            }
            if (liV2FindLastCoreEnd && liV2S2BaseNum == 0U) {
                liV2Info.bN2Start = liV2BN2Idx;
                liV2Info.gS1Start = liV2GS1Idx;
                liV2Info.s2Start = 0;
                liV2FindLastCoreEnd = false;
            }
            liV2S2Loop = liV2KernelConstInfo.isLDOpen ? liV2S2BaseNum : (liV2ActS2Size > 0 ? 1 : 0);
            for (uint32_t liV2S2Idx = 0; liV2S2Idx < liV2S2Loop;) {
                if (liV2FindLastCoreEnd) {
                    liV2Info.bN2Start = liV2BN2Idx;
                    liV2Info.gS1Start = liV2GS1Idx;
                    liV2Info.s2Start = liV2S2Idx;
                    liV2FindLastCoreEnd = false;
                }
                uint32_t liV2S2RemainBaseNum = liV2S2Loop - liV2S2Idx;
                if (liV2LastGS1RemainBlockCnt + liV2S2RemainBaseNum >= liV2CoreDealBlockCnt) {
                    liV2Info.bN2End = liV2BN2Idx;
                    liV2Info.gS1End = liV2GS1Idx;
                    liV2Info.s2End = liV2KernelConstInfo.isLDOpen ?
                                         liV2S2Idx + liV2CoreDealBlockCnt - liV2LastGS1RemainBlockCnt - 1 :
                                         liV2S2BaseNum - 1;
                    if (liV2CoreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (liV2S2Idx == 0 && liV2Info.s2End + 1 < liV2S2BaseNum) {
                            liV2Info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (liV2CoreIdx == coreNum - 1 && liV2Info.bN2End != liV2KernelConstInfo.batchSize - 1) {
                            liV2Info.bN2End = liV2KernelConstInfo.batchSize - 1;
                            liV2Info.gS1End = 0;
                            liV2Info.s2End = 0;
                        }
                        return;
                    }
                    liV2CoreIdx++;
                    liV2FindLastCoreEnd = true;
                    liV2S2Idx = liV2Info.s2End + 1;
                    liV2LastGS1RemainBlockCnt = 0;
                    liV2CoreDealBlockCnt =
                        liV2CoreIdx < liV2Deal1MoreBlockCoreNum ? liV2MinBlockPerCore + 1 : liV2MinBlockPerCore;
                } else {
                    liV2LastGS1RemainBlockCnt += liV2S2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx,
                                                                          uint32_t s1Start)
{
    uint32_t tBase = 0U;
    uint32_t s1Count = liV2LoopInfo.actS1Size;
    if (liV2KernelConstInfo.outputLayout == LI_V2_LAYOUT::TND) {
        tBase = cuSeqlensQGm.GetValue(bIdx);
        s1Count = cuSeqlensQGm.GetValue(bIdx + 1) - tBase;
    }
    LIV2Common::DealActSeqLenIsZero<LI_V2_LAYOUT::TND, LI_V2_LAYOUT::BSND>(
        bIdx, n2Idx, s1Start, tBase, s1Count, liV2KernelConstInfo.topk, liV2KernelConstInfo, liV2VectorService);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *liV2SequsedK,
    __gm__ uint8_t *liV2CmpResidualK, __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
    __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
    const LIV2TilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        liV2BlockIndex = GetBlockIdx(); // vec:0-47
        liV2AiCoreIndex = liV2BlockIndex / 2;
    } else {
        liV2BlockIndex = GetBlockIdx(); // cube:0-23
        liV2AiCoreIndex = liV2BlockIndex;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, liV2SequsedK, liV2CmpResidualK);

    // 获取分核信息
    if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
        SplitCoreByAICPU(liV2AiCoreIndex, liV2BlockIndex, metadataGm);
    } else {
        SplitCore(liV2AiCoreIndex, usedCoreNum, liV2SplitInfo);
    }

    pipe = tPipe;

    uint64_t offset = 0;
    uint32_t topkCountAlign16_ = LIV2Common::Align(liV2KernelConstInfo.topk, Align_16_Bytes); // topkCount对齐到16
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize =
        liV2KernelConstInfo.s1BaseSize *
        LIV2Common::Align((uint64_t)liV2KernelConstInfo.kSeqSize, (uint64_t)liV2KernelConstInfo.s2BaseSize) *
        sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + liV2AiCoreIndex * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;
    // vec 存储需要LD的s1对应的s2的score与index，
    // 大小为s1BaseSize * sparseCount * 2，一个核内最多有两个s1BaseSize需要LD
    GlobalTensor<SCORE_T> ldScoreGm; // 存放进行LD的s2 score
    ldScoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + offset));
    offset +=
        static_cast<uint64_t>(GetBlockNum()) * liV2KernelConstInfo.s1BaseSize * topkCountAlign16_ * 2 * sizeof(SCORE_T);
    GlobalTensor<int32_t> ldIndexGm; // 存放进行LD的s2 Index
    ldIndexGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset +=
        static_cast<uint64_t>(GetBlockNum()) * liV2KernelConstInfo.s1BaseSize * topkCountAlign16_ * 2 * sizeof(int32_t);

    if ASCEND_IS_AIV {
        liV2VectorService.InitParams(liV2KernelConstInfo, liV2LoadInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ float *)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        if (outputIdxOffset != nullptr) {
            isOutputIdxOffsetValid = true;
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t *)outputIdxOffset);
        }
        liV2VectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm, outputIdxOffsetGm);
        liV2VectorService.InitVecWorkspaceTensor(scoreGm, ldScoreGm, ldIndexGm);
    } else {
        matmulService.InitParams(liV2KernelConstInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    LIV2Common::GetBN2Idx(liV2LoopInfo, liV2KernelConstInfo, bN2Idx);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    LIV2Common::CalcS2LoopParams<true>(liV2LoopInfo, liV2KernelConstInfo, liV2SplitInfo, bN2LoopIdx, gS1LoopIdx,
                                       liV2KernelConstInfo.cmpRatio);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(liV2LoopInfo.bIdx, liV2LoopInfo.actS1Size, liV2LoopInfo.actS2Size, liV2LoopInfo.actS2SizeOrig);
    LIV2Common::CalcGS1LoopParams<LAYOUT_T == LI_V2_LAYOUT::BSND>(liV2LoopInfo, liV2KernelConstInfo, liV2SplitInfo,
                                                                  bN2LoopIdx);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                                  LIV2Common::RunInfo &liV2KernelRunInfo)
{
    if (!LIV2Common::InitRunInfo(loop, s2LoopIdx, liV2KernelRunInfo, liV2LoopInfo, liV2KernelConstInfo, liV2SplitInfo,
                                 liV2LoadInfo, isOutputIdxOffsetValid)) {
        return;
    }
    liV2KernelRunInfo.needTndPadding = false;
    if (liV2KernelRunInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(liV2KernelRunInfo.bIdx);
            if (hasSequsedQ) {
                uint32_t curSequsedQ = sequsedQGm.GetValue(liV2KernelRunInfo.bIdx);
                uint32_t nextPrefixSum = cuSeqlensQGm.GetValue(liV2KernelRunInfo.bIdx + 1);
                uint32_t liV2CurCuLensQ = nextPrefixSum - actualSeqQPrefixSum;
                if (curSequsedQ < liV2CurCuLensQ) {
                    liV2KernelRunInfo.needTndPadding = true;
                    liV2KernelRunInfo.curCuSeqlensQ = liV2CurCuLensQ;
                    liV2KernelRunInfo.curSequsedQ = curSequsedQ;
                }
            }
        } else { // BSND
            actualSeqQPrefixSum = (liV2KernelRunInfo.bIdx <= 0) ?
                                      0 :
                                      static_cast<uint64_t>(liV2KernelRunInfo.bIdx) * liV2KernelConstInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * liV2KernelConstInfo.qHeadNum * liV2KernelConstInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset =
            tndBIdxOffset + liV2KernelRunInfo.gS1Idx * liV2KernelConstInfo.mBaseSize * liV2KernelConstInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset =
            actualSeqQPrefixSum * liV2KernelConstInfo.qHeadNum + liV2KernelRunInfo.n2Idx * liV2KernelConstInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.topk +
                              liV2KernelRunInfo.n2Idx * liV2KernelConstInfo.topk;
        // B,S1,N2,k/T,N2,k
        valueOutCoreOffset = actualSeqQPrefixSum * liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.topk +
                             liV2KernelRunInfo.n2Idx * liV2KernelConstInfo.topk;
        outputIdxCoreOffset = (actualSeqQPrefixSum + liV2KernelRunInfo.gS1Idx * liV2KernelConstInfo.s1BaseSize) *
                                  liV2KernelConstInfo.kHeadNum +
                              liV2KernelRunInfo.n2Idx;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_V2_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = cuSeqlensKGm.GetValue(liV2KernelRunInfo.bIdx);
    } else {
        actualSeqKPrefixSum = (liV2KernelRunInfo.bIdx <= 0) ? 0 : liV2KernelRunInfo.bIdx * liV2KernelConstInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + liV2KernelRunInfo.s2Idx * liV2KernelConstInfo.s2BaseSize *
                                            liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.headDim;
    liV2KernelRunInfo.tensorQueryOffset = queryCoreOffset;
    liV2KernelRunInfo.tensorKeyOffset = keyCoreOffset;
    liV2KernelRunInfo.tensorWeightsOffset = weightsCoreOffset;
    liV2KernelRunInfo.indiceOutOffset = indiceOutCoreOffset;
    liV2KernelRunInfo.valueOutOffset = valueOutCoreOffset;
    liV2KernelRunInfo.outputIdxCoreOffset = outputIdxCoreOffset;
    if ASCEND_IS_AIV {
        if (liV2KernelRunInfo.needTndPadding) {
            liV2VectorService.DoTndPadding(liV2KernelRunInfo);
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Process()
{
    if (usedCoreNum == 0 || isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
    ProcessDecode();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize = static_cast<uint64_t>(liV2KernelConstInfo.batchSize) * liV2KernelConstInfo.qSeqSize *
                                   liV2KernelConstInfo.kHeadNum * liV2KernelConstInfo.topk;
        uint64_t singleCoreSize =
            LIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = liV2BlockIndex * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, liV2KernelConstInfo.INVALID_IDX);
            if (liV2KernelConstInfo.returnValueFlag) {
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

                GlobalTensor<uint32_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint32_t> valueOut = valueOutGmTmp[baseSize];

                AscendC::InitGlobalMemory(valueOut, dealSize, liV2KernelConstInfo.NEG_INF_FLOAT);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessMain()
{
    if (!liV2SplitInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        liV2VectorService.AllocEventID();
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    LIV2Common::RunInfo liV2KernelRunInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = liV2SplitInfo.bN2Start; bN2LoopIdx <= liV2SplitInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (liV2LoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(liV2LoopInfo.bIdx, liV2LoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = liV2SplitInfo.gS1Start; gS1LoopIdx <= liV2LoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            liV2KernelRunInfo.s2Start = liV2SplitInfo.s2Start;
            liV2KernelRunInfo.s2LoopEnd = liV2LoopInfo.s2LoopEnd;
            for (int s2LoopIdx = liV2SplitInfo.s2Start; s2LoopIdx <= liV2LoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, liV2KernelRunInfo);
                ++gloop;
            }
            liV2SplitInfo.s2Start = 0;
        }
        if (liV2LoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(liV2LoopInfo.bIdx, liV2LoopInfo.n2Idx, liV2LoopInfo.actS1Size);
        }
        liV2SplitInfo.gS1Start = 0U;
    }

    if ASCEND_IS_AIV {
        liV2VectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 1);
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 0 +
                                                                          LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 1 +
                                                                          LIV2Common::ConstInfo::AIV0_AIV1_OFFSET);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                                       LIV2Common::RunInfo liV2KernelRunInfo)
{
    CalcRunInfo(loop, s2LoopIdx, liV2KernelRunInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(liV2KernelRunInfo);
    } else {
        liV2VectorService.ProcessVec1(liV2KernelRunInfo);
        if (liV2KernelRunInfo.isLastS2InnerLoop) { // 本核s2last
            liV2VectorService.ProcessTopK(liV2KernelRunInfo);
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        liV2VectorService.InitLDBuffers(pipe, liV2LoadInfo);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (liV2LoadInfo.isLdCoreEnable) {
            liV2VectorService.ProcessLD();
        }
    }
}
} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H
