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
 * \file sparse_flash_mla_csa_kernel_arch35.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
#define SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
#include "sparse_flash_mla_common_arch35.h"
#include "sparse_flash_mla_kvcache.h"
#include "sparse_flash_mla_csa_block_cube_arch35.h"
#include "sparse_flash_mla_csa_block_vector_arch35.h"
#include "kernel_operator.h"
#include "../sparse_flash_mla_kernel_metadata.h"

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"
#include "common/buffers_policy_3buff_sfa.h"

#include "kernel_operator_list_tensor_intf.h"
#include "common/smla_kernel_common_arch35.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {
template <typename CubeBlockType, typename VecBlockType>
class SparseFlashMlaCsaKernel {
public:
    SMLA_ARGS_TRAITS;
    __aicore__ inline SparseFlashMlaCsaKernel(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV,
                                __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
                                __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace, const SparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                           __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                           __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                           __gm__ uint8_t *cmpResidualKV);
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
        __gm__ uint8_t *cmpResidualKV, __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
        __gm__ uint8_t *sinks, __gm__ uint8_t *workspace, const SparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    const SparseFlashMlaTilingData *__restrict smlaCsaTilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint32_t PRELOAD_NUM = 3;
    static constexpr uint32_t crossCoreMte2SyncFlagId = 15; // IS_SPLIT_G 核间 MTE2 同步 flag ID
    static constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;

    /* 核间通道 */
    BufferManager<BufferType::GM> smlaCsaV0ResGmBufferManager;

    StaticBuffer<T> smlaCsaBmm1Buffers[2];
    StaticBuffer<T> smlaCsaBmm2Buffers;
    uint32_t smlaCsaBmm1GetFlag = 0;
    uint32_t smlaCsaVUbBase = 0;

    // mm2左矩阵P
    StaticBuffer<Q_T> smlaCsaL1PBuffers[2];
    uint32_t smlaCsaL1PGetFlag = 0;
    uint32_t smlaCsaL1CubeBase = 0;
    GlobalTensor<uint32_t> smlaCsaMetadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvlenGm;
    GlobalTensor<int32_t> actualSeqCmpKvlenGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    GlobalTensor<int32_t> actualSeqQlenGm;
    GlobalTensor<int32_t> oriTopkLengthGm;
    GlobalTensor<int32_t> cmpTopkLengthGm;

    bool hasCuSeqlensQ = false;
    bool hasCuSeqlensOriKv = false;
    bool hasCuSeqlensCmpKv = false;
    bool hasActualSeqQlen = false;
    bool hasActualSeqOriKvlen = false;
    bool hasActualSeqCmpKvlen = false;
    /* workspace 空间 */
    BuffersPolicy3buffSFA<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    BufferManager<BufferType::GM> fdStagingBufferManager;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> fdStagingBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> intraCoreCombineBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> crossCoreCombineBuffer;
    /* 核Index信息 */
    int32_t smlaCsaAicIdx;

    /* Init阶段metadata解析结果 */
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t hasLoad;

    /* 初始化后不变的信息 */
    ConstInfo smlaCsaConstInfo;

    /* 模板库Block */
    CubeBlockType smlaCsaCubeBlock;
    VecBlockType smlaCsaVecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling)
{
    fa_base_matmul::ResetIdCounter();
    smlaCsaConstInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->smlaCsaAicIdx = GetBlockIdx();
        smlaCsaConstInfo.aivIdx = 0;
        this->smlaCsaTilingData = tiling;
    } else {
        smlaCsaConstInfo.aivIdx = GetBlockIdx();
        this->smlaCsaAicIdx = smlaCsaConstInfo.aivIdx >> 1;
        this->smlaCsaTilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->smlaCsaMetadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    smlaCsaConstInfo.s1BaseSize = 64;
    smlaCsaConstInfo.s2BaseSize = 128;

    this->ParseTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV,
                          cmpResidualKV);
    smlaCsaVecBlock.InitVecBlock(cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    smlaCsaVecBlock.CleanOutput(attentionOut, softmaxLse, smlaCsaConstInfo);

    // 从meta data解析分核信息
    bN2StartIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_BN2_START_INDEX, false));
    gS1StartIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_M_START_INDEX, false));
    bN2EndIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_BN2_END_INDEX, false));
    nextGs1Idx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_M_END_INDEX, false));
    hasLoad = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_CORE_ENABLE_INDEX, false));
    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    this->InitGlobalBuffer(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                           cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, seqUsedOriKV, seqUsedCmpKV,
                           cmpResidualKV, oriTopkLength, cmpTopkLength, sinks, workspace, tiling); // gm设置

    if ASCEND_IS_AIV {
        if constexpr ((TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                       TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                       TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                      IS_VEC_S2PHYADDR) {
            this->smlaCsaVecBlock.GetKVPhyAddr(
                hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
                hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm,
                cmpResidualKvGm, actualSeqQlenGm, cuSeqlensQGm, workspace, smlaCsaConstInfo);
        }
    }

    InitMMResBuf(workspace);
    if constexpr (IS_BATCH_CONSISTENCY) {
        smlaCsaVecBlock.InitS2SplitStaging(intraCoreCombineBuffer.Get(), crossCoreCombineBuffer.Get());
    } else {
        smlaCsaVecBlock.InitS2SplitStaging(fdStagingBuffer.Get());
    }
    ComputeConstexpr<TEMPLATE_INTF_ARGS>(this->smlaCsaConstInfo);
    InitLocalBuffer(this->smlaCsaVecBlock, this->smlaCsaCubeBlock, this->smlaCsaConstInfo, this->smlaCsaVUbBase,
                    this->smlaCsaL1CubeBase);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
    __gm__ uint8_t *cmpResidualKV)
{
    auto &sparseFlashMLABaseParams = this->smlaCsaTilingData->baseParams;
    auto &sparseFlashMLACmpParams = this->smlaCsaTilingData->cmpParams;
    smlaCsaConstInfo.bSize = sparseFlashMLABaseParams.batchSize;
    smlaCsaConstInfo.n2Size = 1;
    smlaCsaConstInfo.gSize = sparseFlashMLABaseParams.nNumOfQInOneGroup;
    smlaCsaConstInfo.s1Size = sparseFlashMLABaseParams.qSeqSize;
    smlaCsaConstInfo.s2Size = sparseFlashMLABaseParams.kvSeqSize;
    smlaCsaConstInfo.cmpS2Size = sparseFlashMLACmpParams.cmpKvSeqSize;
    smlaCsaConstInfo.oriSparseBlockCount = sparseFlashMLABaseParams.oriSparseBlockCount;
    smlaCsaConstInfo.cmpSparseBlockCount = sparseFlashMLACmpParams.cmpSparseBlockCount;
    smlaCsaConstInfo.alignedOriSparseBlockCount = (smlaCsaConstInfo.oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                                  SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    smlaCsaConstInfo.alignedCmpSparseBlockCount = (smlaCsaConstInfo.cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                                  SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    smlaCsaConstInfo.cmpRatio = sparseFlashMLACmpParams.cmpRatio;
    smlaCsaConstInfo.oriMaskMode = sparseFlashMLABaseParams.oriMaskMode;
    smlaCsaConstInfo.cmpMaskMode = sparseFlashMLACmpParams.cmpMaskMode;
    smlaCsaConstInfo.oriWinLeft = sparseFlashMLABaseParams.oriWinLeft;
    smlaCsaConstInfo.oriWinRight = sparseFlashMLABaseParams.oriWinRight;
    smlaCsaConstInfo.isSoftmaxLseEnable = sparseFlashMLABaseParams.returnSoftmaxLse;
    smlaCsaConstInfo.oriKvStride = sparseFlashMLABaseParams.oriKeyStride0;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        smlaCsaConstInfo.cmpKvStride = sparseFlashMLACmpParams.cmpKeyStride0;
    }
    if ASCEND_IS_AIV {
        smlaCsaConstInfo.softmaxScale = sparseFlashMLABaseParams.softmaxScale;
    }
    smlaCsaConstInfo.dSize = 512;
    smlaCsaConstInfo.dSizeV = smlaCsaConstInfo.dSize;
    smlaCsaConstInfo.dSizeVInput = smlaCsaConstInfo.dSize;
    smlaCsaConstInfo.sparseBlockSize = 1;
    smlaCsaConstInfo.actualSeqLenSize = smlaCsaConstInfo.bSize + 1;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        this->smlaCsaConstInfo.isActualLenDimsOriKVNull = 0U;
    } else {
        this->smlaCsaConstInfo.isActualLenDimsOriKVNull = (seqUsedOriKV == nullptr);
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        smlaCsaConstInfo.oriBlockSize = sparseFlashMLABaseParams.oriBlockSize;
        smlaCsaConstInfo.cmpBlockSize = sparseFlashMLABaseParams.cmpBlockSize;
        smlaCsaConstInfo.oriMaxBlockNumPerBatch = sparseFlashMLABaseParams.oriMaxBlockNumPerBatch;
        smlaCsaConstInfo.cmpMaxBlockNumPerBatch = sparseFlashMLACmpParams.cmpMaxBlockNumPerBatch;
    }

    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
        hasCuSeqlensOriKv = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            hasCuSeqlensCmpKv = true;
        }
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ);
        hasActualSeqQlen = true;
    }
    if (seqUsedOriKV != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
        hasActualSeqOriKvlen = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (seqUsedCmpKV != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKV != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
        }
    }

    smlaCsaConstInfo.needInit = 0;
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && smlaCsaConstInfo.oriMaskMode != 0) {
        for (uint32_t bIdx = 0; bIdx < smlaCsaConstInfo.bSize; bIdx++) {
            int64_t s2Size;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                s2Size = actualSeqOriKvlenGm.GetValue(bIdx);
            } else {
                s2Size = GetSmlaSeqLen(bIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm,
                                       cuSeqlensOriKvGm, smlaCsaConstInfo.s2Size);
            }
            int64_t s1Size = GetSmlaSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                           smlaCsaConstInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
                expectQs =
                    GetSmlaSeqLen(bIdx, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, smlaCsaConstInfo.s1Size);
            } else {
                expectQs = smlaCsaConstInfo.s1Size;
            }
            if (s1Size > s2Size || s1Size < expectQs) {
                smlaCsaConstInfo.needInit = 1;
                break;
            }
        }
    } else {
        smlaCsaConstInfo.needInit = 1;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling)
{
    smlaCsaVecBlock.InitGlobalBuffer(oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                                     sequsedQ, sinks, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    smlaCsaCubeBlock.InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ,
                                      cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV,
                                      smlaCsaConstInfo);

    if (oriTopkLength != nullptr) {
        smlaCsaConstInfo.hasOriTopkLength = true;
        oriTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)oriTopkLength);
    } else {
        smlaCsaConstInfo.hasOriTopkLength = false;
    }
    if (cmpTopkLength != nullptr) {
        smlaCsaConstInfo.hasCmpTopkLength = true;
        cmpTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)cmpTopkLength);
    } else {
        smlaCsaConstInfo.hasCmpTopkLength = false;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    // L1: [l1P x2][cube L1], l1P 必须放在最前面保证与 vec 申请地址相同
    uint32_t mm2LeftSize = smlaCsaConstInfo.s1BaseSize * smlaCsaConstInfo.s2BaseSize;
    uint32_t l1PBaseAddr = 0;
    smlaCsaL1PBuffers[0] = {LocalTensor<Q_T>(TPosition::A1, l1PBaseAddr, mm2LeftSize), 0};
    l1PBaseAddr += (mm2LeftSize * sizeof(Q_T));
    smlaCsaL1PBuffers[1] = {LocalTensor<Q_T>(TPosition::A1, l1PBaseAddr, mm2LeftSize), 1};
    l1PBaseAddr += (mm2LeftSize * sizeof(Q_T));
    smlaCsaL1CubeBase = l1PBaseAddr;

    // UB: [bmm2][bmm1 x2][vec UB]
    uint32_t mm1ResultSize = smlaCsaConstInfo.s1BaseSize / CV_RATIO * smlaCsaConstInfo.s2BaseSize;
    uint32_t mm2ResultSize = smlaCsaConstInfo.s1BaseSize / CV_RATIO * 512;
    uint32_t ubBaseAddr = 0;
    smlaCsaBmm2Buffers = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, mm2ResultSize), 0};
    ubBaseAddr += (mm2ResultSize * sizeof(T));
    smlaCsaBmm1Buffers[0] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, mm1ResultSize), 0};
    ubBaseAddr += (mm1ResultSize * sizeof(T));
    smlaCsaBmm1Buffers[1] = {LocalTensor<T>(TPosition::VECIN, ubBaseAddr, mm1ResultSize), 1};
    ubBaseAddr += (mm1ResultSize * sizeof(T));
    smlaCsaVUbBase = ubBaseAddr;

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(smlaCsaBmm1Buffers[0].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(smlaCsaBmm1Buffers[1].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    }

    if constexpr (IS_SPLIT_G || TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        uint32_t v0ResSize = smlaCsaConstInfo.s2BaseSize * 512U * sizeof(Q_T);
        int64_t v0ResTotalOffset;
        if constexpr (IS_SPLIT_G) {
            v0ResTotalOffset = v0ResSize * 3 * (smlaCsaAicIdx >> 1U);
        } else {
            v0ResTotalOffset = v0ResSize * 3 * smlaCsaAicIdx;
        }
        smlaCsaV0ResGmBufferManager.Init(workspace + v0ResTotalOffset);
        v0ResGmBuffers.Init(smlaCsaV0ResGmBufferManager, v0ResSize);
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(0));
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(1));
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(2));
    }
    int64_t fdStagingOffset = 0LL;
    if constexpr (IS_SPLIT_G || TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        constexpr int64_t TRIPLE_BUFFER_NUM = 3LL;
        int64_t v0ResSize = static_cast<int64_t>(smlaCsaConstInfo.s2BaseSize) * smlaCsaConstInfo.dSize * sizeof(Q_T);
        uint32_t v0LogicalSlotCount = IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum();
        fdStagingOffset = v0ResSize * TRIPLE_BUFFER_NUM * v0LogicalSlotCount;
        fdStagingOffset += TRIPLE_BUFFER_NUM * smlaCsaConstInfo.s2BaseSize * sizeof(int32_t) * GetBlockNum();
        if constexpr (IS_VEC_S2PHYADDR) {
            int64_t totalBS1 = (LAYOUT_T == SMLA_LAYOUT::TND) ?
                                   static_cast<int64_t>(smlaCsaConstInfo.s1Size) :
                                   static_cast<int64_t>(smlaCsaConstInfo.bSize) * smlaCsaConstInfo.s1Size;
            if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                          TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                fdStagingOffset += totalBS1 * smlaCsaConstInfo.alignedOriSparseBlockCount * sizeof(int64_t);
            }
            if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                          TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                fdStagingOffset += totalBS1 * smlaCsaConstInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
            }
        }
    }
    fdStagingBufferManager.Init(workspace + fdStagingOffset);
    constexpr uint32_t FD_MAX_SUM_REGION_NUM = 2U;
    uint32_t gSize = static_cast<uint32_t>(smlaCsaConstInfo.gSize);
    uint32_t combineElemSize =
        gSize * smlaCsaConstInfo.dSize +
        FD_MAX_SUM_REGION_NUM * gSize * static_cast<uint32_t>(AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW);
    if constexpr (IS_BATCH_CONSISTENCY) {
        uint32_t intraCoreSlotNum = IS_SPLIT_G ? GetBlockNum() : (GetBlockNum() << 1U);
        uint32_t intraCoreCombineSize = intraCoreSlotNum * combineElemSize * sizeof(float);
        uint32_t crossCoreCombineSize =
            GetBlockNum() * BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * combineElemSize * sizeof(float);
        intraCoreCombineBuffer.Init(fdStagingBufferManager, intraCoreCombineSize);
        crossCoreCombineBuffer.Init(fdStagingBufferManager, crossCoreCombineSize);
    } else {
        uint32_t fdSlotCount = static_cast<uint32_t>(AttentionCommon::FD_MAX_S2_SPLIT_NUM) *
                               (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
        fdStagingBuffer.Init(fdStagingBufferManager, fdSlotCount * combineElemSize * sizeof(float));
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if constexpr (IS_VEC_S2PHYADDR) {
        SyncAll<false>();
    } else if (this->smlaCsaConstInfo.needInit) {
        SyncAll<false>();
    }
    FdRunInfo fdRunInfo;
    if ASCEND_IS_AIV {
        ParseFdRunInfo(fdRunInfo, this->smlaCsaConstInfo, this->smlaCsaMetadataGm);
    }
    ProcessMainLoop();
    if ASCEND_IS_AIV {
        SyncAll();
        if (fdRunInfo.coreEnable) {
            this->smlaCsaVecBlock.ProcessFlashDecode(fdRunInfo, this->smlaCsaConstInfo);
        }
    }
    FreeEvent(this->smlaCsaBmm1Buffers, this->smlaCsaCubeBlock, this->smlaCsaVecBlock, this->smlaCsaConstInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_CORE_ENABLE_INDEX, false));
    int64_t smlaMaxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        smlaMaxS2LoopCnt =
            static_cast<int64_t>(smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < smlaMaxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                }
            }
        }
        return;
    }

    // 从meta data解析分核信息
    uint32_t bN2StartIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_BN2_START_INDEX, false));
    uint32_t gS1StartIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_M_START_INDEX, false));
    uint32_t s2StartIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_S2_START_INDEX, false));
    uint32_t bN2EndIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_BN2_END_INDEX, false));
    uint32_t nextGs1Idx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_M_END_INDEX, false));
    uint32_t s2EndIdx = smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_S2_END_INDEX, false));
    uint32_t firstFdDataWorkspaceIdx =
        smlaCsaMetadataGm.GetValue(GetAttrAbsIndex(smlaCsaAicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, false));

    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0 || s2EndIdx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool smlaNotLast = true;
    bool isFirstLoop = true;
    RunInfo smla35RunInfo[4];
    RunParamStr smla35RunParam;
    smla35RunParam.firstFdDataWorkspaceIdx = firstFdDataWorkspaceIdx;
    int64_t multiCoreInnerIdx = 1;
    int64_t s2SplitIdxCounter = 0;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        smla35RunParam.boIdx = bnIdx;
        smla35RunParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(
            smla35RunParam, this->smlaCsaConstInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm, this->cuSeqlensCmpKvGm,
            this->actualSeqQlenGm, this->actualSeqOriKvlenGm, this->actualSeqCmpKvlenGm, this->cmpResidualKvGm,
            this->hasActualSeqQlen, this->hasActualSeqOriKvlen, this->hasActualSeqCmpKvlen, this->hasCuSeqlensCmpKv);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(smla35RunParam, this->smlaCsaConstInfo, lastBN, nextGs1Idx, gS1StartIdx,
                                              s2EndIdx);

        int64_t gS1LoopEnd = lastBN ? (smla35RunParam.gs1LoopEndIdx + PRELOAD_NUM) : smla35RunParam.gs1LoopEndIdx;
        for (int64_t gS1Index = smla35RunParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool smlaNotLastThreeLoop = true;
            bool smlaNotLastTwoLoop = true;
            if (lastBN) {
                int32_t smlaExtraGS1 = gS1Index - smla35RunParam.gs1LoopEndIdx;
                switch (smlaExtraGS1) {
                    case 0:
                        smlaNotLastThreeLoop = false;
                        break;
                    case 1:
                        smlaNotLastTwoLoop = false;
                        smlaNotLastThreeLoop = false;
                        break;
                    case 2:
                        smlaNotLast = false;
                        smlaNotLastTwoLoop = false;
                        smlaNotLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (smlaNotLastThreeLoop) {
                ComputeAxisIdxByBnAndGs1<TEMPLATE_INTF_ARGS>(bnIdx, gS1Index, smla35RunParam, this->smlaCsaConstInfo,
                                                             this->smlaCsaAicIdx);
                bool smlaS1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(smla35RunParam, this->smlaCsaConstInfo,
                                                                           gS1Index, this->cuSeqlensQGm);
                bool smlaS2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(bnIdx, gS1Index, this->cuSeqlensQGm, oriTopkLengthGm,
                                                          cmpTopkLengthGm, smla35RunParam, this->smlaCsaConstInfo);
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t oriLoad = smla35RunParam.s2OriLineEndIdx - smla35RunParam.s2OriLineStartIdx;
                    int64_t cmpLoad = smla35RunParam.s2CmpLineEndIdx - smla35RunParam.s2CmpLineStartIdx;
                    int64_t totalLoad = oriLoad + cmpLoad;
                    int64_t s2BaseSize = static_cast<int64_t>(smlaCsaConstInfo.s2BaseSize);
                    int64_t rawReductionBlockSize = (totalLoad + 31LL) / 32LL;
                    int64_t reductionBlockSize = (rawReductionBlockSize + s2BaseSize - 1LL) / s2BaseSize * s2BaseSize;
                    smla35RunParam.baseBlockNumPerReductionBlock =
                        reductionBlockSize > 0 ? reductionBlockSize / s2BaseSize : 1LL;
                }
                if (!smlaS2NoNeedCalc) {
                    bool isFirstS2RangeTask = (bnIdx == bN2StartIdx && gS1Index == smla35RunParam.gs1LoopStartIdx);
                    bool isLastS2RangeTask = (lastBN && gS1Index == smla35RunParam.gs1LoopEndIdx - 1);
                    int64_t s2StartPoint =
                        ConvertS2MetadataBlockToToken(smla35RunParam, this->smlaCsaConstInfo, s2StartIdx);
                    int64_t s2EndPoint =
                        (isLastS2RangeTask && s2EndIdx == 0) ?
                            0 :
                            ConvertS2MetadataBlockToToken(smla35RunParam, this->smlaCsaConstInfo, s2EndIdx);
                    smlaS2NoNeedCalc = ApplyS2MetadataRange(smla35RunParam, this->smlaCsaConstInfo, s2StartPoint,
                                                            s2EndPoint, isFirstS2RangeTask, isLastS2RangeTask);
                } else {
                    smla35RunParam.isCrossCoreSplit = false;
                }
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (smlaS1NoNeedCalc || smlaS2NoNeedCalc) {
                    continue;
                }
                if constexpr (!IS_BATCH_CONSISTENCY) {
                    if (smla35RunParam.isCrossCoreSplit) {
                        smla35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                s2LoopLimit = smla35RunParam.s2LoopEndIdx - 1;
                if constexpr (IS_SPLIT_G) {
                    smlaMaxS2LoopCnt -= (s2LoopLimit + 1);
                }
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t smla35ReductionBlockSize = smla35RunParam.baseBlockNumPerReductionBlock > 0 ?
                                                           smla35RunParam.baseBlockNumPerReductionBlock :
                                                           1LL;
                    int64_t smla35ReductionLoop = s2LoopCount;
                    if (s2LoopCount >= smla35RunParam.oriKvLoopEndIdx) {
                        smla35ReductionLoop +=
                            (smla35ReductionBlockSize - smla35RunParam.oriKvLoopEndIdx % smla35ReductionBlockSize) %
                            smla35ReductionBlockSize;
                    }
                    if (smla35RunParam.isCrossCoreSplit && smla35ReductionLoop % smla35ReductionBlockSize == 0) {
                        smla35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                              TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                              TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                    if (smlaNotLastThreeLoop) {
                        RunInfo &runInfo1 = smla35RunInfo[taskId % 4];
                        SetRunInfo<TEMPLATE_INTF_ARGS>(runInfo1, smla35RunParam, taskId, s2LoopCount, s2LoopLimit,
                                                       multiCoreInnerIdx, this->smlaCsaConstInfo);
                    }
                    if ASCEND_IS_AIV {
                        if (smlaNotLastThreeLoop) {
                            RunInfo &runInfo1 = smla35RunInfo[taskId % 4];
                            this->smlaCsaVecBlock.ProcessVec0(this->v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                              this->smlaCsaConstInfo);
                        }
                        if (taskId > 1 && smlaNotLast) {
                            uint32_t bmm1Slot = smlaCsaBmm1GetFlag;
                            smlaCsaBmm1GetFlag ^= 1;
                            uint32_t l1PSlot = smlaCsaL1PGetFlag;
                            smlaCsaL1PGetFlag ^= 1;
                            auto &runInfo2 = smla35RunInfo[(taskId + 2) % 4];
                            this->smlaCsaVecBlock.ProcessVec1(this->smlaCsaL1PBuffers[l1PSlot],
                                                              this->smlaCsaBmm1Buffers[bmm1Slot], runInfo2,
                                                              this->smlaCsaConstInfo);
                        }
                        if (taskId > 2) {
                            RunInfo &runInfo3 = smla35RunInfo[(taskId + 1) % 4];
                            this->smlaCsaVecBlock.ProcessVec2(this->smlaCsaBmm2Buffers, runInfo3,
                                                              this->smlaCsaConstInfo);
                        }
                    } else {
                        if (taskId > 0 && smlaNotLastTwoLoop) {
                            RunInfo &runInfo1 = smla35RunInfo[(taskId + 3) % 4];
                            this->smlaCsaCubeBlock.IterateLoadQK(this->v0ResGmBuffers.Get(runInfo1.taskIdMod3),
                                                                 runInfo1, this->smlaCsaConstInfo, isFirstLoop);
                            isFirstLoop = false;
                        } else {
                            if constexpr (IS_SPLIT_G) {
                                if (taskId > 0 && smlaMaxS2LoopCnt > 0) {
                                    smlaMaxS2LoopCnt--;
                                    CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                                    CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                                }
                            }
                        }
                        if (taskId > 1 && smlaNotLast) {
                            uint32_t bmm1Slot = smlaCsaBmm1GetFlag;
                            smlaCsaBmm1GetFlag ^= 1;
                            auto &runInfo2 = smla35RunInfo[(taskId + 2) % 4];
                            RunInfo &runInfoNext = smla35RunInfo[(taskId + 3) % 4];
                            this->smlaCsaCubeBlock.IterateBmm1(this->smlaCsaBmm1Buffers[bmm1Slot], smlaNotLastTwoLoop,
                                                               runInfoNext, runInfo2, this->smlaCsaConstInfo);
                        }
                        if (taskId > 2) {
                            uint32_t l1PSlot = smlaCsaL1PGetFlag;
                            smlaCsaL1PGetFlag ^= 1;
                            RunInfo &runInfo3 = smla35RunInfo[(taskId + 1) % 4];
                            this->smlaCsaCubeBlock.IterateBmm2(this->smlaCsaBmm2Buffers,
                                                               this->smlaCsaL1PBuffers[l1PSlot], runInfo3,
                                                               this->smlaCsaConstInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < smlaMaxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
            }
        }
    }
}

} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
