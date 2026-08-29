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
 * \file quant_sparse_flash_mla_csa_kernel.h
 * \brief
 */

#ifndef QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
#define QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
#include "quant_sparse_flash_mla_common_arch35.h"
#include "quant_sparse_flash_mla_kvcache.h"
#include "quant_sparse_flash_mla_csa_block_cube.h"
#include "quant_sparse_flash_mla_csa_block_vector.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "../quant_sparse_flash_mla_metadata.h"

#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif
#if __has_include("../../../sparse_flash_mla/op_kernel/arch35/common/buffers_policy_3buff_sfa.h")
#include "../../../sparse_flash_mla/op_kernel/arch35/common/buffers_policy_3buff_sfa.h"
#else
#include "../../sparse_flash_mla/arch35/common/buffers_policy_3buff_sfa.h"
#endif

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class QuantSparseFlashMlaCsa {
public:
    ARGS_TRAITS;
    __aicore__ inline QuantSparseFlashMlaCsa(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *qDescale, __gm__ uint8_t *oriKVDescale, __gm__ uint8_t *cmpKVDescale,
                                __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV,
                                __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKv,
                                __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace, const QuantSparseFlashMlaTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline int64_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                           __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv,
                                           __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedCmpKv,
                                           __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *qDescale, __gm__ uint8_t *oriKVDescale,
                                            __gm__ uint8_t *cmpKVDescale, __gm__ uint8_t *oriSparseIndices,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                            __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv,
                                            __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
                                            __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
                                            __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
                                            const QuantSparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void FreeEvent();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    TPipe *pipe;

    const QuantSparseFlashMlaTilingData *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 3;

    uint32_t crossCoreSyncBufId = 0;
    /* 核间通道 */
    BufferManager<BufferType::GM> v0ResGmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    BuffersPolicy3buffSFA<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    /* GM信息 */
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqQlenGm;
    GlobalTensor<int32_t> actualSeqOriKvlenGm;
    GlobalTensor<int32_t> actualSeqCmpKvlenGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
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
    /* 核Index信息 */
    int32_t aicIdx;
    /* 分核信息 */
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t s2StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t s2EndIdx;
    uint32_t hasLoad;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *qDescale,
    __gm__ uint8_t *oriKVDescale, __gm__ uint8_t *cmpKVDescale, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const QuantSparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe)
{
    __gm__ uint8_t *sequsedOriKv = seqUsedOriKV;
    __gm__ uint8_t *sequsedCmpKv = seqUsedCmpKV;
    fa_base_matmul::ResetIdCounter();
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
        this->tilingData = tiling;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    // 从meta data解析分核信息
    bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false));
    bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false));
    hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;

    this->pipe = tPipe;
    this->ParseTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, sequsedOriKv, cuSeqlensCmpKv, sequsedCmpKv,
                          cmpResidualKv);
    this->InitGlobalBuffer(query, oriKV, cmpKV, qDescale, oriKVDescale, cmpKVDescale, oriSparseIndices,
                           cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv,
                           sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, oriTopkLength, cmpTopkLength, sinks,
                           workspace, tiling, tPipe); // gm设置
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    vecBlock.CleanOutput(attentionOut, softmaxLse, constInfo);
    if ASCEND_IS_AIV {
        if constexpr ((TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                       TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                       TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                      IS_VEC_S2PHYADDR) {
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen,
                                        hasCuSeqlensQ, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm,
                                        cuSeqlensOriKvGm, oriTopkLengthGm, hasActualSeqCmpKvlen, hasCuSeqlensCmpKv,
                                        actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm,
                                        actualSeqQlenGm, cuSeqlensQGm, workspace, constInfo);
        }
    }
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::GetSeqLen(
    int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens, GlobalTensor<int32_t> &actualSeqGm,
    GlobalTensor<int32_t> &cuSeqlensGm, int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    } else {
        return defaultSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    auto &quantSparseFlashMlaBaseParams = this->tilingData->baseParams;
    constInfo.bSize = quantSparseFlashMlaBaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = quantSparseFlashMlaBaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = quantSparseFlashMlaBaseParams.qSeqSize;
    constInfo.s2Size = quantSparseFlashMlaBaseParams.kvSeqSize;
    constInfo.cmpS2Size = quantSparseFlashMlaBaseParams.cmpKvSeqSize;
    constInfo.oriSparseBlockCount = quantSparseFlashMlaBaseParams.oriSparseBlockCount;
    constInfo.cmpSparseBlockCount = quantSparseFlashMlaBaseParams.cmpSparseBlockCount;
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    constInfo.alignedOriSparseBlockCount =
        (constInfo.oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    constInfo.alignedCmpSparseBlockCount =
        (constInfo.cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        constInfo.cmpRatio = quantSparseFlashMlaBaseParams.cmpRatio;
    }
    constInfo.oriMaskMode = quantSparseFlashMlaBaseParams.oriMaskMode;
    constInfo.cmpMaskMode = quantSparseFlashMlaBaseParams.cmpMaskMode;
    constInfo.oriWinLeft = quantSparseFlashMlaBaseParams.oriWinLeft;
    constInfo.oriWinRight = quantSparseFlashMlaBaseParams.oriWinRight;
    constInfo.softmaxScale = quantSparseFlashMlaBaseParams.softmaxScale;
    constInfo.oriKvStride = quantSparseFlashMlaBaseParams.oriKvStride;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        constInfo.cmpKvStride = quantSparseFlashMlaBaseParams.cmpKvStride;
    }
    constInfo.dSize = quantSparseFlashMlaBaseParams.dSize;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = quantSparseFlashMlaBaseParams.dSizeVInput;
    constInfo.isSoftmaxLseEnable = quantSparseFlashMlaBaseParams.returnSoftmaxLse;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;

    if constexpr (isPa) {
        constInfo.oriBlockSize = quantSparseFlashMlaBaseParams.paOriBlockSize;
        constInfo.oriMaxBlockNumPerBatch = quantSparseFlashMlaBaseParams.oriMaxBlockNumPerBatch;
        if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            constInfo.cmpBlockSize = quantSparseFlashMlaBaseParams.paCmpBlockSize;
            constInfo.cmpMaxBlockNumPerBatch = quantSparseFlashMlaBaseParams.cmpMaxBlockNumPerBatch;
        }
    }

    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
        hasCuSeqlensOriKv = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            hasCuSeqlensCmpKv = true;
        }
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ);
        hasActualSeqQlen = true;
    }
    if (sequsedOriKv != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
        hasActualSeqOriKvlen = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (sequsedCmpKv != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedCmpKv);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKv != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKv);
        }
    }

    constInfo.needInit = 0;
    if (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && constInfo.oriMaskMode != 0) {
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t s2Size = GetSeqLen(bIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm,
                                       cuSeqlensOriKvGm, constInfo.s2Size);
            int64_t s1Size =
                GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                expectQs = GetSeqLen(bIdx, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            } else {
                expectQs = constInfo.s1Size;
            }
            if (s1Size > s2Size || s1Size < expectQs) {
                constInfo.needInit = 1;
                break;
            }
        }
    } else {
        constInfo.needInit = 1;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *qDescale,
    __gm__ uint8_t *oriKVDescale, __gm__ uint8_t *cmpKVDescale, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const QuantSparseFlashMlaTilingData *__restrict tiling, TPipe *tPipe)
{
    vecBlock.InitGlobalBuffer(oriKV, cmpKV, qDescale, oriKVDescale, cmpKVDescale, oriSparseIndices, cmpSparseIndices,
                              oriBlockTable, cmpBlockTable, sequsedQ, sinks, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    cubeBlock.InitCubeInput(cuSeqlensQ, sequsedQ, constInfo);
    if (oriTopkLength != nullptr) {
        constInfo.hasOriTopkLength = true;
        oriTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)oriTopkLength);
    } else {
        constInfo.hasOriTopkLength = false;
    }
    if (cmpTopkLength != nullptr) {
        constInfo.hasCmpTopkLength = true;
        cmpTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)cmpTopkLength);
    } else {
        constInfo.hasCmpTopkLength = false;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 416 * 1024); // L1P+L1Right总计416k, 其余96k给L1Q使用
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    l1PBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID); // 0
    crossCoreSyncBufId++;
    l1PBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID); // 1
    crossCoreSyncBufId++;

    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID); // 2
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID); // 3
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID); // 4
    crossCoreSyncBufId++;

    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    bmm2Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId); // 5
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId); // 6
    crossCoreSyncBufId++;
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId); // 7
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
    uint32_t v0ResSize = constInfo.s2BaseSize * 512U * sizeof(Q_T);
    int64_t totalOffset = 0;
    if constexpr (IS_SPLIT_G) {
        totalOffset = v0ResSize * 3 * (aicIdx >> 1U);
    } else {
        totalOffset = v0ResSize * 3 * aicIdx;
    }
    v0ResGmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(v0ResGmBufferManager, v0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId); // 8
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId); // 9
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId); // 10
    crossCoreSyncBufId++;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->constInfo.needInit) {
        SyncAll<false>();
    }
    ICachePreLoad(6);
    ProcessMainLoop();
    FreeEvent();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    int64_t maxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(15);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(15);
                }
            }
        }
        return;
    }

    // 从meta data解析分核信息
    uint32_t s2LoopLimit = 0;
    int64_t taskId = 0;
    bool isFirstLoop = true;
    bool notLast = true;
    bool notLastTwoLoop = true;
    RunInfo runInfo[4];
    RunParamStr runParam;
    int64_t multiCoreInnerIdx = 1;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm,
                                              this->cuSeqlensCmpKvGm, this->actualSeqQlenGm, this->actualSeqOriKvlenGm,
                                              this->actualSeqCmpKvlenGm, this->cmpResidualKvGm, this->hasCuSeqlensOriKv,
                                              this->hasCuSeqlensCmpKv, this->hasActualSeqQlen,
                                              this->hasActualSeqOriKvlen, this->hasActualSeqCmpKvlen);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastThreeLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc =
                    ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Index, this->cuSeqlensQGm);
                bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(
                    bnIdx, gS1Index, this->cuSeqlensQGm, oriTopkLengthGm, cmpTopkLengthGm, runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= runParam.s2LoopEndIdx;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastThreeLoop) {
                    RunInfo &runInfo1 = runInfo[taskId % 4];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                }
                if ASCEND_IS_AIV {
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[taskId % 4];
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(runInfo1.taskIdMod3),
                                                   v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1, this->constInfo);
                    }
                    if (taskId > 1 && notLast) {
                        auto &runInfo2 = runInfo[(taskId + 2) % 4];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo2,
                                                   this->constInfo);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                } else {
                    if (taskId > 0 && notLastTwoLoop) {
                        RunInfo &runInfo1 = runInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateLoadQK(this->l1RightBuffers.Get(runInfo1.taskIdMod3),
                                                      v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                      this->constInfo, isFirstLoop);
                        isFirstLoop = false;
                    } else {
                        if constexpr (IS_SPLIT_G) {
                            if (taskId > 0 && maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
                            }
                        }
                    }
                    if (taskId > 1 && notLast) {
                        auto &runInfo2 = runInfo[(taskId + 2) % 4];
                        RunInfo &runInfoNext = runInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(),
                                                    this->l1RightBuffers.Get(runInfo2.taskIdMod3),
                                                    v0ResGmBuffers.Get(runInfo2.taskIdMod3), notLastTwoLoop,
                                                    runInfoNext, runInfo2, this->constInfo);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers,
                                                    this->l1RightBuffers.Get(runInfo3.taskIdMod3), runInfo3,
                                                    this->constInfo);
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
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t halfG = (constInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
        runParam.gSplitSize = (aicIdx % 2 == 0) ? halfG : (constInfo.gSize - halfG);
    } else {
        runParam.goIdx = 0;
        runParam.gSplitSize = constInfo.gSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineOriEndIdx;
    } else {
        runInfo.s2StartIdx = 0;
        runInfo.s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2OriSize = runParam.actualS2OriSize;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::FreeEvent()
{
    if ASCEND_IS_AIC {
        bmm1Buffers.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
        bmm2Buffers.Get().WaitCrossCore();
        this->cubeBlock.FreeEvent();
    } else {
        this->vecBlock.FreeEvent(constInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo &runInfo,
                                                                                            RunParamStr &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize; // D>128 这里需要适配
    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = (runInfo.s2LoopCount >= runParam.oriKvLoopEndIdx) ?
                               (runInfo.s2LoopCount - runParam.oriKvLoopEndIdx) :
                               runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}
} // namespace BaseApi
#endif // QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
