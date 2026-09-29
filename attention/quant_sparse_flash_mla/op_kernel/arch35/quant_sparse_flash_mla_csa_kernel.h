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

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"
#include "../../../sparse_flash_mla/op_kernel/arch35/common/buffers_policy_3buff_sfa.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using AttentionCommon::FdRunInfo;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class QuantSparseFlashMlaCsa {
public:
    QSMLA_ARGS_TRAITS;
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
    __aicore__ inline int64_t GetQsmlaSeqLen(int32_t batchIndex, bool useActualLength, bool useCuSeqlens,
                                             GlobalTensor<int32_t> &actualLengthGm, GlobalTensor<int32_t> &cuLengthGm,
                                             int64_t defaultLength);
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
    __aicore__ inline void ParseFdRunInfo(FdRunInfo &fdRunInfo);
    __aicore__ inline int64_t ConvertS2MetadataBlockToToken(const RunParamStr &qs35RunParam, const ConstInfo &constInfo,
                                                            uint32_t s2BlockIdx);
    __aicore__ inline bool ApplyS2MetadataRange(RunParamStr &qs35RunParam, ConstInfo &constInfo, int64_t s2StartPoint,
                                                int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask);
    __aicore__ inline void SetRunInfo(RunInfo &qs35RunInfo, RunParamStr &qs35RunParam, int64_t taskId,
                                      int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &qs35RunInfo, RunParamStr &qs35RunParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &qs35RunParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &qs35RunParam, RunInfo &qs35RunInfo);
    TPipe *pipe;

    const QuantSparseFlashMlaTilingData *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 3;

    uint32_t crossCoreSyncBufId = 0;
    /* 核间通道 */
    BufferManager<BufferType::GM> v0ResGmBufferManager;
    BufferManager<BufferType::GM> fdStagingBufferManager;
    BuffersPolicySingleBuffer<BufferType::GM> fdStagingBuffer;
    BuffersPolicySingleBuffer<BufferType::GM> intraCoreCombineBuffer;
    BuffersPolicySingleBuffer<BufferType::GM> crossCoreCombineBuffer;

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
    bool qsmlaHasCuSeqlensQ = false;
    bool qsmlaHasCuSeqlensOriKv = false;
    bool qsmlaHasCuSeqlensCmpKv = false;
    bool qsmlaHasActualSeqQlen = false;
    bool qsmlaHasActualSeqOriKvlen = false;
    bool qsmlaHasActualSeqCmpKvlen = false;
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
    if (nextGs1Idx != 0 || s2EndIdx != 0) {
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
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, qsmlaHasActualSeqQlen,
                                        qsmlaHasCuSeqlensQ, qsmlaHasActualSeqOriKvlen, qsmlaHasCuSeqlensOriKv,
                                        actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
                                        qsmlaHasActualSeqCmpKvlen, qsmlaHasCuSeqlensCmpKv, actualSeqCmpKvlenGm,
                                        cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm, actualSeqQlenGm,
                                        cuSeqlensQGm, workspace, constInfo);
        }
    }
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    if constexpr (IS_BATCH_CONSISTENCY) {
        vecBlock.InitS2SplitStaging(intraCoreCombineBuffer.Get(), crossCoreCombineBuffer.Get());
    } else {
        vecBlock.InitS2SplitStaging(fdStagingBuffer.Get());
    }
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::GetQsmlaSeqLen(
    int32_t batchIndex, bool useActualLength, bool useCuSeqlens, GlobalTensor<int32_t> &actualLengthGm,
    GlobalTensor<int32_t> &cuLengthGm, int64_t defaultLength)
{
    if (useActualLength) {
        return actualLengthGm.GetValue(batchIndex);
    } else if (useCuSeqlens) {
        return cuLengthGm.GetValue(batchIndex + 1) - cuLengthGm.GetValue(batchIndex);
    } else {
        return defaultLength;
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
        qsmlaHasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
        qsmlaHasCuSeqlensOriKv = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            qsmlaHasCuSeqlensCmpKv = true;
        }
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ);
        qsmlaHasActualSeqQlen = true;
    }
    if (sequsedOriKv != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedOriKv);
        qsmlaHasActualSeqOriKvlen = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (sequsedCmpKv != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedCmpKv);
            qsmlaHasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKv != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKv);
        }
    }

    constInfo.needInit = 0;
    if (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && constInfo.oriMaskMode != 0) {
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t localS2Size = GetQsmlaSeqLen(bIdx, qsmlaHasActualSeqOriKvlen, qsmlaHasCuSeqlensOriKv,
                                                 actualSeqOriKvlenGm, cuSeqlensOriKvGm, constInfo.s2Size);
            int64_t localS1Size = GetQsmlaSeqLen(bIdx, qsmlaHasActualSeqQlen, qsmlaHasCuSeqlensQ, actualSeqQlenGm,
                                                 cuSeqlensQGm, constInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                expectQs =
                    GetQsmlaSeqLen(bIdx, false, qsmlaHasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            } else {
                expectQs = constInfo.s1Size;
            }
            if (localS1Size > localS2Size || localS1Size < expectQs) {
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
    l1BufferManager.Init(pipe, 416 * 1024);
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    l1PBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1PBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;

    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;
    l1RightBuffers.Get().SetCrossCoreID(crossCoreSyncBufId, INVALID_CROSS_CORE_EVENT_ID);
    crossCoreSyncBufId++;

    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    bmm2Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    bmm1Buffers.Get().SetCrossCoreID(crossCoreSyncBufId, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
    uint32_t v0ResSize = constInfo.s2BaseSize * 512U * sizeof(Q_T);
    int64_t totalOffset;
    if constexpr (IS_SPLIT_G) {
        totalOffset = static_cast<int64_t>(v0ResSize) * 3 * (aicIdx >> 1U);
    } else {
        totalOffset = static_cast<int64_t>(v0ResSize) * 3 * aicIdx;
    }
    v0ResGmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(v0ResGmBufferManager, v0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, crossCoreSyncBufId);
    crossCoreSyncBufId++;

    uint64_t v0RegionSize = static_cast<uint64_t>(v0ResSize) * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
    uint64_t phyAddrRegionSize = 0;
    if constexpr (IS_VEC_S2PHYADDR) {
        uint64_t totalBS1 = LAYOUT_T == QSMLA_LAYOUT::TND ? static_cast<uint64_t>(constInfo.s1Size) :
                                                            static_cast<uint64_t>(constInfo.bSize) * constInfo.s1Size;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            phyAddrRegionSize += totalBS1 * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        }
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            phyAddrRegionSize += totalBS1 * constInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
        }
    }
    fdStagingBufferManager.Init(workspace + v0RegionSize + phyAddrRegionSize);
    constexpr uint32_t FD_MAX_SUM_REGION_NUM = 2U;
    uint32_t gSize = static_cast<uint32_t>(constInfo.gSize);
    if constexpr (IS_BATCH_CONSISTENCY) {
        uint32_t combineElemSize =
            gSize * constInfo.dSize +
            FD_MAX_SUM_REGION_NUM * gSize * static_cast<uint32_t>(AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW);
        uint32_t intraCoreSlotNum = IS_SPLIT_G ? GetBlockNum() : (GetBlockNum() << 1U);
        uint32_t intraCoreCombineSize = intraCoreSlotNum * combineElemSize * sizeof(float);
        uint32_t crossCoreCombineSize =
            GetBlockNum() * BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * combineElemSize * sizeof(float);
        intraCoreCombineBuffer.Init(fdStagingBufferManager, intraCoreCombineSize);
        crossCoreCombineBuffer.Init(fdStagingBufferManager, crossCoreCombineSize);
    } else {
        uint32_t fdSlotCount = static_cast<uint32_t>(AttentionCommon::FD_MAX_S2_SPLIT_NUM) *
                               (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
        uint32_t fdStagingSize =
            fdSlotCount * (gSize * constInfo.dSize * sizeof(float) +
                           FD_MAX_SUM_REGION_NUM * gSize *
                               static_cast<uint32_t>(AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW) * sizeof(float));
        fdStagingBuffer.Init(fdStagingBufferManager, fdStagingSize);
    }
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

    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
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
    SyncAll();
    if ASCEND_IS_AIV {
        FdRunInfo fdRunInfo;
        ParseFdRunInfo(fdRunInfo);
        if (fdRunInfo.coreEnable) {
            this->vecBlock.ProcessFlashDecode(fdRunInfo, this->constInfo);
        }
    }
    FreeEvent();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    int64_t qsmlaMaxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        qsmlaMaxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < qsmlaMaxS2LoopCnt; loopCnt++) {
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
    bool qsmlaNotLast = true;
    bool qsmlaNotLastTwoLoop = true;
    RunInfo qs35RunInfo[4];
    RunParamStr qs35RunParam;
    qs35RunParam.firstFdDataWorkspaceIdx =
        metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, false));
    int64_t qsmlaMultiCoreInnerIdx = 1;
    int64_t s2SplitIdxCounter = 0;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        qs35RunParam.boIdx = bnIdx;
        qs35RunParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(
            qs35RunParam, this->constInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm, this->cuSeqlensCmpKvGm,
            this->actualSeqQlenGm, this->actualSeqOriKvlenGm, this->actualSeqCmpKvlenGm, this->cmpResidualKvGm,
            this->qsmlaHasCuSeqlensOriKv, this->qsmlaHasCuSeqlensCmpKv, this->qsmlaHasActualSeqQlen,
            this->qsmlaHasActualSeqOriKvlen, this->qsmlaHasActualSeqCmpKvlen);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(qs35RunParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx, s2EndIdx);

        int64_t qsmlaGS1LoopEnd = lastBN ? (qs35RunParam.gs1LoopEndIdx + PRELOAD_NUM) : qs35RunParam.gs1LoopEndIdx;
        for (int64_t gS1Index = qs35RunParam.gs1LoopStartIdx; gS1Index < qsmlaGS1LoopEnd; gS1Index++) {
            bool qsmlaNotLastThreeLoop = true;
            if (lastBN) {
                int32_t qsmlaExtraGS1 = gS1Index - qs35RunParam.gs1LoopEndIdx;
                switch (qsmlaExtraGS1) {
                    case 0:
                        qsmlaNotLastThreeLoop = false;
                        break;
                    case 1:
                        qsmlaNotLastTwoLoop = false;
                        qsmlaNotLastThreeLoop = false;
                        break;
                    case 2:
                        qsmlaNotLast = false;
                        qsmlaNotLastTwoLoop = false;
                        qsmlaNotLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (qsmlaNotLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, qs35RunParam);
                bool qsmlaS1NoNeedCalc =
                    ComputeParamS1<TEMPLATE_INTF_ARGS>(qs35RunParam, this->constInfo, gS1Index, this->cuSeqlensQGm);
                bool qsmlaS2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(bnIdx, gS1Index, this->cuSeqlensQGm, oriTopkLengthGm,
                                                          cmpTopkLengthGm, qs35RunParam, this->constInfo);
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t s2Load =
                        qs35RunParam.s2LineOriEndIdx - qs35RunParam.s2LineStartIdx + qs35RunParam.s2CmpLineEndIdx;
                    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
                    int64_t s2PerReduceBlock = ((s2Load + 31LL) / 32LL + s2BaseSize - 1) >> 7 << 7;
                    int64_t baseBlockNum = s2PerReduceBlock >> 7;
                    qs35RunParam.baseBlockNumPerReductionBlock = baseBlockNum > 0 ? baseBlockNum : 1LL;
                }
                if (!qsmlaS2NoNeedCalc) {
                    bool isFirstS2RangeTask = bnIdx == bN2StartIdx && gS1Index == qs35RunParam.gs1LoopStartIdx;
                    bool isLastS2RangeTask = lastBN && gS1Index == qs35RunParam.gs1LoopEndIdx - 1;
                    int64_t s2StartPoint = ConvertS2MetadataBlockToToken(qs35RunParam, this->constInfo, s2StartIdx);
                    int64_t s2EndPoint = isLastS2RangeTask && s2EndIdx == 0 ?
                                             0 :
                                             ConvertS2MetadataBlockToToken(qs35RunParam, this->constInfo, s2EndIdx);
                    qsmlaS2NoNeedCalc = ApplyS2MetadataRange(qs35RunParam, this->constInfo, s2StartPoint, s2EndPoint,
                                                             isFirstS2RangeTask, isLastS2RangeTask);
                } else {
                    qs35RunParam.isCrossCoreSplit = false;
                }
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (qsmlaS1NoNeedCalc || qsmlaS2NoNeedCalc) {
                    continue;
                }
                if constexpr (!IS_BATCH_CONSISTENCY) {
                    if (qs35RunParam.isCrossCoreSplit) {
                        qs35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if constexpr (IS_SPLIT_G) {
                    qsmlaMaxS2LoopCnt -= qs35RunParam.s2LoopEndIdx;
                }
                s2LoopLimit = qs35RunParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t qs35ReductionBlockSize = qs35RunParam.baseBlockNumPerReductionBlock > 0 ?
                                                         qs35RunParam.baseBlockNumPerReductionBlock :
                                                         1LL;
                    int64_t qs35ReductionLoop = s2LoopCount;
                    if (s2LoopCount >= qs35RunParam.oriKvLoopEndIdx) {
                        qs35ReductionLoop +=
                            (qs35ReductionBlockSize - qs35RunParam.oriKvLoopEndIdx % qs35ReductionBlockSize) %
                            qs35ReductionBlockSize;
                    }
                    if (qs35RunParam.isCrossCoreSplit && qs35ReductionLoop % qs35ReductionBlockSize == 0) {
                        qs35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if (qsmlaNotLastThreeLoop) {
                    RunInfo &runInfo1 = qs35RunInfo[taskId % 4];
                    this->SetRunInfo(runInfo1, qs35RunParam, taskId, s2LoopCount, s2LoopLimit, qsmlaMultiCoreInnerIdx);
                }
                if ASCEND_IS_AIV {
                    if (qsmlaNotLastThreeLoop) {
                        RunInfo &runInfo1 = qs35RunInfo[taskId % 4];
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(runInfo1.taskIdMod3),
                                                   v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1, this->constInfo);
                    }
                    if (taskId > 1 && qsmlaNotLast) {
                        auto &runInfo2 = qs35RunInfo[(taskId + 2) % 4];
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo2,
                                                   this->constInfo);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = qs35RunInfo[(taskId + 1) % 4];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                } else {
                    if (taskId > 0 && qsmlaNotLastTwoLoop) {
                        RunInfo &runInfo1 = qs35RunInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateLoadQK(this->l1RightBuffers.Get(runInfo1.taskIdMod3),
                                                      v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                      this->constInfo, isFirstLoop);
                        isFirstLoop = false;
                    } else {
                        if constexpr (IS_SPLIT_G) {
                            if (taskId > 0 && qsmlaMaxS2LoopCnt > 0) {
                                qsmlaMaxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
                            }
                        }
                    }
                    if (taskId > 1 && qsmlaNotLast) {
                        auto &runInfo2 = qs35RunInfo[(taskId + 2) % 4];
                        RunInfo &runInfoNext = qs35RunInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(),
                                                    this->l1RightBuffers.Get(runInfo2.taskIdMod3),
                                                    v0ResGmBuffers.Get(runInfo2.taskIdMod3), qsmlaNotLastTwoLoop,
                                                    runInfoNext, runInfo2, this->constInfo);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = qs35RunInfo[(taskId + 1) % 4];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers,
                                                    this->l1RightBuffers.Get(runInfo3.taskIdMod3), runInfo3,
                                                    this->constInfo);
                    }
                }
                ++taskId;
            }
            ++qsmlaMultiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < qsmlaMaxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ConvertS2MetadataBlockToToken(
    const RunParamStr &qs35RunParam, const ConstInfo &constInfo, uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriLen = qs35RunParam.s2LineOriEndIdx - qs35RunParam.s2LineStartIdx;
    int64_t cmpLen = qs35RunParam.s2CmpLineEndIdx - qs35RunParam.s2CmpLineStartIdx;
    int64_t qs35ReductionBlockSize =
        qs35RunParam.baseBlockNumPerReductionBlock > 0 ? qs35RunParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t reductionBlockSize = qs35ReductionBlockSize * s2BaseSize;
    int64_t oriReductionBlockNum = (oriLen + reductionBlockSize - 1) / reductionBlockSize;
    if (s2BlockIdx <= oriReductionBlockNum) {
        int64_t oriToken = static_cast<int64_t>(s2BlockIdx) * reductionBlockSize;
        return oriToken < oriLen ? oriToken : oriLen;
    }
    int64_t cmpToken = (static_cast<int64_t>(s2BlockIdx) - oriReductionBlockNum) * reductionBlockSize;
    return oriLen + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ApplyS2MetadataRange(
    RunParamStr &qs35RunParam, ConstInfo &constInfo, int64_t s2StartPoint, int64_t s2EndPoint, bool isFirstS2RangeTask,
    bool isLastS2RangeTask)
{
    int64_t oriStart = qs35RunParam.s2LineStartIdx;
    int64_t oriEnd = qs35RunParam.s2LineOriEndIdx;
    int64_t oriLen = oriEnd - oriStart;
    int64_t cmpStart = qs35RunParam.s2CmpLineStartIdx;
    int64_t cmpEnd = qs35RunParam.s2CmpLineEndIdx;
    int64_t cmpLen = cmpEnd - cmpStart;
    int64_t totalLen = oriLen + cmpLen;

    int64_t effectiveS2EndPoint = isLastS2RangeTask && s2EndPoint == 0 ? totalLen : s2EndPoint;
    int64_t rangeStart = isFirstS2RangeTask ? s2StartPoint : 0;
    rangeStart = rangeStart < 0 ? 0 : rangeStart;
    rangeStart = rangeStart < totalLen ? rangeStart : totalLen;
    int64_t rangeEnd = isLastS2RangeTask ? effectiveS2EndPoint : totalLen;
    rangeEnd = rangeEnd < 0 ? 0 : rangeEnd;
    rangeEnd = rangeEnd < totalLen ? rangeEnd : totalLen;
    if (rangeEnd <= rangeStart) {
        qs35RunParam.oriKvLoopEndIdx = 0;
        qs35RunParam.cmpKvLoopEndIdx = 0;
        qs35RunParam.s2LoopEndIdx = 0;
        qs35RunParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    qs35RunParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    qs35RunParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < oriLen ? rangeStart : oriLen;
    int64_t oriRangeEnd = rangeEnd < oriLen ? rangeEnd : oriLen;
    qs35RunParam.s2LineStartIdx = oriStart + oriRangeStart;
    qs35RunParam.s2LineOriEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > oriLen ? rangeStart - oriLen : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > oriLen ? rangeEnd - oriLen : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    qs35RunParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    qs35RunParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriRangeLen = qs35RunParam.s2LineOriEndIdx - qs35RunParam.s2LineStartIdx;
    int64_t cmpRangeLen = qs35RunParam.s2CmpLineEndIdx - qs35RunParam.s2CmpLineStartIdx;
    qs35RunParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    qs35RunParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    qs35RunParam.s2LoopEndIdx = qs35RunParam.oriKvLoopEndIdx + qs35RunParam.cmpKvLoopEndIdx;
    return qs35RunParam.s2LoopEndIdx == 0;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ParseFdRunInfo(FdRunInfo &fdRunInfo)
{
    uint32_t aivIdx = static_cast<uint32_t>(this->constInfo.aivIdx);
    fdRunInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_CORE_ENABLE_INDEX, true)) != 0;
    if (!fdRunInfo.coreEnable) {
        return;
    }
    fdRunInfo.bn2Idx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_BN2_IDX_INDEX, true));
    fdRunInfo.mIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_IDX_INDEX, true));
    fdRunInfo.workspaceIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_IDX_INDEX, true));
    fdRunInfo.workspaceNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_NUM_INDEX, true));
    fdRunInfo.mStartIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_START_INDEX, true));
    fdRunInfo.mNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_NUM_INDEX, true));
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &qs35RunParam)
{
    // GS1合轴, 不切G, 只切S1
    qs35RunParam.s1oIdx = gS1Index * qs35RunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t qsmlaHalfG = (constInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        qs35RunParam.goIdx = (aicIdx % 2 == 0) ? 0 : qsmlaHalfG;
        qs35RunParam.gSplitSize = (aicIdx % 2 == 0) ? qsmlaHalfG : (constInfo.gSize - qsmlaHalfG);
    } else {
        qs35RunParam.goIdx = 0;
        qs35RunParam.gSplitSize = constInfo.gSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &qs35RunInfo, RunParamStr &qs35RunParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < qs35RunParam.oriKvLoopEndIdx) {
        qs35RunInfo.s2StartIdx = qs35RunParam.s2LineStartIdx;
        qs35RunInfo.s2EndIdx = qs35RunParam.s2LineOriEndIdx;
    } else {
        qs35RunInfo.s2StartIdx = qs35RunParam.s2CmpLineStartIdx;
        qs35RunInfo.s2EndIdx = qs35RunParam.s2CmpLineEndIdx;
    }
    qs35RunInfo.s2LoopCount = s2LoopCount;
    if (qs35RunInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        qs35RunInfo.s1oIdx = qs35RunParam.s1oIdx;
        qs35RunInfo.boIdx = qs35RunParam.boIdx;
        qs35RunInfo.n2oIdx = qs35RunParam.n2oIdx;
        qs35RunInfo.goIdx = qs35RunParam.goIdx;
        qs35RunInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        qs35RunInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        qs35RunInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    qs35RunInfo.taskId = taskId;
    qs35RunInfo.taskIdMod2 = taskId & 1;
    qs35RunInfo.taskIdMod3 = taskId % 3;
    qs35RunInfo.s2LoopLimit = s2LoopLimit;

    qs35RunInfo.actualS1Size = qs35RunParam.actualS1Size;
    qs35RunInfo.attentionOutOffset = qs35RunParam.attentionOutOffset;
    qs35RunInfo.sOuterOffset = qs35RunParam.sOuterOffset;
    qs35RunInfo.firstFdDataWorkspaceIdx = qs35RunParam.firstFdDataWorkspaceIdx;
    qs35RunInfo.isCrossCoreSplit = qs35RunParam.isCrossCoreSplit;
    qs35RunInfo.s2SplitIdx = qs35RunParam.s2SplitIdx;
    qs35RunInfo.isFirstS2SplitCore = qs35RunParam.isFirstS2SplitCore;
    int64_t qs35ReductionLoop = s2LoopCount;
    if constexpr (IS_BATCH_CONSISTENCY) {
        // 进入 CMP 时补齐规约计数，不增加实际计算。
        if (s2LoopCount >= qs35RunParam.oriKvLoopEndIdx) {
            qs35ReductionLoop += (qs35RunParam.baseBlockNumPerReductionBlock -
                                  qs35RunParam.oriKvLoopEndIdx % qs35RunParam.baseBlockNumPerReductionBlock) %
                                 qs35RunParam.baseBlockNumPerReductionBlock;
        }
    }
    int64_t baseBlockIdInReduceBlock = qs35ReductionLoop % qs35RunParam.baseBlockNumPerReductionBlock;
    qs35RunInfo.reduceBlockId = qs35ReductionLoop / qs35RunParam.baseBlockNumPerReductionBlock;
    qs35RunInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    qs35RunInfo.isLastBase =
        qs35RunParam.baseBlockNumPerReductionBlock - baseBlockIdInReduceBlock == 1LL || s2LoopCount == s2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        qs35RunInfo.isLastBase = qs35RunInfo.isLastBase || s2LoopCount + 1 == qs35RunParam.oriKvLoopEndIdx;
        qs35RunInfo.isFirstReduce = baseBlockIdInReduceBlock == 1;
    } else {
        qs35RunInfo.isFirstReduce = qs35RunInfo.s2LoopCount == 1;
    }
    qs35RunInfo.needReduce = qs35RunInfo.reduceBlockId > 0;
    this->ComputeBmm1Tail(qs35RunInfo, qs35RunParam);
    InitUniqueRunInfo(qs35RunParam, qs35RunInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &qs35RunParam, RunInfo &qs35RunInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(qs35RunParam, qs35RunInfo);
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
__aicore__ inline void QuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo &qs35RunInfo,
                                                                                            RunParamStr &qs35RunParam)
{
    // ------------------------S1 Base Related---------------------------
    qs35RunInfo.s1RealSize = qs35RunParam.s1RealSize;
    qs35RunInfo.halfS1RealSize = qs35RunParam.halfS1RealSize;
    qs35RunInfo.firstHalfS1RealSize = qs35RunParam.firstHalfS1RealSize;
    qs35RunInfo.mRealSize = qs35RunParam.mRealSize;
    qs35RunInfo.halfMRealSize = qs35RunParam.halfMRealSize;
    qs35RunInfo.firstHalfMRealSize = qs35RunParam.firstHalfMRealSize;

    qs35RunInfo.vec2S1BaseSize = qs35RunInfo.halfS1RealSize; // D>128 这里需要适配
    qs35RunInfo.vec2MBaseSize = qs35RunInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    qs35RunInfo.s2RealSize = constInfo.s2BaseSize;
    qs35RunInfo.s2AlignedSize = qs35RunInfo.s2RealSize;
    int64_t qsmlaCurS2LoopCnt = (qs35RunInfo.s2LoopCount >= qs35RunParam.oriKvLoopEndIdx) ?
                                    (qs35RunInfo.s2LoopCount - qs35RunParam.oriKvLoopEndIdx) :
                                    qs35RunInfo.s2LoopCount;
    if (qs35RunInfo.s2StartIdx + (qsmlaCurS2LoopCnt + 1) * qs35RunInfo.s2RealSize > qs35RunInfo.s2EndIdx) {
        qs35RunInfo.s2RealSize =
            qs35RunInfo.s2EndIdx - qsmlaCurS2LoopCnt * qs35RunInfo.s2RealSize - qs35RunInfo.s2StartIdx;
        qs35RunInfo.s2AlignedSize = Align(qs35RunInfo.s2RealSize);
    }
}
} // namespace BaseApi
#endif // QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
