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
 * \file mixed_quant_sparse_flash_mla_csa_kernel.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
#include "mixed_quant_sparse_flash_mla_tiling_data_arch35.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
#include "mixed_quant_sparse_flash_mla_kvcache.h"
#include "mixed_quant_sparse_flash_mla_csa_block_cube.h"
#include "mixed_quant_sparse_flash_mla_csa_block_vector.h"
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../mixed_quant_sparse_flash_mla_metadata.h"
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
template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
class MixedQuantSparseFlashMlaCsa {
public:
    using CubeBlockType = MqsmlaCsaCubeBlockType;
    using VecBlockType = MqsmlaCsaVecBlockType;
    MQSMLA_ARGS_TRAITS;
    __aicore__ inline MixedQuantSparseFlashMlaCsa(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv,
                                __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
                                __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace, const MixedQuantSparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMqsmlaMainLoop();
    __aicore__ inline int64_t GetMqsmlaSeqLen(int32_t batchIndex, bool preferActualLength, bool deriveFromCuSeqlens,
                                              GlobalTensor<int32_t> &actualLengthGm, GlobalTensor<int32_t> &cuLengthGm,
                                              int64_t fallbackLength);
    __aicore__ inline void ParseMqsmlaTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                 __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv,
                                                 __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedCmpKv,
                                                 __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitMqsmlaGlobalBuffer(
        __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
        __gm__ uint8_t *cmpResidualKv, __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
        __gm__ uint8_t *sinks, __gm__ uint8_t *workspace, const MixedQuantSparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void InitMqsmlaLocalBuffer();
    __aicore__ inline void InitMqsmlaMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeMqsmlaConstexpr();
    __aicore__ inline void SetMqsmlaRunInfo(RunInfo<HIGH_PERF> &mq35RunInfo, RunParamStr<HIGH_PERF> &mq35RunParam,
                                            int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
                                            int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeMqsmlaBmm1Tail(RunInfo<HIGH_PERF> &mq35RunInfo, RunParamStr<HIGH_PERF> &mq35RunParam);
    __aicore__ inline void InitMqsmlaUniqueConstInfo();
    __aicore__ inline void FreeMqsmlaEvent();
    __aicore__ inline void ComputeMqsmlaAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index,
                                                          RunParamStr<HIGH_PERF> &mq35RunParam);
    __aicore__ inline void InitMqsmlaUniqueRunInfo(const RunParamStr<HIGH_PERF> &mq35RunParam,
                                                   RunInfo<HIGH_PERF> &mq35RunInfo);
    __aicore__ inline void ParseMqsmlaFdRunInfo(FdRunInfo &fdRunInfo);
    __aicore__ inline int64_t ConvertMqsmlaS2MetadataBlockToToken(const RunParamStr<HIGH_PERF> &mq35RunParam,
                                                                  const ConstInfo<HIGH_PERF> &mqsmlaCsaConstInfo,
                                                                  uint32_t s2BlockIdx);
    __aicore__ inline bool ApplyMqsmlaS2MetadataRange(RunParamStr<HIGH_PERF> &mq35RunParam,
                                                      ConstInfo<HIGH_PERF> &mqsmlaCsaConstInfo, int64_t s2StartPoint,
                                                      int64_t s2EndPoint, bool isFirstS2RangeTask,
                                                      bool isLastS2RangeTask);

    const MixedQuantSparseFlashMlaTilingData *__restrict mqsmlaCsaTilingData;
    static constexpr uint32_t PRELOAD_NUM = 3;
    /* 核间通道 */
    BufferManager<BufferType::GM> mqsmlaCsaV0ResGmBufferManager;

    StaticBuffer<T> mqsmlaCsaBmm1Buffers[2];
    StaticBuffer<T> mqsmlaCsaBmm2Buffers;
    uint32_t mqsmlaCsaBmm1GetFlag = 0;
    uint32_t mqsmlaCsaVUbBase = 0;

    // mm2左矩阵P
    StaticBuffer<Q_T> mqsmlaCsaL1PBuffers[2];
    uint32_t mqsmlaCsaL1PGetFlag = 0;
    uint32_t mqsmlaCsaL1CubeBase = 0;
    /* GM信息 */
    GlobalTensor<uint32_t> mqsmlaCsaMetadataGm;
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
    BufferManager<BufferType::GM> fdStagingBufferManager;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> fdStagingBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> intraCoreCombineBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> crossCoreCombineBuffer;
    /* 核Index信息 */
    int32_t mqsmlaCsaAicIdx;
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t s2StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t s2EndIdx;
    uint32_t hasLoad;

    /* 初始化后不变的信息 */
    ConstInfo<HIGH_PERF> mqsmlaCsaConstInfo;

    /* 模板库Block */
    MqsmlaCsaCubeBlockType mqsmlaCsaCubeBlock;
    MqsmlaCsaVecBlockType mqsmlaCsaVecBlock;
};

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const MixedQuantSparseFlashMlaTilingData *__restrict tiling)
{
    fa_base_matmul::ResetIdCounter();
    mqsmlaCsaConstInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->mqsmlaCsaAicIdx = GetBlockIdx();
        mqsmlaCsaConstInfo.aivIdx = 0;
        this->mqsmlaCsaTilingData = tiling;
    } else {
        mqsmlaCsaConstInfo.aivIdx = GetBlockIdx();
        this->mqsmlaCsaAicIdx = mqsmlaCsaConstInfo.aivIdx >> 1;
        this->mqsmlaCsaTilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->mqsmlaCsaMetadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    bN2StartIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_BN2_START_INDEX, false));
    gS1StartIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_M_START_INDEX, false));
    s2StartIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_S2_START_INDEX, false));
    bN2EndIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_BN2_END_INDEX, false));
    nextGs1Idx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_M_END_INDEX, false));
    s2EndIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_S2_END_INDEX, false));
    hasLoad = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_CORE_ENABLE_INDEX, false));
    if (nextGs1Idx != 0 || s2EndIdx != 0) {
        bN2EndIdx++;
    }

    mqsmlaCsaConstInfo.s1BaseSize = 64;
    mqsmlaCsaConstInfo.s2BaseSize = 128;

    this->ParseMqsmlaTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, sequsedOriKv, cuSeqlensCmpKv, sequsedCmpKv,
                                cmpResidualKv);
    this->InitMqsmlaGlobalBuffer(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                                 cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv,
                                 cmpResidualKv, oriTopkLength, cmpTopkLength, sinks, workspace, tiling);
    mqsmlaCsaVecBlock.InitVecBlock(cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedOriKv, sequsedCmpKv,
                                   cmpResidualKv);
    mqsmlaCsaVecBlock.CleanOutput(attentionOut, softmaxLse, mqsmlaCsaConstInfo);
    if ASCEND_IS_AIV {
        if constexpr ((TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                       TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                       TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                      IS_VEC_S2PHYADDR) {
            this->mqsmlaCsaVecBlock.GetKVPhyAddr(
                hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen, hasCuSeqlensQ,
                hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
                hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm,
                cmpResidualKvGm, actualSeqQlenGm, cuSeqlensQGm, workspace, mqsmlaCsaConstInfo);
        }
    }
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMqsmlaMMResBuf(workspace);
    if constexpr (IS_BATCH_CONSISTENCY) {
        mqsmlaCsaVecBlock.InitS2SplitStaging(intraCoreCombineBuffer.Get(), crossCoreCombineBuffer.Get());
    } else {
        mqsmlaCsaVecBlock.InitS2SplitStaging(fdStagingBuffer.Get());
    }
    this->ComputeMqsmlaConstexpr();
    this->InitMqsmlaLocalBuffer();
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline int64_t MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::GetMqsmlaSeqLen(
    int32_t batchIndex, bool preferActualLength, bool deriveFromCuSeqlens, GlobalTensor<int32_t> &actualLengthGm,
    GlobalTensor<int32_t> &cuLengthGm, int64_t fallbackLength)
{
    if (preferActualLength) {
        return actualLengthGm.GetValue(batchIndex);
    } else if (deriveFromCuSeqlens) {
        return cuLengthGm.GetValue(batchIndex + 1) - cuLengthGm.GetValue(batchIndex);
    } else {
        return fallbackLength;
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ParseMqsmlaTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *sequsedOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    auto &mixedQuantSparseFlashMlaBaseParams = this->mqsmlaCsaTilingData->baseParams;
    mqsmlaCsaConstInfo.bSize = mixedQuantSparseFlashMlaBaseParams.batchSize;
    mqsmlaCsaConstInfo.n2Size = 1;
    mqsmlaCsaConstInfo.gSize = mixedQuantSparseFlashMlaBaseParams.nNumOfQInOneGroup;
    mqsmlaCsaConstInfo.s1Size = mixedQuantSparseFlashMlaBaseParams.qSeqSize;
    mqsmlaCsaConstInfo.s2Size = mixedQuantSparseFlashMlaBaseParams.kvSeqSize;
    mqsmlaCsaConstInfo.cmpS2Size = mixedQuantSparseFlashMlaBaseParams.cmpKvSeqSize;
    mqsmlaCsaConstInfo.oriSparseBlockCount = mixedQuantSparseFlashMlaBaseParams.oriSparseBlockCount;
    mqsmlaCsaConstInfo.cmpSparseBlockCount = mixedQuantSparseFlashMlaBaseParams.cmpSparseBlockCount;
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    mqsmlaCsaConstInfo.alignedOriSparseBlockCount =
        (mqsmlaCsaConstInfo.oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM *
        SPARSE_BLOCK_ALIGN_NUM;
    mqsmlaCsaConstInfo.alignedCmpSparseBlockCount =
        (mqsmlaCsaConstInfo.cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM *
        SPARSE_BLOCK_ALIGN_NUM;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqsmlaCsaConstInfo.cmpRatio = mixedQuantSparseFlashMlaBaseParams.cmpRatio;
    }
    mqsmlaCsaConstInfo.oriMaskMode = mixedQuantSparseFlashMlaBaseParams.oriMaskMode;
    mqsmlaCsaConstInfo.cmpMaskMode = mixedQuantSparseFlashMlaBaseParams.cmpMaskMode;
    mqsmlaCsaConstInfo.oriWinLeft = mixedQuantSparseFlashMlaBaseParams.oriWinLeft;
    mqsmlaCsaConstInfo.oriWinRight = mixedQuantSparseFlashMlaBaseParams.oriWinRight;
    mqsmlaCsaConstInfo.dSizeRope = mixedQuantSparseFlashMlaBaseParams.ropeHeadDim;
    mqsmlaCsaConstInfo.softmaxScale = mixedQuantSparseFlashMlaBaseParams.softmaxScale;
    mqsmlaCsaConstInfo.oriKvStride = mixedQuantSparseFlashMlaBaseParams.oriKvStride;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqsmlaCsaConstInfo.cmpKvStride = mixedQuantSparseFlashMlaBaseParams.cmpKvStride;
    }
    mqsmlaCsaConstInfo.dSize = mixedQuantSparseFlashMlaBaseParams.dSize;
    mqsmlaCsaConstInfo.dSizeV = mqsmlaCsaConstInfo.dSize;
    mqsmlaCsaConstInfo.dSizeVInput = mixedQuantSparseFlashMlaBaseParams.dSizeVInput;
    mqsmlaCsaConstInfo.dSizeNope = mqsmlaCsaConstInfo.dSize - mqsmlaCsaConstInfo.dSizeRope;
    if constexpr (!HIGH_PERF) {
        mqsmlaCsaConstInfo.isSoftmaxLseEnable = mixedQuantSparseFlashMlaBaseParams.returnSoftmaxLse;
    }
    mqsmlaCsaConstInfo.sparseBlockSize = 1;
    mqsmlaCsaConstInfo.actualSeqLenSize = mqsmlaCsaConstInfo.bSize + 1;

    if constexpr (isPa) {
        mqsmlaCsaConstInfo.oriBlockSize = mixedQuantSparseFlashMlaBaseParams.paOriBlockSize;
        mqsmlaCsaConstInfo.oriMaxBlockNumPerBatch = mixedQuantSparseFlashMlaBaseParams.oriMaxBlockNumPerBatch;
        if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            mqsmlaCsaConstInfo.cmpBlockSize = mixedQuantSparseFlashMlaBaseParams.paCmpBlockSize;
            mqsmlaCsaConstInfo.cmpMaxBlockNumPerBatch = mixedQuantSparseFlashMlaBaseParams.cmpMaxBlockNumPerBatch;
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

    mqsmlaCsaConstInfo.needInit = 0;
    if (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && mqsmlaCsaConstInfo.oriMaskMode != 0) {
        for (uint32_t mqsmlaBIdx = 0; mqsmlaBIdx < mqsmlaCsaConstInfo.bSize; mqsmlaBIdx++) {
            int64_t s2Size = GetMqsmlaSeqLen(mqsmlaBIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm,
                                             cuSeqlensOriKvGm, mqsmlaCsaConstInfo.s2Size);
            int64_t s1Size = GetMqsmlaSeqLen(mqsmlaBIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                             mqsmlaCsaConstInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                expectQs = GetMqsmlaSeqLen(mqsmlaBIdx, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm,
                                           mqsmlaCsaConstInfo.s1Size);
            } else {
                expectQs = mqsmlaCsaConstInfo.s1Size;
            }
            if (s1Size > s2Size || s1Size < expectQs) {
                mqsmlaCsaConstInfo.needInit = 1;
                break;
            }
        }
    } else {
        mqsmlaCsaConstInfo.needInit = 1;
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::InitMqsmlaGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const MixedQuantSparseFlashMlaTilingData *__restrict tiling)
{
    mqsmlaCsaVecBlock.InitGlobalBuffer(oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                                       sequsedQ, sinks, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    mqsmlaCsaCubeBlock.InitGlobalBuffer(query, cuSeqlensQ, sequsedQ, mqsmlaCsaConstInfo);
    if constexpr (!HIGH_PERF) {
        if (oriTopkLength != nullptr) {
            mqsmlaCsaConstInfo.hasOriTopkLength = true;
            oriTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)oriTopkLength);
        } else {
            mqsmlaCsaConstInfo.hasOriTopkLength = false;
        }
        if (cmpTopkLength != nullptr) {
            mqsmlaCsaConstInfo.hasCmpTopkLength = true;
            cmpTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)cmpTopkLength);
        } else {
            mqsmlaCsaConstInfo.hasCmpTopkLength = false;
        }
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::InitMqsmlaMMResBuf(
    __gm__ uint8_t *workspace)
{
    // L1: [l1P x2][cube L1], l1P 必须放在最前面保证与 vec 申请地址相同
    uint32_t mm2LeftSize = mqsmlaCsaConstInfo.s1BaseSize * mqsmlaCsaConstInfo.s2BaseSize;
    uint32_t l1PAddr = 0;
    mqsmlaCsaL1PBuffers[0] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize), 0};
    l1PAddr += (mm2LeftSize * sizeof(Q_T));
    mqsmlaCsaL1PBuffers[1] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize), 1};
    l1PAddr += (mm2LeftSize * sizeof(Q_T));
    mqsmlaCsaL1CubeBase = l1PAddr;

    // UB: [bmm2][bmm1 x2][vec UB]
    uint32_t mm1ResultSize = mqsmlaCsaConstInfo.s1BaseSize / CV_RATIO * mqsmlaCsaConstInfo.s2BaseSize;
    uint32_t mm2ResultSize = mqsmlaCsaConstInfo.s1BaseSize / CV_RATIO * 512;
    uint32_t ubAddr = 0;
    mqsmlaCsaBmm2Buffers = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm2ResultSize), 0};
    ubAddr += (mm2ResultSize * sizeof(T));
    mqsmlaCsaBmm1Buffers[0] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize), 0};
    ubAddr += (mm1ResultSize * sizeof(T));
    mqsmlaCsaBmm1Buffers[1] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize), 1};
    ubAddr += (mm1ResultSize * sizeof(T));
    mqsmlaCsaVUbBase = ubAddr;

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[0].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[1].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    }

    uint32_t v0ResSize = mqsmlaCsaConstInfo.s2BaseSize * 512U * sizeof(Q_T);
    uint64_t totalOffset = IS_SPLIT_G ? static_cast<uint64_t>(v0ResSize) * 3 * (mqsmlaCsaAicIdx >> 1U) :
                                        static_cast<uint64_t>(v0ResSize) * 3 * mqsmlaCsaAicIdx;
    mqsmlaCsaV0ResGmBufferManager.Init(workspace + totalOffset);
    v0ResGmBuffers.Init(mqsmlaCsaV0ResGmBufferManager, v0ResSize);
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(0));
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(1));
    v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(2));

    uint64_t phyAddrRegionSize = 0;
    if constexpr (IS_VEC_S2PHYADDR) {
        uint64_t totalBS1 = (LAYOUT_T == QSMLA_LAYOUT::TND) ?
                                mqsmlaCsaConstInfo.s1Size :
                                static_cast<uint64_t>(mqsmlaCsaConstInfo.bSize) * mqsmlaCsaConstInfo.s1Size;
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            phyAddrRegionSize += totalBS1 * mqsmlaCsaConstInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        }
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            phyAddrRegionSize += totalBS1 * mqsmlaCsaConstInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
        }
    }
    uint64_t v0RegionSize = static_cast<uint64_t>(v0ResSize) * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
    fdStagingBufferManager.Init(workspace + v0RegionSize + phyAddrRegionSize);
    constexpr uint32_t FD_MAX_SUM_REGION_NUM = 2U;
    uint32_t gSize = static_cast<uint32_t>(mqsmlaCsaConstInfo.gSize);
    if constexpr (IS_BATCH_CONSISTENCY) {
        uint32_t combineElemSize =
            gSize * mqsmlaCsaConstInfo.dSize +
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
            fdSlotCount * (gSize * mqsmlaCsaConstInfo.dSize * sizeof(float) +
                           FD_MAX_SUM_REGION_NUM * gSize *
                               static_cast<uint32_t>(AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW) * sizeof(float));
        fdStagingBuffer.Init(fdStagingBufferManager, fdStagingSize);
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::InitMqsmlaLocalBuffer()
{
    mqsmlaCsaVecBlock.InitLocalBuffer(mqsmlaCsaConstInfo, mqsmlaCsaVUbBase);
    mqsmlaCsaCubeBlock.InitLocalBuffer(mqsmlaCsaL1CubeBase);
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ComputeMqsmlaConstexpr()
{
    mqsmlaCsaConstInfo.s1S2 = mqsmlaCsaConstInfo.s1Size * mqsmlaCsaConstInfo.s2Size;
    mqsmlaCsaConstInfo.gS1 = mqsmlaCsaConstInfo.gSize * mqsmlaCsaConstInfo.s1Size;
    mqsmlaCsaConstInfo.n2G = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.gSize;

    mqsmlaCsaConstInfo.s1Dv = mqsmlaCsaConstInfo.s1Size * mqsmlaCsaConstInfo.dSizeV;
    mqsmlaCsaConstInfo.s2Dv = mqsmlaCsaConstInfo.s2Size * mqsmlaCsaConstInfo.dSizeV;
    mqsmlaCsaConstInfo.n2Dv = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.dSizeV;
    mqsmlaCsaConstInfo.gDv = mqsmlaCsaConstInfo.gSize * mqsmlaCsaConstInfo.dSizeV;
    mqsmlaCsaConstInfo.gS1Dv = mqsmlaCsaConstInfo.gSize * mqsmlaCsaConstInfo.s1Dv;
    mqsmlaCsaConstInfo.n2S2Dv = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.s2Dv;
    mqsmlaCsaConstInfo.n2GDv = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.gDv;
    mqsmlaCsaConstInfo.s2BaseN2Dv = mqsmlaCsaConstInfo.s2BaseSize * mqsmlaCsaConstInfo.n2Dv;
    mqsmlaCsaConstInfo.n2GS1Dv = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.gS1Dv;

    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        // (BS)ND
        mqsmlaCsaConstInfo.s1BaseN2GDv = mqsmlaCsaConstInfo.s1BaseSize * mqsmlaCsaConstInfo.n2GDv;

        mqsmlaCsaConstInfo.mm1Ka = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.dSize;
        if ASCEND_IS_AIV {
            mqsmlaCsaConstInfo.attentionOutStride =
                (mqsmlaCsaConstInfo.n2G - mqsmlaCsaConstInfo.gSize) * mqsmlaCsaConstInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        mqsmlaCsaConstInfo.s1BaseN2GDv = mqsmlaCsaConstInfo.s1BaseSize * mqsmlaCsaConstInfo.n2GDv;
        mqsmlaCsaConstInfo.mm1Ka = mqsmlaCsaConstInfo.n2Size * mqsmlaCsaConstInfo.dSize;
        if ASCEND_IS_AIV {
            mqsmlaCsaConstInfo.attentionOutStride =
                (mqsmlaCsaConstInfo.n2G - mqsmlaCsaConstInfo.gSize) * mqsmlaCsaConstInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::Process()
{
    if constexpr (IS_VEC_S2PHYADDR) {
        SyncAll<false>();
    } else {
        if (this->mqsmlaCsaConstInfo.needInit) {
            SyncAll<false>();
        }
    }
    if ASCEND_IS_AIV {
        mqsmlaCsaVecBlock.InitSinks(this->mqsmlaCsaConstInfo);
    }
    FdRunInfo fdRunInfo;
    if ASCEND_IS_AIV {
        ParseMqsmlaFdRunInfo(fdRunInfo);
    }
    ICachePreLoad(6);
    ProcessMqsmlaMainLoop();
    SyncAll();
    if ASCEND_IS_AIV {
        if (fdRunInfo.coreEnable > 0) {
            this->mqsmlaCsaVecBlock.ProcessFlashDecode(fdRunInfo, this->mqsmlaCsaConstInfo);
        }
    }
    FreeMqsmlaEvent();
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ProcessMqsmlaMainLoop()
{
    int64_t mqsmlaMaxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        mqsmlaMaxS2LoopCnt =
            static_cast<int64_t>(mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < mqsmlaMaxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(15);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(15);
                }
            }
        }
        return;
    }

    // 从meta data解析分核信息
    uint32_t firstFdDataWorkspaceIdx =
        mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(mqsmlaCsaAicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, false));
    uint32_t s2LoopLimit = 0;
    int64_t taskId = 0;
    bool isFirstLoop = true;
    bool mqsmlaNotLast = true;
    bool mqsmlaNotLastTwoLoop = true;
    RunInfo<HIGH_PERF> mq35RunInfo[4];
    RunParamStr<HIGH_PERF> mq35RunParam;
    mq35RunParam.firstFdDataWorkspaceIdx = firstFdDataWorkspaceIdx;
    int64_t mqsmlaMultiCoreInnerIdx = 1;
    int64_t s2SplitIdxCounter = 0;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        mq35RunParam.boIdx = bnIdx;
        mq35RunParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(
            mq35RunParam, this->mqsmlaCsaConstInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm, this->cuSeqlensCmpKvGm,
            this->actualSeqQlenGm, this->actualSeqOriKvlenGm, this->actualSeqCmpKvlenGm, this->cmpResidualKvGm,
            this->hasCuSeqlensOriKv, this->hasCuSeqlensCmpKv, this->hasActualSeqQlen, this->hasActualSeqOriKvlen,
            this->hasActualSeqCmpKvlen);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(mq35RunParam, this->mqsmlaCsaConstInfo, lastBN, nextGs1Idx, gS1StartIdx,
                                              s2EndIdx);

        int64_t mqsmlaGS1LoopEnd = lastBN ? (mq35RunParam.gs1LoopEndIdx + PRELOAD_NUM) : mq35RunParam.gs1LoopEndIdx;
        for (int64_t gS1Index = mq35RunParam.gs1LoopStartIdx; gS1Index < mqsmlaGS1LoopEnd; gS1Index++) {
            bool mqsmlaNotLastThreeLoop = true;
            if (lastBN) {
                int32_t mqsmlaExtraGS1 = gS1Index - mq35RunParam.gs1LoopEndIdx;
                switch (mqsmlaExtraGS1) {
                    case 0:
                        mqsmlaNotLastThreeLoop = false;
                        break;
                    case 1:
                        mqsmlaNotLastTwoLoop = false;
                        mqsmlaNotLastThreeLoop = false;
                        break;
                    case 2:
                        mqsmlaNotLast = false;
                        mqsmlaNotLastTwoLoop = false;
                        mqsmlaNotLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (mqsmlaNotLastThreeLoop) {
                this->ComputeMqsmlaAxisIdxByBnAndGs1(bnIdx, gS1Index, mq35RunParam);
                bool mqsmlaS1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(mq35RunParam, this->mqsmlaCsaConstInfo,
                                                                             gS1Index, this->cuSeqlensQGm);
                bool mqsmlaS2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(bnIdx, gS1Index, this->cuSeqlensQGm, oriTopkLengthGm,
                                                          cmpTopkLengthGm, mq35RunParam, this->mqsmlaCsaConstInfo);
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t s2Load =
                        mq35RunParam.s2LineOriEndIdx - mq35RunParam.s2LineStartIdx + mq35RunParam.s2CmpLineEndIdx;
                    int64_t s2BaseSize = static_cast<int64_t>(mqsmlaCsaConstInfo.s2BaseSize);
                    int64_t s2PerReduceBlock = ((s2Load + 31LL) / 32LL + s2BaseSize - 1) >> 7 << 7;
                    int64_t baseBlockNum = s2PerReduceBlock >> 7;
                    mq35RunParam.baseBlockNumPerReductionBlock = baseBlockNum > 0 ? baseBlockNum : 1LL;
                }
                if (!mqsmlaS2NoNeedCalc) {
                    bool isFirstS2RangeTask = (bnIdx == bN2StartIdx && gS1Index == mq35RunParam.gs1LoopStartIdx);
                    bool isLastS2RangeTask = (lastBN && gS1Index == mq35RunParam.gs1LoopEndIdx - 1);
                    int64_t s2StartPoint =
                        ConvertMqsmlaS2MetadataBlockToToken(mq35RunParam, this->mqsmlaCsaConstInfo, s2StartIdx);
                    int64_t s2EndPoint =
                        (isLastS2RangeTask && s2EndIdx == 0) ?
                            0 :
                            ConvertMqsmlaS2MetadataBlockToToken(mq35RunParam, this->mqsmlaCsaConstInfo, s2EndIdx);
                    mqsmlaS2NoNeedCalc =
                        ApplyMqsmlaS2MetadataRange(mq35RunParam, this->mqsmlaCsaConstInfo, s2StartPoint, s2EndPoint,
                                                   isFirstS2RangeTask, isLastS2RangeTask);
                } else {
                    mq35RunParam.isCrossCoreSplit = false;
                }
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (mqsmlaS1NoNeedCalc || mqsmlaS2NoNeedCalc) {
                    continue;
                }
                if constexpr (!IS_BATCH_CONSISTENCY) {
                    if (mq35RunParam.isCrossCoreSplit) {
                        mq35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if constexpr (IS_SPLIT_G) {
                    mqsmlaMaxS2LoopCnt -= mq35RunParam.s2LoopEndIdx;
                }
                s2LoopLimit = mq35RunParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t mq35ReductionBlockSize = mq35RunParam.baseBlockNumPerReductionBlock > 0 ?
                                                         mq35RunParam.baseBlockNumPerReductionBlock :
                                                         1LL;
                    int64_t mq35ReductionLoop = s2LoopCount;
                    if (s2LoopCount >= mq35RunParam.oriKvLoopEndIdx) {
                        mq35ReductionLoop +=
                            (mq35ReductionBlockSize - mq35RunParam.oriKvLoopEndIdx % mq35ReductionBlockSize) %
                            mq35ReductionBlockSize;
                    }
                    if (mq35RunParam.isCrossCoreSplit && mq35ReductionLoop % mq35ReductionBlockSize == 0) {
                        mq35RunParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if (mqsmlaNotLastThreeLoop) {
                    RunInfo<HIGH_PERF> &runInfo1 = mq35RunInfo[taskId % 4];
                    this->SetMqsmlaRunInfo(runInfo1, mq35RunParam, taskId, s2LoopCount, s2LoopLimit,
                                           mqsmlaMultiCoreInnerIdx);
                }
                if ASCEND_IS_AIV {
                    if (mqsmlaNotLastThreeLoop) {
                        RunInfo<HIGH_PERF> &runInfo1 = mq35RunInfo[taskId % 4];
                        this->mqsmlaCsaVecBlock.ProcessVec0(v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                            this->mqsmlaCsaConstInfo);
                    }
                    if (taskId > 1 && mqsmlaNotLast) {
                        uint32_t bmm1Slot = mqsmlaCsaBmm1GetFlag;
                        mqsmlaCsaBmm1GetFlag ^= 1;
                        uint32_t l1PSlot = mqsmlaCsaL1PGetFlag;
                        mqsmlaCsaL1PGetFlag ^= 1;
                        auto &runInfo2 = mq35RunInfo[(taskId + 2) % 4];
                        this->mqsmlaCsaVecBlock.ProcessVec1(this->mqsmlaCsaL1PBuffers[l1PSlot],
                                                            this->mqsmlaCsaBmm1Buffers[bmm1Slot], runInfo2,
                                                            this->mqsmlaCsaConstInfo);
                    }
                    if (taskId > 2) {
                        RunInfo<HIGH_PERF> &runInfo3 = mq35RunInfo[(taskId + 1) % 4];
                        this->mqsmlaCsaVecBlock.ProcessVec2(this->mqsmlaCsaBmm2Buffers, runInfo3,
                                                            this->mqsmlaCsaConstInfo);
                    }
                } else {
                    if (taskId > 0 && mqsmlaNotLastTwoLoop) {
                        RunInfo<HIGH_PERF> &runInfo1 = mq35RunInfo[(taskId + 3) % 4];
                        this->mqsmlaCsaCubeBlock.IterateLoadQK(v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                               this->mqsmlaCsaConstInfo, isFirstLoop);
                        isFirstLoop = false;
                    } else {
                        if constexpr (IS_SPLIT_G) {
                            if (taskId > 0 && mqsmlaMaxS2LoopCnt > 0) {
                                mqsmlaMaxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
                            }
                        }
                    }
                    if (taskId > 1 && mqsmlaNotLast) {
                        uint32_t bmm1Slot = mqsmlaCsaBmm1GetFlag;
                        mqsmlaCsaBmm1GetFlag ^= 1;
                        RunInfo<HIGH_PERF> &runInfo2 = mq35RunInfo[(taskId + 2) % 4];
                        RunInfo<HIGH_PERF> &runInfoNext = mq35RunInfo[(taskId + 3) % 4];
                        this->mqsmlaCsaCubeBlock.IterateBmm1(
                            this->mqsmlaCsaBmm1Buffers[bmm1Slot], v0ResGmBuffers.Get(runInfo2.taskIdMod3),
                            mqsmlaNotLastTwoLoop, runInfoNext, runInfo2, this->mqsmlaCsaConstInfo);
                    }
                    if (taskId > 2) {
                        uint32_t l1PSlot = mqsmlaCsaL1PGetFlag;
                        mqsmlaCsaL1PGetFlag ^= 1;
                        RunInfo<HIGH_PERF> &runInfo3 = mq35RunInfo[(taskId + 1) % 4];
                        this->mqsmlaCsaCubeBlock.IterateBmm2(this->mqsmlaCsaBmm2Buffers,
                                                             this->mqsmlaCsaL1PBuffers[l1PSlot], runInfo3,
                                                             this->mqsmlaCsaConstInfo);
                    }
                }
                ++taskId;
            }
            ++mqsmlaMultiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < mqsmlaMaxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(15);
                CrossCoreWaitFlag<0, PIPE_MTE2>(15);
            }
        }
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline int64_t
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ConvertMqsmlaS2MetadataBlockToToken(
    const RunParamStr<HIGH_PERF> &mq35RunParam, const ConstInfo<HIGH_PERF> &mqsmlaCsaConstInfo, uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(mqsmlaCsaConstInfo.s2BaseSize);
    int64_t oriLen = mq35RunParam.s2LineOriEndIdx - mq35RunParam.s2LineStartIdx;
    int64_t cmpLen = mq35RunParam.s2CmpLineEndIdx - mq35RunParam.s2CmpLineStartIdx;
    int64_t mq35ReductionBlockSize =
        mq35RunParam.baseBlockNumPerReductionBlock > 0 ? mq35RunParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t reductionBlockSize = mq35ReductionBlockSize * s2BaseSize;
    int64_t oriReductionBlockNum = (oriLen + reductionBlockSize - 1) / reductionBlockSize;
    if (s2BlockIdx <= oriReductionBlockNum) {
        int64_t oriToken = static_cast<int64_t>(s2BlockIdx) * reductionBlockSize;
        return oriToken < oriLen ? oriToken : oriLen;
    }
    int64_t cmpToken = (static_cast<int64_t>(s2BlockIdx) - oriReductionBlockNum) * reductionBlockSize;
    return oriLen + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline bool
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ApplyMqsmlaS2MetadataRange(
    RunParamStr<HIGH_PERF> &mq35RunParam, ConstInfo<HIGH_PERF> &mqsmlaCsaConstInfo, int64_t s2StartPoint,
    int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask)
{
    int64_t oriStart = mq35RunParam.s2LineStartIdx;
    int64_t oriEnd = mq35RunParam.s2LineOriEndIdx;
    int64_t oriLen = oriEnd - oriStart;
    int64_t cmpStart = mq35RunParam.s2CmpLineStartIdx;
    int64_t cmpEnd = mq35RunParam.s2CmpLineEndIdx;
    int64_t cmpLen = cmpEnd - cmpStart;
    int64_t totalLen = oriLen + cmpLen;

    int64_t effectiveS2EndPoint = (isLastS2RangeTask && s2EndPoint == 0) ? totalLen : s2EndPoint;
    int64_t rangeStart = isFirstS2RangeTask ? s2StartPoint : 0;
    rangeStart = rangeStart < 0 ? 0 : rangeStart;
    rangeStart = rangeStart < totalLen ? rangeStart : totalLen;
    int64_t rangeEnd = isLastS2RangeTask ? effectiveS2EndPoint : totalLen;
    rangeEnd = rangeEnd < 0 ? 0 : rangeEnd;
    rangeEnd = rangeEnd < totalLen ? rangeEnd : totalLen;
    if (rangeEnd <= rangeStart) {
        mq35RunParam.oriKvLoopEndIdx = 0;
        mq35RunParam.cmpKvLoopEndIdx = 0;
        mq35RunParam.s2LoopEndIdx = 0;
        mq35RunParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    mq35RunParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    mq35RunParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < oriLen ? rangeStart : oriLen;
    int64_t oriRangeEnd = rangeEnd < oriLen ? rangeEnd : oriLen;
    mq35RunParam.s2LineStartIdx = oriStart + oriRangeStart;
    mq35RunParam.s2LineOriEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > oriLen ? rangeStart - oriLen : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > oriLen ? rangeEnd - oriLen : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    mq35RunParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    mq35RunParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(mqsmlaCsaConstInfo.s2BaseSize);
    int64_t oriRangeLen = mq35RunParam.s2LineOriEndIdx - mq35RunParam.s2LineStartIdx;
    int64_t cmpRangeLen = mq35RunParam.s2CmpLineEndIdx - mq35RunParam.s2CmpLineStartIdx;
    mq35RunParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    mq35RunParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    mq35RunParam.s2LoopEndIdx = mq35RunParam.oriKvLoopEndIdx + mq35RunParam.cmpKvLoopEndIdx;
    return mq35RunParam.s2LoopEndIdx == 0;
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ParseMqsmlaFdRunInfo(
    FdRunInfo &fdRunInfo)
{
    uint32_t aivIdx = static_cast<uint32_t>(this->mqsmlaCsaConstInfo.aivIdx);
    fdRunInfo.coreEnable = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_CORE_ENABLE_INDEX, true)) != 0;
    if (!fdRunInfo.coreEnable) {
        return;
    }

    fdRunInfo.bn2Idx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_BN2_IDX_INDEX, true));
    fdRunInfo.mIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_IDX_INDEX, true));
    fdRunInfo.workspaceIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_IDX_INDEX, true));
    fdRunInfo.workspaceNum = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_NUM_INDEX, true));
    fdRunInfo.mStartIdx = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_START_INDEX, true));
    fdRunInfo.mNum = mqsmlaCsaMetadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_NUM_INDEX, true));
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ComputeMqsmlaAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr<HIGH_PERF> &mq35RunParam)
{
    // GS1合轴, 不切G, 只切S1
    mq35RunParam.s1oIdx = gS1Index * mq35RunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t mqsmlaHalfG = (mqsmlaCsaConstInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        mq35RunParam.goIdx = (mqsmlaCsaAicIdx % 2 == 0) ? 0 : mqsmlaHalfG;
        mq35RunParam.gSplitSize = (mqsmlaCsaAicIdx % 2 == 0) ? mqsmlaHalfG : (mqsmlaCsaConstInfo.gSize - mqsmlaHalfG);
    } else {
        mq35RunParam.goIdx = 0;
        mq35RunParam.gSplitSize = mqsmlaCsaConstInfo.gSize;
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::SetMqsmlaRunInfo(
    RunInfo<HIGH_PERF> &mq35RunInfo, RunParamStr<HIGH_PERF> &mq35RunParam, int64_t taskId, int64_t s2LoopCount,
    int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < mq35RunParam.oriKvLoopEndIdx) {
        mq35RunInfo.s2StartIdx = mq35RunParam.s2LineStartIdx;
        mq35RunInfo.s2EndIdx = mq35RunParam.s2LineOriEndIdx;
    } else {
        mq35RunInfo.s2StartIdx = mq35RunParam.s2CmpLineStartIdx;
        mq35RunInfo.s2EndIdx = mq35RunParam.s2CmpLineEndIdx;
    }
    mq35RunInfo.s2LoopCount = s2LoopCount;
    if (mq35RunInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        mq35RunInfo.s1oIdx = mq35RunParam.s1oIdx;
        mq35RunInfo.boIdx = mq35RunParam.boIdx;
        mq35RunInfo.n2oIdx = mq35RunParam.n2oIdx;
        mq35RunInfo.goIdx = mq35RunParam.goIdx;
        mq35RunInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        mq35RunInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        mq35RunInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    mq35RunInfo.taskId = taskId;
    mq35RunInfo.taskIdMod2 = taskId & 1;
    mq35RunInfo.taskIdMod3 = taskId % 3;
    mq35RunInfo.s2LoopLimit = s2LoopLimit;

    mq35RunInfo.actualS1Size = mq35RunParam.actualS1Size;
    mq35RunInfo.attentionOutOffset = mq35RunParam.attentionOutOffset;
    mq35RunInfo.sOuterOffset = mq35RunParam.sOuterOffset;
    mq35RunInfo.firstFdDataWorkspaceIdx = mq35RunParam.firstFdDataWorkspaceIdx;
    mq35RunInfo.isCrossCoreSplit = mq35RunParam.isCrossCoreSplit;
    mq35RunInfo.s2SplitIdx = mq35RunParam.s2SplitIdx;
    mq35RunInfo.isFirstS2SplitCore = mq35RunParam.isFirstS2SplitCore;
    int64_t mq35ReductionBlockSize =
        mq35RunParam.baseBlockNumPerReductionBlock > 0 ? mq35RunParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t mq35ReductionLoop = s2LoopCount;
    if constexpr (IS_BATCH_CONSISTENCY) {
        // 进入 CMP 时补齐规约计数，不增加实际计算。
        if (s2LoopCount >= mq35RunParam.oriKvLoopEndIdx) {
            mq35ReductionLoop += (mq35ReductionBlockSize - mq35RunParam.oriKvLoopEndIdx % mq35ReductionBlockSize) %
                                 mq35ReductionBlockSize;
        }
    }
    int64_t baseBlockIdInReduceBlock = mq35ReductionLoop % mq35ReductionBlockSize;
    mq35RunInfo.reduceBlockId = mq35ReductionLoop / mq35ReductionBlockSize;
    mq35RunInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    mq35RunInfo.isLastBase =
        ((mq35ReductionBlockSize - baseBlockIdInReduceBlock) == 1LL) || (s2LoopCount == s2LoopLimit);
    if constexpr (IS_BATCH_CONSISTENCY) {
        mq35RunInfo.isLastBase = mq35RunInfo.isLastBase || s2LoopCount + 1 == mq35RunParam.oriKvLoopEndIdx;
    }
    mq35RunInfo.needReduce = mq35RunInfo.reduceBlockId > 0;
    this->ComputeMqsmlaBmm1Tail(mq35RunInfo, mq35RunParam);
    InitMqsmlaUniqueRunInfo(mq35RunParam, mq35RunInfo);
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::InitMqsmlaUniqueRunInfo(
    const RunParamStr<HIGH_PERF> &mq35RunParam, RunInfo<HIGH_PERF> &mq35RunInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(mq35RunParam, mq35RunInfo);
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::FreeMqsmlaEvent()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[0].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[0].idx) +
                                                          AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[1].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(mqsmlaCsaBmm1Buffers[1].idx) +
                                                          AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
        this->mqsmlaCsaCubeBlock.ReleaseMqsmlaCubeEvents();
    } else {
        this->mqsmlaCsaVecBlock.FreeEvent(mqsmlaCsaConstInfo);
    }
}

template <typename MqsmlaCsaCubeBlockType, typename MqsmlaCsaVecBlockType>
__aicore__ inline void
MixedQuantSparseFlashMlaCsa<MqsmlaCsaCubeBlockType, MqsmlaCsaVecBlockType>::ComputeMqsmlaBmm1Tail(
    RunInfo<HIGH_PERF> &mq35RunInfo, RunParamStr<HIGH_PERF> &mq35RunParam)
{
    // ------------------------S1 Base Related---------------------------
    mq35RunInfo.s1RealSize = mq35RunParam.s1RealSize;
    mq35RunInfo.halfS1RealSize = mq35RunParam.halfS1RealSize;
    mq35RunInfo.firstHalfS1RealSize = mq35RunParam.firstHalfS1RealSize;
    mq35RunInfo.mRealSize = mq35RunParam.mRealSize;
    mq35RunInfo.halfMRealSize = mq35RunParam.halfMRealSize;
    mq35RunInfo.firstHalfMRealSize = mq35RunParam.firstHalfMRealSize;

    mq35RunInfo.vec2S1BaseSize = mq35RunInfo.halfS1RealSize; // D>128 这里需要适配
    mq35RunInfo.vec2MBaseSize = mq35RunInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    mq35RunInfo.s2RealSize = mqsmlaCsaConstInfo.s2BaseSize;
    mq35RunInfo.s2AlignedSize = mq35RunInfo.s2RealSize;
    int64_t mqsmlaCurS2LoopCnt = (mq35RunInfo.s2LoopCount >= mq35RunParam.oriKvLoopEndIdx) ?
                                     (mq35RunInfo.s2LoopCount - mq35RunParam.oriKvLoopEndIdx) :
                                     mq35RunInfo.s2LoopCount;
    if (mq35RunInfo.s2StartIdx + (mqsmlaCurS2LoopCnt + 1) * mq35RunInfo.s2RealSize > mq35RunInfo.s2EndIdx) {
        mq35RunInfo.s2RealSize =
            mq35RunInfo.s2EndIdx - mqsmlaCurS2LoopCnt * mq35RunInfo.s2RealSize - mq35RunInfo.s2StartIdx;
        mq35RunInfo.s2AlignedSize = Align(mq35RunInfo.s2RealSize);
    }
}
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_H
