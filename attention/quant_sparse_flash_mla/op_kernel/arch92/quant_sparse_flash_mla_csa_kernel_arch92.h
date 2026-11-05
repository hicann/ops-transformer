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
 * \file quant_sparse_flash_mla_csa_kernel_arch92.h
 * \brief
 */

#ifndef QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
#define QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H

#include "quant_sparse_flash_mla_common_arch92.h"
#include "quant_sparse_flash_mla_kvcache_arch92.h"
#include "quant_sparse_flash_mla_csa_block_cube_arch92.h"
#include "quant_sparse_flash_mla_csa_block_vector_arch92.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../quant_sparse_flash_mla_metadata.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/CopyInL1.h"

#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
enum class TaskDealMode : uint8_t {
    CREATE_TASK = 0,
    SKIP = 1,
};
template <typename CubeBlockType, typename VecBlockType>
class QuantSparseFlashMlaScfa {
public:
    ARGS_TRAITS;
    __aicore__ inline QuantSparseFlashMlaScfa(){};

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                __gm__ uint8_t* qDescale, __gm__ uint8_t* oriKVDescale, __gm__ uint8_t* cmpKVDescale,
                                __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV,
                                __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKv,
                                __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                const QuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline int64_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t>& actualSeqGm, GlobalTensor<int32_t>& cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline void ParseTilingData(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                           __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* sequsedOriKv,
                                           __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedCmpKv,
                                           __gm__ uint8_t* cmpResidualKv);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* qDescale, __gm__ uint8_t* oriKVDescale,
                                            __gm__ uint8_t* cmpKVDescale, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                            __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ,
                                            __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv,
                                            __gm__ uint8_t* cmpResidualKv, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* workspace,
                                            const QuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t* workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo& runInfo, RunParamStr& runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo& runInfo, RunParamStr& runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr& runParam, RunInfo& runInfo);
    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx,
                                              int32_t gs1LoopEndIdx);
    __aicore__ inline TaskDealMode GetTaskDealMode(uint32_t& bN2Cur, uint32_t& gS1Cur, RunParamStr& runParam,
                                                   uint32_t& prevBN2Cur, uint32_t& prevGS1Cur, int64_t& s2LoopEndIdx,
                                                   int64_t& gS1LoopEndForBN, int64_t& s2Cur, uint32_t bN2EndIdx,
                                                   int64_t nextGs1Idx, uint32_t bN2StartIdx, uint32_t& gS1StartIdx,
                                                   int64_t& remainingS2Sync, uint32_t s2StartIdx, uint32_t firstGs1,
                                                   int32_t gs1LoopEndIdx);
    __aicore__ inline void CreateTask(int64_t& taskId, int64_t s2Cur, RunInfo runInfo[3], RunParamStr& runParam,
                                      int64_t multiCoreInnerIdx, int64_t s2LoopEndIdx);
    __aicore__ inline bool ShouldExecuteTask(RunInfo runInfo[3]);
    __aicore__ inline void ExecuteTask(int64_t taskId, RunInfo runInfo[3]);
    __aicore__ inline void UpdateAxisInfo(TaskDealMode mode, uint32_t& bN2Cur, uint32_t& gS1Cur, int64_t& s2Cur,
                                          int64_t& multiCoreInnerIdx, uint32_t& gS1StartIdx, int64_t gS1LoopEndForBN,
                                          int64_t s2LoopEndIdx);
    TPipe* pipe;

    const QuantSparseFlashMlaTilingData* __restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 2;
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    /* GM信息 */
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqQlenGm;
    GlobalTensor<int32_t> actualSeqOriKvlenGm;
    GlobalTensor<int32_t> actualSeqCmpKvlenGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    bool hasCuSeqlensQ = false;
    bool hasCuSeqlensOriKv = false;
    bool hasCuSeqlensCmpKv = false;
    bool hasActualSeqQlen = false;
    bool hasActualSeqOriKvlen = false;
    bool hasActualSeqCmpKvlen = false;
    __gm__ int32_t* cuSeqlensQAddr = nullptr;
    __gm__ int32_t* actualSeqKvlenAddr = nullptr;
    __gm__ int32_t* actualSeqQlenAddr = nullptr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
    int64_t createdTaskCount = 0;
    int64_t executedTaskCount = 0;
    int64_t multiCoreInnerIdx = 1;
    int64_t s2LoopEndIdx = 0;
    int64_t gS1LoopEndForBN = 0;
    int64_t s2Cur = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* qDescale,
    __gm__ uint8_t* oriKVDescale, __gm__ uint8_t* cmpKVDescale, __gm__ uint8_t* oriSparseIndices,
    __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
    __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKv,
    __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace, const QuantSparseFlashMlaTilingData* __restrict tiling,
    TPipe* tPipe)
{
    __gm__ uint8_t* sequsedOriKv = seqUsedOriKV; // 复用QSMLA功能
    __gm__ uint8_t* sequsedCmpKv = seqUsedCmpKV;
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
        this->tilingData = tiling;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx;
        this->tilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->metadataGm.SetGlobalBuffer((__gm__ uint32_t*)metadata);

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;

    this->pipe = tPipe;
    this->ParseTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, sequsedOriKv, cuSeqlensCmpKv, sequsedCmpKv,
                          cmpResidualKv);
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    vecBlock.CleanOutput(attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, oriKV, cmpKV, qDescale, oriKVDescale, cmpKVDescale, cmpSparseIndices, oriBlockTable,
                           cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv,
                           sequsedCmpKv, cmpResidualKv, sinks, workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::GetSeqLen(
    int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens, GlobalTensor<int32_t>& actualSeqGm,
    GlobalTensor<int32_t>& cuSeqlensGm, int64_t defaultSize)
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
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* sequsedOriKv,
    __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    auto& QuantSparseFlashMlaBaseParams = this->tilingData->baseParams;
    constInfo.bSize = QuantSparseFlashMlaBaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = QuantSparseFlashMlaBaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = QuantSparseFlashMlaBaseParams.qSeqSize;
    constInfo.s2Size = QuantSparseFlashMlaBaseParams.kvSeqSize;
    constInfo.cmpS2Size = QuantSparseFlashMlaBaseParams.cmpKvSeqSize;
    constInfo.sparseBlockCount = QuantSparseFlashMlaBaseParams.cmpSparseBlockCount;
    constInfo.cmpRatio = QuantSparseFlashMlaBaseParams.cmpRatio;
    constInfo.oriWinLeft = QuantSparseFlashMlaBaseParams.oriWinLeft;
    constInfo.oriWinRight = QuantSparseFlashMlaBaseParams.oriWinRight;
    constInfo.softmaxScale = QuantSparseFlashMlaBaseParams.softmaxScale;
    constInfo.oriKvStride = QuantSparseFlashMlaBaseParams.oriKvStride;
    constInfo.cmpKvStride = QuantSparseFlashMlaBaseParams.cmpKvStride;
    constInfo.dSize = QuantSparseFlashMlaBaseParams.dSize;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = QuantSparseFlashMlaBaseParams.dSizeVInput;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;

    // pageAttention
    if constexpr (isPa) {
        constInfo.oriBlockSize = QuantSparseFlashMlaBaseParams.paOriBlockSize;
        constInfo.cmpBlockSize = QuantSparseFlashMlaBaseParams.paCmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = QuantSparseFlashMlaBaseParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = QuantSparseFlashMlaBaseParams.cmpMaxBlockNumPerBatch;
    }

    // actQ->TND, actKV pa场景任意layout均有
    constInfo.isActualLenDimsKVNull = false;

    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
        hasCuSeqlensOriKv = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
            hasCuSeqlensCmpKv = true;
        }
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t*)sequsedQ);
        hasActualSeqQlen = true;
    }
    if (sequsedOriKv != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t*)sequsedOriKv);
        hasActualSeqOriKvlen = true;
    }
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        if (sequsedCmpKv != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t*)sequsedCmpKv);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKv != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKv);
        }
    }

    constInfo.needInit = 0;

    for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
        int64_t s2Size = GetSeqLen(bIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm,
                                   constInfo.s2Size);
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
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* qDescale,
    __gm__ uint8_t* oriKVDescale, __gm__ uint8_t* cmpKVDescale, __gm__ uint8_t* cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* cuSeqlensQ,
    __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv, __gm__ uint8_t* sinks,
    __gm__ uint8_t* workspace, const QuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t*)cuSeqlensQ;
    }
    if (sequsedOriKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t*)sequsedOriKv;
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t*)sequsedQ;
    }

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, qDescale, oriKVDescale, cmpKVDescale, cmpSparseIndices, oriBlockTable,
                              cmpBlockTable, sequsedQ, sinks, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    cubeBlock.InitCubeInput(cuSeqlensQ, sequsedQ, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t* workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);      // 32Kb
    l1RightBuffers.Init(l1BufferManager, mm1RightSize); // 384kb
    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ComputeConstexpr()
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
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->constInfo.needInit) {
        SyncAll();
    }
    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ShouldDispatchTask(
    uint32_t bN2Cur, uint32_t gS1Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx, int32_t gs1LoopEndIdx)
{
    if (bN2Cur < bN2EndIdx - 1) {
        return true;
    }
    if ((bN2Cur == bN2EndIdx - 1) && static_cast<int64_t>(gS1Cur) < gs1LoopEndIdx) {
        return true;
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline TaskDealMode QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::GetTaskDealMode(
    uint32_t& bN2Cur, uint32_t& gS1Cur, RunParamStr& runParam, uint32_t& prevBN2Cur, uint32_t& prevGS1Cur,
    int64_t& s2LoopEndIdx, int64_t& gS1LoopEndForBN, int64_t& s2Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx,
    uint32_t bN2StartIdx, uint32_t& gS1StartIdx, int64_t& remainingS2Sync, uint32_t s2StartIdx, uint32_t firstGs1,
    int32_t gs1LoopEndIdx)
{
    bool lastBN = (bN2Cur == bN2EndIdx - 1);
    bool isFirstTask = (s2Cur == s2StartIdx && gS1Cur == firstGs1 && bN2Cur == bN2StartIdx);

    if (bN2Cur != prevBN2Cur || isFirstTask) {
        runParam.boIdx = bN2Cur;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm,
                                              this->cuSeqlensCmpKvGm, this->actualSeqQlenGm, this->actualSeqOriKvlenGm,
                                              this->actualSeqCmpKvlenGm, this->cmpResidualKvGm, this->hasCuSeqlensOriKv,
                                              this->hasCuSeqlensCmpKv, this->hasActualSeqQlen,
                                              this->hasActualSeqOriKvlen, this->hasActualSeqCmpKvlen);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx);
        prevBN2Cur = bN2Cur;
        gS1LoopEndForBN = runParam.gs1LoopEndIdx;

        runParam.s1oIdx = gS1Cur * runParam.qSNumInOneBlock;
        if constexpr (IS_SPLIT_G) {
            runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64;
        } else {
            runParam.goIdx = 0;
        }

        // 增加 S1 无效行跳过
        if (gS1Cur >= runParam.gs1LoopEndIdx) {
            s2Cur = 0;
            gS1Cur = 0;
            gS1StartIdx = 0;
            ++bN2Cur;
            return TaskDealMode::SKIP;
        }

        bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Cur, this->cuSeqlensQAddr);
        bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
        if (s1NoNeedCalc || s2NoNeedCalc) {
            ++gS1Cur;
            s2Cur = 0;
            if (gS1Cur >= gS1LoopEndForBN) {
                gS1Cur = 0;
                gS1StartIdx = 0;
                ++bN2Cur;
            }
            return TaskDealMode::SKIP;
        }
        s2LoopEndIdx = runParam.s2LoopEndIdx - 1;
        prevGS1Cur = gS1Cur;
    }

    if (gS1Cur != prevGS1Cur) {
        gS1LoopEndForBN = runParam.gs1LoopEndIdx;

        runParam.s1oIdx = gS1Cur * runParam.qSNumInOneBlock;
        if constexpr (IS_SPLIT_G) {
            runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64;
        } else {
            runParam.goIdx = 0;
        }

        bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Cur, this->cuSeqlensQAddr);
        bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);

        if (s1NoNeedCalc || s2NoNeedCalc) {
            ++gS1Cur;
            s2Cur = 0;
            if (gS1Cur >= gS1LoopEndForBN) {
                gS1Cur = 0;
                gS1StartIdx = 0;
                ++bN2Cur;
            }
            return TaskDealMode::SKIP;
        }
        s2LoopEndIdx = runParam.s2LoopEndIdx - 1;
        prevGS1Cur = gS1Cur;
    }
    return TaskDealMode::CREATE_TASK;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::CreateTask(int64_t& taskId, int64_t s2Cur,
                                                                                        RunInfo runInfo[3],
                                                                                        RunParamStr& runParam,
                                                                                        int64_t multiCoreInnerIdx,
                                                                                        int64_t s2LoopEndIdx)
{
    RunInfo& runInfo1 = runInfo[executedTaskCount % 3];
    this->SetRunInfo(runInfo1, runParam, taskId, s2Cur, s2LoopEndIdx, multiCoreInnerIdx);
    runInfo1.isValid = true;
    ++taskId;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ShouldExecuteTask(RunInfo runInfo[3])
{
    for (uint32_t i = 0; i < 3; i++) {
        if (runInfo[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ExecuteTask(int64_t loop,
                                                                                         RunInfo runInfo[3])
{
    RunInfo& runInfo0 = runInfo[loop % 3];
    RunInfo& runInfo1 = runInfo[(loop + 2) % 3];
    RunInfo& runInfo2 = runInfo[(loop + 1) % 3];
    if (runInfo0.isValid) {
        if ASCEND_IS_AIC {
            this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(), runInfo0, this->constInfo);
        } else {
            this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), runInfo0, this->constInfo);
        }
    }
    if (loop >= 1) {
        if (runInfo1.isValid) {
            if ASCEND_IS_AIV {
                this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo1, this->constInfo);
            } else {
                this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, this->l1RightBuffers.GetReused(),
                                            runInfo1, this->constInfo);
            }
        }
    }
    if (loop >= 2) {
        if (runInfo2.isValid) {
            if ASCEND_IS_AIV {
                this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo2, this->constInfo);
            }
            runInfo2.isValid = false;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::UpdateAxisInfo(
    TaskDealMode mode, uint32_t& bN2Cur, uint32_t& gS1Cur, int64_t& s2Cur, int64_t& multiCoreInnerIdx,
    uint32_t& gS1StartIdx, int64_t gS1LoopEndForBN, int64_t s2LoopEndIdx)
{
    if (mode == TaskDealMode::CREATE_TASK && s2Cur <= s2LoopEndIdx) {
        ++s2Cur;
    }
    if (s2Cur > s2LoopEndIdx) {
        s2Cur = 0;
        ++gS1Cur;
        ++multiCoreInnerIdx;
    }
    if (gS1Cur >= gS1LoopEndForBN) {
        gS1Cur = 0;
        gS1StartIdx = 0;
        ++bN2Cur;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    int64_t maxS2LoopCnt = 0;
    if (hasLoad == 0) {
        return;
    }

    uint32_t bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    uint32_t gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    uint32_t s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false));
    uint32_t bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    uint32_t nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    uint32_t s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false));

    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    RunInfo runInfo[3];
    RunParamStr runParam;
    int64_t remainingS2Sync = maxS2LoopCnt;
    uint32_t prevBN2Cur = bN2StartIdx;
    uint32_t firstGs1 = gS1StartIdx;
    uint32_t prevGS1Cur = gS1StartIdx;
    uint32_t bN2Cur = bN2StartIdx;
    uint32_t gS1Cur = gS1StartIdx;
    int64_t s2Cur = s2StartIdx;
    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    // 计算当前batch的gs1LoopEndIdx
    int32_t sIdx = bN2EndIdx - 1;
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        // actual seq length first
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[sIdx + 1] - cuSeqlensQAddr[sIdx]) :
                                                        actualSeqQlenAddr[sIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size : actualSeqQlenAddr[sIdx];
    }

    int32_t gs1LoopEndIdx = 0;
    int32_t qSNumInOneBlock = 1;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
        gs1LoopEndIdx = actualS1Size;
    } else {
        // 不需要取topk, 每次计算gSize行, 循环qs次
        gs1LoopEndIdx = (actualS1Size + qSNumInOneBlock - 1) / qSNumInOneBlock;
    }

    gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    while (shouldDispatchTask || shouldExecuteTask) {
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, bN2EndIdx, nextGs1Idx, gs1LoopEndIdx);
        if (shouldDispatchTask) {
            TaskDealMode mode = GetTaskDealMode(bN2Cur, gS1Cur, runParam, prevBN2Cur, prevGS1Cur, s2LoopEndIdx,
                                                gS1LoopEndForBN, s2Cur, bN2EndIdx, nextGs1Idx, bN2StartIdx, gS1StartIdx,
                                                remainingS2Sync, s2StartIdx, firstGs1, gs1LoopEndIdx);
            if (mode == TaskDealMode::CREATE_TASK) {
                CreateTask(createdTaskCount, s2Cur, runInfo, runParam, multiCoreInnerIdx, s2LoopEndIdx);
                UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur, multiCoreInnerIdx, gS1StartIdx, gS1LoopEndForBN,
                               s2LoopEndIdx);
            }
        }
        shouldExecuteTask = ShouldExecuteTask(runInfo);
        if (shouldExecuteTask) {
            ExecuteTask(executedTaskCount, runInfo);
            ++executedTaskCount;
        }
    }

    if ASCEND_IS_AIC {
        bmm2Buffers.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo& runInfo, RunParamStr& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
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
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr& runParam, RunInfo& runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void QuantSparseFlashMlaScfa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo& runInfo,
                                                                                             RunParamStr& runParam)
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
#endif // QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
