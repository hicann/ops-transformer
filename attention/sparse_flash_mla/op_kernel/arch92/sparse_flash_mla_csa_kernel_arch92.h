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
 * \file sparse_flash_mla_csa_kernel_arch92.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
#define SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H

#include "sparse_flash_mla_common_arch92.h"
#include "sparse_flash_mla_kvcache_arch92.h"
#include "sparse_flash_mla_csa_block_cube_arch92.h"
#include "sparse_flash_mla_csa_block_vector_arch92.h"
#include "kernel_operator.h"
#include "../sparse_flash_mla_kernel_metadata.h"
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

namespace SMLAKernel {

#ifndef SMLA_TASK_DEAL_MODE_DEFINED
#define SMLA_TASK_DEAL_MODE_DEFINED
enum class TaskDealMode : uint8_t {
    CREATE_TASK = 0,
    SKIP = 1,
};
#endif
template <typename CubeBlockType, typename VecBlockType>
class SparseFlashMlaCsaKernel {
public:
    ARGS_TRAITS;
    __aicore__ inline SparseFlashMlaCsaKernel(){};

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV,
                                __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKV,
                                __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* workspace, const SparseFlashMlaTilingData* __restrict tiling,
                                TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline int64_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t>& actualSeqGm, GlobalTensor<int32_t>& cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline void ParseTilingData(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ,
                                           __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
                                           __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV,
                                           __gm__ uint8_t* cmpResidualKV);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                            __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                            __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                            __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                            __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ,
                                            __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV,
                                            __gm__ uint8_t* cmpResidualKV, __gm__ uint8_t* oriTopkLength,
                                            __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks,
                                            __gm__ uint8_t* workspace,
                                            const SparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t* workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void FreeEvent();
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
    // 将 metadata 中 s2 分核的 block 索引转换为 token 偏移（对齐 A5 基线）
    __aicore__ inline int64_t ConvertS2MetadataBlockToToken(const RunParamStr& runParam, const ConstInfo& constInfo,
                                                            uint32_t s2BlockIdx);
    // 按 metadata 的 s2 分核范围裁剪 runParam 中 ori/cmp 的 s2 行区间（对齐 A5 基线）
    __aicore__ inline bool ApplyS2MetadataRange(RunParamStr& runParam, ConstInfo& constInfo, int64_t s2StartPoint,
                                                int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask);
    __aicore__ inline void CreateTask(int64_t& taskId, int64_t s2Cur, RunInfo runInfo[3], RunParamStr& runParam,
                                      int64_t multiCoreInnerIdx, int64_t s2LoopEndIdx);
    __aicore__ inline bool ShouldExecuteTask(RunInfo runInfo[3]);
    __aicore__ inline void ExecuteTask(int64_t taskId, RunInfo runInfo[3]);
    __aicore__ inline void UpdateAxisInfo(TaskDealMode mode, uint32_t& bN2Cur, uint32_t& gS1Cur, int64_t& s2Cur,
                                          int64_t& multiCoreInnerIdx, uint32_t& gS1StartIdx, int64_t gS1LoopEndForBN,
                                          int64_t s2LoopEndIdx);
    TPipe* pipe;

    const SparseFlashMlaTilingData* __restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint64_t SYNC_MODE = 2;
    static constexpr uint32_t PRELOAD_NUM = 3;
    static constexpr uint32_t crossCoreMte2SyncFlagId = 15; // IS_SPLIT_G 核间 MTE2 同步 flag ID
    static constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    uint32_t crossCoreSyncBufId = 0;
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
    uint32_t s2MetaEndIdx = 0;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices,
    __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
    __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKV,
    __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    const SparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe)
{
    __gm__ uint8_t* sequsedOriKv = seqUsedOriKV;
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
    this->ParseTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV,
                          cmpResidualKV);
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    vecBlock.CleanOutput(attentionOut, softmaxLse, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                           cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, seqUsedOriKV, seqUsedCmpKV,
                           cmpResidualKV, oriTopkLength, cmpTopkLength, sinks, workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::GetSeqLen(
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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* cuSeqlensOriKv,
    __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV,
    __gm__ uint8_t* cmpResidualKV)
{
    auto& sparseFlashMLABaseParams = this->tilingData->baseParams;
    auto& sparseFlashMLACmpParams = this->tilingData->cmpParams;
    constInfo.bSize = sparseFlashMLABaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = sparseFlashMLABaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = sparseFlashMLABaseParams.qSeqSize;
    constInfo.s2Size = sparseFlashMLABaseParams.kvSeqSize;
    constInfo.cmpS2Size = sparseFlashMLACmpParams.cmpKvSeqSize;
    constInfo.oriSparseBlockCount = sparseFlashMLABaseParams.oriSparseBlockCount;
    constInfo.cmpSparseBlockCount = sparseFlashMLACmpParams.cmpSparseBlockCount;
    constInfo.cmpRatio = sparseFlashMLACmpParams.cmpRatio;
    constInfo.oriMaskMode = sparseFlashMLABaseParams.oriMaskMode;
    constInfo.cmpMaskMode = sparseFlashMLACmpParams.cmpMaskMode;
    constInfo.oriWinLeft = sparseFlashMLABaseParams.oriWinLeft;
    constInfo.oriWinRight = sparseFlashMLABaseParams.oriWinRight;
    constInfo.layoutType = sparseFlashMLABaseParams.outputLayout;
    constInfo.returnSoftmaxLse = sparseFlashMLABaseParams.returnSoftmaxLse;
    constInfo.tileSize = 0;
    constInfo.dSizeRope = 64;
    if ASCEND_IS_AIV {
        constInfo.softmaxScale = sparseFlashMLABaseParams.softmaxScale;
    }
    constInfo.dSize = 512;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = constInfo.dSize;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        this->constInfo.isActualLenDimsOriKVNull = 0U;
    } else {
        this->constInfo.isActualLenDimsOriKVNull = (seqUsedOriKV == nullptr);
    }

    // pageAttention
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        constInfo.oriBlockSize = sparseFlashMLABaseParams.oriBlockSize;
        constInfo.cmpBlockSize = sparseFlashMLABaseParams.cmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sparseFlashMLABaseParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = sparseFlashMLACmpParams.cmpMaxBlockNumPerBatch;
        constInfo.oriP = sparseFlashMLABaseParams.oriP;
        constInfo.oriMLow = sparseFlashMLABaseParams.oriMLow;
        constInfo.oriMHigh = sparseFlashMLABaseParams.oriMHigh;
        constInfo.cmpP = sparseFlashMLACmpParams.cmpP;
        constInfo.cmpMLow = sparseFlashMLACmpParams.cmpMLow;
        constInfo.cmpMHigh = sparseFlashMLACmpParams.cmpMHigh;
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            constInfo.cmpKeyStride0 = constInfo.cmpBlockSize * constInfo.dSize;
            constInfo.cmpKvStride = constInfo.cmpBlockSize * constInfo.dSize;
        }
        constInfo.oriKeyStride0 = constInfo.oriBlockSize * constInfo.dSize;
        constInfo.oriKvStride = constInfo.oriBlockSize * constInfo.dSize;
    }

    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensOriKv);
        hasCuSeqlensOriKv = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensCmpKv);
            hasCuSeqlensCmpKv = true;
        }
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t*)sequsedQ);
        hasActualSeqQlen = true;
    }
    if (seqUsedOriKV != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t*)seqUsedOriKV);
        hasActualSeqOriKvlen = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (seqUsedCmpKV != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t*)seqUsedCmpKV);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKV != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKV);
        }
    }

    constInfo.needInit = 0;
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && constInfo.oriMaskMode != 0) {
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t s2Size;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                s2Size = actualSeqOriKvlenGm.GetValue(bIdx);
            } else {
                s2Size = GetSeqLen(bIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm,
                                   constInfo.s2Size);
            }
            int64_t s1Size =
                GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices,
    __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
    __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKV,
    __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
    const SparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t*)cuSeqlensQ;
    }
    if (seqUsedOriKV != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t*)seqUsedOriKV;
    }
    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t*)sequsedQ;
    }

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ,
                              sinks, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    cubeBlock.InitCubeInput(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ, cuSeqlensQ,
                            cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, constInfo);
    if (oriTopkLength != nullptr) {
        constInfo.hasOriTopkLength = true;
        oriTopkLengthGm.SetGlobalBuffer((__gm__ int32_t*)oriTopkLength);
    } else {
        constInfo.hasOriTopkLength = false;
    }
    if (cmpTopkLength != nullptr) {
        constInfo.hasCmpTopkLength = true;
        cmpTopkLengthGm.SetGlobalBuffer((__gm__ int32_t*)cmpTopkLength);
    } else {
        constInfo.hasCmpTopkLength = false;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t* workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    uint32_t mm1LeftSize = constInfo.s1BaseSize * 512 * sizeof(Q_T);
    uint32_t l1BufferSize = mm2LeftSize * 2 + mm1RightSize * 3 + mm1LeftSize;
    l1BufferManager.Init(pipe, l1BufferSize);
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    // v0 稀疏结果从 AIV 写入、由 cube 读取的跨核 L1 通道，无条件分配
    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    if ASCEND_IS_AIC {
        l1PBuffers.Get().SetCrossCore();
        l1PBuffers.Get().SetCrossCore();
    }

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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ComputeConstexpr()
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

    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->constInfo.needInit) {
        SyncAll();
    }
    ProcessMainLoop();
    FreeEvent();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ShouldDispatchTask(
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
__aicore__ inline TaskDealMode SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::GetTaskDealMode(
    uint32_t& bN2Cur, uint32_t& gS1Cur, RunParamStr& runParam, uint32_t& prevBN2Cur, uint32_t& prevGS1Cur,
    int64_t& s2LoopEndIdx, int64_t& gS1LoopEndForBN, int64_t& s2Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx,
    uint32_t bN2StartIdx, uint32_t& gS1StartIdx, int64_t& remainingS2Sync, uint32_t s2StartIdx, uint32_t firstGs1,
    int32_t gs1LoopEndIdx)
{
    bool lastBN = (bN2Cur == bN2EndIdx - 1);
    bool isFirstTask = (s2Cur == 0 && gS1Cur == firstGs1 && bN2Cur == bN2StartIdx);

    if (bN2Cur != prevBN2Cur || isFirstTask) {
        runParam.boIdx = bN2Cur;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(
            runParam, this->constInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm, this->cuSeqlensCmpKvGm,
            this->actualSeqQlenGm, this->actualSeqOriKvlenGm, this->actualSeqCmpKvlenGm, this->cmpResidualKvGm,
            this->hasActualSeqQlen, this->hasActualSeqOriKvlen, this->hasActualSeqCmpKvlen, this->hasCuSeqlensCmpKv);
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

        bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Cur, this->cuSeqlensQGm);
        bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
        bool isFirstS2RangeTask = (bN2Cur == bN2StartIdx && gS1Cur == runParam.gs1LoopStartIdx);
        bool isLastS2RangeTask = (lastBN && gS1Cur == runParam.gs1LoopEndIdx - 1);
        int64_t s2StartPoint = ConvertS2MetadataBlockToToken(runParam, this->constInfo, s2StartIdx);
        int64_t s2EndPoint = (isLastS2RangeTask && this->s2MetaEndIdx == 0) ?
                                 0 :
                                 ConvertS2MetadataBlockToToken(runParam, this->constInfo, this->s2MetaEndIdx);
        s2NoNeedCalc = ApplyS2MetadataRange(runParam, this->constInfo, s2StartPoint, s2EndPoint, isFirstS2RangeTask,
                                            isLastS2RangeTask) ||
                       s2NoNeedCalc;
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

        if (gS1Cur >= runParam.gs1LoopEndIdx) {
            ++gS1Cur;
            s2Cur = 0;
            if (gS1Cur >= gS1LoopEndForBN) {
                gS1Cur = 0;
                gS1StartIdx = 0;
                ++bN2Cur;
            }
            return TaskDealMode::SKIP;
        }

        runParam.s1oIdx = gS1Cur * runParam.qSNumInOneBlock;
        if constexpr (IS_SPLIT_G) {
            runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64;
        } else {
            runParam.goIdx = 0;
        }

        bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Cur, this->cuSeqlensQGm);
        bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
        bool isFirstS2RangeTask = (bN2Cur == bN2StartIdx && gS1Cur == runParam.gs1LoopStartIdx);
        bool isLastS2RangeTask = (lastBN && gS1Cur == runParam.gs1LoopEndIdx - 1);
        int64_t s2StartPoint = ConvertS2MetadataBlockToToken(runParam, this->constInfo, s2StartIdx);
        int64_t s2EndPoint = (isLastS2RangeTask && this->s2MetaEndIdx == 0) ?
                                 0 :
                                 ConvertS2MetadataBlockToToken(runParam, this->constInfo, this->s2MetaEndIdx);
        s2NoNeedCalc = ApplyS2MetadataRange(runParam, this->constInfo, s2StartPoint, s2EndPoint, isFirstS2RangeTask,
                                            isLastS2RangeTask) ||
                       s2NoNeedCalc;

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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::CreateTask(int64_t& taskId, int64_t s2Cur,
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
__aicore__ inline bool SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ShouldExecuteTask(RunInfo runInfo[3])
{
    for (uint32_t i = 0; i < 3; i++) {
        if (runInfo[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ExecuteTask(int64_t loop,
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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::UpdateAxisInfo(
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
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
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
    this->s2MetaEndIdx = s2EndIdx;

    if (nextGs1Idx != 0 || s2EndIdx != 0) {
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
    int64_t s2Cur = 0;
    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    // 计算当前batch的gs1LoopEndIdx
    int32_t sIdx = bN2EndIdx - 1;
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // actual seq length first
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[sIdx + 1] - cuSeqlensQAddr[sIdx]) :
                                                        actualSeqQlenAddr[sIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size : actualSeqQlenAddr[sIdx];
    }
    int32_t gs1LoopEndIdx = 0;
    int32_t qSNumInOneBlock = 1;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        gs1LoopEndIdx = actualS1Size; // 对于CSA, 不切G轴, 每次拷贝一行的topk，只算一行的qs
    } else {                          // SWA/CFA
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
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo& runInfo, RunParamStr& runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2OriLineStartIdx;
        runInfo.s2EndIdx = runParam.s2OriLineEndIdx;
    } else {
        runInfo.s2StartIdx = runParam.s2CmpLineStartIdx;
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
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.firstFdDataWorkspaceIdx = runParam.firstFdDataWorkspaceIdx;
    runInfo.isCrossCoreSplit = runParam.isCrossCoreSplit;
    runInfo.s2SplitIdx = runParam.s2SplitIdx;
    runInfo.isFirstS2SplitCore = runParam.isFirstS2SplitCore;
    int64_t safeBaseBlockNum =
        runParam.baseBlockNumPerReductionBlock > 0 ? runParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t baseBlockIdInReduceBlock = s2LoopCount % safeBaseBlockNum;
    runInfo.reduceBlockId = s2LoopCount / safeBaseBlockNum;
    runInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    runInfo.isLastBase = baseBlockIdInReduceBlock == safeBaseBlockNum - 1LL || s2LoopCount == s2LoopLimit;
    runInfo.needReduce = runInfo.reduceBlockId > 0;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr& runParam, RunInfo& runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo, this->constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo& runInfo,
                                                                                             RunParamStr& runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

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

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ConvertS2MetadataBlockToToken(
    const RunParamStr& runParam, const ConstInfo& constInfo, uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    int64_t reductionBlockSize = runParam.baseBlockNumPerReductionBlock * s2BaseSize;
    int64_t oriReductionBlockNum = (oriLen + reductionBlockSize - 1) / reductionBlockSize;
    int64_t reductionBlockIdx = static_cast<int64_t>(s2BlockIdx);
    if (reductionBlockIdx < oriReductionBlockNum) {
        int64_t oriToken = reductionBlockIdx * reductionBlockSize;
        return oriToken < oriLen ? oriToken : oriLen;
    }
    int64_t cmpToken = (reductionBlockIdx - oriReductionBlockNum) * reductionBlockSize;
    return oriLen + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ApplyS2MetadataRange(
    RunParamStr& runParam, ConstInfo& constInfo, int64_t s2StartPoint, int64_t s2EndPoint, bool isFirstS2RangeTask,
    bool isLastS2RangeTask)
{
    int64_t oriStart = runParam.s2OriLineStartIdx;
    int64_t oriEnd = runParam.s2OriLineEndIdx;
    int64_t oriLen = oriEnd - oriStart;
    int64_t cmpStart = runParam.s2CmpLineStartIdx;
    int64_t cmpEnd = runParam.s2CmpLineEndIdx;
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
        runParam.oriKvLoopEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        runParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    runParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    runParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < oriLen ? rangeStart : oriLen;
    int64_t oriRangeEnd = rangeEnd < oriLen ? rangeEnd : oriLen;
    runParam.s2OriLineStartIdx = oriStart + oriRangeStart;
    runParam.s2OriLineEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > oriLen ? rangeStart - oriLen : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > oriLen ? rangeEnd - oriLen : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    runParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    runParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriRangeLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpRangeLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    runParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return runParam.s2LoopEndIdx == 0;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::FreeEvent()
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

} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
