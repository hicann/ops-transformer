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
 * \file mixed_quant_sparse_flash_mla_csa_kernel_arch92.h
 * \brief
 */

#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H

#include "mixed_quant_sparse_flash_mla_common_arch92.h"
#include "mixed_quant_sparse_flash_mla_kvcache_arch92.h"
#include "mixed_quant_sparse_flash_mla_csa_block_cube_arch92.h"
#include "mixed_quant_sparse_flash_mla_csa_block_vector_arch92.h"
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#include "../mixed_quant_sparse_flash_mla_metadata.h"
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
class MixedQuantSparseFlashMlaCsa {
public:
    ARGS_TRAITS;
    __aicore__ inline MixedQuantSparseFlashMlaCsa(){};

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV,
                                __gm__ uint8_t* oriSparseIndices, __gm__ uint8_t* cmpSparseIndices,
                                __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
                                __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv,
                                __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV,
                                __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKv,
                                __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                const MixedQuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe);
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
                                            __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                                            __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* cuSeqlensQ,
                                            __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedOriKv,
                                            __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv,
                                            __gm__ uint8_t* sinks, __gm__ uint8_t* workspace,
                                            const MixedQuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe);
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
                                                   uint32_t s2StartIdx, uint32_t firstGs1, int32_t gs1LoopEndIdx);
    __aicore__ inline void CreateTaskN128(int64_t& taskId, int64_t s2Cur, RunInfo runInfo[4], RunParamStr& runParam,
                                          int64_t multiCoreInnerIdx, int64_t s2LoopEndIdx);
    __aicore__ inline bool ShouldExecuteTaskN128(RunInfo runInfo[4]);
    __aicore__ inline void ExecuteTaskN128(int64_t taskId, RunInfo runInfo[4]);
    __aicore__ inline void CreateTaskN64(int64_t& taskId, int64_t s2Cur, RunInfo runInfo[3], RunParamStr& runParam,
                                         int64_t multiCoreInnerIdx, int64_t s2LoopEndIdx);
    __aicore__ inline bool ShouldExecuteTaskN64(RunInfo runInfo[3]);
    __aicore__ inline void ExecuteTaskN64(int64_t taskId, RunInfo runInfo[3]);
    __aicore__ inline void UpdateAxisInfo(TaskDealMode mode, uint32_t& bN2Cur, uint32_t& gS1Cur, int64_t& s2Cur,
                                          int64_t& multiCoreInnerIdx, uint32_t& gS1StartIdx, int64_t gS1LoopEndForBN,
                                          int64_t s2LoopEndIdx);
    TPipe* pipe;

    const MixedQuantSparseFlashMlaTilingData* __restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 2;
    static constexpr uint32_t crossCoreMte2SyncFlagId = 10;
    /* 核间通道 */

    BufferManager<BufferType::GM> v0ResGmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers1;

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

    /* workspace 空间 */
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;

    /* 核Index信息 */
    int32_t aicIdx;
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t s2StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t s2EndIdx;
    uint32_t hasLoad;

    int64_t maxS2LoopCnt = 0;

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
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* oriSparseIndices,
    __gm__ uint8_t* cmpSparseIndices, __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv,
    __gm__ uint8_t* sequsedQ, __gm__ uint8_t* seqUsedOriKV, __gm__ uint8_t* seqUsedCmpKV, __gm__ uint8_t* cmpResidualKv,
    __gm__ uint8_t* oriTopkLength, __gm__ uint8_t* cmpTopkLength, __gm__ uint8_t* sinks, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
    const MixedQuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe)
{
    __gm__ uint8_t* sequsedOriKv = seqUsedOriKV; // 复用MQSMLA功能
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
    this->InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,
                           cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv, sequsedCmpKv, cmpResidualKv, sinks,
                           workspace, tiling, tPipe); // gm设置
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    vecBlock.CleanOutput(attentionOut, constInfo);
    if ASCEND_IS_AIV {
        if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE && IS_VEC_S2PHYADDR &&
                      TOPK_VALUE_MODE == TopkValueMode::TOPK_INDEX_MODE) {
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, actualSeqQlenAddr,
                                        cuSeqlensQAddr, actualSeqKvlenAddr, workspace, constInfo);
        }
    }
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf(workspace);
    cubeBlock.InitCubeBlock(pipe, l1BufferManager, query);
    this->ComputeConstexpr();

    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::GetSeqLen(
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
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* sequsedOriKv,
    __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv)
{
    auto& mixedQuantSparseFlashMlaBaseParams = this->tilingData->baseParams;
    constInfo.bSize = mixedQuantSparseFlashMlaBaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = mixedQuantSparseFlashMlaBaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = mixedQuantSparseFlashMlaBaseParams.qSeqSize;
    constInfo.s2Size = mixedQuantSparseFlashMlaBaseParams.kvSeqSize;
    constInfo.cmpS2Size = mixedQuantSparseFlashMlaBaseParams.cmpKvSeqSize;
    constInfo.sparseBlockCount = mixedQuantSparseFlashMlaBaseParams.cmpSparseBlockCount;
    constInfo.cmpRatio = mixedQuantSparseFlashMlaBaseParams.cmpRatio;
    constInfo.oriWinLeft = mixedQuantSparseFlashMlaBaseParams.oriWinLeft;
    constInfo.oriWinRight = mixedQuantSparseFlashMlaBaseParams.oriWinRight;
    constInfo.dSizeRope = mixedQuantSparseFlashMlaBaseParams.ropeHeadDim;
    constInfo.softmaxScale = mixedQuantSparseFlashMlaBaseParams.softmaxScale;
    constInfo.oriKvStride = mixedQuantSparseFlashMlaBaseParams.oriKvStride;
    constInfo.cmpKvStride = mixedQuantSparseFlashMlaBaseParams.cmpKvStride;
    constInfo.dSize = mixedQuantSparseFlashMlaBaseParams.dSize;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = mixedQuantSparseFlashMlaBaseParams.dSizeVInput;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;

    // pageAttention
    if constexpr (isPa) {
        constInfo.oriBlockSize = mixedQuantSparseFlashMlaBaseParams.paOriBlockSize;
        constInfo.cmpBlockSize = mixedQuantSparseFlashMlaBaseParams.paCmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = mixedQuantSparseFlashMlaBaseParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = mixedQuantSparseFlashMlaBaseParams.cmpMaxBlockNumPerBatch;
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
    if constexpr (TEMPLATE_MODE != MQSMLATemplateMode::SWA_TEMPLATE_MODE) {
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
    if constexpr (TEMPLATE_MODE != MQSMLATemplateMode::SWA_TEMPLATE_MODE) {
        if (sequsedCmpKv != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t*)sequsedCmpKv);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKv != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t*)cmpResidualKv);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t* query, __gm__ uint8_t* oriKV, __gm__ uint8_t* cmpKV, __gm__ uint8_t* cmpSparseIndices,
    __gm__ uint8_t* oriBlockTable, __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t* cuSeqlensQ,
    __gm__ uint8_t* cuSeqlensOriKv, __gm__ uint8_t* cuSeqlensCmpKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedOriKv, __gm__ uint8_t* sequsedCmpKv, __gm__ uint8_t* cmpResidualKv, __gm__ uint8_t* sinks,
    __gm__ uint8_t* workspace, const MixedQuantSparseFlashMlaTilingData* __restrict tiling, TPipe* tPipe)
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

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ, sinks,
                              sequsedOriKv, sequsedCmpKv, cmpResidualKv);
    cubeBlock.InitCubeInput(cuSeqlensQ, sequsedQ, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t* workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 128 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);      // 32Kb
    l1RightBuffers.Init(l1BufferManager, mm1RightSize); // 384kb
    ubBufferManager.Init(pipe, mm1ResultSize + mm2ResultSize * 2);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    bmm2Buffers1.Init(ubBufferManager, mm2ResultSize);
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
        bmm2Buffers1.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
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
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeConstexpr()
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

    if constexpr (LAYOUT_T == MQSMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == MQSMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::Process()
{
    ICachePreLoad(7);
    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ShouldDispatchTask(
    uint32_t bN2Cur, uint32_t gS1Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx, int32_t gs1LoopEndIdx)
{
    if (bN2Cur < bN2EndIdx - 1) {
        return true;
    }
    int64_t Gs1Idx = gs1LoopEndIdx;

    if (bN2Cur == bN2EndIdx - 1) {
        if (nextGs1Idx == 0) {
            Gs1Idx = GetSeqLen(bN2Cur, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        }
        if (gS1Cur < Gs1Idx) {
            return true;
        }
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline TaskDealMode MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::GetTaskDealMode(
    uint32_t& bN2Cur, uint32_t& gS1Cur, RunParamStr& runParam, uint32_t& prevBN2Cur, uint32_t& prevGS1Cur,
    int64_t& s2LoopEndIdx, int64_t& gS1LoopEndForBN, int64_t& s2Cur, uint32_t bN2EndIdx, int64_t nextGs1Idx,
    uint32_t bN2StartIdx, uint32_t& gS1StartIdx, uint32_t s2StartIdx, uint32_t firstGs1, int32_t gs1LoopEndIdx)
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
        gS1Cur = runParam.gs1LoopStartIdx;
        int64_t expectQs;
        if constexpr (LAYOUT_T == MQSMLA_LAYOUT::TND) {
            expectQs = GetSeqLen(bN2Cur, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
        } else {
            expectQs = constInfo.s1Size;
        }
        while (gS1Cur > gS1StartIdx) {
            vecBlock.InitOutput(constInfo, runParam, bN2Cur, gS1Cur);
            gS1StartIdx++;
        }
        while ((bN2Cur < bN2EndIdx - 1 || (bN2Cur == bN2EndIdx - 1 && nextGs1Idx == 0)) && expectQs > gS1LoopEndForBN) {
            vecBlock.InitOutput(constInfo, runParam, bN2Cur, gS1Cur);
            expectQs--;
        }
        runParam.s1oIdx = gS1Cur * runParam.qSNumInOneBlock;
        int64_t halfG = (constInfo.gSize + 1) / 2;
        if constexpr (IS_SPLIT_G) {
            runParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
        } else {
            runParam.goIdx = 0;
        }

        if (gS1Cur >= gS1LoopEndForBN) {
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
        if constexpr (IS_SPLIT_G) {
            maxS2LoopCnt -= runParam.s2LoopEndIdx;
        }
        s2LoopEndIdx = runParam.s2LoopEndIdx - 1;
        prevGS1Cur = gS1Cur;
    }
    if (gS1Cur != prevGS1Cur) {
        runParam.s1oIdx = gS1Cur * runParam.qSNumInOneBlock;
        int64_t halfG = (constInfo.gSize + 1) / 2;
        if constexpr (IS_SPLIT_G) {
            runParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
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
        if constexpr (IS_SPLIT_G) {
            maxS2LoopCnt -= runParam.s2LoopEndIdx;
        }
        s2LoopEndIdx = runParam.s2LoopEndIdx - 1;
        prevGS1Cur = gS1Cur;
    }
    return TaskDealMode::CREATE_TASK;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::CreateTaskN128(
    int64_t& taskId, int64_t s2Cur, RunInfo runInfo[4], RunParamStr& runParam, int64_t multiCoreInnerIdx,
    int64_t s2LoopEndIdx)
{
    RunInfo& runInfo1 = runInfo[executedTaskCount % 4];
    this->SetRunInfo(runInfo1, runParam, taskId, s2Cur, s2LoopEndIdx, multiCoreInnerIdx);
    runInfo1.isValid = true;
    ++taskId;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ShouldExecuteTaskN128(
    RunInfo runInfo[4])
{
    for (uint32_t i = 0; i < 4; i++) {
        if (runInfo[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ExecuteTaskN128(int64_t loop,
                                                                                                 RunInfo runInfo[4])
{
    RunInfo& runInfo0 = runInfo[loop % 4];
    RunInfo& runInfo1 = runInfo[(loop + 3) % 4];
    RunInfo& runInfo2 = runInfo[(loop + 2) % 4];
    RunInfo& runInfo3 = runInfo[(loop + 1) % 4];
    if (runInfo0.isValid) {
        if ASCEND_IS_AIV {
            this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), v0ResGmBuffers.Get(), runInfo0, this->constInfo);
        }
    }
    if (loop >= 1) {
        if (runInfo1.isValid) {
            if ASCEND_IS_AIC {
                this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(), v0ResGmBuffers.Get(),
                                            runInfo1, this->constInfo);
            }
        }
    }
    if (loop >= 2) {
        if (runInfo2.isValid) {
            if ASCEND_IS_AIV {
                this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo2, this->constInfo);
            } else {
                this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->bmm2Buffers1.Get(), this->l1PBuffers,
                                            this->l1RightBuffers.GetReused(), runInfo2, this->constInfo);
            }
        }
    }
    if (loop >= 3) {
        if (runInfo3.isValid) {
            if ASCEND_IS_AIV {
                this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), this->bmm2Buffers1.Get(), runInfo3,
                                           this->constInfo);
            }
            runInfo3.isValid = false;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::CreateTaskN64(
    int64_t& taskId, int64_t s2Cur, RunInfo runInfoN64[3], RunParamStr& runParam, int64_t multiCoreInnerIdx,
    int64_t s2LoopEndIdx)
{
    RunInfo& runInfo1 = runInfoN64[executedTaskCount % 3];
    this->SetRunInfo(runInfo1, runParam, taskId, s2Cur, s2LoopEndIdx, multiCoreInnerIdx);
    runInfo1.isValid = true;
    ++taskId;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ShouldExecuteTaskN64(
    RunInfo runInfoN64[3])
{
    for (uint32_t i = 0; i < 3; i++) {
        if (runInfoN64[i].isValid) {
            return true;
        }
    }
    return false;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ExecuteTaskN64(int64_t loop,
                                                                                                RunInfo runInfoN64[3])
{
    RunInfo& runInfo0 = runInfoN64[loop % 3];
    RunInfo& runInfo1 = runInfoN64[(loop + 2) % 3];
    RunInfo& runInfo2 = runInfoN64[(loop + 1) % 3];
    if (runInfo0.isValid) {
        if ASCEND_IS_AIV {
            this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), v0ResGmBuffers.Get(), runInfo0, this->constInfo);
        } else {
            this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(), v0ResGmBuffers.Get(),
                                        runInfo0, this->constInfo);
        }
    }
    if (loop >= 1) {
        if (runInfo1.isValid) {
            if ASCEND_IS_AIC {
                this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->bmm2Buffers1.Get(), this->l1PBuffers,
                                            this->l1RightBuffers.GetReused(), runInfo1, this->constInfo);
            } else {
                this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo1, this->constInfo);
            }
        }
    }
    if (loop >= 2) {
        if (runInfo2.isValid) {
            if ASCEND_IS_AIV {
                this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), this->bmm2Buffers1.Get(), runInfo2,
                                           this->constInfo);
            }
            runInfo2.isValid = false;
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::UpdateAxisInfo(
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
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    maxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                }
            }
            bmm2Buffers.Get().WaitCrossCore();
            bmm2Buffers1.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            this->cubeBlock.FreeEvent();
        } else {
            this->vecBlock.FreeEvent();
        }
        return;
    }
    RunInfo runInfo[4];
    RunInfo runInfoN64[3];
    RunParamStr runParam;
    uint32_t prevBN2Cur = bN2StartIdx;
    uint32_t firstGs1 = gS1StartIdx;
    uint32_t prevGS1Cur = gS1StartIdx;
    uint32_t bN2Cur = bN2StartIdx;
    uint32_t gS1Cur = gS1StartIdx;
    int64_t s2Cur = s2StartIdx;
    bool shouldDispatchTask = true;
    bool shouldExecuteTask = false;
    int32_t sIdx = bN2EndIdx - 1;
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == MQSMLA_LAYOUT::TND) {
        // actual seq length first
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[sIdx + 1] - cuSeqlensQAddr[sIdx]) :
                                                        actualSeqQlenAddr[sIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size : actualSeqQlenAddr[sIdx];
    }
    int32_t gs1LoopEndIdx = 0;
    int32_t qSNumInOneBlock = 0;
    qSNumInOneBlock = 1;
    if constexpr (TEMPLATE_MODE == MQSMLATemplateMode::CSA_TEMPLATE_MODE) {
        gs1LoopEndIdx = actualS1Size; //
    } else {
        // 不需要取topk, 每次计算gSize行, 循环qs次
        gs1LoopEndIdx = (actualS1Size + qSNumInOneBlock - 1) / qSNumInOneBlock;
    }
    gs1LoopEndIdx = nextGs1Idx == 0 ? gs1LoopEndIdx : nextGs1Idx;
    while (shouldDispatchTask || shouldExecuteTask) {
        shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, bN2EndIdx, nextGs1Idx, gs1LoopEndIdx);
        if (shouldDispatchTask) {
            TaskDealMode mode =
                GetTaskDealMode(bN2Cur, gS1Cur, runParam, prevBN2Cur, prevGS1Cur, s2LoopEndIdx, gS1LoopEndForBN, s2Cur,
                                bN2EndIdx, nextGs1Idx, bN2StartIdx, gS1StartIdx, s2StartIdx, firstGs1, gs1LoopEndIdx);
            if (mode == TaskDealMode::CREATE_TASK) {
                if constexpr (IS_SPLIT_G) {
                    CreateTaskN128(createdTaskCount, s2Cur, runInfo, runParam, multiCoreInnerIdx, s2LoopEndIdx);
                } else {
                    CreateTaskN64(createdTaskCount, s2Cur, runInfoN64, runParam, multiCoreInnerIdx, s2LoopEndIdx);
                }
                UpdateAxisInfo(mode, bN2Cur, gS1Cur, s2Cur, multiCoreInnerIdx, gS1StartIdx, gS1LoopEndForBN,
                               s2LoopEndIdx);
            }
        }
        if constexpr (IS_SPLIT_G) {
            shouldExecuteTask = ShouldExecuteTaskN128(runInfo);
        } else {
            shouldExecuteTask = ShouldExecuteTaskN64(runInfoN64);
        }
        if (shouldExecuteTask) {
            if constexpr (IS_SPLIT_G) {
                ExecuteTaskN128(executedTaskCount, runInfo);
            } else {
                ExecuteTaskN64(executedTaskCount, runInfoN64);
            }
            ++executedTaskCount;
        }
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
            }
        }
        bmm2Buffers.Get().WaitCrossCore();
        bmm2Buffers1.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
        this->cubeBlock.FreeEvent();
    } else {
        this->vecBlock.FreeEvent();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::SetRunInfo(
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
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr& runParam, RunInfo& runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void MixedQuantSparseFlashMlaCsa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo& runInfo,
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
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_KERNEL_ARCH92_H
