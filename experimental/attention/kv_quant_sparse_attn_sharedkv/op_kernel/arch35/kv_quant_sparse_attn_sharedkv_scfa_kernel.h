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
 * \file kv_quant_sparse_attn_sharedkv_scfa_kernel.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "kv_quant_sparse_attn_sharedkv_kvcache.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_cube.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_vector.h"
#include "kernel_operator.h"
#include "../kv_quant_sparse_attn_sharedkv_metadata.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class KvQuantSparseAttnSharedkvScfa {
public:
    ARGS_TRAITS;
    __aicore__ inline KvQuantSparseAttnSharedkvScfa(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *workspace,
                                            const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    TPipe *pipe;

    const KvQuantSparseAttnSharedkvTilingData *__restrict tilingData;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint32_t PRELOAD_NUM = 3;
    /* 核间通道 */
    BufferManager<BufferType::GM> v0ResGmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    /* GM信息 */
    GlobalTensor<uint32_t> metadataGm;
    __gm__ int32_t *cuSeqlensQAddr = nullptr;
    __gm__ int32_t *actualSeqKvlenAddr = nullptr;
    __gm__ int32_t *actualSeqQlenAddr = nullptr;
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

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
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
    this->ParseTilingData(cuSeqlensQ, sequsedKv);
    this->InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, sequsedQ,
                           sequsedKv, sinks, workspace, tiling, tPipe); // gm设置
    vecBlock.InitVecBlock(tPipe, cuSeqlensQ, sequsedKv);
    vecBlock.CleanOutput(attentionOut, constInfo);
    /* 先完成主流程UB分配，地址计算复用stage0OutBuf和stage2OutBuf */
    InitMMResBuf(workspace);
    this->InitLocalBuffer();
    if ASCEND_IS_AIV {
        if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE && IS_VEC_S2PHYADDR) {
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, actualSeqQlenAddr,
                                        cuSeqlensQAddr, actualSeqKvlenAddr, workspace, constInfo);
            this->vecBlock.SetKvPhyAddrReady();
        }
    }
    /* cube侧不依赖sharedParams的scalar前置 */
    cubeBlock.InitCubeBlock(pipe, query);
    this->ComputeConstexpr();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    constInfo.bSize = sparseAttnSharedkvBaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    constInfo.s2Size = sparseAttnSharedkvBaseParams.kvSeqSize;
    constInfo.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    constInfo.cmpRatio = sparseAttnSharedkvBaseParams.cmpRatio;
    constInfo.oriWinLeft = sparseAttnSharedkvBaseParams.oriWinLeft;
    constInfo.oriWinRight = sparseAttnSharedkvBaseParams.oriWinRight;
    constInfo.tileSize = sparseAttnSharedkvBaseParams.tileSize;
    constInfo.dSizeRope = sparseAttnSharedkvBaseParams.ropeHeadDim;
    constInfo.softmaxScale = sparseAttnSharedkvBaseParams.softmaxScale;
    constInfo.oriKvStride = sparseAttnSharedkvBaseParams.oriKvStride;
    constInfo.cmpKvStride = sparseAttnSharedkvBaseParams.cmpKvStride;
    constInfo.dSize = sparseAttnSharedkvBaseParams.dSize;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = sparseAttnSharedkvBaseParams.dSizeVInput;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;
    constInfo.actualSeqLenKVSize = constInfo.bSize;

    // pageAttention
    if constexpr (isPa) {
        constInfo.oriBlockSize = sparseAttnSharedkvBaseParams.paOriBlockSize;
        constInfo.cmpBlockSize = sparseAttnSharedkvBaseParams.paCmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.cmpMaxBlockNumPerBatch;
    }

    // actQ->TND, actKV pa场景任意layout均有
    constInfo.isActualLenDimsKVNull = false;

    constInfo.needInit = 0;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)sequsedKv);
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        GlobalTensor<int32_t> cuSeqlensQGm;
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            int64_t s1Size = cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx);
            if (s1Size > s2Size) {
                constInfo.needInit = 1;
                break;
            }
        }
    } else {
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            int64_t s1Size = constInfo.s1Size;
            if (s1Size > s2Size) {
                constInfo.needInit = 1;
                break;
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t *)cuSeqlensQ;
    }
    if (sequsedKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)sequsedKv;
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)sequsedQ;
    }

    vecBlock.InitGlobalBuffer(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ, sinks);
    cubeBlock.InitCubeInput(cuSeqlensQ, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitMMResBuf(
    __gm__ uint8_t *workspace)
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    l1BufferManager.Init(pipe, mm2LeftSize * 2); // 仅L1P使用, L1Right已迁移至cube侧通过tPipe分配
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);

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
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // This operator fixes n2Size to 1 in ParseTilingData. Keep the derived
    // values here in sync if shared-KV later supports N2 > 1.
    // constInfo.n2Size = 1;
    constInfo.n2G = constInfo.gSize;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.n2GDv = constInfo.gDv;
    constInfo.mm1Ka = constInfo.dSize;
    if ASCEND_IS_AIV {
        // n2G == gSize when n2Size is fixed to 1, so adjacent output rows are
        // contiguous and the MTE3 dst stride is always 0.
        constInfo.attentionOutStride = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->constInfo.needInit) {
        SyncAll<false>();
    }
    ICachePreLoad(6);

    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    int64_t maxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(10);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(10);
                }
            }
        }
        this->vecBlock.WaitKvPhyAddrBeforeRead();
        return;
    }
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
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, this->cuSeqlensQAddr, this->actualSeqQlenAddr,
                                              this->actualSeqKvlenAddr);
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
                    ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Index, this->cuSeqlensQAddr);
                bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
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
                        this->vecBlock.ProcessVec0(v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1, this->constInfo);
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
                        this->cubeBlock.IterateLoadQK(v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                      this->constInfo, isFirstLoop);
                        isFirstLoop = false;
                    } else {
                        if constexpr (IS_SPLIT_G) {
                            if (taskId > 0 && maxS2LoopCnt > 0) {
                                maxS2LoopCnt--;
                                CrossCoreSetFlag<0, PIPE_MTE2>(10);
                                CrossCoreWaitFlag<0, PIPE_MTE2>(10);
                            }
                        }
                    }
                    if (taskId > 1 && notLast) {
                        auto &runInfo2 = runInfo[(taskId + 2) % 4];
                        RunInfo &runInfoNext = runInfo[(taskId + 3) % 4];
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), v0ResGmBuffers.Get(runInfo2.taskIdMod3),
                                                    notLastTwoLoop, runInfoNext, runInfo2, this->constInfo);
                    }
                    if (taskId > 2) {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, runInfo3,
                                                    this->constInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    // hasLoad非零但没有有效Vec0的AIV也必须参与同一次同步。
    this->vecBlock.WaitKvPhyAddrBeforeRead();
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(10);
                CrossCoreWaitFlag<0, PIPE_MTE2>(10);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : 64;
    } else {
        runParam.goIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineEndIdx;
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
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &runInfo, RunParamStr &runParam)
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
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
