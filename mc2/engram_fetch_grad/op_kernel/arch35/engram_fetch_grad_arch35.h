/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_ARCH35_H
#define ENGRAM_FETCH_GRAD_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_VERSION_NUM >= 90200000) && (HCOMM_VERSION_NUM >= 90200000)
#define ENABLE_ENGRAM_FETCH_GRAD_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_grad_tiling_data.h"
#include "../engram_fetch_grad_utils.h"
#include "adv_api/hccl/hccl.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif

#include "engram_fetch_grad_unique.h"
#include "sort/sort_lib.h"
#include "all_to_all_v_udma_put.h"
#include "adv_api/reduce/sum.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_GRAD_KERNEL)

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchGradSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchGradArch35 {
public:
    __aicore__ inline EngramFetchGradArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR perm, GM_ADDR sendCounts,
                                GM_ADDR recvCounts, GM_ADDR recvLocalEntry, GM_ADDR numRecv, GM_ADDR gradUniqueOut,
                                GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut, GM_ADDR workspaceGM, TPipe* pipe,
                                const EngramFetchGradTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void DrainChecked(uint64_t handle);
    __aicore__ inline void UnsortGrad();
    __aicore__ inline uint32_t LoadGradChunk(int64_t pos, int64_t end, LocalTensor<uint8_t>& buf, int32_t bufIdx,
                                             LocalTensor<int32_t>& idxUb, uint32_t tokensPerBuf);
    __aicore__ inline void StoreGradChunk(int64_t base, uint32_t count, LocalTensor<uint8_t>& buf, int32_t bufIdx);
    __aicore__ inline void InitFlagsAndDispls();
    __aicore__ inline void LoadTilingData(const EngramFetchGradTilingData* tilingData);
    __aicore__ inline void CalcWorkSpaceOffset(const EngramFetchGradTilingData* tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void SetAllToAllvConfig();
    __aicore__ inline void RunSortPhase(uint32_t numRecv);
    __aicore__ inline void CrossRankBarrierIssue();
    __aicore__ inline void CrossRankBarrierWait();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank, uint32_t pipelineIdx);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);

    // ══ 核分区掩盖 ══
    __aicore__ inline void InitCorePartition(uint32_t numRecv);
    __aicore__ inline void SoftBarrier();         // sort 子集软屏障(每代一页)
    __aicore__ inline void InitSortBarrierArea(); // 屏障页区初始清零(core0, 一次性)
    GM_ADDR gradFetchedGM_{nullptr};
    GM_ADDR permGM_{nullptr};
    GM_ADDR recvLocalEntryGM_{nullptr};
    GM_ADDR numRecvGM_{nullptr};
    GM_ADDR gradUniqueOutGM_{nullptr};
    GM_ADDR uniqueLocalEntryOutGM_{nullptr};
    GM_ADDR numUniqueOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    __gm__ EngramCommContext* ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t channelsPerRank_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    int64_t totalRecv_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    uint64_t winSize_{0};
    uint64_t ubSize_{0};
    uint32_t tileBytes_{0};
    uint32_t gradSubBatch_{Mc2Kernel::GRAD_SUB_BATCH};
    uint32_t uniqueEntryBytes_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    GM_ADDR gradSortedGM_{nullptr};
    GM_ADDR recvGradGM_{nullptr};
    GM_ADDR sendCountsGM_{nullptr};
    GM_ADDR recvCountsGM_{nullptr};
    GM_ADDR sdisplsGM_{nullptr};
    GM_ADDR rdisplsGM_{nullptr};
    GM_ADDR counterScratchGM_{nullptr};
    GM_ADDR flagScratchGM_{nullptr};
    GM_ADDR segCountGM_{nullptr};
    GM_ADDR coreStartGM_{nullptr};
    GM_ADDR sortCompanionGM_{nullptr};
    GM_ADDR sortWorkspaceGm_{nullptr};

    uint32_t sortNumTileData_{0U}; // SortLib tile 元素数
    uint32_t sortTmpUbSize_{0U};   // SortLib AscendC::Sort 临时 UB
    uint32_t aivNumTiling_{0U};    // tiling 侧平台核数(策略换算基准; 运行期实际核数=totalBlocks_)
    uint32_t sortCoreCnt_{0U};     // tiling 决定的 sort 参与核数(编排策略, 运行期钳位执行)
    uint32_t chunkElems_{0U};

    TBuf<> entryBuf_;
    TBuf<> gradBuf_;
    AscendC::TPipe* tpipe_{nullptr};
    int32_t eventMte2ToMte3_[2] = {0, 0}; // UnsortGrad/LoadGradChunk 乒乓事件(LocalCopy 事件归 allToAllv_)
    int32_t eventMte3ToMte2_[2] = {0, 0};
    Mc2Kernel::AllToAllvPut allToAllv_; // 变长全交换模块(pipelinesPerRank/窗口布局/credit 协议内聚)
    TBuf<> hcommBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> indicesBuf_;
    TBuf<> offBuf_;
    uint32_t indicesBufElements_{0};
    TBuf<> castFp32Buf_;
    TBuf<> accumBuf_;
    uint32_t statusBufBytes_{0U};
    uint32_t tempBufBytes_{0U};
    uint32_t indicesBufBytes_{0U};
    uint32_t castBufBytes_{0U};
    uint32_t accumBufBytes_{0U};

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    EngramFetchGradUnique::EngramFetchGradUnique uniqueScatter_;

    bool isAllToAllvCore_{false};
    bool isSortCore_{false};
    bool useCorePartition_{false};
    uint32_t sortCoreIdx_{0};
    uint32_t sortCoreBase_{0};
    uint32_t numSortCores_{0};
    static constexpr uint32_t SOFT_BARRIER_MAX_GENS = 16U;
    uint32_t softBarrierGen_{0};
    uint32_t softBarrierPageInts_{0};
    GM_ADDR sortBarrierGM_{nullptr};
    TBuf<> softBarrierScratchBuf_;

    friend struct EngramSortSoftBarrier;
};

struct EngramSortSoftBarrier {
    EngramFetchGradArch35* self;
    __aicore__ inline void operator()() const
    {
        self->SoftBarrier();
    }
};

__aicore__ inline void EngramFetchGradArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchGradArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
    ascendc_assert(ret == 0, "Drain failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank, uint32_t pipelineIdx)
{
    uint32_t channelIdx = pipelineIdx;
    if (channelIdx >= channelsPerRank_) {
        channelIdx = 0U;
    }
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

// 跨 rank barrier 用(窗口地址 = commBuffer[dst] + offset)
__aicore__ inline GM_ADDR EngramFetchGradArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline void EngramFetchGradArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    while (sumOfFlag != compareFlag) {
        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            __gm__ int32_t* slotPtr = (__gm__ int32_t*)(statusWinBase + i * STATE_OFFSET);
            sumOfFlag += AscendC::ReadGmBypassDCache(slotPtr);
        }
    }
}

__aicore__ inline void EngramFetchGradArch35::ClearStatusFlags(GM_ADDR statusWinBase)
{
    LocalTensor<int32_t> cleanTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanTensor, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> statusGM;
    statusGM.SetGlobalBuffer((__gm__ int32_t*)statusWinBase);
    DataCopy(statusGM, cleanTensor, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline static uint32_t AlignRowStride(uint32_t hiddenBytes)
{
    return (hiddenBytes + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
}

__aicore__ inline static uint32_t UnsortTokensPerBuf(int64_t hiddenBytes, uint32_t tileBytes, uint32_t indicesElems)
{
    uint32_t tokensPerBuf = 1U;
    if (hiddenBytes > 0) {
        uint32_t byTile = static_cast<uint32_t>(static_cast<uint64_t>(tileBytes) / static_cast<uint64_t>(hiddenBytes));
        tokensPerBuf = (byTile > 1U) ? byTile : 1U;
    }
    if (tokensPerBuf > indicesElems) {
        tokensPerBuf = indicesElems;
    }
    return (tokensPerBuf > 0U) ? tokensPerBuf : 1U;
}

__aicore__ inline static uint32_t MaxGradRowsPerPing(uint32_t hiddenBytes, uint32_t gradSubBatch)
{
    uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / AlignRowStride(hiddenBytes);
    if (maxByPong < 1U) {
        maxByPong = 1U;
    }
    if (maxByPong > gradSubBatch) {
        maxByPong = gradSubBatch;
    }
    return maxByPong;
}

__aicore__ inline static uint32_t CastBufBytes(uint32_t hiddenDim, uint32_t maxByPong)
{
    uint32_t fp32RowStride =
        (hiddenDim * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    return Ceil(fp32RowStride * maxByPong * 2U, UB_ALIGN) * UB_ALIGN;
}

__aicore__ inline static uint32_t AccumBufBytes(uint32_t hiddenDim)
{
    return Ceil(hiddenDim * sizeof(float), UB_ALIGN) * UB_ALIGN * Mc2Kernel::ACCUM_BUF_COPIES;
}

__aicore__ inline static uint32_t UniqueEntryBytes(uint32_t hiddenDim, int32_t outputDtype)
{
    uint32_t bytes = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
    if (outputDtype != Mc2Kernel::ENGRAM_DT_FLOAT) {
        uint32_t castHalfBytes =
            (hiddenDim * 2U + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        bytes += 2U * castHalfBytes;
    }
    return bytes;
}

__aicore__ inline void EngramFetchGradArch35::LoadTilingData(const EngramFetchGradTilingData* tilingData)
{
    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    totalRecv_ = tilingData->totalRecv;
    inputDtype_ = tilingData->inputDtype;
    outputDtype_ = tilingData->outputDtype;
    ubSize_ = tilingData->ubSize;
    winSize_ = tilingData->commBufferSize;
    gradSubBatch_ = tilingData->gradSubBatch; // 一次GM加载到累加的粒度，scatter 阶段子批行数上限
    sortNumTileData_ = tilingData->sortNumTileData;
    sortTmpUbSize_ = tilingData->sortTmpUbSize;
    chunkElems_ = tilingData->chunkElems; // 行太大时需要切分多个chunk，chunk元素数2048
    aivNumTiling_ = tilingData->aivNum;
    sortCoreCnt_ = tilingData->sortCoreCnt;

    ascendc_assert(aivNumTiling_ != 0U && sortCoreCnt_ != 0U && gradSubBatch_ != 0U,
                   "invalid tiling: aivNum=%u, sortCoreCnt=%u, gradSubBatch=%u", aivNumTiling_, sortCoreCnt_,
                   gradSubBatch_);

    // ub装不下时需要按chunk处理
    if (chunkElems_ > 0U) {
        uint32_t chunkFp32Stride =
            (chunkElems_ * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        castBufBytes_ = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) ? chunkFp32Stride : 0U;
        accumBufBytes_ = 2U * chunkFp32Stride; // accum使用双缓冲
        // 必要的unique阶段大小; 非 fp32 输出再加 cast staging 双缓冲
        uniqueEntryBytes_ = Mc2Kernel::FLUSH_CAST_HEAD_BYTES +
                            ((outputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) ? 2U * chunkFp32Stride : 0U);
    }
}

// workspace 布局 12 个 GM 区段地址。
// gradSorted | recvGrad | sdispls | rdispls | counterScratch | flagScratch |
// segCount | coreStart | sortCompanion | sortWorkspace(6段) | sortBarrier(软同步区)
__aicore__ inline void EngramFetchGradArch35::CalcWorkSpaceOffset(const EngramFetchGradTilingData* tilingData)
{
    uint64_t workSpaceOffset = 0;
    gradSortedGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    int64_t totalRecvUB = totalRecv_;
    recvGradGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += static_cast<uint64_t>(totalRecvUB) * static_cast<uint64_t>(hiddenBytes_);
    sdisplsGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += numRanks_ * UB_ALIGN;
    rdisplsGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += numRanks_ * UB_ALIGN;
    counterScratchGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += static_cast<uint64_t>(aivNumTiling_) * UB_ALIGN;
    flagScratchGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += UB_ALIGN;
    segCountGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    coreStartGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    uint32_t maxSortCount = static_cast<uint32_t>(totalRecv_);
    uint64_t sortCompanionSize = Ceil(static_cast<uint64_t>(maxSortCount) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    sortCompanionGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += sortCompanionSize;
    constexpr uint64_t kRadixRounds = 4U;
    uint64_t slTileCount = static_cast<uint64_t>(tilingData->sortTileCount);
    uint64_t seg0 = Ceil(256U * kRadixRounds * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg1 = Ceil(slTileCount * 256U * kRadixRounds * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg2 = Ceil(static_cast<uint64_t>(totalRecv_) * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg3 = 2U * Ceil(slTileCount * 256U * 2U, UB_ALIGN) * UB_ALIGN;
    uint64_t seg4 = Ceil(slTileCount * static_cast<uint64_t>(sortNumTileData_), UB_ALIGN) * UB_ALIGN;
    uint64_t seg5 = Ceil(static_cast<uint64_t>(totalRecv_) * 4U, UB_ALIGN) * UB_ALIGN;
    uint64_t sortWorkspaceSize = seg0 + seg1 + seg2 + seg3 + seg4 + seg5;
    sortWorkspaceGm_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += sortWorkspaceSize;
    softBarrierPageInts_ = totalBlocks_ * STATE_OFFSET / sizeof(int32_t);
    uint64_t sortBarrierBytes = static_cast<uint64_t>(SOFT_BARRIER_MAX_GENS) *
                                Ceil(static_cast<uint64_t>(totalBlocks_) * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    sortBarrierGM_ = workspaceGM_ + workSpaceOffset;
    workSpaceOffset += sortBarrierBytes;
}

__aicore__ inline void EngramFetchGradArch35::InitBuffers()
{
    // hcomm 通信库工作区
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    // 跨 rank barrier 状态标志区(每 rank 32B 槽)
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    statusBufBytes_ = statusBufSize;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);

    // temp临时存储区，反复用于独立不同阶段
    uint32_t tempBufSize = statusBufSize;
    uint32_t entryBatchBytes = ENTRY_BATCH_CAP * sizeof(int32_t);
    if (tempBufSize < entryBatchBytes) {
        tempBufSize = entryBatchBytes;
    }
    uint32_t coreArrayBytes = Ceil(Ceil(static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t), UB_ALIGN) +
                                       static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t),
                                   UB_ALIGN) *
                              UB_ALIGN;
    if (tempBufSize < coreArrayBytes) {
        tempBufSize = coreArrayBytes;
    }
    // displs 批量构造区：sdisplUb+rdisplUb 两段各 numRanks_*UB_ALIGN 字节（tempBuf_ 容量必须覆盖）
    uint32_t displsBatchBytes = 2U * numRanks_ * UB_ALIGN;
    if (tempBufSize < displsBatchBytes) {
        tempBufSize = displsBatchBytes;
    }
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    tempBufBytes_ = tempBufSize;

    // unsort perm 索引 + 判重 entry 加载 + counts 暂存
    uint32_t indicesBufSize = Mc2Kernel::IDX_BUF_BYTES;
    if (indicesBufSize < statusBufSize) {
        indicesBufSize = statusBufSize;
    }
    // countsBuf 暂存区：recvCounts 占 numRanks_*4B，sendCounts 暂存位于偏移 numRanks_*UB_ALIGN、
    // 长度 numRanks_*UB_ALIGN，尾部合计 2*numRanks_*UB_ALIGN 字节
    uint32_t stagingBytes = 2U * numRanks_ * UB_ALIGN;
    if (indicesBufSize < stagingBytes) {
        indicesBufSize = stagingBytes;
    }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    indicesBufBytes_ = indicesBufSize;
    indicesBufElements_ = indicesBufSize / sizeof(int32_t);

    // unsort 批量行偏移(int64)
    uint32_t unsortBatch = UnsortTokensPerBuf(hiddenBytes_, tileBytes_, indicesBufElements_);
    uint32_t offBufBytes =
        (unsortBatch * sizeof(int64_t) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    tpipe_->InitBuffer(offBuf_, offBufBytes);

    // 全行模式的 cast/accum/entry 尾部字节覆盖推导(chunk 模式已在 LoadTilingData 算好)
    uint32_t maxByPong = MaxGradRowsPerPing(static_cast<uint32_t>(hiddenBytes_), gradSubBatch_);
    if (chunkElems_ == 0U) {
        if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
            if (inputDtype_ == outputDtype_) {
                castBufBytes_ = (static_cast<uint32_t>(hiddenDim_) * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) /
                                Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            } else {
                castBufBytes_ = CastBufBytes(hiddenDim_, maxByPong);
            }
        } else {
            castBufBytes_ = 0U;
        }
        accumBufBytes_ = AccumBufBytes(hiddenDim_);
        uniqueEntryBytes_ = UniqueEntryBytes(hiddenDim_, outputDtype_);
    }

    // scatter 阶段峰值容量断言(防 UB 溢出)
    uint32_t scatterPeak = Mc2Kernel::GRAD_BUF_BYTES + uniqueEntryBytes_ + castBufBytes_ + accumBufBytes_;
    uint32_t uniqueBufSize = (scatterPeak > Mc2Kernel::COMM_BUF_BYTES) ? scatterPeak : Mc2Kernel::COMM_BUF_BYTES;
    uint64_t permanentUsed = Mc2Kernel::HCOMM_INIT_SIZE + statusBufBytes_ + tempBufBytes_ + indicesBufBytes_;
    ascendc_assert(static_cast<uint64_t>(uniqueBufSize) <= ubSize_ - permanentUsed,
                   "scatter stage buffers overflow: need=%u, permanent=%llu, ubSize=%llu", uniqueBufSize, permanentUsed,
                   ubSize_);

    // 判重组表(entryBuf) + grad 加载乒乓(gradBuf)
    tpipe_->InitBuffer(entryBuf_, Mc2Kernel::ENTRY_BUF_BYTES);
    tpipe_->InitBuffer(gradBuf_, Mc2Kernel::GRAD_BUF_BYTES);
}

// allToAllv 模块的参数装配 + 绑定传输层
// (pipelinesPerRank/布局/角色在 Process 的 InitJobs 注入; 本地段拷贝乒乓复用 entryBuf/gradBuf,
//  与 UnsortGrad 相位不重叠)
__aicore__ inline void EngramFetchGradArch35::SetAllToAllvConfig()
{
    Mc2Kernel::AllToAllvConfig config;
    config.rankId = rankId_;
    config.rankSize = numRanks_;
    config.channelsPerRank = channelsPerRank_;
    config.winSize = winSize_;
    config.numSlots = NUM_SLOTS;
    config.rowBytes = static_cast<uint64_t>(hiddenBytes_);
    config.srcGM = gradSortedGM_;
    config.dstGM = recvGradGM_;
    config.sendCountsGM = sendCountsGM_;
    config.recvCountsGM = recvCountsGM_;
    config.sdisplsGM = sdisplsGM_;
    config.rdisplsGM = rdisplsGM_;
    config.creditScratchGM = counterScratchGM_;
    allToAllv_.Init(config, ctxPtr_->hcommHandle, ctxPtr_->commBuffer, hcomm_, statusBuf_.Get<int32_t>(),
                    entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(), tpipe_);
}

__aicore__ inline void EngramFetchGradArch35::Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR perm,
                                                   GM_ADDR sendCounts, GM_ADDR recvCounts, GM_ADDR recvLocalEntry,
                                                   GM_ADDR numRecv, GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut,
                                                   GM_ADDR numUniqueOut, GM_ADDR workspaceGM, TPipe* pipe,
                                                   const EngramFetchGradTilingData* tilingData)
{
    tpipe_ = pipe;
    // 输入
    gradFetchedGM_ = gradFetched;
    permGM_ = perm;
    sendCountsGM_ = sendCounts;
    recvCountsGM_ = recvCounts;
    recvLocalEntryGM_ = recvLocalEntry;
    numRecvGM_ = numRecv;
    gradUniqueOutGM_ = gradUniqueOut;

    // 输出
    uniqueLocalEntryOutGM_ = uniqueLocalEntryOut;
    numUniqueOutGM_ = numUniqueOut;

    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    totalBlocks_ = GetBlockNum();
    tileBytes_ = TILE_BYTES; // 一次datacopy上限64k

    ctxPtr_ = (__gm__ EngramCommContext*)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    // 必须一致性校验，否则 workspace 的 displs 区按 Host rankSize 规划而 Kernel 按 numRanks_ 写入会越界
    ascendc_assert(numRanks_ != 0U && numRanks_ <= Mc2Kernel::MAX_QP_SIZE && numRanks_ == tilingData->rankSize,
                   "invalid rankSize: commContext=%u, tiling=%u", numRanks_, tilingData->rankSize);
    channelsPerRank_ = (ctxPtr_->channelsPerRank == 0U) ? 1U : ctxPtr_->channelsPerRank;

    // 加载tilingdata及相关参数设置
    LoadTilingData(tilingData);

    // pingpong同步id获取
    eventMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    eventMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    eventMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    // workspace划分计算
    CalcWorkSpaceOffset(tilingData);

    // UB缓冲池分配
    InitBuffers();

    uniqueScatter_.Init(aivId_, totalBlocks_, rankId_, numRanks_, numEntriesPerRank_, hiddenDim_, hiddenBytes_,
                        inputDtype_, outputDtype_, tpipe_, entryBuf_, gradBuf_, indicesBuf_, tempBuf_, statusBuf_,
                        castFp32Buf_, accumBuf_);
    uniqueScatter_.SetChunkElems(chunkElems_);

    // allToAllv 模块需要的 params 装配与初始化
    SetAllToAllvConfig();
}

__aicore__ inline uint32_t EngramFetchGradArch35::LoadGradChunk(int64_t pos, int64_t end, LocalTensor<uint8_t>& buf,
                                                                int32_t bufIdx, LocalTensor<int32_t>& idxUb,
                                                                uint32_t tokensPerBuf)
{
    uint32_t n = 0;
    if (end > pos) {
        n = static_cast<uint32_t>(end - pos);
        if (n > tokensPerBuf) {
            n = tokensPerBuf;
        }
    }
    if (n == 0) {
        return 0;
    }

    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t*)permGM_);
    DataCopyExtParams cpParams{1U, static_cast<uint32_t>(n) * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t*)gradFetchedGM_);
    DataCopyPadExtParams<uint8_t> gPad{false, 0, 0, 0};
    uint32_t hb32 = static_cast<uint32_t>(hiddenBytes_);
    LocalTensor<int64_t> offUb = offBuf_.Get<int64_t>();
    AscendC::Cast<int64_t, int32_t>(offUb, idxUb, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(n));
    AscendC::Muls<int64_t>(offUb, offUb, static_cast<int64_t>(hiddenBytes_), static_cast<int32_t>(n));
    EngramFetchGradSyncFunc<HardEvent::V_S>();
    for (uint32_t j = 0; j < n; j++) {
        int64_t srcOff = offUb.GetValue(j);
        DataCopyExtParams gParams{1U, hb32, 0U, 0U, 0U};
        DataCopyPad(buf[static_cast<uint64_t>(j) * hiddenBytes_], srcGm[srcOff], gParams, gPad);
    }
    AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_[bufIdx]);
    return n;
}

__aicore__ inline void EngramFetchGradArch35::StoreGradChunk(int64_t base, uint32_t count, LocalTensor<uint8_t>& buf,
                                                             int32_t bufIdx)
{
    AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventMte2ToMte3_[bufIdx]);

    GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer((__gm__ uint8_t*)gradSortedGM_);
    uint64_t dstOff = static_cast<uint64_t>(base) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t totalBytes = static_cast<uint64_t>(count) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t srcOff = 0;
    while (totalBytes > 0) {
        uint32_t chunkBytes = (totalBytes > MAX_BLOCK_BYTES) ? MAX_BLOCK_BYTES : static_cast<uint32_t>(totalBytes);
        DataCopyExtParams sParams{1U, chunkBytes, 0U, 0U, 0U};
        DataCopyPad(dstGm[dstOff + srcOff], buf[srcOff], sParams);
        srcOff += chunkBytes;
        totalBytes -= chunkBytes;
    }
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_[bufIdx]);
}

__aicore__ inline void EngramFetchGradArch35::UnsortGrad()
{
    int64_t totalPerCore = (numTokens_ + static_cast<int64_t>(totalBlocks_) - 1) / static_cast<int64_t>(totalBlocks_);
    int64_t start = static_cast<int64_t>(aivId_) * totalPerCore;
    int64_t end = start + totalPerCore;
    if (end > numTokens_) {
        end = numTokens_;
    }
    if (start >= end) {
        return;
    }

    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t*)permGM_);

    if ((static_cast<uint64_t>(hiddenBytes_) % Mc2Kernel::UB_ALIGN != 0U) ||
        (static_cast<uint64_t>(hiddenBytes_) > tileBytes_)) {
        LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();
        int64_t pos = start;
        while (pos < end) {
            uint32_t batch = static_cast<uint32_t>(end - pos);
            if (batch > indicesBufElements_) {
                batch = indicesBufElements_;
            }
            DataCopyExtParams cpParams{1U, batch * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
            EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

            uint32_t j = 0;
            while (j < batch) {
                int32_t runStartIdx = idxUb.GetValue(j);
                uint32_t runLen = 1;
                while (j + runLen < batch) {
                    int32_t nextIdx = idxUb.GetValue(j + runLen);
                    if (nextIdx == runStartIdx + static_cast<int32_t>(runLen)) {
                        runLen++;
                    } else {
                        break;
                    }
                }

                GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(runStartIdx) * static_cast<uint64_t>(hiddenBytes_);
                GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(pos + j) * static_cast<uint64_t>(hiddenBytes_);
                allToAllv_.LocalCopy(dst, src, static_cast<uint64_t>(runLen) * static_cast<uint64_t>(hiddenBytes_));

                j += runLen;
            }
            pos += batch;
        }
        return;
    }

    uint32_t tokensPerBuf = UnsortTokensPerBuf(hiddenBytes_, tileBytes_, indicesBufElements_);

    LocalTensor<uint8_t> buf0 = entryBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = gradBuf_.Get<uint8_t>();
    LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();

    int32_t curBuf = 0;
    int64_t pos = start;
    int64_t base0 = start;
    uint32_t n0 = LoadGradChunk(pos, end, buf0, curBuf, idxUb, tokensPerBuf);
    pos += n0;
    bool stored0 = false;
    bool stored1 = false;

    while (pos < end) {
        int32_t nxtBuf = 1 - curBuf;
        if (stored0 && nxtBuf == 0) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_[0]);
        }
        if (stored1 && nxtBuf == 1) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_[1]);
        }
        uint32_t n1 = (nxtBuf == 0) ? LoadGradChunk(pos, end, buf0, nxtBuf, idxUb, tokensPerBuf) :
                                      LoadGradChunk(pos, end, buf1, nxtBuf, idxUb, tokensPerBuf);
        pos += n1;

        if (curBuf == 0) {
            StoreGradChunk(base0, n0, buf0, curBuf);
            stored0 = true;
        } else {
            StoreGradChunk(base0, n0, buf1, curBuf);
            stored1 = true;
        }
        base0 += n0;
        curBuf = nxtBuf;
        n0 = n1;
    }

    if (curBuf == 0) {
        StoreGradChunk(base0, n0, buf0, curBuf);
        stored0 = true;
    } else {
        StoreGradChunk(base0, n0, buf1, curBuf);
        stored1 = true;
    }

    if (stored0) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_[0]);
    }
    if (stored1) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_[1]);
    }
}

__aicore__ inline void EngramFetchGradArch35::InitFlagsAndDispls()
{
    if (aivId_ != 0) {
        return;
    }

    // 填充 1 到 flagScratch(32B), 为后续 CrossRankBarrierIssue 提供核0 写入本 rank 窗口标志槽的数据源
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagInit;
    flagInit.SetGlobalBuffer((__gm__ int32_t*)flagScratchGM_);
    DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    // 读两份计数进 indicesBuf: recvCounts(紧排 int32，8B)在前, sendCounts(每 rank 32B 槽)在后
    // recv紧排 | 空隙 | send(32B间隔)
    LocalTensor<int32_t> countsBuf = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> recvCountsGM;
    recvCountsGM.SetGlobalBuffer((__gm__ int32_t*)recvCountsGM_);
    DataCopyExtParams rcParams{1U, numRanks_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> i32Pad{false, 0, 0, 0};
    DataCopyPad(countsBuf, recvCountsGM, rcParams, i32Pad);

    uint32_t sendUbIdx = numRanks_;
    GlobalTensor<int32_t> sendCountsGM;
    sendCountsGM.SetGlobalBuffer((__gm__ int32_t*)sendCountsGM_);
    DataCopyExtParams scParams{1U, numRanks_ * UB_ALIGN, 0U, 0U, 0U};
    DataCopyPad(countsBuf[sendUbIdx * SENDCOUNT_STRIDE_RATIO], sendCountsGM, scParams, i32Pad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    // 标量前缀和算 displs: 每 rank 的发送/接收起始行号, 按 32B 槽距填成 int64 表
    int64_t sAccum = 0;
    int64_t rAccum = 0;
    LocalTensor<int64_t> sdisplUb = tempBuf_.Get<int64_t>();
    LocalTensor<int64_t> rdisplUb = sdisplUb[numRanks_ * (UB_ALIGN / sizeof(int64_t))];
    for (uint32_t r = 0; r < numRanks_; r++) {
        int32_t sCount = countsBuf.GetValue(sendUbIdx * SENDCOUNT_STRIDE_RATIO + r * SENDCOUNT_STRIDE_RATIO);
        int32_t rCount = countsBuf.GetValue(r);
        sdisplUb.SetValue(r * (UB_ALIGN / sizeof(int64_t)), sAccum);
        rdisplUb.SetValue(r * (UB_ALIGN / sizeof(int64_t)), rAccum);
        sAccum += sCount;
        rAccum += rCount;
    }
    // 两张表各一次 MTE3 落到 GM 的 displs 区, a2a 模块按槽索引读
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    DataCopyParams displParams = {1U, static_cast<uint16_t>(numRanks_ * UB_ALIGN), 0U, 0U};
    GlobalTensor<int64_t> sdGM;
    sdGM.SetGlobalBuffer((__gm__ int64_t*)sdisplsGM_);
    DataCopyPad(sdGM, sdisplUb, displParams);
    GlobalTensor<int64_t> rdGM;
    rdGM.SetGlobalBuffer((__gm__ int64_t*)rdisplsGM_);
    DataCopyPad(rdGM, rdisplUb, displParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::CrossRankBarrierIssue()
{
    if (isSender_) {
        // sender 核数可能少于 rank 数（控核场景 numSendCores_ < numRanks_）：按 numSendCores_
        // 为 stride 循环覆盖全部目的 rank，否则只有部分 rank 能收到本 rank 的 barrier flag
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            if (dstRank != rankId_) {
                uint64_t handle = GetCommHandle(dstRank, 0U);
                GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, 0U) + rankId_ * STATE_OFFSET; // 标志区固定窗口偏移0
                WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
            }
        }
    }
    if (aivId_ == 0) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagAddr = localWinBase + rankId_ * STATE_OFFSET; // 标志区固定窗口偏移0
        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> flagGM;
        flagGM.SetGlobalBuffer((__gm__ int32_t*)localFlagAddr);
        DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchGradArch35::CrossRankBarrierWait()
{
    if (isSender_) {
        // 与 CrossRankBarrierIssue 对称：drain 掉本核发出的全部 flag 写
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            if (dstRank != rankId_) {
                uint64_t handle = GetCommHandle(dstRank, 0U);
                DrainChecked(handle);
            }
        }
    }
    SyncAll<true>();

    if (isFlagCore_) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagBase = localWinBase; // 标志区固定窗口偏移0
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::InitCorePartition(uint32_t numRecv)
{
    // flag 核(最后一个核)与分区模式无关: 承担跨 rank barrier
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    const bool withSort = (numRecv > 0U);
    const uint32_t sortMin = withSort ? 1U : 0U;
    const uint32_t allToAllMin = 2U; // 1 send + 1 recv(至少)
    // 根据recv的数目看是否需要sort，以及是否总核数大于至少并行所需的核数（3个），否则会退化为串行
    useCorePartition_ = withSort && (totalBlocks_ >= allToAllMin + sortMin);
    uint32_t allToAllBudget;
    if (!useCorePartition_) {
        allToAllBudget = totalBlocks_; // a2a使用全核
    } else {
        uint32_t sortBudget = static_cast<uint32_t>(
            (static_cast<uint64_t>(totalBlocks_) * static_cast<uint64_t>(sortCoreCnt_) + aivNumTiling_ - 1U) /
            aivNumTiling_);
        if (sortBudget > totalBlocks_ - allToAllMin) {
            sortBudget = totalBlocks_ - allToAllMin;
        }
        allToAllBudget = totalBlocks_ - sortBudget;
    }

    allToAllv_.InitJobs(allToAllBudget, aivId_); // 推导 pipelinesPerRank / send-recv job 划分 / 窗口布局
    numSendCores_ = allToAllv_.sendJobCnt_;
    numRecvCores_ = allToAllv_.recvJobCnt_;

    // 在并行策略中判别核身份（recv or send or sort）
    if (useCorePartition_) {
        sortCoreBase_ = numSendCores_ + numRecvCores_;
        numSortCores_ = totalBlocks_ - sortCoreBase_;
        isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
        isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < numSendCores_ + numRecvCores_) || (totalBlocks_ <= 1U);
        isAllToAllvCore_ = isSender_ || isReceiver_;
        isSortCore_ = !isAllToAllvCore_;
        sortCoreIdx_ = aivId_ - sortCoreBase_; // sort 核连续, 除开send和recv后续的就是sort了
    } else {
        // 纯串
        sortCoreBase_ = 0;
        numSortCores_ = withSort ? totalBlocks_ : 0U;
        isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
        isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < numSendCores_ + numRecvCores_) || (totalBlocks_ <= 1U);
        isAllToAllvCore_ = true;
        isSortCore_ = false;
        sortCoreIdx_ = aivId_;
    }
}

__aicore__ inline void EngramFetchGradArch35::InitSortBarrierArea()
{
    if (aivId_ != 0) {
        return;
    }
    LocalTensor<int32_t> zeroLocal = tempBuf_.Get<int32_t>();
    Duplicate<int32_t>(zeroLocal, 0, softBarrierPageInts_);
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> softBarrierGM;
    softBarrierGM.SetGlobalBuffer((__gm__ int32_t*)sortBarrierGM_);
    uint64_t pageInts = static_cast<uint64_t>(softBarrierPageInts_);
    for (uint32_t g = 0; g < SOFT_BARRIER_MAX_GENS; g++) {
        DataCopy(softBarrierGM[g * pageInts], zeroLocal, softBarrierPageInts_);
    }
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::SoftBarrier()
{
    if (!isSortCore_ || numSortCores_ <= 1U) {
        softBarrierGen_++;
        return;
    }
    if (softBarrierGen_ >= SOFT_BARRIER_MAX_GENS) {
        ascendc_assert(false, "soft barrier page overflow: gen=%u, max=%u", softBarrierGen_, SOFT_BARRIER_MAX_GENS);
    }
    uint32_t statusCnt = numSortCores_ * STATE_OFFSET / sizeof(float); // 槽区 float 数(含 32B 对齐尾)
    GM_ADDR pageBase = sortBarrierGM_ + static_cast<uint64_t>(softBarrierGen_) * softBarrierPageInts_ * sizeof(int32_t);
    float minTarget = static_cast<float>(numSortCores_) - 0.5f;
    float maxTarget = static_cast<float>(numSortCores_) + 0.5f;
    AscendC::PipeBarrier<PIPE_ALL>();
    LocalTensor<float> softBarrierLocal = softBarrierScratchBuf_.Get<float>();
    for (uint32_t i = 1U; i < STATE_OFFSET / sizeof(float); i++) {
        softBarrierLocal.SetValue(i, 0.0f);
    }
    softBarrierLocal.SetValue(0, 1.0f);
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<float> mySlotGM;
    mySlotGM.SetGlobalBuffer((__gm__ float*)(pageBase + static_cast<uint64_t>(sortCoreIdx_) * STATE_OFFSET));
    DataCopyParams arriveParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(mySlotGM, softBarrierLocal, arriveParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    GlobalTensor<float> softBarrierGM;
    softBarrierGM.SetGlobalBuffer((__gm__ float*)pageBase);
    SumParams sumParams{1U, statusCnt, statusCnt};
    float flag = 0.0f;
    while ((flag < minTarget) || (flag > maxTarget)) {
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(statusCnt * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> cpPad{false, 0, 0, 0};
        DataCopyPad(softBarrierLocal, softBarrierGM, cpParams, cpPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_V>();
        AscendC::Sum<float>(softBarrierLocal, softBarrierLocal, sumParams); // 原地: 归约结果落 softBarrierLocal[0]
        EngramFetchGradSyncFunc<HardEvent::V_S>();
        flag = softBarrierLocal.GetValue(0);
    }
    softBarrierGen_++;
}

__aicore__ inline void EngramFetchGradArch35::RunSortPhase(uint32_t numRecv)
{
    AscendC::PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    tpipe_->InitBuffer(softBarrierScratchBuf_, numSortCores_ * STATE_OFFSET);
    softBarrierGen_ = 0; // 起始软同步次数为0
    SortLib::SortParams sortParams;
    sortParams.numTileData = sortNumTileData_;
    sortParams.tileCount = (numRecv + sortParams.numTileData - 1U) / sortParams.numTileData;
    sortParams.activeCores = (sortParams.tileCount < numSortCores_) ? sortParams.tileCount : numSortCores_;
    sortParams.tmpUbSize = sortTmpUbSize_;
    sortParams.totalElements = static_cast<int64_t>(numRecv);
    sortParams.isSingleCore = 0;
    sortParams.jobIndex = aivId_ - sortCoreBase_; // sort 子集内的连续编号
    sortParams.totalJobs = numSortCores_;
    EngramSortSoftBarrier softBarrier{this};
    SortLib::SortInvoke<int32_t, int32_t, uint32_t, false, EngramSortSoftBarrier>(
        tpipe_, (__gm__ int32_t*)recvLocalEntryGM_, (__gm__ int32_t*)recvLocalEntryGM_,
        (__gm__ int32_t*)sortCompanionGM_, (__gm__ char*)sortWorkspaceGm_, sortParams, softBarrier);
}

__aicore__ inline void EngramFetchGradArch35::Process()
{
    if ASCEND_IS_AIV {
        GlobalTensor<int32_t> numRecvGM;
        numRecvGM.SetGlobalBuffer((__gm__ int32_t*)numRecvGM_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
        uint32_t numRecv = static_cast<uint32_t>(numRecvGM.GetValue(0));
        // 初始化核预算/核角色/pipelinesPerRank（每rank分核数）/窗口布局
        InitCorePartition(numRecv);

        InitFlagsAndDispls();
        // write/read计数器清零
        allToAllv_.ClearCounters(entryBuf_.Get<int32_t>());
        // 清零sort软同步需要使用的状态区
        InitSortBarrierArea();
        SyncAll<true>();
        // 发送跨rank barrier标志写入
        CrossRankBarrierIssue();
        UnsortGrad();
        // 等待全部rank的barrier标志到齐并清零
        CrossRankBarrierWait();

        if (numRecv == 0) {
            // 纯a2a，无sort
            allToAllv_.SendPhase();
            allToAllv_.RecvPhase();
            allToAllv_.Drain();
            SyncAll<true>();
            uniqueScatter_.WriteNumUniqueZero(numUniqueOutGM_);
            return;
        } else if (useCorePartition_) {
            // 若采用并行策略且为a2a核
            if (isAllToAllvCore_) {
                allToAllv_.SendPhase();
                allToAllv_.RecvPhase();
                allToAllv_.Drain();
            } else {
                // 采用并且策略且为sort核
                RunSortPhase(numRecv);
            }
            // 确保a2a与排序均已完成，为后续scatteradd做准备
            SyncAll<true>();
        } else {
            // 串行
            allToAllv_.SendPhase();
            allToAllv_.RecvPhase();
            allToAllv_.Drain();
            SyncAll<true>();
            RunSortPhase(numRecv);
        }
        // scatteradd阶段
        AscendC::PipeBarrier<PIPE_ALL>();
        tpipe_->Reset();
        tpipe_->InitBuffer(statusBuf_, statusBufBytes_);
        tpipe_->InitBuffer(tempBuf_, tempBufBytes_);
        tpipe_->InitBuffer(indicesBuf_, indicesBufBytes_);
        tpipe_->InitBuffer(entryBuf_, uniqueEntryBytes_);
        tpipe_->InitBuffer(gradBuf_, Mc2Kernel::GRAD_BUF_BYTES);
        if (castBufBytes_ > 0U) {
            tpipe_->InitBuffer(castFp32Buf_, castBufBytes_);
        } else {
            tpipe_->InitBuffer(castFp32Buf_, Mc2Kernel::UB_ALIGN);
        }
        tpipe_->InitBuffer(accumBuf_, accumBufBytes_);
        uniqueScatter_.SetCastBuf(castFp32Buf_);
        uniqueScatter_.SetAccumBuf(accumBuf_);
        uniqueScatter_.SetGradSubBatch(gradSubBatch_);
        uniqueScatter_.SetEntryBufBytes(uniqueEntryBytes_);
        // 均分numRecv -> 每核一段[rawStart, rawEnd) -> 保证一个重复组归一个核 -> 向量计算unique的entry数
        // -> 写回GM，coreStart[核] = 实际起点，segCount[核] = 段内unique数
        uniqueScatter_.CountUniquesParallel(numRecv, recvLocalEntryGM_, coreStartGM_, segCountGM_);
        SyncAll<true>();
        uniqueScatter_.RunScatterCast(numRecv, recvLocalEntryGM_, uniqueLocalEntryOutGM_, numUniqueOutGM_,
                                      gradUniqueOutGM_, recvGradGM_, coreStartGM_, segCountGM_, sortCompanionGM_);
    }
}

#endif // defined(ENABLE_ENGRAM_FETCH_GRAD_KERNEL)

} // namespace Mc2Kernel

#endif
