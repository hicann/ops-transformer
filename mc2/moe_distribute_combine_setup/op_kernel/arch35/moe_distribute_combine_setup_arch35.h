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
 * \file moe_distribute_combine_setup_arch35.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hcomm/hcomm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_combine_setup_base.h"
#include "../moe_distribute_combine_setup_tiling_data.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"

namespace MoeDistributeCombineSetupImpl {

#define TemplateMC2TypeClass typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeFunc ExpandXType, ExpandIdxType

using namespace AscendC;
using namespace Mc2Aclnn;

template <TemplateMC2TypeClass>
class MoeDistributeCombineSetup {
    constexpr static uint8_t BUFFER_NUM = 2;              // 多buf
    constexpr static uint64_t STATE_OFFSET = 512U;        // 状态空间偏移地址
    constexpr static uint32_t STATE_SIZE = 1024U * 1024U; // 1M
    constexpr static uint32_t UB_ALIGN = 32U;             // UB按32字节对齐
    // 每 aiv 控制块大小（win [950KB,1MB) 区内的 D_S/C_S/C flag 步长）
    constexpr static uint64_t STATE_SIZE_PER_CORE = 512U;
    constexpr static uint32_t STATE_COUNT_THRESHOLD = 512U; // moeExpertNumPerRank*epWorldSize状态数阈值
    constexpr static uint64_t COMM_CMD_INFO_BYTES = 64U;    // commCmdInfoOut 每个 (a+rank) 条目 16 * int32
    // 性能优化分界：bs<=16 本卡数据量小，拆半会把一次拷贝拆成两块；bs>16 才走双 buffer
    constexpr static uint32_t BS_PERF_THRESHOLD = 16U;

public:
    __aicore__ inline MoeDistributeCombineSetup(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                GM_ADDR quantExpandX, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling,
                                __gm__ void *mc2CcTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void CurRankComm(const LocalTensor<int32_t> &assistInfoForCombineLocal,
                                       uint32_t curRankExpertNum);
    __aicore__ inline void Communication();
    __aicore__ inline void BuffInit();
    __aicore__ inline void AssistInfoLocalCopy();

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> assistInfoForCombineGlobal_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGM_;
    GM_ADDR expandXGM_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    const MoeDistributeCombineSetupInfo *moeDistributeCombineSetupInfo_{nullptr};
    uint32_t axisMaxBS_{0};
    uint32_t moeSendNum_{0}; // moeExpertPerRankNum * epWorldSize
    uint64_t epDataOffsetOnWin_{0};
    uint64_t epStateOffsetOnWin_{0};
    uint64_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0}; // 当前核处理的起始卡号
    uint32_t endRankId_{0};   // 当前核处理的结束卡号
    uint32_t sendRankNum_{0};
    uint32_t dataState_{0};
    uint64_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t remain_ub_space{0};
    bool isShardExpert_{false};

    TQue<QuePosition::VECIN, 1> assistInfoQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> expertTokenTmpQueue_;
    uint32_t localCopyQueueNum_{1};

    // =============================== Urma =============================== //
    uint32_t aivId_{0};
    uint32_t epRankId_{0};

    GM_ADDR quantExpandXGM_{nullptr}; // 仅用于结束时打印输出
    GM_ADDR statusDataSpaceGM_;
    GM_ADDR statusFlagGM_; // 本卡状态flag源地址，用于远端WriteNbi
    constexpr static uint64_t HCOMM_INIT_SIZE = 512UL;
    constexpr static uint32_t STATUS_FLAG_VALUE = 0x3f800000U; // float 1.0

    LocalTensor<uint8_t> hcommTensor_;

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > epRankId_ ? rankId - 1 : rankId;
        // hcommHandle_ 在 GM(context) 上，读前 DCCI 单 cacheline，避免脏/陈旧 handle
        GlobalTensor<uint64_t> handleGT;
        handleGT.SetGlobalBuffer((__gm__ uint64_t *)&mc2Context_->hcommHandle_[index]);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(handleGT);
        return handleGT(0);
    }

    __aicore__ inline GM_ADDR GetUrmaAddrByRankId(const int32_t rankId, const uint8_t expertLocalId = 0U)
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_ +
                         expertPerSizeOnWin_ * static_cast<uint64_t>(expertLocalId));
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(const int32_t rankId)
    {
        // C 专用状态区，与 D bank 物理隔离；C_S 独立乒乓
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + COMBINE_STATUS_BASE +
                         dataState_ * COMBINE_STATE_BANK_STRIDE);
    }
    // =============================== Urma =============================== //
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR quantExpandX,
    GM_ADDR /* commCmdInfoOut */, GM_ADDR /* workspaceGM */, TPipe *pipe,
    const MoeDistributeCombineSetupTilingData *tilingData, __gm__ void *mc2InitTiling, __gm__ void *mc2CcTiling)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    moeDistributeCombineSetupInfo_ = &(tilingData->moeDistributeCombineSetupInfo);
    mc2Context_ = (__gm__ Mc2MoeContext *)context;
    epRankId_ = mc2Context_->epRankId;

    quantExpandXGM_ = quantExpandX;

    // 获取 win 控制区：本算子只读写 C_S，与 D_S 独立，避免联跑抢乒乓
    GlobalTensor<int32_t> selfDataStatusTensor;
    statusDataSpaceGM_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankId_]);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGM_ + STATE_WIN_OFFSET +
                                                            STATE_SIZE_PER_CORE * static_cast<uint64_t>(aivId_) +
                                                            CTRL_COMBINE_STATE_OFFSET));

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0); // C_S：combine 专用 0/1
    // 本卡 status flag 源（控制块 +64），供远端 WriteNbi；须在 epHcclBuffer_ 注册窗内
    statusFlagGM_ = statusDataSpaceGM_ + STATE_WIN_OFFSET + STATE_SIZE_PER_CORE * static_cast<uint64_t>(aivId_) +
                    CTRL_COMBINE_FLAG_OFFSET;

    GlobalTensor<int32_t> statusFlagTensor;
    statusFlagTensor.SetGlobalBuffer((__gm__ int32_t *)statusFlagGM_);
    statusFlagTensor.SetValue(0, static_cast<int32_t>(STATUS_FLAG_VALUE));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusFlagTensor);

    expandXGM_ = expandX;
    assistInfoForCombineGlobal_.SetGlobalBuffer((__gm__ int32_t *)assistInfoForCombine);

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    axisMaxBS_ = moeDistributeCombineSetupInfo_->globalBs / moeDistributeCombineSetupInfo_->epWorldSize;
    moeSendNum_ = moeDistributeCombineSetupInfo_->epWorldSize * moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    isShardExpert_ = (epRankId_ < moeDistributeCombineSetupInfo_->sharedExpertRankNum); // 当前rank是否为共享专家

    axisHExpandXTypeSize_ = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->h) *
                            static_cast<uint64_t>(sizeof(ExpandXType));              // 一个token占用内存
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * axisHExpandXTypeSize_; // 单个专家可能占用的最大空间

    // C 固定占用数据区后半，区内按 C_S 做 totalWin/4 乒乓
    {
        const uint64_t totalWin = static_cast<uint64_t>(moeDistributeCombineSetupInfo_->totalWinSize);
        winDataSizeOffset_ = (totalWin >> 1) + static_cast<uint64_t>(dataState_) * (totalWin >> 2);
    }
    stateOffset_ = (moeSendNum_ > STATE_COUNT_THRESHOLD) ? (STATE_OFFSET >> 1) : STATE_OFFSET;
    epStateOffsetOnWin_ = static_cast<uint64_t>(epRankId_) * stateOffset_;
    epDataOffsetOnWin_ = static_cast<uint64_t>(epRankId_) *
                         static_cast<uint64_t>(moeDistributeCombineSetupInfo_->moeExpertPerRankNum) *
                         expertPerSizeOnWin_; // 前面rank数据区占用内存地址偏移

    epWindowGM_ = GetUrmaAddrByRankId(epRankId_);
    epStatusSpaceGM_ = GetUrmaStateAddrByRankId(epRankId_);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    OOMCheckAddrRange<ExpandXType>((__gm__ ExpandXType *)(epWindowGM_),
                                   moeDistributeCombineSetupInfo_->totalWinSize / 4U);
    OOMCheckAddrRange<float>((__gm__ float *)(epStatusSpaceGM_), COMBINE_STATE_BANK_STRIDE);
#endif

    SplitCoreCal(); // 分核计算
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::SplitCoreCal()
{
    // 对worldSize按卡分核，得到每个核上处理的卡的数量，保证一个jetty只由一个核维护
    uint32_t coreIdxNew = (aivId_ + epRankId_) % moeDistributeCombineSetupInfo_->aivNum; // 按照卡去进行偏移
    sendRankNum_ = moeDistributeCombineSetupInfo_->epWorldSize / moeDistributeCombineSetupInfo_->aivNum;
    uint32_t remainderRankNum = moeDistributeCombineSetupInfo_->epWorldSize % moeDistributeCombineSetupInfo_->aivNum;
    startRankId_ = sendRankNum_ * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        ++sendRankNum_;
        startRankId_ += coreIdxNew;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::BuffInit()
{
    uint64_t totalUbSize = moeDistributeCombineSetupInfo_->totalUbSize;
    uint32_t assistInfoQueueSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint64_t reservedUbSize = HCOMM_INIT_SIZE + assistInfoQueueSize;
    const uint64_t leftover = totalUbSize - reservedUbSize;
    const uint64_t halfAlign = (leftover / 2U) / UB_ALIGN * UB_ALIGN;
    tpipe_->Reset();
    if (moeDistributeCombineSetupInfo_->bs > BS_PERF_THRESHOLD && halfAlign > 0U) {
        remain_ub_space = halfAlign;
        localCopyQueueNum_ = 2U;
        tpipe_->InitBuffer(assistInfoQueue_, 1, static_cast<uint32_t>(reservedUbSize));
        tpipe_->InitBuffer(expertTokenTmpQueue_, 2, static_cast<uint32_t>(remain_ub_space));
    } else {
        remain_ub_space = leftover;
        localCopyQueueNum_ = 1U;
        tpipe_->InitBuffer(assistInfoQueue_, 1, static_cast<uint32_t>(reservedUbSize));
        tpipe_->InitBuffer(expertTokenTmpQueue_, 1, static_cast<uint32_t>(remain_ub_space));
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::AssistInfoLocalCopy()
{
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.AllocTensor<int32_t>();

    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        // 对于共享专家来说assistInfoForCombine输入维度为epWorldSize个
        epSendCntParams = {1U, moeDistributeCombineSetupInfo_->epWorldSize * static_cast<uint32_t>(sizeof(uint32_t)),
                           0U, 0U, 0U};
    } else {
        epSendCntParams = {1U, moeSendNum_ * static_cast<uint32_t>(sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(assistInfoForCombineLocal, assistInfoForCombineGlobal_, epSendCntParams, copyPadParams);
    assistInfoQueue_.EnQue(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Communication()
{
    // 本卡专家数
    uint32_t curRankExpertNum = (isShardExpert_) ? 1U : moeDistributeCombineSetupInfo_->moeExpertPerRankNum;
    AssistInfoLocalCopy();
    LocalTensor<int32_t> assistInfoForCombineLocal = assistInfoQueue_.DeQue<int32_t>();
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等assistInfoQueue_的GM->Local，后续标量读

    uint32_t assistInfoSize = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<uint8_t> assistStorage = assistInfoForCombineLocal.ReinterpretCast<uint8_t>();
    hcommTensor_ = assistStorage[assistInfoSize];
    int32_t ret = hcomm_.Init(hcommTensor_, static_cast<uint32_t>(HCOMM_INIT_SIZE));
    if (ret != HCOMM_SUCCESS) {
        assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
        return;
    }

    // 数据和状态分离：有token才发数据，每个dstRank始终单独发一次状态
    uint32_t middleRank = epRankId_ < startRankId_ ? startRankId_ : epRankId_;
    for (uint32_t dstRankId = middleRank; dstRankId < endRankId_; ++dstRankId) {
        if (unlikely(dstRankId == epRankId_)) {
            continue;
        }
        // 每 dst 只取一次 channelId（内含单行 DCCI）
        auto channelId = GetCommHandle(dstRankId);
        GM_ADDR dstStatusAddr = GetUrmaStateAddrByRankId(dstRankId) + epStateOffsetOnWin_;
        for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
            uint32_t preCount = 0U;
            uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + dstRankId;
            if (likely(assistInfoIdx > 0U)) {
                // 计算其他卡或专家已经发了多少token
                preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
            }
            const uint32_t curCount = assistInfoForCombineLocal.GetValue(assistInfoIdx);
            // 当前要发送的token数量
            uint32_t curTokenNum = curCount - preCount;
            if (curTokenNum == 0U) {
                continue;
            }
            GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
            GM_ADDR dstAddr = GetUrmaAddrByRankId(dstRankId, expertIdx) + epDataOffsetOnWin_;
            uint64_t lengthU64 = axisHExpandXTypeSize_ * static_cast<uint64_t>(curTokenNum);
            // 数据与状态在同一 Commit 批次里，状态排在最后：任一数据 WQE 失败都会让队列进 error 态
            // 并 flush 掉后面的状态 WQE，表现为对端状态恒为 0。所以这里必须查返回值。
            int32_t dataRet = hcomm_.WriteNbi<false>(channelId, dstAddr, srcAddr, static_cast<uint32_t>(lengthU64));
        }
        // 无论token是否为0，都独立下发状态，避免WriteWithNotify在length=0时丢notify
        hcomm_.WriteNbi<false>(channelId, dstStatusAddr, statusFlagGM_, UB_ALIGN);
        hcomm_.Commit(channelId);
    }

    middleRank = epRankId_ < endRankId_ ? epRankId_ : endRankId_;
    for (uint32_t dstRankId = startRankId_; dstRankId < middleRank; ++dstRankId) {
        if (unlikely(dstRankId == epRankId_)) {
            continue;
        }
        // 每 dst 只取一次 channelId（内含单行 DCCI）
        auto channelId = GetCommHandle(dstRankId);
        GM_ADDR dstStatusAddr = GetUrmaStateAddrByRankId(dstRankId) + epStateOffsetOnWin_;
        for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
            uint32_t preCount = 0U;
            uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + dstRankId;
            if (likely(assistInfoIdx > 0U)) {
                // 计算其他卡或专家已经发了多少token
                preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
            }
            const uint32_t curCount = assistInfoForCombineLocal.GetValue(assistInfoIdx);
            // 当前要发送的token数量
            uint32_t curTokenNum = curCount - preCount;
            if (curTokenNum == 0U) {
                continue;
            }
            GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
            GM_ADDR dstAddr = GetUrmaAddrByRankId(dstRankId, expertIdx) + epDataOffsetOnWin_;
            uint64_t lengthU64 = axisHExpandXTypeSize_ * static_cast<uint64_t>(curTokenNum);
            // 数据与状态在同一 Commit 批次里，状态排在最后：任一数据 WQE 失败都会让队列进 error 态
            // 并 flush 掉后面的状态 WQE，表现为对端状态恒为 0。所以这里必须查返回值。
            int32_t dataRet = hcomm_.WriteNbi<false>(channelId, dstAddr, srcAddr, static_cast<uint32_t>(lengthU64));
        }
        // 无论token是否为0，都独立下发状态，避免WriteWithNotify在length=0时丢notify
        hcomm_.WriteNbi<false>(channelId, dstStatusAddr, statusFlagGM_, UB_ALIGN);
        hcomm_.Commit(channelId);
    }

    if (unlikely(epRankId_ >= startRankId_ && epRankId_ < endRankId_)) {
        // 远端已 Commit，本卡拷贝与 NIC 重叠
        CurRankComm(assistInfoForCombineLocal, curRankExpertNum);
    }

    assistInfoQueue_.FreeTensor<int32_t>(assistInfoForCombineLocal);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::CurRankComm(
    const LocalTensor<int32_t> &assistInfoForCombineLocal, uint32_t curRankExpertNum)
{
    uint32_t curTokenNum = 0;
    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; ++expertIdx) {
        uint32_t preCount = 0U;
        uint32_t assistInfoIdx = expertIdx * moeDistributeCombineSetupInfo_->epWorldSize + epRankId_;
        if (likely(assistInfoIdx > 0U)) {
            // 计算其他卡或专家已经发了多少token
            preCount = assistInfoForCombineLocal.GetValue(assistInfoIdx - 1U);
        }
        // 当前要发送的token数量
        curTokenNum = assistInfoForCombineLocal.GetValue(assistInfoIdx) - preCount;
        if (unlikely(curTokenNum == 0U)) {
            continue;
        }

        GM_ADDR srcAddr = expandXGM_ + static_cast<uint64_t>(preCount) * axisHExpandXTypeSize_;
        GM_ADDR dstAddr = GetUrmaAddrByRankId(epRankId_, expertIdx) + epDataOffsetOnWin_;

        // expandX在GM上，winIn也属于GM，
        // 因此，数据需要GM -> local -> winIn
        GlobalTensor<uint8_t> selfDataSrcTensor;
        GlobalTensor<uint8_t> selfDataDstTensor;
        selfDataSrcTensor.SetGlobalBuffer((__gm__ uint8_t *)srcAddr);
        selfDataDstTensor.SetGlobalBuffer((__gm__ uint8_t *)dstAddr);

        uint64_t sendBytes = static_cast<uint64_t>(curTokenNum) * axisHExpandXTypeSize_;
        DataCopyPadExtParams<uint8_t> padParams{false, 0U, 0U, 0U};
        if (localCopyQueueNum_ > 1U && remain_ub_space > 0U) {
            const uint64_t totalBytes = sendBytes;
            const uint64_t chunkNum = (totalBytes + remain_ub_space - 1UL) / remain_ub_space;
            DataCopyExtParams chunkParams{1U, 0U, 0U, 0U, 0U};
            chunkParams.blockLen = static_cast<uint32_t>((totalBytes > remain_ub_space) ? remain_ub_space : totalBytes);
            LocalTensor<uint8_t> preBuf = expertTokenTmpQueue_.AllocTensor<uint8_t>();
            DataCopyPad(preBuf, selfDataSrcTensor[0], chunkParams, padParams);
            expertTokenTmpQueue_.EnQue(preBuf);
            for (uint64_t i = 0UL; i < chunkNum; ++i) {
                const uint64_t curOff = i * remain_ub_space;
                const uint64_t curRest = totalBytes - curOff;
                const uint32_t curLen = static_cast<uint32_t>((curRest > remain_ub_space) ? remain_ub_space : curRest);
                LocalTensor<uint8_t> curBuf = expertTokenTmpQueue_.DeQue<uint8_t>();
                if (i + 1UL < chunkNum) {
                    const uint64_t nextOff = curOff + remain_ub_space;
                    const uint64_t nextRest = totalBytes - nextOff;
                    chunkParams.blockLen =
                        static_cast<uint32_t>((nextRest > remain_ub_space) ? remain_ub_space : nextRest);
                    LocalTensor<uint8_t> nextBuf = expertTokenTmpQueue_.AllocTensor<uint8_t>();
                    DataCopyPad(nextBuf, selfDataSrcTensor[nextOff], chunkParams, padParams);
                    expertTokenTmpQueue_.EnQue(nextBuf);
                }
                DataCopyExtParams outParams{1U, curLen, 0U, 0U, 0U};
                DataCopyPad(selfDataDstTensor[curOff], curBuf, outParams);
                expertTokenTmpQueue_.FreeTensor<uint8_t>(curBuf);
            }
        } else {
            LocalTensor<uint8_t> expertTokenTmpU8 = expertTokenTmpQueue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(remain_ub_space), 0U, 0U, 0U};

            uint64_t i = 0;
            for (; sendBytes > remain_ub_space; sendBytes -= remain_ub_space) {
                DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[i * remain_ub_space], copyParams, padParams);
                expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
                expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
                DataCopyPad(selfDataDstTensor[i * remain_ub_space], expertTokenTmpU8, copyParams);
                ++i;

                AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }

            copyParams.blockLen = sendBytes;
            DataCopyPad(expertTokenTmpU8, selfDataSrcTensor[i * remain_ub_space], copyParams, padParams);
            expertTokenTmpQueue_.EnQue(expertTokenTmpU8);
            expertTokenTmpU8 = expertTokenTmpQueue_.DeQue<uint8_t>();
            DataCopyPad(selfDataDstTensor[i * remain_ub_space], expertTokenTmpU8, copyParams);

            expertTokenTmpQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
        }
    }

    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 数据拷贝完后才能写状态
    // 向本卡状态区写状态
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(GetUrmaStateAddrByRankId(epRankId_) + epStateOffsetOnWin_));
    selfStatusTensor.SetValue(0, static_cast<int32_t>(STATUS_FLAG_VALUE));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeCombineSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        // 空闲核不参与通信，但仍要走到下面的 SyncAll，否则其他核会等不到它
        if (startRankId_ < moeDistributeCombineSetupInfo_->epWorldSize) {
            BuffInit();
            Communication();
        }
    }
}
} // namespace MoeDistributeCombineSetupImpl

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_ARCH35_H
