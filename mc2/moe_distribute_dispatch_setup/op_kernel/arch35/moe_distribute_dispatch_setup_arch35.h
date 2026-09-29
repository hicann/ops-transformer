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
 * \file moe_distribute_dispatch_setup.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_H
#define MOE_DISTRIBUTE_DISPATCH_SETUP_H

#include "adv_api/hcomm/hcomm.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_setup_tiling.h"
#include "../moe_distribute_dispatch_setup_base.h"
#include "../../../common/op_kernel/quantize_functions.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../moe_distribute_dispatch_setup_common.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

namespace Mc2Kernel {
constexpr uint8_t BUFFER_NUM = 2;       // 多buf
constexpr uint32_t STATE_OFFSET = 512U; // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024U * 1024U;
constexpr uint32_t UB_ALIGN = 32U; // UB按32字节对齐
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t SQE_START_OFFSET = 10U << 20;
constexpr uint32_t WRITE_SQE_SIZE = 64U;
constexpr uint32_t NORMAL_CQE_SIZE = 64U;
constexpr uint32_t CQ_DEPTH_256 =
    256U; // 为cqeBuf申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
constexpr uint32_t UNQUANT = 0;
constexpr uint32_t STATIC_QUANT = 1;
constexpr uint32_t PERTOKEN_DYNAMIC_QUANT = 2;
constexpr uint32_t PERGROUP_DYNAMIC_QUANT = 3;
constexpr uint32_t MX_QUANT = 4;
constexpr uint8_t QUANT_PADDING_VALUE = 0;

constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;

struct BatchWriteItem {
    uint64_t type;
    uint32_t res1[5];
    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t res2[4];
};

#define TemplateMC2TypeClass typename XType, typename YOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateMC2TypeFunc XType, YOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace Mc2Aclnn;

template <TemplateMC2TypeClass>
class MoeDistributeDispatchSetup {
public:
    __aicore__ inline MoeDistributeDispatchSetup(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR YOut, GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM,
                                TPipe *pipe, const MoeDistributeDispatchSetupTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CurRankComm(LocalTensor<int32_t> &statusTensor_, uint32_t &moeStartToken);
    __aicore__ inline void CalMoeStartTokenNum(uint32_t startRankId, LocalTensor<int32_t> &statusTensor,
                                               uint32_t &tokenNum);
    __aicore__ inline void TokenScatter();
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void ActiveMaskCalCnt();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess(LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal,
                                        uint32_t expertIndex);
    __aicore__ inline void QuantStatic(LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal,
                                       uint32_t expertIndex);
    __aicore__ inline void QuantDynamicPerToken(LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal,
                                                uint32_t expertIndex);
    __aicore__ inline void QuantDynamicPerGroup(LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal,
                                                uint32_t expertIndex);
    __aicore__ inline void QuantDynamicMxFp8(LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal);
    __aicore__ inline void SetStatus();
    __aicore__ inline void QuantInit();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt_, uint32_t curUseAivNum, uint32_t &startTokenId_,
                                       uint32_t &endTokenId_, uint32_t &sendTokenNum_, bool isFront = true);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);

    __aicore__ inline uint32_t MIN(uint32_t x1, uint32_t x2)
    {
        return (x1 < x2) ? x1 : x2;
    }

    __aicore__ inline int32_t ReduceSumWorkNeedSize(int32_t calCnt)
    {
        int32_t elementsPerBlock = 32 / static_cast<int>(sizeof(int32_t));
        int32_t iter1OutputCount = calCnt;
        int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        return iter1AlignEnd;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<YOutType> expandXOutGMTensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;

    LocalTensor<YOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<YOutType> xOutTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<float> expertIdsF32LT_;
    LocalTensor<int32_t> indexLT_;
    LocalTensor<uint32_t> sortedIndex1LT_;
    LocalTensor<float> sortedOutF32LT_;
    TBuf<> tempBuf_;
    TBuf<> sortedBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> batchItemInfoBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> rowMaxBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_; // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TQue<QuePosition::VECIN, 1> smoothScalesInQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tmpTokenQueue_;

    GM_ADDR yOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR commCmdInfoOutGM_{nullptr}; // 仅用于结束时打印输出
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR statusDataSpaceGM_;

    DataCopyExtParams dataCopyParamsFloat_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t rankNumPerSharedExpert_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t scalesCount_{0};
    uint64_t totalWinSize_{0};
    uint32_t startStatusIndex_{0};
    uint32_t axisHCommu{0};
    uint32_t sdmaUsedStreamPerCore_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t curSendCnt_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t activeMaskBsCnt_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    int32_t expertIdsCnt_{0};
    int32_t expertIdsActCnt_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t recvWinBlockNum_; // 接收Win区块数
    uint64_t remain_ub_space{0};
    bool isSendShared_{false};
    bool isInputActiveMaskFlag_ = false;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams xCopyParams_;
    DataCopyExtParams hCommuCopyOutParams_;
    DataCopyParams scalesInParams_;
    DataCopyPadParams scalesPadParams_;

    // URMA通信新增函数
    __aicore__ inline void Communication();
    __aicore__ inline void InitCommBuffer();

    uint32_t sortNum_;
    uint32_t sortRepeat_;
    uint64_t totalUbSize_;

    // =============================== Urma =============================== //
    TBuf<> hcommBuf_;
    LocalTensor<uint8_t> hcommTensor_;

    constexpr static uint64_t HCOMM_INIT_SIZE = 512UL;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文
    __gm__ Mc2MoeContext *mc2Context_{nullptr};

    __aicore__ inline uint64_t GetCommHandle(uint32_t rankId)
    {
        uint32_t index = rankId > epRankId_ ? rankId - 1 : rankId;
        // hcommHandle_ 在 GM(context) 上，读前 DCCI 单 cacheline，避免脏/陈旧 handle
        GlobalTensor<uint64_t> handleGT;
        handleGT.SetGlobalBuffer((__gm__ uint64_t *)&mc2Context_->hcommHandle_[index]);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(handleGT);
        return handleGT(0);
    }

    __aicore__ inline GM_ADDR GetUrmaAddrByRankId(const int32_t rankId)
    {
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + STATE_SIZE + winDataSizeOffset_);
    }

    __aicore__ inline GM_ADDR GetUrmaStateAddrByRankId(const int32_t rankId)
    {
        // D 专用状态区 [0,512KB)，按 D_S 选 bank；与 C [512KB,768KB) 隔离
        return (GM_ADDR)(mc2Context_->epHcclBuffer_[rankId] + dataState_ * WIN_STATE_OFFSET);
    }
    // =============================== Urma =============================== //
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR YOut,
    GM_ADDR expandIdxOut, GM_ADDR commCmdInfoOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchSetupTilingData *tilingData)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    aivNum_ = tilingData->moeDistributeDispatchSetupInfo.aivNum;
    totalUbSize_ = tilingData->moeDistributeDispatchSetupInfo.totalUbSize;
    if (aivId_ >= aivNum_) {
        return;
    }
    mc2Context_ = (__gm__ Mc2MoeContext *)context;
    epRankId_ = mc2Context_->epRankId;

    GlobalTensor<int32_t> selfDataStatusTensor;
    statusDataSpaceGM_ = (GM_ADDR)(mc2Context_->epHcclBuffer_[epRankId_]);
    // 只读写 D_S（控制块 +0），与 C_S 独立
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGM_ + STATE_WIN_OFFSET +
                                                            aivId_ * WIN_ADDR_ALIGN + CTRL_DISPATCH_STATE_OFFSET));
    axisBS_ = tilingData->moeDistributeDispatchSetupInfo.bs;
    axisH_ = tilingData->moeDistributeDispatchSetupInfo.h;
    epWorldSize_ = tilingData->moeDistributeDispatchSetupInfo.epWorldSize;
    axisMaxBS_ = tilingData->moeDistributeDispatchSetupInfo.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData->moeDistributeDispatchSetupInfo.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchSetupInfo.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchSetupInfo.sharedExpertRankNum;
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    isInputActiveMaskFlag_ = tilingData->moeDistributeDispatchSetupInfo.isActiveMask;
    axisK_ = tilingData->moeDistributeDispatchSetupInfo.k;
    scalesCount_ = tilingData->moeDistributeDispatchSetupInfo.scalesCount;
    totalWinSize_ = tilingData->moeDistributeDispatchSetupInfo.totalWinSize;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ YOutType *)YOut);
    if constexpr (IsSmoothScaleExist) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    }
    yOutGM_ = YOut;
    expandIdxOutGM_ = expandIdxOut;
    commCmdInfoOutGM_ = commCmdInfoOut;

    hOutSize_ = axisH_ * sizeof(YOutType);
    QuantInit();
    hAlignWinSize_ = Ceil(hOutSizeAlign_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN; // win区token起始地址对齐512
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(YOutType);
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertPerSizeOnWin_ = axisMaxBS_ * hAlignWinSize_;
    expertIdsCnt_ = axisBS_ * axisK_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    stateOffset_ = ((recvWinBlockNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    rscvStatusNum_ = recvWinBlockNum_;
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_; // 每个aiv需要处理的专家数
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_; // + sharedExpertRankNum_, 每个aiv发送的
    if (aivId_ < remainderRankNum_) {                  // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }
    uint32_t statusBufCntAlign = Ceil(recvWinBlockNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign * UB_ALIGN);
    statusTensor_ = statusBuf_.Get<int32_t>(); // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8); // 8 = UB_ALIGN / sizeof(int32_t)
    statusSpaceGm_ = GetUrmaStateAddrByRankId(epRankId_);
    uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为flag
    PipeBarrier<PIPE_V>();
    // 0x40000000 是 float 的 2.0；dispatch_teardown 的 sumTarget_ 取 1.0，
    // 靠比较 compareTarget * 2 来匹配，两侧改动必须同步
    Duplicate<int32_t>(statusTensor_, 0x40000000, mask, statusBufCntAlign / 8, 1, 8);

    // D 固定占用数据区前半，区内按 D_S 做 totalWin/4 乒乓
    winDataSizeOffset_ = dataState_ * (tilingData->moeDistributeDispatchSetupInfo.totalWinSize / 4);
    windowGM_ = GetUrmaAddrByRankId(epRankId_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    hOutAlignUbSize_ = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN;
    expertIdsCnt_ = axisBS_ * axisK_;
    if constexpr (QuantMode > UNQUANT) {
        tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hAlignSize_);       // 14K * 2
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7K * 2 + 32 + 6
        uint32_t hFp32Size = axisH_ * sizeof(float);
        tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size);
        floatLocalTemp_ = receiveDataCastFloatBuf_.Get<float>();
        tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
        dstExpBuf_ = receiveDataCastFloatBuf_; // 内存复用 // 这里直接用hFp32Size申请空间有风险
        subExpBuf_ = smoothScalesBuf_;         // 内存复用
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
    } else {
        uint32_t expertIdxSize = expertIdsCnt_ * sizeof(int32_t);
        tpipe_->InitBuffer(dstExpBuf_, expertIdxSize);             // BS * K * 4 = 32K
        tpipe_->InitBuffer(subExpBuf_, expertIdxSize);             // BS * K * 4 = 32K
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_); // 7k*2 + 32 + 6
    }
    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    sortNum_ = Align<uint32_t, uint32_t>(expertIdsCnt_, 32);
    sortRepeat_ = Ceil<uint32_t, uint32_t>(expertIdsCnt_, 32);
    uint32_t expertIdsSize = axisBS_ * axisK_ * sizeof(int32_t); // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(expertCountBuf_, expertIdsSize); // BS * K * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();
    tpipe_->InitBuffer(workLocalBuf_, sortNum_ * sizeof(float));
    expertIdsF32LT_ = workLocalBuf_.Get<float>();
    tpipe_->InitBuffer(workLocalBuf_, sortNum_ * sizeof(int32_t));
    indexLT_ = workLocalBuf_.Get<int32_t>();
    tpipe_->InitBuffer(tempBuf_, sortNum_ * sizeof(int32_t) * 2);
    tpipe_->InitBuffer(sortedBuf_, sortNum_ * sizeof(int32_t) * 2);
    tpipe_->InitBuffer(workLocalBuf_, sortNum_ * sizeof(uint32_t));
    sortedIndex1LT_ = workLocalBuf_.Get<uint32_t>();
    tpipe_->InitBuffer(workLocalBuf_, sortNum_ * sizeof(float));
    sortedOutF32LT_ = workLocalBuf_.Get<float>();
    // reduceSum计算所需的Tensor空间，取最大统一前面申请
    int32_t reduceSumWorkNeedSize = ReduceSumWorkNeedSize(expertIdsCnt_);
    tpipe_->InitBuffer(workLocalBuf_, reduceSumWorkNeedSize * sizeof(int32_t));
    workLocalTensor_ = workLocalBuf_.Get<float>();

    axisHCommu = hOutSizeAlign_ / sizeof(YOutType); // 有效搬运长度
    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    hCommuCopyOutParams_ = {1U, static_cast<uint32_t>(axisHCommu * sizeof(YOutType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(YOutType)), 0U, 0U, 0U};
    scalesInParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(float)), 0U, 0U};
    scalesPadParams_ = {false, 0, 0, 0};
    activeMaskBsCnt_ = axisBS_;
    if (isInputActiveMaskFlag_) {
        ActiveMaskCalCnt();
    }

    // 发送数据分核
    isSendShared_ = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared_) {
        curSendCnt_ = activeMaskBsCnt_ * sharedExpertNum_;
        SplitToCore(curSendCnt_, sharedUsedAivNum_, startTokenId_, endTokenId_, sendTokenNum_, false);
    } else {
        curSendCnt_ = activeMaskBsCnt_ * axisK_;
        SplitToCore(curSendCnt_, moeUsedAivNum_, startTokenId_, endTokenId_, sendTokenNum_);
    }
    // 发送状态分核
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantInit()
{
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN;           // scale起始放置偏移
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN; // 用于搬入token数据xInQueue_大小申请
    if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        hOutSizeAlign_ += sizeof(float);
    } else if constexpr (QuantMode == MX_QUANT) {
        hOutSizeAlign_ = Align256(axisH_) * sizeof(YOutType);
        hAlignSize_ = Align128(axisH_) * sizeof(XType); // MX量化计算scale时每次搬入128个数据
        hOutSizeAlign_ += Align2(Ceil32(axisH_));
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        hOutSizeAlign_ = Align128(axisH_) * sizeof(YOutType);
        hAlignSize_ = Align128(axisH_) * sizeof(XType); // PERGROUP量化计算scale时每次搬入128个数据
        hOutSizeAlign_ += Ceil128(axisH_) * sizeof(float);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SendToSharedExpert()
{
    if (startTokenId_ >= curSendCnt_) {
        return;
    }
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_; // 计算目的共享专家卡在其所在共享专家组的id
    GlobalTensor<YOutType> yOutGMTensor;
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    for (uint32_t virtualTokenIndex = startTokenId_; virtualTokenIndex < endTokenId_; ++virtualTokenIndex) {
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        yOutGMTensor.SetGlobalBuffer((__gm__ YOutType *)(yOutGM_ + axisBS_ * axisK_ * hAlignWinSize_));
        uint32_t tokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        if constexpr (QuantMode > UNQUANT) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            LocalTensor<uint8_t> singleByteTok = xInTensor_.template ReinterpretCast<uint8_t>();
            // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
            if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT)) {
                Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
                SyncFunc<HardEvent::V_MTE2>();
            }
            DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<YOutType>();
            QuantProcess(xOutTensor_, xInTensor_, toSharedExpertIndex);
            xOutQueue_.EnQue(xOutTensor_);
            xInQueue_.FreeTensor<XType>(xInTensor_);
            xOutTensor_ = xOutQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[virtualTokenIndex * hAlignWinCnt_], xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<YOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[virtualTokenIndex * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<YOutType>(xTmpTensor_);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId,
                                                                                              int32_t calCnt,
                                                                                              int32_t &curExpertCnt)
{
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt > curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // 由于是后面的核作为发送的共享专家，因此需要换算
    }
    startTokenId = newAivId * sendTokenNum;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SendToMoeExpert()
{
    if (startTokenId_ >= curSendCnt_) {
        return;
    }
    GlobalTensor<YOutType> dstWinGMTensor;
    GlobalTensor<YOutType> yOutGMTensor;
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams batchItemCopyOutParams{1U, sizeof(BatchWriteItem), 0U, 0U, 0U};
    for (int32_t tokenIndex = startTokenId_; tokenIndex < endTokenId_; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor_(tokenIndex);
        int32_t curExpertCnt = 0;
        if (likely(tokenIndex > 0)) {
            CalTokenSendExpertCnt(dstExpertId, tokenIndex, curExpertCnt);
        }
        expertCountTensor_(tokenIndex - startTokenId_) = curExpertCnt;
        yOutGMTensor.SetGlobalBuffer((__gm__ YOutType *)(yOutGM_));
        if constexpr (QuantMode > UNQUANT) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            LocalTensor<uint8_t> singleByteTok = xInTensor_.template ReinterpretCast<uint8_t>();
            // 由于MX以及PERGROUP量化在计算scales时每次搬入256字节数据，所以在token搬入前需要对空间填0，避免引入脏数据
            if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT)) {
                Duplicate(singleByteTok, QUANT_PADDING_VALUE, Align128(axisH_) * sizeof(XType));
                SyncFunc<HardEvent::V_MTE2>();
            }
            DataCopyPad(xInTensor_, xGMTensor_[sortedIndex1LT_(tokenIndex) / axisK_ * axisH_], xCopyParams_,
                        copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<YOutType>();
            QuantProcess(xOutTensor_, xInTensor_, dstExpertId + sharedExpertNum_);
            xOutQueue_.EnQue(xOutTensor_);
            xInQueue_.FreeTensor<XType>(xInTensor_);
            xOutTensor_ = xOutQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[tokenIndex * hAlignWinCnt_], xOutTensor_, hCommuCopyOutParams_);
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<YOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[sortedIndex1LT_(tokenIndex) / axisK_ * axisH_], xCopyParams_,
                        copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<YOutType>();
            DataCopyPad(yOutGMTensor[tokenIndex * hAlignWinCnt_], xTmpTensor_, hCommuCopyOutParams_);
            xQueue_.FreeTensor<YOutType>(xTmpTensor_);
        }
    }
    GlobalTensor<int32_t> expandIdxGMTensor;
    expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOutGM_ + startTokenId_ * sizeof(int32_t)));
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(sendTokenNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(expandIdxGMTensor, expertCountTensor_, expertIdsCntParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::ActiveMaskCalCnt()
{
    // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<half> tempTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> xActiveMaskTensor;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    xActiveMaskTensor = dstExpBuf_.Get<bool>();
    tempTensor = subExpBuf_.Get<half>();
    sumOutTensor = expertIdsBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGMTensor_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

/*
共享专家卡：所有核用于给moe专家发送数据
moe专家卡：部分核用于给共享专家发送数据，部分核用于给moe专家发送数据
*/
template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::AlltoAllDispatch()
{
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    expertIdsActCnt_ = activeMaskBsCnt_ * axisK_;

    // 后面的核向共享专家发数据
    if (isSendShared_) {
        SendToSharedExpert();
    }

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (isSendShared_) { // 用于send共享专家数据的核，也需要搬运expertIds，后续会重新分核写状态位置，该核可能用于写moe专家flag
        return;
    }
    // 把 token 整理到 YOut 临时区
    SendToMoeExpert();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::SetStatus()
{
    if (activeMaskBsCnt_ > 0) { // 需要发送的再算
        for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
            int32_t curExpertCnt = 0;
            int32_t cntPosIndex =
                curExpertId * 8 + 1; // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
            if (curExpertId <
                sharedExpertRankNum_) { // 当前处理专家为共享专家 shared:Cnt -> win -> LocalCopy(moe+share)
                if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                    curExpertCnt = activeMaskBsCnt_;
                }
            } else { // 当前处理卡为moe专家卡
                int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
                CalTokenSendExpertCnt(curMoeExpertId, expertIdsActCnt_, curExpertCnt);
            }
            statusTensor_(cntPosIndex) = curExpertCnt;
        }
    }
    GlobalTensor<int32_t> windowExpGMTensor;
    // 状态区拷贝到windowOut
    windowExpGMTensor.SetGlobalBuffer(
        (__gm__ int32_t *)(statusDataSpaceGM_ + DISPATCH_STAGING_OFFSET + startExpertId_ * 8 * sizeof(int32_t)));
    DataCopy<int32_t>(windowExpGMTensor, statusTensor_[startExpertId_ * 8],
                      sendExpertNum_ * 8); // 8时数据大小，按32对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;       // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL; // 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {
        // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantProcess(LocalTensor<YOutType> &outLocal,
                                                                                     LocalTensor<XType> &inLocal,
                                                                                     uint32_t expertIndex)
{
    if constexpr (QuantMode == STATIC_QUANT) {
        QuantStatic(outLocal, inLocal, expertIndex);
    } else if constexpr (QuantMode == PERTOKEN_DYNAMIC_QUANT) {
        QuantDynamicPerToken(outLocal, inLocal, expertIndex);
    } else if constexpr (QuantMode == MX_QUANT) {
        QuantDynamicMxFp8(outLocal, inLocal);
    } else if constexpr (QuantMode == PERGROUP_DYNAMIC_QUANT) {
        QuantDynamicPerGroup(outLocal, inLocal, expertIndex);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantStatic(LocalTensor<YOutType> &outLocal,
                                                                                    LocalTensor<XType> &inLocal,
                                                                                    uint32_t expertIndex)
{
    Cast(floatLocalTemp_, inLocal, RoundMode::CAST_NONE, axisH_);
    if constexpr (Std::IsSame<YOutType, int8_t>::value) {
        if (scalesCount_ == 1) { // 所有专家共享scales，维度为(1,)
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGMTensor_);
            float scaleVal = scalesGMTensor_(0);
            Muls(floatLocalTemp_, floatLocalTemp_, scaleVal, axisH_);
        } else if (scalesCount_ == axisH_) { // 所有专家共享scales，维度为(h,)
            DataCopyPad(smoothScalesTensor_, scalesGMTensor_, scalesInParams_, scalesPadParams_);
        } else { // 所有专家不共享scales
            DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesInParams_, scalesPadParams_);
        }
        if (scalesCount_ != 1) {
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Mul(floatLocalTemp_, floatLocalTemp_, smoothScalesTensor_, axisH_);
        }

        LocalTensor<int32_t> tokenI32LT = floatLocalTemp_.ReinterpretCast<int32_t>();
        Cast(tokenI32LT, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
        LocalTensor<half> tokenF16LT = floatLocalTemp_.ReinterpretCast<half>();
        Cast(tokenF16LT, tokenI32LT, RoundMode::CAST_ROUND, axisH_);
        Cast(outLocal, tokenF16LT, RoundMode::CAST_TRUNC, axisH_);
    } else if constexpr (Std::IsSame<YOutType, hifloat8_t>::value) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(scalesGMTensor_);
        float scaleVal = scalesGMTensor_(0);
        Muls(floatLocalTemp_, floatLocalTemp_, scaleVal, axisH_);
        Cast(outLocal, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantDynamicPerToken(
    LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal, uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    float maxVal = INT8_MAX_VALUE; // 获取输出类型的最大值（AscendC未提供相关接口）
    if constexpr (Std::IsSame<YOutType, fp8_e5m2_t>::value) {
        maxVal = FP8_E5M2_MAX_VALUE;
    } else if constexpr (Std::IsSame<YOutType, fp8_e4m3fn_t>::value) {
        maxVal = FP8_E4M3_MAX_VALUE;
    }

    Cast(floatLocalTemp_, inLocal, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        SyncFunc<AscendC::HardEvent::V_MTE2>(); // ub复用，循环同步
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        Mul(floatLocalTemp_, floatLocalTemp_, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
    Abs(floatLocalAbsTemp, floatLocalTemp_, axisH_);
    PipeBarrier<PIPE_V>();
    ReduceMaxInplace(floatLocalAbsTemp, axisH_); // 获取最大值
    SyncFunc<AscendC::HardEvent::V_S>();
    dynamicScale = maxVal / floatLocalAbsTemp.GetValue(0);
    Muls(floatLocalTemp_, floatLocalTemp_, dynamicScale, axisH_);
    PipeBarrier<PIPE_V>();
    if constexpr (Std::IsSame<YOutType, int8_t>::value) {
        LocalTensor<half> halfLocalTemp = floatLocalTemp_.ReinterpretCast<half>();
        LocalTensor<int32_t> int32LocalTemp = floatLocalTemp_.ReinterpretCast<int32_t>();
        Cast(int32LocalTemp, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
        Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
        Cast(outLocal, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    } else if constexpr (Std::IsSame<YOutType, fp8_e4m3fn_t>::value || Std::IsSame<YOutType, fp8_e5m2_t>::value) {
        Cast(outLocal, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
    }

    LocalTensor<float> tokenF32Tmp = outLocal.template ReinterpretCast<float>();
    tokenF32Tmp.SetValue((Ceil(axisH_, UB_ALIGN) * UB_ALIGN) / sizeof(float),
                         float(1.0) / dynamicScale); // int8->float32
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantDynamicPerGroup(
    LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal, uint32_t expertIndex)
{
    if constexpr (Std::IsSame<YOutType, fp8_e4m3fn_t>::value || Std::IsSame<YOutType, fp8_e5m2_t>::value) {
        if constexpr (IsSmoothScaleExist) { // 平滑系数
            DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesInParams_, scalesPadParams_);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
        }

        __ubuf__ XType *srcAddr = (__ubuf__ XType *)inLocal.GetPhyAddr();
        __ubuf__ float *smoothLocalAddr = (__ubuf__ float *)smoothScalesTensor_.GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
        __ubuf__ float *scaleOutLocalAddr = (__ubuf__ float *)outLocal[Align128<uint32_t>(axisH_)].GetPhyAddr();

        Quant::ComputePerTileDynamic<XType, YOutType, AscendC::RoundMode::CAST_RINT, IsSmoothScaleExist>(
            srcAddr, smoothLocalAddr, scaleOutLocalAddr, outLocalAddr, axisH_);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::QuantDynamicMxFp8(
    LocalTensor<YOutType> &outLocal, LocalTensor<XType> &inLocal)
{
    if constexpr (Std::IsSame<YOutType, fp8_e4m3fn_t>::value || Std::IsSame<YOutType, fp8_e5m2_t>::value) {
        uint32_t mxScaleNum = Align2(Ceil32(axisH_));
        __ubuf__ XType *srcAddr = (__ubuf__ XType *)inLocal.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)floatLocalTemp_.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleLocalAddr = (__ubuf__ uint16_t *)floatLocalTemp_[Align32(mxScaleNum)].GetPhyAddr();
        __ubuf__ int8_t *outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleLocalAddr = (__ubuf__ uint16_t *)outLocal[Align256<uint32_t>(axisH_)].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, axisH_);                                           // 计算最大Exp
        Quant::ComputeScale<YOutType>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum); // 计算scales并填充
        Quant::ComputeFp8Data<XType, YOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleLocalAddr, outLocalAddr, axisH_); // 计算量化后的expandx并填充
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::InitCommBuffer()
{
    tpipe_->Reset();
    uint32_t statusBufCntAlign = Ceil(recvWinBlockNum_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    tpipe_->InitBuffer(statusBuf_, statusBufCntAlign * UB_ALIGN);
    uint32_t sqInfoSize = Ceil(sizeof(HcclAiRMAWQ), UB_ALIGN) * UB_ALIGN;
    uint32_t cqInfoSize = Ceil(sizeof(HcclAiRMACQ), UB_ALIGN) * UB_ALIGN;
    remain_ub_space = (totalUbSize_ - statusBufCntAlign * UB_ALIGN - sqInfoSize - cqInfoSize - 6 * UB_ALIGN -
                       UB_ALIGN * CQ_DEPTH_256 - WRITE_SQE_SIZE - WRITE_SQE_SIZE * moeExpertNumPerRank_ * 2) /
                      2;
    tpipe_->InitBuffer(tmpTokenQueue_, 2, remain_ub_space); // double buffer

    // Urma
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::TokenScatter()
{
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    constexpr float MAX_FP32 = 2147483647;
    Duplicate(expertIdsF32LT_, MAX_FP32, sortNum_);
    CreateVecIndex(indexLT_, 0, sortNum_);
    SyncFunc<HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    Cast(expertIdsF32LT_, expertIdsTensor_, RoundMode::CAST_RINT, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    Muls(expertIdsF32LT_, expertIdsF32LT_, (float)-1, sortNum_);
    LocalTensor<float> concatLocal;
    LocalTensor<float> tempTensor = tempBuf_.Get<float>(GetSortLen<float>(sortNum_));
    PipeBarrier<PIPE_V>();
    Concat(concatLocal, expertIdsF32LT_, tempTensor, sortRepeat_);
    LocalTensor<uint32_t> indexU32LT = indexLT_.ReinterpretCast<uint32_t>();
    LocalTensor<float> sortedLocal = sortedBuf_.Get<float>(GetSortLen<float>(sortNum_));
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedLocal, concatLocal, indexU32LT, tempTensor, sortRepeat_);
    PipeBarrier<PIPE_V>();
    Extract(sortedOutF32LT_, sortedIndex1LT_, sortedLocal, sortRepeat_);
    SyncFunc<HardEvent::V_MTE2>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::CalMoeStartTokenNum(
    uint32_t startRankId, LocalTensor<int32_t> &statusTensor, uint32_t &tokenNum)
{
    if (startRankId <= sharedExpertRankNum_) {
        return;
    }
    // statusTensor 上一段连续专家的 token 数，调用前已 MTE2_S，标量前缀和即可。
    const uint32_t preExpertCnt = (startRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
    uint32_t preTokenSum = 0U;
    for (uint32_t i = 0U; i < preExpertCnt; ++i) {
        preTokenSum += static_cast<uint32_t>(statusTensor((sharedExpertRankNum_ + i) * 8 + 1));
    }
    tokenNum += preTokenSum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::CurRankComm(LocalTensor<int32_t> &statusTensor,
                                                                                    uint32_t &moeStartToken)
{
    uint32_t preExpertNum = sharedExpertRankNum_ + (epRankId_ - sharedExpertRankNum_) * moeExpertNumPerRank_;
    uint32_t calCnt = 0;
    for (uint32_t expertIdx = 0; expertIdx < moeExpertNumPerRank_; expertIdx++) {
        uint32_t tokenNum = static_cast<uint32_t>(statusTensor_((preExpertNum + expertIdx) * 8 + 1));
        uint64_t sendBytes = static_cast<uint64_t>(tokenNum) * hAlignWinSize_;
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(remain_ub_space), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> copyPadExtParams{false, 0U, 0U, 0U};
        GlobalTensor<uint8_t> srcAddr_T;
        GlobalTensor<uint8_t> dstAddr_T;
        LocalTensor<uint8_t> expertTokenTmpU8 = tmpTokenQueue_.AllocTensor<uint8_t>();

        srcAddr_T.SetGlobalBuffer((__gm__ uint8_t *)(yOutGM_ + (moeStartToken + calCnt) * hAlignWinSize_));
        dstAddr_T.SetGlobalBuffer(
            (__gm__ uint8_t *)(GetUrmaAddrByRankId(epRankId_) +
                               (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + expertIdx))));
        uint64_t i = 0;
        for (; sendBytes > remain_ub_space; sendBytes -= remain_ub_space) {
            DataCopyPad(expertTokenTmpU8, srcAddr_T[i * remain_ub_space], copyParams, copyPadExtParams);
            tmpTokenQueue_.EnQue(expertTokenTmpU8);
            expertTokenTmpU8 = tmpTokenQueue_.DeQue<uint8_t>();
            DataCopyPad(dstAddr_T[i * remain_ub_space], expertTokenTmpU8, copyParams);
            ++i;
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        }
        copyParams.blockLen = sendBytes;
        DataCopyPad(expertTokenTmpU8, srcAddr_T[i * remain_ub_space], copyParams, copyPadExtParams);
        tmpTokenQueue_.EnQue(expertTokenTmpU8);
        expertTokenTmpU8 = tmpTokenQueue_.DeQue<uint8_t>();
        DataCopyPad(dstAddr_T[i * remain_ub_space], expertTokenTmpU8, copyParams);
        // 发状态
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        GlobalTensor<int32_t> dstAddrStatus;
        dstAddrStatus.SetGlobalBuffer((__gm__ int32_t *)(GetUrmaStateAddrByRankId(epRankId_) +
                                                         (epRankId_ + expertIdx * epWorldSize_) * stateOffset_));
        DataCopy<int32_t>(dstAddrStatus, statusTensor_[(expertIdx + preExpertNum) * 8], 8);
        calCnt += tokenNum;
        tmpTokenQueue_.FreeTensor<uint8_t>(expertTokenTmpU8);
    }
    moeStartToken += calCnt;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::Communication()
{
    // 按卡分核，并与 combine_setup 保持同一旋转规则，保证同一 hcommHandle/jetty 始终由同一 AIV 维护
    uint32_t coreIdxNew = (aivId_ + epRankId_) % aivNum_;
    uint32_t sendRankNum = epWorldSize_ / aivNum_;
    uint32_t remainderRankNum = epWorldSize_ % aivNum_;
    uint32_t startRankId = sendRankNum * coreIdxNew;
    if (coreIdxNew < remainderRankNum) {
        ++sendRankNum;
        startRankId += coreIdxNew;
    } else {
        startRankId += remainderRankNum;
    }
    uint32_t endRankId = startRankId + sendRankNum;
    // 空闲核跳过 hcomm.Init / TPipe::Reset：通信并行度上限是 epWorldSize
    if (sendRankNum == 0 || startRankId >= epWorldSize_) {
        return;
    }
    InitCommBuffer();

    GlobalTensor<int32_t> windowExpGMTensor;
    windowExpGMTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGM_ + DISPATCH_STAGING_OFFSET));
    uint32_t len = sharedExpertRankNum_;
    if (endRankId > sharedExpertRankNum_) {
        len += (endRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
    }

    DataCopy<int32_t>(statusTensor_, windowExpGMTensor, len * 8); // 8时数据大小，按32对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint32_t moeStartToken = 0; // 当前核处理的第一张moe卡之前的token数，根据该值计算起始偏移地址
    CalMoeStartTokenNum(startRankId, statusTensor_, moeStartToken);

    uint32_t middleRank = epRankId_ < startRankId ? startRankId : epRankId_;
    for (uint32_t dstRankId = middleRank; dstRankId < endRankId; ++dstRankId) {
        if (dstRankId < sharedExpertRankNum_) {
            continue;
        }
        // 本卡通信：仅搬运数据
        if (dstRankId == epRankId_) {
            CurRankComm(statusTensor_, moeStartToken);
            continue;
        }

        // 给共享专家卡发数据和状态，每张共享专家卡上只会有一个共享专家
        if (dstRankId < sharedExpertRankNum_) {
        }
        // 给moe专家卡发数据和状态，每张moe专家卡上有moeExpertNumPerRank_个moe专家
        else {
            // 非本卡通信：发完 epRankId_-->dstExpertId 的所有token后，统一敲doorbell
            auto channelId = GetCommHandle(dstRankId);
            uint32_t preExpertNum = sharedExpertRankNum_ + (dstRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
            for (uint32_t dstExpertId = 0; dstExpertId < moeExpertNumPerRank_; ++dstExpertId) {
                // 当前rank --> dstExpertId 的所有 token 数量
                uint32_t tokenNum = statusTensor_((preExpertNum + dstExpertId) * 8 + 1);

                // 发送token
                if (tokenNum > 0) {
                    // 起始地址应该为rankId在本卡的起始偏移加上专家偏移，卡内的偏移为先前moe专家的偏移总量
                    GM_ADDR srcTokenAddr = (GM_ADDR)(yOutGM_ + moeStartToken * hAlignWinSize_);
                    // 目的地址的token偏移量
                    uint32_t tokenOffset = expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId);
                    GM_ADDR dstTokenAddr = (GM_ADDR)(GetUrmaAddrByRankId(dstRankId) + tokenOffset);
                    uint32_t length = tokenNum * hAlignWinSize_;
                    hcomm_.WriteNbi<false>(channelId, dstTokenAddr, srcTokenAddr, length);
                    moeStartToken += tokenNum;
                }

                // 发送状态
                GM_ADDR srcStatusAddr = (GM_ADDR)(statusDataSpaceGM_ + DISPATCH_STAGING_OFFSET +
                                                  (preExpertNum + dstExpertId) * 8 * sizeof(uint32_t));
                uint32_t offset = stateOffset_ * epRankId_;
                if (moeExpertNumPerRank_ > 1) {
                    offset = (epRankId_ + dstExpertId * epWorldSize_) * stateOffset_;
                }
                GM_ADDR dstStatusAddr = (GM_ADDR)(GetUrmaStateAddrByRankId(dstRankId) + offset);
                hcomm_.WriteNbi<false>(channelId, dstStatusAddr, srcStatusAddr, UB_ALIGN);
            }

            if (moeExpertNumPerRank_ > 0) {
                hcomm_.Commit(channelId);
            }
        }
    }

    middleRank = epRankId_ < endRankId ? epRankId_ : endRankId;
    for (uint32_t dstRankId = startRankId; dstRankId < middleRank; ++dstRankId) {
        if (dstRankId < sharedExpertRankNum_) {
            continue;
        }
        // 本卡通信：仅搬运数据
        if (dstRankId == epRankId_) {
            CurRankComm(statusTensor_, moeStartToken);
            continue;
        }

        // 给共享专家卡发数据和状态，每张共享专家卡上只会有一个共享专家
        if (dstRankId < sharedExpertRankNum_) {
        }
        // 给moe专家卡发数据和状态，每张moe专家卡上有moeExpertNumPerRank_个moe专家
        else {
            // 非本卡通信：发完 epRankId_-->dstExpertId 的所有token后，统一敲doorbell
            auto channelId = GetCommHandle(dstRankId);
            uint32_t preExpertNum = sharedExpertRankNum_ + (dstRankId - sharedExpertRankNum_) * moeExpertNumPerRank_;
            for (uint32_t dstExpertId = 0; dstExpertId < moeExpertNumPerRank_; ++dstExpertId) {
                // 当前rank --> dstExpertId 的所有 token 数量
                uint32_t tokenNum = statusTensor_((preExpertNum + dstExpertId) * 8 + 1);

                // 发送token
                if (tokenNum > 0) {
                    // 起始地址应该为rankId在本卡的起始偏移加上专家偏移，卡内的偏移为先前moe专家的偏移总量
                    GM_ADDR srcTokenAddr = (GM_ADDR)(yOutGM_ + moeStartToken * hAlignWinSize_);
                    // 目的地址的token偏移量
                    uint32_t tokenOffset = expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId);
                    GM_ADDR dstTokenAddr = (GM_ADDR)(GetUrmaAddrByRankId(dstRankId) + tokenOffset);
                    uint32_t length = tokenNum * hAlignWinSize_;
                    hcomm_.WriteNbi<false>(channelId, dstTokenAddr, srcTokenAddr, length);
                    moeStartToken += tokenNum;
                }

                // 发送状态
                GM_ADDR srcStatusAddr = (GM_ADDR)(statusDataSpaceGM_ + DISPATCH_STAGING_OFFSET +
                                                  (preExpertNum + dstExpertId) * 8 * sizeof(uint32_t));
                uint32_t offset = stateOffset_ * epRankId_;
                if (moeExpertNumPerRank_ > 1) {
                    offset = (epRankId_ + dstExpertId * epWorldSize_) * stateOffset_;
                }
                GM_ADDR dstStatusAddr = (GM_ADDR)(GetUrmaStateAddrByRankId(dstRankId) + offset);
                hcomm_.WriteNbi<false>(channelId, dstStatusAddr, srcStatusAddr, UB_ALIGN);
            }

            if (moeExpertNumPerRank_ > 0) {
                hcomm_.Commit(channelId);
            }
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchSetup<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        TokenScatter();
        AlltoAllDispatch();
        SetStatus();
        SyncAll<true>();
        Communication();
    }
}

} // namespace Mc2Kernel
#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_H
