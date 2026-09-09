/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mte_comm.h
 * \brief quant_all_reduce 与quant_reduce_scatter MTE通信相关 kernel代码公共部分
 */

#ifndef MTE_COMMON_H
#define MTE_COMMON_H

#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "quant_reduce_scatter_context.h"

namespace QuantMTECommImpl {

using namespace AscendC;
// 后缀_BYTES 表示单位为字节大小B, _NUM 表示单位为个
constexpr static uint32_t UB_ALIGN_BYTES = 32U;    // UB按32B对齐
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;        // float格式下32B对齐需要 32/4 =8个
constexpr uint32_t BUFFER_NUM = 2U;                // 用于double buffer
constexpr static uint32_t SCALE_BLCOK_BYTES = 32U; // 一个scale数据块固定32B，搬运要求32B对齐
constexpr static uint32_t X_BLOCK_BYTES = 1024U; // 当前一个x数据块固定1024B = 128 * 8(PT) = 32 * 32(MX)，为穿刺取值
constexpr static uint64_t STATE_WIN_SIZE =
    1024UL * 1024UL; // Win区前1MB作为状态区相关，状态0区(462KB) + 状态1区(462KB) + 标志位区(100KB)
constexpr static uint64_t SINGLE_STATE_REGION_SIZE = 462UL * 1024UL; // 0/1分区单个状态区大小
constexpr static uint32_t ZERONE_STATE_POS = 0U;                     // 0/1分区标志位的index
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;                    // 每个核的标志位间512B对齐

#define TemplateTypeClass typename XType, typename ScalesType, typename OutputType
#define TemplateType XType, ScalesType, OutputType

template <TemplateTypeClass>
class MTECommunication {
public:
    __aicore__ inline MTECommunication(){};
    __aicore__ inline void InitHcclContext(GM_ADDR context, uint64_t hcclBufferSize);
    __aicore__ inline void InitParams();
    __aicore__ inline void InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output, uint64_t alignedXSize,
                                        uint64_t dataSpaceGmSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum,
                                        uint64_t scaleNumsPerBlock = 0);
    template <bool isReduceScatter = false>
    __aicore__ inline void CopyDataToWin(uint64_t xSliceSizeNums = 0, uint64_t scaleSliceNums = 0);
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float> &localResultTensor,
                                              uint32_t count);
    __aicore__ inline void ComputeTailAivId(uint64_t totalAivCount);
    __aicore__ inline GM_ADDR GetWinAddrGm(uint32_t rankId, uint64_t offset = 0);
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);

    uint32_t rankIdHccl_{0};
    uint32_t rankDimHccl_{0};
    GM_ADDR windowsIn_[HCCL_MAX_RANK_SIZE];
    GM_ADDR windowsOut_[HCCL_MAX_RANK_SIZE];
    __gm__ QuantReduceScatterContext *mc2Context_{nullptr};

    uint32_t aivId_{0};
    uint64_t aivNum_{0};
    uint32_t round_{0};
    uint32_t tailBlockNums_{0};
    uint32_t assignedBlockNums_{0};
    uint64_t scaleNumsPerBlock_{0};
    uint64_t xOffset_{0};
    uint64_t scaleOffset_{0};
    uint64_t lastAivId_{0};
    uint32_t winBufferFlags_{0};

private:
    uint32_t xNumPerBlock_{0};
    uint64_t tailXNums_{0};

    __aicore__ inline void CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset, uint32_t count);

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;
    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<ScalesType> localWinScaleGMTensor_;
    GlobalTensor<OutputType> outputTensor_;
    GlobalTensor<uint32_t> selfWinFlagGMTensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<ScalesType> scaleTmpTensor_;
    LocalTensor<float> stateResetTensor_;
    LocalTensor<OutputType> xOutTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_, scaleQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> winFlagsBuf_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;
};

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitHcclContext(GM_ADDR context, uint64_t hcclBufferSize)
{
    mc2Context_ = (__gm__ QuantReduceScatterContext *)context;
    rankIdHccl_ = mc2Context_->epRankId;
    rankDimHccl_ = mc2Context_->rankSizePerServer;
    for (int i = 0; i < rankDimHccl_; i++) {
        windowsIn_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
        windowsOut_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i] + hcclBufferSize / 2;
    }
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitParams()
{
    aivId_ = GetBlockIdx(); // 获取当前核Id
    // scaleNumsPerBlock_ 由 SetBlockSize 设置（与 xNumPerBlock_ 配套），此处不再覆盖
    assignedBlockNums_ =
        aivId_ < tailBlockNums_ ? round_ + 1 : round_; // 当前核分配到的数据块数量，顺序分核，序号小的核多搬一轮
    uint64_t blockIdx =
        aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_); // 计算当前核分派到的首个数据块序列号
    xOffset_ = blockIdx * xNumPerBlock_;          // 块数 * 一块有多少数据，得到要搬第几个x数据
    scaleOffset_ = blockIdx * scaleNumsPerBlock_; // 计算要搬第几个 scale
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitGMTensor(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                                                    uint64_t xSize, uint64_t winSpaceGmSize)
{
    // 入参相关数据的GMTensor
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
    outputTensor_.SetGlobalBuffer((__gm__ OutputType *)output);

// =========== Win区相关 ===========
// Win区OOM检测适配，告知OOM框架Win区地址和大小
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (uint64_t curRank = 0; curRank < rankDimHccl_; ++curRank) {
        OOMCheckAddrRange(GetWinAddrGm(curRank), winSpaceGmSize);
    }
#endif

    //  处理 0/1 分区 标志位
    uint64_t currCoreFlagOffset =
        2UL * SINGLE_STATE_REGION_SIZE + aivId_ * WIN_ADDR_ALIGN; // 计算当前核的标志位在Win区的偏移
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t *)GetWinAddrGm(rankIdHccl_, currCoreFlagOffset));
    LocalTensor<uint32_t> winFlagLocalTensor = winFlagsBuf_.Get<uint32_t>();
    DataCopy(winFlagLocalTensor, selfWinFlagGMTensor_, UB_ALIGN_BYTES / sizeof(uint32_t)); // GM -> UB
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    winBufferFlags_ = winFlagLocalTensor.GetValue(ZERONE_STATE_POS);    // 获取标志位
    winFlagLocalTensor.SetValue(ZERONE_STATE_POS, 1 - winBufferFlags_); // 翻转标志位, 0→1, 1→0
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfWinFlagGMTensor_, winFlagLocalTensor, UB_ALIGN_BYTES / sizeof(uint32_t)); // 覆写标志位回GM

    // 获取本卡地址写数据
    // 通过rankId和0/1分区标志位获取本地winIn区地址对应卡的数据区域
    GM_ADDR localDataSpaceGm = GetWinDataAddrGm(rankIdHccl_, winBufferFlags_);
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType *)localDataSpaceGm);
    localWinScaleGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)(localDataSpaceGm + xSize)); // sclae数据跟在x后
}

template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::InitBuffer(TPipe *tPipe)
{
    // xQueue_/scaleQueue_ 与 xNumPerBlock_/scaleNumsPerBlock_ 配套（SetBlockSize 设定）
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(XType));
    uint64_t scaleQueueBytes = scaleNumsPerBlock_ * sizeof(ScalesType);
    if (scaleQueueBytes < UB_ALIGN_BYTES) {
        scaleQueueBytes = UB_ALIGN_BYTES; // 至少 32B（DataCopy 对齐要求）
    }
    tPipe->InitBuffer(scaleQueue_, BUFFER_NUM, scaleQueueBytes);
    tPipe->InitBuffer(xOutQueue_, BUFFER_NUM, xNumPerBlock_ * sizeof(OutputType)); // 用于输出的OutPutTensor
    tPipe->InitBuffer(winFlagsBuf_, UB_ALIGN_BYTES);                               // 用于读取0/1分区的标志位
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES);                             // 状态位每一个按32B对齐
    tPipe->InitBuffer(readStateBuf_, rankDimHccl_ * UB_ALIGN_BYTES);               // 每次读 rankDim 个状态位
    tPipe->InitBuffer(stateResetBuf_, rankDimHccl_ * UB_ALIGN_BYTES);              // 用于清理状态区

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0,
                     static_cast<uint32_t>(rankDimHccl_ * FLOAT_UB_ALIGN_NUM)); // 用于状态区清零
}

/**
 * @brief 配置数据块划分参数，用于多核并行计算
 *
 * @param elementsPerBlock 每个标准数据块包含的X元素数量
 * @param aivNum AIV核的总数，用于数据分发和负载均衡
 * @param lastBlockNum 最后一个核处理的尾部数据块元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum,
                                                                    uint64_t lastBlockNum, uint64_t scaleNumsPerBlock)
{
    xNumPerBlock_ = elementsPerBlock;
    aivNum_ = aivNum;
    tailXNums_ = lastBlockNum;
    // scaleNumsPerBlock 为 0 时按默认逻辑（SCALE_BLCOK_BYTES/sizeof(ScalesType)）计算，与旧接口兼容
    scaleNumsPerBlock_ = (scaleNumsPerBlock != 0) ? scaleNumsPerBlock : (SCALE_BLCOK_BYTES / sizeof(ScalesType));
}

/**
 * @brief 计算负责处理尾部数据的AI核ID
 *
 * @param totalAivCount AIV核的总数量
 *
 * @note 当数据量较小时，此时计算仅仅只有一轮（round_ == 0），此时处理尾块的aiv并非最后一个，
 *       需根据当前数据块数量计算。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ComputeTailAivId(uint64_t totalAivCount)
{
    if (round_ == 0) {
        // 小数据量时，如果只有一轮搬运，负责尾块的aiv由此时计算的总块数决定
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        // 轮次大于一轮时，负责尾块的aiv必定是最后一个
        lastAivId_ = totalAivCount - 1;
    }
}

/**
 * @brief 复制数据块从GM到Win区
 *
 * 复制流程：
 * 1. 量化数据（x）复制：GM → UB（队列分配缓冲区） → Win区
 * 2. 缩放因子（scale）复制：GM → UB（队列分配缓冲区） → Win区
 *
 * @param curXOffset 量化数据x在全局内存(GM)中的偏移量
 * @param curScaleOffset 缩放因子scale在全局内存(GM)中的偏移量
 * @param count 当前x数据块dataCopy的元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataBlock(uint64_t curXOffset, uint64_t curScaleOffset,
                                                                     uint32_t count)
{
    // 先拷贝data数据， 再拷贝scales
    /* x 从 GM -> UB -> Win */
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor_, xGMTensor_[curXOffset], count);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    DataCopy(localWinXGMTensor_[curXOffset], xTmpTensor_, count);
    xQueue_.FreeTensor<XType>(xTmpTensor_);

    /* scale 从 GM -> UB -> Win（DataCopyPad 处理 sub-32B 对齐） */
    scaleTmpTensor_ = scaleQueue_.AllocTensor<ScalesType>();
    DataCopyParams scaleGmToUbParams;
    scaleGmToUbParams.blockLen = scaleNumsPerBlock_ * sizeof(ScalesType);
    scaleGmToUbParams.blockCount = 1;
    DataCopyPadParams scaleGmToUbPadParams;
    DataCopyPad(scaleTmpTensor_, scalesGMTensor_[curScaleOffset], scaleGmToUbParams, scaleGmToUbPadParams);
    scaleQueue_.EnQue(scaleTmpTensor_);
    scaleTmpTensor_ = scaleQueue_.DeQue<ScalesType>();
    DataCopyParams scaleUbToWinParams;
    scaleUbToWinParams.blockLen = scaleNumsPerBlock_ * sizeof(ScalesType);
    scaleUbToWinParams.blockCount = 1;
    DataCopyPad(localWinScaleGMTensor_[curScaleOffset], scaleTmpTensor_, scaleUbToWinParams);
    scaleQueue_.FreeTensor<ScalesType>(scaleTmpTensor_);
}

/**
 * @brief 将数据复制到Win区（支持ReduceScatter和AllReduce两种通信模式）
 *
 * 该函数根据通信模式将数据从全局内存GM全部复制到本卡Win区，支持两种分布式通信模式：
 * 1. ReduceScatter模式：由于数据按Rank（设备）分片，需按rankDim遍历以搬运所有数据
 * 2. AllReduce模式：数据全局复制，每个核直接将数据复制到Win区
 *
 * @tparam isReduceScatter 通信模式标识，true表示ReduceScatter模式，false表示AllReduce模式
 *
 * @param[in] xSliceSizeNums 每个Rank的数据分片大小（元素数量），ReduceScatter模式使用
 * @param[in] scaleSliceNums 每个Rank的缩放因子分片大小（元素数量），ReduceScatter模式使用
 */
template <TemplateTypeClass>
template <bool isReduceScatter>
__aicore__ inline void MTECommunication<TemplateType>::CopyDataToWin(uint64_t xSliceSizeNums, uint64_t scaleSliceNums)
{
    // 遍历每个核需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_;              // 计算现在搬第几个x
        uint64_t curScaleOffset = scaleOffset_ + curBlock * scaleNumsPerBlock_; // 计算现在搬第几个scale
        uint32_t copyBlockNum = xNumPerBlock_;
        if ((aivId_ == lastAivId_) && (curBlock == assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_; // 检测是否为最后的尾块搬运
        }
        if constexpr (isReduceScatter) {
            // ReduceScatter过程，数据按卡均分，需要对卡进行遍历
            for (uint64_t curRank = 0; curRank < rankDimHccl_; ++curRank) {
                // all2all过程，加上卡偏移
                uint64_t curRankXOffset = curXOffset + curRank * xSliceSizeNums;
                uint64_t curRankScaleOffset = curScaleOffset + curRank * scaleSliceNums;

                // 搬运当前数据块到Win区
                CopyDataBlock(curRankXOffset, curRankScaleOffset, copyBlockNum);
            }
        } else {
            // AllReduce过程，allgather直接搬运
            CopyDataBlock(curXOffset, curScaleOffset, copyBlockNum);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

/**
 * @brief 向Win区状态区写入本核完成数据搬运状态
 *
 * 该函数负责将当前AI Core（核）的数据搬运完成状态写入到所有Rank的状态Win区中，
 * 通知其他设备当前核的数据准备已完成。这是集群通信软同步机制的一部分，
 * 每个核需要向所有Rank（包括本机和其他设备）的状态窗口写入标识，
 * 确保所有设备都能感知到当前核的完成状态。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * rankDimHccl_; // Win区大小为 aivNum * rankDim, 此处计算核偏移
    // 遍历每一张卡，给每一张卡都要写入状态
    for (uint32_t curRank = 0; curRank < rankDimHccl_; ++curRank) {
        // 写入状态到对端，每个核写一个状态，表示自己的数据块已经写完
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_,
                        FLOAT_UB_ALIGN_NUM); // 先重置statusTensor数据，后面累加需要Tensor内全部数据，防止脏数据
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;                                            // 用1标识
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank, winBufferFlags_); // 获取当前要写对端卡的状态区地址
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float *)remoteWinStateGM);
        // 不同卡上的核的状态写到相邻位置，读时可以一次读rankDim个状态, 状态区大小设计为 aivNum * ranDim
        uint64_t curOffset = (coreOffset + rankIdHccl_) * FLOAT_UB_ALIGN_NUM; // 当前核偏移 + 卡偏移， 按32B对齐
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM); // 按32B对齐拷贝
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

/**
 * @brief 读取Win区状态区并等待同步状态
 *
 * 该函数从本Rank的状态Win区读取同步标识，并在本地进行忙等待，直到所有参与通信的Rank
 * 都已完成状态设置。这种软同步机制确保所有Rank上当前核（AI Core）所需的数据都已
 * 准备就绪，从而避免跨设备数据不一致性问题。
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(rankIdHccl_, winBufferFlags_); // 获取本卡的状态区用于读取
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * rankDimHccl_ * FLOAT_UB_ALIGN_NUM; // 获取当前核所需读取状态位的头地址，状态按32B对齐
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float *)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0;                                         // 用于计算状态和
    uint32_t statusCnt = rankDimHccl_ * FLOAT_UB_ALIGN_NUM; // 一次读rankDim个，按32B对齐
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = rankDimHccl_ - (float)0.5;
    float maxTarget = rankDimHccl_ + (float)0.5;
    // 读取statusCnt个数据求和
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt); // 相关状态区重新置零
}

/**
 * @brief 将计算得到的UB上的结果Tensor的数据复制到GM上的OutPutTensor，并进行类型转换（如果需要）。
 *
 * @param outputOffset 输出OutputTensor的GM偏移量，用于指定目标位置。
 * @param sourceTensor 本地UB上计算结果Tensor，包含计算完成的数据。
 * @param count 当前每次处理数据块的元素数量
 */
template <TemplateTypeClass>
__aicore__ inline void MTECommunication<TemplateType>::CopyResultToOutput(uint64_t outOffsetGM,
                                                                          LocalTensor<float> &localResultTensor,
                                                                          uint32_t count)
{
    // 将计算好的数据拷贝到输出tensor，如果是非float数据类型需要先转换成目标数据类型
    xOutTensor_ = xOutQueue_.AllocTensor<OutputType>();
    if constexpr (AscendC::IsSameType<OutputType, float>::value) {
        DataCopy(xOutTensor_, localResultTensor, count);
    } else {
        Cast(xOutTensor_, localResultTensor, RoundMode::CAST_RINT, count);
    }
    xOutQueue_.EnQue(xOutTensor_);
    xOutTensor_ = xOutQueue_.DeQue<OutputType>();
    DataCopy(outputTensor_[outOffsetGM], xOutTensor_, count);
    xOutQueue_.FreeTensor(xOutTensor_);
}

// 获取指定rank, 指定偏移下Win区的GM地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinAddrGm(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(windowsIn_[rankId] + offset);
}

// 根据0/1分区标志位，获取对应rank的Win区数据区的地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        // 若使用 0 分区, 加上状态区的偏移
        return GetWinAddrGm(rankId, STATE_WIN_SIZE);
    } else {
        // 若使用 1 分区，即WinOut
        return (GM_ADDR)(windowsOut_[rankId]);
    }
}

// 根据0/1分区标志位，获取对应rank的Win区状态区的地址
template <TemplateTypeClass>
__aicore__ inline GM_ADDR MTECommunication<TemplateType>::GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        // 若使用 0 分区, 即无偏移
        return GetWinAddrGm(rankId);
    } else {
        // 若使用 1 分区，则加上状态0区的大小偏移
        return GetWinAddrGm(rankId, SINGLE_STATE_REGION_SIZE);
    }
}
} // namespace QuantMTECommImpl
#endif // MTE_COMMON_H
