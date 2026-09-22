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
 * \file moe_distribute_combine_a2_layered.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#define MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../moe_distribute_combine_tiling.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_a2_base.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_a2_adump.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_a2_constant.h"

#include "../../../common/op_kernel/mc2_combine_math.h"

#include "../../../common/op_kernel/mc2_combine_type_traits.h"

namespace MoeDistributeCombineA2Impl {

constexpr int UB_ALIGN_SIZE = 32;
constexpr uint64_t CACHELINE_SIZE = 64;

#define TemplateMC2TypeA2layeredClass typename ExpandXType, typename ExpandIdxType, typename ExpandXTransType
#define TemplateMC2TypeA2layeredFunc ExpandXType, ExpandIdxType, ExpandXTransType

using Mc2Combine::OutputType;
using Mc2Combine::OutputType_t;

using namespace AscendC;
template <TemplateMC2TypeA2layeredClass>
class MoeDistributeCombineA2Layered {
public:
    constexpr static uint32_t UB_ALIGN = 32U; // UB按32字节对齐

    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B16_PER_BLOCK = 16U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static uint32_t SERVER_RANK_SIZE = 8U;
    constexpr static uint32_t VEC_LEN = 256U;
    constexpr static bool DynamicQuant = std::is_same<ExpandXTransType, int8_t>::value;
    constexpr static uint32_t TBUF_TEMP_OFFSET = 0U;
    constexpr static uint32_t IPC_REDUCE_USED_CORE_NUM = 32U; // 拉起远端IPC和机内reduce需要的核数
    constexpr static uint32_t WEIGHT_VALUE_NUM = 16U;         // token(h * sizeof(bf/fp16)) + scale(32B) = (h + 16) * 2B
    constexpr static uint64_t GM2IPC_SYNC_FLAG = 12345ULL;
    constexpr static uint64_t RDMA_TOKEN_ARRIVED_FLAG = 123ULL;
    constexpr static uint64_t RDMA_TOKEN_END_FLAG = 321ULL;

    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        return Mc2Combine::RoundUp(val, align);
    }
    template <typename T>
    inline __aicore__ T CeilDiv(const T dividend, const T divisor)
    {
        return Mc2Combine::CeilDiv(dividend, divisor);
    }

    __aicore__ inline MoeDistributeCombineA2Layered(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM);
    __aicore__ inline void Process();
    __aicore__ inline void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                           uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo);

private:
    __aicore__ inline int PrepareQuantizedRecvWindow();
    // Tensor views preserve the existing UB offsets and buffer reuse across reduction stages.
    struct WindowSumBuffers {
        LocalTensor<int16_t> countReduceLocal;
        LocalTensor<int32_t> offsetReduceLocal;
        LocalTensor<uint32_t> tmpTokenIdxLocal;
        LocalTensor<uint32_t> offsetIndexs;
        LocalTensor<uint32_t> copyNums;
        LocalTensor<uint32_t> dataOffsets;
        LocalTensor<ExpandXType> dataInLocal;
        LocalTensor<float> castInFloatLocal;
        LocalTensor<float> sumFloatLocal;
        LocalTensor<float> inUbTemp;
        LocalTensor<ExpandXTransType> castDataInt8;
        LocalTensor<ExpandXType> scaleData;
        LocalTensor<ExpandXType> sumHalfLocal;
        LocalTensor<ExpandXType> reduceMaxOutTensor;
        LocalTensor<ExpandXType> absScaleTensor;
        LocalTensor<half> halfLocal;
        LocalTensor<float> reduceMaxTensorFloat;
        LocalTensor<uint64_t> rdmaFlagLocal;
    };

    struct ServerSumBuffers {
        LocalTensor<float> sumFloatLocal;
        LocalTensor<ExpandXType> sumFpAndBfLocal;
        LocalTensor<ExpandXType> dataIn;
        LocalTensor<float> castFp32;
        LocalTensor<ExpandXType> sumFp16Local;
        LocalTensor<ExpandXTransType> dataInt8;
        LocalTensor<ExpandXType> scaleData;
        LocalTensor<ExpandXType> castFp16;
        LocalTensor<ExpandXType> scaleDup;
        LocalTensor<float> sumFloatLocal1;
        LocalTensor<ExpandXType> sumBf16Local;
        LocalTensor<ExpandXTransType> dataInUbInt8;
        LocalTensor<ExpandXType> scaleDataBf16;
        LocalTensor<half> castDataHalf;
        LocalTensor<float> castDataFp32;
        LocalTensor<float> castFp32scale;
        LocalTensor<float> castFp32ScaleBrcb;
    };

    __aicore__ inline void CopyLocalServerTokens(LocalTensor<ExpandXTransType> &tmpLowUb_,
                                                 GlobalTensor<ExpandXTransType> &aivSrcGlobal,
                                                 GlobalTensor<ExpandXTransType> &aivDstGlobal, uint32_t copyLenAlign_,
                                                 uint32_t copySum, uint32_t copyOnceNum);
    __aicore__ inline void WriteRdmaWqe(__gm__ uint8_t *wqeAddr, GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr,
                                        uint64_t messageLen, uint64_t destRankId, uint64_t curHead,
                                        __gm__ HcclAiRMAInfo *QpInfo);
    __aicore__ inline void ReduceWindowTokens(WindowSumBuffers &buffers, uint32_t targetServerId, uint32_t targetRankId,
                                              uint32_t coreNumPerServer, uint32_t totalCopyLen,
                                              uint32_t offsetNumPerExpert);
    __aicore__ inline void AccumulateIpcWindowToken(WindowSumBuffers &buffers, uint32_t targetRankId,
                                                    uint32_t offsetNumPerExpert, uint32_t offsetIndexStart);
    __aicore__ inline void BuildWindowTokenList(WindowSumBuffers &buffers, uint32_t targetServerId,
                                                uint32_t coreNumPerServer, uint32_t &baseBuffOffset,
                                                uint32_t &totalCopyLen);
    __aicore__ inline void InitWindowSumBuffers(WindowSumBuffers &buffers, uint32_t baseBuffOffset);
    __aicore__ inline void QuantizeBf16WindowToken(WindowSumBuffers &buffers, uint32_t dataOffset, uint32_t tokenOffset,
                                                   const DataCopyParams &flagDataCopyParams);
    __aicore__ inline void QuantizeFp16WindowToken(WindowSumBuffers &buffers, uint32_t dataOffset, uint32_t tokenOffset,
                                                   const DataCopyParams &flagDataCopyParams);
    __aicore__ inline void InitServerSumBuffers(ServerSumBuffers &buffers);
    __aicore__ inline void SumUnquantizedToServer(ServerSumBuffers &buffers, int copyNum, uint32_t i);
    __aicore__ inline void SumBf16ToServer(ServerSumBuffers &buffers, int copyNum, uint32_t i);
    __aicore__ inline void SumFp16ToServer(ServerSumBuffers &buffers, int copyNum, uint32_t i);
    __aicore__ inline void WaitRdmaTokens(LocalTensor<uint64_t> &checkRdmaLocal, uint32_t copyOnceNum,
                                          uint32_t &copySum, bool &stopFlag,
                                          const DataCopyExtParams &flagDataCopyParams,
                                          const DataCopyPadExtParams<uint64_t> &flagPadParams);
    __aicore__ inline void NotifyIpcReady();
    __aicore__ inline void UpdateRdmaDoorbell(__gm__ HcclAiRMAWQ *qp_ctx_entry, uint64_t curHead,
                                              uint64_t curHardwareHead);
    __aicore__ inline void InitSendCountsAndStatus(GM_ADDR sendCount);
    __aicore__ inline void InitQuantParams();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void GM2IPC();
    __aicore__ inline void WaitIPC();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline uint32_t GatherValidTokenIdx(WindowSumBuffers &buffers, uint32_t realBS);
    __aicore__ inline void FillWindowOffsetLists(WindowSumBuffers &buffers, uint32_t coreNumPerServer,
                                                 uint32_t tokenNum, uint32_t &baseBuffOffset, uint32_t &totalCopyLen);
    __aicore__ inline void InitServerUnquantizedBuffers(ServerSumBuffers &buffers, uint32_t &baseBuffOffset);
    __aicore__ inline void InitServerFp16Buffers(ServerSumBuffers &buffers, uint32_t &fpBaseBuffOffset);
    __aicore__ inline void InitServerBf16Buffers(ServerSumBuffers &buffers, uint32_t &bfBaseBuffOffset);
    __aicore__ inline void CopyTokensToIpc(LocalTensor<ExpandIdxType> &sendCountLocal,
                                           LocalTensor<float> &expandScalesLocal, LocalTensor<ExpandXType> &inUb,
                                           LocalTensor<float> &inUbTemp);
    __aicore__ inline void InitWindowCommonBuffers(WindowSumBuffers &buffers, uint32_t &baseBuffOffset,
                                                   uint32_t &fp16baseBuffOffset);
    __aicore__ inline void InitWindowFp16Buffers(WindowSumBuffers &buffers, uint32_t &fp16baseBuffOffset);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM);
    __aicore__ inline void InitGlobalBuffers(GM_ADDR expandX, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR XOut,
                                             GM_ADDR performanceInfo);
    __aicore__ inline void DispatchServerTokens(LocalTensor<uint64_t> &checkRdmaLocal,
                                                LocalTensor<ExpandXTransType> &tmpLowUb_,
                                                GlobalTensor<ExpandXTransType> &aivSrcGlobal,
                                                GlobalTensor<ExpandXTransType> &aivDstGlobal, uint32_t dstRankId,
                                                uint32_t &copySum, uint32_t copyOnceNum, uint32_t copyLen_,
                                                uint32_t copyLenAlign_, uint64_t srcrdmaAddr, uint64_t dstrdmaAddr,
                                                const DataCopyExtParams &flagDataCopyParams,
                                                const DataCopyPadExtParams<uint64_t> &flagPadParams);
    __aicore__ inline void SumToServer();
    __aicore__ inline void Preload();
    __aicore__ inline void ToWindowPreload();
    __aicore__ inline void CopyPerformanceInfo();

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> expandScalesGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;

    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXTransType> localInWindow_;
    GlobalTensor<uint64_t> readStateGlobal_;
    GlobalTensor<int32_t> performanceInfoI32GMTensor_;

    // 低精度需要用到的变量
    GlobalTensor<ExpandXType> localInScaleWindow_;
    OutputType_t<ExpandXType> scaleMulVal;
    uint32_t mask;

    __gm__ HcclAiRMAInfo *qp_info_;

    GlobalTensor<uint64_t> shareFlagGlobal_;
    GlobalTensor<ExpandXType> shareMemGlobal_;
    GlobalTensor<ExpandXType> dstshareMemGlobal_;
    GlobalTensor<int32_t> offsetInnerGlobal_;
    GlobalTensor<int16_t> countInnerGlobal_;
    GlobalTensor<int32_t> offsetOuterGlobal_;
    GlobalTensor<int32_t> countOuterGlobal_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t globalBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0}; // topK
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};           // aiv id
    uint32_t moeExpertNum_{0};      // moe专家数, 等于worldSize_ - 共享专家卡数
    uint32_t localMoeExpertNum_{0}; // 每张卡的专家数
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t serverNum_{0};
    uint32_t performanceInfoSize_{0};
    bool needPerformanceInfo_{false};

    TBuf<QuePosition::VECCALC> tBuf;
    TBuf<TPosition::VECOUT> rdmaInBuf_;
    TBuf<TPosition::VECOUT> rdmaInBuf2_;
    TBuf<> statusBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<TPosition::LCM> aDumpBuf_;

    uint64_t sumTarget_{0};
    uint32_t startBs{0};
    uint32_t endBs{0};
    uint32_t processNum{0};
    uint32_t resNum{0};
    uint32_t offsetIndex{0};
    uint32_t maxBs_{0};
    uint32_t stepCoreNum{0};
    uint64_t magicValue_{0};
    LocalTensor<int32_t> offsetReduceLocal_;
    LocalTensor<int32_t> countReduceLocal_;
    LocalTensor<uint64_t> ubLocal;
    LocalTensor<uint32_t> ubLocalHead;
    LocalTensor<int32_t> performanceInfoI32Tensor_;

    // 低精度相关
    uint32_t repeatNum{0};
    uint32_t scaleNum;
    uint32_t scaleNumAlign;
    uint32_t SCALE_GRANU;

    MoeDistributeA2Base::MoeDistributeA2CombineAddrInfo addrInfo_;
    Mc2A2Kernel::MoeDistributeA2ADump aDump_;
};

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::UpdateRdmaDoorbell(
    __gm__ HcclAiRMAWQ *qp_ctx_entry, uint64_t curHead, uint64_t curHardwareHead)
{
    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;                    // [0:23] DB_TAG (qp_num)
    doorBellInfo |= 0UL << 24UL;                          // [24:27] DB_CMD = HNS_ROCE_V2_SQ_DB (0)
    doorBellInfo |= (curHead % 65536UL) << 32UL;          // [32:47] DB_PI = sq.head
    doorBellInfo |= (uint64_t)(qp_ctx_entry->sl) << 48UL; // [48:50] DB_SL = qp.sl

    __gm__ uint64_t *doorBellAddr = (__gm__ uint64_t *)(qp_ctx_entry->dbAddr);
    PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, (uint32_t)curHead);
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WriteRdmaWqe(
    __gm__ uint8_t *wqeAddr, GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t messageLen, uint64_t destRankId,
    uint64_t curHead, __gm__ HcclAiRMAInfo *QpInfo)
{
    auto mem_info_table = QpInfo->memPtr;
    auto sizeof_memdetail = QpInfo->sizeOfAiRMAMem;
    uint64_t shift = 15U;
    // Write the WQE to GM
    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;                     // [0:4] opcode=0x3(RDMA_WRITE)
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U); // [7] owner_bit
    byte_4 |= 1U << 8U;                       // [8:8] IBV_SEND_SIGNALED

    *(__gm__ uint32_t *)(wqeAddr) = byte_4;         // Control set by local parameter see above lines
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen; // message size
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;          // immtdata is always 0 till we provide poll CQ flow in AIV
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U; // [120:127] num_sge = 1
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;         // [128:151] start_sge_idx = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)destDmaAddr; // destination VA

    // Setup SGE and write to GM
    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key; // L_Key
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) =
        (uint64_t)srcDmaAddr; // src VA addr memory registered by RNIC
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AIVRDMAPostSend(
    GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId, uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo)
{
    auto qpNum = ((__gm__ HcclAiRMAInfo *)QpInfo)->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ *)(((__gm__ HcclAiRMAInfo *)QpInfo)->sqPtr +
                               destRankId * qpNum * (uint64_t)(((__gm__ HcclAiRMAInfo *)QpInfo)->sizeOfAiRMAWQ));
    auto cur_rank_id = (((__gm__ HcclAiRMAInfo *)QpInfo)->curRankId);
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t *)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t *)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    auto QP_DEPTH = qp_ctx_entry->depth;

    PipeBarrier<PIPE_ALL>();

    // Make sure we don't overflow the SQ in an infinite loop - no need to mitigate endless loop as the host
    // will timeout and kill the kernel, same as all2all kernel if it fails to complete (e.g. in case of link loss)
    while (1) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle(); // add this line to solve slow poll CQ issue
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    WriteRdmaWqe(wqeAddr, srcDmaAddr, destDmaAddr, messageLen, destRankId, curHead, QpInfo);
    // wqe & sge cache flush
    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    PipeBarrier<PIPE_ALL>();
    curHead++;

    UpdateRdmaDoorbell(qp_ctx_entry, curHead, curHardwareHead);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitQuantParams()
{
    if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
        SCALE_GRANU = 16U;
        scaleNum = axisH_ / SCALE_GRANU;
        scaleNumAlign = RoundUp(scaleNum, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(ExpandXType));
        if (axisH_ >= vecNum) {
            mask = vecNum;
        } else {
            mask = axisH_;
        }

    } else { // bf16
        SCALE_GRANU = 8U;
        scaleNum = axisH_ / SCALE_GRANU;
        scaleNumAlign = RoundUp(scaleNum, (uint32_t)(UB_ALIGN / sizeof(ExpandXType)));
        repeatNum = CeilDiv(axisH_, (VEC_LEN / static_cast<uint32_t>(sizeof(float))));
        uint32_t vecNum = VEC_LEN / static_cast<uint32_t>(sizeof(float)); // Brcb 8个datablock(32Bytes)
        if (axisH_ >= vecNum) {
            mask = vecNum;
        } else {
            mask = axisH_;
        }
    }
    scaleMulVal = 1 / 127.;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitSendCountsAndStatus(
    GM_ADDR sendCount)
{
    uint64_t send_counts_inner_offset = static_cast<uint64_t>(worldSize_ * localMoeExpertNum_);
    uint64_t offset_inner_offset = send_counts_inner_offset + static_cast<uint64_t>(globalBs_ * serverNum_ / 2);
    uint64_t send_counts_outer_offset = offset_inner_offset + static_cast<uint64_t>(globalBs_ * axisK_ * serverNum_);
    uint64_t offset_outer_offset = send_counts_outer_offset + static_cast<uint64_t>(axisBS_);

    countInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(sendCount) + send_counts_inner_offset * 2);
    offsetInnerGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_inner_offset);
    countOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + send_counts_outer_offset);
    offsetOuterGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCount) + offset_outer_offset);

    PipeBarrier<PIPE_ALL>();
    magicValue_ = addrInfo_.GetMagicValue();
    sumTarget_ = magicValue_;
    if (coreIdx_ == 0U) {
        AscendC::LocalTensor<uint64_t> tempLocal = tBuf.Get<uint64_t>();
        tempLocal(0) = sumTarget_;
        auto copyParam = AscendC::DataCopyExtParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
        AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(readStateGlobal_, tempLocal, copyParam);
    }
    LocalTensor<uint8_t> aDumpTensor = aDumpBuf_.Get<uint8_t>();
    GM_ADDR aDumpAddr = addrInfo_.GetAdumpDataAddr();
    aDump_.Init(aDumpAddr, coreIdx_, rankId_, worldSize_, rankId_, moeExpertNum_, worldSize_, globalBs_,
                (magicValue_ - 1UL) & 0x1, Mc2A2Kernel::IS_HIERARCHY, aivNum_, aDumpTensor, axisH_, axisBS_);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR performanceInfo,
    GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData,
    GM_ADDR contextGM)
{
    tpipe_ = pipe;
    InitTilingAttrs(tilingData, contextGM);
    InitGlobalBuffers(expandX, sendCount, scales, XOut, performanceInfo);
    // coreIdx_ < serverNum_
    BuffInit();

    SplitCoreCal();

    InitSendCountsAndStatus(sendCount);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitTilingAttrs(
    const MoeDistributeCombineA2TilingData *tilingData, GM_ADDR contextGM)
{
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;

    globalBs_ = tilingData->moeDistributeCombineInfo.globalBs;
    maxBs_ = globalBs_ / worldSize_;
    InitQuantParams();
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineA2TilingData, mc2CcTiling));
    qp_info_ = (__gm__ HcclAiRMAInfo *)(((__gm__ HcclA2CombineOpParam *)contextGM)->aiRMAInfo);

    serverNum_ = worldSize_ / SERVER_RANK_SIZE;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    addrInfo_.Init(rankId_, maxBs_, worldSize_, axisH_, axisK_, localMoeExpertNum_, aivNum_);

    coreIdx_ = GetBlockIdx();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitGlobalBuffers(
    GM_ADDR expandX, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR XOut, GM_ADDR performanceInfo)
{
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    expandScalesGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    readStateGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(addrInfo_.GetLocalSendBuffFlagAddr()));
    axisHFloatSize_ = axisH_ * static_cast<uint32_t>(sizeof(float));
    axisHExpandXTypeSize_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));

    needPerformanceInfo_ = performanceInfo != nullptr;
    if (unlikely(needPerformanceInfo_)) {
        performanceInfoSize_ = worldSize_;
        performanceInfoI32GMTensor_.SetGlobalBuffer((__gm__ int32_t *)performanceInfo);
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSize_ * sizeof(int64_t));
        performanceInfoI32Tensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoI32Tensor_, 0, performanceInfoSize_ * sizeof(int64_t) / sizeof(int32_t));
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::BuffInit()
{
    // 状态tBuf
    tpipe_->InitBuffer(statusBuf_, worldSize_ * UB_ALIGN);
    uint32_t leftBufSize =
        AscendC::TOTAL_UB_SIZE - worldSize_ * UB_ALIGN - RoundUp(performanceInfoSize_, B64_PER_BLOCK) * sizeof(int64_t);

    // AIVRDMAPostSend函数需要的tBuf
    tpipe_->InitBuffer(rdmaInBuf_, UB_ALIGN);
    ubLocal = rdmaInBuf_.Get<uint64_t>();

    tpipe_->InitBuffer(rdmaInBuf2_, UB_ALIGN);
    ubLocalHead = rdmaInBuf2_.Get<uint32_t>();
    leftBufSize -= UB_ALIGN * 2;

    // aDumpBuf_
    uint32_t aDumpSize =
        Std::max(RoundUp(Mc2A2Kernel::H_POS + 1U, B32_PER_BLOCK), B32_PER_BLOCK * 2U) * sizeof(uint32_t);
    tpipe_->InitBuffer(aDumpBuf_, aDumpSize);
    leftBufSize -= aDumpSize;

    // 总tBuf
    tpipe_->InitBuffer(tBuf, leftBufSize);
}
template <TemplateMC2TypeA2layeredClass>
__aicore__ inline int MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::PrepareQuantizedRecvWindow()
{
    int offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex) % axisBS_) *
                      (axisH_ * sizeof(ExpandXTransType) + scaleNumAlign * sizeof(ExpandXType)) /
                      sizeof(ExpandXTransType);
    uint32_t srcServerId = offsetReduceLocal_.GetValue(offsetIndex) / axisBS_;
    uint32_t srcRankId = srcServerId * SERVER_RANK_SIZE + rankId_ % SERVER_RANK_SIZE;
    GM_ADDR addr = addrInfo_.GetLocalRecvBuffDataAddr(srcRankId);
    localInWindow_.SetGlobalBuffer((__gm__ ExpandXTransType *)(addr));
    localInScaleWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(addr));
    SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
    return offsetOnIpc;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SplitCoreCal()
{
    Mc2Combine::SplitCoreRange(worldSize_, aivNum_, coreIdx_, sendRankNum_, startRankId_, endRankId_);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::NotifyIpcReady()
{
    SyncAll<true>();
    if (coreIdx_ < SERVER_RANK_SIZE) {
        uint32_t dstRankId = (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE + coreIdx_;
        shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)addrInfo_.GetRemoteIpcSyncFlagAddr(dstRankId));
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        inUb(0) = GM2IPC_SYNC_FLAG + magicValue_;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_, inUb, B64_PER_BLOCK);
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::GM2IPC()
{
    // 初始化baseBuffOffset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    // 申请LocalTensor : sendCount 以及计算偏移
    LocalTensor<ExpandIdxType> sendCountLocal =
        tBuf.GetWithOffset<int32_t>(RoundUp(moeExpertNum_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(moeExpertNum_, B32_PER_BLOCK);

    // 申请LocalTensor : expandScales 以及计算偏移
    LocalTensor<float> expandScalesLocal =
        tBuf.GetWithOffset<float>((maxBs_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN, baseBuffOffset);
    baseBuffOffset += sizeof(float) * ((maxBs_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);

    // 申请LocalTensor : InUb。 token：【data】(H * fp16/bf16) + expandScales(32B)
    LocalTensor<ExpandXType> inUb = tBuf.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    LocalTensor<float> inUbTemp = inUb[axisH_].template ReinterpretCast<float>();

    DataCopy(sendCountLocal, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK)); // mte2
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    CopyTokensToIpc(sendCountLocal, expandScalesLocal, inUb, inUbTemp);
    NotifyIpcReady();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::CopyTokensToIpc(
    LocalTensor<ExpandIdxType> &sendCountLocal, LocalTensor<float> &expandScalesLocal, LocalTensor<ExpandXType> &inUb,
    LocalTensor<float> &inUbTemp)
{
    for (uint32_t expertId = 0U; expertId < localMoeExpertNum_; ++expertId) {
        for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; ++dstRankId) {
            uint32_t preCount = 0U;

            if (expertId != 0U || dstRankId != 0U) {
                preCount = static_cast<uint32_t>(sendCountLocal.GetValue(expertId * worldSize_ + dstRankId - 1));
            }
            dstshareMemGlobal_.SetGlobalBuffer(
                (__gm__ ExpandXType *)(addrInfo_.GetIpcDataAddr(rankId_, expertId, dstRankId)));

            uint32_t tokenNum = sendCountLocal.GetValue(expertId * worldSize_ + dstRankId) - preCount;
            uint32_t startTokenAddr = preCount * axisH_;
            PipeBarrier<PIPE_ALL>();
            DataCopy(expandScalesLocal, expandScalesGlobal_[preCount], (tokenNum + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint64_t rankTokenNum = 0U;
            for (uint32_t tokenId = 0U; tokenId < tokenNum; ++tokenId) {
                float scaleVal = expandScalesLocal.GetValue(tokenId);
                inUbTemp(0) = scaleVal;
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                DataCopy(inUb, expandXGlobal_[startTokenAddr], axisH_);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopy(dstshareMemGlobal_[rankTokenNum * (axisH_ + WEIGHT_VALUE_NUM)], inUb,
                         axisH_ + WEIGHT_VALUE_NUM);
                startTokenAddr += axisH_;
                rankTokenNum++;
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitIPC()
{
    ///***
    uint32_t stepCoreNum_ = SERVER_RANK_SIZE;
    // 只要8个core分别wait 来自8卡的flag，然后sync一下 再进行流水

    if (coreIdx_ < stepCoreNum_) {
        uint32_t srcRankId = (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE + coreIdx_;
        shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)addrInfo_.GetLocalIpcSyncFlagAddr(srcRankId));
        int64_t startTime = GetCurrentTimestampUs();
        LocalTensor<uint64_t> inUb = statusBuf_.Get<uint64_t>();
        while (true) {
            DataCopy(inUb, shareFlagGlobal_, B64_PER_BLOCK);
            PipeBarrier<PIPE_ALL>();
            if (inUb(0) >= (GM2IPC_SYNC_FLAG + magicValue_)) {
                break;
            }
        }
        inUb(0) = 0;
        PipeBarrier<PIPE_ALL>();
        DataCopy(shareFlagGlobal_, inUb, B64_PER_BLOCK);
        PipeBarrier<PIPE_ALL>();
        if (unlikely(needPerformanceInfo_)) {
            auto srcRankId = (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE + coreIdx_;
            RecordRankCommDuration(performanceInfoI32Tensor_, srcRankId, startTime);
        }
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::QuantizeFp16WindowToken(
    WindowSumBuffers &buffers, uint32_t dataOffset, uint32_t tokenOffset, const DataCopyParams &flagDataCopyParams)
{
    Cast(buffers.sumHalfLocal, buffers.sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    Abs(buffers.absScaleTensor, buffers.sumHalfLocal, axisH_);
    PipeBarrier<PIPE_V>();
    BlockReduceMax(buffers.reduceMaxOutTensor, buffers.absScaleTensor, repeatNum, mask, 1, 1, 8); // g16
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Muls(buffers.scaleData, buffers.reduceMaxOutTensor, scaleMulVal, scaleNum); // 1/scale = dmax / 127
    PipeBarrier<PIPE_V>();
    Brcb(buffers.absScaleTensor, buffers.scaleData, repeatNum, {1, 8}); // 填充scale值
    PipeBarrier<PIPE_V>();

    Div(buffers.sumHalfLocal, buffers.sumHalfLocal, buffers.absScaleTensor, axisH_); // data_fp16/(1/scale)
    PipeBarrier<PIPE_V>();
    Cast(buffers.castDataInt8, buffers.sumHalfLocal, RoundMode::CAST_RINT, axisH_); // fp16->int8 四舍六入五成双
    PipeBarrier<PIPE_V>();

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(localOutWindow_[dataOffset], buffers.dataInLocal, axisH_ / 2 + scaleNumAlign); // int8数据+scale值
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(shareFlagGlobal_[tokenOffset], buffers.rdmaFlagLocal, flagDataCopyParams);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::QuantizeBf16WindowToken(
    WindowSumBuffers &buffers, uint32_t dataOffset, uint32_t tokenOffset, const DataCopyParams &flagDataCopyParams)
{
    PipeBarrier<PIPE_V>();
    Abs(buffers.castInFloatLocal, buffers.sumFloatLocal, axisH_); // 求fp32张量的绝对值
    PipeBarrier<PIPE_V>();
    BlockReduceMax(buffers.reduceMaxTensorFloat, buffers.castInFloatLocal, repeatNum, mask, 1, 1, 8); // fp32的g16
    PipeBarrier<PIPE_V>();
    Muls(buffers.reduceMaxTensorFloat, buffers.reduceMaxTensorFloat, scaleMulVal, scaleNum); // scale = dmax * 1/127
    PipeBarrier<PIPE_V>();
    Brcb(buffers.castInFloatLocal, buffers.reduceMaxTensorFloat, repeatNum, {1, 8}); // 填充fp32的scale值
    PipeBarrier<PIPE_V>();
    Div(buffers.sumFloatLocal, buffers.sumFloatLocal, buffers.castInFloatLocal, axisH_); // data_fp32/(1/scale)
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Cast(buffers.scaleData, buffers.reduceMaxTensorFloat, RoundMode::CAST_RINT, scaleNum); // 1/scale从fp32量化成bf16
    PipeBarrier<PIPE_V>();
    Cast(buffers.halfLocal, buffers.sumFloatLocal, RoundMode::CAST_RINT, axisH_); // token数据fp32->bf16 四舍六入五成双
    PipeBarrier<PIPE_V>();
    Cast(buffers.castDataInt8, buffers.halfLocal, RoundMode::CAST_RINT, axisH_); // token数据bf16->int8 四舍六入五成双
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(localOutWindow_[dataOffset], buffers.dataInLocal, axisH_ / 2 + scaleNumAlign); // int8数据+scale值
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(shareFlagGlobal_[tokenOffset], buffers.rdmaFlagLocal, flagDataCopyParams);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitWindowSumBuffers(
    WindowSumBuffers &buffers, uint32_t baseBuffOffset)
{
    uint32_t fp16baseBuffOffset = 0U;
    InitWindowCommonBuffers(buffers, baseBuffOffset, fp16baseBuffOffset);
    InitWindowFp16Buffers(buffers, fp16baseBuffOffset);

    // 量化 bf16 复用sumFloatLocal
    buffers.halfLocal = tBuf.GetWithOffset<half>(axisH_, baseBuffOffset);

    baseBuffOffset += sizeof(float) * (axisH_); // 复用sumFloatLocal，但是offset要加上sumFloatLocal大小
    buffers.reduceMaxTensorFloat = tBuf.GetWithOffset<float>(scaleNum, baseBuffOffset);
    baseBuffOffset += sizeof(float) * (scaleNum);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitWindowCommonBuffers(
    WindowSumBuffers &buffers, uint32_t &baseBuffOffset, uint32_t &fp16baseBuffOffset)
{
    // 量化和不量化都要用

    buffers.dataInLocal = tBuf.GetWithOffset<ExpandXType>(axisH_ + WEIGHT_VALUE_NUM, baseBuffOffset);
    baseBuffOffset += sizeof(ExpandXType) * (axisH_ + WEIGHT_VALUE_NUM);

    // 初始化 fp16所需tBuf偏移的base Offset
    fp16baseBuffOffset = baseBuffOffset;

    // 量化和不量化都要用 同时也为bf16的Brcb函数扩充复用，扩充到H个，至少要256B对齐
    buffers.castInFloatLocal =
        tBuf.GetWithOffset<float>(RoundUp(axisHFloatSize_, VEC_LEN) / sizeof(float), baseBuffOffset);
    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value &&
                  std::is_same<ExpandXType, bfloat16_t>::value) {
        Duplicate<float>(buffers.castInFloatLocal, 0.0, buffers.castInFloatLocal.GetSize());
    }
    baseBuffOffset += RoundUp(axisHFloatSize_, VEC_LEN);

    // 量化和不量化都要用
    buffers.sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);

    // token格式: data(H*sizeof(ExpandXType)) + weight值(32B)
    buffers.inUbTemp = buffers.dataInLocal[axisH_].template ReinterpretCast<float>();

    // 量化 dataInLocal复用 存放 int8的data fp/bf16的scale
    buffers.castDataInt8 = buffers.dataInLocal.template ReinterpretCast<ExpandXTransType>();
    buffers.scaleData = buffers.dataInLocal[axisH_ / 2].template ReinterpretCast<ExpandXType>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitWindowFp16Buffers(
    WindowSumBuffers &buffers, uint32_t &fp16baseBuffOffset)
{
    // 量化fp16
    buffers.sumHalfLocal = tBuf.GetWithOffset<ExpandXType>(axisH_, fp16baseBuffOffset); // 复用castInFloatLocal
    fp16baseBuffOffset += axisH_ * sizeof(ExpandXType);

    // 16个数取最大值
    buffers.reduceMaxOutTensor = tBuf.GetWithOffset<ExpandXType>(scaleNum, fp16baseBuffOffset);

    // 将scale利用Brcb函数扩充到H个，至少要256B对齐   复用reduceMaxOutTensor
    buffers.absScaleTensor = tBuf.GetWithOffset<ExpandXType>(
        RoundUp(axisHExpandXTypeSize_, VEC_LEN) / sizeof(ExpandXType), fp16baseBuffOffset);
    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value &&
                  std::is_same<ExpandXType, half>::value) {
        Duplicate<ExpandXType>(buffers.absScaleTensor, 0.0, buffers.absScaleTensor.GetSize());
    }
    fp16baseBuffOffset +=
        Std::max(scaleNum * static_cast<uint32_t>(sizeof(ExpandXType)), RoundUp(axisHExpandXTypeSize_, VEC_LEN));
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::BuildWindowTokenList(
    WindowSumBuffers &buffers, uint32_t targetServerId, uint32_t coreNumPerServer, uint32_t &baseBuffOffset,
    uint32_t &totalCopyLen)
{
    // 初始baseBuffOffset
    baseBuffOffset = TBUF_TEMP_OFFSET;
    uint32_t realBS = static_cast<uint32_t>(countInnerGlobal_.GetValue(globalBs_ * targetServerId));
    buffers.countReduceLocal = tBuf.GetWithOffset<int16_t>(RoundUp(realBS, B16_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int16_t) * RoundUp(realBS, B16_PER_BLOCK); // 需要32字节对齐

    buffers.offsetReduceLocal = tBuf.GetWithOffset<int32_t>(RoundUp(realBS * axisK_, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(int32_t) * RoundUp(realBS * axisK_, B32_PER_BLOCK);

    // buffers.tmpTokenIdxLocal 和后续的Ub使用不冲突，可被复用
    buffers.tmpTokenIdxLocal = tBuf.GetWithOffset<uint32_t>(RoundUp(realBS, B32_PER_BLOCK), baseBuffOffset);
    baseBuffOffset += sizeof(uint32_t) * RoundUp(realBS, B32_PER_BLOCK);

    DataCopy(buffers.countReduceLocal, countInnerGlobal_[globalBs_ * targetServerId + 1],
             RoundUp(realBS, B16_PER_BLOCK));
    DataCopy(buffers.offsetReduceLocal, offsetInnerGlobal_[globalBs_ * axisK_ * targetServerId],
             RoundUp(realBS * axisK_, B32_PER_BLOCK));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    totalCopyLen = 0;

    uint32_t tokenNum = GatherValidTokenIdx(buffers, realBS);
    FillWindowOffsetLists(buffers, coreNumPerServer, tokenNum, baseBuffOffset, totalCopyLen);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline uint32_t MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::GatherValidTokenIdx(
    WindowSumBuffers &buffers, uint32_t realBS)
{
    uint32_t tokenNum = 0;
    for (uint32_t i = 0U; i < realBS; i++) {
        uint32_t copyNum = static_cast<uint32_t>(buffers.countReduceLocal.GetValue(i));
        if (copyNum > 0) {
            buffers.tmpTokenIdxLocal(tokenNum) = i;
            ++tokenNum;
        }
    }
    return tokenNum;
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::FillWindowOffsetLists(
    WindowSumBuffers &buffers, uint32_t coreNumPerServer, uint32_t tokenNum, uint32_t &baseBuffOffset,
    uint32_t &totalCopyLen)
{
    // 计算offsetIndex, copyNum, dataOffset
    uint32_t listLen = RoundUp(maxBs_ / coreNumPerServer + 1, B32_PER_BLOCK); // maxBs_ / coreNumPerServer;
    buffers.offsetIndexs = tBuf.GetWithOffset<uint32_t>(listLen, baseBuffOffset);
    baseBuffOffset += listLen * sizeof(uint32_t);
    buffers.copyNums = tBuf.GetWithOffset<uint32_t>(listLen, baseBuffOffset);
    baseBuffOffset += listLen * sizeof(uint32_t);
    buffers.dataOffsets = tBuf.GetWithOffset<uint32_t>(listLen, baseBuffOffset);
    baseBuffOffset += listLen * sizeof(uint32_t);

    // 每个核使用的链路要岔开，不能有冲突
    uint32_t processNum = 0;
    for (uint32_t i = 0U; i < tokenNum; i++) {
        if ((i % coreNumPerServer) == (coreIdx_ % coreNumPerServer)) {
            uint32_t index = buffers.tmpTokenIdxLocal(i);
            int copyNum = static_cast<int32_t>(buffers.countReduceLocal.GetValue(index));
            uint32_t dataOffset = i * (axisH_ / 2U + scaleNumAlign); // uint8的数据
            buffers.copyNums(processNum) = static_cast<uint32_t>(copyNum);
            buffers.offsetIndexs(processNum) = index * axisK_;
            buffers.dataOffsets(processNum) = dataOffset;
            totalCopyLen += static_cast<uint32_t>(copyNum);
            processNum++;
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AccumulateIpcWindowToken(
    WindowSumBuffers &buffers, uint32_t targetRankId, uint32_t offsetNumPerExpert, uint32_t offsetIndexStart)
{
    uint32_t targetLocalServerExpertId = buffers.offsetReduceLocal.GetValue(offsetIndex) / offsetNumPerExpert;
    uint32_t targetIpcRank =
        (targetLocalServerExpertId / localMoeExpertNum_) + (rankId_ / SERVER_RANK_SIZE) * SERVER_RANK_SIZE;
    uint32_t targetLocalExpertId = targetLocalServerExpertId % localMoeExpertNum_;
    uint64_t targetIpcOffset =
        (buffers.offsetReduceLocal.GetValue(offsetIndex) % offsetNumPerExpert) * (axisH_ + WEIGHT_VALUE_NUM);

    shareMemGlobal_.SetGlobalBuffer(
        (__gm__ ExpandXType *)addrInfo_.GetIpcDataAddr(targetIpcRank, targetLocalExpertId, targetRankId));
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopy(buffers.dataInLocal, shareMemGlobal_[targetIpcOffset], axisH_ + WEIGHT_VALUE_NUM); // mte2
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    float scaleVal = buffers.inUbTemp(0);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(buffers.castInFloatLocal, buffers.dataInLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    if ((offsetIndex - offsetIndexStart) == 0U) {
        Muls(buffers.sumFloatLocal, buffers.castInFloatLocal, scaleVal, axisH_);
    } else {
        Axpy(buffers.sumFloatLocal, buffers.castInFloatLocal, scaleVal, axisH_);
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::ReduceWindowTokens(
    WindowSumBuffers &buffers, uint32_t targetServerId, uint32_t targetRankId, uint32_t coreNumPerServer,
    uint32_t totalCopyLen, uint32_t offsetNumPerExpert)
{
    uint32_t processTokenNum = 0;
    uint32_t offsetIndexStart = buffers.offsetIndexs(processTokenNum);
    offsetIndex = buffers.offsetIndexs(processTokenNum);
    uint32_t copyNum = buffers.copyNums(processTokenNum);
    uint32_t dataOffset = buffers.dataOffsets(processTokenNum);
    uint32_t tokenOffset = 0;
    auto flagDataCopyParams = DataCopyParams{1U, static_cast<uint16_t>(sizeof(uint64_t)), 0, 0};
    shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)addrInfo_.GetIpcTokenFlagAddr(targetServerId));
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0U; i < totalCopyLen; i++) {
        AccumulateIpcWindowToken(buffers, targetRankId, offsetNumPerExpert, offsetIndexStart);
        offsetIndex += 1U;

        PipeBarrier<PIPE_V>();
        if ((offsetIndex - offsetIndexStart) == copyNum) {
            tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
            if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
                if constexpr (std::is_same<ExpandXType, half>::value) {
                    QuantizeFp16WindowToken(buffers, dataOffset, tokenOffset, flagDataCopyParams);
                } else {
                    QuantizeBf16WindowToken(buffers, dataOffset, tokenOffset, flagDataCopyParams);
                }
            } else {
                PipeBarrier<PIPE_V>();
                Cast(buffers.dataInLocal, buffers.sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopy(localOutWindow_[tokenOffset * axisH_], buffers.dataInLocal, axisH_); // FP16/BF16数据
                PipeBarrier<PIPE_MTE3>();
                DataCopyPad(shareFlagGlobal_[tokenOffset], buffers.rdmaFlagLocal, flagDataCopyParams);
            }
            processTokenNum++;
            offsetIndex = buffers.offsetIndexs(processTokenNum);
            copyNum = buffers.copyNums(processTokenNum);
            dataOffset = buffers.dataOffsets(processTokenNum);
            offsetIndexStart = offsetIndex;
        }
    }
    PipeBarrier<PIPE_ALL>();
    buffers.rdmaFlagLocal(0) = RDMA_TOKEN_END_FLAG + magicValue_;
    tokenOffset = coreNumPerServer * processTokenNum + coreIdx_ % coreNumPerServer;
    DataCopyPad(shareFlagGlobal_[tokenOffset], buffers.rdmaFlagLocal, flagDataCopyParams);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToWindow()
{
    WindowSumBuffers buffers;
    // 32core流水并行
    uint32_t offsetNumPerExpert = globalBs_;
    uint32_t coreNumPerServer = stepCoreNum / serverNum_;
    uint32_t targetServerId = coreIdx_ / coreNumPerServer;
    uint32_t targetRankId = rankId_ % SERVER_RANK_SIZE + targetServerId * SERVER_RANK_SIZE;

    uint32_t baseBuffOffset = 0U;
    uint32_t totalCopyLen = 0U;
    BuildWindowTokenList(buffers, targetServerId, coreNumPerServer, baseBuffOffset, totalCopyLen);
    InitWindowSumBuffers(buffers, baseBuffOffset);
    localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(addrInfo_.GetLocalSendBuffDataAddr(targetRankId)));
    buffers.rdmaFlagLocal = statusBuf_.Get<uint64_t>();
    buffers.rdmaFlagLocal(0) = RDMA_TOKEN_ARRIVED_FLAG + magicValue_;
    ReduceWindowTokens(buffers, targetServerId, targetRankId, coreNumPerServer, totalCopyLen, offsetNumPerExpert);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitRdmaTokens(
    LocalTensor<uint64_t> &checkRdmaLocal, uint32_t copyOnceNum, uint32_t &copySum, bool &stopFlag,
    const DataCopyExtParams &flagDataCopyParams, const DataCopyPadExtParams<uint64_t> &flagPadParams)
{
    for (uint32_t i = 0U; i < copyOnceNum; i++) {
        while (true) {
            DataCopyPad(checkRdmaLocal[64], shareFlagGlobal_[copySum], flagDataCopyParams, flagPadParams);
            PipeBarrier<PIPE_ALL>();
            if (checkRdmaLocal.GetValue(64) == (RDMA_TOKEN_ARRIVED_FLAG + magicValue_)) {
                copySum++;
                break;
            } else if (checkRdmaLocal.GetValue(64) == (RDMA_TOKEN_END_FLAG + magicValue_) || copySum == maxBs_) {
                stopFlag = true;
                break;
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (stopFlag) {
            break;
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::CopyLocalServerTokens(
    LocalTensor<ExpandXTransType> &tmpLowUb_, GlobalTensor<ExpandXTransType> &aivSrcGlobal,
    GlobalTensor<ExpandXTransType> &aivDstGlobal, uint32_t copyLenAlign_, uint32_t copySum, uint32_t copyOnceNum)
{
    uint32_t cpNum = 0U;
    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        cpNum = axisH_ + scaleNumAlign * static_cast<uint32_t>(sizeof(ExpandXType)) /
                             static_cast<uint32_t>(sizeof(ExpandXTransType));
    } else {
        cpNum = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType)) / static_cast<uint32_t>(sizeof(ExpandXTransType));
    }
    for (uint32_t k = 0U; k < copyOnceNum; k++) {
        DataCopy(tmpLowUb_, aivSrcGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXTransType)],
                 cpNum);
        PipeBarrier<PIPE_ALL>();
        DataCopy(aivDstGlobal[copyLenAlign_ * (copySum - copyOnceNum + k) / sizeof(ExpandXTransType)], tmpLowUb_,
                 cpNum);
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::AlltoAllServerDispatch()
{
    LocalTensor<uint64_t> checkRdmaLocal = statusBuf_.Get<uint64_t>();
    LocalTensor<ExpandXTransType> tmpLowUb_ = tBuf.Get<ExpandXTransType>();
    uint32_t checkServer = coreIdx_ - stepCoreNum;
    GlobalTensor<ExpandXTransType> aivSrcGlobal;
    GlobalTensor<ExpandXTransType> aivDstGlobal;
    uint32_t dstRankId = rankId_ % SERVER_RANK_SIZE + SERVER_RANK_SIZE * checkServer;
    uint32_t copySum = 0;
    uint32_t copyOnceNum = 1;
    uint32_t copyLen_;
    uint32_t copyLenAlign_;

    if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
        copyLen_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
                   scaleNum * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXTransType)) +
                        scaleNumAlign * static_cast<uint32_t>(sizeof(ExpandXType));
    } else {
        copyLen_ = axisH_ * static_cast<uint32_t>(sizeof(ExpandXType));
        copyLenAlign_ = copyLen_;
    }
    uint64_t srcrdmaAddr = (uint64_t)(addrInfo_.GetLocalSendBuffDataAddr(dstRankId));
    uint64_t dstrdmaAddr = (uint64_t)(addrInfo_.GetRemoteRecvBuffDataAddr(dstRankId));
    auto flagDataCopyParams = DataCopyExtParams{1U, static_cast<uint16_t>(sizeof(uint64_t)), 0, 0, 0};
    auto flagPadParams = DataCopyPadExtParams<uint64_t>{false, 0, 0, 0};
    shareFlagGlobal_.SetGlobalBuffer((__gm__ uint64_t *)addrInfo_.GetIpcTokenFlagAddr(checkServer));
    DispatchServerTokens(checkRdmaLocal, tmpLowUb_, aivSrcGlobal, aivDstGlobal, dstRankId, copySum, copyOnceNum,
                         copyLen_, copyLenAlign_, srcrdmaAddr, dstrdmaAddr, flagDataCopyParams, flagPadParams);
    if (rankId_ != dstRankId) {
        AIVRDMAPostSend((GM_ADDR)((uint64_t)(readStateGlobal_.GetPhyAddr())),
                        (GM_ADDR)((uint64_t)(addrInfo_.GetRemoteRecvBuffFlagAddr(dstRankId))), dstRankId, 32, qp_info_);
    }
    aDump_.UpdateHierarchyInterNodeSendOrRecv(checkServer, copySum, Mc2A2Kernel::IS_SEND);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::DispatchServerTokens(
    LocalTensor<uint64_t> &checkRdmaLocal, LocalTensor<ExpandXTransType> &tmpLowUb_,
    GlobalTensor<ExpandXTransType> &aivSrcGlobal, GlobalTensor<ExpandXTransType> &aivDstGlobal, uint32_t dstRankId,
    uint32_t &copySum, uint32_t copyOnceNum, uint32_t copyLen_, uint32_t copyLenAlign_, uint64_t srcrdmaAddr,
    uint64_t dstrdmaAddr, const DataCopyExtParams &flagDataCopyParams,
    const DataCopyPadExtParams<uint64_t> &flagPadParams)
{
    bool stopFlag = false;
    while (!stopFlag) {
        WaitRdmaTokens(checkRdmaLocal, copyOnceNum, copySum, stopFlag, flagDataCopyParams, flagPadParams);
        if (copySum > 0U) {
            if (rankId_ != dstRankId) {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                AIVRDMAPostSend((GM_ADDR)(srcrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)),
                                (GM_ADDR)(dstrdmaAddr + copyLenAlign_ * (copySum - copyOnceNum)), dstRankId,
                                copyLen_ * copyOnceNum, qp_info_);
            } else {
                aivSrcGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(srcrdmaAddr));
                aivDstGlobal.SetGlobalBuffer((__gm__ ExpandXTransType *)(dstrdmaAddr));
                CopyLocalServerTokens(tmpLowUb_, aivSrcGlobal, aivDstGlobal, copyLenAlign_, copySum, copyOnceNum);
            }
        }
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::WaitDispatch()
{
    if ((coreIdx_ < serverNum_) && (coreIdx_ != (rankId_ / SERVER_RANK_SIZE))) {
        uint32_t srcRankId = rankId_ % SERVER_RANK_SIZE + coreIdx_ * SERVER_RANK_SIZE;
        LocalTensor<uint64_t> statusTensor = statusBuf_.Get<uint64_t>();
        uint32_t readNum = 1U;
        DataCopyParams intriParams{static_cast<uint16_t>(readNum), 1, 15, 0}; // srcStride为15个block
        GlobalTensor<uint64_t> statusSpaceGlobal;                             // win区状态位置拷入相关参数
        statusSpaceGlobal.SetGlobalBuffer((__gm__ uint64_t *)addrInfo_.GetLocalRecvBuffFlagAddr(srcRankId));

        int64_t startTime = GetSystemCycle() / TIME_CYCLE;
        while (true) {
            DataCopy(statusTensor, statusSpaceGlobal, intriParams);
            PipeBarrier<PIPE_ALL>();
            uint64_t sumOfFlag = statusTensor.GetValue(0);
            if (sumOfFlag == sumTarget_) {
                break;
            }
        }
        if (unlikely(needPerformanceInfo_)) {
            RecordRankCommDuration(performanceInfoI32Tensor_, srcRankId, startTime);
        }
        uint32_t dstServerId = coreIdx_;
        aDump_.UpdateHierarchyInterNodeSendOrRecv(dstServerId, 0U, Mc2A2Kernel::IS_RECV);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Preload()
{
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        return;
    }
    processNum = axisBS_ / reduceCore;
    resNum = axisBS_ - processNum * reduceCore;
    startBs = 0U;
    endBs = 0U;
    if (coreIdx_ < resNum) {
        processNum += 1U;
        startBs = coreIdx_ * processNum;
        endBs = startBs + processNum;
    } else {
        startBs = coreIdx_ * processNum + resNum;
        endBs = startBs + processNum;
    }

    // 初始化offset
    uint32_t baseBuffOffset = TBUF_TEMP_OFFSET;
    offsetReduceLocal_ = tBuf.GetWithOffset<int32_t>(
        RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);
    baseBuffOffset += sizeof(uint32_t) * RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));

    countReduceLocal_ =
        tBuf.GetWithOffset<int32_t>(RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t))), baseBuffOffset);

    DataCopy(offsetReduceLocal_, offsetOuterGlobal_,
             RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t))));
    DataCopy(countReduceLocal_, countOuterGlobal_, RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)))); // 256 * 4
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumFp16ToServer(
    ServerSumBuffers &buffers, int copyNum, uint32_t i)
{
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate(buffers.sumFp16Local, static_cast<ExpandXType>(0.0), axisH_);
    for (int j = 0; j < copyNum; j++) {
        int offsetOnIpc = PrepareQuantizedRecvWindow();
        DataCopy(buffers.dataInt8, localInWindow_[offsetOnIpc], axisH_);
        DataCopy(buffers.scaleData,
                 localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)],
                 scaleNumAlign);

        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(buffers.castFp16, buffers.dataInt8, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Brcb(buffers.scaleDup, buffers.scaleData, repeatNum, {1, 8}); // 填充scale值
        PipeBarrier<PIPE_V>();
        MulAddDst(buffers.sumFp16Local, buffers.castFp16, buffers.scaleDup, axisH_); // fp16乘加scale值
        PipeBarrier<PIPE_V>();

        offsetIndex++;
    }
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(expandOutGlobal_[i * axisH_], buffers.sumFp16Local, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumBf16ToServer(
    ServerSumBuffers &buffers, int copyNum, uint32_t i)
{
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Duplicate(buffers.sumFloatLocal1, 0.0f, axisH_);

    for (int j = 0; j < copyNum; j++) {
        int offsetOnIpc = PrepareQuantizedRecvWindow();
        DataCopy(buffers.dataInUbInt8, localInWindow_[offsetOnIpc], axisH_);
        DataCopy(buffers.scaleDataBf16,
                 localInScaleWindow_[((offsetOnIpc + axisH_) * sizeof(ExpandXTransType)) / sizeof(ExpandXType)],
                 scaleNumAlign);

        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // cast before muls
        Cast(buffers.castDataHalf, buffers.dataInUbInt8, AscendC::RoundMode::CAST_NONE, axisH_); // data:int8->fp16
        PipeBarrier<PIPE_V>();
        Cast(buffers.castDataFp32, buffers.castDataHalf, AscendC::RoundMode::CAST_NONE, axisH_); // data:fp16->fp32
        PipeBarrier<PIPE_V>();
        Cast(buffers.castFp32scale, buffers.scaleDataBf16, AscendC::RoundMode::CAST_NONE, scaleNum); // scale:bf16->fp32
        PipeBarrier<PIPE_V>();
        Brcb(buffers.castFp32ScaleBrcb, buffers.castFp32scale, repeatNum, {1, 8}); // 填充fp32的scale值
        PipeBarrier<PIPE_V>();
        MulAddDst(buffers.sumFloatLocal1, buffers.castDataFp32, buffers.castFp32ScaleBrcb, axisH_); // fp16乘加scale值
        PipeBarrier<PIPE_V>();
        offsetIndex++;
    }
    PipeBarrier<PIPE_V>();
    Cast(buffers.sumBf16Local, buffers.sumFloatLocal1, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(expandOutGlobal_[i * axisH_], buffers.sumBf16Local, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumUnquantizedToServer(
    ServerSumBuffers &buffers, int copyNum, uint32_t i)
{
    Duplicate(buffers.sumFloatLocal, 0.0f, axisH_);
    for (int j = 0; j < copyNum; j++) {
        int offsetOnIpc = (offsetReduceLocal_.GetValue(offsetIndex) % axisBS_) * axisH_;
        uint32_t srcServerId = offsetReduceLocal_.GetValue(offsetIndex) / axisBS_;
        uint32_t srcRankId = srcServerId * SERVER_RANK_SIZE + rankId_ % SERVER_RANK_SIZE;
        GM_ADDR addr = addrInfo_.GetLocalRecvBuffDataAddr(srcRankId);
        localInWindow_.SetGlobalBuffer((__gm__ ExpandXTransType *)(addr));
        SyncFunc<AscendC::HardEvent::V_MTE2>(); // 下一个token用的buffer和上一个token用的buffer之间进行同步
        DataCopy(buffers.dataIn, localInWindow_[offsetOnIpc], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // cast before muls
        Cast(buffers.castFp32, buffers.dataIn, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        // add mulBufLocal to sumFloatBufLocal
        AscendC::Add(buffers.sumFloatLocal, buffers.sumFloatLocal, buffers.castFp32, axisH_);
        offsetIndex++;
    }
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    Cast(buffers.sumFpAndBfLocal, buffers.sumFloatLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(expandOutGlobal_[i * axisH_], buffers.sumFpAndBfLocal, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitServerSumBuffers(
    ServerSumBuffers &buffers)
{
    // 初始化 fp16  bf16的offset
    uint32_t baseBuffOffset = sizeof(uint32_t) * RoundUp(axisBS_ * serverNum_, (uint32_t)(UB_ALIGN / sizeof(int32_t))) +
                              sizeof(int32_t) * RoundUp(axisBS_, (uint32_t)(UB_ALIGN / sizeof(int32_t)));
    uint32_t fpBaseBuffOffset = baseBuffOffset;
    uint32_t bfBaseBuffOffset = baseBuffOffset;

    InitServerUnquantizedBuffers(buffers, baseBuffOffset);
    InitServerFp16Buffers(buffers, fpBaseBuffOffset);
    InitServerBf16Buffers(buffers, bfBaseBuffOffset);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitServerUnquantizedBuffers(
    ServerSumBuffers &buffers, uint32_t &baseBuffOffset)
{
    // 不量化
    buffers.sumFloatLocal = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);
    buffers.sumFpAndBfLocal = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(float);

    buffers.dataIn = tBuf.GetWithOffset<ExpandXType>(axisH_, baseBuffOffset);
    baseBuffOffset += axisH_ * sizeof(ExpandXType);
    buffers.castFp32 = tBuf.GetWithOffset<float>(axisH_, baseBuffOffset);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitServerFp16Buffers(
    ServerSumBuffers &buffers, uint32_t &fpBaseBuffOffset)
{
    // 量化 fp16
    buffers.sumFp16Local = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    buffers.dataInt8 = tBuf.GetWithOffset<ExpandXTransType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    buffers.scaleData = tBuf.GetWithOffset<ExpandXType>(scaleNumAlign, fpBaseBuffOffset);
    fpBaseBuffOffset += scaleNumAlign * sizeof(ExpandXType);

    buffers.castFp16 = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
    fpBaseBuffOffset += axisH_ * sizeof(ExpandXType);

    buffers.scaleDup = tBuf.GetWithOffset<ExpandXType>(axisH_, fpBaseBuffOffset);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::InitServerBf16Buffers(
    ServerSumBuffers &buffers, uint32_t &bfBaseBuffOffset)
{
    // 量化 bf16
    buffers.sumFloatLocal1 = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    buffers.sumBf16Local = tBuf.GetWithOffset<ExpandXType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    buffers.dataInUbInt8 = tBuf.GetWithOffset<ExpandXTransType>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(ExpandXTransType);

    buffers.scaleDataBf16 = tBuf.GetWithOffset<ExpandXType>(scaleNumAlign, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign * sizeof(ExpandXType);

    buffers.castDataHalf = tBuf.GetWithOffset<half>(axisH_, bfBaseBuffOffset); // Bf16 用half代替
    bfBaseBuffOffset += axisH_ * sizeof(half);

    buffers.castDataFp32 = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
    bfBaseBuffOffset += axisH_ * sizeof(float);

    buffers.castFp32scale = tBuf.GetWithOffset<float>(scaleNum, bfBaseBuffOffset);
    bfBaseBuffOffset += scaleNumAlign * sizeof(float);

    buffers.castFp32ScaleBrcb = tBuf.GetWithOffset<float>(axisH_, bfBaseBuffOffset);
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::SumToServer()
{
    ServerSumBuffers buffers;
    uint32_t reduceCore = 8U;
    if (coreIdx_ >= reduceCore) {
        SyncAll<true>();
        return;
    }
    InitServerSumBuffers(buffers);
    for (uint32_t i = startBs; i < endBs; i++) {
        int copyNum = countReduceLocal_.GetValue(i);
        offsetIndex = i * serverNum_;
        PipeBarrier<PIPE_ALL>(); // 高精度为了同步加入的 PIPE_ALL
        if constexpr (DynamicQuant && std::is_same<ExpandXTransType, int8_t>::value) {
            if constexpr (std::is_same<ExpandXType, half>::value) { // fp16
                SumFp16ToServer(buffers, copyNum, i);
            } else { // bf16
                SumBf16ToServer(buffers, copyNum, i);
            }
        } else {
            SumUnquantizedToServer(buffers, copyNum, i);
        }
    }

    SyncAll<true>();
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::CopyPerformanceInfo()
{
    if (unlikely(needPerformanceInfo_)) {
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopy(performanceInfoI32GMTensor_, performanceInfoI32Tensor_,
                          performanceInfoSize_ * sizeof(int64_t) / sizeof(int32_t));
        AscendC::SetAtomicNone();
    }
}

template <TemplateMC2TypeA2layeredClass>
__aicore__ inline void MoeDistributeCombineA2Layered<TemplateMC2TypeA2layeredFunc>::Process()
{
    if ASCEND_IS_AIV {
        GM2IPC();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_HIERARCHY_COMBINE_GM2IPC);
        WaitIPC();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_HIERARCHY_COMBINE_WAITIPC);
        stepCoreNum = IPC_REDUCE_USED_CORE_NUM;
        if (coreIdx_ < stepCoreNum) {
            SumToWindow();
        } else if (coreIdx_ < (stepCoreNum + serverNum_)) {
            AlltoAllServerDispatch();
        }
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_HIERARCHY_COMBINE_INTER_SEND);
        SyncAll<true>();
        Preload();
        WaitDispatch();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_HIERARCHY_COMBINE_INTER_RECV);
        SumToServer();
        CopyPerformanceInfo();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_END);
        hccl_.Finalize();
    }
}
} // namespace MoeDistributeCombineA2Impl
#endif // MOE_DISTRIBUTE_COMBINE_A2_LAYERED_H
