/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 **/

/*!
 * \file moe_distribute_combine_v2_host_kfc.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_HOST_KFC_H
#define MOE_DISTRIBUTE_COMBINE_V2_HOST_KFC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "../moe_distribute_combine_v2_tiling.h"
#if __has_include("../../common/moe_distribute_base.h")
#include "../../common/moe_distribute_base.h"
#include "../../moe_distribute_dispatch_v2/check_winsize.h"
#include "../../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/check_winsize.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

namespace MoeDistributeCombineV2Impl {
constexpr uint8_t BUFFER_NUM = 2;                      // 多buf
constexpr uint32_t STATE_OFFSET = 32U;                 // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024UL * 1024UL;       // 1M
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U; // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint32_t UB_ALIGN = 32U; // UB按32字节对齐
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 975UL * 1024UL; // 预留48*512内存
constexpr uint64_t TIMEOUT_OFFSET = 1000UL * 1024UL;
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint64_t TIMEOUT_DETECTION_TX_UNITS = 8UL;
constexpr uint64_t STATE_HCCL_OFFSET = 32UL;
constexpr uint32_t EXPAND_IDX_INFO = 3U; // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ALIGNED_LEN = 256U;   // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr float SCALE_PARAM = 127.0;     // 计算量化参数所需的缩放倍数
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t REDUCE_NUM = 8U;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1U;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2U;
constexpr uint8_t MOE_NUM_IDX = 3U;
constexpr uint32_t DIM_NUM = 2;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t BLOCK_NUM = ALIGNED_LEN / UB_ALIGN; // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr size_t SYSTEM_NEED_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr size_t MAX_REDUCE_TILE_SIZE = 32;
constexpr uint32_t BATCH_SIZE = 4;
constexpr uint64_t READY_FLAG = 12345UL;
constexpr uint64_t PACKED_TOKEN_READY_FLAG = 1UL;
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint8_t MAX_SERVER_RANK_SIZE = 16;
constexpr uint32_t SHARE_FLAG_ENTRY_STRIDE_BYTES = 512U;
constexpr uint32_t TOKEN_META_BYTES = 12U;    // scale(float)+originRankId(u32)+originTokenId(u32)
constexpr uint32_t TOKEN_ONE_META_BYTES = 4U; // originTokenId(u32)
constexpr uint32_t FLAG_CNT_U64 = 2;          // flag + cnt 数量为2
constexpr size_t TARGET_CNT_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t BLOCK_COPY_BYTES_FIRST = 256U; // 第一个copy chunk的字节数 8*32B
constexpr uint32_t BLOCK_COPY_BYTES_SECOND = SPLIT_BLOCK_SIZE - BLOCK_COPY_BYTES_FIRST; // 第二个copy chunk的字节数
constexpr uint32_t MIN_AVAIL_UB_SIZE = 8192U;                                           // 单位字节
constexpr uint32_t SCALE_FACTOR_CONSTANT = 4U;                                          // 单位字节

#define CombineV2HostTypeClass typename ExpandXType, typename XType, typename ExpandIdxType, bool IsInt8Quant
#define CombineV2HostTypeFunc ExpandXType, XType, ExpandIdxType, IsInt8Quant

using namespace MoeDistributeV2Base;
using namespace AscendC;

template <CombineV2HostTypeClass>
class MoeDistributeCombineV2HostKfc {
public:
    __aicore__ inline MoeDistributeCombineV2HostKfc(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR expertScales, GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
                                GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                              GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR expandScales,
                                              GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo,
                                              GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                              GM_ADDR constExpertV, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void BuffInit();
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void InitAlltoAllBuffers();

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return (x > y) ? x : y;
    }

    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

    template <typename T>
    inline __aicore__ T DivCeil(const T val, const T div)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (div == 0 || val + div - 1 < val) {
            return val;
        }
        return (val + div - 1) / div;
    }

    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId)
    {
        return GetBaseWindAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winDataSizeOffsetEp_;
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId)
    {
        return GetBaseWindStateAddrByRankId(epWinContext_, rankId, epRankIdOriginal_) + winStatusOffset_;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> elasticInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<float> expandScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_; // 用于存对端window的变量
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    GM_ADDR epWindowGM_;
    GM_ADDR stateGM_;
    GM_ADDR maskCalcWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<float> stateResetTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t epRankId_{0};
    uint32_t epRankIdOriginal_{0};
    uint32_t coreIdx_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0}; // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};          // moeExpertPerRankNum_ * epWorldSize_
    uint32_t bufferNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t moeExpertNum_{0};

    __gm__ HcclOpParam *epWinContext_{nullptr};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint32_t selfSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hAlign32Size_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t expertScaleBeginIdx_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;

    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> expertScalesBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> expertMaskBuf_;
    TBuf<> elasticInfoBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool enableSpecialExpert_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<XType> scaleDivTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<float> expertScalesLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t scaleNum_{0};
    float scaleValFloat_;

private:
    __aicore__ inline void CommunInit(const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR workspaceGM);
    __aicore__ inline void SplitCoreByToken(const uint32_t totalSendCnt);
    __aicore__ inline void SplitCoreByServer();
    // 偏移量计算相关函数
    __aicore__ inline void PrepareServerShareLayout(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void CalcLocalTargetCnt(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void BuildPrefixBaseOnCore0();
    __aicore__ inline void LoadLocalBaseFromGm();
    // 1、Server内通信
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void SumToWindow();
    __aicore__ inline void DispatchTokensToShareMem(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void DispatchTokenInner(uint32_t globalIdx, uint32_t originRankId, uint32_t originTokenId,
                                              uint32_t topkId, uint32_t tokenIdInServer, uint64_t shareDataAddr);
    __aicore__ inline void UpdateShareFlag();
    __aicore__ inline void ProcessOneServer(uint32_t toServerId, LocalTensor<float> sumTileUb,
                                            LocalTensor<uint32_t> existFlagUb);
    __aicore__ inline void AccumulateRankDataToUb(uint32_t fromLocalRank, uint32_t targetServerId, uint32_t baseId,
                                                  uint32_t endId, LocalTensor<float> sumTileUb,
                                                  LocalTensor<uint32_t> existFlagUb);
    __aicore__ inline void ProcessBatchTokens(LocalTensor<ExpandXType> inputBatchUb, uint32_t curBatch,
                                              uint32_t targetServerId, uint32_t baseId, uint32_t endId,
                                              LocalTensor<float> sumTileUb, LocalTensor<uint32_t> existFlagUb,
                                              LocalTensor<float> tmpUb);
    __aicore__ inline void ReadRankTokenCnt(uint32_t fromLocalRank, uint32_t &tokenCnt, GM_ADDR shareBase);
    __aicore__ inline void TokenToWinOut(GM_ADDR dstTokenBase, uint32_t tokenIdInServer,
                                         LocalTensor<float> srcSumTensor);
    __aicore__ inline void WriteWinOutHeader(GlobalTensor<uint64_t> headerGm, uint32_t winTokenCnt);
    // 2、Server间通信
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void WaitWinInCount();
    __aicore__ inline void WaitTokenBlockReady(GM_ADDR winInTkAddr, uint32_t countIdx);
    __aicore__ inline void LoadTokenToUb(LocalTensor<ExpandXType> &outTokenUb, GM_ADDR winInTkAddr, uint32_t countIdx);
    __aicore__ inline void AlltoAllCombine();
    __aicore__ inline void LoadTokenCounts(LocalTensor<uint32_t> &tokenCntArray);
    __aicore__ inline void ProcessSingleToken(uint32_t tokenId, LocalTensor<uint32_t> &tokenCntArray);
    __aicore__ inline void SearchAndAccumulateToken(uint32_t targetTokenId, LocalTensor<uint32_t> &tokenCntArray,
                                                    LocalTensor<float> &sumLocal, bool &foundAny);
    __aicore__ inline void AccumulateTokenFromServer(GM_ADDR winInTkAddr, uint32_t hitIdx,
                                                     LocalTensor<float> &sumLocal);
    __aicore__ inline void ProcessSharedExpertAndOutput(uint32_t tokenId, LocalTensor<float> &sumLocal);

private:
    uint32_t globalBs_{0};
    uint32_t serverTokenBs_{0};
    uint32_t tokenIdBaseInServer_{0};
    uint32_t tokenIdEndInServer_{0};
    // ================= 分核（按 toServerId） =================
    uint32_t startServerId_{0};
    uint32_t endServerId_{0};
    uint32_t sendServerNum_{0}; // 本核负责的 toServer 数
    // ================= Server 拓扑 =================
    uint32_t serverRankSize_{0}; // 一个 Server 内 Rank 数（host 传入）
    uint32_t serverNum_{0};      // epWorldSize_ / serverRankSize_
    uint32_t serverId_{0};       // 本卡所在Server Id
    uint32_t localRankId_{0};    // Server 内本卡的局部 Rank Id
    // ================= Token  =================
    uint32_t maxLocalBs_{0};
    uint32_t tokenElemNum_{0};        // axisH， 每个 token 包含的元素个数
    uint32_t tokenDataBytes_{0};      // axisH * sizeof(ExpandXType)
    uint32_t tokenDataBytesAlign_{0}; // tokenDataBytes_ 按 UB_ALIGN 对齐后的大小
    uint32_t metaBytesAlign_{0};      // 12Bytes的三元组UB对其后的字节数，32B
    uint32_t tokenMetaBytes_{0};      // tokenData + MetaInfo 按 UB_ALIGN 对齐后的大小
    uint32_t tokenDataBlockNum_{0};   // WinOut中每个 token拆分成多少个480Block
    uint32_t packedTokenBytes_{0};    // 处理后的token在WinOut中占用的字节数
    // ================= Win区 布局 =================
    // Server之间通信
    uint32_t winHeaderBytes_{SPLIT_BLOCK_SIZE};
    uint32_t winOutSliceBytes_{0};
    uint32_t winOutTotalBytes_{0};
    uint32_t winInSliceBytes_{0};

    uint32_t safeReduceTileSize_{MAX_REDUCE_TILE_SIZE};
    uint32_t winInTotalBytes_{0};
    // Server内通信
    uint32_t shareFlagSliceBytes_{0};
    uint32_t shareFlagTotalBytes_{0};
    uint32_t shareDataSliceBytes_{0};
    uint32_t shareDataTotalBytes_{0};
    uint64_t flagPadOffset_{0};
    uint32_t rowStrideBytes_{0};
    uint32_t rowStrideElems_{0};
    uint32_t coreTableElems_{0};
    // ================= ShareData 布局 =================
    uint32_t flagU64CopyCntAlign_{0}; // 每个 fromLocalRank 需要拷贝的 flag + cnt 数量（uint64_t）

    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    uint32_t localMoeExpertNum_{0}; // 每张卡的moe专家数
    GM_ADDR workspaceGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_; // Server之间通信使用
    // Server 内共享区基址
    uint64_t serverShareAddr_[MAX_SERVER_RANK_SIZE]; // [targetLocalRank]
    // 共享数据token写偏移，shape: [aivNum_][serverRankSize_]
    GlobalTensor<uint32_t> gmCoreTargetCnt_;  // 每核->每 targetLocalRank 的 token 数
    GlobalTensor<uint32_t> gmCoreTargetBase_; // prefix base：每核->每 targetLocalRank 的起始 offset
    GM_ADDR cntGM_;
    GM_ADDR baseGM_;
    // UB侧，每核私有，用于运行时统计/递增。三个都是长度 serverRankSize_ 的小数组
    TBuf<> localTargetCntBuf_;
    TBuf<> localTargetBaseBuf_;
    TBuf<> localTargetRunBuf_;
    TBuf<> prefixBuf_;
    TBuf<> baseRowBuf_;
    TBuf<> shareFlagUbBuf_;
    LocalTensor<uint32_t> localTargetCnt_;            // 本核 token → targetLocalRank 的计数
    LocalTensor<uint32_t> localTargetBase_;           // 本核从 gmCoreTargetBase_ 读回的 base
    LocalTensor<uint32_t> localTargetRun_;            // 本核运行时递增 offset
    LocalTensor<uint32_t> rankTotalCntToTargetLocal_; // 本rank总的 token → targetLocalRank 的计数
    LocalTensor<float> sumFloatLocal_;
    LocalTensor<uint64_t> batchWriteItemLocalB64;
    LocalTensor<uint32_t> batchWriteItemLocalB32;
    GlobalTensor<ExpandXType> tokenLocalOutWindow_;
    GlobalTensor<ExpandXType> tokenLocalInWindow_;
    GlobalTensor<uint64_t> flagLocalOutWindow_;
    GlobalTensor<uint64_t> flagLocalInWindow_;
    LocalTensor<uint32_t> countTensor;
    TBuf<> countBuf_;
    TBuf<> sumBuf_;
    TBuf<> tokenIdBuf_;
    TBuf<> localCntBuf_;
    TBuf<> localTokenIdBuf_;
    TBuf<> localOutTensorBuf_;
    TBuf<> localFlagBuf_;
    TBuf<> tempBuf_;
    TBuf<> outBuf1_;
    TBuf<> localOutTempBuf_;
    LocalTensor<uint64_t> localCntTensor_;
};

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
    GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
    TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;

    maskCalcWorkspaceGM_ = workspaceGM + SYSTEM_NEED_WORKSPACE + coreIdx_ * MASK_CALC_NEED_WORKSPACE;
    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, expandScales, xActiveMask,
                       sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut);
    InitAttrs(tilingData);
    // 检查hcclwinsize是否越界
    auto realWinSize = GetWinSize(epWinContext_);
    CheckWindowSize(totalWinSizeEp_, realWinSize, tpipe_, XOut);
    PipeBarrier<PIPE_ALL>();
    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffsetEp_ =
        static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeEp / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWinAddrByRankId(epRankIdOriginal_);
    CommunInit(tilingData, workspaceGM);

#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    for (int tempepRankId = 0; tempepRankId < epWorldSize_; tempepRankId++) {
        OOMCheckAddrRange<XType>((__gm__ XType *)(GetWinAddrByRankId(tempepRankId)), totalWinSizeEp_);
        OOMCheckAddrRange<float>((__gm__ float *)(GetWinStateAddrByRankId(tempepRankId)), STATE_SIZE);
    }
#endif
    if (isShareExpertRankFlag_) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1); // moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    }
    SplitCoreByToken(selfSendCnt_);
    SplitCoreByServer();
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, tokenMetaBytes_);
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::CommunInit(
    const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR workspaceGM)
{
    // 1. Server 拓扑
    serverRankSize_ = 2;
    serverNum_ = epWorldSize_ / serverRankSize_;
    localRankId_ = epRankId_ % serverRankSize_;
    serverId_ = epRankId_ / serverRankSize_;
    localMoeExpertNum_ = moeExpertNum_ / epWorldSize_;
    serverTokenBs_ = globalBs_;
    tokenIdBaseInServer_ = localRankId_ * axisBS_; // 本卡在 server 内的 base
    tokenIdEndInServer_ = tokenIdBaseInServer_ + axisBS_;
    // 2. Token
    tokenElemNum_ = axisH_;
    tokenDataBytes_ = axisH_ * sizeof(ExpandXType);
    tokenDataBytesAlign_ = RoundUp<uint32_t>(tokenDataBytes_, UB_ALIGN); // UB_ALIGN=32B
    metaBytesAlign_ = RoundUp<uint32_t>(TOKEN_META_BYTES, UB_ALIGN);
    tokenMetaBytes_ = tokenDataBytesAlign_ + metaBytesAlign_;
    flagU64CopyCntAlign_ = RoundUp<uint32_t>(FLAG_CNT_U64 * sizeof(uint64_t), UB_ALIGN) / sizeof(uint64_t); // 4
    maxLocalBs_ = globalBs_ * axisK_ / serverRankSize_;
    // token 数据需要的 480B block 数（向上取整）
    tokenDataBlockNum_ = DivCeil<uint32_t>(tokenDataBytes_, SPLIT_BLOCK_DATA_SIZE);
    // WinOut Data 区：一个 token 的 bytes
    packedTokenBytes_ = static_cast<uint64_t>(tokenDataBlockNum_ + 1) * WIN_ADDR_ALIGN; // WIN_ADDR_ALIGN=512B
    // 每个 toServer slice 总大小（bytes）：header(512B) + maxLocalBs_ * tokenStride
    winOutSliceBytes_ = WIN_ADDR_ALIGN + maxLocalBs_ * packedTokenBytes_;
    winOutTotalBytes_ = winOutSliceBytes_ * serverNum_; // 发送区
    winInTotalBytes_ = winOutTotalBytes_;               // 接收区
    windowOutGM_ = GetWinAddrByRankId(epRankId_);
    windowInGM_ = windowOutGM_ + winOutTotalBytes_;
    // 3. Share 区
    shareFlagSliceBytes_ = WIN_ADDR_ALIGN;
    shareFlagTotalBytes_ = shareFlagSliceBytes_ * serverRankSize_; // share区的flag区
    shareDataSliceBytes_ = tokenMetaBytes_ * maxLocalBs_;
    shareDataTotalBytes_ = shareDataSliceBytes_ * serverRankSize_; // share区的data区
    for (int i = 0; i < MIN(serverRankSize_, MAX_SERVER_RANK_SIZE); i++) {
        // 一个Server内的全部RankId号，epRankId_为本Rank的Id号
        uint32_t rankIdServerInner = epRankId_ / serverRankSize_ * serverRankSize_ + i;
        serverShareAddr_[i] =
            reinterpret_cast<uint64_t>(winOutTotalBytes_ + winInTotalBytes_ + GetWinAddrByRankId(rankIdServerInner));
    }
    // 4. token offset 表
    const uint32_t rowBytes = serverRankSize_ * sizeof(uint32_t);
    rowStrideBytes_ = RoundUp<uint32_t>(rowBytes, 64); // 64B cacheline
    rowStrideElems_ = rowStrideBytes_ / sizeof(uint32_t);
    coreTableElems_ = aivNum_ * rowStrideElems_;
    cntGM_ = workspaceGM + SYSTEM_NEED_WORKSPACE + aivNum_ * MASK_CALC_NEED_WORKSPACE;
    baseGM_ = cntGM_ + static_cast<uint64_t>(coreTableElems_) * sizeof(uint32_t);
    gmCoreTargetCnt_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(cntGM_), coreTableElems_);
    gmCoreTargetBase_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(baseGM_), coreTableElems_);
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
    GM_ADDR expandScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut)
{
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float *)expertScales);
    expandScalesGM_.SetGlobalBuffer((__gm__ float *)expandScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType *)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t *)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType *)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertV);
    expandOutGlobal_.SetGlobalBuffer((__gm__ XType *)XOut);
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::InitTilingAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    globalBs_ = tilingData->moeDistributeCombineV2Info.globalBs;
    hasElasticInfoFlag_ = tilingData->moeDistributeCombineV2Info.hasElasticInfo;
    epWorldSizeOriginal_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    totalWinSizeEp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeEp;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    bufferNum_ = tilingData->moeDistributeCombineV2Info.bufferNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineV2Info.constExpertNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::InitAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpParam *)contextGM0;
    statusDataSpaceGm_ = GetStatusDataSpaceGm(epWinContext_); // (GM_ADDR)(epWinContext_->localWindowsExp);
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = 0; // 标志位，标志0区还是1区
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_; // 部署在所有Rank上的总专家数
    if (epRankId_ < sharedExpertRankNum) {
        isShareExpertRankFlag_ = true;
    }
    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
    // A5 Server内/间通信 Buffer初始化
    uint32_t floatTokenBytes = axisH_ * sizeof(float);
    uint32_t floatTokenBytesAlign = RoundUp<uint32_t>(floatTokenBytes, UB_ALIGN);
    tpipe_->InitBuffer(tempBuf_, BATCH_SIZE * tokenMetaBytes_);
    tpipe_->InitBuffer(localOutTempBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(shareFlagUbBuf_, (flagU64CopyCntAlign_) * sizeof(uint64_t));
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(localOutTempBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(tempBuf_, BATCH_SIZE * tokenMetaBytes_);
    tpipe_->InitBuffer(shareFlagUbBuf_, flagU64CopyCntAlign_ * sizeof(uint64_t));
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    uint32_t bsKFloatAlign = Ceil(bsKNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size_ > bsKFloatAlign ? hFloatAlign256Size_ : bsKFloatAlign;
    tpipe_->InitBuffer(outBuf1_, (tokenDataBlockNum_ * SPLIT_BLOCK_FLAG_SIZE));
    uint32_t floatTokenBytes = axisH_ * sizeof(float);
    uint32_t floatTokenBytesAlign = RoundUp<uint32_t>(floatTokenBytes, UB_ALIGN);
    uint32_t usedUbEstimate = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size_ +
                              tokenMetaBytes_ + (tokenDataBlockNum_ * SPLIT_BLOCK_FLAG_SIZE) +
                              UB_ALIGN * SCALE_FACTOR_CONSTANT;
    uint32_t availableUb = (usedUbEstimate < ubSize_) ? (ubSize_ - usedUbEstimate) : MIN_AVAIL_UB_SIZE;
    uint32_t perTokenBytes = floatTokenBytesAlign + sizeof(uint32_t);
    safeReduceTileSize_ = (perTokenBytes == 0) ? 1 : (availableUb / perTokenBytes);
    safeReduceTileSize_ = (safeReduceTileSize_ > MAX_REDUCE_TILE_SIZE) ? MAX_REDUCE_TILE_SIZE : safeReduceTileSize_;
    safeReduceTileSize_ = (safeReduceTileSize_ > axisBS_) ? axisBS_ : safeReduceTileSize_;
    safeReduceTileSize_ = (safeReduceTileSize_ < 1) ? 1 : safeReduceTileSize_;
    tpipe_->InitBuffer(sumBuf_, floatTokenBytesAlign * safeReduceTileSize_);
    // countBuf_ 需要能容纳 serverNum_ 个元素，取两者较大值避免越界写
    uint32_t countBufSize = MAX(safeReduceTileSize_, serverNum_);
    tpipe_->InitBuffer(countBuf_, countBufSize * sizeof(uint32_t));
    // Server间通信buffer初始化
    tpipe_->InitBuffer(localCntBuf_, UB_ALIGN);
    tpipe_->InitBuffer(localTokenIdBuf_, UB_ALIGN);
    tpipe_->InitBuffer(localOutTensorBuf_, tokenDataBytesAlign_);
    tpipe_->InitBuffer(localFlagBuf_, UB_ALIGN);
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::SplitCoreByToken(
    const uint32_t totalSendCnt)
{
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = totalSendCnt / aivNum_;
    uint32_t remainderRankNum = totalSendCnt % aivNum_;
    startTokenId_ = sendCntNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::SplitCoreByServer()
{
    // 要处理的总量是 serverNum_（toServerId 的数量）
    if (serverNum_ == 0 || aivNum_ == 0) {
        sendServerNum_ = 0;
        startServerId_ = 0;
        endServerId_ = 0;
        return;
    }
    // 均匀分核
    sendServerNum_ = serverNum_ / aivNum_;
    uint32_t remainder = serverNum_ % aivNum_;
    startServerId_ = sendServerNum_ * coreIdx_;
    if (coreIdx_ < remainder) {
        sendServerNum_++;
        startServerId_ += coreIdx_;
    } else {
        startServerId_ += remainder;
    }
    endServerId_ = startServerId_ + sendServerNum_;
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::PrepareServerShareLayout(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    // Step1：统计本核 -> 每个 targetLocalRank 的 token 数。写 gmCoreTargetCnt_[coreIdx_][t]
    CalcLocalTargetCnt(expandIdxLocal);
    AscendC::SyncAll<true>(); // 等所有核写完 cnt
    // Step2core0 计算 prefix base
    BuildPrefixBaseOnCore0();
    AscendC::SyncAll<true>(); // 等 base 计算完成
    // Step3：每核读回自己的 base，并清空 run
    LoadLocalBaseFromGm();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::CalcLocalTargetCnt(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    tpipe_->InitBuffer(localTargetCntBuf_, rowStrideBytes_);
    localTargetCnt_ = localTargetCntBuf_.Get<uint32_t>();

    for (uint32_t i = 0; i < rowStrideElems_; ++i) {
        localTargetCnt_.SetValue(i, 0U);
    }

    //  必须和 AlltoAllDispatch 的 token 遍历顺序一致
    for (uint32_t loop = 0; loop < sendCntNum_; ++loop) {
        const uint32_t baseOffset = loop * EXPAND_IDX_INFO;
        uint32_t originRankId = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        const uint32_t targetLocalRank = originRankId % serverRankSize_;
        const uint32_t cnt = localTargetCnt_.GetValue(targetLocalRank) + 1U;
        localTargetCnt_.SetValue(targetLocalRank, cnt);
    }
    // SyncFunc<AscendC::HardEvent::S_MTE3>();
    GlobalTensor<uint32_t> gmRow;
    gmRow.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t *>(cntGM_ + static_cast<uint64_t>(coreIdx_) * rowStrideBytes_));
    DataCopy(gmRow, localTargetCnt_, rowStrideElems_);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::BuildPrefixBaseOnCore0()
{
    if (coreIdx_ != 0) {
        return;
    }
    // UB：prefix[t] + baseRow[t] 两个数组，都按 rowStrideBytes_ 分配更省事
    tpipe_->InitBuffer(prefixBuf_, rowStrideBytes_);
    tpipe_->InitBuffer(baseRowBuf_, rowStrideBytes_);
    LocalTensor<uint32_t> prefixUb = prefixBuf_.Get<uint32_t>();
    LocalTensor<uint32_t> baseRowUb = baseRowBuf_.Get<uint32_t>();
    // prefix 清零（含 padding）
    for (uint32_t i = 0; i < rowStrideElems_; ++i) {
        prefixUb.SetValue(i, 0U);
    }
    // 逐行生成 base 并写回
    for (uint32_t c = 0; c < aivNum_; ++c) {
        // baseRowUb 清零（含 padding）
        for (uint32_t i = 0; i < rowStrideElems_; ++i)
            baseRowUb.SetValue(i, 0U);
        const uint32_t cntBase = c * rowStrideElems_;
        for (uint32_t t = 0; t < serverRankSize_; ++t) {
            uint32_t p = prefixUb.GetValue(t);
            baseRowUb.SetValue(t, p);
            uint32_t cnt = gmCoreTargetCnt_.GetValue(cntBase + t); // Scalar 读 GM（允许，读不需要 DCCI）
            prefixUb.SetValue(t, p + cnt);
        }
        GlobalTensor<uint32_t> baseGmRow;
        baseGmRow.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint32_t *>(baseGM_ + static_cast<uint64_t>(c) * rowStrideBytes_));
        DataCopy(baseGmRow, baseRowUb, rowStrideElems_);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::LoadLocalBaseFromGm()
{
    tpipe_->InitBuffer(localTargetBaseBuf_, serverRankSize_ * sizeof(uint32_t));
    tpipe_->InitBuffer(localTargetRunBuf_, serverRankSize_ * sizeof(uint32_t));
    localTargetBase_ = localTargetBaseBuf_.Get<uint32_t>();
    localTargetRun_ = localTargetRunBuf_.Get<uint32_t>();
    const uint32_t rowBase = coreIdx_ * rowStrideElems_;
    for (uint32_t t = 0; t < serverRankSize_; ++t) {
        localTargetBase_.SetValue(t, gmCoreTargetBase_.GetValue(rowBase + t));
        localTargetRun_.SetValue(t, 0U);
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AlltoAllDispatch()
{
    if (sendCntNum_ == 0U) {
        return;
    }
    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U,
                                      0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // 1、计算要发送的token的数量与偏移 gmCoreTargetCnt_ / gmCoreTargetBase_
    PrepareServerShareLayout(expandIdxLocal);
    AscendC::PipeBarrier<PIPE_ALL>();
    // 2、Server内AlltoAll发送 token：发往ShareData[targetLocalRank][fromLocalRank][tokenOffset]
    DispatchTokensToShareMem(expandIdxLocal);
    AscendC::SyncAll<true>(); // 等所有核把 ShareData 写完
    // 3、一个核负责写 ShareFlag
    UpdateShareFlag();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::DispatchTokensToShareMem(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        const uint32_t localIdx = loop;
        const uint32_t globalIdx = startTokenId_ + loop;
        const uint32_t baseOffset = localIdx * EXPAND_IDX_INFO;
        uint32_t originRankId = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        uint32_t originTokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1));
        uint32_t topkId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 2));
        uint32_t targetLocalRank = originRankId % serverRankSize_;
        uint32_t tokenIdInServer = targetLocalRank * axisBS_ + originTokenId;
        const uint32_t runVal = localTargetRun_.GetValue(targetLocalRank);
        const uint32_t tokenOffset = localTargetBase_.GetValue(targetLocalRank) + runVal;
        localTargetRun_.SetValue(targetLocalRank, runVal + 1U);
        const uint64_t targetShareBase = serverShareAddr_[targetLocalRank];
        const uint64_t shareDataAddr = targetShareBase + static_cast<uint64_t>(shareFlagTotalBytes_) +
                                       static_cast<uint64_t>(localRankId_) * shareDataSliceBytes_ +
                                       static_cast<uint64_t>(tokenOffset) * tokenMetaBytes_;
        DispatchTokenInner(globalIdx, originRankId, originTokenId, topkId, tokenIdInServer, shareDataAddr);
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::DispatchTokenInner(
    uint32_t globalIdx, uint32_t originRankId, uint32_t originTokenId, uint32_t topkId, uint32_t tokenIdInServer,
    uint64_t shareDataAddr)
{
    LocalTensor<ExpandXType> tokenUb = tempBuf_.Get<ExpandXType>();
    DataCopyExtParams inputCopyParams{1U, static_cast<uint32_t>(tokenDataBytes_), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    // 1) 生成 tokenData 写入 tokenUb
    DataCopyPad(tokenUb, expandXGM_[globalIdx * axisH_], inputCopyParams, copyPadExtParams);
    // 2) 写 meta：scaleVal + originRankId + originTokenId
    float scaleVal = 0.0f;
    if (topkId >= axisK_) {
        scaleVal = 1.0f;
    } else if (globalIdx < selfSendCnt_) {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                          AscendC::DcciDst::CACHELINE_OUT>(expandScalesGM_[globalIdx]);
        scaleVal = expandScalesGM_.GetValue(globalIdx);
    }
    LocalTensor<float> metaFP32 = tokenUb.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    metaFP32.SetValue(tokenDataBytesAlign_ / sizeof(float), scaleVal);
    LocalTensor<uint32_t> metaU32 = tokenUb.template ReinterpretCast<uint32_t>();
    metaU32.SetValue((tokenDataBytesAlign_ + sizeof(float)) / sizeof(uint32_t), originRankId);
    metaU32.SetValue((tokenDataBytesAlign_ + sizeof(float) + sizeof(uint32_t)) / sizeof(uint32_t), tokenIdInServer);
    // 3) 写 ShareData
    LocalTensor<uint8_t> payloadUb = tempBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> shareDataByteGm;
    shareDataByteGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(shareDataAddr));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(shareDataByteGm, payloadUb, static_cast<uint32_t>(tokenMetaBytes_));
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::UpdateShareFlag()
{
    if (coreIdx_ != 0) {
        return;
    }
    LocalTensor<uint64_t> flagUb = shareFlagUbBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
    for (uint32_t targetLocalRank = 0; targetLocalRank < serverRankSize_; ++targetLocalRank) {
        // 1) 汇总 cnt：Σ gmCoreTargetCnt_[c][targetLocalRank]
        uint64_t cnt = 0;
        for (uint32_t c = 0; c < aivNum_; ++c) {
            cnt += static_cast<uint64_t>(gmCoreTargetCnt_.GetValue(c * rowStrideElems_ + targetLocalRank));
        }
        // 2) 写 flag + cnt
        flagUb.SetValue(0, READY_FLAG);
        flagUb.SetValue(1, cnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GM_ADDR flagAddr = reinterpret_cast<__gm__ uint8_t *>(serverShareAddr_[targetLocalRank]) +
                           static_cast<uint64_t>(localRankId_) * shareFlagSliceBytes_;
        GlobalTensor<uint64_t> flagGm;
        flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
        DataCopy(flagGm, flagUb, flagU64CopyCntAlign_);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

// Server内加权求和
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::SumToWindow()
{
    // Step1. 等待本卡 ShareFlag[fromLocalRank][*] 全部 READY
    if (coreIdx_ == 0) {
        GM_ADDR shareFlagBase = reinterpret_cast<__gm__ uint8_t *>(serverShareAddr_[localRankId_]);
        LocalTensor<uint64_t> flagUb = shareFlagUbBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
        for (uint32_t fromLocalRank = 0; fromLocalRank < serverRankSize_; ++fromLocalRank) {
            GM_ADDR flagAddr = shareFlagBase + static_cast<uint64_t>(fromLocalRank) * shareFlagSliceBytes_;
            GlobalTensor<uint64_t> flagGm;
            flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
            while (true) {
                DataCopy(flagUb, flagGm, flagU64CopyCntAlign_);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                if (flagUb.GetValue(0) == READY_FLAG) {
                    break;
                }
            }
        }
    }
    AscendC::SyncAll<true>();
    // Step2. ShareData → WinOut（按 toServerId 分核）
    LocalTensor<float> sumTileUb = sumBuf_.Get<float>();
    LocalTensor<uint32_t> existFlagUb = countBuf_.Get<uint32_t>();
    for (uint32_t toServerId = startServerId_; toServerId < endServerId_; ++toServerId) {
        ProcessOneServer(toServerId, sumTileUb, existFlagUb);
    }
    AscendC::SyncAll<true>();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::ProcessOneServer(
    uint32_t toServerId, LocalTensor<float> sumTileUb, LocalTensor<uint32_t> existFlagUb)

{
    GM_ADDR winOutSliceBase = windowOutGM_ + (toServerId * winOutSliceBytes_);
    GlobalTensor<uint64_t> winOutHeaderGm;
    winOutHeaderGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(winOutSliceBase));
    GM_ADDR winOutDataBase = winOutSliceBase + WIN_ADDR_ALIGN;
    uint64_t currentWinDataOffset = 0;
    uint32_t currentWinTokenCnt = 0;
    for (uint32_t baseId = tokenIdBaseInServer_; baseId < tokenIdEndInServer_; baseId += safeReduceTileSize_) {
        uint32_t endId = baseId + safeReduceTileSize_;
        if (endId > tokenIdEndInServer_)
            endId = tokenIdEndInServer_;
        uint32_t currentTileLen = endId - baseId;
        // 1. 清零
        const uint32_t sumStrideF = hFloatAlign32Size_ / sizeof(float);
        Duplicate(sumTileUb, 0.0f, currentTileLen * sumStrideF);
        Duplicate(existFlagUb, static_cast<uint32_t>(0), currentTileLen);
        PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        // 2. 累加所有的Rank
        for (uint32_t fromRank = 0; fromRank < serverRankSize_; ++fromRank) {
            AccumulateRankDataToUb(fromRank, toServerId, baseId, endId, sumTileUb, existFlagUb);
        }
        PipeBarrier<PIPE_ALL>();
        // 3. 写入 WinOut
        for (uint32_t i = 0; i < currentTileLen; ++i) {
            if (existFlagUb.GetValue(i) == 0) {
                continue;
            }
            uint32_t tokenIdInServer = baseId + i;
            uint32_t originRankId = tokenIdInServer / axisBS_;
            uint32_t tokenServerId = originRankId / serverRankSize_;
            if (tokenServerId != toServerId)
                continue;
            GM_ADDR dstTokenBase = winOutDataBase + currentWinDataOffset;
            LocalTensor<float> tokenSumSlice = sumTileUb[i * sumStrideF];
            TokenToWinOut(dstTokenBase, tokenIdInServer, tokenSumSlice);

            currentWinTokenCnt++;
            currentWinDataOffset += packedTokenBytes_;
        }
    }
    WriteWinOutHeader(winOutHeaderGm, currentWinTokenCnt);
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AccumulateRankDataToUb(
    uint32_t fromLocalRank, uint32_t targetServerId, uint32_t baseId, uint32_t endId, LocalTensor<float> sumTileUb,
    LocalTensor<uint32_t> existFlagUb)
{
    uint32_t cnt = 0;
    GM_ADDR shareBase = reinterpret_cast<__gm__ uint8_t *>(serverShareAddr_[localRankId_]);
    ReadRankTokenCnt(fromLocalRank, cnt, shareBase);
    if (cnt == 0U) {
        return;
    }
    GM_ADDR shareDataBase =
        shareBase + static_cast<uint64_t>(shareFlagTotalBytes_ + (fromLocalRank * shareDataSliceBytes_));
    LocalTensor<float> tmpUb = localOutTempBuf_.Get<float>();
    LocalTensor<ExpandXType> inputBatchUb = tempBuf_.Get<ExpandXType>();
    uint32_t processed = 0;
    while (processed < cnt) {
        uint32_t curBatch = (cnt - processed) > BATCH_SIZE ? BATCH_SIZE : (cnt - processed);
        GlobalTensor<ExpandXType> batchGm;
        batchGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ ExpandXType *>(shareDataBase + static_cast<uint64_t>(processed * tokenMetaBytes_)));
        uint32_t copyLen = (curBatch * tokenMetaBytes_) / sizeof(ExpandXType); // tokenMetaBytes_是32B对齐的
        DataCopy(inputBatchUb, batchGm, copyLen);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ProcessBatchTokens(inputBatchUb, curBatch, targetServerId, baseId, endId, sumTileUb, existFlagUb, tmpUb);
        processed += curBatch;
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::ProcessBatchTokens(
    LocalTensor<ExpandXType> inputBatchUb, uint32_t curBatch, uint32_t targetServerId, uint32_t baseId, uint32_t endId,
    LocalTensor<float> sumTileUb, LocalTensor<uint32_t> existFlagUb, LocalTensor<float> tmpUb)
{
    LocalTensor<uint8_t> batchBytes = inputBatchUb.template ReinterpretCast<uint8_t>();
    for (uint32_t k = 0; k < curBatch; ++k) {
        const uint32_t packBaseBytes = k * tokenMetaBytes_;
        LocalTensor<float> metaF = batchBytes[packBaseBytes + tokenDataBytesAlign_].template ReinterpretCast<float>();
        float scaleVal = metaF.GetValue(0);
        LocalTensor<uint32_t> metaU32 =
            batchBytes[packBaseBytes + tokenDataBytesAlign_ + sizeof(float)].template ReinterpretCast<uint32_t>();
        uint32_t originRankId = metaU32.GetValue(0);
        uint32_t tokenIdInServer = metaU32.GetValue(1);
        AscendC::PipeBarrier<PIPE_ALL>();
        if ((originRankId / serverRankSize_) != targetServerId || tokenIdInServer < baseId ||
            tokenIdInServer >= endId) {
            AscendC::PipeBarrier<PIPE_ALL>();
            continue;
        }
        uint32_t offsetInTile = tokenIdInServer - baseId;
        const uint32_t sumStrideF = hFloatAlign32Size_ / sizeof(float);
        AscendC::PipeBarrier<PIPE_ALL>();
        existFlagUb.SetValue(offsetInTile, 1U);
        LocalTensor<ExpandXType> srcToken = batchBytes[packBaseBytes].template ReinterpretCast<ExpandXType>();
        LocalTensor<float> dstSum = sumTileUb[offsetInTile * sumStrideF];
        Cast(tmpUb, srcToken, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Muls(tmpUb, tmpUb, scaleVal, axisH_);
        PipeBarrier<PIPE_V>();
        Add(dstSum, dstSum, tmpUb, axisH_);
        PipeBarrier<PIPE_V>();
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::ReadRankTokenCnt(uint32_t fromLocalRank,
                                                                                              uint32_t &tokenCnt,
                                                                                              GM_ADDR shareBase)
{
    GM_ADDR flagAddr = shareBase + static_cast<uint64_t>(fromLocalRank * shareFlagSliceBytes_);
    GlobalTensor<uint64_t> flagGm;
    flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
    LocalTensor<uint64_t> flagUb = shareFlagUbBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
    DataCopy(flagUb, flagGm, flagU64CopyCntAlign_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    tokenCnt = static_cast<uint32_t>(flagUb.GetValue(1));
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::TokenToWinOut(
    GM_ADDR dstTokenBase, uint32_t tokenIdInServer, LocalTensor<float> srcSumTensor)
{
    LocalTensor<uint32_t> headerU32 = tempBuf_.Get<uint32_t>();
    constexpr uint32_t HEADER_U32_CNT = SPLIT_BLOCK_SIZE / sizeof(uint32_t);
    headerU32.SetValue(0, tokenIdInServer);
    // 1、写tokenID头，占512B
    GlobalTensor<uint32_t> headerGm;
    headerGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dstTokenBase));
    DataCopy(headerGm, headerU32, HEADER_U32_CNT);
    // 2、写Token
    // 2.1 写tokenData：480B
    LocalTensor<ExpandXType> tokenUb = localOutTempBuf_.Get<ExpandXType>();
    uint32_t totalElements = hFloatAlign32Size_ / sizeof(ExpandXType);
    Duplicate(tokenUb, static_cast<ExpandXType>(0), totalElements);
    PipeBarrier<PIPE_V>();
    Cast(tokenUb, srcSumTensor, AscendC::RoundMode::CAST_RINT, axisH_);
    GM_ADDR dataBlockBase = dstTokenBase + SPLIT_BLOCK_SIZE;
    GlobalTensor<ExpandXType> dataDstGm;
    dataDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXType *>(dataBlockBase));
    DataCopyExtParams dataCopyParams = {static_cast<uint16_t>(tokenDataBlockNum_), SPLIT_BLOCK_DATA_SIZE, 0U, UB_ALIGN,
                                        0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(dataDstGm, tokenUb, dataCopyParams);
    // 2.2 padding写flag：32B
    uint64_t mask[1] = {0x0101010101010101};
    uint8_t repeatTime = static_cast<uint8_t>(Ceil((tokenDataBlockNum_ + 1) * UB_ALIGN, 256));
    LocalTensor<uint32_t> flagU32 = outBuf1_.Get<uint32_t>();
    Duplicate<uint32_t>(flagU32, uint32_t(1), mask, repeatTime, uint16_t(1), uint8_t(8));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    GlobalTensor<uint32_t> flagDstGm;
    flagDstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(dstTokenBase));
    constexpr uint32_t PAD_OFFSET_IN_U32 = SPLIT_BLOCK_DATA_SIZE / sizeof(uint32_t);
    DataCopyExtParams flagCopyParams = {static_cast<uint16_t>(tokenDataBlockNum_ + 1), UB_ALIGN, 0U,
                                        SPLIT_BLOCK_DATA_SIZE, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(flagDstGm[PAD_OFFSET_IN_U32], flagU32, flagCopyParams);
    PipeBarrier<PIPE_ALL>();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::WriteWinOutHeader(
    GlobalTensor<uint64_t> headerGm, uint32_t winTokenCnt)
{
    LocalTensor<uint64_t> headerUb = shareFlagUbBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
    Duplicate<uint64_t>(headerUb, 0ULL, flagU64CopyCntAlign_);
    headerUb.SetValue(0, READY_FLAG);
    headerUb.SetValue(1, winTokenCnt);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(headerGm, headerUb, flagU64CopyCntAlign_);
}

// 传递token
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AlltoAllServerDispatch()
{
    SyncAll<true>();
    // 0核处理 hccl_.BatchWrite<true>()
    if (coreIdx_ == (startServerId_ / sendServerNum_)) {
        GM_ADDR srcAddr = GetWinAddrByRankId(localRankId_);
        GM_ADDR dstAddr = srcAddr + static_cast<uint64_t>(winOutTotalBytes_);
        flagLocalInWindow_.SetGlobalBuffer((__gm__ uint64_t *)(dstAddr));
        flagLocalOutWindow_.SetGlobalBuffer((__gm__ uint64_t *)(srcAddr));
        LocalTensor<uint64_t> flagInUb = moeQueue_.AllocTensor<uint64_t>();
        DataCopy(flagInUb, flagLocalOutWindow_, flagU64CopyCntAlign_);
        moeQueue_.EnQue(flagInUb);
        LocalTensor<uint64_t> flagOutUb = moeQueue_.DeQue<uint64_t>();
        DataCopy(flagLocalInWindow_, flagOutUb, flagU64CopyCntAlign_);
        moeQueue_.FreeTensor<uint64_t>(flagOutUb);
        uint32_t tokenAlignCnt = packedTokenBytes_ / sizeof(ExpandXType);
        tokenLocalOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(srcAddr + WIN_ADDR_ALIGN));
        tokenLocalInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(dstAddr + WIN_ADDR_ALIGN));
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        for (uint32_t tokenId = 0U; tokenId < maxLocalBs_; tokenId++) {
            LocalTensor<ExpandXType> inUb = moeQueue_.AllocTensor<ExpandXType>();
            DataCopy(inUb, tokenLocalOutWindow_[tokenId * tokenAlignCnt], tokenAlignCnt);
            moeQueue_.EnQue(inUb);
            LocalTensor<ExpandXType> outUb = moeQueue_.DeQue<ExpandXType>();
            DataCopy(tokenLocalInWindow_[tokenId * tokenAlignCnt], outUb, tokenAlignCnt);
            moeQueue_.FreeTensor<ExpandXType>(outUb);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

// 读取count的flag位，确保所有server组都执行结束
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::WaitWinInCount()
{
    if (coreIdx_ < serverNum_) {
        tpipe_->InitBuffer(localCntBuf_, UB_ALIGN);
        localCntTensor_ = localCntBuf_.Get<uint64_t>();
        GlobalTensor<uint64_t> flagGm;
        flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(windowInGM_));
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        // 对winIn中的count的flag位进行等待求和判断
        for (uint32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
            while (true) {
                uint32_t flagOffset = serverIndex * winInSliceBytes_;
                DataCopy(localCntTensor_, flagGm[flagOffset], flagU64CopyCntAlign_);
                PipeBarrier<PIPE_ALL>();
                uint64_t flagValue = localCntTensor_.GetValue(0);
                if (flagValue == READY_FLAG) {
                    break;
                }
            }
        }
    }
    SyncAll<true>();
}

// 等待token拆分后的block块到齐
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::WaitTokenBlockReady(GM_ADDR winInTkAddr,
                                                                                                 uint32_t countIdx)
{
    GM_ADDR tokenBase = winInTkAddr + (countIdx * packedTokenBytes_);
    GM_ADDR tokenDataBase = tokenBase + SPLIT_BLOCK_SIZE;
    GlobalTensor<uint32_t> gmU32;
    gmU32.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tokenDataBase));
    while (true) {
        uint32_t readyCount = 0;
        for (uint32_t blockIndex = 0; blockIndex < tokenDataBlockNum_; ++blockIndex) {
            LocalTensor<uint32_t> localFlagTensor = localFlagBuf_.Get<uint32_t>();
            uint32_t offsetOfFlag =
                ((blockIndex + 1) * SPLIT_BLOCK_DATA_SIZE + blockIndex * SPLIT_BLOCK_FLAG_SIZE) / sizeof(uint32_t);
            DataCopy(localFlagTensor, gmU32[offsetOfFlag], flagU64CopyCntAlign_ * 2);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t cnt = localFlagTensor.GetValue(0);
            if (cnt != 0) {
                readyCount++;
            }
        }
        if (readyCount == tokenDataBlockNum_) {
            break;
        }
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::LoadTokenToUb(
    LocalTensor<ExpandXType> &outTokenUb, GM_ADDR winInTkAddr, uint32_t countIdx)
{
    GM_ADDR tokenBase = winInTkAddr + (countIdx * packedTokenBytes_);
    GM_ADDR tokenDataBase = tokenBase + SPLIT_BLOCK_SIZE;
    GlobalTensor<ExpandXType> blockGm;
    blockGm.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXType *>(tokenDataBase));
    DataCopyExtParams dataCopyParams = {static_cast<uint16_t>(tokenDataBlockNum_), SPLIT_BLOCK_DATA_SIZE, UB_ALIGN, 0U,
                                        0U};
    DataCopyPadExtParams<ExpandXType> padParams{true, 0, 0, 0};
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyPad(outTokenUb, blockGm, dataCopyParams, padParams);
}

// 对来自不同server的相同token id进行求和
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AlltoAllCombine()
{
    SplitCoreByToken(axisBS_);
    LocalTensor<uint32_t> tokenCntArray = countBuf_.Get<uint32_t>();
    LoadTokenCounts(tokenCntArray);
    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; tokenId++) {
        ProcessSingleToken(tokenId, tokenCntArray);
    }
}

// 统计winIn每个server分区token个数
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::LoadTokenCounts(
    LocalTensor<uint32_t> &tokenCntArray)
{
    GlobalTensor<uint64_t> flagGmU64;
    flagGmU64.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(windowInGM_));
    for (uint32_t serverIdx = 0; serverIdx < serverNum_; serverIdx++) {
        localCntTensor_ = localCntBuf_.Get<uint64_t>();
        uint32_t offsetOfNum = (serverIdx * winInSliceBytes_) / sizeof(uint64_t);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopy(localCntTensor_, flagGmU64[offsetOfNum], flagU64CopyCntAlign_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t count = static_cast<uint32_t>(localCntTensor_.GetValue(1));
        tokenCntArray.SetValue(serverIdx, count);
    }
    PipeBarrier<PIPE_ALL>();
}

// 按照token进行分核并开始累加求和
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::ProcessSingleToken(
    uint32_t tokenId, LocalTensor<uint32_t> &tokenCntArray)
{
    const uint32_t targetTokenId = tokenIdBaseInServer_ + tokenId;
    LocalTensor<float> sumLocal = sumBuf_.Get<float>();
    AscendC::Duplicate(sumLocal, 0.0f, hFloatAlign32Size_ / sizeof(float));
    AscendC::PipeBarrier<PIPE_V>();
    bool foundAny = false;
    SearchAndAccumulateToken(targetTokenId, tokenCntArray, sumLocal, foundAny);
    if (!foundAny) {
        return;
    }
    ProcessSharedExpertAndOutput(tokenId, sumLocal);
}

// 按照server分区查找对应token
template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::SearchAndAccumulateToken(
    uint32_t targetTokenId, LocalTensor<uint32_t> &tokenCntArray, LocalTensor<float> &sumLocal, bool &foundAny)
{
    for (uint32_t serverIdx = 0; serverIdx < serverNum_; serverIdx++) {
        uint32_t tokenCnt = tokenCntArray.GetValue(serverIdx);
        if (tokenCnt == 0) {
            continue;
        }
        GM_ADDR winInTkAddr = windowInGM_ + winHeaderBytes_ + serverIdx * winInSliceBytes_;
        uint32_t hitIdx = 0;
        bool hit = false;
        GlobalTensor<uint32_t> idGm;
        idGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(winInTkAddr));
        LocalTensor<uint32_t> localTokenIdTensor = localTokenIdBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < tokenCnt; i++) {
            uint32_t offsetOfToken = (i * packedTokenBytes_) / sizeof(uint32_t);
            DataCopy(localTokenIdTensor, idGm[offsetOfToken], flagU64CopyCntAlign_ * 2);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t currTokenId = localTokenIdTensor.GetValue(0);
            if (currTokenId == targetTokenId) {
                hitIdx = i;
                hit = true;
                break;
            }
        }
        if (!hit) {
            continue;
        }
        foundAny = true;
        AccumulateTokenFromServer(winInTkAddr, hitIdx, sumLocal);
    }
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::AccumulateTokenFromServer(
    GM_ADDR winInTkAddr, uint32_t hitIdx, LocalTensor<float> &sumLocal)
{
    WaitTokenBlockReady(winInTkAddr, hitIdx);
    LocalTensor<ExpandXType> tokenUb = localOutTensorBuf_.Get<ExpandXType>();
    LoadTokenToUb(tokenUb, winInTkAddr, hitIdx);
    AscendC::PipeBarrier<PIPE_ALL>();
    LocalTensor<float> tokenFp32Ub = localOutTempBuf_.Get<float>();
    AscendC::Cast(tokenFp32Ub, tokenUb, AscendC::RoundMode::CAST_NONE, axisH_);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(sumLocal, sumLocal, tokenFp32Ub, axisH_);
    AscendC::PipeBarrier<PIPE_V>();
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::ProcessSharedExpertAndOutput(
    uint32_t tokenId, LocalTensor<float> &sumLocal)
{
    if (hasSharedExpertX_) {
        const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
        const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
        LocalTensor<XType> tokenUb = localOutTensorBuf_.Get<XType>();
        SyncFunc<AscendC::HardEvent::V_MTE2>(); // 与结果搬出Cast同地址
        DataCopyPad(tokenUb, sharedExpertXGM_[tokenId * axisH_], expandXCopyParams, copyPadExtParams);
        LocalTensor<float> tokenFp32Ub = localOutTempBuf_.Get<float>();
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(tokenFp32Ub, tokenUb, AscendC::RoundMode::CAST_NONE, axisH_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sumLocal, sumLocal, tokenFp32Ub, axisH_);
        AscendC::PipeBarrier<PIPE_V>();
    }
    LocalTensor<ExpandXType> outUb = localOutTensorBuf_.Get<ExpandXType>();
    AscendC::Cast(outUb, sumLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    GlobalTensor<ExpandXType> outGm = expandOutGlobal_[tokenId * axisH_];
    DataCopy(outGm, outUb, tokenDataBytesAlign_ / sizeof(ExpandXType));
}

template <CombineV2HostTypeClass>
__aicore__ inline void MoeDistributeCombineV2HostKfc<CombineV2HostTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        BuffInit();
        // 1、Server内
        AlltoAllDispatch();
        AlltoAllBuffInitAndMaskCal();
        SumToWindow();
        // 2、Server间
        AlltoAllServerDispatch();
        WaitWinInCount();
        AlltoAllCombine();
    }
}

} // namespace MoeDistributeCombineV2Impl
#endif // MOE_DISTRIBUTE_COMBINE_V2_HOST_KFC_H
