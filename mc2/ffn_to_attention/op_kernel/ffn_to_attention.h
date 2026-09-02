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
 * \file ffn_to_attention.h
 * \brief
 */

#ifndef FFN_TO_ATTENTION_H
#define FFN_TO_ATTENTION_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "ffn_to_attention_tiling.h"
#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace FFNToAttentionImpl {
constexpr uint8_t BUFFER_NUM = 2; // 多buf
constexpr uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr uint8_t WIN_OFFSET_CNT = 2;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;

#define TemplateFFNToAttentionTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionTypeFunc xType, isInputRankTable

using namespace AscendC;
template <TemplateFFNToAttentionTypeClass>
class FFNToAttention {
public:
    __aicore__ inline FFNToAttention(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds, GM_ADDR tokenIds,
                                GM_ADDR expertOffsets, GM_ADDR actualTokenNum, GM_ADDR attnRankTable,
                                GM_ADDR workspaceGM, TPipe *pipe, const FFNToAttentionTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
    __aicore__ inline void ReadTokenMetaData(ReadTokenMetaDataStruct &metaDataStruct, uint32_t YOffset);
    TPipe *tpipe_{nullptr};
    GlobalTensor<xType> xGMTensor_;
    GlobalTensor<int32_t> layerIdsGMTensor_;
    GlobalTensor<int32_t> sessionIdsGMTensor_;
    GlobalTensor<int32_t> microBatchIdsGMTensor_;
    GlobalTensor<int32_t> tokenIdsGMTensor_;
    GlobalTensor<int32_t> expertOffsetsGMTensor_;
    GlobalTensor<int64_t> actualTokenNumGMTensor_;
    GlobalTensor<int32_t> attnRankTableGMTensor_;
    LocalTensor<xType> xTmpTensor_;
    LocalTensor<int32_t> statusTensor_;

    // TBuf<> attnTableBuf_;
    TBuf<> statusBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;

    uint32_t aivId_{0};
    uint64_t actualTokenNum_{0};
    uint32_t axisH_{0};
    uint32_t axisHS_{0};
    uint32_t axisA_{0};
    uint32_t microBatchNum_{0};
    uint32_t axisBS_{0};
    uint32_t expertNumPerToken_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t batchSizeSendCnt_{0};
    uint64_t perMicroBatchOffset_{0};
    uint64_t perExpertOffset_{0};
    uint64_t hSize_{0};
    uint64_t winTokenInfoTableSize_{0};
    uint8_t windowType_{0};
    __gm__ HcclOpResParam *winContext_{nullptr};
};

template <TemplateFFNToAttentionTypeClass>
__aicore__ inline void FFNToAttention<TemplateFFNToAttentionTypeFunc>::Init(
    GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds, GM_ADDR tokenIds, GM_ADDR expertOffsets,
    GM_ADDR actualTokenNum, GM_ADDR attnRankTable, GM_ADDR workspaceGM, TPipe *pipe,
    const FFNToAttentionTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    axisBS_ = tilingData->ffnToAttentionInfo.BS;
    axisH_ = tilingData->ffnToAttentionInfo.H;
    axisHS_ = tilingData->ffnToAttentionInfo.HS;
    axisA_ = tilingData->ffnToAttentionInfo.A;
    microBatchNum_ = tilingData->ffnToAttentionInfo.microBatchNum;
    expertNumPerToken_ = tilingData->ffnToAttentionInfo.expertNumPerToken; // 专家总数
    aivNum_ = tilingData->ffnToAttentionInfo.aivNum;
    worldSize_ = tilingData->ffnToAttentionInfo.worldSize;
    windowType_ = tilingData->ffnToAttentionInfo.windowType;

    xGMTensor_.SetGlobalBuffer((__gm__ xType *)x);
    sessionIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sessionIds);
    microBatchIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)microBatchIds);
    tokenIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)tokenIds);
    expertOffsetsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertOffsets);
    actualTokenNumGMTensor_.SetGlobalBuffer((__gm__ int64_t *)actualTokenNum);
    if constexpr (isInputRankTable) {
        attnRankTableGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(attnRankTable));
    }

    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(actualTokenNumGMTensor_);
    actualTokenNum_ = actualTokenNumGMTensor_.GetValue(0);
    hSize_ = axisH_ * sizeof(xType);
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_); // H * 2
    tpipe_->InitBuffer(statusBuf_, UB_ALIGN);        // 对齐32B，flag位
    statusTensor_ = statusBuf_.Get<int32_t>();       // 标志位刷新
    statusTensor_(0) = 1;

    batchSizeSendCnt_ = axisBS_ * expertNumPerToken_;
    perMicroBatchOffset_ = batchSizeSendCnt_ * axisHS_;
    perExpertOffset_ = expertNumPerToken_ * axisHS_;
    winTokenInfoTableSize_ = Ceil(microBatchNum_ * batchSizeSendCnt_ * sizeof(int32_t), WIN_ADDR_ALIGN) *
                             WIN_ADDR_ALIGN; // win区划分token_info_table在前, 偏移向上取整
}

template <TemplateFFNToAttentionTypeClass>
__aicore__ inline GM_ADDR FFNToAttention<TemplateFFNToAttentionTypeFunc>::GetWindowAddr(uint32_t curAttenWorkRank)
{
    // 当 userMemType 为 0 时，使用远程设备内存; 当 userMemType 不为 0 时，使用用户自定义内存
    if (winContext_->userMemType == 0) {
        return (__gm__ uint8_t *)(((HcclRankRelationResV2 *)(winContext_->remoteRes[curAttenWorkRank].nextDevicePtr))
                                      ->windowsIn);
    } else {
        return (__gm__ uint8_t *)(winContext_->userMemRes[curAttenWorkRank].addr);
    }
}

template <TemplateFFNToAttentionTypeClass>
__aicore__ inline void FFNToAttention<TemplateFFNToAttentionTypeFunc>::ReadTokenMetaData(
    ReadTokenMetaDataStruct &metaDataStruct, uint32_t YOffset)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        sessionIdsGMTensor_[YOffset]);
    metaDataStruct.curAttenWorkIds = sessionIdsGMTensor_(YOffset); // 直接去gm取数值 当前token属于哪个attnWork
    metaDataStruct.curAttenWorkRank = metaDataStruct.curAttenWorkIds;

    if constexpr (isInputRankTable) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            attnRankTableGMTensor_[metaDataStruct.curAttenWorkIds]);
        metaDataStruct.curAttenWorkRank =
            attnRankTableGMTensor_(metaDataStruct.curAttenWorkIds); // 直接去gm取数值 当前token发送的目标卡
    }

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        microBatchIdsGMTensor_[YOffset]);
    metaDataStruct.curMicroBatchIds = microBatchIdsGMTensor_(YOffset); // 直接去gm取数值 当前token属于哪个microBatch

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tokenIdsGMTensor_[YOffset]);
    metaDataStruct.curTokenBatchOffset = tokenIdsGMTensor_(YOffset); // 直接去gm取数值 当前Token属于Batch中的第几个

    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        expertOffsetsGMTensor_[YOffset]);
    metaDataStruct.curTokenTopkOffset = expertOffsetsGMTensor_(YOffset); // 直接去gm取数值 当前token属于TopK中的第几个
}

template <TemplateFFNToAttentionTypeClass>
__aicore__ inline void FFNToAttention<TemplateFFNToAttentionTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        uint32_t calStartTokenId = 0;
        uint32_t sendTokenNum = actualTokenNum_ / aivNum_;
        uint32_t remainderRankNum = actualTokenNum_ % aivNum_;
        if (aivId_ < remainderRankNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
            sendTokenNum += 1;
            calStartTokenId += aivId_;
        } else {
            calStartTokenId += remainderRankNum;
        }
        if (calStartTokenId >= actualTokenNum_) {
            sendTokenNum = 0;
        }
        GM_ADDR curRankWinAddr;
        GlobalTensor<xType> tokenDataGMTensor;
        GlobalTensor<int32_t> tokenInfoTableGMTensor;
        DataCopyExtParams dataCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(axisH_ * sizeof(xType)), 0U, 0U, 0U};
        DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
        for (uint32_t tokenCnt = 0; tokenCnt < sendTokenNum; ++tokenCnt) {
            uint32_t YOffset = aivId_ + (tokenCnt * aivNum_); // 当前token在Y的偏移

            // 1. 加载token
            xTmpTensor_ = xQueue_.AllocTensor<xType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[YOffset * axisH_], xCopyParams, copyPadExtParams);

            // 2. 计算偏移数据
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaData(metaDataStruct, YOffset);

            // 3. 计算win区地址
            curRankWinAddr = GetWindowAddr(metaDataStruct.curAttenWorkRank);

            // 4. 计算info和data在win区的地址偏移
            uint64_t tokenInfoTableOffset = metaDataStruct.curMicroBatchIds * batchSizeSendCnt_;
            uint64_t tokenDataOffset =
                (metaDataStruct.curMicroBatchIds * perMicroBatchOffset_ +
                 metaDataStruct.curTokenBatchOffset * perExpertOffset_ + metaDataStruct.curTokenTopkOffset * axisHS_) *
                sizeof(xType);
            tokenInfoTableGMTensor.SetGlobalBuffer(
                (__gm__ int32_t *)(curRankWinAddr +
                                   (tokenInfoTableOffset + metaDataStruct.curTokenBatchOffset * expertNumPerToken_ +
                                    metaDataStruct.curTokenTopkOffset) *
                                       sizeof(int32_t))); // win区分布，前面放info 后面放data
            tokenDataGMTensor.SetGlobalBuffer(
                (__gm__ xType *)(curRankWinAddr + winTokenInfoTableSize_ + tokenDataOffset));

            // 5. 复制数据到win区
            if (tokenCnt == 0) {
                SyncFunc<AscendC::HardEvent::S_MTE3>();
            }
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<xType>();
            DataCopyPad(tokenDataGMTensor, xTmpTensor_, xCopyParams); // 数据搬入token_data位置上
            xQueue_.FreeTensor<xType>(xTmpTensor_);
            PipeBarrier<PIPE_MTE3>();
            DataCopyPad(tokenInfoTableGMTensor, statusTensor_, dataCopyParams); // 状态位
        }
    }
}
} // namespace FFNToAttentionImpl
#endif
