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
 * \file flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "flash_attn_metadata.h"
#include "flash_attn_metadata_aicpu.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../flash_attn/op_host/fa_adjust_sinner_souter.h"

#define FA_KERNEL_STATUS_OK             0
#define FA_KERNEL_STATUS_PARAM_INVALID  1

using namespace optiling;

namespace aicpu {
uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes {};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    return FA_KERNEL_STATUS_OK;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                         GetAttrValue(ctx, "head_dim", headDim_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    return ParamsInit();
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

void FlashAttnMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void FlashAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    uint32_t qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == load_balance::Layout::BSH || baseInfo.layoutQuery == load_balance::Layout::BSND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == load_balance::Layout::TND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_TND;
    }
    optiling::flash_attn::fa_tiling_util::AdjustSinnerAndSouter(baseInfo.headDim, maxSeqlenQ_,
        maxSeqlenKv_, baseInfo.sparseMode, baseInfo.preToken, baseInfo.nextToken, qlayout,
        mBaseSize_, s2BaseSize_);
    mBaseSize_ *= (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96U * 1024U * 1024U;      // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb
    param.fdTolerance = 10;                  // 10: tolerance block
    param.fdLeastBlock = 3;                  // 3: least block
    param.fdOn = true;                  // TODO: turn off fd for flash attn
}

void FlashAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = batchSize_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.preToken = (winLeft_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winLeft_);
    baseInfo.nextToken = (winRight_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winRight_);
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    LoadActualQuerySeq();
    LoadActualKvSeq();
}

void FlashAttnMetadataCpuKernel::LoadActualQuerySeq()
{
    baseInfo.actualQuerySeqSize.clear();
    baseInfo.isCumulativeQuerySeq = (layoutQ_ == "TND" || layoutQ_ == "NTD");

    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(sequsedQ_, sequsedQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.querySeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeQuerySeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualQuerySeqSize.begin());
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1U;
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensQ_, cuSeqlensQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.querySeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeQuerySeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.querySeqSize = std::max(baseInfo.querySeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

void FlashAttnMetadataCpuKernel::LoadActualKvSeq()
{
    baseInfo.actualKvSeqSize.clear();
    baseInfo.isCumulativeKvSeq = (layoutKv_ == "TND" || layoutKv_ == "NTD");

    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(sequsedKv_, sequsedKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.kvSeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeKvSeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualKvSeqSize.begin());
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensKv_, cuSeqlensKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.kvSeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeKvSeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.kvSeqSize = std::max(baseInfo.kvSeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult &splitRes)
{
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum);
    faMetadata.Clear();         // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
        [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead { static_cast<uint32_t>(aicCoreNum_) }; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto &faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto &prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bN2End[prevLastCore] : faRes.bN2End[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.gS1End[prevLastCore] : faRes.gS1End[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX,   mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX,  s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX,   faRes.bN2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX,     faRes.gS1End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX,    faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto &fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX,       fdRes.bN2Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX,         fdRes.gS1Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX,       fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX,         fdRes.mLen[aivIdx]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
