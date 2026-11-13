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
 * \file flash_mla_metadata_aicpu.cpp
 * \brief
 */

#include "flash_mla_metadata_aicpu.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "log.h"
#include "status.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "flash_mla_metadata.h"
#include "../../flash_mla/op_kernel/utils/flash_mla_config.h"

constexpr uint32_t FA_KERNEL_STATUS_OK = 0;
constexpr uint32_t FA_KERNEL_STATUS_PARAM_INVALID = 1;

using namespace optiling;

namespace aicpu {
uint32_t FlashMlaMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = CheckMetadataCapacity(splitRes.sectionNum);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Metadata buffer capacity insufficient!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    return FA_KERNEL_STATUS_OK;
}

bool FlashMlaMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cacheSeqlens_ = ctx.Input(static_cast<uint32_t>(ParamId::cacheSeqlens));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim_qk", headDimQk_) && GetAttrValue(ctx, "head_dim_v", headDimV_) &&
        GetAttrValue(ctx, "soc_version", socVersion_) && GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
        GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);

    KERNEL_CHECK_FALSE(ParamsCheck(), false, "Params check failed");
    return ParamsInit();
}

bool FlashMlaMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

bool FlashMlaMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_FALSE(CheckAttrs(), false, "Check attrs failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    KERNEL_CHECK_FALSE(!actualSeqlenKv_.empty(), false, "cacheSeqlens must be provided to derive batch_size");
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(static_cast<int64_t>(actualSeqlenKv_.size())), false,
                       "Check query sequence failed");
    return true;
}

bool FlashMlaMetadataCpuKernel::CheckAttrs()
{
    KERNEL_CHECK_FALSE(numHeadsQ_ >= 1 && numHeadsQ_ <= 128, false, "numHeadsQ must be in [1, 128], but got %d",
                       numHeadsQ_);
    KERNEL_CHECK_FALSE(numHeadsKv_ == FlashMlaCommon::MLA_KV_HEADS, false, "MLA kv head num must be %ld, but got %d",
                       FlashMlaCommon::MLA_KV_HEADS, numHeadsKv_);
    KERNEL_CHECK_FALSE(headDimQk_ == FlashMlaCommon::MLA_HEAD_DIM_QK, false,
                       "MLA headDimQk (nope 512 + rope 64) must be %ld, but got %d", FlashMlaCommon::MLA_HEAD_DIM_QK,
                       headDimQk_);
    KERNEL_CHECK_FALSE(headDimV_ == FlashMlaCommon::MLA_HEAD_DIM_V, false,
                       "MLA headDimV (value/output width) must be %ld, but got %d", FlashMlaCommon::MLA_HEAD_DIM_V,
                       headDimV_);
    KERNEL_CHECK_FALSE((maskMode_ == 0 || maskMode_ == 3 || maskMode_ == 4), false,
                       "maskMode only supports 0(NO_MASK), 3(RIGHT_DOWN), 4(BAND), but got %d", maskMode_);
    KERNEL_CHECK_FALSE(winLeft_ >= -1 && winLeft_ <= INT32_MAX && winRight_ >= -1 && winRight_ <= INT32_MAX, false,
                       "win_left and win_right must be in [-1, INT32_MAX]");
    KERNEL_CHECK_FALSE(maskMode_ == 4 || (winLeft_ == -1 && winRight_ == -1), false,
                       "win_left and win_right must be -1 for mask_mode 0 or 3");
    KERNEL_CHECK_FALSE((layoutQ_ == "TND" || layoutQ_ == "BNSD" || layoutQ_ == "BSND"), false,
                       "layoutQ only supports TND, BNSD, BSND, but got %s", layoutQ_.c_str());
    KERNEL_CHECK_FALSE(aicCoreNum_ > 0, false, "aicCoreNum must be positive, but got %d", aicCoreNum_);
    KERNEL_CHECK_FALSE(aivCoreNum_ > 0, false, "aivCoreNum must be positive, but got %d", aivCoreNum_);
    if (socVersion_.find("Ascend960") != std::string::npos) {
        KERNEL_CHECK_FALSE(aicCoreNum_ <= 64 && aivCoreNum_ == aicCoreNum_, false,
                           "arch92 MLA requires at most 64 AICs and CV 1:1, but got AIC=%d AIV=%d", aicCoreNum_,
                           aivCoreNum_);
    }
    return true;
}

bool FlashMlaMetadataCpuKernel::CheckMetadataCapacity(uint32_t sectionNum)
{
    const uint64_t maxSections = static_cast<uint64_t>(actualSeqlenKv_.size()) * numHeadsKv_;
    KERNEL_CHECK_FALSE(sectionNum > 0 && sectionNum <= maxSections, false, "sectionNum must be in [1, %lu], but got %u",
                       maxSections, sectionNum);
    const int64_t needWords =
        FlashMlaCommon::HEAD_METADATA_STRIDE +
        static_cast<int64_t>(sectionNum) * (static_cast<int64_t>(aicCoreNum_) * FlashMlaCommon::FA_METADATA_STRIDE +
                                            static_cast<int64_t>(aivCoreNum_) * FlashMlaCommon::FD_METADATA_STRIDE);
    KERNEL_CHECK_FALSE(metadata_->NumElements() >= needWords, false,
                       "metadata buffer too small: %ld words, but sectionNum %u needs %ld words",
                       metadata_->NumElements(), sectionNum, needWords);
    return true;
}

bool FlashMlaMetadataCpuKernel::CheckActualQuerySeq(int64_t batchSize)
{
    isActualSeqlenQAccum_ = false;
    actualSeqlenQ_.clear();
    std::vector<int64_t> cuSeqlensQ{};
    std::vector<int64_t> sequsedQ{};

    cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    sequsedQ = GetTensorDataAsInt64(sequsedQ_);

    if (!cuSeqlensQ.empty() && static_cast<int64_t>(cuSeqlensQ.size()) != batchSize + 1) {
        KERNEL_LOG_ERROR("cuSeqlensQ shape must be (batchSize+1,)=(%ld,), but got %zu", batchSize + 1,
                         cuSeqlensQ.size());
        return false;
    }
    if (!sequsedQ.empty() && static_cast<int64_t>(sequsedQ.size()) != batchSize) {
        KERNEL_LOG_ERROR("sequsedQ shape must be (batchSize,)=(%ld,), but got %zu", batchSize, sequsedQ.size());
        return false;
    }

    if (layoutQ_ == "TND") {
        if (cuSeqlensQ.empty()) {
            KERNEL_LOG_ERROR("When layoutQ is TND, cuSeqlensQ should be provided, but got empty");
            return false;
        }
    } else if (!cuSeqlensQ.empty()) {
        KERNEL_LOG_ERROR("When layoutQ is not TND, cuSeqlensQ should not be provided, but got non-empty");
        return false;
    }

    for (size_t i = 0; i < sequsedQ.size(); ++i) {
        if (sequsedQ[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedQ must be non-negative, but %zuth element is %ld", i, sequsedQ[i]);
            return false;
        }
    }

    if (!cuSeqlensQ.empty()) {
        if (cuSeqlensQ[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensQ must be 0, but got %ld", cuSeqlensQ[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensQ.size(); ++i) {
        if (cuSeqlensQ[i] < cuSeqlensQ[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensQ must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensQ[i], cuSeqlensQ[i - 1]);
            return false;
        }
        if (!sequsedQ.empty() && sequsedQ[i - 1] > cuSeqlensQ[i] - cuSeqlensQ[i - 1]) {
            KERNEL_LOG_ERROR("sequsedQ[%zu]=%ld exceeds the physical query length %ld", i - 1, sequsedQ[i - 1],
                             cuSeqlensQ[i] - cuSeqlensQ[i - 1]);
            return false;
        }
    }

    if (!sequsedQ.empty()) {
        isActualSeqlenQAccum_ = false;
        actualSeqlenQ_ = sequsedQ;
    } else if (!cuSeqlensQ.empty()) {
        isActualSeqlenQAccum_ = true;
        actualSeqlenQ_.assign(cuSeqlensQ.begin() + 1, cuSeqlensQ.end());
    }

    return true;
}

bool FlashMlaMetadataCpuKernel::CheckActualKvSeq()
{
    isActualSeqlenKvAccum_ = false;
    actualSeqlenKv_.clear();
    std::vector<int64_t> cacheSeqlens{};

    cacheSeqlens = GetTensorDataAsInt64(cacheSeqlens_);

    for (size_t i = 0; i < cacheSeqlens.size(); ++i) {
        if (cacheSeqlens[i] < 0) {
            KERNEL_LOG_ERROR("The elements of cacheSeqlens must be non-negative, but %zuth element is %ld", i,
                             cacheSeqlens[i]);
            return false;
        }
    }

    actualSeqlenKv_ = cacheSeqlens;
    return true;
}

void FlashMlaMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void FlashMlaMetadataCpuKernel::InitLoadBalanceParams()
{
    const auto tile =
        socVersion_.find("Ascend960") != std::string::npos ? FlashMlaConfig::ARCH92 : FlashMlaConfig::ARCH35;
    mBaseSize_ = tile.m;
    s2BaseSize_ = tile.s2;
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    // M 方向总行数 = g * S1 = (N1/N2) * maxSeqlenQ。若总行数 <= 2*mBaseSize（M 最多 2 块），
    // 多 section 切分无收益，仅增加 metadata/re-init/SyncAll 开销，故关闭。
    {
        uint32_t g = numHeadsQ_ / numHeadsKv_;
        if (g * maxSeqlenQ_ <= param.mBaseSize * 2) {
            param.l2Byte = 0U; // 0: disable section splitting
        } else {
            // Reserve one quarter of Ascend350's 64 MiB L2, matching the existing 950 budget ratio.
            const uint32_t l2BudgetMiB = socVersion_.find("Ascend350") != std::string::npos ? 48U : 96U;
            param.l2Byte = l2BudgetMiB * 1024U * 1024U;
        }
    }
    param.fdTolerance = 10; // 10: tolerance block
    param.fdLeastBlock = 3; // 3: least block
    param.fdOn = true;
}

void FlashMlaMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = actualSeqlenKv_.size();
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDimQk = static_cast<uint32_t>(headDimQk_);
    baseInfo.headDimV = static_cast<uint32_t>(headDimV_);
    load_balance::SparseMode maskMode = load_balance::SparseMode::BUTT;
    if (maskMode_ != 0) {
        maskMode = static_cast<load_balance::SparseMode>(maskMode_);
    }
    baseInfo.attenMaskFlag = (maskMode != load_balance::SparseMode::BUTT);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode);
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winLeft_);
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winRight_);
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = isActualSeqlenKvAccum_;
    baseInfo.actualKvSeqSize = actualSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = isActualSeqlenQAccum_;
    baseInfo.actualQuerySeqSize = actualSeqlenQ_;
}

bool FlashMlaMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult& splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashMlaMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult& splitRes)
{
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum, static_cast<uint32_t>(aicCoreNum_),
                                  static_cast<uint32_t>(aivCoreNum_));
    faMetadata.Clear(); // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    // Emit the same architecture tile used to partition tasks.
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, static_cast<FA_METADATA_T>(aicCoreNum_));
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, static_cast<FA_METADATA_T>(aivCoreNum_));
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX,
                               static_cast<FA_METADATA_T>(load_balance::KernelSplitMode::BN2_S1G_S2));
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
    return true;
}

namespace {
static const char* kernelType = "FlashMlaMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashMlaMetadataCpuKernel);
} // namespace

} // namespace aicpu
