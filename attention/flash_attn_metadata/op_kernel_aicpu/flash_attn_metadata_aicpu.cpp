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

#include "flash_attn_metadata_aicpu.h"
#include <cmath>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include "log.h"
#include "status.h"
#include "flash_attn_grad_metadata_split.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../flash_attn/op_host/fa_adjust_sinner_souter.h"
#include "../../flash_attn/op_host/arch92/fa_adjust_sinner_souter_arch92.h"

constexpr uint32_t FA_KERNEL_STATUS_OK = 0;
constexpr uint32_t FA_KERNEL_STATUS_PARAM_INVALID = 1;

using namespace optiling;

namespace aicpu {
namespace {
// FAG uses cu_seqlens exclusively for physical packed offsets, and seqused
// (when present) for the effective attention rectangle within each batch.
bool CheckFagSequence(const std::vector<int64_t>& cu, const std::vector<int64_t>& used, int64_t batchSize,
                      int64_t maxSeqlen, bool isTnd, const char* name)
{
    if (!isTnd) {
        if (!cu.empty()) {
            KERNEL_LOG_ERROR("FAG %s cu_seqlens is supported only for TND layout", name);
            return false;
        }
        if (used.empty()) {
            return true;
        }
        // Optional per the interface doc (seqused or max_seqlen, one of the
        // two): when this side provides seqused, max_seqlen may be omitted.
        // Skip the upper-bound check in that case; the missing-side
        // requirement (max as the default fill source) is enforced in
        // ParamsCheck.
        if (batchSize <= 0 || used.size() != static_cast<size_t>(batchSize)) {
            KERNEL_LOG_ERROR("FAG %s seqused requires B entries", name);
            return false;
        }
        for (size_t i = 0; i < used.size(); ++i) {
            if (used[i] < 0 || (maxSeqlen >= 0 && used[i] > maxSeqlen)) {
                KERNEL_LOG_ERROR("FAG %s seqused at batch %zu must be in [0, physical S]", name, i);
                return false;
            }
        }
        return true;
    }
    if (batchSize <= 0 || maxSeqlen < 0 || cu.size() != static_cast<size_t>(batchSize) + 1 || cu[0] != 0) {
        KERNEL_LOG_ERROR("FAG %s requires B+1 cumulative lengths starting at zero and non-negative max_seqlen", name);
        return false;
    }
    if (!used.empty() && used.size() != static_cast<size_t>(batchSize)) {
        KERNEL_LOG_ERROR("FAG %s seqused must have exactly B entries", name);
        return false;
    }
    for (size_t i = 0; i < static_cast<size_t>(batchSize); ++i) {
        if (cu[i + 1] < cu[i] || cu[i + 1] - cu[i] > maxSeqlen) {
            KERNEL_LOG_ERROR("FAG %s physical length at batch %zu must be in [0, max_seqlen]", name, i);
            return false;
        }
        if (!used.empty() && (used[i] < 0 || used[i] > cu[i + 1] - cu[i])) {
            KERNEL_LOG_ERROR("FAG %s seqused at batch %zu must be in [0, physical length]", name, i);
            return false;
        }
    }
    return true;
}
} // namespace

uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    // ===== FAG metadata start =====
    // Re-read seqused tensors right before FAG via direct int32 access.
    // When an AicpuTaskSpace cache hit causes the reused task descriptor to
    // carry a stale (undefined) dtype for an optional-input slot, the
    // GetTensorDataAsInt64 switch falls through and returns a zero-filled
    // vector. The zeros then masquerade as a legitimate all-zero seqused input
    // in CheckActualQuerySeq/CheckActualKvSeq, polluting actualSeqlenQ_/Kv_.
    // Direct pointer reads bypass the dtype dispatch and always see the real
    // device data, so we use them to detect and recover from this condition.
    // Only triggers when the early-read vector is all-zero AND the direct
    // read yields non-zero values — legitimate all-zero seqused inputs are
    // never touched.
    if (fagIsGradEnabled_) {
        auto isAllZero = [](const std::vector<int64_t>& v) {
            if (v.empty()) {
                return false;
            }
            for (size_t i = 0; i < v.size(); i++) {
                if (v[i] != 0) {
                    return false;
                }
            }
            return true;
        };
        auto directReadInt32 = [](Tensor* t) -> std::vector<int64_t> {
            // NOTE: This assumes INT32 data (enforced by host-side
            // FlashAttnMetadataCheck::CheckSeqLens in flash_attn_metadata_check.h).
            // If the dtype restriction is relaxed, this reader must be updated.
            if (t == nullptr || t->GetData() == nullptr) {
                return {};
            }
            auto sh = t->GetTensorShape();
            const int64_t n = (sh != nullptr && sh->GetDims() > 0) ? sh->GetDimSize(0) : 0;
            if (n <= 0 || n > 65536) {
                return {};
            }
            const int32_t* p = static_cast<const int32_t*>(t->GetData());
            std::vector<int64_t> v;
            v.reserve(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; i++) {
                v.push_back(p[i]);
            }
            return v;
        };
        std::vector<int64_t> sqDirect = directReadInt32(sequsedQ_);
        if (!sqDirect.empty() && isAllZero(actualSeqlenQ_) && !isAllZero(sqDirect)) {
            actualSeqlenQ_ = sqDirect;
            isActualSeqlenQAccum_ = false;
        }
        std::vector<int64_t> skvDirect = directReadInt32(sequsedKv_);
        if (!skvDirect.empty() && isAllZero(actualSeqlenKv_) && !isAllZero(skvDirect)) {
            actualSeqlenKv_ = skvDirect;
            isActualSeqlenKvAccum_ = false;
        }
    }

    // FAG (Flash Attn Grad) metadata: only executed when is_grad_enabled=true.
    if (fagIsGradEnabled_) {
        InitFagParams();
        DoFagSparse();
        GenFagMetadata(splitRes.sectionNum);
    }
    // ===== FAG metadata end =====

    return FA_KERNEL_STATUS_OK;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
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
    GetAttrValueOpt(ctx, "head_dim_v", headDimV_);
    // head_dim_v 未指定(-1)时, v 的 head_dim 等于 head_dim
    if (headDimV_ == -1) {
        headDimV_ = headDim_;
    }
    GetAttrValueOpt(ctx, "is_grad_enabled", fagIsGradEnabled_);

    KERNEL_CHECK_FALSE(ParamsCheck(), false, "Params check failed");
    return ParamsInit();
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    InitBaseInfo();
    InitLoadBalanceParams();
    InitDeviceInfo();
    return true;
}

bool FlashAttnMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(), false, "Check query sequence failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    // Non-TND layouts: metadata has no q/k/v tensors, so a side without
    // seqused can only derive its physical length (default fill in
    // InitFagActualSeqlen and the split grid) from max_seqlen. When that side
    // provides seqused, max_seqlen is optional per the interface doc.
    // TND layouts derive max_seqlen from cu_seqlens (see CheckActual*Seq).
    if (fagIsGradEnabled_ && (layoutQ_ != "TND" || layoutKv_ != "TND")) {
        KERNEL_CHECK_FALSE(batchSize_ > 0, false, "FAG metadata requires B > 0");
    }
    return true;
}

bool FlashAttnMetadataCpuKernel::CheckActualQuerySeq()
{
    isActualSeqlenQAccum_ = false;
    actualSeqlenQ_.clear();
    std::vector<int64_t> cuSeqlensQ{};
    std::vector<int64_t> sequsedQ{};

    cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    sequsedQ = GetTensorDataAsInt64(sequsedQ_);
    // Optional per the interface doc: under TND layout, max_seqlen_q may be
    // omitted (<0) and is derived from cu_seqlens_q. Use the packed token
    // total (cu.back()) so host (q.shape T-dim fallback) and AICPU share the
    // same derived value in grid/split formulas.
    if (fagIsGradEnabled_ && layoutQ_ == "TND" && maxSeqlenQ_ < 0 && !cuSeqlensQ.empty()) {
        maxSeqlenQ_ = static_cast<int32_t>(cuSeqlensQ.back());
    }
    if (fagIsGradEnabled_ &&
        !CheckFagSequence(cuSeqlensQ, sequsedQ, batchSize_, maxSeqlenQ_, layoutQ_ == "TND", "query")) {
        return false;
    }
    fagPhysicalTokensQ_ = cuSeqlensQ.empty() ? 0 : cuSeqlensQ.back();
    fagHasSeqused_ = !sequsedQ.empty();

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

bool FlashAttnMetadataCpuKernel::CheckActualKvSeq()
{
    isActualSeqlenKvAccum_ = false;
    actualSeqlenKv_.clear();
    std::vector<int64_t> cuSeqlensKv{};
    std::vector<int64_t> sequsedKv{};

    cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_);
    sequsedKv = GetTensorDataAsInt64(sequsedKv_);
    // Optional per the interface doc: under TND layout, max_seqlen_kv may be
    // omitted (<0) and is derived from cu_seqlens_kv (packed token total).
    if (fagIsGradEnabled_ && layoutKv_ == "TND" && maxSeqlenKv_ < 0 && !cuSeqlensKv.empty()) {
        maxSeqlenKv_ = static_cast<int32_t>(cuSeqlensKv.back());
    }
    if (fagIsGradEnabled_ &&
        !CheckFagSequence(cuSeqlensKv, sequsedKv, batchSize_, maxSeqlenKv_, layoutKv_ == "TND", "kv")) {
        return false;
    }
    fagPhysicalTokensKv_ = cuSeqlensKv.empty() ? 0 : cuSeqlensKv.back();
    fagHasSeqused_ = fagHasSeqused_ || !sequsedKv.empty();

    for (size_t i = 0; i < sequsedKv.size(); ++i) {
        if (sequsedKv[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedKv must be non-negative, but %zuth element is %ld", i,
                             sequsedKv[i]);
            return false;
        }
    }

    if (!cuSeqlensKv.empty()) {
        if (cuSeqlensKv[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensKv must be 0, but got %ld", cuSeqlensKv[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensKv.size(); ++i) {
        if (cuSeqlensKv[i] < cuSeqlensKv[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensKv must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensKv[i], cuSeqlensKv[i - 1]);
            return false;
        }
    }

    if (!sequsedKv.empty()) {
        isActualSeqlenKvAccum_ = false;
        actualSeqlenKv_ = sequsedKv;
    } else if (!cuSeqlensKv.empty()) {
        isActualSeqlenKvAccum_ = true;
        actualSeqlenKv_.assign(cuSeqlensKv.begin() + 1, cuSeqlensKv.end());
    }

    return true;
}

void FlashAttnMetadataCpuKernel::InitDeviceInfo()
{
    // Keep aicCoreNum_/aivCoreNum_ unchanged: they define metadata strides and the FAG layout.
    uint32_t maxUsedAicCores = optiling::flash_attn::fa_tiling_util::GetMaxUsedAicCores(
        aicCoreNum_, baseInfo.batchSize, numHeadsKv_, numHeadsQ_ / numHeadsKv_, maxSeqlenQ_, maxSeqlenKv_, mBaseSize_,
        s2BaseSize_);
    uint32_t maxUsedAivCores = maxUsedAicCores * (aivCoreNum_ / aicCoreNum_);
    deviceInfo.aicCoreMaxNum = maxUsedAicCores;
    deviceInfo.aivCoreMaxNum = maxUsedAivCores;
    deviceInfo.aicCoreMinNum = maxUsedAicCores;
    deviceInfo.aivCoreMinNum = maxUsedAivCores;
}

int64_t FlashAttnMetadataCpuKernel::CostFunc(uint32_t basicM, uint32_t basicS2)
{
    const uint32_t mAlignCoef = 16U;
    const uint32_t s2AlignCoef = 64U;
    const float mCoef = -0.037f;
    const float s2Coef = 0.11f;
    const float constant = 3.45f;

    float alignBasicM = static_cast<float>(load_balance::CeilDiv(basicM, mAlignCoef));
    float alignBasicS2 = static_cast<float>(load_balance::CeilDiv(basicS2, s2AlignCoef));

    return static_cast<int64_t>(mCoef * alignBasicM + s2Coef * alignBasicS2 + constant);
}

void FlashAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    uint32_t qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == load_balance::Layout::BSH || baseInfo.layoutQuery == load_balance::Layout::BSND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == load_balance::Layout::TND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_TND;
    }
    uint32_t gSize = static_cast<uint32_t>(numHeadsQ_ / numHeadsKv_);
    bool dnUnmergedRouted = false;
    if (socVersion_.find("Ascend960") != std::string::npos) {
        // arch92: 基本块按 vHeadDim 档位决定, 同步输出 templateId/mBaseSize/s2BaseSize
        uint32_t templateId = 0;
        optiling::flash_attn::fa_tiling_util::GetConfigInfo(static_cast<uint32_t>(headDimV_), maskMode_, maxSeqlenQ_,
                                                            templateId, mBaseSize_, s2BaseSize_);
        dnUnmergedRouted = templateId == optiling::flash_attn::fa_tiling_util::FA_TEMPLATE_DN_UNMERGED;
        param.costFunc = nullptr;
    } else {
        optiling::flash_attn::fa_tiling_util::AdjustSinnerAndSouter(headDimV_, gSize, maxSeqlenQ_, maxSeqlenKv_,
                                                                    maskMode_, baseInfo.preToken, baseInfo.nextToken,
                                                                    qlayout, mBaseSize_, s2BaseSize_);
        // 不合轴 DN(预埋): 路由开关打开时按 BN1_S1 分核(B轴*N1轴, M轴=纯S1); 判定逻辑与 host
        // 侧 UpdateTilingKeyTemplateId 共享 fa_tiling_util::GetFlashAttnTemplateId, 保证 metadata
        // 分核模式与 kernel 模板选择严格一致; 开关关闭时维持原 BN2_S1G 分核, 不影响现网行为。
        dnUnmergedRouted = optiling::flash_attn::fa_tiling_util::DN_UNMERGED_ROUTE_ENABLED &&
                           optiling::flash_attn::fa_tiling_util::GetFlashAttnTemplateId(
                               static_cast<uint32_t>(headDim_), static_cast<uint32_t>(headDimV_)) ==
                               optiling::flash_attn::fa_tiling_util::FA_TEMPLATE_DN_UNMERGED;
        param.costFunc = FlashAttnMetadataCpuKernel::CostFunc;
    }
    mBaseSize_ *= (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96U * 1024U * 1024U; // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb
    param.fdTolerance = 10;             // 10: tolerance block
    param.fdLeastBlock = 3;             // 3: least block
    param.fdOn = true;
    param.kernelSplitMode =
        dnUnmergedRouted ? load_balance::KernelSplitMode::BN1_S1_S2 : load_balance::KernelSplitMode::BN2_S1G_S2;
}

void FlashAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = (actualSeqlenQ_.empty()) ? batchSize_ : actualSeqlenQ_.size();
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDimV_;
    load_balance::SparseMode maskMode = load_balance::SparseMode::BUTT;
    if (maskMode_ != 0) {
        maskMode = static_cast<load_balance::SparseMode>(maskMode_);
    }
    baseInfo.attenMaskFlag = (maskMode != load_balance::SparseMode::BUTT);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode);
    baseInfo.preToken = (winLeft_ == -1) ? static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) : winLeft_;
    baseInfo.nextToken = (winRight_ == -1) ? static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) : winRight_;
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = isActualSeqlenKvAccum_;
    baseInfo.actualKvSeqSize = actualSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = isActualSeqlenQAccum_;
    baseInfo.actualQuerySeqSize = actualSeqlenQ_;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult& splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult& splitRes)
{
    detail::FaMetadata faMetadata(aicCoreNum_, aivCoreNum_, splitRes.sectionNum, metadata_->GetData());
    faMetadata.Clear(); // set to all 0

    SetMetadataHead(splitRes, faMetadata);
    SetMetadataFa(splitRes, faMetadata);
    SetMetadataFd(splitRes, faMetadata);
    return true;
}

void FlashAttnMetadataCpuKernel::SetMetadataHead(const load_balance::SectionStreamKResult& splitRes,
                                                 optiling::detail::FaMetadata& faMetadata)
{
    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, aicCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, aivCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX, static_cast<FA_METADATA_T>(param.kernelSplitMode));
}

void FlashAttnMetadataCpuKernel::SetMetadataFa(const load_balance::SectionStreamKResult& splitRes,
                                               optiling::detail::FaMetadata& faMetadata)
{
    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bnStart = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_START_INDEX, bnStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }
    }
}

void FlashAttnMetadataCpuKernel::SetMetadataFd(const load_balance::SectionStreamKResult& splitRes,
                                               optiling::detail::FaMetadata& faMetadata)
{
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
}

namespace {
static const char* kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
