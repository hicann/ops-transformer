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
 * \file quant_flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "quant_flash_attn_metadata_aicpu.h"
#include "../../quant_flash_attn/op_host/qfa_adjust_sinner_souter.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantFlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantFlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    GetAttrValueOpt(ctx, "quant_compute_mode", quantMode_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_q_descale", layoutQDescale_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    GetAttrValueOpt(ctx, "is_grad_enabled", isGradEnabled_);
    GetAttrValueOpt(ctx, "head_dim_v", headDimV_);
    return ParamsInit();
}

std::vector<int64_t> QuantFlashAttnMetadataCpuKernel::GetTensorDataAsInt64(Tensor *tensor, size_t size)
{
    std::vector<int64_t> result(size);
    if (tensor == nullptr || tensor->GetData() == nullptr || size == 0) {
        return result;
    }

    DataType dataType = tensor->GetDataType();
    void *data = tensor->GetData();

    switch (dataType) {
        case DT_INT32: {
            int32_t *ptr = static_cast<int32_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_INT64: {
            int64_t *ptr = static_cast<int64_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = ptr[i];
            }
            break;
        }
        case DT_INT16: {
            int16_t *ptr = static_cast<int16_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT32: {
            uint32_t *ptr = static_cast<uint32_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT64: {
            uint64_t *ptr = static_cast<uint64_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT16: {
            uint16_t *ptr = static_cast<uint16_t *>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        default:
            break;
    }
    return result;
}

bool QuantFlashAttnMetadataCpuKernel::ParamsInit()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aicCoreMinNum = 1;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND" || layoutQ_ == "NTD";
    baseInfo.isCumulativeKvSeq = layoutKv_ == "TND" || layoutKv_ == "NTD";
    if (batchSize_ > 0) {
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        baseInfo.actualKvSeqSize.resize(batchSize_, baseInfo.kvSeqSize);
        if (baseInfo.isCumulativeQuerySeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
        }
        if (baseInfo.isCumulativeKvSeq) {
            for (uint32_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
        }
    }
    if (baseInfo.isCumulativeQuerySeq && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, batchSize_ + 1);
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            baseInfo.querySeqSize =
                std::max(static_cast<int64_t>(baseInfo.querySeqSize), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, batchSize_);
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            if (baseInfo.isCumulativeQuerySeq && (i > 0)) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
            baseInfo.querySeqSize = std::max(static_cast<int64_t>(baseInfo.querySeqSize), sequsedQ[i]);
        }
    }
    if (baseInfo.isCumulativeKvSeq && cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        batchSize_ = cuSeqlensKv_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_, batchSize_ + 1);
        baseInfo.actualKvSeqSize.resize(batchSize_, baseInfo.kvSeqSize);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i + 1];
            baseInfo.kvSeqSize =
                std::max(static_cast<int64_t>(baseInfo.kvSeqSize), cuSeqlensKv[i + 1] - cuSeqlensKv[i]);
        }
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        batchSize_ = sequsedKv_->GetTensorShape()->GetDimSize(0);
        auto sequsedKv = GetTensorDataAsInt64(sequsedKv_, batchSize_);
        baseInfo.actualKvSeqSize.resize(batchSize_, baseInfo.kvSeqSize);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
            if (baseInfo.isCumulativeKvSeq && (i > 0)) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
            baseInfo.kvSeqSize = std::max(static_cast<int64_t>(baseInfo.kvSeqSize), sequsedKv[i]);
        }
    }
    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    bool isDecode = (layoutQDescale_ == "N2TGD");
    baseInfo.kvHeadNum = isDecode ? numHeadsKv_ : numHeadsQ_;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDim_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    if (quantMode_ == 1 || quantMode_ == 6 || quantMode_ == 0) {
        baseInfo.queryType = load_balance::DataType::INT8;
        baseInfo.kvType = load_balance::DataType::INT8;
    }
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    optiling::quant_flash_attn::qfa_tiling_util::AdjustSinnerAndSouter(
        static_cast<uint32_t>(headDim_), static_cast<int64_t>(maxSeqlenQ_), static_cast<int64_t>(maxSeqlenKv_),
        maskMode_, static_cast<int64_t>(winLeft_), static_cast<int64_t>(winRight_),
        optiling::quant_flash_attn::qfa_tiling_util::LAYOUT_BSND, static_cast<uint32_t>(quantMode_), sOuterFactor,
        sInnerFactor);
    mBaseSize_ = sOuterFactor;
    s2BaseSize_ = sInnerFactor;
    mBaseSize_ = mBaseSize_ * (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 0;
    param.fdTolerance = 300;
    param.fdOn = 0;
    param.outputLayout = load_balance::OutputLayout::BN2_S1G;

    if (isGradEnabled_) {
        int64_t deterMaxRound = CalDeterMaxRound();
        detail::QuantFAGMetaData quantFAGMetaData(metaData_->GetData());
        quantFAGMetaData.SetDeterMaxRound(optiling::QUANT_FAG_DETER_MAX_NUM_INDEX, deterMaxRound);
    }
    needInitOutput_ = CheckNeedInitOutput();
    return true;
}

// 确定性计算最大循环次数
int64_t QuantFlashAttnMetadataCpuKernel::CalDeterMaxRound()
{
    s1Size_ = GetS1SeqSize(0);
    s2Size_ = GetS2SeqSize(0);
    // 非TND场景
    int64_t b = batchSize_ * baseInfo.kvHeadNum;
    int64_t m = CeilDiv<int64_t>(s1Size_, 512);
    int64_t n = CeilDiv<int64_t>(s2Size_, 512);
    int64_t k = aicCoreNum_;

    if (n == 1) {
        return std::max(CeilDiv<int64_t>(m * b, k), m);
    } else {
        return CeilDiv<int64_t>(n * b, std::min(k, m * b)) * m;
    }
}

uint32_t QuantFlashAttnMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }

    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t QuantFlashAttnMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }

    if (layoutKv_ == "TND") {
        if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensKv_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenKv_);
}

bool QuantFlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool QuantFlashAttnMetadataCpuKernel::CheckNeedInitOutput()
{
    const bool hasCuQ = cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr;
    const bool hasCuKv = cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr;
    const bool hasSeqQ = sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr;
    const bool hasSeqKv = sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr;
    const uint32_t bSize = static_cast<uint32_t>(batchSize_);
    const bool hasVarlen = hasCuQ || hasCuKv || hasSeqQ || hasSeqKv;
    if (!hasVarlen || bSize == 0) {
        // 无 varlen 信息时用全局长度兜底
        return maskMode_ == 3 && baseInfo.querySeqSize > baseInfo.kvSeqSize;
    }
    // 直接读原始 varlen 数据 (cu_seqlens 长度为 b+1 且首元素为 0, seqused 长度为 b),
    // 不使用 ParamsInit 中被累加覆盖的 actualQuerySeqSize/actualKvSeqSize
    std::vector<int64_t> cuSeqlensQ;
    std::vector<int64_t> cuSeqlensKv;
    std::vector<int64_t> seqUsedQ;
    std::vector<int64_t> seqUsedKv;
    if (hasCuQ) {
        cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, bSize + 1);
    }
    if (hasCuKv) {
        cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_, bSize + 1);
    }
    if (hasSeqQ) {
        seqUsedQ = GetTensorDataAsInt64(sequsedQ_, bSize);
    }
    if (hasSeqKv) {
        seqUsedKv = GetTensorDataAsInt64(sequsedKv_, bSize);
    }
    for (uint32_t bIdx = 0; bIdx < bSize; ++bIdx) {
        // Q 侧: cu_seqlens_q 差分为 0 → 零长 batch
        int64_t qAllocLen = hasCuQ ? cuSeqlensQ[bIdx + 1] - cuSeqlensQ[bIdx] : -1;
        if (qAllocLen == 0) {
            return true;
        }
        // Q 侧: seqused_q 为 0, 或小于 cu_seqlens_q 分配长度 → 输出存在 padding 行, 需要清零
        int64_t qUsedLen = hasSeqQ ? seqUsedQ[bIdx] : (qAllocLen > 0 ? qAllocLen : baseInfo.querySeqSize);
        if (qUsedLen == 0) {
            return true;
        }
        if (qAllocLen > 0 && qUsedLen < qAllocLen) {
            return true;
        }
        // KV 侧: cu_seqlens_kv 差分为 0 → 零长 batch
        int64_t kvAllocLen = hasCuKv ? cuSeqlensKv[bIdx + 1] - cuSeqlensKv[bIdx] : -1;
        if (kvAllocLen == 0) {
            return true;
        }
        // KV 侧: seqused_kv 为 0 → 该 batch 无有效 kv, 输出应全 0
        int64_t kvLen = hasSeqKv ? seqUsedKv[bIdx] : (kvAllocLen > 0 ? kvAllocLen : baseInfo.kvSeqSize);
        if (kvLen == 0) {
            return true;
        }
        // CAUSAL 下单 batch q > kv → 产生全 mask 行, 输出应为 0
        if (maskMode_ == 3 && qUsedLen > kvLen) {
            return true;
        }
    }
    return false;
}

bool QuantFlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult &splitRes)
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    uint32_t sectionNum = splitRes.sectionNum;
    detail::FaMetaData faMetadata(metaData_->GetData(), sectionNum);
    faMetadata.SetHeadMedata(optiling::HEAD_SECTION_NUM_INDEX, sectionNum);

    faMetadata.SetHeadMedata(optiling::HEAD_IS_FD_INDEX, 0);
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        if (fdSplitRes.usedVecNum > 0) {
            faMetadata.SetHeadMedata(optiling::HEAD_IS_FD_INDEX, 1);
        }
    }

    faMetadata.SetHeadMedata(optiling::HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMedata(optiling::HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);

    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_BN_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, 0U);
        }

        auto faSplitRes = splitRes.sectionFaResult[sectionId];
        for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            if (i > 0) {
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_START_INDEX, faSplitRes.bNEnd[i - 1]);
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, faSplitRes.mEnd[i - 1]);
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, faSplitRes.s2End[i - 1]);
            } else if (sectionId > 0) {
                auto preFaSplitRes = splitRes.sectionFaResult[sectionId - 1];
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_START_INDEX,
                                         preFaSplitRes.bNEnd[preFaSplitRes.usedCoreNum - 1]);
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX,
                                         preFaSplitRes.mEnd[preFaSplitRes.usedCoreNum - 1]);
                faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX,
                                         preFaSplitRes.s2End[preFaSplitRes.usedCoreNum - 1]);
            }
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_END_INDEX, faSplitRes.bNEnd[i]);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, faSplitRes.mEnd[i]);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, faSplitRes.s2End[i]);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faSplitRes.firstFdDataWorkspaceIdx[i]);
        }

        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        for (uint32_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
            uint32_t curTaskIdx = fdSplitRes.taskIdx[i];
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_BN_IDX_INDEX, fdSplitRes.bNIdx[curTaskIdx]);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, fdSplitRes.mIdx[curTaskIdx]);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX,
                                     fdSplitRes.workspaceIdx[curTaskIdx]);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, fdSplitRes.s2SplitNum[curTaskIdx]);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, fdSplitRes.mStart[i]);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, fdSplitRes.mLen[i]);
        }
    }
    faMetadata.SetHeadMedata(optiling::HEAD_NEED_INIT_OUTPUT_INDEX, needInitOutput_ ? 1U : 0U);
    return true;
}

namespace {
static const char *kernelType = "QuantFlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantFlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
