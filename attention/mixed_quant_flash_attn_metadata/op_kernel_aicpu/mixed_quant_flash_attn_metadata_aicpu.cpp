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
 * \file mixed_quant_flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "mixed_quant_flash_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t MixedQuantFlashAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool MixedQuantFlashAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
        GetAttrValue(ctx, "quant_compute_mode", quantMode_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutAttnOut_);
    return ParamsInit();
}

std::vector<int64_t> MixedQuantFlashAttnMetadataCpuKernel::GetTensorDataAsInt64(Tensor* tensor, size_t size)
{
    std::vector<int64_t> result(size);
    if (tensor == nullptr || tensor->GetData() == nullptr || size == 0) {
        return result;
    }

    DataType dataType = tensor->GetDataType();
    void* data = tensor->GetData();

    switch (dataType) {
        case DT_INT32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_INT64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = ptr[i];
            }
            break;
        }
        case DT_INT16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            for (size_t i = 0; i < size; ++i) {
                result[i] = static_cast<int64_t>(ptr[i]);
            }
            break;
        }
        case DT_UINT16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
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

inline int64_t MixedQuantFlashAttnMetadataCpuKernel::CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    int64_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    int64_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

bool MixedQuantFlashAttnMetadataCpuKernel::ParamsInit()
{
    // Device info
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    // deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    // baseInfo
    // actual seq size
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND" || layoutQ_ == "NTD";
    baseInfo.isCumulativeKvSeq = layoutKv_ == "TND" || layoutKv_ == "NTD";
    if (batchSize_ > 0) {
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
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
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, batchSize_);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            if (baseInfo.isCumulativeQuerySeq && (i > 0)) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), sequsedQ[i]);
        }
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        batchSize_ = sequsedKv_->GetTensorShape()->GetDimSize(0);
        auto sequsedKv = GetTensorDataAsInt64(sequsedKv_, batchSize_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
            if (baseInfo.isCumulativeKvSeq && (i > 0)) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), sequsedKv[i]);
        }
    }
    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDim_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16; // anti-quant 场景下 Q 固定 位宽（2 bytes）
    // KV 按量化模式选择数据类型，此处 DataType 仅用于表达元素位宽：
    //   quantMode_ == 1 → INT4（0.5 byte，4-bit 量化）
    //   quantMode_ != 1 → INT8（1 byte，8-bit 量化）
    baseInfo.kvType = quantMode_ == 1 ? load_balance::DataType::INT4 : load_balance::DataType::INT8;
    // param
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }

    uint32_t qlayout = optiling::mixed_quant_flash_attn::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == Layout::BSH || baseInfo.layoutQuery == Layout::BSND) {
        qlayout = optiling::mixed_quant_flash_attn::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == Layout::TND) {
        qlayout = optiling::mixed_quant_flash_attn::fa_tiling_util::LAYOUT_TND;
    }
    optiling::mixed_quant_flash_attn::fa_tiling_util::AdjustSinnerAndSouter(
        headDim_, groupSize_, baseInfo.querySeqSize, baseInfo.kvSeqSize, baseInfo.sparseMode, baseInfo.preToken,
        baseInfo.nextToken, qlayout, mBaseSize_, s2BaseSize_);

    mBaseSize_ = mBaseSize_ * (aivCoreNum_ / aicCoreNum_); // CV_Radio
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.costFunc = CalcCost;
    // param.outputLayout = load_balance::OutputLayout::BN1_S1;
    param.l2Byte = 0U; // Set sectionNum to 1
    param.fdTolerance = 0;
    param.fdLeastBlock = 0;
    param.fdOn = (maskMode_ != 4); // TODO: turn off fd for flash attn
    return true;
}

bool MixedQuantFlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult& splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool MixedQuantFlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult& splitRes)
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    uint32_t sectionNum = splitRes.sectionNum;
    bool isS1G = (param.outputLayout == load_balance::OutputLayout::BN2_S1G);
    detail::FaMetaData faMetadata(metaData_->GetData(), sectionNum, static_cast<uint32_t>(aicCoreNum_),
                                  static_cast<uint32_t>(aivCoreNum_));
    faMetadata.SetHeadMetadata(optiling::HEAD_AIC_CORE_NUM_INDEX, static_cast<uint32_t>(aicCoreNum_));
    faMetadata.SetHeadMetadata(optiling::HEAD_AIV_CORE_NUM_INDEX, static_cast<uint32_t>(aivCoreNum_));
    faMetadata.SetHeadMetadata(optiling::HEAD_IS_S1G_INDEX, isS1G ? 1U : 0U);
    faMetadata.SetHeadMetadata(optiling::HEAD_SECTION_NUM_INDEX, sectionNum);

    faMetadata.SetHeadMetadata(optiling::HEAD_IS_FD_INDEX, 0);
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        if (fdSplitRes.usedVecNum > 0) {
            faMetadata.SetHeadMetadata(optiling::HEAD_IS_FD_INDEX, 1);
        }
    }

    faMetadata.SetHeadMetadata(optiling::HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(optiling::HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);

    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        // Clear
        for (uint32_t i = 0; i < static_cast<uint32_t>(aicCoreNum_); ++i) {
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_START_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, 0U);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(aivCoreNum_); ++i) {
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_BN_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_START_INDEX, 0U);
            faMetadata.SetFdMetadata(sectionId, i, optiling::FD_M_NUM_INDEX, 0U);
        }

        // FA Metadata Generate
        auto faSplitRes = splitRes.sectionFaResult[sectionId];
        for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            // FA start
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
            // FA end
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_BN_END_INDEX, faSplitRes.bNEnd[i]);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_M_END_INDEX, faSplitRes.mEnd[i]);
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_S2_END_INDEX, faSplitRes.s2End[i]);
            // FA idx
            faMetadata.SetFaMetadata(sectionId, i, optiling::FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faSplitRes.firstFdDataWorkspaceIdx[i]);
        }

        // FD Metadata Generate
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
    return true;
}

namespace {
static const char* kernelType = "MixedQuantFlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, MixedQuantFlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
