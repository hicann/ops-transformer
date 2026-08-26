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
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    dequantScaleV_ = ctx.Input(static_cast<uint32_t>(ParamId::dequantScaleV));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "quant_mode", quantMode_);
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
    return ParamsInit();
    // return true;
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
    // Device info
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    // deviceInfo.socVersion = socVersion_;
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
    if (!baseInfo.isCumulativeQuerySeq && sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, batchSize_);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), sequsedQ[i]);
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, batchSize_ + 1);
        baseInfo.actualQuerySeqSize.resize(batchSize_, maxSeqlenQ_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            maxSeqlenQ_ = std::max(static_cast<int64_t>(maxSeqlenQ_), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (!baseInfo.isCumulativeKvSeq && sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        batchSize_ = sequsedKv_->GetTensorShape()->GetDimSize(0);
        auto sequsedKv = GetTensorDataAsInt64(sequsedKv_, batchSize_);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = sequsedKv[i];
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), sequsedKv[i]);
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        batchSize_ = cuSeqlensKv_->GetTensorShape()->GetDimSize(0) - 1;
        auto cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_, batchSize_ + 1);
        baseInfo.actualKvSeqSize.resize(batchSize_, maxSeqlenKv_);
        for (uint32_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualKvSeqSize[i] = cuSeqlensKv[i + 1];
            maxSeqlenKv_ = std::max(static_cast<int64_t>(maxSeqlenKv_), cuSeqlensKv[i + 1] - cuSeqlensKv[i]);
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
    baseInfo.sparseMode = maskMode_;
    baseInfo.preToken = winLeft_ == -1 ? std::numeric_limits<uint32_t>::max() : winLeft_;
    baseInfo.nextToken = winRight_ == -1 ? std::numeric_limits<uint32_t>::max() : winRight_;
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = ConvertToLayout(layoutKv_);
    if (quantMode_ == 1 || quantMode_ == 2) {
        baseInfo.queryType = load_balance::DataType::INT8;
        baseInfo.kvType = load_balance::DataType::INT8;
    }

    // param
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
        groupSize_ = 1;
    } else {
        groupSize_ = numHeadsQ_ / numHeadsKv_;
    }
    mBaseSize_ = 128;
    s2BaseSize_ = 256;
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 0U; // sectionNum = 1
    param.fdOn = false;

    // 校验 dequant_scale_v: BNSD layout 下, shape 应为 (B, N, ceil(S/64), D, 2)
    if (dequantScaleV_ != nullptr && dequantScaleV_->GetTensorShape() != nullptr) {
        const int64_t DEQUANT_SCALE_V_GROUP_SIZE = 64;
        bool isBnsdLayout = layoutKv_ == "BNSD";
        if (isBnsdLayout) {
            int64_t maxActualKvSeq = 0;
            for (int32_t i = 0; i < batchSize_; ++i) {
                if (i < static_cast<int32_t>(baseInfo.actualKvSeqSize.size())) {
                    maxActualKvSeq = std::max(maxActualKvSeq, static_cast<int64_t>(baseInfo.actualKvSeqSize[i]));
                }
            }
            // seqused_kv 未传入时, actualKvSeqSize 为空或由 max_seqlen_kv 填充
            if (maxActualKvSeq <= 0) {
                maxActualKvSeq = maxSeqlenKv_;
            }
            int64_t expectedSGroup = (maxActualKvSeq + DEQUANT_SCALE_V_GROUP_SIZE - 1) / DEQUANT_SCALE_V_GROUP_SIZE;
            int64_t actualSGroup = dequantScaleV_->GetTensorShape()->GetDimSize(0);
            int64_t actualBND2 = batchSize_ * numHeadsKv_ * headDim_ * 2;
            if (actualBND2 == 0) {
                return true;
            }
            actualSGroup = actualSGroup / actualBND2;
            if (expectedSGroup > 0 && expectedSGroup != actualSGroup) {
                KERNEL_LOG_ERROR(
                    "dequant_scale_v dim2 should be ceil(max(seqused_kv)/64) = %ld when layout is BNSD, but got %ld",
                    expectedSGroup, actualSGroup);
                return false;
            }
        }
    }
    return true;
}

bool QuantFlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool QuantFlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult &splitRes)
{
    uint32_t sectionNum = splitRes.sectionNum;
    detail::QFaMetaData faMetadata(metaData_->GetData(), sectionNum);
    uint32_t *ptr = (uint32_t *)metaData_->GetData();
    ptr[1] = mBaseSize_;
    ptr[2] = s2BaseSize_;
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        for (uint32_t i = 0; i < AIC_CORE_NUM; ++i) {
            // faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_CORE_ENABLE_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_END_INDEX, 0U);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX, 0U);
        }
        for (uint32_t i = 0; i < AIV_CORE_NUM; ++i) {
            // faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_CORE_ENABLE_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_BN2_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_IDX_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_NUM_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_START_INDEX, 0U);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_NUM_INDEX, 0U);
        }
        // QFA Metadata Generate
        auto faSplitRes = splitRes.sectionFaResult[sectionId];
        for (uint32_t i = 0; i < faSplitRes.usedCoreNum; ++i) {
            // faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_CORE_ENABLE_INDEX, 1U);
            // QFA start
            if (i > 0) {
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX, faSplitRes.bNEnd[i - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX, faSplitRes.mEnd[i - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX, faSplitRes.s2End[i - 1]);
            } else if (sectionId > 0) {
                auto preQFaSplitRes = splitRes.sectionFaResult[sectionId - 1];
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX,
                                          preQFaSplitRes.bNEnd[preQFaSplitRes.usedCoreNum - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX,
                                          preQFaSplitRes.mEnd[preQFaSplitRes.usedCoreNum - 1]);
                faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX,
                                          preQFaSplitRes.s2End[preQFaSplitRes.usedCoreNum - 1]);
            }
            // QFA end
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_BN2_END_INDEX, faSplitRes.bNEnd[i]);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_M_END_INDEX, faSplitRes.mEnd[i]);
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_S2_END_INDEX, faSplitRes.s2End[i]);
            // QFA idx
            faMetadata.setQFaMetadata(sectionId, i, optiling::QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX,
                                      faSplitRes.firstFdDataWorkspaceIdx[i]);
        }
        // QFD Metadata Generate
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        for (uint32_t i = 0; i < fdSplitRes.usedVecNum; ++i) {
            // faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_CORE_ENABLE_INDEX, 1U);
            uint32_t curTaskIdx = fdSplitRes.taskIdx[i];
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_BN2_IDX_INDEX, fdSplitRes.bNIdx[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_IDX_INDEX, fdSplitRes.mIdx[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_IDX_INDEX,
                                      fdSplitRes.workspaceIdx[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_NUM_INDEX,
                                      fdSplitRes.s2SplitNum[curTaskIdx]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_START_INDEX, fdSplitRes.mStart[i]);
            faMetadata.setQFdMetadata(sectionId, i, optiling::QFD_M_NUM_INDEX, fdSplitRes.mLen[i]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "QuantFlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantFlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
