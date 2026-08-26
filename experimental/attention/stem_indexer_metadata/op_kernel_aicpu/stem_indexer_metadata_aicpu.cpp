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
 * \file stem_indexer_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "stem_indexer_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
namespace {
constexpr uint32_t STEM_M_BASE_SIZE = 64U;
constexpr uint32_t STEM_S2_BASE_SIZE = 256U;
} // namespace

uint32_t StemIndexerMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    SectionStreamKResult result;
    success = BalanceSchedule(result);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    success = GenMetadata(result);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool StemIndexerMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    qSeqLens_ = ctx.Input(static_cast<uint32_t>(ParamId::qSeqLens));
    kvSeqLens_ = ctx.Input(static_cast<uint32_t>(ParamId::kvSeqLens));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs =
        GetAttrValue(ctx, "q_heads", numHeadsQ_) && GetAttrValue(ctx, "kv_heads", numHeadsKv_) &&
        GetAttrValue(ctx, "causal", causal_) && GetAttrValue(ctx, "stem_block_size", stemBlockSize_) &&
        GetAttrValue(ctx, "window_size", windowSize_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "dim_qkflat", headDim_);
    return true;
}

std::vector<int64_t> StemIndexerMetadataCpuKernel::GetTensorDataAsInt64(Tensor *tensor, size_t size)
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

bool StemIndexerMetadataCpuKernel::BalanceSchedule(SectionStreamKResult &result)
{
    DeviceInfo deviceInfo{};
    StemIndexerBaseInfo baseInfo{};
    load_balance::SectionStreamKParam param{};
    auto success = GenerateDeviceInfo(deviceInfo) && GenerateBaseInfo(baseInfo) && GenerateSectionStreamKParam(param);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, result) == SECTION_STREAM_K_SUCCESS;
}

bool StemIndexerMetadataCpuKernel::GenerateDeviceInfo(DeviceInfo &deviceInfo)
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
    return true;
}

bool StemIndexerMetadataCpuKernel::GenerateBaseInfo(StemIndexerBaseInfo &baseInfo)
{
    KERNEL_CHECK_NULLPTR(qSeqLens_, false, "q_seq_len is nullptr!");
    KERNEL_CHECK_NULLPTR(kvSeqLens_, false, "kv_seq_len is nullptr!");
    KERNEL_CHECK_FALSE(qSeqLens_->NumElements() == kvSeqLens_->NumElements(), false,
                       "q_seq_len(%ld) has different length with kv_seq_len(%ld)", qSeqLens_->NumElements(),
                       kvSeqLens_->NumElements());

    size_t batchSize = qSeqLens_->NumElements();

    baseInfo.batchSize = static_cast<uint32_t>(batchSize);
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.querySeqSize = 0;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.kvSeqSize = 0;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDim_;
    baseInfo.attenMaskFlag = causal_;
    baseInfo.sparseMode = (causal_) ? static_cast<uint32_t>(load_balance::SparseMode::RIGHT_DOWN_CAUSAL) :
                                      static_cast<uint32_t>(load_balance::SparseMode::BUTT);
    baseInfo.preToken = -1;
    baseInfo.nextToken = -1;
    baseInfo.layoutQuery = load_balance::Layout::BUTT;
    baseInfo.layoutKv = load_balance::Layout::BUTT;
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeQuerySeq = false;
    baseInfo.isCumulativeKvSeq = false;
    baseInfo.actualQuerySeqSize = GetTensorDataAsInt64(qSeqLens_, batchSize);
    baseInfo.actualKvSeqSize = GetTensorDataAsInt64(kvSeqLens_, batchSize);
    for (size_t i = 0; i < batchSize; ++i) {
        baseInfo.actualQuerySeqSize[i] = load_balance::CeilDiv(baseInfo.actualQuerySeqSize[i], stemBlockSize_);
        baseInfo.actualKvSeqSize[i] = load_balance::CeilDiv(baseInfo.actualKvSeqSize[i], stemBlockSize_);
    }
    // New
    baseInfo.tailSize = windowSize_;

    return true;
}

bool StemIndexerMetadataCpuKernel::GenerateSectionStreamKParam(load_balance::SectionStreamKParam &param)
{
    param.l2Byte = 96U * 1024U * 1024U;
    param.mBaseSize = STEM_M_BASE_SIZE;
    param.s2BaseSize = STEM_S2_BASE_SIZE;
    param.fdOn = false;
    param.outputLayout = load_balance::OutputLayout::BN2_S1G;
    return true;
}

bool StemIndexerMetadataCpuKernel::GenMetadata(SectionStreamKResult &result)
{
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    uint64_t requiredElems = optiling::GetMetadataRequiredElems(result.sectionNum);
    if (static_cast<uint64_t>(metadata_->NumElements()) < requiredElems) {
        KERNEL_LOG_ERROR("metadata has %ld elements, but sectionNum %u requires at least %llu elements",
                         metadata_->NumElements(), result.sectionNum, static_cast<unsigned long long>(requiredElems));
        return false;
    }

    optiling::detail::SliMetadata sliMetadata(metadata_->GetData(), result.sectionNum);
    sliMetadata.Clear();

    sliMetadata.SetHeadMetadata(optiling::HEAD_SECTION_NUM_INDEX, result.sectionNum);

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)};
    for (uint32_t secIdx = 0; secIdx < result.sectionNum; ++secIdx) {
        auto &faRes = result.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto &prevFaRes = (secIdx == 0U) ? dummyHead : result.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            SLI_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            SLI_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            SLI_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_BN_START_INDEX, bn2Start);
            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_M_START_INDEX, mStart);
            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_S2_START_INDEX, s2Start);
            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_BN_END_INDEX, faRes.bNEnd[aicIdx]);
            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_M_END_INDEX, faRes.mEnd[aicIdx]);
            sliMetadata.SetFaMetadata(secIdx, aicIdx, optiling::SLI_SEC_S2_END_INDEX, faRes.s2End[aicIdx]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "StemIndexerMetadata";
REGISTER_CPU_KERNEL(kernelType, StemIndexerMetadataCpuKernel);
} // namespace

} // namespace aicpu
