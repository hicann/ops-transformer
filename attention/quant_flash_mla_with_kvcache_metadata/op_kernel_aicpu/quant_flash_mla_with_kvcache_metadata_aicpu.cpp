/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_with_kvcache_metadata_aicpu.cpp
 * \brief QuantFlashMlaWithKvcacheMetadata AICPU算子实现: 使用SectionStreamK负载均衡算法分核并生成metadata
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "quant_flash_mla_with_kvcache_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantFlashMlaWithKvcacheMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantFlashMlaWithKvcacheMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    cacheSeqlens_ = ctx.Input(static_cast<uint32_t>(ParamId::cacheSeqlens));
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "soc_version", socVersion_) && GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
        GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    GetAttrValueOpt(ctx, "quant_mode", quantMode_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "head_dim_qk", headDimQk_);
    GetAttrValueOpt(ctx, "head_dim_v", headDimV_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    return ParamsInit();
}

std::vector<int64_t> QuantFlashMlaWithKvcacheMetadataCpuKernel::GetTensorDataAsInt64(Tensor* tensor, size_t size)
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

bool QuantFlashMlaWithKvcacheMetadataCpuKernel::ParamsInit()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = 1;
    // BaseInfo序列长度为uint32, maxSeqlen未传(-1)时以0初始化, 后续从varlen张量推导
    baseInfo.querySeqSize = (maxSeqlenQ_ > 0) ? static_cast<uint32_t>(maxSeqlenQ_) : 0U;
    baseInfo.kvSeqSize = (maxSeqlenKv_ > 0) ? static_cast<uint32_t>(maxSeqlenKv_) : 0U;
    // MLA PA场景: q侧TND时按cu_seqlens_q(累积, B+1), cache_seqlens为每batch KV实际长度(B个元素, BY_BATCH模式)
    baseInfo.isCumulativeQuerySeq = layoutQ_ == "TND";
    baseInfo.isCumulativeKvSeq = false; // PA场景下cache_seqlens为每batch实际长度, 非累积

    // KV侧: cache_seqlens为必选输入, shape为(batchSize,)
    const bool hasCacheSeqlens = cacheSeqlens_ != nullptr && cacheSeqlens_->GetData() != nullptr;
    if (hasCacheSeqlens) {
        auto cacheSeqlens = GetTensorDataAsInt64(cacheSeqlens_, cacheSeqlens_->NumElements());
        // batch size 由 cacheSeqlens 实际长度推导(每batch一项), 与flmwkcc非量化版本一致
        batchSize_ = static_cast<int32_t>(cacheSeqlens.size());
        baseInfo.actualKvSeqSize.resize(batchSize_, baseInfo.kvSeqSize);
        for (int64_t i = 0; i < batchSize_; ++i) {
            // PA场景: cache_seqlens为每batch实际长度, 直接存入actualKvSeqSize和kvSeqSize
            baseInfo.actualKvSeqSize[i] = cacheSeqlens[i];
            baseInfo.kvSeqSize = std::max(static_cast<int64_t>(baseInfo.kvSeqSize), cacheSeqlens[i]);
        }
    } else if (batchSize_ > 0) {
        // 防御性兜底, 正常不会走到(aclnn层已校验cache_seqlens必选)
        baseInfo.actualKvSeqSize.resize(batchSize_, baseInfo.kvSeqSize);
        if (baseInfo.isCumulativeKvSeq) {
            for (int64_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualKvSeqSize[i] += baseInfo.actualKvSeqSize[i - 1];
            }
        }
    }

    // Q侧: 与qfa接口metadata一致, 读取cu_seqlens_q/seqused_q tensor数据.
    // 先用maxSeqlenQ初始化, 再用tensor数据覆盖.
    if (batchSize_ > 0 && baseInfo.querySeqSize > 0) {
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        if (baseInfo.isCumulativeQuerySeq) {
            for (int64_t i = 1; i < batchSize_; ++i) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
        }
    }
    if (baseInfo.isCumulativeQuerySeq && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        auto cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, cuSeqlensQ_->NumElements());
        batchSize_ = static_cast<int32_t>(cuSeqlensQ.size()) - 1;
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        for (int64_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = cuSeqlensQ[i + 1];
            baseInfo.querySeqSize =
                std::max(static_cast<int64_t>(baseInfo.querySeqSize), cuSeqlensQ[i + 1] - cuSeqlensQ[i]);
        }
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        auto sequsedQ = GetTensorDataAsInt64(sequsedQ_, sequsedQ_->NumElements());
        batchSize_ = static_cast<int32_t>(sequsedQ.size());
        baseInfo.actualQuerySeqSize.resize(batchSize_, baseInfo.querySeqSize);
        for (int64_t i = 0; i < batchSize_; ++i) {
            baseInfo.actualQuerySeqSize[i] = sequsedQ[i];
            if (baseInfo.isCumulativeQuerySeq && (i > 0)) {
                baseInfo.actualQuerySeqSize[i] += baseInfo.actualQuerySeqSize[i - 1];
            }
            baseInfo.querySeqSize = std::max(static_cast<int64_t>(baseInfo.querySeqSize), sequsedQ[i]);
        }
    }

    baseInfo.batchSize = batchSize_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvHeadNum = numHeadsKv_; // MLA固定KV_N=1
    baseInfo.headDimQk = headDimQk_;
    baseInfo.headDimV = headDimV_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.layoutQuery = ConvertToLayout(layoutQ_);
    // KV cache为PA布局, 分核侧统一按BSND处理
    baseInfo.layoutKv = ConvertToLayout("BSND");
    // MLA量化场景: quantMode 1=FP8_E4M3, 0=HIFLOAT8, 按量化模式设置分核代价估算的数据类型(均为1字节)
    load_balance::DataType quantDtype = (quantMode_ == 1) ? load_balance::DataType::FP8_E4M3FN :
                                        (quantMode_ == 0) ? load_balance::DataType::HIFP8 :
                                                            load_balance::DataType::INT8;
    baseInfo.queryType = quantDtype;
    baseInfo.kvType = quantDtype;

    // MLA固定切分: SOuter=32, SInner=256 (与QMLA kernel保持一致)
    s2BaseSize_ = 256U;                                // SInner
    mBaseSize_ = NUM_32 * (aivCoreNum_ / aicCoreNum_); // SOuter * cvRatio
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96 * 1024 * 1024;
    // param.fdTolerance = 300;
    param.fdOn = true;
    // MLA的M轴为GS1合轴(n2=1, g=numHeadsQ), 与FA kernel的M轴遍历方式一致
    param.kernelSplitMode = load_balance::KernelSplitMode::BN2_S1G_S2;

    // 防御: 分核需要有效的Q/KV长度, 无任何varlen信息且max_seqlen未传时报错
    if (baseInfo.querySeqSize == 0U || baseInfo.kvSeqSize == 0U) {
        KERNEL_LOG_ERROR("invalid seq size, querySeqSize: %u, kvSeqSize: %u, need seqused_q/cu_seqlens_q or "
                         "max_seqlen_q/max_seqlen_kv",
                         baseInfo.querySeqSize, baseInfo.kvSeqSize);
        return false;
    }

    needInitOutput_ = CheckNeedInitOutput();
    return true;
}

bool QuantFlashMlaWithKvcacheMetadataCpuKernel::BalanceSchedule(SectionStreamKResult& splitRes)
{
    auto ret = load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
    return ret;
}

bool QuantFlashMlaWithKvcacheMetadataCpuKernel::CheckNeedInitOutput()
{
    // 与qfa接口metadata一致, 读取Q侧和KV侧tensor数据判断是否需要init输出.
    // QMLA PA场景: KV侧用cache_seqlens(等价于qfa的seqused_kv).
    const bool hasCuQ = cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr;
    const bool hasSeqQ = sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr;
    const bool hasCacheSeqlens = cacheSeqlens_ != nullptr && cacheSeqlens_->GetData() != nullptr;
    const uint32_t bSize = static_cast<uint32_t>(batchSize_);
    const bool hasVarlen = hasCuQ || hasSeqQ || hasCacheSeqlens;
    if (!hasVarlen || bSize == 0) {
        return maskMode_ == 3 && baseInfo.querySeqSize > baseInfo.kvSeqSize;
    }
    std::vector<int64_t> cuSeqlensQ;
    std::vector<int64_t> seqUsedQ;
    std::vector<int64_t> cacheSeqlens;
    if (hasCuQ) {
        cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_, bSize + 1);
    }
    if (hasSeqQ) {
        seqUsedQ = GetTensorDataAsInt64(sequsedQ_, bSize);
    }
    if (hasCacheSeqlens) {
        cacheSeqlens = GetTensorDataAsInt64(cacheSeqlens_, bSize);
    }
    for (uint32_t bIdx = 0; bIdx < bSize; ++bIdx) {
        // Q侧: cu_seqlens_q差分为0 → 零长batch
        int64_t qAllocLen = hasCuQ ? cuSeqlensQ[bIdx + 1] - cuSeqlensQ[bIdx] : -1;
        if (qAllocLen == 0) {
            return true;
        }
        // Q侧: seqused_q为0, 或小于分配长度 → 输出存在padding行, 需要清零
        int64_t qUsedLen = hasSeqQ ? seqUsedQ[bIdx] : (qAllocLen > 0 ? qAllocLen : baseInfo.querySeqSize);
        if (qUsedLen == 0) {
            return true;
        }
        if (qAllocLen > 0 && qUsedLen < qAllocLen) {
            return true;
        }
        // KV侧: cache_seqlens为每batch实际长度(PA场景), 为0 → 该batch无有效kv, 输出应全0
        int64_t kvLen = hasCacheSeqlens ? cacheSeqlens[bIdx] : static_cast<int64_t>(baseInfo.kvSeqSize);
        if (kvLen == 0) {
            return true;
        }
        // CAUSAL下单batch q > kv → 产生全mask行, 输出应为0
        if (maskMode_ == 3 && qUsedLen > kvLen) {
            return true;
        }
    }
    return false;
}

bool QuantFlashMlaWithKvcacheMetadataCpuKernel::GenMetaData(SectionStreamKResult& splitRes)
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
static const char* kernelType = "QuantFlashMlaWithKvcacheMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantFlashMlaWithKvcacheMetadataCpuKernel);
} // namespace

} // namespace aicpu
