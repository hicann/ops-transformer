/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_lightning_indexer_v2_metadata_aicpu.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "status.h"

using namespace optiling;

namespace aicpu {
namespace {
// fast负载差分数组的容量上限，不依赖metadata的动态存储布局
constexpr uint32_t A5_FAST_MAX_AIC_CORE_NUM = 80U;
constexpr uint32_t A5_FAST_QUERY_TILE_SIZE = 16U;
constexpr uint32_t A5_FAST_KEY_TILE_SIZE = 2048U;
constexpr uint32_t A5_FAST_MIN_KEY_SIZE = 16384U;
constexpr uint64_t A5_FAST_MIN_TOTAL_QUERY_SIZE = 1024U;
} // namespace
uint32_t QuantLightningIndexerV2MetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (IsFastTemplate()) {
        SplitResult splitRes{aicCoreNum_, aivCoreNum_};
        success = FastSplit(splitRes) && GenMetadata(splitRes);
        return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantLightningIndexerV2MetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    cuSeqlensK_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenK));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    sequsedK_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedK));
    cmpResidualK_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualK));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs =
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
        GetAttrValue(ctx, "soc_version", socVersion_) && GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
        GetAttrValue(ctx, "num_heads_k", numHeadsK_) && GetAttrValue(ctx, "head_dim", headDim_) &&
        GetAttrValue(ctx, "topk", topk_) && GetAttrValue(ctx, "quant_mode", quantMode_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

bool QuantLightningIndexerV2MetadataCpuKernel::ParamsCheck()
{
    if (metadata_ == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    } else if (metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata data is nullptr");
        return false;
    }
    const int32_t batchSize = GetQueryBatchSize();
    return CheckCuSeqlensQ(batchSize) && CheckCuSeqlensK(batchSize) && CheckSequsedQ(batchSize) &&
           CheckSequsedK(batchSize) && CheckCmpResidualK(batchSize);
}

bool QuantLightningIndexerV2MetadataCpuKernel::CheckCuSeqlensQ(int32_t batchSize) const
{
    if (layoutQ_ != "TND" || cuSeqlensQ_ == nullptr || cuSeqlensQ_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cuSeqlensQPtr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
    if (cuSeqlensQPtr[0] != 0) {
        KERNEL_LOG_ERROR("The first element of cu_seqlens_q should be 0, but got %d", cuSeqlensQPtr[0]);
        return false;
    }
    for (int32_t i = 1; i < batchSize + 1; ++i) {
        if (cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
            KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                             "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
                             i - 1, cuSeqlensQPtr[i - 1], i, cuSeqlensQPtr[i]);
            return false;
        }
    }
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::CheckCuSeqlensK(int32_t batchSize) const
{
    if (layoutK_ != "TND" || cuSeqlensK_ == nullptr || cuSeqlensK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cuSeqlensKPtr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
    if (cuSeqlensKPtr[0] != 0) {
        KERNEL_LOG_ERROR("The first element of cu_seqlens_k should be 0, but got %d", cuSeqlensKPtr[0]);
        return false;
    }
    for (int32_t i = 1; i < batchSize + 1; ++i) {
        if (cuSeqlensKPtr[i - 1] > cuSeqlensKPtr[i]) {
            KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                             "but got cu_seqlens_k[%d] = %d, cu_seqlens_k[%d] = %d",
                             i - 1, cuSeqlensKPtr[i - 1], i, cuSeqlensKPtr[i]);
            return false;
        }
    }
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::CheckSequsedQ(int32_t batchSize) const
{
    if (sequsedQ_ == nullptr || sequsedQ_->GetData() == nullptr) {
        return true;
    }
    const int32_t* sequsedQPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
    const int32_t* cuSeqlensQPtr = (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) ?
                                       static_cast<const int32_t*>(cuSeqlensQ_->GetData()) :
                                       nullptr;
    for (int32_t i = 0; i < batchSize; ++i) {
        if (sequsedQPtr[i] < 0) {
            KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i, sequsedQPtr[i]);
            return false;
        }
        if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
            KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than max_seqlen_q %d, "
                             "but got seqused_q[%d] = %d",
                             maxSeqlenQ_, i, sequsedQPtr[i]);
            return false;
        }
        if (cuSeqlensQPtr != nullptr) {
            const int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
            if (sequsedQPtr[i] > seqLen) {
                KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than the sequence length "
                                 "from cu_seqlens_q %d, but got seqused_q[%d] = %d",
                                 seqLen, i, sequsedQPtr[i]);
                return false;
            }
        }
    }
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::CheckSequsedK(int32_t batchSize) const
{
    if (sequsedK_ == nullptr || sequsedK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* sequsedKPtr = static_cast<const int32_t*>(sequsedK_->GetData());
    const int32_t* cuSeqlensKPtr = (layoutK_ == "TND" && cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) ?
                                       static_cast<const int32_t*>(cuSeqlensK_->GetData()) :
                                       nullptr;
    for (int32_t i = 0; i < batchSize; ++i) {
        if (sequsedKPtr[i] < 0) {
            KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d", i, sequsedKPtr[i]);
            return false;
        }
        if (layoutK_ == "BSND" && sequsedKPtr[i] > maxSeqlenK_) {
            KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than max_seqlen_k %d, "
                             "but got seqused_k[%d] = %d",
                             maxSeqlenK_, i, sequsedKPtr[i]);
            return false;
        }
        if (cuSeqlensKPtr != nullptr) {
            const int32_t seqLen = cuSeqlensKPtr[i + 1] - cuSeqlensKPtr[i];
            if (sequsedKPtr[i] > seqLen) {
                KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than the sequence length "
                                 "from cu_seqlens_k %d, but got seqused_k[%d] = %d",
                                 seqLen, i, sequsedKPtr[i]);
                return false;
            }
        }
    }
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::CheckCmpResidualK(int32_t batchSize) const
{
    if (cmpResidualK_ == nullptr || cmpResidualK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cmpResidualKPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
    for (int32_t i = 0; i < batchSize; ++i) {
        if (cmpResidualKPtr[i] < 0 || cmpResidualKPtr[i] >= cmpRatio_) {
            KERNEL_LOG_ERROR("The elements in cmp_residual_k should be in [0, cmpRatio_(%d)), but got "
                             "cmp_residual_k[%d] = %d",
                             cmpRatio_, i, cmpResidualKPtr[i]);
            return false;
        }
    }
    return true;
}

int32_t QuantLightningIndexerV2MetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_传了，使用sequsedQ_获取BatchSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取BatchSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用batchSize_
    return batchSize_;
}

ValidSocVersion QuantLightningIndexerV2MetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    const std::string ascend960 = "Ascend960";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else if (socVersion_.find(ascend960) != std::string::npos) {
        return ValidSocVersion::ASCEND960;
    } else {
        return ValidSocVersion::ASCEND910B;
    }
}

bool QuantLightningIndexerV2MetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == SparseMode::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == SparseMode::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = static_cast<uint32_t>(numHeadsQ_) / static_cast<uint32_t>(numHeadsK_);
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910B) {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 2048U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (static_cast<uint32_t>(topk_) > TOPK_6K) {
            s1BaseSize_ = S1_BASE_SIZE_SMALL;
        }
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    }
    // fast 模板：每个 S1 tile 覆盖的 token 数 = QLI_FAST_S1_TILE_ROWS / gSize
    // （gSize=24 -> 16，gSize=32 -> 12），与 fast kernel KernelConfig::S1_TILE 一致
    if (IsFastTemplate()) {
        fastS1Tile_ = QLI_FAST_S1_TILE_ROWS / groupSize_;
    }
    return true;
}

uint32_t QuantLightningIndexerV2MetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 sequsedQ_ 传了，直接使用
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 querySeqSize_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t QuantLightningIndexerV2MetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 如果 seqUsedKv_ 传了，直接使用
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // seqUsedKv_ 没传，判断 Layout
    if (layoutK_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenOriKv_
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t* s2Ptr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 使用 max_seqlen_k
    return static_cast<uint32_t>(maxSeqlenK_);
}

uint64_t QuantLightningIndexerV2MetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetS2SeqSize(bIdx);
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t* residualPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) +
               static_cast<uint64_t>(residualPtr[bIdx]);
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsA5FastInputSupported() const
{
    const auto present = [](const Tensor* t) { return t != nullptr && t->GetData() != nullptr; };
    if ((maskMode_ != static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
         maskMode_ != static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) ||
        cmpRatio_ != 1 || layoutQ_ != "TND" || (layoutK_ != "TND" && layoutK_ != "PA_BBND") || batchSize_ <= 0 ||
        !present(cuSeqlensQ_) || present(sequsedQ_) || aicCoreNum_ == 0U || aicCoreNum_ > A5_FAST_MAX_AIC_CORE_NUM) {
        return false;
    }
    return layoutK_ == "TND" ? (present(cuSeqlensK_) && !present(sequsedK_)) : present(sequsedK_);
}

bool QuantLightningIndexerV2MetadataCpuKernel::CollectA5FastWorkload(A5FastWorkload& workload)
{
    // 统计实际工作量及16Q/2048K任务数，保留Q/K尾块对利用率的影响
    workload.firstKeyTiles = (GetS2SeqSize(0U) + A5_FAST_KEY_TILE_SIZE - 1U) / A5_FAST_KEY_TILE_SIZE;
    for (uint32_t b = 0U; b < static_cast<uint32_t>(batchSize_); ++b) {
        const uint32_t q = GetS1SeqSize(b), k = GetS2SeqSize(b);
        if (q == 0U || k < A5_FAST_MIN_KEY_SIZE ||
            static_cast<uint64_t>(k) < static_cast<uint64_t>(q) + A5_FAST_KEY_TILE_SIZE) {
            return false;
        }
        const uint32_t keyTiles = (k + A5_FAST_KEY_TILE_SIZE - 1U) / A5_FAST_KEY_TILE_SIZE;
        workload.totalQ += q;
        workload.totalK += k;
        workload.totalTasks += (q + A5_FAST_QUERY_TILE_SIZE - 1U) / A5_FAST_QUERY_TILE_SIZE;
        workload.minK = std::min(workload.minK, k);
        workload.uniformKeyTiles = workload.uniformKeyTiles && (keyTiles == workload.firstKeyTiles);
        workload.usefulQueryKeyRows += static_cast<uint64_t>(q) * k;
    }
    return workload.totalQ >= A5_FAST_MIN_TOTAL_QUERY_SIZE;
}

uint64_t QuantLightningIndexerV2MetadataCpuKernel::CalcA5FastMaxCoreKeyTiles(const A5FastWorkload& workload)
{
    // K块数相同时，最重核负载可直接由全局任务数推导；Q/K尾块损耗留到后续利用率比较中计入
    uint64_t maxCoreKeyTiles = ((workload.totalTasks + aicCoreNum_ - 1U) / aicCoreNum_) * workload.firstKeyTiles;
    if (!workload.uniformKeyTiles) {
        // K块数不同时，将各batch拆为完整轮次和环形余数区间，用差分数组计算最重核负载
        std::array<int64_t, A5_FAST_MAX_AIC_CORE_NUM + 1U> loadDelta{};
        uint64_t fullRoundLoad = 0U;
        int64_t taskOffset = 0;
        for (uint32_t b = 0U; b < static_cast<uint32_t>(batchSize_); ++b) {
            const uint32_t tasks = (GetS1SeqSize(b) + A5_FAST_QUERY_TILE_SIZE - 1U) / A5_FAST_QUERY_TILE_SIZE;
            const uint32_t keyTiles = (GetS2SeqSize(b) + A5_FAST_KEY_TILE_SIZE - 1U) / A5_FAST_KEY_TILE_SIZE;
            fullRoundLoad += static_cast<uint64_t>(tasks / aicCoreNum_) * keyTiles;
            const int64_t end = taskOffset + tasks % aicCoreNum_;
            loadDelta[taskOffset] += keyTiles;
            loadDelta[std::min(end, static_cast<int64_t>(aicCoreNum_))] -= keyTiles;
            if (end > aicCoreNum_) {
                loadDelta[0] += keyTiles;
                loadDelta[end - aicCoreNum_] -= keyTiles;
            }
            taskOffset = (end >= aicCoreNum_) ? end - aicCoreNum_ : end;
        }
        int64_t remainderLoad = 0;
        maxCoreKeyTiles = 0U;
        for (uint32_t core = 0U; core < aicCoreNum_; ++core) {
            remainderLoad += loadDelta[core];
            maxCoreKeyTiles = std::max(maxCoreKeyTiles, fullRoundLoad + static_cast<uint64_t>(remainderLoad));
        }
    }

    return maxCoreKeyTiles;
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsA5FastWorkloadEligible(const A5FastWorkload& workload)
{
    // 短K通过平均Q摊薄初始化和最终筛选成本，分页场景从32K起准入
    // 分页FP8的scale预检还需按K加权约束Q；metadata无blockSize信息，按最小合法页16保守估计
    const bool fp8 = quantMode_ == static_cast<int32_t>(QUANT_MODE_FP8);
    // 短FP8筛选指K小于64K且TopK小于2048；K小于32K时平均Q至少128，否则短筛选至少32
    const bool shortFp8Selection = fp8 && (workload.minK < 65536U) && (topk_ < 2048);
    const uint64_t minAverageQuery = (workload.minK < 32768U) ? 128U : (shortFp8Selection ? 32U : 0U);
    // 分页FP8至少保留256行的K加权Query预算，用于摊薄scale预检成本
    constexpr uint64_t PAGE_FP8_SCALE_QUERY_BUDGET = 256U;
    const bool pagedFp8 = fp8 && (layoutK_ == "PA_BBND");
    const uint64_t minWeightedQuery = pagedFp8 ? PAGE_FP8_SCALE_QUERY_BUDGET : (shortFp8Selection ? 32U : 0U);
    if ((workload.minK < 32768U && layoutK_ != "TND") ||
        workload.totalQ < static_cast<uint64_t>(batchSize_) * minAverageQuery ||
        workload.usefulQueryKeyRows < workload.totalK * minWeightedQuery) {
        return false;
    }

    return IsA5FastUtilizationSufficient(workload, CalcA5FastMaxCoreKeyTiles(workload), pagedFp8);
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsA5FastUtilizationSufficient(const A5FastWorkload& workload,
                                                                             uint64_t maxCoreKeyTiles,
                                                                             bool pagedFp8) const
{
    // 按最短K查表选择利用率门槛，16K档要求满载；使用整数比例比较，允许恰好等于门槛的场景准入
    const uint32_t tier =
        static_cast<uint32_t>(workload.minK >= 32768U) + static_cast<uint32_t>(workload.minK >= 65536U) +
        static_cast<uint32_t>(workload.minK >= 131072U) + static_cast<uint32_t>(workload.minK >= 262144U) +
        static_cast<uint32_t>(workload.minK >= 393216U);
    // 表的两行依次为TND、PA_BBND；六列依次为K的[16K,32K)、[32K,64K)、
    // [64K,128K)、[128K,256K)、[256K,384K)、[384K,+∞)档，按分子/分母表示利用率门槛
    // TND门槛依次为100%、82%、74%、72%、70%、2/3；PA依次为100%、85%、78%、75%、73%、70%
    static constexpr uint64_t NUMERATORS[2][6] = {{100U, 82U, 74U, 72U, 70U, 2U}, {100U, 85U, 78U, 75U, 73U, 70U}};
    static constexpr uint64_t DENOMINATORS[2][6] = {{100U, 100U, 100U, 100U, 100U, 3U},
                                                    {100U, 100U, 100U, 100U, 100U, 100U}};
    const uint32_t paged = static_cast<uint32_t>(layoutK_ == "PA_BBND");
    const uint64_t thresholdNumerator = NUMERATORS[paged][tier];
    const uint64_t thresholdDenominator = DENOMINATORS[paged][tier];
    const uint64_t usefulQueryKeyTiles = workload.usefulQueryKeyRows / A5_FAST_KEY_TILE_SIZE;
    const uint64_t scheduledQueryKeyTiles = maxCoreKeyTiles * A5_FAST_QUERY_TILE_SIZE * aicCoreNum_;
    // 按8头补齐的比例提高短K门槛；K>=256K沿用已验证可覆盖补齐成本的门槛
    // 普通档门槛最高截为满载，16K且TopK<2048时保留超出满载的门槛，避免误放行
    // 7是8头向上对齐的偏置；tier从0计数，tier>=4对应K>=256K
    const uint64_t physicalHeads = (static_cast<uint64_t>(numHeadsQ_) + 7U) / 8U * 8U;
    const uint64_t referenceHeads =
        (tier >= 4U) ? physicalHeads : std::max(static_cast<uint64_t>(8U), static_cast<uint64_t>(numHeadsQ_));
    const uint64_t headDenominator = thresholdDenominator * referenceHeads;
    const uint64_t paddedNumerator = thresholdNumerator * physicalHeads;
    const uint64_t headNumerator =
        ((tier == 0U) && (topk_ < 2048)) ? paddedNumerator : std::min(paddedNumerator, headDenominator);
    // 为超出单batch、Q=1024性能参照的分页FP8预检预留额外预算，额外K行按1/8个Q-K块折算
    // 先比较主流水预算，再以除法检查剩余裕量，避免额外成本交叉相乘放大中间值
    const uint64_t referenceKeyRows = workload.usefulQueryKeyRows / A5_FAST_MIN_TOTAL_QUERY_SIZE;
    const uint64_t extraScaleKeyRows = pagedFp8 ? workload.totalK - std::min(workload.totalK, referenceKeyRows) : 0U;
    const uint64_t scaleCheckBudget = (extraScaleKeyRows + 7U) / 8U;
    const uint64_t usefulBudget = usefulQueryKeyTiles * headDenominator;
    const uint64_t scheduledBudget = scheduledQueryKeyTiles * headNumerator;
    return (usefulBudget >= scheduledBudget) &&
           (((usefulBudget - scheduledBudget) / headDenominator) >= scaleCheckBudget);
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsA5FastEligible()
{
    // 静态范围、输入组合、工作量统计及收益判断分层执行，未授权时沿用原模板
    constexpr int32_t MAX_HEAD_NUM = 32;
    constexpr int32_t HEAD_DIM = 128;
    if (ProcessSocVersion() != ValidSocVersion::ASCEND950 ||
        (quantMode_ != static_cast<int32_t>(QUANT_MODE_FP8) && quantMode_ != static_cast<int32_t>(QUANT_MODE_MXFP8)) ||
        numHeadsK_ != 1 || numHeadsQ_ < 1 || numHeadsQ_ > MAX_HEAD_NUM || headDim_ != HEAD_DIM || topk_ < 1 ||
        topk_ > static_cast<int32_t>(QLI_V2_A5_FAST_SPARSE_CAPACITY) || !IsA5FastInputSupported()) {
        return false;
    }
    A5FastWorkload workload;
    return CollectA5FastWorkload(workload) && IsA5FastWorkloadEligible(workload);
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcSplitInfo(SplitContext& splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo& splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        maxS2Size_ = std::max(maxS2Size_, s2Size);
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND950 || validSocVersion == ValidSocVersion::ASCEND960) {
        if (maxS2Size_ < static_cast<uint32_t>(topk_)) {
            supportFd_ = false;
            return;
        }
        if (maxS2Size_ > fdToleranceRatio * s2BaseSize_) {
            supportFd_ = true;
            return;
        }
    }
}

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size) const
{
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size) const
{
    auto mode = static_cast<SparseMode>(maskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcCost(uint32_t basicM, uint32_t basicS2) const
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(COST_WEIGHT_M * alignBasicM + COST_WEIGHT_S2 * alignBasicS2);
}

BlockCost<int64_t> QuantLightningIndexerV2MetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                                           uint32_t s1GTailSize,
                                                                           uint32_t s2TailSize) const
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<int64_t> QuantLightningIndexerV2MetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache& batchCache)
{
    // no mask
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.revertS2Size));
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in two or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    return std::make_pair(s2FirstToken, s2LastToken);
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext& splitContext,
                                                              BatchCache& batchCache)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.revertS2Size = GetRevertS2Size(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext& splitContext,
                                                            const BatchCache& batchCache, S1GCache& s1GCache)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0 || splitInfo.s2BaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    auto s2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    int64_t s2FirstToken = s2TokenRange.first;
    int64_t s2LastToken = s2TokenRange.second;
    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));

    // get block start & end
    int64_t s2CmpLength = (s2LastToken - s2FirstToken + 1) / cmpRatio_;
    if (s2CmpLength <= 0) {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = 0U;
    } else {
        s1GCache.s2Start = 0U;
        // end of block index, Right-open interval
        s1GCache.s2End = (static_cast<uint32_t>(s2CmpLength) + s2BaseSize_ - 1) / s2BaseSize_;
    }

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    uint32_t s2TailSize = static_cast<uint32_t>(s2CmpLength) % s2BaseSize_;

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = s2TailSize != 0 ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;

    BlockCost<int64_t> typeCost =
        CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[batchCache.bIdx], s2TailSize);

    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost =
            typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[TAIL_BLOCK][TAIL_BLOCK] : typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost =
            typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[NORMAL_BLOCK][TAIL_BLOCK] : typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext& splitContext,
                                                             CostInfo& costInfo)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        // 更新最大S1G行开销
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcCostInfo(SplitContext& splitContext)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    CostInfo& costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsK_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * static_cast<uint32_t>(numHeadsK_);
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::UpdateCursor(const SplitContext& splitContext,
                                                            AssignContext& assignContext)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    const CostInfo& costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx ==
        static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_)) { // 所有负载全部分配完
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_) != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_);
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::AssignByBatch(const SplitContext& splitContext,
                                                             AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo& costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_)) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_) != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_);
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::AssignByRow(const SplitContext& splitContext,
                                                           AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::AssignByBlock(const SplitContext& splitContext,
                                                             AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::ForceAssign(const SplitContext& splitContext,
                                                           AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext& assignContext,
                                                                  const SplitResult& splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void QuantLightningIndexerV2MetadataCpuKernel::RecordFDInfo(const SplitContext& splitContext,
                                                            const AssignContext& assignContext, SplitResult& result)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / static_cast<uint32_t>(numHeadsK_);
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize =
        (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
            (static_cast<uint64_t>(s1Size) * groupSize_ - static_cast<uint64_t>(splitS1GIdx) * mBaseSize_) :
            mBaseSize_;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize / groupSize_;
    result.numOfFdHead++;
}

void QuantLightningIndexerV2MetadataCpuKernel::AssignBlockToCore(const SplitContext& splitContext,
                                                                 AssignContext& assignContext, SplitResult& result)
{
    const CostInfo& costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    assignContext.coreCache = {};
    assignContext.coreCache.costLimit = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    if (!supportFd_) {
        assignContext.coreCache.costLimit = costInfo.maxS1GCost > assignContext.coreCache.costLimit ?
                                                costInfo.maxS1GCost :
                                                assignContext.coreCache.costLimit;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    if (supportFd_) {
        AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    // 对之前的归约信息进行记录并清理
    if (supportFd_ && IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    // 更新S2切分信息
    if (supportFd_ && assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext& splitContext,
                                                             SplitResult& result)
{
    const CostInfo& costInfo = splitContext.costInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlockToCore(splitContext, assignContext, result);
        // Keep trailing zero-cost batches in the last AIC range so their outputs are initialized.
        if (!assignContext.isFinished && assignContext.unassignedCost <= 0) {
            result.bN2End[i] = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_);
            result.gS1End[i] = 0U;
            result.s2End[i] = 0U;
            assignContext.isFinished = true;
        }
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void QuantLightningIndexerV2MetadataCpuKernel::SplitFD(SplitResult& splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    // 向上取整，避免核负载为0
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 冗余vec核数为0，此时规约任务无法进行更小的切分，只能1个vec核计算1个规约任务
        if (emptyVectorNum == 0U) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
            splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
            curCoreIndex++;
            continue;
        }
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        // 需要使用的vec核数与当前剩余可用vec核数取最小
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        // FD负载分配
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
        // 更新冗余vec核数
        emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool QuantLightningIndexerV2MetadataCpuKernel::BalanceSchedule(SplitResult& splitRes)
{
    SplitContext splitContext(batchSize_);
    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_);
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (supportFd_ && splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::IsFastTemplate()
{
    // fast 模板仅存在于 arch22。AICPU 侧无 PlatformAscendC，只能用 soc_version attr
    // 判定：ASCEND910B 分支表示"非 Ascend950/960"，覆盖现有 arch22 芯片（910B/910_93）
    if (ProcessSocVersion() != ValidSocVersion::ASCEND910B) {
        return false;
    }
    // 此处只判 gSize/sparse_mode/cmp_ratio；layout/dtype/return_value 等其余条件由
    // 主算子 host tiling（IsFastTemplateEligible）统一校验
    const int32_t gSize = (numHeadsK_ == 0) ? 0 : numHeadsQ_ / numHeadsK_;
    return (maskMode_ == QLI_FAST_SPARSE_MODE_LOWER) && (cmpRatio_ == 1) &&
           ((gSize == QLI_FAST_GSIZE_24) || (gSize == QLI_FAST_GSIZE_32));
}

uint32_t QuantLightningIndexerV2MetadataCpuKernel::FastGetS1SeqSize(uint32_t bIdx) const
{
    // fast 模板 layout_q 恒为 TND，序列长度取 cu_seqlens_q 差分；cu 序列不保证升序，
    // 差分为负时按 0 处理，避免转无符号后放大（与 fast kernel 侧物理截断行为一致）
    if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
        const int32_t diff = s1Ptr[bIdx + 1U] - s1Ptr[bIdx];
        return diff > 0 ? static_cast<uint32_t>(diff) : 0U;
    }
    return static_cast<uint32_t>(maxSeqlenQ_);
}

bool QuantLightningIndexerV2MetadataCpuKernel::FastSplit(SplitResult& splitRes)
{
    // 分核算法与 fast kernel 原 device 侧均分逻辑一致：
    // 工作单元是 S1 tile（QLI_FAST_S1_TILE_ROWS / gSize 个 token），tile 总数按每个
    // batch 的 S1 长度独立向上取整后累加（不剔除 S2 为 0 的 batch，与 device 侧的
    // tile↔batch 映射保持一致），再均分到 min(totalTiles, aicCoreNum) 个 Mix group
    if (aicCoreNum_ == 0U || fastS1Tile_ == 0U) {
        return false;
    }
    uint64_t totalTileCount = 0;
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        const uint32_t s1Size = FastGetS1SeqSize(bIdx);
        totalTileCount += (static_cast<uint64_t>(s1Size) + fastS1Tile_ - 1U) / fastS1Tile_;
    }
    if (totalTileCount == 0U) {
        // 全空 case：仅 0 号核空转，与慢模板 isKvSeqAllZero 的处理一致
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = 0U;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }

    const uint32_t activeGroups = static_cast<uint32_t>(std::min<uint64_t>(totalTileCount, aicCoreNum_));
    const uint64_t baseTileCount = totalTileCount / activeGroups;
    const uint64_t remainder = totalTileCount % activeGroups;
    uint64_t cursor = 0;
    for (uint32_t coreIdx = 0; coreIdx < activeGroups; coreIdx++) {
        const uint64_t begin = cursor;
        const uint64_t end = begin + baseTileCount + (coreIdx < remainder ? 1U : 0U);
        cursor = end;
        // gS1End 承载 tile 前缀端点（左闭右开），GenMetadata 据此写入扩展区每 group
        // 的 [begin, end)；bN2End/s2End 仅按通用 SplitResult 结构填充，fast 路径不消费
        splitRes.bN2End[coreIdx] = 1U;
        // tile 端点不超过 cu_seqlens_q 累计值（int32 域），收窄安全
        splitRes.gS1End[coreIdx] = static_cast<uint32_t>(end);
        splitRes.s2End[coreIdx] = 1U;
    }
    splitRes.usedCoreNum = activeGroups;
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::GenMetadata(SplitResult& splitRes)
{
    const uint64_t metadataSize = GetQliV2MetadataSize(aicCoreNum_, aivCoreNum_);
    if ((aicCoreNum_ == 0) || (aivCoreNum_ == 0) || (metadata_->NumElements() < metadataSize)) {
        return false;
    }
    auto* metadataPtr = static_cast<uint32_t*>(metadata_->GetData());
    std::fill_n(metadataPtr, metadataSize, 0U);
    metadataPtr[QLI_V2_METADATA_AIC_NUM_INDEX] = aicCoreNum_;
    metadataPtr[QLI_V2_METADATA_AIV_NUM_INDEX] = aivCoreNum_;
    auto* extensionMetadata = metadataPtr + GetQliV2MetadataExtensionOffset(aicCoreNum_, aivCoreNum_);

    if (IsFastTemplate()) {
        // arch22 fast 分核数据写入扩展区（参考 A5：fast 专属信息不再占用分核记录区，
        // 记录区仅承载 slow 分核）。3 槽连续紧凑布局：每 group 依次为 enable、tile 起点、
        // tile 终点（左闭右开）；0 号 group 从 0 开始，后续 group 取前一 group 的 END。
        // 每 group 占 3 槽，fastGroups 已按扩展区容量截断（256 int32 最多 85 group）；
        // 未启用 group 保持 fill_n 的 0（disable）
        constexpr uint32_t maxFastGroups = QLI_V2_METADATA_EXTENSION_SIZE / QLI_V2_A3_FAST_GROUP_SLOT_NUM;
        const uint32_t fastGroups = std::min<uint32_t>(splitRes.usedCoreNum, maxFastGroups);
        for (size_t i = 0; i < fastGroups; ++i) {
            const size_t groupBase = i * QLI_V2_A3_FAST_GROUP_SLOT_NUM;
            extensionMetadata[groupBase + QLI_V2_A3_FAST_ENABLE_SLOT] = 1; // group enable
            extensionMetadata[groupBase + QLI_V2_A3_FAST_START_SLOT] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            extensionMetadata[groupBase + QLI_V2_A3_FAST_END_SLOT] = splitRes.gS1End[i];
        }
        return true;
    }

    // LI Metadata Generate（slow：标准分核记录区）
    auto* qliMetadata = metadataPtr + QLI_V2_METADATA_HEADER_SIZE;
    auto* qldMetadata = qliMetadata + static_cast<uint64_t>(aicCoreNum_) * QLI_V2_METADATA_SIZE;
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
        // FA END
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_BN2_END_INDEX] = splitRes.bN2End[i];
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_M_END_INDEX] = splitRes.gS1End[i];
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_S2_END_INDEX] = splitRes.s2End[i];
        qliMetadata[i * QLI_V2_METADATA_SIZE + QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // LD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        qldMetadata[i * QLD_V2_METADATA_SIZE + QLD_V2_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    // 先生成完整的原模板调度信息再写入准入标志，供kernel检查失败时回退使用
    if (ProcessSocVersion() == ValidSocVersion::ASCEND950) {
        extensionMetadata[QLI_V2_A5_FAST_FLAG_OFFSET] =
            IsA5FastEligible() ? QLI_V2_A5_FAST_ENABLE : QLI_V2_A5_FAST_DISABLE;
    }
    return true;
}

namespace {
static const char* kernelType = "QuantLightningIndexerV2Metadata";
REGISTER_CPU_KERNEL(kernelType, QuantLightningIndexerV2MetadataCpuKernel);
} // namespace
}; // namespace aicpu
