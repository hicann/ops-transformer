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
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cmath>
#include "quant_flash_attn_metadata_aicpu.h"
#include "../../quant_flash_attn/op_host/qfa_adjust_sinner_souter.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantFlashAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SectionStreamKResult splitRes;
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    if (success && isGradEnabled_) {
        success = GenQuantFagGradMetaData();
    }
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantFlashAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    for (Tensor** slot : {&sequsedQ_, &sequsedKv_}) {
        Tensor* tensor = *slot;
        if (tensor == nullptr) {
            continue;
        }
        auto shape = tensor->GetTensorShape();
        bool empty = tensor->GetData() == nullptr;
        if (shape != nullptr) {
            for (int32_t axis = 0; axis < shape->GetDims(); ++axis) {
                empty = empty || shape->GetDimSize(axis) == 0;
            }
        }
        if (empty) {
            *slot = nullptr;
        }
    }
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
    GetAttrValueOpt(ctx, "metadata_dim_num", metadataDimNum_);
    GetAttrValueOpt(ctx, "metadata_row_size", metadataRowSize_);
    return ParamsInit();
}

std::vector<int64_t> QuantFlashAttnMetadataCpuKernel::GetTensorDataAsInt64(Tensor* tensor, size_t size)
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

bool QuantFlashAttnMetadataCpuKernel::ParamsInit()
{
    if (isGradEnabled_ && (layoutQ_ == "TND" || layoutKv_ == "TND")) {
        if (layoutQ_ != "TND" || layoutKv_ != "TND" || cuSeqlensQ_ == nullptr || cuSeqlensKv_ == nullptr ||
            cuSeqlensQ_->GetData() == nullptr || cuSeqlensKv_->GetData() == nullptr) {
            KERNEL_LOG_ERROR("QuantFAG TND requires both cumulative sequence tensors");
            return false;
        }
        if (cuSeqlensQ_->GetDataType() != DT_INT32 || cuSeqlensKv_->GetDataType() != DT_INT32 ||
            cuSeqlensQ_->GetTensorShape() == nullptr || cuSeqlensKv_->GetTensorShape() == nullptr ||
            cuSeqlensQ_->GetTensorShape()->GetDims() != 1 || cuSeqlensKv_->GetTensorShape()->GetDims() != 1) {
            KERNEL_LOG_ERROR("QuantFAG TND cu_seqlens must be rank 1 int32 tensors");
            return false;
        }
        const int64_t count = cuSeqlensQ_->GetTensorShape()->GetDimSize(0);
        if (count < 2 || cuSeqlensKv_->GetTensorShape()->GetDimSize(0) != count ||
            (batchSize_ > 0 && batchSize_ != count - 1)) {
            KERNEL_LOG_ERROR("QuantFAG TND cumulative sequence shape/batch mismatch");
            return false;
        }
        // Missing optional inputs may be represented by Tensor objects with null data.
        for (auto used : {sequsedQ_, sequsedKv_}) {
            if (used != nullptr && used->GetData() != nullptr &&
                (used->GetDataType() != DT_INT32 || used->GetTensorShape() == nullptr ||
                 used->GetTensorShape()->GetDims() != 1 || used->GetTensorShape()->GetDimSize(0) != count - 1)) {
                auto shape = used->GetTensorShape();
                KERNEL_LOG_ERROR(
                    "QuantFAG TND seqused_%s invalid: dtype=%d rank=%ld dim0=%ld expected_B=%ld data_present=%d",
                    used == sequsedQ_ ? "q" : "kv", static_cast<int>(used->GetDataType()),
                    shape == nullptr ? -1L : static_cast<int64_t>(shape->GetDims()),
                    shape == nullptr || shape->GetDims() == 0 ? -1L : shape->GetDimSize(0), count - 1,
                    used->GetData() != nullptr ? 1 : 0);
                return false;
            }
        }
        auto qSeq = GetTensorDataAsInt64(cuSeqlensQ_, count);
        auto kvSeq = GetTensorDataAsInt64(cuSeqlensKv_, count);
        if (qSeq[0] != 0 || kvSeq[0] != 0) {
            KERNEL_LOG_ERROR("QuantFAG TND cumulative sequences must start at zero");
            return false;
        }
        for (int64_t i = 1; i < count; ++i) {
            if (qSeq[i] < qSeq[i - 1] || kvSeq[i] < kvSeq[i - 1]) {
                KERNEL_LOG_ERROR("QuantFAG TND cumulative sequences must be nondecreasing");
                return false;
            }
        }
        for (int64_t i = 0; i < count - 1; ++i) {
            const int64_t qUsed = sequsedQ_ == nullptr || sequsedQ_->GetData() == nullptr ?
                                      qSeq[i + 1] - qSeq[i] :
                                      static_cast<const int32_t*>(sequsedQ_->GetData())[i];
            const int64_t kvUsed = sequsedKv_ == nullptr || sequsedKv_->GetData() == nullptr ?
                                       kvSeq[i + 1] - kvSeq[i] :
                                       static_cast<const int32_t*>(sequsedKv_->GetData())[i];
            if (qUsed < 0 || qUsed > qSeq[i + 1] - qSeq[i] || kvUsed < 0 || kvUsed > kvSeq[i + 1] - kvSeq[i]) {
                KERNEL_LOG_ERROR("QuantFAG TND seqused must be nonnegative and within each cu_seqlens segment");
                return false;
            }
        }
    }

    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
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
        baseInfo.queryType = load_balance::DataType::FP8_E4M3FN;
        baseInfo.kvType = load_balance::DataType::FP8_E4M3FN;
    }
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    if (socVersion_.find("Ascend960") != std::string::npos) {
        // 基本块随 decode/prefill 与 vHeadDim 分级，按 vHeadDim 判档
        optiling::quant_flash_attn::qfa_tiling_util::AdjustSinnerAndSouterArch92(
            static_cast<uint32_t>(headDimV_), static_cast<int64_t>(maxSeqlenQ_), static_cast<int64_t>(maxSeqlenKv_),
            maskMode_, static_cast<int64_t>(winLeft_), static_cast<int64_t>(winRight_),
            optiling::quant_flash_attn::qfa_tiling_util::LAYOUT_BSND, static_cast<uint32_t>(quantMode_), isDecode,
            sOuterFactor, sInnerFactor);
    } else {
        // 保持原有逻辑与入参不变
        optiling::quant_flash_attn::qfa_tiling_util::AdjustSinnerAndSouter(
            static_cast<uint32_t>(headDim_), static_cast<int64_t>(maxSeqlenQ_), static_cast<int64_t>(maxSeqlenKv_),
            maskMode_, static_cast<int64_t>(winLeft_), static_cast<int64_t>(winRight_),
            optiling::quant_flash_attn::qfa_tiling_util::LAYOUT_BSND, static_cast<uint32_t>(quantMode_), sOuterFactor,
            sInnerFactor);
    }
    mBaseSize_ = sOuterFactor;
    s2BaseSize_ = sInnerFactor;
    mBaseSize_ = mBaseSize_ * (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    if (quantMode_ == 1 &&
        socVersion_.find("Ascend960") == std::string::npos) { // 仅 MXFP8 在支持 FlashDecode 的平台开启 FD
        param.l2Byte = 96U * 1024U * 1024U;                   // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb
        param.fdTolerance = 10;                               // 10: tolerance block
        param.fdLeastBlock = 3;                               // 3: least block
        param.fdOn = true;
    } else {
        param.l2Byte = 0;
        param.fdTolerance = 300;
        param.fdOn = false;
    }
    param.kernelSplitMode = load_balance::KernelSplitMode::BN2_S1G_S2;

    if (isGradEnabled_) {
        // 等长场景的 deterMaxRound, 由 GenMetaData 在 Clear 之后写入第二行槽位 0。
        // varlen(seqused) 场景该值不适用, 会被 GenQuantFagGradMetaData 覆盖为 roundPrefix[b]。
        fagDeterMaxRound_ = CalDeterMaxRound();
    }
    needInitOutput_ = CheckNeedInitOutput();
    return true;
}

bool QuantFlashAttnMetadataCpuKernel::HasVarlenSeq() const
{
    return (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) ||
           (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) ||
           (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) ||
           (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr);
}

// 确定性计算最大循环次数 (等长场景, 与 quantFAG kernel 的 cal_deter_max_loop_num 保持一致)
int64_t QuantFlashAttnMetadataCpuKernel::CalDeterMaxRound()
{
    s1Size_ = GetS1SeqSize(0);
    s2Size_ = GetS2SeqSize(0);
    // 非TND场景
    int64_t b = batchSize_ * baseInfo.kvHeadNum;
    int64_t m = CeilDiv<int64_t>(s1Size_, 512);
    int64_t n = CeilDiv<int64_t>(s2Size_, 512);
    int64_t k = aicCoreNum_;

    if (m == 0 || n == 0 || b == 0) {
        return 0;
    }
    if (n == 1) {
        return std::max(CeilDiv<int64_t>(m * b, k), m);
    } else {
        return CeilDiv<int64_t>(n * b, std::min(k, m * b)) * m;
    }
}

namespace {
constexpr int64_t MASK_MODE_CAUSAL = 3;
constexpr int64_t MASK_MODE_BAND = 4;
constexpr int64_t TND_LINE_KIND_RIGHT_DOWN = 3;
constexpr int64_t TND_LINE_KIND_BAND = 4;
constexpr int64_t TND_LINE_KIND_BAND_DENSE = 43;
constexpr int64_t TND_LINE_TINY_AREA = 8;
constexpr int64_t TND_LINE_TOKEN_UNLIMITED = 100000000LL;

inline int64_t TndLineCeilDiv(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t GetRightDownRowShift(int64_t m, int64_t n)
{
    return std::max(static_cast<int64_t>(0), m - n - 1);
}

inline void GetRightDownVirt(int64_t m, int64_t n, int64_t& virtM, int64_t& virtN)
{
    m -= GetRightDownRowShift(m, n);
    if (m < 1 || n < 1) {
        virtM = 0;
        virtN = 0;
        return;
    }
    virtM = m;
    virtN = 2 * n - m + 3;
}

inline bool TndLineRunIsWorthy(int64_t k, int64_t virtM, int64_t virtN, int64_t pairCount)
{
    if (virtM <= 0 || virtN <= 0 || virtM * virtN <= TND_LINE_TINY_AREA) {
        return false;
    }
    return pairCount > 0 && virtN * pairCount >= k && virtM >= std::min(k, virtN);
}

inline int64_t CalBandWideCols(int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t nNew = std::min(m - 1 + q, n);
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = nNew - l1 - l2;
    const int64_t foldCols = std::max(static_cast<int64_t>(0), l3 - p + 1);
    return nNew - foldCols;
}

inline int64_t CalBandNarrowRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t l1 = q - 1;
    const int64_t l2 = std::min(n - q + 1, m + 2 - p - q);
    const int64_t l3 = std::max(static_cast<int64_t>(0), std::min(p + n - m - 1, p + q - 2));
    int64_t overlap = 0;
    const int64_t nNew = l1 + l2 + l3;
    if (l3 != 0 && p <= l3) {
        overlap = std::min(l3 - p + 1, l1);
    }
    const int64_t seg = p + q - 1;
    return seg * TndLineCeilDiv((nNew - overlap) * n1 + overlap, k);
}

inline bool TndLineRunEmpty(const TndLineRunShape& shape, bool isBand)
{
    return shape.m < 1 || shape.n < 1 || (isBand && shape.p + shape.q < 2);
}
} // namespace

bool QuantFlashAttnMetadataCpuKernel::IsTndLineBandMode() const
{
    return maskMode_ == MASK_MODE_BAND;
}

void QuantFlashAttnMetadataCpuKernel::CalTndLineActualToken(uint32_t bIdx, int64_t& s1Token, int64_t& s2Token)
{
    if (IsTndLineBandMode()) {
        s1Token = (winLeft_ == -1) ? TND_LINE_TOKEN_UNLIMITED : static_cast<int64_t>(winLeft_);
        s2Token = (winRight_ == -1) ? TND_LINE_TOKEN_UNLIMITED : static_cast<int64_t>(winRight_);
    } else {
        s1Token = TND_LINE_TOKEN_UNLIMITED;
        s2Token = 0;
    }
    const int64_t s1Len = static_cast<int64_t>(GetS1SeqSize(bIdx));
    const int64_t s2Len = static_cast<int64_t>(GetS2SeqSize(bIdx));
    s1Token = s1Token + s1Len - s2Len;
    s2Token = s2Token - s1Len + s2Len;
}

void QuantFlashAttnMetadataCpuKernel::GetTndLineOuterMN(uint32_t bIdx, int64_t& m, int64_t& n)
{
    const int64_t s1Len = static_cast<int64_t>(GetS1SeqSize(bIdx));
    const int64_t s2Len = static_cast<int64_t>(GetS2SeqSize(bIdx));
    if (s1Len <= 0 || s2Len <= 0) {
        m = 0;
        n = 0;
        return;
    }
    m = TndLineCeilDiv(s1Len, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_M));
    n = TndLineCeilDiv(s2Len, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_N));
    m = m < 1 ? 1 : m;
    n = n < 1 ? 1 : n;
}

TndLineRunShape QuantFlashAttnMetadataCpuKernel::MakeTndLineRunShape(uint32_t bIdx)
{
    TndLineRunShape shape;
    GetTndLineOuterMN(bIdx, shape.m, shape.n);
    if (!IsTndLineBandMode()) {
        shape.m -= GetRightDownRowShift(shape.m, shape.n);
        shape.kind = TND_LINE_KIND_RIGHT_DOWN;
        return shape;
    }

    shape.kind = TND_LINE_KIND_BAND;
    if (shape.m < 1 || shape.n < 1) {
        return shape;
    }
    int64_t s1Token = 0;
    int64_t s2Token = 0;
    CalTndLineActualToken(bIdx, s1Token, s2Token);
    shape.p = TndLineCeilDiv(s1Token, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_M)) + 1;
    shape.q = TndLineCeilDiv(s2Token, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_N)) + 1;
    shape.p = shape.p > shape.m ? shape.m : shape.p;
    shape.q = shape.q > shape.n ? shape.n : shape.q;
    if (shape.p < 0) {
        shape.n = shape.n + shape.p;
        shape.q = shape.p + shape.q;
        shape.p = 1;
    } else if (shape.q < 0) {
        shape.m = shape.m + shape.q;
        shape.p = shape.p + shape.q;
        shape.q = 1;
    }
    int64_t actualM = shape.m;
    int64_t actualN = shape.n;
    if (shape.p + shape.q <= shape.m) {
        const int64_t l1 = shape.q - 1;
        const int64_t l2 = std::min(shape.n - shape.q + 1, shape.m + 2 - shape.p - shape.q);
        const int64_t l3 =
            std::max(static_cast<int64_t>(0), std::min(shape.p + shape.n - shape.m - 1, shape.p + shape.q - 2));
        actualM = (l3 == 0) ? (shape.p + shape.q + l2 - 2) : shape.m;
        actualN = l1 + l2 + l3;
    } else {
        actualM = shape.m;
        actualN = std::min(shape.m - 1 + shape.q, shape.n);
    }
    shape.m = actualM;
    shape.n = actualN;
    shape.kind = (shape.p >= shape.m) ? TND_LINE_KIND_BAND_DENSE : TND_LINE_KIND_BAND;
    return shape;
}

int64_t QuantFlashAttnMetadataCpuKernel::CountTndLineSameShapeRun(uint32_t start, TndLineRunShape& shape)
{
    shape = MakeTndLineRunShape(start);
    uint32_t end = start;
    while (end + 1 < static_cast<uint32_t>(batchSize_)) {
        const TndLineRunShape next = MakeTndLineRunShape(end + 1);
        if (next.m != shape.m || next.n != shape.n || next.p != shape.p || next.q != shape.q ||
            next.kind != shape.kind) {
            break;
        }
        end += 1;
    }
    return static_cast<int64_t>(end - start) + 1;
}

int64_t QuantFlashAttnMetadataCpuKernel::CalTndLineRunRounds(const TndLineRunShape& shape, int64_t total)
{
    const int64_t k = static_cast<int64_t>(aicCoreNum_);
    const bool isBand = IsTndLineBandMode();
    if (TndLineRunEmpty(shape, isBand)) {
        return 0;
    }
    if (!isBand) {
        int64_t virtM = 0;
        int64_t virtN = 0;
        GetRightDownVirt(shape.m, shape.n, virtM, virtN);
        const int64_t runPairs = total / 2;
        const int64_t pairRounds = (runPairs > 0 && virtN > 0) ? TndLineCeilDiv(virtN * runPairs, k) * virtM : 0;
        const int64_t singleRounds = (total % 2 == 1) ? TndLineCeilDiv(shape.n, k) * shape.m : 0;
        return pairRounds + singleRounds;
    }
    if (shape.kind == TND_LINE_KIND_BAND_DENSE) {
        return TndLineCeilDiv(shape.n * total, k) * shape.m;
    }
    if (shape.p + shape.q <= shape.m) {
        return CalBandNarrowRounds(k, total, shape.m, shape.n, shape.p, shape.q);
    }
    return TndLineCeilDiv(CalBandWideCols(shape.m, shape.n, shape.p, shape.q) * total, k) * shape.m;
}

bool QuantFlashAttnMetadataCpuKernel::PreferTndLineSwizzle()
{
    const int64_t k = static_cast<int64_t>(aicCoreNum_);
    const int64_t n1 = static_cast<int64_t>(baseInfo.kvHeadNum);
    const int64_t bSize = static_cast<int64_t>(batchSize_);
    const bool isBand = IsTndLineBandMode();
    if (k <= 0 || n1 <= 0 || bSize <= 0) {
        return false;
    }
    if (numHeadsQ_ != numHeadsKv_ || n1 < 2) {
        return false;
    }
    for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
        const int64_t s1Len = static_cast<int64_t>(GetS1SeqSize(static_cast<uint32_t>(bIdx)));
        const int64_t s2Len = static_cast<int64_t>(GetS2SeqSize(static_cast<uint32_t>(bIdx)));
        if (s1Len == 0 || s2Len == 0) {
            continue;
        }
        if (!isBand && s1Len != s2Len) {
            return false;
        }
    }

    bool hasWork = false;
    int64_t i = 0;
    while (i < bSize) {
        TndLineRunShape shape;
        const int64_t groupSize = CountTndLineSameShapeRun(static_cast<uint32_t>(i), shape);
        const int64_t total = n1 * groupSize;
        if (TndLineRunEmpty(shape, isBand)) {
            i += groupSize;
            continue;
        }
        hasWork = true;
        bool worthy = true;
        if (!isBand) {
            int64_t virtM = 0;
            int64_t virtN = 0;
            GetRightDownVirt(shape.m, shape.n, virtM, virtN);
            worthy = (shape.m == shape.n && total / 2 > 0) || TndLineRunIsWorthy(k, virtM, virtN, total / 2);
            if (worthy && (total % 2 == 1) && shape.m < std::min(k, shape.n)) {
                worthy = false;
            }
        } else if (shape.kind == TND_LINE_KIND_BAND_DENSE) {
            worthy = TndLineRunIsWorthy(k, shape.m, shape.n, total);
        } else if (shape.p + shape.q <= shape.m) {
            worthy = shape.m * shape.n > TND_LINE_TINY_AREA;
        } else {
            worthy = TndLineRunIsWorthy(k, shape.m, CalBandWideCols(shape.m, shape.n, shape.p, shape.q), total);
        }
        if (!worthy) {
            return false;
        }
        i += groupSize;
    }
    return hasWork;
}

bool QuantFlashAttnMetadataCpuKernel::CalTndLineSwizzleSchedule(TndLineSchedule& sched)
{
    const int64_t bSize = static_cast<int64_t>(batchSize_);
    const int64_t n1 = static_cast<int64_t>(baseInfo.kvHeadNum);
    sched.roundPrefix.assign(bSize + 1, 0);
    sched.s1Outer.assign(bSize, 0);
    sched.s2Outer.assign(bSize, 0);
    sched.lineM.assign(bSize, 0);
    sched.lineN.assign(bSize, 0);
    sched.lineP.assign(bSize, 0);
    sched.lineQ.assign(bSize, 0);
    sched.runSize.assign(bSize, 0);
    sched.s1Token.assign(bSize, 0);
    sched.s2Token.assign(bSize, 0);

    for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
        int64_t m = 0;
        int64_t n = 0;
        GetTndLineOuterMN(static_cast<uint32_t>(bIdx), m, n);
        sched.s1Outer[bIdx] = m;
        sched.s2Outer[bIdx] = n;
        CalTndLineActualToken(static_cast<uint32_t>(bIdx), sched.s1Token[bIdx], sched.s2Token[bIdx]);
    }

    int64_t i = 0;
    while (i < bSize) {
        TndLineRunShape shape;
        const int64_t groupSize = CountTndLineSameShapeRun(static_cast<uint32_t>(i), shape);
        const int64_t runRounds = CalTndLineRunRounds(shape, n1 * groupSize);
        if (runRounds > INT32_MAX - sched.roundPrefix[i]) {
            KERNEL_LOG_ERROR("QuantFAG TND line swizzle exceeds int32 metadata round capacity");
            return false;
        }
        const int64_t soloEnd = sched.roundPrefix[i] + runRounds;
        for (int64_t r = 0; r < groupSize; ++r) {
            sched.roundPrefix[i + r + 1] = soloEnd;
            sched.lineM[i + r] = std::max(static_cast<int64_t>(0), shape.m);
            sched.lineN[i + r] = std::max(static_cast<int64_t>(0), shape.n);
            sched.lineP[i + r] = std::max(static_cast<int64_t>(0), shape.p);
            sched.lineQ[i + r] = std::max(static_cast<int64_t>(0), shape.q);
            sched.runSize[i + r] = groupSize;
        }
        i += groupSize;
    }
    return sched.roundPrefix[bSize] > 0;
}

bool QuantFlashAttnMetadataCpuKernel::GenQuantFagTndLineSchedule(optiling::detail::QuantFAGMetaData& gradMetaData)
{
    TndLineSchedule sched;
    if (!CalTndLineSwizzleSchedule(sched)) {
        return false;
    }
    const int64_t bSize = static_cast<int64_t>(batchSize_);
    const int64_t roundPrefixBase = static_cast<int64_t>(optiling::QUANT_FAG_ARRAY_BASE_INDEX);
    const int64_t s1OuterBase = roundPrefixBase + bSize + 1;
    const int64_t s2OuterBase = s1OuterBase + bSize;
    const int64_t lineMBase = s2OuterBase + bSize;
    const int64_t lineNBase = lineMBase + bSize;
    const int64_t linePBase = lineNBase + bSize;
    const int64_t lineQBase = linePBase + bSize;
    const int64_t runSizeBase = lineQBase + bSize;
    const int64_t s1TokenBase = runSizeBase + bSize;
    const int64_t s2TokenBase = s1TokenBase + bSize;
    const int64_t needSize = s2TokenBase + bSize;
    const int64_t fagRowSize = static_cast<int64_t>(GetFagOffset());
    if (needSize > fagRowSize) {
        KERNEL_LOG_INFO("QuantFAG TND line swizzle: batchSize %ld needs %ld metadata slots but row size is only %ld, "
                        "falling back to dense swizzle",
                        bSize, needSize, fagRowSize);
        return false;
    }

    gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_VALID_INDEX, 1);
    gradMetaData.SetDeterMaxRound(optiling::QUANT_FAG_DETER_MAX_NUM_INDEX, sched.roundPrefix[bSize]);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_ROUND_PREFIX_OFFSET_INDEX, roundPrefixBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_S1_OUTER_OFFSET_INDEX, s1OuterBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_S2_OUTER_OFFSET_INDEX, s2OuterBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_MODE_INDEX, optiling::QUANT_FAG_TND_LINE_MODE);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_TND_LINE_M_OFFSET_INDEX, lineMBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_TND_LINE_N_OFFSET_INDEX, lineNBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_TND_LINE_P_OFFSET_INDEX, linePBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_TND_LINE_Q_OFFSET_INDEX, lineQBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_TND_LINE_RUN_SIZE_OFFSET_INDEX, runSizeBase);
    for (int64_t bIdx = 0; bIdx <= bSize; ++bIdx) {
        gradMetaData.SetMetaData(static_cast<uint32_t>(roundPrefixBase + bIdx), sched.roundPrefix[bIdx]);
    }
    for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
        gradMetaData.SetMetaData(static_cast<uint32_t>(s1OuterBase + bIdx), sched.s1Outer[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(s2OuterBase + bIdx), sched.s2Outer[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(lineMBase + bIdx), sched.lineM[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(lineNBase + bIdx), sched.lineN[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(linePBase + bIdx), sched.lineP[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(lineQBase + bIdx), sched.lineQ[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(runSizeBase + bIdx), sched.runSize[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(s1TokenBase + bIdx), sched.s1Token[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(s2TokenBase + bIdx), sched.s2Token[bIdx]);
    }
    gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_KIND_INDEX, optiling::QUANT_FAG_TND_LINE_SWIZZLE);
    KERNEL_LOG_INFO("QuantFAG TND line swizzle: batchSize %ld, aicNum %d, loopMax %ld, rowSize %ld", bSize, aicCoreNum_,
                    sched.roundPrefix[bSize], fagRowSize);
    return true;
}

// varlen 确定性 swizzle 分核: 逐 batch 累计轮数
//   prefix[0]   = 0
//   prefix[b+1] = prefix[b] + ceil(n_b * n1 / k) * m_b
// 对标 FAG flash_attention_score_grad_tiling_varlen_regbase.cpp 的 CalcTNDSwizzleParam dense 分支,
// kernel 端解码见 deter.h 的 CalTNDDenseSwizzleIndex。
bool QuantFlashAttnMetadataCpuKernel::CalDeterSwizzleSchedule(std::vector<int64_t>& roundPrefix,
                                                              std::vector<int64_t>& s1OuterList,
                                                              std::vector<int64_t>& s2OuterList,
                                                              std::vector<std::vector<int64_t>>& sparseData)
{
    const int64_t k = static_cast<int64_t>(aicCoreNum_);
    const int64_t n1 = static_cast<int64_t>(baseInfo.kvHeadNum);
    const int64_t bSize = static_cast<int64_t>(batchSize_);
    if (k <= 0 || n1 <= 0 || bSize <= 0) {
        KERNEL_LOG_ERROR("QuantFAG grad schedule: invalid aicCoreNum %ld, headNum %ld or batchSize %ld", k, n1, bSize);
        return false;
    }
    // 反向 swizzle 调度只支持 N1 == N2 (g == 1)
    if (numHeadsQ_ != numHeadsKv_) {
        KERNEL_LOG_ERROR("QuantFAG grad schedule: only N1 == N2 supported, got numHeadsQ %d numHeadsKv %d", numHeadsQ_,
                         numHeadsKv_);
        return false;
    }

    const bool sparse = maskMode_ == 3 || maskMode_ == 4;
    const bool packed = layoutQ_ == "TND" && layoutKv_ == "TND";
    const bool extendedSchedule = sparse || packed;
    sparseData.assign(extendedSchedule ? optiling::QUANT_FAG_SPARSE_ARRAY_COUNT : 0, std::vector<int64_t>(bSize));
    roundPrefix.assign(bSize + 1, 0);
    s1OuterList.assign(bSize, 0);
    s2OuterList.assign(bSize, 0);
    for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
        const int64_t s1Len = static_cast<int64_t>(GetS1SeqSize(static_cast<uint32_t>(bIdx)));
        const int64_t s2Len = static_cast<int64_t>(GetS2SeqSize(static_cast<uint32_t>(bIdx)));
        if (s1Len < 0 || s2Len < 0 || (maxSeqlenQ_ >= 0 && s1Len > maxSeqlenQ_) ||
            (maxSeqlenKv_ >= 0 && s2Len > maxSeqlenKv_)) {
            KERNEL_LOG_ERROR("QuantFAG grad schedule: batch %ld has invalid sequence lengths (s1 %ld, s2 %ld)", bIdx,
                             s1Len, s2Len);
            return false;
        }
        if (s1Len == 0 || s2Len == 0) {
            roundPrefix[bIdx + 1] = roundPrefix[bIdx];
            continue;
        }
        const int64_t m = CeilDiv<int64_t>(s1Len, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_M));
        const int64_t n = CeilDiv<int64_t>(s2Len, static_cast<int64_t>(optiling::QUANT_FAG_CUBE_BASE_N));
        // swizzle 一轮内同一 head 最多 min(k, n) 个核并发, 它们的 dQ 坐标为 (s2o + delta) % m,
        // 要求互不重叠才能保证确定性累加顺序, 故 m >= min(k, n)。
        if (!extendedSchedule && m < std::min(k, n)) {
            KERNEL_LOG_ERROR("QuantFAG grad schedule: batch %ld s1Outer %ld < min(aicNum %ld, s2Outer %ld), "
                             "deterministic swizzle schedule is unsafe (s1 %ld, s2 %ld)",
                             bIdx, m, k, n, s1Len, s2Len);
            return false;
        }
        s1OuterList[bIdx] = m;
        s2OuterList[bIdx] = n;
        int64_t scheduleRows = m;
        int64_t scheduleCols = n;
        if (sparse) {
            // FAG right-down token correction, using effective batch lengths.
            // Clamping an unlimited/large window to max(Lq,Lkv) preserves its mask.
            const int64_t limit = std::max(s1Len, s2Len);
            const int64_t left = maskMode_ == 3 || winLeft_ == -1 ? limit : std::min<int64_t>(winLeft_, limit);
            const int64_t right = maskMode_ == 3 ? 0 : (winRight_ == -1 ? limit : std::min<int64_t>(winRight_, limit));
            const int64_t s1Token = left + s1Len - s2Len;
            const int64_t s2Token = right - s1Len + s2Len;
            const int64_t rowBegin = std::max<int64_t>(0, -s2Token) / optiling::QUANT_FAG_CUBE_BASE_M;
            const int64_t colBegin = std::max<int64_t>(0, -s1Token) / optiling::QUANT_FAG_CUBE_BASE_N;
            const int64_t rowEnd = CeilDiv<int64_t>(std::min(s1Len, s2Len + s1Token), optiling::QUANT_FAG_CUBE_BASE_M);
            const int64_t colEnd = CeilDiv<int64_t>(std::min(s2Len, s1Len + s2Token), optiling::QUANT_FAG_CUBE_BASE_N);
            scheduleCols = colEnd - colBegin;
            // Empty virtual rows prevent dQ collisions even for short Q / long KV.
            // Every (head, column) keeps the same core across its entire row sweep.
            scheduleRows = std::max(rowEnd - rowBegin, std::min(k, scheduleCols));
            sparseData[0][bIdx] = rowBegin;
            sparseData[1][bIdx] = colBegin;
            sparseData[2][bIdx] = scheduleRows;
            sparseData[3][bIdx] = scheduleCols;
            sparseData[4][bIdx] = s1Token;
            sparseData[5][bIdx] = s2Token;
        }
        if (packed && !sparse) {
            // Keep a unique dQ row per active core even when Q is shorter than KV.
            scheduleRows = std::max(m, std::min(k, n));
            sparseData[0][bIdx] = 0;
            sparseData[1][bIdx] = 0;
            sparseData[2][bIdx] = scheduleRows;
            sparseData[3][bIdx] = scheduleCols;
            sparseData[4][bIdx] = 0;
            sparseData[5][bIdx] = 0;
        }
        const int64_t rounds = CeilDiv<int64_t>(scheduleCols * n1, k) * scheduleRows;
        if (rounds > INT32_MAX - roundPrefix[bIdx]) {
            KERNEL_LOG_ERROR("QuantFAG grad schedule exceeds int32 metadata round capacity");
            return false;
        }
        roundPrefix[bIdx + 1] = roundPrefix[bIdx] + rounds;
    }
    return true;
}

bool QuantFlashAttnMetadataCpuKernel::GenQuantFagGradMetaData()
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("QuantFAG grad schedule: metadata is empty");
        return false;
    }
    // 偏移与 GenMetaData 写 deterMaxRound 时同源, 也与 kernel 侧的 tiling.metadata_len 一致
    detail::QuantFAGMetaData gradMetaData(metaData_->GetData(), GetFagOffset());
    gradMetaData.SetMetaData(optiling::QUANT_FAG_NEED_INIT_OUTPUT_INDEX, needInitOutput_ ? 1 : 0);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_BATCH_SIZE_INDEX, batchSize_);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_AIC_CORE_NUM_INDEX, aicCoreNum_);

    if (!HasVarlenSeq()) {
        // 等长场景: kernel 走定长分核, deterMaxRound 已由 GenMetaData 写入(fagDeterMaxRound_),
        // 这里不重复计算, 只标记调度数据不可用
        gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_VALID_INDEX, 0);
        return true;
    }

    if ((maskMode_ == MASK_MODE_CAUSAL || maskMode_ == MASK_MODE_BAND) && PreferTndLineSwizzle() &&
        GenQuantFagTndLineSchedule(gradMetaData)) {
        return true;
    }

    std::vector<int64_t> roundPrefix;
    std::vector<int64_t> s1OuterList;
    std::vector<int64_t> s2OuterList;
    std::vector<std::vector<int64_t>> sparseData;
    if (!CalDeterSwizzleSchedule(roundPrefix, s1OuterList, s2OuterList, sparseData)) {
        gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_VALID_INDEX, 0);
        gradMetaData.SetDeterMaxRound(optiling::QUANT_FAG_DETER_MAX_NUM_INDEX, 0);
        return false;
    }

    // 三个 per-batch 数组按 batch 紧凑排布, 起点写进标量槽位供 kernel 读取
    const int64_t bSize = static_cast<int64_t>(batchSize_);
    const int64_t roundPrefixBase = static_cast<int64_t>(optiling::QUANT_FAG_ARRAY_BASE_INDEX);
    const int64_t s1OuterBase = roundPrefixBase + bSize + 1; // roundPrefix 有 b+1 项
    const int64_t s2OuterBase = s1OuterBase + bSize;
    const int64_t sparseBase = s2OuterBase + bSize;
    const int64_t needSize = sparseBase + static_cast<int64_t>(sparseData.size()) * bSize;
    // 容量校验: 数组区不得越出 metadata 第二行。第二行长度 = 第一行长度 = fagOffset
    const int64_t fagRowSize = static_cast<int64_t>(GetFagOffset());
    if (needSize > fagRowSize) {
        KERNEL_LOG_ERROR("QuantFAG grad schedule: batchSize %ld needs %ld metadata slots but row size is only %ld, "
                         "please allocate metadata by 16 + (aicNum + aivNum) * 16 * batch * numHeadsKv",
                         bSize, needSize, fagRowSize);
        gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_VALID_INDEX, 0);
        return false;
    }

    gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_VALID_INDEX, 1);
    // varlen 下等长公式算出的 fagDeterMaxRound_ 不适用, 覆盖为按 seqused 逐 batch 累加的真实轮数
    gradMetaData.SetDeterMaxRound(optiling::QUANT_FAG_DETER_MAX_NUM_INDEX, roundPrefix[batchSize_]);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_ROUND_PREFIX_OFFSET_INDEX, roundPrefixBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_S1_OUTER_OFFSET_INDEX, s1OuterBase);
    gradMetaData.SetMetaData(optiling::QUANT_FAG_S2_OUTER_OFFSET_INDEX, s2OuterBase);
    for (int64_t bIdx = 0; bIdx <= bSize; ++bIdx) {
        gradMetaData.SetMetaData(static_cast<uint32_t>(roundPrefixBase + bIdx), roundPrefix[bIdx]);
    }
    for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
        gradMetaData.SetMetaData(static_cast<uint32_t>(s1OuterBase + bIdx), s1OuterList[bIdx]);
        gradMetaData.SetMetaData(static_cast<uint32_t>(s2OuterBase + bIdx), s2OuterList[bIdx]);
    }
    gradMetaData.SetMetaData(optiling::QUANT_FAG_SCHEDULE_KIND_INDEX,
                             sparseData.empty() ? 1 : optiling::QUANT_FAG_DENSE_SWIZZLE);
    for (size_t field = 0; field < sparseData.size(); ++field) {
        const int64_t fieldBase = sparseBase + static_cast<int64_t>(field) * bSize;
        gradMetaData.SetMetaData(optiling::QUANT_FAG_ROW_BEGIN_OFFSET_INDEX + field, fieldBase);
        for (int64_t bIdx = 0; bIdx < bSize; ++bIdx) {
            gradMetaData.SetMetaData(static_cast<uint32_t>(fieldBase + bIdx), sparseData[field][bIdx]);
        }
    }
    KERNEL_LOG_INFO("QuantFAG grad schedule: batchSize %ld, aicNum %d, loopMax %ld, offsets %ld/%ld/%ld, rowSize %ld",
                    bSize, aicCoreNum_, roundPrefix[batchSize_], roundPrefixBase, s1OuterBase, s2OuterBase, fagRowSize);
    return true;
}

uint32_t QuantFlashAttnMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }

    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t QuantFlashAttnMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }

    if (layoutKv_ == "TND") {
        if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
            const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensKv_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenKv_);
}

bool QuantFlashAttnMetadataCpuKernel::BalanceSchedule(SectionStreamKResult& splitRes)
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

bool QuantFlashAttnMetadataCpuKernel::GenMetaData(SectionStreamKResult& splitRes)
{
    if (metaData_ == nullptr || metaData_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    int32_t dimNum = 0;
    int64_t rowSize = 0;
    GetMetadataRowInfo(dimNum, rowSize);

    // 容量与维度校验: FA/FD 调度区需求 16 + sectionNum*(aic+aiv)*16 不得超过单行长度
    // (2D 为 dim1, 1D 为 dim0); shape 信息完全不可用时跳过校验(兼容旧调用方)
    if (dimNum > 0 && rowSize > 0) {
        int64_t needSize = optiling::METADATA_STRIDE + static_cast<int64_t>(splitRes.sectionNum) *
                                                           (aicCoreNum_ + aivCoreNum_) * optiling::METADATA_STRIDE;
        if (needSize > rowSize) {
            KERNEL_LOG_ERROR("metadata row size %ld is smaller than required %ld (sectionNum %d, aic %d, aiv %d), "
                             "please allocate metadata by 16 + (aicNum + aivNum) * 16 * batch * numHeadsKv",
                             rowSize, needSize, splitRes.sectionNum, aicCoreNum_, aivCoreNum_);
            return false;
        }
        if (isGradEnabled_ && dimNum < 2) {
            KERNEL_LOG_ERROR("metadata must be 2D (2, scheduleSize) when is_grad_enabled is true, but got %dD", dimNum);
            return false;
        }
    }
    detail::FaMetaData faMetadata(aicCoreNum_, aivCoreNum_, splitRes.sectionNum, metaData_->GetData());
    faMetadata.Clear(); // set to all 0

    SetMetadataHead(splitRes, faMetadata);
    SetMetadataFa(splitRes, faMetadata);
    SetMetadataFd(splitRes, faMetadata);

    // FAG 写入必须在 Clear 之后(避免被 FA/FD 清零覆盖), 且偏移取第二行起点
    // (= dim1, grad 侧按 GetDim(1) 推导, 保持一致); shape 不可用时回退旧常量
    if (isGradEnabled_) {
        detail::QuantFAGMetaData quantFAGMetaData(metaData_->GetData(), GetFagOffset());
        quantFAGMetaData.SetDeterMaxRound(optiling::QUANT_FAG_DETER_MAX_NUM_INDEX, fagDeterMaxRound_);
    }
    return true;
}

// 第二行(反向 FAG 区)起点。优先使用宿主侧经 attr 下发的 shape(aclnn 层读取, 可靠),
// attr 缺失时回退 AICPU 侧 TensorShape(部分平台不填充, 可能得到 -1/0)。
void QuantFlashAttnMetadataCpuKernel::GetMetadataRowInfo(int32_t& dimNum, int64_t& rowSize)
{
    dimNum = static_cast<int32_t>(metadataDimNum_);
    rowSize = metadataRowSize_;
    if (rowSize <= 0 || dimNum <= 0) {
        auto outShape = metaData_ == nullptr ? nullptr : metaData_->GetTensorShape();
        if (outShape != nullptr && outShape->GetDims() >= 1) {
            dimNum = outShape->GetDims();
            rowSize = dimNum >= 2 ? outShape->GetDimSize(1) : outShape->GetDimSize(0);
        }
    }
}

uint32_t QuantFlashAttnMetadataCpuKernel::GetFagOffset()
{
    int32_t dimNum = 0;
    int64_t rowSize = 0;
    GetMetadataRowInfo(dimNum, rowSize);
    return (dimNum >= 2 && rowSize > 0) ? static_cast<uint32_t>(rowSize) : optiling::QUANT_FAG_METADATA_SIZE;
}
void QuantFlashAttnMetadataCpuKernel::SetMetadataHead(const SectionStreamKResult& splitRes,
                                                      optiling::detail::FaMetaData& faMetadata)
{
    faMetadata.SetHeadMedata(optiling::HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);

    faMetadata.SetHeadMedata(optiling::HEAD_IS_FD_INDEX, 0);
    for (uint32_t sectionId = 0; sectionId < splitRes.sectionNum; ++sectionId) {
        auto fdSplitRes = splitRes.sectionFdResult[sectionId];
        if (fdSplitRes.usedVecNum > 0) {
            faMetadata.SetHeadMedata(optiling::HEAD_IS_FD_INDEX, 1);
        }
    }

    faMetadata.SetHeadMedata(optiling::HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMedata(optiling::HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    faMetadata.SetHeadMedata(optiling::HEAD_AIC_NUM_INDEX, static_cast<uint32_t>(aicCoreNum_));
    faMetadata.SetHeadMedata(optiling::HEAD_AIV_NUM_INDEX, static_cast<uint32_t>(aivCoreNum_));
    faMetadata.SetHeadMedata(optiling::HEAD_NEED_INIT_OUTPUT_INDEX, needInitOutput_ ? 1U : 0U);
}

void QuantFlashAttnMetadataCpuKernel::SetMetadataFa(const SectionStreamKResult& splitRes,
                                                    optiling::detail::FaMetaData& faMetadata)
{
    for (uint32_t sectionId = 0; sectionId < splitRes.sectionNum; ++sectionId) {
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
    }
}

void QuantFlashAttnMetadataCpuKernel::SetMetadataFd(const SectionStreamKResult& splitRes,
                                                    optiling::detail::FaMetaData& faMetadata)
{
    for (uint32_t sectionId = 0; sectionId < splitRes.sectionNum; ++sectionId) {
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
}

namespace {
static const char* kernelType = "QuantFlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantFlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
