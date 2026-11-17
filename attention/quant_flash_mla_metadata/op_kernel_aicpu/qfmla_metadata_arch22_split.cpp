/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <numeric>
#include <algorithm>
#include <vector>

#include "../../common/op_kernel/aicpu_common.h"
#include "qfmla_metadata_arch22_split.h"

namespace aicpu {
namespace arch22 {
namespace {

template <typename T>
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

static int64_t CeilDivision(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

constexpr int64_t M_FD_BASE_SIZE_MLA = 8;
constexpr uint32_t MAX_SPLIT_RATIO = 2U;
constexpr uint32_t SEQ_LEN_MIN = 4096U;
constexpr uint32_t SEQ_LEN_MAX = 5120U;
constexpr uint32_t SEQ_LEN_MIN_V2 = 37000U;
constexpr uint32_t SEQ_LEN_MAX_V2 = 65536U;
constexpr uint32_t HEAD_DIM_FD = 512U;
constexpr uint32_t HEAD_DIM_V_FD = 512U;
constexpr uint32_t SPARSE_MODE_FD = 3U;
constexpr uint32_t BYTE_BLOCK = 32U;
constexpr uint32_t MAX_SPLIT_SIZE = 8192U;
constexpr uint32_t FIA_BALANCE_SG_BASIC_SIZE = 128U;
constexpr uint32_t PRE_LOAD_NUM_MLA = 2U;
constexpr uint32_t DEQUANT_PER_CHANNEL_MODE = 0U;
constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_MODE = 3U;
constexpr int64_t QS_MAX_MLA = 32;
constexpr int64_t MSD_VEC_LOAD = 1024;

} // namespace

bool Arch22IfaSplit::Run(const Arch22ScheduleInput& input, Arch22ScheduleResult& result)
{
    if (!ParseInputs(input)) {
        return false;
    }
    if (!PreProcess()) {
        return false;
    }
    if (!Split()) {
        return false;
    }
    FillResult(result);
    return true;
}

bool Arch22IfaSplit::ParseInputs(const Arch22ScheduleInput& input)
{
    cacheSeqlens_ = input.cacheSeqlens;
    cuSeqlensQ_ = input.cuSeqlensQ;
    sequsedQ_ = input.sequsedQ;

    batchSize_ = static_cast<uint32_t>(cacheSeqlens_.size());
    if (batchSize_ == 0U) {
        KERNEL_LOG_ERROR("arch22: cache_seqlens is required and must be non-empty for PA scene");
        return false;
    }
    if (input.batchSize > 0 && static_cast<uint32_t>(input.batchSize) != batchSize_) {
        KERNEL_LOG_ERROR("arch22: batchSize %u mismatch with cache_seqlens size %u",
                         static_cast<uint32_t>(input.batchSize), batchSize_);
        return false;
    }
    numHeads_ = static_cast<uint32_t>(input.numHeadsQ);
    numKvHeads_ = static_cast<uint32_t>(input.numHeadsKv);
    if (numHeads_ == 0U) {
        KERNEL_LOG_ERROR("arch22: the query's heads num must not be zero");
        return false;
    }
    if (numKvHeads_ == 0U) {
        numKvHeads_ = numHeads_;
    }
    if (numKvHeads_ > numHeads_ || numHeads_ % numKvHeads_ != 0U) {
        KERNEL_LOG_ERROR("arch22: numKeyValueHeads(%u) must not be greater than numHeads(%u) "
                         "and numHeads must be divisible by numKvHeads",
                         numKvHeads_, numHeads_);
        return false;
    }
    nNumOfQInOneGroup_ = numHeads_ / numKvHeads_;
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = 1U;
    s1Outer_ = 1U;

    headDim_ = static_cast<uint32_t>(input.headDimQk > 0 ? input.headDimQk : 576);
    blockSize_ = 128U;
    maxSeqlenQ_ = input.maxSeqlenQ;
    maxSeqlenKv_ = input.maxSeqlenKv;
    maskMode_ = input.maskMode;
    sparseMode_ = static_cast<uint32_t>(maskMode_);
    attenMaskFlag_ = (maskMode_ != 0);
    aicNum_ = input.aicCoreNum;
    aivNum_ = input.aivCoreNum;
    if (aicNum_ == 0U || aivNum_ == 0U) {
        KERNEL_LOG_ERROR("arch22: num of core obtained is 0 (aic %u, aiv %u)", aicNum_, aivNum_);
        return false;
    }
    cvRatio_ = aivNum_ / aicNum_;

    if (input.socVersion.find("310P") != std::string::npos) {
        socVersion_ = IfaSocVersion::SOC_ASCEND_310P;
    } else {
        socVersion_ = IfaSocVersion::SOC_ASCEND_910B;
    }

    const std::string& layout = input.layoutQ;
    if (layout == "TND" || layout == "TND_NTD") {
        inputLayout_ = IfaLayout::TND;
        outputLayout_ = (layout == "TND") ? IfaLayout::TND : IfaLayout::NTD;
    } else if (layout == "BSH" || layout == "BSH_NBSD") {
        inputLayout_ = IfaLayout::BSH_BSND;
        outputLayout_ = (layout == "BSH") ? IfaLayout::BSH_BSND : IfaLayout::NBSD;
    } else if (layout == "BSND" || layout == "BSND_NBSD") {
        inputLayout_ = IfaLayout::BSH_BSND;
        outputLayout_ = (layout == "BSND") ? IfaLayout::BSH_BSND : IfaLayout::NBSD;
    } else if (layout == "BNSD" || layout == "BNSD_NBSD") {
        inputLayout_ = IfaLayout::BNSD;
        outputLayout_ = (layout == "BNSD") ? IfaLayout::BNSD : IfaLayout::NBSD;
    } else {
        KERNEL_LOG_ERROR("arch22: only support BSH, BSND, BNSD and TND, got %s", layout.c_str());
        return false;
    }
    inputKvLayout_ = IfaLayout::NZ;

    if (inputLayout_ == IfaLayout::TND) {
        if (!QKVPreProcess4TND()) {
            return false;
        }
    } else {
        if (maxSeqlenQ_ <= 0) {
            KERNEL_LOG_ERROR("arch22: BSH/BSND scene requires max_seqlen_q > 0 to derive qSeqSize");
            return false;
        }
        qSeqSize_ = static_cast<uint32_t>(maxSeqlenQ_);
        if (qSeqSize_ > 32U) {
            KERNEL_LOG_ERROR("arch22: Query S(%u) should not be bigger than 32 in MLA", qSeqSize_);
            return false;
        }
        if (!sequsedQ_.empty()) {
            if (sequsedQ_.size() != batchSize_) {
                KERNEL_LOG_ERROR("arch22: seqused_q must contain %u elements, got %zu", batchSize_, sequsedQ_.size());
                return false;
            }
            for (uint32_t b = 0; b < batchSize_; ++b) {
                if (sequsedQ_[b] < 0 || sequsedQ_[b] > QS_MAX_MLA || (maxSeqlenQ_ > 0 && sequsedQ_[b] > maxSeqlenQ_)) {
                    KERNEL_LOG_ERROR("arch22: seqused_q[%u]=%ld must be in [0, 32] and not exceed max_seqlen_q=%d", b,
                                     sequsedQ_[b], maxSeqlenQ_);
                    return false;
                }
            }
        }
    }
    s1SplitSize_ = qSeqSize_;

    if (maxSeqlenKv_ > 0) {
        sMax_ = static_cast<uint32_t>(maxSeqlenKv_);
    } else {
        int64_t maxKv = 0;
        for (int64_t len : cacheSeqlens_) {
            maxKv = std::max(maxKv, len);
        }
        sMax_ = Align<uint32_t>(static_cast<uint32_t>(maxKv), blockSize_);
    }
    seqSize_ = sMax_;

    if (!ProcessActualSeqLen()) {
        return false;
    }
    if (maxSeqlenKv_ > 0 && maxActualseq_ > sMax_) {
        KERNEL_LOG_ERROR("arch22: invalid actual seq length %u for PA, max actual seq length is larger than "
                         "capacity %u (max_seqlen_kv)",
                         maxActualseq_, sMax_);
        return false;
    }
    return true;
}

bool Arch22IfaSplit::QKVPreProcess4TND()
{
    if (cuSeqlensQ_.empty()) {
        KERNEL_LOG_ERROR("arch22: TND scene requires cu_seqlens_q (actualSeqLengthsQ)");
        return false;
    }
    if (cuSeqlensQ_[0] != 0) {
        KERNEL_LOG_ERROR("arch22: cu_seqlens_q must start with 0, got %ld", cuSeqlensQ_[0]);
        return false;
    }
    ifaCuSeqQ_.assign(cuSeqlensQ_.begin() + 1, cuSeqlensQ_.end());
    actualLenQDims_ = static_cast<uint32_t>(ifaCuSeqQ_.size());
    if (actualLenQDims_ == 0U) {
        KERNEL_LOG_ERROR("arch22: TND the length of the query's actual sequence lengths should be greater than 0");
        return false;
    }
    if (cacheSeqlens_.size() != ifaCuSeqQ_.size()) {
        KERNEL_LOG_ERROR("arch22: TND the length of the query's(%zu) and the key/value's(%zu) "
                         "actual sequence lengths should be equal",
                         ifaCuSeqQ_.size(), cacheSeqlens_.size());
        return false;
    }
    batchSizeQ_ = batchSize_ = actualLenQDims_;

    tSeqSize_ = static_cast<uint32_t>(ifaCuSeqQ_[actualLenQDims_ - 1]);
    if (maxSeqlenQ_ > 0) {
        if (static_cast<uint32_t>(maxSeqlenQ_) != tSeqSize_) {
            KERNEL_LOG_ERROR("arch22: TND query T(%d) should be equal to the last element(%u) "
                             "of cu_seqlens_q",
                             maxSeqlenQ_, tSeqSize_);
            return false;
        }
    }

    int64_t tmpQSeqSize = 0;
    for (uint32_t b = 0; b < actualLenQDims_; ++b) {
        int64_t qs = (b == 0U) ? ifaCuSeqQ_[0] : (ifaCuSeqQ_[b] - ifaCuSeqQ_[b - 1]);
        if (qs < 0 || qs > QS_MAX_MLA) {
            KERNEL_LOG_ERROR("arch22: TND QS of batch %u computed by cu_seqlens_q should be in range [0, 32], got %ld",
                             b, qs);
            return false;
        }
        tmpQSeqSize = std::max(tmpQSeqSize, qs);
        qSeqSquareSum_ += static_cast<uint64_t>(qs) * static_cast<uint64_t>(qs);
    }
    qSeqSize_ = static_cast<uint32_t>(tmpQSeqSize);
    return true;
}

bool Arch22IfaSplit::ProcessActualSeqLen()
{
    actualSeqLenFlag_ = true;
    if (inputLayout_ == IfaLayout::TND) {
        if (cacheSeqlens_.empty()) {
            KERNEL_LOG_ERROR("arch22: TND scene requires cache_seqlens (actualSeqLengths)");
            return false;
        }
        actualLenDims_ = static_cast<uint32_t>(cacheSeqlens_.size());
        maxActualseq_ = sMax_;
    } else {
        if (cacheSeqlens_.empty()) {
            KERNEL_LOG_ERROR("arch22: PA scene requires cache_seqlens (actualSeqLengths)");
            return false;
        }
        actualLenDims_ = static_cast<uint32_t>(cacheSeqlens_.size());
        if (actualLenDims_ == 0U) {
            KERNEL_LOG_ERROR("arch22: the key/value's actual sequence lengths size can not be zero in PA scene");
            return false;
        }
        if (actualLenDims_ != 1U && actualLenDims_ < batchSize_) {
            KERNEL_LOG_ERROR("arch22: the size of the key/value's actual sequence lengths(%u) "
                             "should be greater than q batch(%u) or equal to 1",
                             actualLenDims_, batchSize_);
            return false;
        }
        actualLenDims_ = std::min(actualLenDims_, batchSize_);
    }

    uint32_t loop = (actualLenDims_ == 1U) ? 1U : batchSize_;
    for (uint32_t i = 0U; i < loop; ++i) {
        int64_t actLen = (actualLenDims_ == 1U) ? cacheSeqlens_[0] : cacheSeqlens_[i];
        if (actLen < 0) {
            KERNEL_LOG_ERROR("arch22: the value of the key/value's actual sequence lengths[%u](%ld) "
                             "must be greater than or equal to 0",
                             i, actLen);
            return false;
        }
        maxActualseq_ = (maxActualseq_ < static_cast<uint32_t>(actLen)) ? static_cast<uint32_t>(actLen) : maxActualseq_;
        if (actLen == 0) {
            hasZeroActualseq_ = true;
        }
        if (cacheSeqlens_[i] != cacheSeqlens_[0]) {
            isSameActualseq_ = false;
        }
    }
    return true;
}

bool Arch22IfaSplit::PreProcess()
{
    SetupPerfMode();
    IsFdBalanceCase();
    balanceModeFlag_ = IsBalanceSplitCore();
    SetCoreNum();
    return true;
}

void Arch22IfaSplit::SetupPerfMode()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
        return;
    }
    perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
}

void Arch22IfaSplit::SetCoreNum()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        coreNum_ = aicNum_;
        return;
    }
    if (balanceModeFlag_) {
        coreNum_ = aicNum_;
        return;
    }
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
        perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        coreNum_ = aicNum_;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        coreNum_ = aicNum_;
    } else {
        coreNum_ = aivNum_;
    }
}

void Arch22IfaSplit::IsFdBalanceCase()
{
    if (IsValidFlag() || IsValidFlag3B() || IsValidFlag560B()) {
        fdBalanceFlag_ = 1U;
    } else {
        fdBalanceFlag_ = 0U;
    }
}

bool Arch22IfaSplit::CheckCommonConditions(const ValidityConfigFD& config) const
{
    return CheckBatchAndQSeqSize(config) && CheckHeadDimensions(config) && CheckQuantizationFlags(config.sparseMode) &&
           CheckActualSeqLengths(config.expectedActualSeqLength);
}

bool Arch22IfaSplit::CheckBatchAndQSeqSize(const ValidityConfigFD& config) const
{
    if (std::find(config.validBatchSizes.begin(), config.validBatchSizes.end(), static_cast<int32_t>(batchSize_)) ==
        config.validBatchSizes.end()) {
        return false;
    }
    if (std::find(config.validQSeqSizes.begin(), config.validQSeqSizes.end(), static_cast<int32_t>(qSeqSize_)) ==
        config.validQSeqSizes.end()) {
        return false;
    }
    return true;
}

bool Arch22IfaSplit::CheckHeadDimensions(const ValidityConfigFD& config) const
{
    return (numHeads_ == static_cast<uint32_t>(config.numHeads) &&
            numKvHeads_ == static_cast<uint32_t>(config.numKvHeads) &&
            headDim_ == static_cast<uint32_t>(config.headDim) && headDimV_ == static_cast<uint32_t>(config.headDimV));
}

bool Arch22IfaSplit::CheckQuantizationFlags(int32_t sparseMode) const
{
    if (!quantFlag_) {
        return false;
    }
    if (sparseMode_ != static_cast<uint32_t>(sparseMode)) {
        return false;
    }
    if (inputKvLayout_ != IfaLayout::NZ || !pageAttentionFlag_) {
        return false;
    }
    const int64_t queryQuantMode = DEQUANT_PER_TOKEN_HEAD_MODE;
    const int64_t keyAntiQuantMode = DEQUANT_PER_CHANNEL_MODE;
    const int64_t valueAntiQuantMode = DEQUANT_PER_CHANNEL_MODE;
    if (queryQuantMode != static_cast<int64_t>(DEQUANT_PER_TOKEN_HEAD_MODE) ||
        keyAntiQuantMode != static_cast<int64_t>(DEQUANT_PER_CHANNEL_MODE) ||
        valueAntiQuantMode != static_cast<int64_t>(DEQUANT_PER_CHANNEL_MODE)) {
        return false;
    }
    return true;
}

bool Arch22IfaSplit::CheckActualSeqLengths(int64_t expectedActualSeqLength) const
{
    if (cacheSeqlens_.empty()) {
        return false;
    }
    for (uint32_t bIdx = 0U; bIdx < batchSize_; ++bIdx) {
        int64_t s2 = (actualLenDims_ == 1U) ? cacheSeqlens_[0] : cacheSeqlens_[bIdx];
        if (expectedActualSeqLength >= 0) {
            if (s2 < static_cast<int64_t>(SEQ_LEN_MIN_V2) || s2 > static_cast<int64_t>(SEQ_LEN_MAX_V2)) {
                return false;
            }
        } else {
            if (s2 < static_cast<int64_t>(SEQ_LEN_MIN) || s2 > static_cast<int64_t>(SEQ_LEN_MAX)) {
                return false;
            }
        }
    }
    return true;
}

bool Arch22IfaSplit::IsValidFlag()
{
    ValidityConfigFD cfg;
    cfg.validBatchSizes = {1};
    cfg.validQSeqSizes = {2};
    cfg.numHeads = 128;
    cfg.numKvHeads = 1;
    cfg.headDim = static_cast<int32_t>(HEAD_DIM_FD);
    cfg.headDimV = static_cast<int32_t>(HEAD_DIM_V_FD);
    cfg.sparseMode = static_cast<int32_t>(SPARSE_MODE_FD);
    cfg.expectedActualSeqLength = 1;
    return CheckCommonConditions(cfg);
}

bool Arch22IfaSplit::IsValidFlag3B()
{
    ValidityConfigFD cfg;
    cfg.validBatchSizes = {1, 2, 4};
    cfg.validQSeqSizes = {4};
    cfg.numHeads = 8;
    cfg.numKvHeads = 1;
    cfg.headDim = static_cast<int32_t>(HEAD_DIM_FD);
    cfg.headDimV = static_cast<int32_t>(HEAD_DIM_V_FD);
    cfg.sparseMode = static_cast<int32_t>(SPARSE_MODE_FD);
    cfg.expectedActualSeqLength = -1; // 表示 [4096, 5120]
    return CheckCommonConditions(cfg);
}

bool Arch22IfaSplit::IsValidFlag560B()
{
    ValidityConfigFD cfg;
    cfg.validBatchSizes = {4};
    cfg.validQSeqSizes = {2, 3, 4};
    cfg.numHeads = 64;
    cfg.numKvHeads = 1;
    cfg.headDim = static_cast<int32_t>(HEAD_DIM_FD);
    cfg.headDimV = static_cast<int32_t>(HEAD_DIM_V_FD);
    cfg.sparseMode = static_cast<int32_t>(SPARSE_MODE_FD);
    cfg.expectedActualSeqLength = -1;
    return CheckCommonConditions(cfg);
}

bool Arch22IfaSplit::DealSameSeqEachBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return isSameActualseq_;
        } else {
            return isSameSeqAllKVTensor_;
        }
    }
    return isSameActualseq_;
}

bool Arch22IfaSplit::HasZeroSeqBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return hasZeroActualseq_;
        } else {
            return isSameSeqAllKVTensor_ ? hasZeroActualseq_ : hasZeroActualseq_;
        }
    }
    return hasZeroActualseq_;
}

bool Arch22IfaSplit::IsBalanceSplitCore()
{
    if (fdBalanceFlag_ == 1U) {
        return true;
    }
    if (perfMode_ != IfaPerfMode::CUBE_VIEW_MM_MLA) {
        return false;
    }
    if (inputLayout_ == IfaLayout::TND) {
        return true;
    }
    if (antiQuantFlag_) {
        return false;
    }
    if (inputLayout_ == IfaLayout::BNSD) {
        return false;
    }
    if (isWorkspace_) {
        return false;
    }
    return !DealSameSeqEachBatch();
}

bool Arch22IfaSplit::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            return false;
        }
    }
    if (balanceModeFlag_) {
        return tndFDCoreArrLen_ > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false;
    }
    if (gqaMtpFlag_) {
        return false;
    }
    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }
    return false;
}

bool Arch22IfaSplit::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = 0.4f; // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5f; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5f;
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }
    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        return true;
    }
    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        return true;
    }
    return false;
}

Arch22IfaSplit::IfaAmlaMode Arch22IfaSplit::GetAmlaMode() const
{
    if (!ropeFlag_ || inputKvLayout_ != IfaLayout::NZ || antiQuantFlag_ || quantFlag_) {
        return IfaAmlaMode::DISABLE_AMLA;
    }
    if (blockSize_ == 128U) {
        return IfaAmlaMode::AMLA;
    }
    return IfaAmlaMode::DISABLE_AMLA;
}

bool Arch22IfaSplit::Split()
{
    amlaMode_ = GetAmlaMode();
    if (balanceModeFlag_) {
        if (fdBalanceFlag_ == 1U) {
            return SplitBalancedFd();
        }
        return SplitBalanced();
    }
    return SplitUnbalanced();
}

void Arch22IfaSplit::GetActualSeqInfo(const std::vector<int64_t>& actualSeqKv, ActualSeqInfo& actualSeqInfo) const
{
    uint32_t bSize = batchSize_;
    if (inputLayout_ == IfaLayout::TND) {
        for (uint32_t b = 0; b < bSize; ++b) {
            actualSeqInfo.actualSeqQ[b] = (b == 0U) ? ifaCuSeqQ_[0] : (ifaCuSeqQ_[b] - ifaCuSeqQ_[b - 1]);
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    } else {
        for (uint32_t b = 0; b < bSize; ++b) {
            actualSeqInfo.actualSeqQ[b] = static_cast<int64_t>(qSeqSize_);
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    }
}

void Arch22IfaSplit::GetSeqTilingInfo(const std::vector<int64_t>& actualSeqKv, const ActualSeqInfo& actualSeqInfo,
                                      SeqTilingInfo& seqTilingInfo) const
{
    uint32_t bSize = batchSize_;
    uint32_t souter = s1SplitSize_;
    uint32_t s2BasicSize = sInnerSize_;
    uint32_t n2Size = numKvHeads_;
    uint64_t totalS2Length = 0U;
    for (uint32_t bIdx = 0U; bIdx < bSize; ++bIdx) {
        uint32_t s1 = static_cast<uint32_t>(actualSeqInfo.actualSeqQ[bIdx]);
        uint64_t actualSeqLen = static_cast<uint64_t>(actualSeqKv[0]);
        if (actualLenDims_ != 1U) {
            actualSeqLen = static_cast<uint64_t>(actualSeqKv[bIdx]);
        }
        if (actualSeqInfo.maxActualseqkv <= 0) { // kv actseq全为0
            seqTilingInfo.s1OuterNum[bIdx] = 1U;
            seqTilingInfo.s2OuterNum[bIdx] = 1U;
        } else {
            seqTilingInfo.s1OuterNum[bIdx] = (souter == 0U) ? 0U : (s1 + (souter - 1U)) / souter;
            seqTilingInfo.s2OuterNum[bIdx] =
                static_cast<uint32_t>((actualSeqLen + s2BasicSize - 1U) / s2BasicSize); // 线段长度
        }

        uint32_t bTotalS2Length = seqTilingInfo.s1OuterNum[bIdx] * seqTilingInfo.s2OuterNum[bIdx] * n2Size;
        totalS2Length += bTotalS2Length;
        if ((seqTilingInfo.s1OuterNum[bIdx] > 0U) && (seqTilingInfo.s2OuterNum[bIdx] > 0U)) {
            seqTilingInfo.lastValidBIdx = bIdx;
        }
    }
    if (totalS2Length > coreNum_) {
        seqTilingInfo.avgS2Length = (totalS2Length + coreNum_ - 1U) / coreNum_; // 平均长度向上取整
    }
}

bool Arch22IfaSplit::FillBalancedSplitCoreInfo(const TilingIndexes& tilingIdx, BalancedSplitTilingInfo& tilingInfo)
{
    if (tilingInfo.currCoreIdx >= MAX_AIC_CORE_NUM) {
        KERNEL_LOG_ERROR("arch22: balanced split core index %u exceeds max aic core num %u", tilingInfo.currCoreIdx,
                         MAX_AIC_CORE_NUM);
        return false;
    }
    coreBEnd_[tilingInfo.currCoreIdx] = tilingIdx.bIdx;
    coreS1OuterEnd_[tilingInfo.currCoreIdx] = tilingIdx.s1Idx;
    coreS2End_[tilingInfo.currCoreIdx] = tilingIdx.s2Idx;
    tilingInfo.coreLoad[tilingInfo.currCoreIdx] = tilingInfo.accumS2Length;
    tilingInfo.currCoreIdx += 1U;
    return true;
}

void Arch22IfaSplit::EndSplitForCurrentCore(const TilingIndexes& tilingIdx, const SeqTilingInfo& seqTilingInfo,
                                            uint32_t& currKvSplitPart, BalancedSplitTilingInfo& tilingInfo)
{
    tilingInfo.accumS2Length += 1U;
    // 更新当前核的End分核信息
    if (!FillBalancedSplitCoreInfo(tilingIdx, tilingInfo)) {
        return;
    }
    if (tilingIdx.s2Idx < seqTilingInfo.s2OuterNum[tilingIdx.bIdx] - 1U) {
        currKvSplitPart += 1U;
        balanceFDCoreStartKVSplitNum_[tilingInfo.currCoreIdx] = currKvSplitPart - 1U;
    } else {
        balanceFDCoreStartKVSplitNum_[tilingInfo.currCoreIdx] = 0U;
    }
    tilingInfo.needUpdate = false;
}

void Arch22IfaSplit::SplitBalancedForEachHead(uint32_t bIdx, const SeqTilingInfo& seqTilingInfo,
                                              BalancedSplitTilingInfo& tilingInfo)
{
    balanceFDCoreStartKVSplitNum_[0] = 0U;

    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; ++s1OuterIdx) {
        uint32_t currKvSplitPart = 1U;
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; ++s2Idx) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart,
                                       tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            if (tilingInfo.tndFDCoreArrLen >= MAX_AIC_CORE_NUM) {
                KERNEL_LOG_ERROR("arch22: tndFDCoreArrLen %u exceeds max aic core num %u", tilingInfo.tndFDCoreArrLen,
                                 MAX_AIC_CORE_NUM);
                return;
            }
            balanceFDCoreBArr_[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr_[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr_[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void Arch22IfaSplit::SplitBalancedForEachHeadFd(uint32_t bIdx, const SeqTilingInfo& seqTilingInfo,
                                                BalancedSplitTilingInfo& tilingInfo,
                                                std::vector<int64_t>& gS1SplitNumOfFdHead, uint32_t s1)
{
    balanceFDCoreStartKVSplitNum_[0] = 0U;

    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; ++s1OuterIdx) {
        uint32_t currKvSplitPart = 1U;
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; ++s2Idx) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart,
                                       tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            if (tilingInfo.tndFDCoreArrLen >= MAX_AIC_CORE_NUM) {
                KERNEL_LOG_ERROR("arch22: tndFDCoreArrLen %u exceeds max aic core num %u", tilingInfo.tndFDCoreArrLen,
                                 MAX_AIC_CORE_NUM);
                return;
            }
            balanceFDCoreBArr_[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr_[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr_[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            int64_t gSize = static_cast<int64_t>(nNumOfQInOneGroup_);
            int64_t currFdS1gSize =
                (s1OuterIdx == seqTilingInfo.s1OuterNum[bIdx] - 1) ?
                    static_cast<int64_t>(s1) * gSize - (static_cast<int64_t>(seqTilingInfo.s1OuterNum[bIdx]) - 1) *
                                                           static_cast<int64_t>(s1SplitSize_) * gSize :
                    static_cast<int64_t>(s1SplitSize_) * gSize; // 处理尾块
            int64_t currFdS1gSplitSize = (currFdS1gSize + M_FD_BASE_SIZE_MLA - 1) / M_FD_BASE_SIZE_MLA;
            int64_t currFdS1gLastPartSize = currFdS1gSize % M_FD_BASE_SIZE_MLA;
            if (currFdS1gLastPartSize == 0) {
                currFdS1gLastPartSize = M_FD_BASE_SIZE_MLA;
            }
            gS1SplitNumOfFdHead[tilingInfo.tndFDCoreArrLen] = currFdS1gSplitSize;
            gS1LastPartSizeOfFdHead_[tilingInfo.tndFDCoreArrLen] = static_cast<uint32_t>(currFdS1gLastPartSize);
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void Arch22IfaSplit::SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t>& gS1SplitNumOfFdHead,
                                uint32_t* s2SplitNumOfFdHead, uint32_t aivCoreNum, SeqTilingInfo& seqTilingInfo)
{
    (void)seqTilingInfo;

    uint32_t totalFDLoad = 0U;
    uint32_t totalFDHeadSplit = 0U;

    for (uint32_t i = 0U; i < tndFDCoreArrLen; ++i) {
        totalFDLoad += s2SplitNumOfFdHead[i] * static_cast<uint32_t>(gS1SplitNumOfFdHead[i]);
        totalFDHeadSplit += static_cast<uint32_t>(gS1SplitNumOfFdHead[i]);
    }
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, aivCoreNum);
    if (maxVectorNum == 0U) {
        usedVecNumOfFd_ = 0U;
        return;
    }
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0U;
    uint32_t preTmpFDIndexEndOfFdHead = 0U;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0U;

    for (uint32_t i = 0U; i < tndFDCoreArrLen; ++i) {
        uint32_t fDKVSplitNum = s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0U; gS1SplitIdx < static_cast<uint32_t>(gS1SplitNumOfFdHead[i]); ++gS1SplitIdx) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector);
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (static_cast<double>(fDKVSplitNum) > remainSpace * static_cast<double>(MAX_SPLIT_RATIO)) {
                if (curCoreIndex >= MAX_AIC_CORE_NUM_X2) {
                    KERNEL_LOG_ERROR("arch22: fd vec core index %u exceeds max %u", curCoreIndex, MAX_AIC_CORE_NUM_X2);
                    return;
                }
                gS1IdxEndOfFdHead_[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                gS1IdxEndOfFdHeadSplit_[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector); // 当前未分配的总负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += static_cast<int64_t>(fDKVSplitNum);
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    if (curCoreIndex >= MAX_AIC_CORE_NUM_X2) {
        KERNEL_LOG_ERROR("arch22: fd vec core index %u exceeds max %u", curCoreIndex, MAX_AIC_CORE_NUM_X2);
        return;
    }
    gS1IdxEndOfFdHead_[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    gS1IdxEndOfFdHeadSplit_[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    usedVecNumOfFd_ = curCoreIndex + 1U;
}

bool Arch22IfaSplit::IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                                        const std::vector<uint32_t>& s1OuterNum,
                                        const std::vector<uint32_t>& s2OuterNum) const
{
    if (needUpdate) {
        return false;
    }
    if ((lastValidBMoreIdx <= 0U) || (lastValidBMoreIdx >= bSize)) {
        return false;
    }
    if (s1OuterNum[lastValidBMoreIdx] == 0) {
        return false;
    }
    if (s2OuterNum[lastValidBMoreIdx] != 0) {
        return false;
    }
    return true;
}

bool Arch22IfaSplit::SplitBalanced()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    if (s1SplitSize_ == 0U) {
        KERNEL_LOG_ERROR("arch22: balanced split requires numHeadsQ/numHeadsKv(%u) <= %u", gSize, s1gBasicSize);
        return false;
    }

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const std::vector<int64_t>& actualSeqKv = cacheSeqlens_;
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; ++bIdx) {
        for (uint32_t nIdx = 0U; nIdx < n2Size; ++nIdx) {
            SplitBalancedForEachHead(bIdx, seqTilingInfo, tilingInfo);
        }
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        if (!FillBalancedSplitCoreInfo(
                TilingIndexes(seqTilingInfo.lastValidBIdx, seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                              seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
                tilingInfo)) {
            return false;
        }
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum,
                           seqTilingInfo.s2OuterNum)) {
        if (!FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo)) {
            return false;
        }
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tndFDCoreArrLen_ = tilingInfo.tndFDCoreArrLen;

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
    }

    return true;
}

bool Arch22IfaSplit::SplitBalancedFd()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    if (s1SplitSize_ == 0U) {
        KERNEL_LOG_ERROR("arch22: balanced fd split requires numHeadsQ/numHeadsKv(%u) <= %u", gSize, s1gBasicSize);
        return false;
    }

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const std::vector<int64_t>& actualSeqKv = cacheSeqlens_;
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);
    std::vector<int64_t> gS1SplitNumOfFdHead(MAX_AIC_CORE_NUM_X2);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; ++bIdx) {
        uint32_t s1 = static_cast<uint32_t>(actualSeqInfo.actualSeqQ[bIdx]);
        for (uint32_t nIdx = 0U; nIdx < n2Size; ++nIdx) {
            SplitBalancedForEachHeadFd(bIdx, seqTilingInfo, tilingInfo, gS1SplitNumOfFdHead, s1);
        }
    }
    for (uint32_t i = 0U; i < tilingInfo.tndFDCoreArrLen; ++i) {
        gS1SplitNumOfFdHeadMla_[i] = static_cast<uint32_t>(gS1SplitNumOfFdHead[i]);
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        if (!FillBalancedSplitCoreInfo(
                TilingIndexes(seqTilingInfo.lastValidBIdx, seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                              seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
                tilingInfo)) {
            return false;
        }
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum,
                           seqTilingInfo.s2OuterNum)) {
        if (!FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo)) {
            return false;
        }
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tndFDCoreArrLen_ = tilingInfo.tndFDCoreArrLen;

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
        // 使能fd负载均衡
        uint32_t* s2SplitNumOfFdHead = balanceFDCoreKVSplitArr_;
        uint32_t aivCoreNumOfFd = (aivNum_ / aicNum_) * usedCoreNum_;
        SplitFDMLa(tilingInfo.tndFDCoreArrLen, gS1SplitNumOfFdHead, s2SplitNumOfFdHead, aivCoreNumOfFd, seqTilingInfo);
    }

    return true;
}

bool Arch22IfaSplit::SplitUnbalanced()
{
    if (amlaMode_ == IfaAmlaMode::AMLA_3BUF) {
        gMax_ = 512U; // 3buf场景下切512
    }
    if (gqaMtpFlag_) {
        gMax_ = antiQuantFlag_ ? 128U : 256U;
    }

    if (ropeFlag_ || gqaMtpFlag_) {
        if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNzFlagForSplit() && inputLayout_ == IfaLayout::TND)) {
            s1SplitSize_ = gMax_ / nNumOfQInOneGroup_;
            if (s1SplitSize_ == 0U) {
                KERNEL_LOG_ERROR("arch22: unbalanced split requires numHeadsQ/numHeadsKv(%u) <= %u", nNumOfQInOneGroup_,
                                 gMax_);
                return false;
            }
            if (s1SplitSize_ > qSeqSize_) {
                s1SplitSize_ = qSeqSize_;
            }
            if (s1SplitSize_ == 0U) {
                KERNEL_LOG_ERROR("arch22: invalid qSeqSize 0 for unbalanced split");
                return false;
            }
            s1Outer_ = (qSeqSize_ + s1SplitSize_ - 1U) / s1SplitSize_;
        } else {
            // mla 模式支持G切分
            if (qSeqSize_ == 0U) {
                KERNEL_LOG_ERROR("arch22: invalid qSeqSize 0 for group split");
                return false;
            }
            groupSplitSize_ = gMax_ / qSeqSize_;
            groupSplitSize_ -= (groupSplitSize_ & 1U);
            if (groupSplitSize_ > nNumOfQInOneGroup_) {
                groupSplitSize_ = nNumOfQInOneGroup_;
            }
            if (groupSplitSize_ == 0U) {
                KERNEL_LOG_ERROR("arch22: invalid groupSplitSize 0 (gMax %u, qSeqSize %u)", gMax_, qSeqSize_);
                return false;
            }
            gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;
        }
    }

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }

    CalcInnerSize(seqSize_);
    return SplitBN();
}

bool Arch22IfaSplit::gqaKvNzFlagForSplit() const
{
    return false;
}

bool Arch22IfaSplit::SplitBN()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNzFlagForSplit() && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }

    for (uint32_t i = 0U; i < MAX_CORE_NUM; ++i) {
        startIdxEachCore_[i] = bn;
    }

    if (actualLenDims_ == 1U || bn <= coreNum_ || DealSameSeqEachBatch()) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray;
    if (actualLenDims_ > 0U) {
        validArray = InitSparseValidArray(cacheSeqlens_.data());
    } else {
        validArray = InitSparseValidArray(nullptr);
    }

    SetSparseStartIdx(validArray, static_cast<int64_t>(bn), static_cast<int64_t>(coreNum_), startIdxEachCore_,
                      CeilDivision(static_cast<int64_t>(bn), static_cast<int64_t>(coreNum_)));

    usedCoreNum_ = coreNum_;
    return true;
}

void Arch22IfaSplit::GetEstimatedLoad(int64_t& estimatedLoad) const
{
    if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
        estimatedLoad = MSD_VEC_LOAD;
    } else if (estimatedLoad == 0) {
        estimatedLoad = 1;
    }
}

std::vector<int64_t> Arch22IfaSplit::InitSparseValidArray(const int64_t* actualLens) const
{
    uint32_t outer;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNzFlagForSplit() && inputLayout_ == IfaLayout::TND)) {
        outer = s1Outer_;
    } else {
        outer = gOuter_;
    }
    std::vector<int64_t> res(batchSize_ * numKvHeads_ * outer);
    for (uint32_t b = 0U; b < batchSize_; ++b) {
        for (uint32_t n = 0U; n < numKvHeads_ * outer; ++n) {
            int64_t estimatedLoad = static_cast<int64_t>(seqSize_);
            if (actualLens != nullptr) {
                estimatedLoad = actualLens[b];
                GetEstimatedLoad(estimatedLoad);
            }
            res[b * numKvHeads_ * outer + n] = estimatedLoad;
        }
    }
    return res;
}

bool Arch22IfaSplit::BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                 std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

void Arch22IfaSplit::InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                   const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end,
                                              static_cast<int64_t>(0));
        } else {
            break;
        }
    }
}

void Arch22IfaSplit::SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize,
                                       int64_t validAivNum, uint32_t* sparseStartIdx, int64_t splitFactorSize)
{
    // initLoad: 使用均分策略, 保证后续不会比均分差
    std::vector<int64_t> localSparseStartIdx(MAX_CORE_NUM, totalSize);
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAivNum, 0);
    InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAivNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(MAX_CORE_NUM, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), static_cast<int64_t>(0));
    int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

    tmpsparseStartIdx[0] = 0;
    for (uint32_t idx = 1; idx < MAX_CORE_NUM; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
                singleLoadValue = redoSingleLoadValue;
            }
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAivNum - static_cast<int64_t>(idx)));
        }
    }

    InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        sparseStartIdx[idx] = static_cast<uint32_t>(localSparseStartIdx[idx]);
    }
}

bool Arch22IfaSplit::SplitBN_V0()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNzFlagForSplit() && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        coreNum_ = static_cast<uint32_t>(CeilDivision(
            static_cast<int64_t>(bn), CeilDivision(static_cast<int64_t>(bn), static_cast<int64_t>(aicNum_))));
    }
    if (coreNum_ == 0U) {
        KERNEL_LOG_ERROR("arch22: invalid bn 0 for SplitBN_V0");
        return false;
    }
    formerCoreNum_ = bn % coreNum_;
    if (formerCoreNum_ == 0U) {
        blockSplitBn2Range_ = bn / coreNum_;
        tailSplitedBatchRange_ = blockSplitBn2Range_;
    } else {
        blockSplitBn2Range_ = bn / coreNum_ + 1U;
        tailSplitedBatchRange_ = blockSplitBn2Range_ - 1U;
    }

    usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

    for (uint32_t i = 0U; i < formerCoreNum_; ++i) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; ++i) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return true;
}

bool Arch22IfaSplit::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    if (bn == 0U) {
        KERNEL_LOG_ERROR("arch22: invalid bn 0 for SplitBNS");
        return false;
    }
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align<uint32_t>(computeSeqSize, blockSize_);
    }
    CalcInnerSize(computeSeqSize);
    return true;
}

bool Arch22IfaSplit::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = MAX_SPLIT_SIZE; // 8192
    if (antiQuantFlag_) {
        if (nNumOfQInOneGroup_ > 1U) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
                sInnerSize_ = 2048U;
            } else {
                sInnerSize_ = 1024U;
            }
        } else if (nNumOfQInOneGroup_ == 1U) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                sInnerSize_ = 1024U;
            }
        }
    } else {
        uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
        uint32_t idx = std::min(nNumOfQInOneGroup_ / 5U, 2U);
        sInnerSize_ = sInnerSize[idx];
    }

    if (ropeFlag_) {
        sInnerSize_ = 512U;
        if (!isWorkspace_ && splitKVFlag_ && inputLayout_ != IfaLayout::TND) {
            if (seqSize == 256U) {
                sInnerSize_ = 128U;
            } else if (seqSize > 256U && seqSize <= sInnerSize_) {
                sInnerSize_ = (sInnerSize_ + 1U) / 2U;
            }
        }
        if (antiQuantFlag_) {
            sInnerSize_ = 1024U;
        }
    }

    if (pageAttentionFlag_ && blockSize_ != 0U) {
        if (sInnerSize_ % blockSize_ != 0U) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    if (sInnerSize_ > seqSize && (!(isWorkspace_ && splitKVFlag_))) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align<uint32_t>(sInnerSize_, BYTE_BLOCK);
    return true;
}

void Arch22IfaSplit::FillResult(Arch22ScheduleResult& result)
{
    uint32_t sInnerLoopSize = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        sInnerLoopSize = Align<uint32_t>(sInnerLoopSize, blockSize_);
    }
    sInnerLoopSize_ = sInnerLoopSize;

    result.balanceMode = balanceModeFlag_ ? 1U : 0U;
    result.splitKVFlag = splitKVFlag_;
    result.kvSplitPart = splitKVFlag_ ? kvSplitPart_ : 0U;
    result.fdBalanceFlag = fdBalanceFlag_;
    result.sInnerSize = sInnerSize_;
    result.sInnerSizeAlign = sInnerSizeAlign_;
    result.sInnerLoopTimes = sInnerLoopTimes_;
    result.sInnerSizeTail = sInnerSizeTail_;
    result.s1SplitSize = s1SplitSize_;
    result.groupSplitSize = groupSplitSize_;
    result.gOuter = gOuter_;
    result.s1Outer = s1Outer_;
    result.usedCoreNum = usedCoreNum_;
    result.sInnerLoopSize = sInnerLoopSize_;
    result.maxActualseq = maxActualseq_;
    result.hasZeroActualseq = hasZeroActualseq_;
    result.isSameActualseq = isSameActualseq_;
    result.qSeqSquareSum = qSeqSquareSum_;
    result.qSeqSize = qSeqSize_;
    result.tSeqSize = tSeqSize_;
    result.seqSize = seqSize_;
    result.coreNum = coreNum_;
    result.actualLenQDims = actualLenQDims_;
    result.actualLenDims = actualLenDims_;
    result.maskMode = maskMode_;
    result.blockSize = blockSize_;
    result.numHeadsQ = numHeads_;
    result.numHeadsKv = numKvHeads_;
    result.nNumOfQInOneGroup = nNumOfQInOneGroup_;
    result.batchSize = batchSize_;
    result.outputLayout = static_cast<uint32_t>(outputLayout_);
    result.kvLayout = 2U;
    result.tndFDCoreArrLen = tndFDCoreArrLen_;
    result.usedVecNumOfFd = usedVecNumOfFd_;

    const uint32_t faNum = std::min(usedCoreNum_, static_cast<uint32_t>(qfmla_arch22::ARCH22_MAX_FA_RECORDS));
    result.faRecords.resize(faNum);
    for (uint32_t i = 0U; i < faNum; ++i) {
        Arch22FaCoreRecord& rec = result.faRecords[i];
        if (balanceModeFlag_) {
            rec.sIdxStart = 0U;
            rec.sIdxEnd = 0U;
            rec.bEnd = (i < MAX_AIC_CORE_NUM) ? coreBEnd_[i] : 0U;
            rec.s1OuterEnd = (i < MAX_AIC_CORE_NUM) ? coreS1OuterEnd_[i] : 0U;
            rec.s2End = (i < MAX_AIC_CORE_NUM) ? coreS2End_[i] : 0U;
            rec.startKvSplitPos = (i < MAX_AIC_CORE_NUM) ? balanceFDCoreStartKVSplitNum_[i] : 0U;
        } else {
            rec.sIdxStart = startIdxEachCore_[i];
            rec.sIdxEnd = (i + 1U < MAX_CORE_NUM) ? startIdxEachCore_[i + 1U] : 0U;
            rec.bEnd = 0U;
            rec.s1OuterEnd = 0U;
            rec.s2End = 0U;
            rec.startKvSplitPos = 0U;
        }
    }

    const uint32_t fdNum = std::min(std::max(tndFDCoreArrLen_, usedVecNumOfFd_),
                                    static_cast<uint32_t>(qfmla_arch22::ARCH22_MAX_FD_RECORDS));
    result.fdRecords.resize(fdNum);
    for (uint32_t i = 0U; i < fdNum; ++i) {
        Arch22FdRecord& rec = result.fdRecords[i];
        rec.bIdx = (i < MAX_AIC_CORE_NUM) ? balanceFDCoreBArr_[i] : 0U;
        rec.s1Idx = (i < MAX_AIC_CORE_NUM) ? balanceFDCoreS1Arr_[i] : 0U;
        rec.kvSplitNum = (i < MAX_AIC_CORE_NUM) ? balanceFDCoreKVSplitArr_[i] : 0U;
        rec.s1SplitNum = (i < MAX_AIC_CORE_NUM_X2) ? gS1SplitNumOfFdHeadMla_[i] : 0U;
        rec.s1LastPartSize = (i < MAX_AIC_CORE_NUM_X2) ? gS1LastPartSizeOfFdHead_[i] : 0U;
        rec.s1IdxEnd = (i < MAX_AIC_CORE_NUM_X2) ? gS1IdxEndOfFdHead_[i] : 0U;
        rec.s1IdxEndSplit = (i < MAX_AIC_CORE_NUM_X2) ? gS1IdxEndOfFdHeadSplit_[i] : 0U;
        rec.workspaceIdx = 0U;
    }
}

bool Arch22Schedule(const Arch22ScheduleInput& input, Arch22ScheduleResult& result)
{
    Arch22IfaSplit split;
    return split.Run(input, result);
}

bool GenMetaDataArch22(const Arch22ScheduleResult& result, int32_t* metadataPtr, int64_t rowSize, int64_t dimNum)
{
    if (metadataPtr == nullptr) {
        KERNEL_LOG_ERROR("arch22: metadata buffer is null");
        return false;
    }
    if (dimNum > 0 && dimNum != 2) {
        KERNEL_LOG_ERROR("arch22: metadata must be 2D, but got %ldD", dimNum);
        return false;
    }
    if (rowSize <= 0) {
        KERNEL_LOG_ERROR("arch22: metadata row size must be positive, got %ld", rowSize);
        return false;
    }
    const uint32_t faRecords = static_cast<uint32_t>(result.faRecords.size());
    const uint32_t fdRecords = static_cast<uint32_t>(result.fdRecords.size());
    if (faRecords > qfmla_arch22::ARCH22_MAX_FA_RECORDS || fdRecords > qfmla_arch22::ARCH22_MAX_FD_RECORDS) {
        KERNEL_LOG_ERROR("arch22: task records exceed capacity (fa %u > %u or fd %u > %u)", faRecords,
                         qfmla_arch22::ARCH22_MAX_FA_RECORDS, fdRecords, qfmla_arch22::ARCH22_MAX_FD_RECORDS);
        return false;
    }
    const int64_t requiredSize = static_cast<int64_t>(qfmla_arch22::GetArch22RequiredSize(faRecords, fdRecords));
    if (rowSize < requiredSize) {
        KERNEL_LOG_ERROR("arch22: metadata row size %ld is smaller than required %ld (fa %u, fd %u)", rowSize,
                         requiredSize, faRecords, fdRecords);
        return false;
    }

    const int64_t totalSlots = 2 * rowSize;
    for (int64_t i = 0; i < totalSlots; ++i) {
        metadataPtr[i] = 0;
    }

    int32_t* base = metadataPtr;
    auto WriteHead = [&base](uint32_t idx, uint32_t val) { base[idx] = static_cast<int32_t>(val); };

    WriteHead(qfmla_arch22::ARCH22_HEAD_BALANCE_MODE, result.balanceMode);
    WriteHead(qfmla_arch22::ARCH22_HEAD_SPLIT_KV_FLAG, result.splitKVFlag ? 1U : 0U);
    WriteHead(qfmla_arch22::ARCH22_HEAD_KV_SPLIT_PART, result.kvSplitPart);
    WriteHead(qfmla_arch22::ARCH22_HEAD_FD_MODE, result.fdBalanceFlag);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S_INNER_SIZE, result.sInnerSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S_INNER_SIZE_ALIGN, result.sInnerSizeAlign);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S_INNER_LOOP_TIMES, result.sInnerLoopTimes);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S_INNER_SIZE_TAIL, result.sInnerSizeTail);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S1_SPLIT_SIZE, result.s1SplitSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_GROUP_SPLIT_SIZE, result.groupSplitSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_G_OUTER, result.gOuter);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S1_OUTER, result.s1Outer);
    WriteHead(qfmla_arch22::ARCH22_HEAD_USED_CORE_NUM, result.usedCoreNum);
    WriteHead(qfmla_arch22::ARCH22_HEAD_PRELOAD_NUM, PRE_LOAD_NUM_MLA);
    WriteHead(qfmla_arch22::ARCH22_HEAD_NEED_INIT_OUTPUT, result.needInitOutput);
    WriteHead(qfmla_arch22::ARCH22_HEAD_IS_FD, result.splitKVFlag ? 1U : 0U);
    WriteHead(qfmla_arch22::ARCH22_HEAD_TND_FD_CORE_ARR_LEN, result.tndFDCoreArrLen);
    WriteHead(qfmla_arch22::ARCH22_HEAD_USED_VEC_NUM_FD, result.usedVecNumOfFd);
    WriteHead(qfmla_arch22::ARCH22_HEAD_FD_RECORD_NUM, fdRecords);
    WriteHead(qfmla_arch22::ARCH22_HEAD_FA_RECORD_NUM, faRecords);
    WriteHead(qfmla_arch22::ARCH22_HEAD_S_INNER_LOOP_SIZE, result.sInnerLoopSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_MAX_ACTUAL_SEQ_KV, result.maxActualseq);
    WriteHead(qfmla_arch22::ARCH22_HEAD_MASK_MODE, static_cast<uint32_t>(result.maskMode));
    WriteHead(qfmla_arch22::ARCH22_HEAD_SPARSE_MODE, static_cast<uint32_t>(result.maskMode));
    WriteHead(qfmla_arch22::ARCH22_HEAD_BLOCK_SIZE, result.blockSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_NUM_HEADS_Q, result.numHeadsQ);
    WriteHead(qfmla_arch22::ARCH22_HEAD_NUM_HEADS_KV, result.numHeadsKv);
    WriteHead(qfmla_arch22::ARCH22_HEAD_N_NUM_OF_Q_IN_ONE_GROUP, result.nNumOfQInOneGroup);
    WriteHead(qfmla_arch22::ARCH22_HEAD_Q_SEQ_SIZE, result.qSeqSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_T_SEQ_SIZE, result.tSeqSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_BATCH_SIZE, result.batchSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_KV_SEQ_SIZE, result.seqSize);
    WriteHead(qfmla_arch22::ARCH22_HEAD_ATTEN_MASK_FLAG, (result.maskMode != 0) ? 1U : 0U);
    WriteHead(qfmla_arch22::ARCH22_HEAD_OUTPUT_LAYOUT, result.outputLayout);
    WriteHead(qfmla_arch22::ARCH22_HEAD_SOFTMAX_LSE_FLAG, 0U);
    WriteHead(qfmla_arch22::ARCH22_HEAD_ACTUAL_LEN_Q_DIMS, result.actualLenQDims);
    WriteHead(qfmla_arch22::ARCH22_HEAD_ACTUAL_LEN_DIMS, result.actualLenDims);
    WriteHead(qfmla_arch22::ARCH22_HEAD_MAX_BLOCK_NUM_PER_BATCH, 0U);
    WriteHead(qfmla_arch22::ARCH22_HEAD_KV_LAYOUT, result.kvLayout);

    int32_t* faBase = base + qfmla_arch22::ARCH22_HEAD_SIZE;
    for (uint32_t i = 0U; i < faRecords; ++i) {
        int32_t* rec = faBase + i * qfmla_arch22::ARCH22_FA_RECORD_SIZE;
        const Arch22FaCoreRecord& src = result.faRecords[i];
        rec[qfmla_arch22::ARCH22_FA_S_IDX_START] = static_cast<int32_t>(src.sIdxStart);
        rec[qfmla_arch22::ARCH22_FA_S_IDX_END] = static_cast<int32_t>(src.sIdxEnd);
        rec[qfmla_arch22::ARCH22_FA_B_END] = static_cast<int32_t>(src.bEnd);
        rec[qfmla_arch22::ARCH22_FA_S1_OUTER_END] = static_cast<int32_t>(src.s1OuterEnd);
        rec[qfmla_arch22::ARCH22_FA_S2_END] = static_cast<int32_t>(src.s2End);
        rec[qfmla_arch22::ARCH22_FA_START_KV_SPLIT_POS] = static_cast<int32_t>(src.startKvSplitPos);
    }

    int32_t* fdBase = faBase + faRecords * qfmla_arch22::ARCH22_FA_RECORD_SIZE;
    for (uint32_t i = 0U; i < fdRecords; ++i) {
        int32_t* rec = fdBase + i * qfmla_arch22::ARCH22_FD_RECORD_SIZE;
        const Arch22FdRecord& src = result.fdRecords[i];
        rec[qfmla_arch22::ARCH22_FD_B_IDX] = static_cast<int32_t>(src.bIdx);
        rec[qfmla_arch22::ARCH22_FD_S1_IDX] = static_cast<int32_t>(src.s1Idx);
        rec[qfmla_arch22::ARCH22_FD_KV_SPLIT_NUM] = static_cast<int32_t>(src.kvSplitNum);
        rec[qfmla_arch22::ARCH22_FD_S1_SPLIT_NUM] = static_cast<int32_t>(src.s1SplitNum);
        rec[qfmla_arch22::ARCH22_FD_S1_LAST_PART_SIZE] = static_cast<int32_t>(src.s1LastPartSize);
        rec[qfmla_arch22::ARCH22_FD_S1_IDX_END] = static_cast<int32_t>(src.s1IdxEnd);
        rec[qfmla_arch22::ARCH22_FD_S1_IDX_END_SPLIT] = static_cast<int32_t>(src.s1IdxEndSplit);
        rec[qfmla_arch22::ARCH22_FD_WORKSPACE_IDX] = static_cast<int32_t>(src.workspaceIdx);
    }
    return true;
}

} // namespace arch22
} // namespace aicpu
