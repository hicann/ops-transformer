/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling.cpp
 * \brief
 */
#include <iostream>
#include <numeric>
#include "incre_flash_attention_tiling_base.h"
#include "incre_flash_attention_tiling_impl.h"
#include "log/log.h"
#include "../op_kernel/incre_flash_attention_tilingdata.h"

using std::pair;
namespace optiling {

const int64_t tokenDefault = 2147483647; // 2147483647 for token default value
const int32_t sparseDefault = 0;
constexpr int64_t mFdBaseSizeMla = 8;
constexpr uint32_t MAX_SPLIT_RATIO = 2;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t SEQ_LEN_MIN = 4096;
constexpr uint32_t SEQ_LEN_MAX = 5120;
constexpr uint32_t SEQ_LEN_MIN_V2 = 37000;
constexpr uint32_t SEQ_LEN_MAX_V2 = 65536;
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t HEAD_DIM_V = 512;
constexpr uint32_t SPARSE_MODE = 3;

uint64_t PFA_BENCHMARK_TILING_KEY = 1000000000000000000;

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

// 获取公约数
static uint32_t increGcd(uint32_t a, uint32_t b)
{
    if (a % b == 0U) {
        return b;
    }
    return increGcd(b, a % b);
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args>
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
constexpr uint64_t IFA_TILINGKEYOFFSET = uint64_t(10000000000000000UL);          // 10^16
constexpr uint64_t IFA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); // 10^15
template <typename... Args>
constexpr uint64_t IFA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

constexpr uint32_t PRE_LOAD_NUM = 4;
constexpr uint32_t BUF_NUM = 6;
constexpr uint32_t PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t ATB_MODE_VAL = 3;
constexpr uint32_t ATB_INNER_PRECISE = 2;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

void IFATiling::SetCoreNum()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        coreNum_ = aicNum_; // use aic num in 310p
    } else {
        if (balanceModeFlag_) {
            coreNum_ = aicNum_;
            return;
        }
        if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
            perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
            coreNum_ = aicNum_; // use aiv num when compute mm by vec
        } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
            coreNum_ = aicNum_;
        } else {
            coreNum_ = aivNum_; // split core by vec core
        }
    }
}

custom::graphStatus IFATiling::GetNpuInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    libapiSize_ = ascendcPlatform->GetLibApiWorkSpaceSize();

    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform->GetCoreNumAiv();
    aicNum_ = ascendcPlatform->GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(ifaContext_->opName, "num of core obtained is 0."),
                return custom::graphStatus::GRAPH_FAILED);

    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        socVersion_ = IfaSocVersion::SOC_ASCEND_310P;
    } else {
        socVersion_ = IfaSocVersion::SOC_ASCEND_910B;

        cvRatio_ = aivNum_ / aicNum_;
        OP_CHECK_IF((cvRatio_ != 1U) && (cvRatio_ != 2U),
                    OP_LOGE(ifaContext_->opName, "aicNum(%u):aivNum(%u) only support 1:1 or 1:2.", aicNum_, aivNum_),
                    return custom::graphStatus::GRAPH_FAILED);
    }

    OP_LOGI(ifaContext_->opName, "FIA aicNum: %u, aivNum:%u, cvRatio:%u.", aicNum_, aivNum_, cvRatio_);

    return custom::graphStatus::GRAPH_SUCCESS;
}

// 该demo支持输入参数的强校验
custom::graphStatus IFATiling::PreCheck()
{
    OP_CHECK_IF(ifaContext_->pseShift.hasValue, OP_LOGE(ifaContext_->opName, "this FA demo does not support pseShift"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSeqLengthsQ.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support actualSeqLengthsQ"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support deqScale1"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support quantScale1"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support deqScale2"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support quantScale2"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support antiquantScale"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support antiquantOffset"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryPaddingSize.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support queryPaddingSize"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support kvPaddingSize"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyAntiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support keyAntiquantOffset"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueAntiquantOffset.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support valueAntiquantOffset"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support keySharedPrefix"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support valueSharedPrefix"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSharedPrefixLen.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support actualSharedPrefixLen"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryRope.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support queryRope"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRope.hasValue, OP_LOGE(ifaContext_->opName, "this FA demo does not support keyRope"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRopeAntiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support keyRopeAntiquantScale"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->dequantScaleQuery.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support dequantScaleQuery"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->qStartIdx.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support qStartIdx"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvStartIdx.hasValue,
                OP_LOGE(ifaContext_->opName, "this FA demo does not support kvStartIdx"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->blockSize != 128 && ifaContext_->blockSize != 512,
                OP_LOGE(ifaContext_->opName, "this FA demo only support blockSize = 128/512"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->innerPrecise != 1,
                OP_LOGE(ifaContext_->opName, "this FA demo only support innerPrecise = 1"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->sparseMode != 0 && ifaContext_->sparseMode != 3,
                OP_LOGE(ifaContext_->opName, "this FA demo only support sparseMode = 0/3"),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(!ifaContext_->keyAntiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "keyAntiquantScale must be provided"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!ifaContext_->valueAntiquantScale.hasValue,
                OP_LOGE(ifaContext_->opName, "valueAntiquantScale must be provided"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(!ifaContext_->blockTable.hasValue, OP_LOGE(ifaContext_->opName, "blockTable must be provided"),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}
custom::graphStatus IFATiling::PreProcess()
{
    if (PreCheck() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (ProcessBaseInputs() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    bool ret = CheckIfRollBack();
    if (ret) {
        passToOldTiling_ = true;
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (ProcessOptionalTensors() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    SetupPerfMode();
    balanceModeFlag_ = IsBalanceSplitCore();
    SetCoreNum();
    UpdateL2CacheOffFlag();

    return custom::graphStatus::GRAPH_SUCCESS;
}

bool IFATiling::CheckCommonConditions(const ValidityConfigFD& config) const
{
    return CheckBatchAndQSeqSize(config.validBatchSizes, config.validQSeqSizes) &&
           CheckHeadDimensions(config.numHeads, config.numKvHeads, config.headDim, config.headDimV) &&
           CheckQuantizationFlags(config.sparseMode) && CheckActualSeqLengths(config.expectedActualSeqLength);
}

bool IFATiling::CheckBatchAndQSeqSize(const std::vector<int32_t>& validBatchSizes,
                                      const std::vector<int32_t>& validQSeqSizes) const
{
    if (std::find(validBatchSizes.begin(), validBatchSizes.end(), batchSize_) == validBatchSizes.end())
        return false;

    if (std::find(validQSeqSizes.begin(), validQSeqSizes.end(), qSeqSize_) == validQSeqSizes.end())
        return false;

    return true;
}

bool IFATiling::CheckHeadDimensions(int32_t numHeads, int32_t numKvHeads, int32_t headDim, int32_t headDimV) const
{
    return (numHeads_ == numHeads && numKvHeads_ == numKvHeads && headDim_ == headDim && headDimV_ == headDimV);
}

bool IFATiling::CheckQuantizationFlags(int32_t sparseMode) const
{
    if (!quantFlag_)
        return false;

    if (sparseMode_ != sparseMode)
        return false;

    if (inputKvLayout_ != IfaLayout::NZ || !pageAttentionFlag_)
        return false;

    int64_t queryQuantMode = ifaContext_->queryQuantMode;
    int64_t keyAntiQuantMode = ifaContext_->keyAntiquantMode;
    int64_t valueAntiQuantMode = ifaContext_->valueAntiquantMode;

    if (queryQuantMode != DEQUANT_PER_TOKEN_HEAD_MODE || keyAntiQuantMode != DEQUANT_PER_CHANNEL_MODE ||
        valueAntiQuantMode != DEQUANT_PER_CHANNEL_MODE)
        return false;

    return true;
}

bool IFATiling::CheckActualSeqLengths(int64_t expectedActualSeqLength) const
{
    return true;
}

bool IFATiling::IsValidFlag3B()
{
    ValidityConfigFD cfg{
        .validBatchSizes = {1, 2, 4}, // 表示有效的BatchSize大小
        .validQSeqSizes = {4},        // 表示有效的Qseq大小
        .numHeads = 8,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = -1 // 表示 [4096, 5120]
    };
    return CheckCommonConditions(cfg);
}

bool IFATiling::IsValidFlag560B()
{
    ValidityConfigFD cfg{.validBatchSizes = {4},
                         .validQSeqSizes = {2, 3, 4},
                         .numHeads = 64,
                         .numKvHeads = 1,
                         .headDim = HEAD_DIM,
                         .headDimV = HEAD_DIM_V,
                         .sparseMode = SPARSE_MODE,
                         .expectedActualSeqLength = -1};
    return CheckCommonConditions(cfg);
}

bool IFATiling::IsValidFlag()
{
    ValidityConfigFD cfg{
        .validBatchSizes = {1},
        .validQSeqSizes = {2},
        .numHeads = 128,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = 1 // 1表示整网场景下s2范围：[37000,65536]
    };
    return CheckCommonConditions(cfg);
}

void IFATiling::IsFdBalanceCase() {}

bool IFATiling::IsBalanceSplitCore()
{
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

    if (isWorkspace_) { // tiling下沉不走balance
        return false;
    }

    return !DealSameSeqEachBatch();
}

uint32_t IFATiling::GetTypeSize(at::ScalarType dtype) const
{
    switch (dtype) {
        case at::ScalarType::Float:
            return NUM_BYTES_FLOAT; // 4
        case at::ScalarType::Half:
            return NUM_BYTES_FLOAT16; // 2
        case at::ScalarType::BFloat16:
            return NUM_BYTES_BF16; // 2
        case at::ScalarType::Bool:
            return NUM_BYTES_BOOL; // 1
        case at::ScalarType::Char:
            return NUM_BYTES_INT8; // int8
        case at::ScalarType::Byte:
            return NUM_BYTES_INT8; // uint8
        default:
            return NUM_BYTES_UNDEF;
    }
}

static int64_t GetShapeSize(const at::IntArrayRef& shape)
{
    if (shape.empty()) {
        return 0;
    }
    int64_t n = 1;
    for (auto d : shape) {
        n *= d;
    }
    return n;
}

custom::graphStatus IFATiling::SetL2CacheFlag()
{
    auto kDType = ifaContext_->key.dType;
    uint32_t kvTypeSize = GetTypeSize(kDType);
    if (kvTypeSize == NUM_BYTES_UNDEF) {
        OP_LOGE(ifaContext_->opName, "Data type %s is not currently supported.",
                DataTypeToSerialString(kDType).c_str());
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (ropeFlag_) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    uint64_t kvSize = 0;
    auto batchOfQuery = ifaContext_->query.shape[0];
    auto batchOfKey = ifaContext_->key.shape[0];
    if (ifaContext_->blockTable.hasValue) {
        kvSize = ifaContext_->key.GetShapeSize();
    } else if (batchOfQuery != batchOfKey) { /* kv noncontinuous */
        for (int64_t size = 0; size < batchOfQuery; ++size) {
            auto keyTensorInList = ifaContext_->kCache[size];
            kvSize += GetShapeSize(keyTensorInList);
        }
    } else {
        kvSize = ifaContext_->key.GetShapeSize();
    }

    uint64_t l2CacheSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    // 考虑K、V，1.2为关闭L2Cache的系数
    if (static_cast<double>(kvSize) * kvTypeSize * 2.0f >= l2CacheSize * 1.2) {
        OP_LOGD(ifaContext_->opName, "L2 cache off");
        l2CacheOffFlag_ = 1U;
    }

    OP_LOGD(ifaContext_->opName, "l2CacheOffFlag_: %u, kvSize: %lu, kvTypeSize: %u, l2CacheSize: %lu", l2CacheOffFlag_,
            kvSize, kvTypeSize, l2CacheSize);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::QKVPreProcess()
{
    OP_CHECK_IF(ifaContext_->key.dType != ifaContext_->value.dType,
                OP_LOGE(ifaContext_->opName, "datatype of k tensor and value tensor is different"),
                return custom::graphStatus::GRAPH_FAILED);
    batchSizeQ_ = batchSize_ = ifaContext_->query.shape[0];
    inputQType_ = ifaContext_->query.dType;
    inputKvType_ = ifaContext_->key.dType;
    outputType_ = ifaContext_->attenOut.dType;

    numHeads_ = ifaContext_->numHeads;
    numKvHeads_ = ifaContext_->kvHeadNums;
    scaleValue_ = ifaContext_->scaleValue;
    blockSize_ = ifaContext_->blockSize;

    OP_LOGI(ifaContext_->opName, "scaleValue_ is %f.", scaleValue_);
    OP_CHECK_IF(numHeads_ == 0U, OP_LOGE(ifaContext_->opName, "the query's heads num is zero"),
                return custom::graphStatus::GRAPH_FAILED);
    if (numKvHeads_ == 0U) {
        numKvHeads_ = numHeads_;
    }
    OP_CHECK_IF(
        ((numKvHeads_ > numHeads_) || (numHeads_ % numKvHeads_ != 0U)),
        OP_LOGE(ifaContext_->opName, "Attr num_key_value_heads is invalid, n: %u, the key/value's heads num: %u",
                numHeads_, numKvHeads_),
        return custom::graphStatus::GRAPH_FAILED);
    nNumOfQInOneGroup_ = numHeads_ / numKvHeads_;
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = 1U;
    s1Outer_ = 1U;

    std::string layout(ifaContext_->layOut);
    uint32_t sOfQuery = 0U;
    uint32_t sOfHeadnum = 0U;
    uint32_t kDimNum = ifaContext_->key.shape.size();
    if (GetInOutLayoutAndProcessQInfo(layout, sOfQuery, sOfHeadnum, kDimNum) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    qSeqSize_ = sOfQuery;
    if (layout == "TND" || layout == "TND_NTD") {
        if (QKVPreProcess4TND(layout) != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
        sOfQuery = qSeqSize_;
        if (isWorkspace_ && ifaContext_->blockTable.hasValue) {
            uint32_t tndBatchSize = ifaContext_->blockTable.shape[0];
            if (tndBatchSize != 0) {
                sOfQuery = (tSeqSize_ + tndBatchSize - 1) / tndBatchSize;
            }
        }
    }
    s1SplitSize_ = qSeqSize_;

    if (GetRopeAndGqaFlag(sOfQuery, kDimNum, sOfHeadnum, layout) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetInOutLayoutAndProcessQInfo(const std::string layout, uint32_t& sOfQuery,
                                                             uint32_t& sOfHeadnum, const uint32_t kDimNum)
{
    bool prefixFlag = ifaContext_->keySharedPrefix.hasValue || ifaContext_->valueSharedPrefix.hasValue;
    if (layout == "BSH" || layout == "BSH_NBSD") {
        if (GetInOutLayout4BSH(layout, sOfQuery, sOfHeadnum, kDimNum, prefixFlag) !=
            custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    } else if (layout == "BSND" || layout == "BSND_NBSD") {
        inputLayout_ = IfaLayout::BSH_BSND;
        outputLayout_ = (layout == "BSND") ? inputLayout_ : IfaLayout::NBSD;
        sOfQuery = ifaContext_->query.shape[1];   // 1, dim of S
        headDim_ = ifaContext_->query.shape[3];   // 3, dim of D
        sOfHeadnum = ifaContext_->query.shape[2]; // 2, dim of N
    } else if (layout == "BNSD" || layout == "BNSD_NBSD") {
        inputLayout_ = IfaLayout::BNSD;
        outputLayout_ = (layout == "BNSD") ? inputLayout_ : IfaLayout::NBSD;
        sOfQuery = ifaContext_->query.shape[2];   // 2, dim of S
        headDim_ = ifaContext_->query.shape[3];   // 3, dim of D
        sOfHeadnum = ifaContext_->query.shape[1]; // 1, dim of N
    } else if (layout == "TND" || layout == "TND_NTD") {
        inputLayout_ = IfaLayout::TND;
        outputLayout_ = (layout == "TND") ? inputLayout_ : IfaLayout::NTD;
        sOfHeadnum = ifaContext_->query.shape[1]; // 2, dim of N
        headDim_ = ifaContext_->query.shape[2];   // 3, dim of D
        sOfQuery = ifaContext_->query.shape[0];   // 1, dim of T, S == 1, D = T / S = T
    } else {
        OP_LOGE(ifaContext_->opName, "Only support input_layout(BSH, BNSD, BSND, TND), actually is %s", layout.c_str());
        return custom::graphStatus::GRAPH_FAILED;
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetInOutLayout4BSH(const std::string layout, uint32_t& sOfQuery, uint32_t& sOfHeadnum,
                                                  const uint32_t kDimNum, bool prefixFlag)
{
    inputLayout_ = IfaLayout::BSH_BSND;
    outputLayout_ = (layout == "BSH") ? inputLayout_ : IfaLayout::NBSD;
    OP_CHECK_IF(ifaContext_->query.shape[2] % numHeads_ != 0U,
                OP_LOGE(ifaContext_->opName, "H should be an interger multiple of numHeads"),
                return custom::graphStatus::GRAPH_FAILED);
    sOfQuery = ifaContext_->query.shape[1];
    headDim_ = ifaContext_->query.shape[2] / numHeads_; // 2, QK dim of H
    int32_t tmpWindowSize = -1;
    tmpWindowSize = static_cast<int32_t>(ifaContext_->windowSize);
    slidingFlag_ = (layout == "BSH") && (sOfQuery == 1) && (tmpWindowSize > 0) && (ifaContext_->blockTable.hasValue) &&
                   (!prefixFlag) && (ifaContext_->value.shape.size() == DIM_BSH) &&
                   (!ifaContext_->queryRope.hasValue) && (inputQType_ == inputKvType_) &&
                   ((inputQType_ == at::ScalarType::BFloat16) || (inputQType_ == at::ScalarType::Half)) &&
                   (socVersion_ == IfaSocVersion::SOC_ASCEND_910B);
    if (slidingFlag_) {
        headDimV_ = ifaContext_->value.shape[2] / numKvHeads_; // 2, V dim of H
    }
    sOfHeadnum = numHeads_;
    if ((ifaContext_->kvHeadNums != 0U) && (kDimNum == 3U)) { // 3, dim of kv when the layout of kv is BSH
        sOfHeadnum = headDim_ * numHeads_ * numKvHeads_ / ifaContext_->key.shape[2]; // 2, dim of H
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetRopeAndGqaFlag(const uint32_t sOfQuery, const uint32_t kDimNum,
                                                 const uint32_t sOfHeadnum, const std::string layout)
{
    bool prefixFlag = ifaContext_->keySharedPrefix.hasValue || ifaContext_->valueSharedPrefix.hasValue;
    if (numHeads_ != sOfHeadnum && !prefixFlag) {
        OP_LOGE(ifaContext_->opName, "the query's heads num should be equal to qOfHeadnum");
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (!slidingFlag_) {
        headDimV_ = headDim_;
    }
    // 元素个数按照基本块大小对齐
    headDimAlign_ = Align(headDim_, BYTE_BLOCK);
    headDimVAlign_ = Align(headDimV_, BYTE_BLOCK);

    OP_CHECK_IF(
        (ifaContext_->queryRope.hasValue && !ifaContext_->keyRope.hasValue),
        OP_LOGE(ifaContext_->opName, "KeyRope is null, but queryRope exists, they should be both null or exist. "),
        return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(
        (!ifaContext_->queryRope.hasValue && ifaContext_->keyRope.hasValue),
        OP_LOGE(ifaContext_->opName, "QueryRope is null, but keyRope exists, they should be both null or exist. "),
        return custom::graphStatus::GRAPH_FAILED);

    if (ifaContext_->keyRope.hasValue && ifaContext_->queryRope.hasValue) {
        ropeFlag_ = true;
    }

    if (sOfQuery > 1U && sOfQuery <= 16U && !ropeFlag_) { // 投机推理场景，QS在1到16之间
        gqaMtpFlag_ = true;
    }
    if (kDimNum == 5U && !ropeFlag_) {
        gqaKvNZFlag_ = true;
    }
    if (!ropeFlag_ && !gqaMtpFlag_) {
        OP_CHECK_IF(sOfQuery != 1U,
                    OP_LOGE(ifaContext_->opName,
                            "In case where MLA is not applied, S of Query:%u is invalid. It should be in range [1, 16]",
                            sOfQuery),
                    return custom::graphStatus::GRAPH_FAILED);
    } else if (layout != "TND" && layout != "TND_NTD") {
        OP_CHECK_IF(sOfQuery > 16,
                    OP_LOGE(ifaContext_->opName, "QueryS(%u) should not be bigger than 16 in MLA.", sOfQuery),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    OP_CHECK_IF(layout == "TND" && headDim_ == 512 && !ropeFlag_,
                OP_LOGE(ifaContext_->opName, "When D is 512, inputlayout %s q_rope and k_rope should not be null!",
                        layout.c_str()),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::QKVPreProcess4TND(const std::string layout)
{
    auto qType = ifaContext_->query.dType;
    uint32_t qTypeSize = GetTypeSize(qType);
    if (qTypeSize == NUM_BYTES_UNDEF) {
        OP_LOGE(ifaContext_->opName, "Data type %s is not currently supported.", DataTypeToSerialString(qType).c_str());
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (isWorkspace_) { // TND+tiling下沉场景，不做校验
        tSeqSize_ = qSeqSize_ = ifaContext_->query.shape[0];
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                    OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
    } else {
        actualLenQDims_ = ifaContext_->actualSeqLengthsQ.GetShapeSize();
        actualLenDims_ = ifaContext_->actualSeqLengths.GetShapeSize();

        OP_CHECK_IF((actualLenQDims_ <= 0U),
                    OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths shape size(%u) should > 0",
                            layout.c_str(), actualLenQDims_),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF((actualLenQDims_ != actualLenDims_),
                    OP_LOGE(ifaContext_->opName,
                            "%s the query's actual sequence lengths shape size(%u) should equal the key/value's actual "
                            "sequence lengths shape size(%u)",
                            layout.c_str(), actualLenQDims_, actualLenDims_),
                    return custom::graphStatus::GRAPH_FAILED);

        batchSizeQ_ = batchSize_ = ifaContext_->actualSeqLengthsQ.GetShapeSize();
        tSeqSize_ = ifaContext_->query.shape[0];
        uint32_t tKVSeqSize = ifaContext_->key.shape[0];
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                    OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::InputAttrsPreProcess()
{
    innerPrecise_ = ifaContext_->innerPrecise; // 默认高性能
    OP_CHECK_IF(((innerPrecise_ != IFA_HIGH_PERFORMANCE) && (innerPrecise_ != IFA_HIGH_PRECISION)),
                OP_LOGE(ifaContext_->opName, "precision mode[%u] should be 0 or 1", innerPrecise_),
                return custom::graphStatus::GRAPH_FAILED); // 当前只支持高精度0和高性能1
    OP_LOGD(ifaContext_->opName, "innerPrecise is %u", innerPrecise_);

    blockTypeSize_ = sizeof(float); // 默认按照float计算
    pageAttentionFlag_ = ifaContext_->blockTable.hasValue;

    if (!pageAttentionFlag_) {
        uint32_t kvBatch = ifaContext_->key.shape[0];
        batchContinuousFlag_ = (batchSize_ == kvBatch);
    } else {
        uint32_t blockTableDim0 = ifaContext_->blockTable.shape[0];
        uint32_t blockTableDim1 = ifaContext_->blockTable.shape[1];
        OP_LOGI(ifaContext_->opName, "pageAttentionFlag_ is true. The shape of blockTable is [%u, %u].", blockTableDim0,
                blockTableDim1);
        OP_CHECK_IF(ifaContext_->blockTable.GetShapeSize() == 0,
                    OP_LOGE(ifaContext_->opName, "check blockTable shape failed, blockTable shapeSize is zero."),
                    return custom::graphStatus::GRAPH_FAILED);
        if (inputLayout_ == IfaLayout::TND && !isWorkspace_) {
            OP_CHECK_IF(
                blockTableDim0 != actualLenQDims_,
                OP_LOGE(
                    ifaContext_->opName,
                    "The actual sequence length dimension for Q[%u] in TND must match the B-axis of block table[%u].",
                    actualLenQDims_, blockTableDim0),
                return custom::graphStatus::GRAPH_FAILED);
        }
    }
    // achieve windowSize when sliding
    windowSize_ = static_cast<int32_t>(ifaContext_->windowSize);
    softmaxLseFlag_ = ifaContext_->softmaxLseFlag;

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SetQuantFlag()
{
    antiQuantFlag_ = (inputQType_ != inputKvType_) && (inputKvType_ == at::ScalarType::Char);
    if (antiQuantFlag_) {
        if (innerPrecise_ == IFA_HIGH_PRECISION) {
            msdIterNum_ = HIGH_PRECISION_ITER_NUM;
        } else {
            msdIterNum_ = ITER_NUM;
        }
    }

    // 全量化基本校验
    if (inputQType_ == at::ScalarType::Char) {
        OP_CHECK_IF(
            !pageAttentionFlag_ || inputKvLayout_ != IfaLayout::NZ,
            OP_LOGE(ifaContext_->opName,
                    "when the dtype of query is int8 in MLA, PageAttetion should be enabled and KV layout must be NZ"),
            return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessBaseInputs()
{
    if ((QKVPreProcess() != custom::graphStatus::GRAPH_SUCCESS) ||
        (InputAttrsPreProcess() != custom::graphStatus::GRAPH_SUCCESS) ||
        (KvShapePostProcess() != custom::graphStatus::GRAPH_SUCCESS) ||
        (SetL2CacheFlag() != custom::graphStatus::GRAPH_SUCCESS) ||
        (SetQuantFlag() != custom::graphStatus::GRAPH_SUCCESS) ||
        (InitInOutMode() != custom::graphStatus::GRAPH_SUCCESS)) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::SetupPerfMode()
{
    // 310P
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        if (EnableGQA310P()) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM;
        } else {
            perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
        }
    } else {
        if (gqaKvNZFlag_) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM_DD;
        } else if (ropeFlag_) {
            perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
        } else {
            if (EnableAllVec()) {
                perfMode_ = IfaPerfMode::BMM_ALL_BY_VEC;
            } else if (EnableCubeViewMM()) {
                perfMode_ = IfaPerfMode::CUBE_VIEW_MM;
            }
        }
    }
}

void IFATiling::UpdatePerfMode()
{
    if (EnableC1V1()) {
        perfMode_ = IfaPerfMode::C1_V1;
    }
}

void IFATiling::UpdateL2CacheOffFlag()
{
    if (perfMode_ != IfaPerfMode::CUBE_VIEW_MM_DD) {
        return;
    }
    if (inputKvLayout_ == IfaLayout::NZ || inputKvLayout_ == IfaLayout::BNSD) {
        l2CacheOffFlag_ = 1U;
        OP_LOGD(ifaContext_->opName, "KV layout is NZ or BNSD, set l2CacheOffFlag_ = 1.");
    } else if (inputKvLayout_ == IfaLayout::BSH_BSND) {
        if (numKvHeads_ == 1U) {
            l2CacheOffFlag_ = 1U;
            OP_LOGD(ifaContext_->opName,
                    "KV layout is BSH and the key/value's heads num is 1, set l2CacheOffFlag_ = 1.");
        }
    }
}

custom::graphStatus IFATiling::ProcessPageAttentionFlag()
{
    maxBlockNumPerBatch_ = ifaContext_->blockTable.shape[1];
    sMax_ = maxBlockNumPerBatch_ * blockSize_;
    seqSize_ = sMax_;
    uint32_t kDimNum = ifaContext_->key.shape.size();
    if (inputLayout_ == IfaLayout::TND) {
        if (kDimNum == 3U) { // BSH
            inputKvLayout_ = IfaLayout::BSH_BSND;
        } else if (kDimNum == 4U) { // BNSD
            inputKvLayout_ = IfaLayout::BNSD;
        } else if (kDimNum == 5U) { // NZ: block_num, N, D/12, block_size, 16
            inputKvLayout_ = IfaLayout::NZ;
        }
    } else {
        if (ropeFlag_ || gqaKvNZFlag_) {
            if (kDimNum == 3U) { // BSH
                inputKvLayout_ = IfaLayout::BSH_BSND;
            } else if (kDimNum == 4U) { // BNSD
                inputKvLayout_ = IfaLayout::BNSD;
            } else if (kDimNum == 5U) { // NZ: block_num, N, D/(32/sizeof(KV_T)), block_size, (32/sizeof(KV_T))
                inputKvLayout_ = IfaLayout::NZ;
            } else {
                OP_LOGE(ifaContext_->opName, "The dim of keyShape[%u] should be one of 3,4,5.", kDimNum);
                return custom::graphStatus::GRAPH_FAILED;
            }
        } else {
            if (kDimNum == 3U) { // BSH
                inputLayout_ = IfaLayout::BSH_BSND;
            } else { // BNSD
                inputLayout_ = IfaLayout::BNSD;
            }
        }
    }

    const std::string inputLayoutStr = ifaContext_->layOut;
    bool isPAinputLayoutStr = inputLayoutStr != "BNSD" && inputLayoutStr != "TND" && inputLayoutStr != "TND_NTD" &&
                              inputLayoutStr != "BNSD_NBSD" && ifaContext_->innerPrecise != ATB_INNER_PRECISE;
    if (inputQType_ == at::ScalarType::Char && inputKvType_ == at::ScalarType::Char && ifaContext_->keyRope.hasValue &&
        ifaContext_->queryRope.hasValue && headDim_ == 512) { // 512 : for MLA
        OP_CHECK_IF((kDimNum == DIM_BNSD),
                    OP_LOGE(ifaContext_->opName, "when the dtype of query is int8 in MLA, KV layout must be NZ"),
                    return custom::graphStatus::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((kDimNum == DIM_BNSD && isPAinputLayoutStr),
                    OP_LOGE(ifaContext_->opName, "when Page Attention scene, kvcache is BNBD, query layout must be "
                                                 "BNSD, BNSD_NBSD, TND, TND_NTD or BNSD_BSND"),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::KvShapePostProcess()
{
    if (pageAttentionFlag_) {
        return ProcessPageAttentionFlag();
    }

    auto batchOfQuery = ifaContext_->query.shape[0];
    auto batchOfKey = ifaContext_->key.shape[0];

    uint32_t tmpSeqSize = 0U;
    for (size_t i = 0U; i < ifaContext_->kCache.size(); i++) {
        auto keyShape = ifaContext_->kCache[i];
        auto valueShape = ifaContext_->vCache[i];

        if ((keyShape.empty()) || (valueShape.empty())) {
            OP_LOGE(ifaContext_->opName,
                    "kv tensor list length should be greater than or equal to q batch, "
                    "kv tensor list index[%lu] is null, q batch: %ld",
                    i, batchOfQuery);
            return custom::graphStatus::GRAPH_FAILED;
        }

        if (!ShapeEqual(keyShape, valueShape)) {
            OP_LOGE(ifaContext_->opName, "k v shape shoud be same ");
            return custom::graphStatus::GRAPH_FAILED;
        }

        uint32_t seqSize;
        if (inputLayout_ == IfaLayout::BSH_BSND) {
            seqSize = keyShape[1];
        } else {
            seqSize = keyShape[2]; // 2, dim of S
        }

        /* 原则上空tensor为S=0，兼容ShapeSize=0场景 */
        if (seqSize != 0U && GetShapeSize(keyShape) == 0) {
            seqSize = 0U;
        }

        if (seqSize == 0U) {
            hasZeroSeqKVTensor_ = true;
        }
        if (i == 0U) {
            tmpSeqSize = seqSize;
        } else {
            if (seqSize != tmpSeqSize) {
                isSameSeqAllKVTensor_ = false;
            }
        }

        sMax_ = std::max(seqSize, sMax_);
        kvListSeqLens_.push_back(seqSize);
    }
    seqSize_ = sMax_;
    inputKvLayout_ = inputLayout_;
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ZeroTensorProcess()
{
    if (sMax_ == 0U) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        seqSize_ = 1024U;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::InitInOutMode()
{
    if (inputQType_ == at::ScalarType::Char && outputType_ == at::ScalarType::Char) {
        inOutMode_ = TilingInOutMode::INT8_INT8;
    } else if (inputQType_ == at::ScalarType::Char && outputType_ == at::ScalarType::Half) {
        inOutMode_ = TilingInOutMode::INT8_FP16;
    } else if (inputQType_ == at::ScalarType::Half && outputType_ == at::ScalarType::Char) {
        inOutMode_ = TilingInOutMode::FP16_INT8;
    } else if (inputQType_ == at::ScalarType::Half && outputType_ == at::ScalarType::Half) {
        inOutMode_ = TilingInOutMode::FP16_FP16;
    } else if (inputQType_ == at::ScalarType::BFloat16 && outputType_ == at::ScalarType::BFloat16) {
        inOutMode_ = TilingInOutMode::BF16_BF16;
    } else if (inputQType_ == at::ScalarType::BFloat16 && outputType_ == at::ScalarType::Char) {
        inOutMode_ = TilingInOutMode::BF16_INT8;
    } else if (inputQType_ == at::ScalarType::Float && outputType_ == at::ScalarType::Float) {
        inOutMode_ = TilingInOutMode::FP32_FP32;
    } else if (inputQType_ == at::ScalarType::Char && outputType_ == at::ScalarType::Half) {
        inOutMode_ = TilingInOutMode::INT8_BF16;
    } else {
        OP_LOGE(ifaContext_->opName, "input dtype %d with output dtype %d is not currently supported.", inputQType_,
                outputType_);
        return custom::graphStatus::GRAPH_FAILED;
    }
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (inOutMode_ != TilingInOutMode::FP16_FP16)) {
        OP_LOGE(ifaContext_->opName,
                "input dtype float16 with output dtype float16 is currently supported when 310P, but "
                "current input dtype is %d and output dtype is %d",
                inputQType_, outputType_);
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessOptionalTensors()
{
    if ((ProcessActualSeqLen() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessPseShift() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessQuant1() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessQuant2() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessDequant1() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessDequant2() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessQuant() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessAntiQuant() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessAttenMask() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessBlockTable() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessKVPaddingSize() != custom::graphStatus::GRAPH_SUCCESS) ||
        // (ProcessMlaRope() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessCvRatio() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessGqaKvNz() != custom::graphStatus::GRAPH_SUCCESS)) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    // for kv shared prefix
    if ((ProcessSharedPrefix() != custom::graphStatus::GRAPH_SUCCESS) ||
        (ProcessSharedPrefixLen() != custom::graphStatus::GRAPH_SUCCESS)) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessActualSeqLen()
{
    if (inputLayout_ == IfaLayout::TND) {
        if (isWorkspace_) {
            actualSeqLenFlag_ = true;
            maxActualseq_ = sMax_;
            return custom::graphStatus::GRAPH_SUCCESS;
        }

        if (CheckActualSeqLens() != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }

        actualSeqLenFlag_ = true;
        maxActualseq_ = sMax_;
    } else {
        if (!ifaContext_->actualSeqLengths.hasValue) {
            OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is nullptr");
            maxActualseq_ = sMax_;
            // pa场景必须带actual_seq_lens；第1次tiling调用时(isWorkspace为true)
            // actualSeqLengths会被强制置None，需要跳过校验
            OP_LOGD(ifaContext_->opName, "isWorkspace: %d", isWorkspace_);
            if (pageAttentionFlag_ && (!isWorkspace_)) {
                OP_LOGE(ifaContext_->opName,
                        "the key/value's actual sequence lengths is null, it must exist in pageAttention scene");
                return custom::graphStatus::GRAPH_FAILED;
            }
            return custom::graphStatus::GRAPH_SUCCESS;
        }
        OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is not nullptr");

        actualSeqLenFlag_ = true;
        actualLenDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengths.GetShapeSize());
        OP_LOGD(ifaContext_->opName, "number of elements in the key/value's actual sequence lengths is %u",
                actualLenDims_);
        if (actualLenDims_ == 0U) {
            // pa场景必须带actual_seq_lens
            if (pageAttentionFlag_) {
                OP_LOGW(ifaContext_->opName,
                        "the key/value's actual sequence lengths size[%u] can not be zero in pageAttention scene",
                        actualLenDims_);
            }
            maxActualseq_ = sMax_;
            return custom::graphStatus::GRAPH_SUCCESS;
        }

        OP_CHECK_IF(
            actualLenDims_ != 1U && actualLenDims_ < batchSize_,
            OP_LOGE(
                ifaContext_->opName,
                "the key/value's actual sequence lengths size[%u] should be greater than q batch[%u] or equal to 1.",
                actualLenDims_, batchSize_),
            return custom::graphStatus::GRAPH_FAILED);

        actualLenDims_ = std::min(actualLenDims_, batchSize_);
    }

    if (ParseActualSeqLens() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckActualSeqLens()
{
    // Q的actual_seq要求非空
    if (!ifaContext_->actualSeqLengthsQ.hasValue) {
        OP_LOGE(ifaContext_->opName, "TND the query's actual sequence lengths should not be null!");
        return custom::graphStatus::GRAPH_FAILED;
    }
    actualLenQDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengthsQ.GetShapeSize());
    if (actualLenQDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenQDims_ is 0!");
        return custom::graphStatus::GRAPH_FAILED;
    }

    // KV的actual_seq要求非空
    if (!ifaContext_->actualSeqLengths.hasValue) {
        OP_LOGE(ifaContext_->opName, "TND the key/value's actual sequence lengths should not be null!");
        return custom::graphStatus::GRAPH_FAILED;
    }
    actualLenDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengths.GetShapeSize());
    if (actualLenDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenDims_ is 0!");
        return custom::graphStatus::GRAPH_FAILED;
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ParseActualSeqLens()
{
    OP_LOGD(ifaContext_->opName, "data of the key/value's actual sequence lengths is nullptr");
    // pa场景必须带actual_seq_lens
    if (pageAttentionFlag_ && (!isWorkspace_)) {
        OP_LOGW(ifaContext_->opName,
                "data of the key/value's actual sequence lengths can not be nullptr in pageAttention scene");
    }
    maxActualseq_ = sMax_;

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessQuant1() const
{
    if (inputQType_ != at::ScalarType::Char) {
        OP_CHECK_IF(ifaContext_->deqScale1.hasValue,
                    OP_LOGE(ifaContext_->opName, "when input type is not int8, dqtScale1 should be null"),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantScale1.hasValue,
                    OP_LOGE(ifaContext_->opName, "when input type is not int8, qtScale1 should be null"),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->deqScale2.hasValue,
                    OP_LOGE(ifaContext_->opName, "when input type is not int8, dqtScale2 should be null"),
                    return custom::graphStatus::GRAPH_FAILED);
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessQuant()
{
    if (inputQType_ != at::ScalarType::Char) {
        OP_CHECK_IF(ifaContext_->dequantScaleQuery.hasValue,
                    OP_LOGE(ifaContext_->opName,
                            "when the dtype of query is not int8, the query's dequant scale should be null"),
                    return custom::graphStatus::GRAPH_FAILED);
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        !ropeFlag_,
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, query_rope and key rope should not be null"),
        return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputLayout_ == IfaLayout::BNSD),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8 in MLA, layout BNSD/BNSD_NBSD is not support"),
        return custom::graphStatus::GRAPH_FAILED);

    // 全量化暂不支持quantScale1/deqScale1/deqScale2
    OP_CHECK_IF(
        (ifaContext_->quantScale1.hasValue || ifaContext_->deqScale1.hasValue || ifaContext_->deqScale2.hasValue),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, quantScale1/deqScale1/deqScale2 should be null"),
        return custom::graphStatus::GRAPH_FAILED);

    // 全量化暂不支持atiquantScale/antiquantOffset
    OP_CHECK_IF((ifaContext_->antiquantScale.hasValue || ifaContext_->antiquantOffset.hasValue),
                OP_LOGE(ifaContext_->opName,
                        "when the dtype of query is int8 in MLA, antiquantScale/antiquantOffset should be null"),
                return custom::graphStatus::GRAPH_FAILED);

    // 全量化暂不支持keyAntiquantOffset/valueAntiquantOffset
    OP_CHECK_IF((ifaContext_->keyAntiquantOffset.hasValue || ifaContext_->valueAntiquantOffset.hasValue),
                OP_LOGE(ifaContext_->opName,
                        "when the dtype of query is int8 in MLA, the key's/value's dequant offset should be null"),
                return custom::graphStatus::GRAPH_FAILED);

    OP_CHECK_IF((!ifaContext_->queryRope.hasValue || !ifaContext_->keyRope.hasValue),
                OP_LOGE(ifaContext_->opName,
                        "when the dtype of query is int8, query_rope and key rope desc should not be null"),
                return custom::graphStatus::GRAPH_FAILED);
    OP_CHECK_IF(
        (ifaContext_->queryRope.dType != at::ScalarType::BFloat16 ||
         ifaContext_->keyRope.dType != at::ScalarType::BFloat16),
        OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, query_rope and key rope dtype should be bf16"),
        return custom::graphStatus::GRAPH_FAILED);

    quantFlag_ = true;

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessQuant2Dtype()
{
    if (outputType_ == at::ScalarType::Char) {
        OP_CHECK_IF(!ifaContext_->quantScale2.hasValue,
                    OP_LOGE(ifaContext_->opName,
                            "output data type is int8, but input tensor of the output's dequant scale is null"),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(!ifaContext_->quantScale2.hasValue,
                    OP_LOGE(ifaContext_->opName, "Desc of the output's dequant scale input tensor is null."),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantScale2.dType != at::ScalarType::BFloat16 &&
                        ifaContext_->quantScale2.dType != at::ScalarType::Float,
                    OP_LOGE(ifaContext_->opName, "the output's dequant scale type(%d) should be bf16 or fp32",
                            ifaContext_->quantScale2.dType),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            ifaContext_->quantOffset2.hasValue && ifaContext_->quantScale2.dType != ifaContext_->quantOffset2.dType,
            OP_LOGE(ifaContext_->opName, "the output's dequant scale dtype(%d) and offset dtype(%d) are not the same",
                    ifaContext_->quantScale2.dType, ifaContext_->quantOffset2.dType),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            inputQType_ != at::ScalarType::BFloat16 && ifaContext_->quantScale2.dType == at::ScalarType::BFloat16,
            OP_LOGE(ifaContext_->opName, "the output's dequant scale and offset support bf16 when inputQ type is bf16"),
            return custom::graphStatus::GRAPH_FAILED);
        if (ifaContext_->quantScale2.dType == at::ScalarType::BFloat16) {
            isOutQuantTypeBf16_ = true;
        }
    } else {
        OP_CHECK_IF(ifaContext_->quantScale2.hasValue,
                    OP_LOGE(ifaContext_->opName,
                            "the output's dequant scale exist, output data type should be INT8, but now it's %s",
                            DataTypeToSerialString(outputType_).c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->quantOffset2.hasValue,
                    OP_LOGE(ifaContext_->opName,
                            "the output's dequant offset exist, output data type should be INT8, but now it's %s",
                            DataTypeToSerialString(outputType_).c_str()),
                    return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessQuant2()
{
    auto& qtScale2 = ifaContext_->quantScale2;
    auto& qtOffset2 = ifaContext_->quantOffset2;

    if (ProcessQuant2Dtype() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (outputType_ == at::ScalarType::Char) {
        if (qtScale2.GetShapeSize() == 1) {
            OP_LOGD(ifaContext_->opName, "the output's dequant scale is a const value.");
        } else {
            OP_LOGD(ifaContext_->opName, "the output's dequant scale is a tensor.");
            isOutQuantPerChnOut_ = true;
        }

        // for offset optional
        if (qtOffset2.hasValue && qtScale2.hasValue) {
            if (qtScale2.dType != qtOffset2.dType) {
                OP_LOGE(ifaContext_->opName, "the output's dequant scale and offset should have the same data type.");
                return custom::graphStatus::GRAPH_FAILED;
            }
            if (qtOffset2.GetShapeSize() == 1) {
                OP_LOGD(ifaContext_->opName, "the output's dequant offset is a const value.");
            } else {
                OP_LOGD(ifaContext_->opName, "the output's dequant offset is a tensor.");
                isOutQuantPerChnOut_ = true;
            }
        }
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessDequant1() const
{
    if (!ifaContext_->deqScale1.hasValue) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessDequant2() const
{
    if (!ifaContext_->deqScale2.hasValue) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKVAntiQuantParamsShapeInPagedAttention(const at::IntArrayRef& inputParaShape) const
{
    if (antiquantPerHeadFlag_ != 0U) { // per-token-head, [block_num, kv_head_num, block_size]
        OP_CHECK_IF(
            (inputParaShape[0] != totalBlockNum_),
            OP_LOGE(ifaContext_->opName, "The 1st dim of antiquant parameter should be %u instead of the current %ld",
                    totalBlockNum_, inputParaShape[0]),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape[1] != numKvHeads_),
            OP_LOGE(ifaContext_->opName, "The 2nd dim of antiquant parameter should be %u instead of the current %ld",
                    numKvHeads_, inputParaShape[1]),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape[2] != blockSize_),
            OP_LOGE(ifaContext_->opName, "The 3rd dim of antiquant parameter should be %u instead of the current %ld",
                    blockSize_, inputParaShape[2]),
            return custom::graphStatus::GRAPH_FAILED);
    } else { // per-token, [block_num, block_size]
        OP_CHECK_IF(
            (inputParaShape[0] != totalBlockNum_),
            OP_LOGE(ifaContext_->opName, "The 1st dim of antiquant parameter should be %u instead of the current %ld",
                    totalBlockNum_, inputParaShape[0]),
            return custom::graphStatus::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape[1] != blockSize_),
            OP_LOGE(ifaContext_->opName, "The 2nd dim of antiquant parameter should be %u instead of the current %ld",
                    blockSize_, inputParaShape[1]),
            return custom::graphStatus::GRAPH_FAILED);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKVAntiQuantParamsInPagedAttention() const
{
    auto& KeyAntiquantScaleShape = ifaContext_->keyAntiquantScale.shape;
    if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantScaleShape) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    auto& keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset;
    if (keyAntiquantOffsetTensor.hasValue) {
        auto KeyAntiquantOffsetShape = keyAntiquantOffsetTensor.shape;
        if (CheckKVAntiQuantParamsShapeInPagedAttention(KeyAntiquantOffsetShape) !=
            custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKVAntiQuantMode()
{
    if (gqaKvNZFlag_ && (antiquantMode_ != DEQUANT_PER_CHANNEL_MODE) && (antiquantMode_ != DEQUANT_PER_TOKEN_MODE)) {
        OP_LOGE(ifaContext_->opName, "antiquantMode value[%u] should be 0 or 1 in GQA KV NZ", antiquantMode_);
        return custom::graphStatus::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND && antiquantMode_ == DEQUANT_PER_TOKEN_MODE,
        OP_LOGE(ifaContext_->opName, "Per token antiquant mode is not supported when layout is TND in GQA KV NZ."),
        return custom::graphStatus::GRAPH_FAILED);
    if ((antiquantMode_ != DEQUANT_PER_CHANNEL_MODE) && (antiquantMode_ != DEQUANT_PER_TOKEN_MODE) &&
        (antiquantMode_ != DEQUANT_PER_TENSOR_HEAD_MODE) && (antiquantMode_ != DEQUANT_PER_TOKEN_HEAD_MODE) &&
        (antiquantMode_ != DEQUANT_PER_TOKEN_PA_MODE) && (antiquantMode_ != DEQUANT_PER_TOKEN_HEAD_PA_MODE)) {
        OP_LOGE(ifaContext_->opName, "antiquantMode value:%u is invalid, it should be 0、1、2、3、4 or 5",
                antiquantMode_);
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (antiquantMode_ == DEQUANT_PER_TENSOR_HEAD_MODE) { // 2:per tensor head
        antiquantMode_ = PER_CHANNEL_MODE;
        antiquantPerHeadFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_HEAD_MODE) { // 3:per token head
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantPerHeadFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_PA_MODE) { // 4:per token + pageAttention scale/offset
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantParamsInPagedAttentionFlag_ = 1U;
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_HEAD_PA_MODE) { // 5:per token head + pageAttention scale/offset
        antiquantMode_ = PER_TOKEN_MODE;
        antiquantPerHeadFlag_ = 1U;
        antiquantParamsInPagedAttentionFlag_ = 1U;
    }
    OP_CHECK_IF((antiquantParamsInPagedAttentionFlag_ != 0U) && !pageAttentionFlag_,
                OP_LOGE(ifaContext_->opName, "the key/value's quant mode 4 and 5 use page attention to manage "
                                             "scale/offset, must be used in page attention scene"),
                return custom::graphStatus::GRAPH_FAILED);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessAntiQuant()
{
    auto& antiquantScaleTensor = ifaContext_->antiquantScale;
    auto& antiquantOffsetTensor = ifaContext_->antiquantOffset;
    auto& keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset;
    auto& keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale;
    auto& valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale;
    auto& valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset;
    auto& keyRopeAntiquantScaleTensor = ifaContext_->keyRopeAntiquantScale;

    if ((!antiQuantFlag_ &&
         (antiquantScaleTensor.hasValue || antiquantOffsetTensor.hasValue || keyAntiquantScaleTensor.hasValue ||
          keyAntiquantOffsetTensor.hasValue || valueAntiquantScaleTensor.hasValue ||
          valueAntiquantOffsetTensor.hasValue || keyRopeAntiquantScaleTensor.hasValue)) &&
        !quantFlag_) {
        OP_LOGE(ifaContext_->opName, "KV antiquant is unenabled, but antiquant antiquantScale/antiquantOffset exist");
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (!antiQuantFlag_) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    if (CheckKeyAndValueAntiquantScaleOffset() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    uint32_t keyAntiquantMode_kvSep = static_cast<uint32_t>(ifaContext_->keyAntiquantMode);
    uint32_t valueAntiquantMode_kvSep = static_cast<uint32_t>(ifaContext_->valueAntiquantMode);
    if (CheckKeyAndValueAntiquantOffset(keyAntiquantMode_kvSep, valueAntiquantMode_kvSep) !=
        custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (kvAntiParamSplitFlag_) {
        if (CheckKvAntiquant4SplitMode() != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }

        if (ProcessAntiQuantMode() != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    } else {
        OP_LOGD(ifaContext_->opName, "kv antiquant is not split mode");
        antiquantMode_ = static_cast<uint32_t>(ifaContext_->antiquantMode);
    }
    antiqSeqSize_ = GetAntiquantSeqLength();
    OP_LOGD(ifaContext_->opName, "antiquant info, iter num:%u, antiquant mode:%u", msdIterNum_, antiquantMode_);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKeyAndValueAntiquantScaleOffset()
{
    auto& keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale;
    auto& keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset;
    auto& valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale;
    auto& valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset;
    auto& keyRopeAntiquantScaleTensor = ifaContext_->keyRopeAntiquantScale;

    kvAntiParamSplitFlag_ = false;
    if (keyAntiquantScaleTensor.hasValue && !valueAntiquantScaleTensor.hasValue) {
        OP_LOGE(ifaContext_->opName, "the value's dequant scale is null, but the key's dequant scale exist");
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (valueAntiquantScaleTensor.hasValue && !keyAntiquantScaleTensor.hasValue) {
        OP_LOGE(ifaContext_->opName, "the key's dequant scale is null, but the value's dequant scale exist");
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (keyAntiquantOffsetTensor.hasValue && !valueAntiquantOffsetTensor.hasValue) {
        OP_LOGE(ifaContext_->opName, "value's dequant offset is null, but the key's dequant offset exist");
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (valueAntiquantOffsetTensor.hasValue && !keyAntiquantOffsetTensor.hasValue) {
        OP_LOGE(ifaContext_->opName, "the key's dequant offset is null, but the value's dequant offset exist");
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (!keyAntiquantScaleTensor.hasValue && keyAntiquantOffsetTensor.hasValue) {
        OP_LOGE(ifaContext_->opName, "the key's dequant scale is null, but the key's dequant offset exist");
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (ropeFlag_) {
        if (keyAntiquantScaleTensor.hasValue && !keyRopeAntiquantScaleTensor.hasValue) {
            OP_LOGE(ifaContext_->opName, "Mla mode: the tensor of the key_rope's dequant scale is null, but the tensor "
                                         "of the key's dequant scale exist");
            return custom::graphStatus::GRAPH_FAILED;
        }
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKeyAndValueAntiquantOffset(const uint32_t keyAntiquantModeKvSep,
                                                               const uint32_t valueAntiquantModeKvSep)
{
    auto& keyAntiquantOffsetTensor = ifaContext_->keyAntiquantOffset;
    auto& valueAntiquantOffsetTensor = ifaContext_->valueAntiquantOffset;
    auto& keyAntiquantScaleTensor = ifaContext_->keyAntiquantScale;
    auto& valueAntiquantScaleTensor = ifaContext_->valueAntiquantScale;

    if (keyAntiquantOffsetTensor.hasValue && valueAntiquantOffsetTensor.hasValue) {
        if (keyAntiquantModeKvSep != 0U || valueAntiquantModeKvSep != 1U) {
            OP_CHECK_IF(
                (keyAntiquantOffsetTensor.dType != valueAntiquantOffsetTensor.dType),
                OP_LOGE(ifaContext_->opName,
                        "the description of the key's and the value's dequant offset should have the same data type"),
                return custom::graphStatus::GRAPH_FAILED);
            if (!ShapeEqual(keyAntiquantOffsetTensor.shape, valueAntiquantOffsetTensor.shape)) {
                OP_LOGE(ifaContext_->opName,
                        "the tensor of the key's and the value's dequant offset should have the same shape");
                return custom::graphStatus::GRAPH_FAILED;
            }
        }
    }

    if (keyAntiquantScaleTensor.hasValue && valueAntiquantScaleTensor.hasValue) {
        if (keyAntiquantModeKvSep != 0U || valueAntiquantModeKvSep != 1U) {
            if (!ShapeEqual(keyAntiquantScaleTensor.shape, valueAntiquantScaleTensor.shape)) {
                OP_LOGE(ifaContext_->opName,
                        "The tensor of the key's and the value's dequant scale should have the same shape");
                return custom::graphStatus::GRAPH_FAILED;
            }
        }
        kvAntiParamSplitFlag_ = true;
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CheckKvAntiquant4SplitMode() const
{
    OP_LOGD(ifaContext_->opName, "kv antiquant is split mode");
    uint32_t keyAntiquantMode = static_cast<uint32_t>(ifaContext_->keyAntiquantMode);
    uint32_t valueAntiquantMode = static_cast<uint32_t>(ifaContext_->valueAntiquantMode);
    if (keyAntiquantMode != valueAntiquantMode) {
        if (keyAntiquantMode != 0U || valueAntiquantMode != 1U) {
            OP_LOGE(ifaContext_->opName, "the key's quant mode and the value's quant mode should be the same");
            return custom::graphStatus::GRAPH_FAILED;
        }
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessAntiQuantMode()
{
    uint32_t keyAntiquantMode = static_cast<uint32_t>(ifaContext_->keyAntiquantMode);
    uint32_t valueAntiquantMode = static_cast<uint32_t>(ifaContext_->valueAntiquantMode);

    if (keyAntiquantMode == 0U && valueAntiquantMode == 1U) {
        antiquantMode_ = PER_CHANNEL_TOKEN_MODE;
    } else {
        antiquantMode_ = keyAntiquantMode;
        OP_LOGD(ifaContext_->opName, "org antiquantMode value:%u", antiquantMode_);
        if (CheckKVAntiQuantMode() != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessBlockTable()
{
    if (!pageAttentionFlag_) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    // gm到l1，copynd2nz的srcDValue最大支持65535
    if (inputLayout_ == IfaLayout::BSH_BSND) { // 0: BSH
        if ((numKvHeads_ * headDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) {
            OP_LOGE(ifaContext_->opName,
                    "When input kvcache layout is BSH, the N * D of kvcache is %u, "
                    "exceeds the maximum limit (%u) of the datacopy instruction.",
                    numKvHeads_ * headDim_, COPYND2NZ_SRC_STRIDE_LIMITATION);
            return custom::graphStatus::GRAPH_FAILED;
        }

        if (slidingFlag_ && (numKvHeads_ * headDimV_ > COPYND2NZ_SRC_STRIDE_LIMITATION)) {
            OP_LOGE(ifaContext_->opName,
                    "When input kvcache layout is BSH, the N * D of vcache is %u, "
                    "exceeds the maximum limit (%u) of the datacopy instruction.",
                    numKvHeads_ * headDimV_, COPYND2NZ_SRC_STRIDE_LIMITATION);
            return custom::graphStatus::GRAPH_FAILED;
        }
    }

    if (CheckPABlockSize() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    totalBlockNum_ = ifaContext_->kCache[0][0];
    OP_CHECK_IF(maxActualseq_ > blockSize_ * maxBlockNumPerBatch_,
                OP_LOGE(ifaContext_->opName,
                        "Invalid actual seq length for PA, max actual seq length(%u) "
                        "is larger than blocksize(%u) * max block num per batch(%u)",
                        maxActualseq_, blockSize_, maxBlockNumPerBatch_),
                return custom::graphStatus::GRAPH_FAILED);

    if (antiquantParamsInPagedAttentionFlag_ != 0U) {
        // 在处理pa相关信息时，才能获取到totalBlockNum_用于scale/offset校验
        if (CheckKVAntiQuantParamsInPagedAttention() != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    }

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessKVPaddingSize()
{
    auto& kvPaddingSize = ifaContext_->kvPaddingSize;
    if (!kvPaddingSize.hasValue) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor is nullptr: %d",
                !ifaContext_->kvPaddingSize.hasValue);
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    if (kvPaddingSize.GetShapeSize() == 0) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition: kvPaddingSize.tensor shape is empty: %d",
                kvPaddingSize.GetShapeSize() == 0);
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    custom::graphStatus ret = CheckSupportKVLeftPadding();

    return ret;
}

custom::graphStatus IFATiling::ProcessSharedPrefix()
{
    if (!ifaContext_->keySharedPrefix.hasValue && !ifaContext_->valueSharedPrefix.hasValue) {
        sysPrefixFlag_ = false;
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    if (SharedPrefixCheckBasic() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    auto keyShape = ifaContext_->keySharedPrefix.shape;
    auto valueShape = ifaContext_->valueSharedPrefix.shape;
    if (SharedPrefixCheckShapes(keyShape, valueShape) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        sMaxPrefix_ = keyShape[1]; // 1:BSH的S维
    } else {
        sMaxPrefix_ = keyShape[2]; // 2:BNSD的S维
    }

    if (ifaContext_->keySharedPrefix.GetShapeSize() == 0) { // 兼容空tensor场景
        sMaxPrefix_ = 0U;
    }

    sysPrefixFlag_ = true;

    return custom::graphStatus::GRAPH_SUCCESS;
}

uint32_t IFATiling::GetAntiquantSeqLength() const
{
    if (antiquantParamsInPagedAttentionFlag_ != 0U) {
        return seqSize_;
    }
    if (kvAntiParamSplitFlag_ && ifaContext_->valueAntiquantScale.shape.size() != DIM_NUM_THREE) {
        OP_LOGW(ifaContext_->opName,
                "the pertoken antiquant's dimensions is [%u], which neeb to be (1, Batch, AntiquantSeqlen)",
                ifaContext_->valueAntiquantScale.shape.size());
    }
    const size_t antiquantSIdx =
        (gqaKvNZFlag_) || (kvAntiParamSplitFlag_ && ifaContext_->valueAntiquantScale.shape.size() == DIM_NUM_TWO) ? 1U :
                                                                                                                    2U;
    return kvAntiParamSplitFlag_ ? ifaContext_->valueAntiquantScale.shape[antiquantSIdx] :
                                   ifaContext_->antiquantScale.shape[antiquantSIdx];
}

custom::graphStatus IFATiling::ProcessSharedPrefixLen()
{
    auto tensor = ifaContext_->actualSharedPrefixLen;
    if (!tensor.hasValue || tensor.GetShapeSize() == 0 || !sysPrefixFlag_) {
        maxActualPrefixLen_ = sMaxPrefix_;
        return custom::graphStatus::GRAPH_SUCCESS;
    }

    maxActualPrefixLen_ = sMaxPrefix_;
    auto actulLenShape = ifaContext_->actualSharedPrefixLen.shape;

    OP_CHECK_IF((actulLenShape.size() != 1U || actulLenShape[0] != 1U),
                OP_LOGE(ifaContext_->opName, "actual shared prefix shape[%lu] must be 1", actulLenShape.size()),
                return custom::graphStatus::GRAPH_FAILED);

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessCvRatio()
{
    // CV1:1 只支持MLA 全量化和非量化
    if ((cvRatio_ == 1) && (!quantFlag_ || !ropeFlag_)) {
        OP_LOGE(ifaContext_->opName, "when CV 1:1, only support MLA non-quantization(QKV type both are FP16 or BF16) "
                                     "and MLA fully quantization(QKV type both are int8)");
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::ProcessMlaRope()
{
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::Split()
{
    amlaMode_ = GetAmlaMode();
    if (balanceModeFlag_) {
        if (isWorkspace_) {
            CalcInnerSize(seqSize_);
            OP_LOGI(ifaContext_->opName, "aicNum_(use for usedCoreNum_ and kvSplitPart_):%u\n", aicNum_);
            usedCoreNum_ = aicNum_;
            splitKVFlag_ = true;
            kvSplit_++;
            kvSplitPart_ = aicNum_;
            return custom::graphStatus::GRAPH_SUCCESS;
        } else {
            return SplitBalanced();
        }
    } else {
        return SplitUnbalanced();
    }
}

custom::graphStatus IFATiling::ProcessGqaKvNz() const
{
    if (!gqaKvNZFlag_) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    if (CheckGqaTensor() != custom::graphStatus::GRAPH_SUCCESS ||
        CheckGqaAttribute() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::GetSeqTilingInfo(const at::IntArrayRef& actualSeqKv, const ActualSeqInfo& actualSeqInfo,
                                 SeqTilingInfo& seqTilingInfo) const
{
    uint32_t bSize = batchSize_;
    uint32_t souter = s1SplitSize_;
    uint32_t s2BasicSize = sInnerSize_;
    uint32_t n2Size = numKvHeads_;
    uint64_t totalS2Length = 0U;
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        uint64_t actualSeqLen = static_cast<uint64_t>(actualSeqKv[0]);
        if (actualLenDims_ != 1U) {
            actualSeqLen = static_cast<uint64_t>(actualSeqKv[bIdx]);
        }
        if (actualSeqInfo.maxActualseqkv <= 0) { // kv actseq全为0
            seqTilingInfo.s1OuterNum[bIdx] = 1U; // kv actseq全为0时，s1不切
            seqTilingInfo.s2OuterNum[bIdx] = 1U; // kv actseq全为0时，每个batch s2份数强制为1
        } else {
            seqTilingInfo.s1OuterNum[bIdx] = (s1 + (souter - 1U)) / souter; // s1总共切多少分，即线段个数
            seqTilingInfo.s2OuterNum[bIdx] =
                static_cast<uint32_t>((actualSeqLen + s2BasicSize - 1U) / s2BasicSize); // 线段长度
        }

        uint32_t bTotalS2Length = seqTilingInfo.s1OuterNum[bIdx] * seqTilingInfo.s2OuterNum[bIdx] *
                                  n2Size; // 线段个数*线段长度*N2=当前batch线段总长度

        totalS2Length += bTotalS2Length; // 更新一次总长度
        OP_LOGI(ifaContext_->opName, "s1OuterNum[%u]:%u, s2OuterNum[%u]:%u\n", bIdx, seqTilingInfo.s1OuterNum[bIdx],
                bIdx, seqTilingInfo.s2OuterNum[bIdx]);
        if ((seqTilingInfo.s1OuterNum[bIdx] > 0U) && (seqTilingInfo.s2OuterNum[bIdx] > 0U)) {
            seqTilingInfo.lastValidBIdx = bIdx;
        }
    }
    if (totalS2Length > coreNum_) {
        seqTilingInfo.avgS2Length = (totalS2Length + coreNum_ - 1U) / coreNum_; // 平均长度向上取整
    }
    OP_LOGI(ifaContext_->opName, "totalS2Length:%lu, avgS2Length:%lu, coreNum_:%u\n", totalS2Length,
            seqTilingInfo.avgS2Length, coreNum_);
}

void IFATiling::FillBalancedSplitCoreInfo(const TilingIndexes& tilingIdx, BalancedSplitTilingInfo& tilingInfo)
{
    tilingInfo.coreLoad[tilingInfo.currCoreIdx] = tilingInfo.accumS2Length;
    tilingInfo.currCoreIdx += 1U;
}

void IFATiling::EndSplitForCurrentCore(const TilingIndexes& tilingIdx, const SeqTilingInfo& seqTilingInfo,
                                       uint32_t& currKvSplitPart, BalancedSplitTilingInfo& tilingInfo)
{
    tilingInfo.accumS2Length += 1U;
    // 更新当前核的End分核信息
    FillBalancedSplitCoreInfo(tilingIdx, tilingInfo);
    if (tilingIdx.s2Idx <
        seqTilingInfo.s2OuterNum[tilingIdx.bIdx] - 1U) { // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
        currKvSplitPart += 1U;
    }
    tilingInfo.needUpdate = false;
}

custom::graphStatus IFATiling::SplitBalanced()
{
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SplitUnbalanced()
{
    if (amlaMode_ == IfaAmlaMode::AMLA_3BUF) {
        gMax_ = 512U; // 3buf场景下切512
    }
    if (gqaMtpFlag_) {
        gMax_ = antiQuantFlag_ ? 128U : 256U; // 伪量化场景下，g切32，非伪量化场景切64
    }

    if (ropeFlag_ || gqaMtpFlag_) {
        if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
            s1SplitSize_ = gMax_ / nNumOfQInOneGroup_;
            if (s1SplitSize_ > qSeqSize_) {
                s1SplitSize_ = qSeqSize_;
            }
            s1Outer_ = (qSeqSize_ + s1SplitSize_ - 1U) / s1SplitSize_;
        } else {
            // mla 模式支持G切分
            groupSplitSize_ = gMax_ / qSeqSize_;
            groupSplitSize_ -= (groupSplitSize_ & 1U);
            if (groupSplitSize_ > nNumOfQInOneGroup_) {
                groupSplitSize_ = nNumOfQInOneGroup_;
            }
            gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;
        }
    }
    if (IsFlashDecode(coreNum_, perfMode_)) {
        OP_LOGI(ifaContext_->opName, "Enable flashdecode.");
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }
    CalcInnerSize(seqSize_);
    return SplitBN();
}

custom::graphStatus IFATiling::SplitBN()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }

    for (auto& elem : startIdxEachCore_) {
        elem = bn;
    }

    if (isSysPrefixTiling_) {
        return SplitBN_V0();
    }

    if (actualLenDims_ == 1U || bn <= coreNum_ || (actualLenDims_ == 0U && kvListSeqLens_.size() == 1U) ||
        DealSameSeqEachBatch()) {
        return SplitBN_V0();
    }

    usedCoreNum_ = coreNum_;
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::GetEstimatedLoad(int64_t& estimatedLoad) const
{
    if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
        estimatedLoad = MSD_VEC_LOAD;
    } else if (estimatedLoad == 0) {
        // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
        estimatedLoad = 1;
    }
}

std::vector<int64_t> IFATiling::InitSparseValidArray(const int64_t* actualLens) const
{
    uint32_t outer;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        outer = s1Outer_;
    } else {
        outer = gOuter_;
    }
    std::vector<int64_t> res((batchSize_ * numKvHeads_ * outer));
    for (uint32_t b = 0U; b < batchSize_; b++) {
        for (uint32_t n = 0U; n < numKvHeads_ * outer; n++) {
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

bool IFATiling::BalanceLoad(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
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

void IFATiling::InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                              const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
        } else {
            break;
        }
    }
}

void IFATiling::SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                  uint32_t* sparseStartIdx, int64_t splitFactorSize) const
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
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
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
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
}

custom::graphStatus IFATiling::SplitBN_V0()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }
    // MLA场景，在单核的最大负载不变的情况下，尽可能少的启动核
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        coreNum_ = static_cast<uint32_t>(CeilDivision(
            static_cast<int64_t>(bn), CeilDivision(static_cast<int64_t>(bn), static_cast<int64_t>(aicNum_))));
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

    for (uint32_t i = 0U; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align(computeSeqSize, blockSize_);
    }

    CalcInnerSize(computeSeqSize);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CalcInnerSize(uint32_t seqSize)
{
    /**
     * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
     * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
     *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
     *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
     *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    sInnerSize_ = MAX_SPLIT_SIZE; // 8192
    if (antiQuantFlag_) {
        if (nNumOfQInOneGroup_ > 1U) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
                sInnerSize_ = 2048U;
            } else {
                sInnerSize_ = 1024U;
            }
        } else if (nNumOfQInOneGroup_ == 1) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                sInnerSize_ = 1024U;
            }
        }
    } else {
        /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
         *  GQA场景下 nNumOfQInOneGroup_和sInnerSize_切分大小直接影响workspace大小,
         *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup_将sInnerSize_
         *  分为8192，4096，2048三档，nNumOfQInOneGroup_增大时减小sInnerSize_，
         *  保证最终workspace大小不超过32MB。
         */
        uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
        uint32_t idx = std::min(nNumOfQInOneGroup_ / 5U, 2U);
        sInnerSize_ = sInnerSize[idx];
    }
    // for ifa_cube_310P
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (nNumOfQInOneGroup_ * qSeqSize_ > 1U)) {
        auto bmm1BufferSize = 40U * 1024U;
        sInnerSize_ = std::min(bmm1BufferSize / nNumOfQInOneGroup_ / qSeqSize_ / static_cast<uint32_t>(sizeof(float)),
                               MAX_SPLIT_SIZE);
        sInnerSize_ = (sInnerSize_ / 128U) * 128U;
    }

    if (ropeFlag_) {
        sInnerSize_ = 512U;
        // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
        if (!isWorkspace_ && splitKVFlag_ && inputLayout_ != IfaLayout::TND) {
            if (seqSize == 256U) {
                sInnerSize_ = 128U;
            } else if (seqSize > 256U && seqSize <= sInnerSize_) {
                sInnerSize_ = (sInnerSize_ + 1U) / 2U;
            }
        }
        if (antiQuantFlag_) {
            sInnerSize_ = 1024U; // 1024: 伪量化场景下，sInnerSize_设置为1024
        }
    }
    // PA特性泛化场景，blockSize_可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (pageAttentionFlag_ && blockSize_ != 0U) {
        if (sInnerSize_ % blockSize_ != 0U) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }
    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize && (!(isWorkspace_ && splitKVFlag_))) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::SetDealBN2Num()
{
    uint32_t elemNum = Align(maxActualseq_, 32U) * ((nNumOfQInOneGroup_ + 1U) / 2U);
    dealBN2Num_ = (8192U + elemNum - 1U) / elemNum;
    if (dealBN2Num_ < DEAL_BN2_NUM) {
        dealBN2Num_ = DEAL_BN2_NUM;
    }
}

bool IFATiling::CalcUbBmm()
{
    uint32_t cubeMSize = nNumOfQInOneGroup_ * qSeqSize_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        uint32_t maxMSize = (inputLayout_ == IfaLayout::TND) ? FIA_BALANCE_SG_BASIC_SIZE : gMax_;
        if (cubeMSize > maxMSize) {
            cubeMSize = maxMSize;
        }
        if (amlaMode_ != IfaAmlaMode::DISABLE_AMLA) {
            cubeMSize = Align(cubeMSize, 16U);
        }
    }
    if (ropeFlag_) {
        mmResUbSize_ = static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        bmm2ResUbSize_ = static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
    } else {
        mmResUbSize_ =
            static_cast<size_t>(msdIterNum_) * static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        if (slidingFlag_) {
            bmm2ResUbSize_ =
                static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimVAlign_) * static_cast<size_t>(cubeMSize);
        } else {
            bmm2ResUbSize_ =
                static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
        }
    }
    qPreSizeMla_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(nNumOfQInOneGroup_ * (headDimAlign_ + 64U)) *
                   static_cast<size_t>(qSeqSize_);
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        SetDealBN2Num();
        mmResUbSize_ *= dealBN2Num_;
        bmm2ResUbSize_ *= dealBN2Num_;
    }

    if (isSysPrefixTiling_) {
        mmResUbSize_ *= batchSizeQ_;
        bmm2ResUbSize_ *= batchSizeQ_;
    }
    return true;
}

bool IFATiling::CalcUbAttenMask()
{
    if (!attenMaskFlag_) {
        selectWithByteMaskTmpMinSize_ = 0U;
        return true;
    }
    // bool/int8/uint8类型的mask，每个占1字节
    attenMaskTypeSize_ = 1U; // 1:sizeof(bool)
    if (ropeFlag_) {
        auto selectWithByteMaskTmpShape =
            ge::Shape({nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
        selectWithByteMaskTmpMinSize_ = AscendC::GetSelectWithBytesMaskMinTmpSize(
            selectWithByteMaskTmpShape, ge::Shape({1, 1}), FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);
    } else {
        auto selectWithByteMaskTmpShape =
            ge::Shape({msdIterNum_ * nNumOfQInOneGroup_, Align(sInnerSize_, BYTE_BLOCK / attenMaskTypeSize_)});
        selectWithByteMaskTmpMinSize_ = AscendC::GetSelectWithBytesMaskMinTmpSize(
            selectWithByteMaskTmpShape, ge::Shape({1, 1}), FP32_BYTES, selectWithByteMaskTmpShape, FP32_BYTES, false);
    }

    return true;
}

std::pair<uint32_t, uint32_t> IFATiling::GetPreLoadNumAndActCoreNum() const
{
    uint32_t preLoadNum = 1U;
    uint32_t actCoreNum = coreNum_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        preLoadNum = PRE_LOAD_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        preLoadNum = BUF_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        preLoadNum = PRE_LOAD_NUM_MLA;
        if (isWorkspace_) {
            actCoreNum = aicNum_;
        }
    }
    return std::make_pair(preLoadNum, actCoreNum);
}

void IFATiling::CalcWorkSpaceForBmmAll(const IfaWorkSpaceSizeParams& params, uint32_t preLoadNum, uint32_t actCoreNum)
{
    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * params.mmResElemSize);
    if (ropeFlag_) {
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(mmResUbSize_ * msdIterNum_ * actCoreNum * params.vec1ResElemSize) *
                                params.kvDtypeRatio);
    } else {
        workspaceSize_ +=
            preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * params.vec1ResElemSize) *
                                             params.kvDtypeRatio);
    }
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.bmm2ResElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.vec2ResElemSize;
    if (ropeFlag_) {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
                                           static_cast<float>(qPreSizeMla_ * actCoreNum * params.qPreProcResElemSize) *
                                           params.kvDtypeRatio);
    } else {
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(bmm2ResUbSize_ * actCoreNum * params.qPreProcResElemSize) *
                                params.kvDtypeRatio);
    }
    workspaceSize_ +=
        static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.nUpdateElemSize; // aMla nUpdate, Gmax=128
    workspaceSize_ +=
        static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.softmaxSumElemSize; // aMla softmaxSum, Gmax=128
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        uint32_t tmpWsNum = 1U; // 1块workspace: aMaxBmm1Gm
        if (antiquantMode_ == PER_TOKEN_MODE) {
            tmpWsNum = 4U; // 4块workspace: aMaxBmm1Gm/qRowSumGm/softmaxResAmaxGm/softmaxResRowSumGm
        }
        workspaceSize_ += preLoadNum *
                          static_cast<size_t>(static_cast<float>(actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)) *
                          tmpWsNum;
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
                                           actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)); // SoftmaxSumGm
    }
}

custom::graphStatus IFATiling::CalcWorkSpace()
{
    IfaWorkSpaceSizeParams params{};
    if (antiQuantFlag_) {
        if (ropeFlag_) {
            params.mmResElemSize = 2U;   // 2: S已经被转换成FP16存在workspace上
            params.bmm2ResElemSize = 2U; // 2: O已经被转换成FP16在workspace上
            params.vec2ResElemSize = 2U;
        } else {
            params.mmResElemSize = 4U;   // 4:int32
            params.bmm2ResElemSize = 4U; // 4:int32
            params.vec2ResElemSize = 4U; // 4:float
        }
        params.vec1ResElemSize = 1U;     // 1:int8
        params.qPreProcResElemSize = 1U; // int8
        params.kvDtypeRatio = 1.0;       // 0.5:int4 1.0:elseType
    }

    workspaceSize_ = libapiSize_;
    std::pair<uint32_t, uint32_t> preLoadInfo = GetPreLoadNumAndActCoreNum();
    uint32_t preLoadNum = preLoadInfo.first;
    uint32_t actCoreNum = preLoadInfo.second;
    if (perfMode_ != IfaPerfMode::BMM_ALL_BY_VEC) {
        CalcWorkSpaceForBmmAll(params, preLoadNum, actCoreNum);
    }
    CalcFDWorkSpace(actCoreNum);

    uint32_t mmPACallBackDataSize = 64; // 64: matmul回调信息需要7个uint32值，dcci cacheline需要64B对齐
    if (pageAttentionFlag_) {
        workspaceSize_ += static_cast<size_t>(actCoreNum) * static_cast<size_t>(mmPACallBackDataSize) *
                          static_cast<size_t>(2); // bmm1 bmm2 2份
    }

    if (isSysPrefixTiling_) {
        if (antiQuantFlag_) {
            size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) *
                               static_cast<size_t>(batchSizeQ_);
            workspaceSize_ += actCoreNum * blockSize * 4U; // 4, rowMax1 rowMax2 rowSum1 rowSum2
        }

        size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) *
                           static_cast<size_t>(batchSizeQ_);
        workspaceSize_ += actCoreNum * blockSize * 3U; // 3, sum log exp

        if (!antiQuantFlag_) {
            workspaceSize_ += static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(nNumOfQInOneGroup_) *
                              static_cast<size_t>(headDimAlign_) * static_cast<size_t>(2) *
                              static_cast<size_t>(actCoreNum); // 2:fp16/bf16
        }
    }
    ifaContext_->workSpaceSize = workspaceSize_;
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        MLACalcFDWorkSpace(actCoreNum);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

void IFATiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0U;
        uint32_t logSumExpSize = 0U;
        uint32_t FDParamNums = balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        if (slidingFlag_) {
            accumOutSize = FDParamNums * headDimVAlign_;
        } else {
            accumOutSize = FDParamNums * headDimAlign_;
        }
        logSumExpSize = NUM2 * FDParamNums * (BYTE_BLOCK / blockTypeSize_); // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
        if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32U; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void IFATiling::MLACalcFDWorkSpace(const uint32_t actCoreNum)
{
    uint32_t accumOutSize = 0U;
    uint32_t logSumExpSize = 0U;

    // BALANCE
    uint32_t FDParamNumsBalance = CalcBalanceFDParamNums(actCoreNum);
    uint32_t accumOutSizeBalance = FDParamNumsBalance * headDimAlign_;
    uint32_t logSumExpSizeBalance = 2U * FDParamNumsBalance * (BYTE_BLOCK / blockTypeSize_);

    // UNBALANCE
    uint32_t FDParamNumsUnbalance = CalcUnbalanceFDParamNums();
    uint32_t accumOutSizeUnbalance = FDParamNumsUnbalance * headDimAlign_;
    uint32_t logSumExpSizeUnbalance = 2U * FDParamNumsUnbalance * (BYTE_BLOCK / blockTypeSize_);

    if (isWorkspace_) { // tiling下沉场景，无法通过actual_seq_kv判断走哪种分核，针对不同layout采取不同workspace分配策略
        if (inputLayout_ == IfaLayout::TND) {
            accumOutSize = accumOutSizeBalance;
            logSumExpSize = logSumExpSizeBalance;
        } else if (inputLayout_ == IfaLayout::BSH_BSND) {
            accumOutSize = std::max(accumOutSizeBalance, accumOutSizeUnbalance);
            logSumExpSize = std::max(logSumExpSizeBalance, logSumExpSizeUnbalance);
        } else {
            accumOutSize = accumOutSizeUnbalance;
            logSumExpSize = logSumExpSizeUnbalance;
        }
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

uint32_t IFATiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint32_t IFATiling::CalcUnbalanceFDParamNums() const
{
    return batchSizeQ_ * numHeads_ * kvSplitPart_;
}

custom::graphStatus IFATiling::FillTiling()
{
    FillTilingBaseParams();
    FillTilingSplitKV();
    FillTilingCoreParams();
    FillTilingSingleCoreParams();
    FillTilingSingleCoreTensorSize();

    FillTilingOutputParams();
    return custom::graphStatus::GRAPH_SUCCESS;
}

void IFATiling::FillTilingBaseParams()
{
    tilingData_->baseParams.set_batchSize(batchSize_);
    tilingData_->baseParams.set_seqSize(sMax_);
    tilingData_->baseParams.set_qSeqSize(qSeqSize_);
    tilingData_->baseParams.set_headSize(headDim_);
    tilingData_->baseParams.set_headSizeV(headDimV_);
    tilingData_->baseParams.set_blockSize(blockSize_);
    tilingData_->baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingData_->baseParams.set_scaleValue(scaleValue_);
    tilingData_->baseParams.set_kvHeadNum(numKvHeads_);
    tilingData_->baseParams.set_qHeadNum(numHeads_);
    tilingData_->baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);
    tilingData_->baseParams.set_batchContinuousFlag(batchContinuousFlag_);
    tilingData_->baseParams.set_pseShiftFlag((pseShiftFlag_) ? 1 : 0);
    tilingData_->baseParams.set_pseShiftB(pseShiftBatch_);
    tilingData_->baseParams.set_pseShiftS(pseShiftS1_);
    tilingData_->baseParams.set_selectWithByteMaskTmpMinSize(selectWithByteMaskTmpMinSize_); // mask
    tilingData_->baseParams.set_actualLenQDims(actualLenQDims_);
    tilingData_->baseParams.set_actualLenDims(isSysPrefixTiling_ ? actualLenDimsPrefix_ : actualLenDims_);
    tilingData_->baseParams.set_msdIterNum(msdIterNum_);
    tilingData_->baseParams.set_kvPaddingFlag(kvPaddingSizeFlag_ ? 1 : 0);
    tilingData_->baseParams.set_antiquantPerTensorFlag(antiquantPerTensorFlag_);
    tilingData_->baseParams.set_antiquantPerHeadFlag(antiquantPerHeadFlag_);
    tilingData_->baseParams.set_antiquantParamsInPagedAttentionFlag(antiquantParamsInPagedAttentionFlag_);
    tilingData_->baseParams.set_attenMaskFlag(attenMaskFlag_ ? 1 : 0);
    tilingData_->baseParams.set_attenMaskSize(attenMaskSize_);
    tilingData_->baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
    tilingData_->baseParams.set_softmaxLseFlag(softmaxLseFlag_); // whether return lse
    tilingData_->baseParams.set_totalBlockNum(totalBlockNum_);
    tilingData_->baseParams.set_antiqSeqSize(antiqSeqSize_);
    tilingData_->baseParams.set_sparseMode(sparseMode_);
    tilingData_->baseParams.set_slidingFlag(slidingFlag_ ? 1 : 0);
    tilingData_->baseParams.set_windowSize(windowSize_);
}

// for flash decode
void IFATiling::FillTilingSplitKV() const
{
    tilingData_->splitKVParams.set_s2(kvSplitPart_);
    uint32_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
        OP_LOGD(ifaContext_->opName, "PA FlashDecode is enabled, sInnerLoopSize is %u, blockSize is %u",
                sInnerLoopSize_, blockSize_);
    }

    tilingData_->splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    if (inputLayout_ == IfaLayout::TND) {
        tilingData_->splitKVParams.set_accumOutSize(tSeqSize_ * numHeads_ * kvSplitPart_ * headDimAlign_);
        tilingData_->splitKVParams.set_logSumExpSize(NUM2 * batchSizeQ_ * numHeads_ * kvSplitPart_ * qSeqSize_ * // 2份
                                                     (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    } else {
        if (slidingFlag_) {
            tilingData_->splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimVAlign_);
        } else {
            tilingData_->splitKVParams.set_accumOutSize(batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_);
        }
        tilingData_->splitKVParams.set_logSumExpSize(NUM2 * batchSizeQ_ * numHeads_ * kvSplitPart_ *
                                                     (BYTE_BLOCK / blockTypeSize_)); // 2: sum + max
    }
    if (!splitKVFlag_) {
        tilingData_->splitKVParams.set_s2(0);
    }
}

void IFATiling::FillTilingCoreParams() const
{
    uint32_t* coreStartIdx = tilingData_->increFlashAttentionCoreParams.get_coreSidxEnd();
    memcpy_s(coreStartIdx, MAX_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_CORE_NUM * sizeof(uint32_t));
}

void IFATiling::FillTilingSingleCoreParams() const
{
    tilingData_->increFlashAttentionSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
    tilingData_->increFlashAttentionSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail_);
    tilingData_->increFlashAttentionSingleCoreParams.set_usedCoreNum(usedCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_formerCoreNum(formerCoreNum_);
    tilingData_->increFlashAttentionSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range_);
    tilingData_->increFlashAttentionSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange_);
    tilingData_->increFlashAttentionSingleCoreParams.set_groupSplitSize(groupSplitSize_);
    tilingData_->increFlashAttentionSingleCoreParams.set_s1SplitSize(s1SplitSize_);
}

void IFATiling::FillTilingSingleCoreTensorSize() const
{
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_->increFlashAttentionSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

// for zero output
void IFATiling::FillTilingOutputParams() const
{
    tilingData_->outputParams.set_isOutQuantTypeBf16(isOutQuantTypeBf16_);
    tilingData_->outputParams.set_isPerChnOut(isOutQuantPerChnOut_);
}

// S1 * G > 128
IfaAmlaMode IFATiling::GetAmlaMode() const
{
    // 1.只有NZ才应该走进这个函数，提前判断, 2.伪量化和全量化不走amla
    if ((!ropeFlag_) || inputKvLayout_ != IfaLayout::NZ || antiQuantFlag_ || quantFlag_) {
        return IfaAmlaMode::DISABLE_AMLA;
    }
    OP_LOGI(ifaContext_->opName, "input Key and Value format is NZ and blockSize_[%u], nNumOfQInOneGroup_[%u]",
            blockSize_, nNumOfQInOneGroup_);
    // 当前暂时只支持 大于24batch G128 MTP1,3 512对齐 走3buf模板
    if (blockSize_ == 128U) {
        return IfaAmlaMode::AMLA; // s1 * G <= 128 nz amla
    }
    return IfaAmlaMode::DISABLE_AMLA;
}

custom::graphStatus IFATiling::GetKvLayoutInfo(KvLayoutInfo& kvLayoutInfo) const
{
    switch (inputKvLayout_) {
        case IfaLayout::NZ:
            kvLayoutInfo.kvLayoutVal = 2U;
            kvLayoutInfo.amlaMode = static_cast<uint8_t>(amlaMode_);
            break;
        case IfaLayout::BSH_BSND:
            kvLayoutInfo.kvLayoutVal = 1U;
            break;
        case IfaLayout::BNSD:
            kvLayoutInfo.kvLayoutVal = 0U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputKvLayout%u", kvLayoutInfo.kvLayoutVal);
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetInputLayoutVal(uint8_t& layoutVal) const
{
    switch (inputLayout_) {
        case IfaLayout::TND:
            layoutVal = 3U; // 2:TND
            break;
        case IfaLayout::BSH_BSND:
            layoutVal = 0U;
            break;
        case IfaLayout::BNSD:
            layoutVal = 1U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputLayout%u", static_cast<uint32_t>(inputLayout_));
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetInputQueryVal(uint8_t& inputQVal) const
{
    switch (inputQType_) {
        case at::ScalarType::Half:
            inputQVal = 0U;
            break;
        case at::ScalarType::BFloat16:
            inputQVal = 2U;
            break;
        case at::ScalarType::Char:
            inputQVal = 3U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputQType%d", inputQType_);
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetInputKvVal(uint8_t& inputKvVal) const
{
    switch (inputKvType_) {
        case at::ScalarType::Half:
            inputKvVal = 0U;
            break;
        case at::ScalarType::BFloat16:
            inputKvVal = 2U;
            break;
        case at::ScalarType::Char:
            inputKvVal = 3U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support inputKvType%d", inputKvType_);
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GetOutputVal(uint8_t& outputVal) const
{
    switch (outputType_) {
        case at::ScalarType::Half:
            outputVal = 0U;
            break;
        case at::ScalarType::BFloat16:
            outputVal = 2U;
            break;
        case at::ScalarType::Char:
            outputVal = 3U;
            break;
        default:
            OP_LOGE(ifaContext_->opName, "not support outputType %d", outputType_);
            return custom::graphStatus::GRAPH_FAILED;
    }
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::GenTilingKey() const
{
    uint8_t layoutVal = 0U;
    uint8_t inputQVal = 0U;
    uint8_t inputKvVal = 0U;
    uint8_t outputVal = 0U;
    uint8_t originVal = 0U;
    uint8_t cvRatioVal = 0U;
    uint8_t splitKvVal = kvSplit_ > 0U ? 1U : 0U;
    uint8_t paVal = pageAttentionFlag_ == true ? 1U * 2U : 0U;
    uint8_t antiquantModeVal = antiquantMode_ == PER_TOKEN_MODE ? 1U * 4U : 0U;
    uint64_t modeVal = sysPrefixFlag_ ? 2U : 1U;
    if (atbRunFlag_) {
        modeVal = static_cast<uint64_t>(ATB_MODE_VAL);
    }
    KvLayoutInfo kvLayoutInfo{};
    uint8_t balanceMode = static_cast<uint8_t>(balanceModeFlag_);

    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        if (GetKvLayoutInfo(kvLayoutInfo) != custom::graphStatus::GRAPH_SUCCESS) {
            return custom::graphStatus::GRAPH_FAILED;
        }
    }

    // page attention 新模板上线后删除这里的特殊处理
    if (pageAttentionFlag_ && sMax_ == 0) {
        paVal = 0;
    }

    if (GetInputLayoutVal(layoutVal) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if (GetInputQueryVal(inputQVal) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (GetInputKvVal(inputKvVal) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (GetOutputVal(outputVal) != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    originVal = inputQVal;
    if (ropeFlag_ && quantFlag_) {
        originVal = outputVal;                // 此处应该获取ROPE的类型，需要修改
        cvRatioVal = (cvRatio_ == 1) ? 1 : 0; // CV1:1场景为1，其他场景为0
    }
    ifaContext_->fdFlag = splitKvVal;
    ifaContext_->layoutVal = layoutVal;
    ifaContext_->antiquantMode_ = static_cast<uint8_t>(antiquantMode_);

    OP_LOGI(ifaContext_->opName, "IFA tilingKey: %lu.", ifaContext_->tilingKey);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CalcNumBlocks()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto aicNum = aicNum_;
    auto aivNum = aivNum_;
    UpdatePerfMode();
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        aivNum = aicNum;
    } else {
        if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || !splitKVFlag_) {
            if (perfMode_ == IfaPerfMode::C1_V1) { // 2:bn数不超过vector core一半时，CV开启CV 1:1
                aivNum = usedCoreNum_;             // CV 1:1时,GetTaskRation()的结果为1,所以aivNum与aicNum相等
                aicNum = aivNum;
            } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
                       perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                aivNum = cvRatio_ * usedCoreNum_;
                aicNum = usedCoreNum_;
            } else {
                aivNum = Align(usedCoreNum_, 2U); // aivNum必须为偶数达成CV 1:2
                aicNum = (aivNum + 1U) / 2U;      // cube核的数量为vector核的数量按2向上对齐
            }
        }
    }
    ifaContext_->numBlocks = ascendcPlatform->CalcTschBlockDim(aivNum, aicNum, aivNum); // 暂时与当前代码一致
    OP_LOGI(ifaContext_->opName, "IFA block dim: %u aiv Num: %u aic Num: %u.", ifaContext_->numBlocks, aivNum, aicNum);
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SharedPrefixTiling()
{
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::FillSysPrefixTiling()
{
    tilingDataPrefix_->set_prefixAttenOutOffset(prefixAttenOutOffset_);
    tilingDataPrefix_->set_userPromptAttenOutOffset(userPromptAttenOutOffset_);
    tilingDataPrefix_->set_tmpLseOffset(tmpLseOffset_);
    tilingDataPrefix_->set_prefixLen(maxActualPrefixLen_);
    tilingDataPrefix_->set_formerCoreNum(formerCoreNumSp_);
    tilingDataPrefix_->set_blockSplitBn2Range(blockSplitBn2RangeSp_);
    tilingDataPrefix_->set_tailSplitedBatchRange(tailSplitedBatchRangeSp_);
    tilingDataPrefix_->set_usedCoreNum(combinUsedCore_);
    tilingDataPrefix_->set_batchSizeQ(batchSizeQ_);
    tilingData_ = &tilingDataPrefix_->base;
    return FillTiling();
}

custom::graphStatus IFATiling::CalcSysPrefixWorkSpace()
{
    size_t size0 = workspaceSize_;
    size_t outSize = static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(numHeads_) *
                     static_cast<size_t>(headDimAlign_) * static_cast<size_t>(blockTypeSize_);
    size_t lseSize = static_cast<size_t>(4) * static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(numHeads_) *
                     static_cast<size_t>(BYTE_BLOCK);

    CalcWorkSpace();

    workspaceSize_ = std::max(workspaceSize_, size0);
    workspaceSize_ = Align(workspaceSize_, 512UL);
    prefixAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;
    userPromptAttenOutOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += outSize;

    tmpLseOffset_ = workspaceSize_ - libapiSize_;
    workspaceSize_ += lseSize;

    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::CalcSysPrefixNumBlocks()
{
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::SplitForLseCombine()
{
    uint32_t coreNum = usedCoreNum_;

    uint32_t bn = batchSizeQ_ * numKvHeads_;
    formerCoreNumSp_ = bn % coreNum;
    if (formerCoreNumSp_ == 0U) {
        blockSplitBn2RangeSp_ = bn / coreNum;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_;
    } else {
        blockSplitBn2RangeSp_ = bn / coreNum + 1U;
        tailSplitedBatchRangeSp_ = blockSplitBn2RangeSp_ - 1U;
    }
    combinUsedCore_ = bn > coreNum ? coreNum : bn;
    return custom::graphStatus::GRAPH_SUCCESS;
}

custom::graphStatus IFATiling::RunBigKernelTiling(IFAContext& context, IncreFlashAttentionTilingDataV2* tilingData,
                                                  bool isWorkspace)
{
    this->ifaContext_ = &context;
    this->tilingData_ = &(tilingData->tilingBase);

    if (this->tilingData_ == nullptr) {
        OP_LOGI(ifaContext_->opName, " tiling data is nullptr.");
    }

    this->isWorkspace_ = true;

    if ((GetNpuInfo() != custom::graphStatus::GRAPH_SUCCESS) || (PreProcess() != custom::graphStatus::GRAPH_SUCCESS)) {
        return custom::graphStatus::GRAPH_FAILED;
    }

    if ((ZeroTensorProcess() != custom::graphStatus::GRAPH_SUCCESS) ||
        (Split() != custom::graphStatus::GRAPH_SUCCESS) || (FillTiling() != custom::graphStatus::GRAPH_SUCCESS) ||
        (CalcWorkSpace() != custom::graphStatus::GRAPH_SUCCESS) ||
        (CalcNumBlocks() != custom::graphStatus::GRAPH_SUCCESS)) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    if (sysPrefixFlag_ && SharedPrefixTiling() != custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_FAILED;
    }
    return GenTilingKey();
}

std::string DataTypeToSerialString(at::ScalarType type)
{
    switch (type) {
        case at::ScalarType::Byte:
            return "DT_UINT8";
        case at::ScalarType::Char:
            return "DT_INT8";
        case at::ScalarType::Short:
            return "DT_INT16";
        case at::ScalarType::Int:
            return "DT_INT32";
        case at::ScalarType::Long:
            return "DT_INT64";
        case at::ScalarType::Half:
            return "DT_FLOAT16";
        case at::ScalarType::Float:
            return "DT_FLOAT";
        case at::ScalarType::Double:
            return "DT_DOUBLE";
        case at::ScalarType::BFloat16:
            return "DT_BFLOAT16";
        case at::ScalarType::Bool:
            return "DT_BOOL";
        default:
            OP_LOGE("IncreFlashAttention", "scalar type %d not support", static_cast<int>(type));
            return "UNDEFINED";
    }
}

uint32_t IFATiling::GetTotalWorkspaceSize() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return static_cast<uint32_t>(libapiSize_);
    }

    // 根据实际的numBlocks减少下
    uint32_t usrWorkspaceSize =
        static_cast<uint32_t>(WS_REPEAT_NUM * aivNum_ * BLOCKSIZE_CALC_256 * static_cast<uint32_t>(headDim_) *
                              NUM_BYTES_FLOAT16) +
        static_cast<uint32_t>(WS_REPEAT_NUM * aivNum_ * WS_TMP_SIZE_PER_CORE * NUM_BYTES_FLOAT16);
    return usrWorkspaceSize + static_cast<uint32_t>(libapiSize_);
}

uint32_t IFATiling::GetHeadSize() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return ifaContext_->query.shape[1] * BLOCK_SIZE / numHeads_;
    }

    if (ifaContext_->query.shape.size() == DIM_NUM_TWO) {
        return ifaContext_->query.shape[DIM_IDX_1] / numHeads_;
    } else if (ifaContext_->query.shape.size() == DIM_NUM_THREE) {
        return ifaContext_->query.shape[DIM_IDX_2];
    } else {
        return ifaContext_->query.shape[DIM_IDX_3];
    }
}

uint32_t IFATiling::CalcSeqStepQ() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return MAX_MATMUL_BASE_M;
    } else if (ifaContext_->actualSeqLengthsQ.hasValue) {
        uint32_t mmADataNum = static_cast<uint32_t>(L0B_SIZE / DOUBLE_BUFFER_NUM / NUM_BYTES_FLOAT16);
        uint32_t maxSeqStepQ = std::min(MAX_MATMUL_BASE_M, mmADataNum / headDim_ / BLOCK_SIZE * BLOCK_SIZE);
        return maxSeqStepQ;
    }

    return MAX_MATMUL_BASE_M;
}

uint32_t IFATiling::GetTotalQBlockNum() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return batchSize_ * numHeads_;
    }

    return batchSize_ * numHeads_;
}

uint32_t IFATiling::CalcSeqStepKv() const
{
    uint32_t tmpSeqStepKv = blockSize_;
    while (tmpSeqStepKv > seqStepQ_) {
        // blockSize不断折半，直到小于cube的seqStep
        tmpSeqStepKv = static_cast<uint32_t>(tmpSeqStepKv / HALF_REDUCE_RATE);
    }

    return tmpSeqStepKv;
}

custom::graphStatus IFATiling::DoSubOpTiling(IFAContext& ifaContext)
{
    ifaTilingData = &(ifaContext.tilingData);
    if (RunBigKernelTiling(ifaContext, ifaTilingData) == custom::graphStatus::GRAPH_SUCCESS) {
        return custom::graphStatus::GRAPH_SUCCESS;
    }
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    return custom::graphStatus::GRAPH_FAILED;
}
} // namespace optiling
