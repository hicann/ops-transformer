/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_selected_attention_infer_tiling.cpp
 * \brief
 */

#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "nsa_selected_attention_infer_tiling_base.h"
#include "nsa_selected_attention_infer_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

template <typename T> inline T Align(T num, T rnd)
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

constexpr uint32_t PRE_LOAD_NUM = 4;

void NsaSelectTiling::SetCoreNum()
{
    coreNum_ = aicNum_;
}

ge::graphStatus NsaSelectTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("NsaSelectedAttentionInfer", "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    socVersion_ = NsaSocVersion::A2;

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR("NsaSelectedAttentionInfer", "num of core obtained is 0."), return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::PreProcess()
{
    if (ProcessBaseInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ProcessOptionalTensors() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetupPerfMode();
    SetCoreNum();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::QKVPreProcess()
{
    OP_CHECK_IF(context_->key.desc->GetDataType() != context_->value.desc->GetDataType(),
        OP_LOGE(context_->opName, "datatype of k tensor and value tensor is different"), return ge::GRAPH_FAILED);
    batchSizeQ_ = batchSize_ = context_->query.shape->GetStorageShape().GetDim(0);
    inputQType_ = context_->query.desc->GetDataType();
    inputKvType_ = context_->key.desc->GetDataType();
    outputType_ = context_->attenOut.desc->GetDataType();
    std::string layout(context_->layOut);
    // B S1 N2 K
    uint32_t selectedBlockCountShapeRange = 0;
    if (layout == "TND") {
        batchSizeQ_ = batchSize_ = context_->actualQSeqLengths.tensor->GetShapeSize();
        selectedBlockCountShapeRange = context_->topkIndices.shape->GetStorageShape().GetDim(dimTwo);
    } else {
        selectedBlockCountShapeRange = context_->topkIndices.shape->GetStorageShape().GetDim(dimThree);
    }

    numHeads_ = *context_->numHeads;
    numKvHeads_ = *context_->kvHeadNums;
    scaleValue_ = *context_->scaleValue;
    blockSize_ = *context_->blockSize;
    selectedBlockSize_ = *context_->selectedBlockSize;
    selectedBlockCount_ = *context_->selectedBlockCount;
    
    OP_CHECK_IF((selectedBlockCount_ != -1) && (selectedBlockCount_ != selectedBlockCountShapeRange), OP_LOGE(context_->opName,
        "selectedBlockCount is not -1, %ld; or the same with topkindices last dim shape %u", selectedBlockCount_, selectedBlockCountShapeRange), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeads_ <= 0, OP_LOGE(context_->opName, "numHeads %ld is less than 1", numHeads_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numKvHeads_ <= 0, OP_LOGE(context_->opName, "numKvHeads %ld is less than 1", numKvHeads_), return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectedBlockCount_ <= 0, OP_LOGE(context_->opName, "selectedBlockCount %ld is less than 1", selectedBlockCount_), return ge::GRAPH_FAILED);
    if (numKvHeads_ == 0) {
        numKvHeads_ = numHeads_;
    }
    nNumOfQInOneGroup_ = static_cast<uint32_t>(numHeads_ / numKvHeads_);
    OP_CHECK_IF(((numKvHeads_ > numHeads_) || (numHeads_ % numKvHeads_ != 0) || (nNumOfQInOneGroup_ > 16)),
        OP_LOGE(context_->opName, "groupSize should be less than or equal to 16, Attr num_key_value_heads is invalid, n: %ld, kvHeadNum: %ld", numHeads_,
            numKvHeads_),
        return ge::GRAPH_FAILED);
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = 1U;

    const int64_t *actualQLenData = context_->actualQSeqLengths.tensor->GetData<int64_t>();
    const int64_t *actualKVLenData = context_->actualKVSeqLengths.tensor->GetData<int64_t>();
    actualLenQDims_ = context_->actualQSeqLengths.tensor->GetShapeSize();
    for (uint32_t i = 0; i < batchSize_; i++) {
        int64_t actQLen = actualQLenData[i];
        int64_t actKVLen = actualKVLenData[i];
        OP_CHECK_IF(
            actQLen > actKVLen,
            OP_LOGE(context_->opName,
                    "the value of actual_q_seq_lengths %ld must be less than or equal to the actual_kv_seq_lengths %ld", actQLen, actKVLen),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            actQLen < 1 || actQLen > 4, // actualQSeqLengths必须大于0，小于4
            OP_LOGE(context_->opName,
                    "the value of actual_q_seq_lengths[%u] must be greater than or equal to 1 and  less than 4, but it is %ld", i, actQLen),
            return ge::GRAPH_FAILED);
        maxactLen_ = std::max(maxactLen_, actQLen);
    }

    uint32_t sOfQuery = 0;
    uint32_t sOfHeadnum = 0;
    if (layout == "BSH") {
        inputLayout_ = NsaLayout::BSH;
        OP_CHECK_IF(context_->query.shape->GetStorageShape().GetDim(dimTwo) % numHeads_ != 0,
            OP_LOGE(context_->opName, "H should be an interger multiple of numHeads"),
            return ge::GRAPH_FAILED);
        sOfQuery = context_->query.shape->GetStorageShape().GetDim(dimOne); // 1, dim of H
        headDim_ = context_->query.shape->GetStorageShape().GetDim(dimTwo) / numHeads_; // 2, dim of H
        headDimV_ = context_->value.shape->GetStorageShape().GetDim(dimTwo) / numKvHeads_; // 2, dim of H
        sOfHeadnum = static_cast<uint32_t>(numHeads_);
        uint32_t kDimNum = context_->key.shape->GetStorageShape().GetDimNum();
        if ((*context_->kvHeadNums != 0) && (kDimNum == 3U)) { // 3, dim of kv when the layout of kv is BSH
            sOfHeadnum = static_cast<uint32_t>(headDim_ * numHeads_ * numKvHeads_) / context_->key.shape->GetStorageShape().GetDim(2); // 2, dim of H
        }
    } else if (layout == "BSND") {
        inputLayout_ = NsaLayout::BSND;
        sOfQuery = context_->query.shape->GetStorageShape().GetDim(dimOne);   // 1, dim of S
        headDim_ = context_->query.shape->GetStorageShape().GetDim(dimThree);   // 3, dim of D
        sOfHeadnum = context_->query.shape->GetStorageShape().GetDim(dimTwo); // 2, dim of N
        headDimV_ = context_->value.shape->GetStorageShape().GetDim(dimThree); // 3, dim of D
    } else if (layout == "TND") {
        inputLayout_ = NsaLayout::TND;
        sOfQuery = static_cast<uint32_t>(maxactLen_);   // 1, dim of S
        headDim_ = context_->query.shape->GetStorageShape().GetDim(dimTwo);   // 2, dim of D
        sOfHeadnum = context_->query.shape->GetStorageShape().GetDim(dimOne); // 1, dim of N
        headDimV_ = context_->value.shape->GetStorageShape().GetDim(dimTwo) / numKvHeads_; // 2, dim of D
    } else {
        OP_LOGE(context_->opName, "Only support input_layout(BSH, BSND, TND), actually is %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(numHeads_) != sOfHeadnum) {
        OP_LOGE(context_->opName, "numHeads should be equal to qOfHeadnum");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(sOfQuery > 4 || sOfQuery < 1, OP_LOGE(context_->opName, " S of Query:%u is invalid, it should be 1~4", sOfQuery),
                return ge::GRAPH_FAILED);

    qSeqSize_ = sOfQuery;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::InputAttrsPreProcess() const
{
    OP_CHECK_IF(
        context_->blockTable.tensor->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE(context_->opName, "check blockTable shape failed, blockTable shapeSize is zero."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ProcessBaseInputs()
{
    if ((CheckBaseInputsNull() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
    if ((QKVPreProcess() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
    }
    if ((InputAttrsPreProcess() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
    }
    if ((KvShapePostProcess() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
    }
    if ((CheckQKOutShape() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
    }
    if ((CheckInputFormatAndLimits() != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
    }
    if (InitInOutMode() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void NsaSelectTiling::SetupPerfMode()
{
    perfMode_ = NsaPerfMode::CUBE_VIEW_MM;
}

ge::graphStatus NsaSelectTiling::KvShapePostProcess()
{
    maxBlockNumPerBatch_ = context_->blockTable.tensor->GetStorageShape().GetDim(1);
    sMax_ = maxBlockNumPerBatch_ * blockSize_;
    seqSize_ = sMax_;
    inputKvLayout_ = inputLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ZeroTensorProcess()
{
    if (sMax_ == 0U) {
        /*
        * 1024，空tensor场景下，作为默认值完成后续计算
        * 避免matmal tiling  softmax tiling异常
        * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
        */
        seqSize_ = 1024U;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::InitInOutMode()
{
    if (inputQType_ == ge::DT_FLOAT16 && outputType_ == ge::DT_FLOAT16) {
        inOutMode_ = TilingInOutMode::FP16_FP16;
    } else if (inputQType_ == ge::DT_BF16 && outputType_ == ge::DT_BF16) {
        inOutMode_ = TilingInOutMode::BF16_BF16;
    } else {
        OP_LOGE(context_->opName, "input dtype %d with output dtype %d is not currently supported.", inputQType_,
                outputType_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ProcessOptionalTensors()
{
    if ((ProcessActualSeqLen() != ge::GRAPH_SUCCESS) ||
        (ProcessBlockTable() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ProcessActualSeqLen()
{
    actualSeqLenFlag_ = true;
    actualLenDims_ = this->isWorkspace_ ? MAX_SIZE_BATCH : context_->actualKVSeqLengths.tensor->GetShapeSize();
    if (actualLenDims_ == 0U) {
        OP_LOGE(context_->opName, "actual_seq_lengths size[%u] can not be zero in pageAttention scene", actualLenDims_);
        maxActualseq_ = sMax_;
        return ge::GRAPH_SUCCESS;
    }
    actualLenDims_ = std::min(actualLenDims_, batchSize_);
    const int64_t *actualLenData = context_->actualKVSeqLengths.tensor->GetData<int64_t>();
    if (this->isWorkspace_){
        maxActualseq_ = sMax_;
    }
    if (actualLenData != nullptr && !this->isWorkspace_) {
        int32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : batchSize_;
        for (int32_t i = 0; i < loop; i++) {
            int64_t actLen = (actualLenDims_ == 1) ? actualLenData[0] : actualLenData[i];
            OP_CHECK_IF(actLen < 0,
                OP_LOGE(context_->opName,
                        "the value of actual_seq_lengths[%d] must be greater than or equal to 0, but it is %ld", i, actLen),
                return ge::GRAPH_FAILED);
            maxActualseq_ = maxActualseq_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : maxActualseq_;
        }
    } else if (!this->isWorkspace_) {
        OP_LOGE(context_->opName, "data of actual_seq_lengths can not be nullptr in pageAttention scene");
    }
    const int64_t *actualQLenData = context_->actualQSeqLengths.tensor->GetData<int64_t>();
    actualLenQDims_ = context_->actualQSeqLengths.tensor->GetShapeSize();
    for (uint32_t i = 0; i < batchSize_; i++) {
        int64_t actLen = actualQLenData[i];
        OP_CHECK_IF(
            actLen < 1 || actLen > 4, // actualQSeqLengths必须大于0，小于等于4
            OP_LOGE(context_->opName,
                    "the value of actual_q_seq_lengths[%u] must be greater than or equal to 1 and  less than 4, but it is %ld", i,
                    actLen),
            return ge::GRAPH_FAILED);
        totalQseq_ += static_cast<uint32_t>(actLen);
        if (actLen > 1) {
            isMtpFlag_ = true;
        }
    }
return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ProcessBlockTable()
{
    // gm到l1，copynd2nz的srcDValue最大支持65535
    if (numKvHeads_ * headDim_ > COPYND2NZ_SRC_STRIDE_LIMITATION) { // 0: BSH
        OP_LOGE(context_->opName,
            "When input kvcache layout is BSH or BSND or TND, the N * D of kvcache is %u, "
            "exceeds the maximum limit (%u) of the datacopy instruction.",
            static_cast<uint32_t>(numKvHeads_) * headDim_, COPYND2NZ_SRC_STRIDE_LIMITATION);
        return ge::GRAPH_FAILED;
    }

    if (CheckPABlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        maxActualseq_ > blockSize_ * maxBlockNumPerBatch_,
        OP_LOGE(context_->opName,
            "Invalid actual seq length for PA, max actual seq length(%u) "
            "is larger than blocksize(%ld) * max block num per batch(%u)",
            maxActualseq_, blockSize_, maxBlockNumPerBatch_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::Split()
{
    CalcInnerSize(seqSize_);
    return SplitBN();
}

ge::graphStatus NsaSelectTiling::SplitBN()
{
    uint32_t bn = batchSize_ * numKvHeads_ * gOuter_;
    if (isMtpFlag_) {
        bn = static_cast<uint32_t>(numKvHeads_) * totalQseq_;
    }

    for (uint32_t i = 0; i < MAX_CORE_NUM; i++) {
        startIdxEachCore_[i] = bn;
    }

    if (actualLenDims_ == 1 || bn <= coreNum_ || this->isWorkspace_) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray;
    const int64_t *actualLenData = context_->actualKVSeqLengths.tensor->GetData<int64_t>();
    validArray = InitSparseValidArray(actualLenData);

    SetSparseStartIdx(validArray, bn, coreNum_, startIdxEachCore_, CeilDivision(bn, coreNum_));

    usedCoreNum_ = coreNum_;
    return ge::GRAPH_SUCCESS;
}

std::vector<int64_t> NsaSelectTiling::InitSparseValidArray(const int64_t *actualLens) const
{
    std::vector<int64_t> res(isMtpFlag_ ? (totalQseq_ * numKvHeads_) : (batchSize_ * numKvHeads_ * gOuter_));
    const int64_t *actualQLenData = context_->actualQSeqLengths.tensor->GetData<int64_t>();
    uint32_t idx = 0;
    for (uint32_t b = 0; b < batchSize_; b++) {
        int64_t s1BlockNum = actualQLenData != nullptr? CeilDivision(actualQLenData[b], static_cast<int64_t>(s1BasicBlock)) : 1;
        for (uint32_t n = 0; n < numKvHeads_ * gOuter_; n++) {
            int64_t estimatedLoad = static_cast<int64_t>(seqSize_);
            if (actualLens != nullptr) {
                estimatedLoad = actualLens[b] == 0 ? 1 : actualLens[b];
            }
            for (uint32_t j = 0; j < static_cast<uint32_t>(s1BlockNum); j++) {
                res[idx++] = estimatedLoad;
            }
        }
    }
    return res;
}
// code copy from flash_attention_score_tiling
bool NsaSelectTiling::BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single
    // core
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

void NsaSelectTiling::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                            const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue) const
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

void NsaSelectTiling::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize, int64_t validAivNum,
                                uint32_t *sparseStartIdx, int64_t splitFactorSize)
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
            avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
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

ge::graphStatus NsaSelectTiling::SplitBN_V0()
{
    uint32_t bn = batchSize_ * numKvHeads_ * gOuter_;
    if (isMtpFlag_) {
        bn = static_cast<uint32_t>(numKvHeads_) * totalQseq_;
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

    for (uint32_t i = 0; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = MAX_SPLIT_SIZE;
    uint32_t sInnerSize[3U] = {4096U, 2048U, 1024U};
    uint32_t idx = std::min(nNumOfQInOneGroup_ / 64U, 2U);
    sInnerSize_ = sInnerSize[idx];

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
    return ge::GRAPH_SUCCESS;
}

bool NsaSelectTiling::CalcUbBmm()
{
    headDimAlign_ = Align(headDimV_, 16U);
    mmResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * sInnerSizeAlign_ * qSeqSize_;
    bmm2ResUbSize_ = msdIterNum_ * nNumOfQInOneGroup_ * headDimAlign_ * qSeqSize_;
    return true;
}

ge::graphStatus NsaSelectTiling::CalcWorkSpace()
{
    uint32_t mmResElemSize = 4;         // 4:fp32
    uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4;       // 4:fp32
    uint32_t vec2ResElemSize = 4;       // 4:fp32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * coreNum_ * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>((static_cast<float>((mmResUbSize_ * coreNum_ * vec1ResElemSize)) * kvDtypeRatio));
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * coreNum_ * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * coreNum_ * vec2ResElemSize;
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::FillTiling()
{
    FillTilingBaseParams();
    FillTilingSplitKV();
    FillTilingCoreParams();
    FillTilingSingleCoreParams();
    FillTilingSingleCoreTensorSize();
    return ge::GRAPH_SUCCESS;
}

void NsaSelectTiling::FillTilingBaseParams() const
{
    tilingData_->baseParams.set_batchSize(batchSize_);
    tilingData_->baseParams.set_seqSize(sMax_);
    tilingData_->baseParams.set_qSeqSize(qSeqSize_);
    tilingData_->baseParams.set_headSize(headDim_);
    tilingData_->baseParams.set_headSizeV(headDimV_);
    tilingData_->baseParams.set_headSizeRope(headDim_ - headDimV_);
    tilingData_->baseParams.set_selectedBlockSize(selectedBlockSize_);
    tilingData_->baseParams.set_selectedBlockCount(selectedBlockCount_);
    tilingData_->baseParams.set_headSizeV(headDimV_);
    tilingData_->baseParams.set_blockSize(blockSize_);
    tilingData_->baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingData_->baseParams.set_scaleValue(scaleValue_);
    tilingData_->baseParams.set_kvHeadNum(numKvHeads_);
    tilingData_->baseParams.set_qHeadNum(numHeads_);
    tilingData_->baseParams.set_nNumOfQInOneGroup(numHeads_ / numKvHeads_);

    tilingData_->baseParams.set_actualLenDims(actualLenDims_);
    tilingData_->baseParams.set_msdIterNum(msdIterNum_);
    tilingData_->baseParams.set_attenMaskFlag(attenMaskFlag_ ? 1 : 0);
    tilingData_->baseParams.set_attenMaskSize(attenMaskSize_);
    tilingData_->baseParams.set_isMtpFlag(isMtpFlag_);
    tilingData_->baseParams.set_actualLenQDims(actualLenQDims_);
}

// for flash decode
void NsaSelectTiling::FillTilingSplitKV() const
{
    tilingData_->splitKVParams.set_s2(kvSplitPart_);
    int64_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1)) / kvSplitPart_;
    sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
    OP_LOGD(context_->opName, "PA FlashDecode is not enabled, sInnerLoopSize is %ld, blockSize is %ld",
                sInnerLoopSize_, blockSize_);
    tilingData_->splitKVParams.set_sInnerLoopSize(sInnerLoopSize_);
    if (!splitKVFlag_) {
        tilingData_->splitKVParams.set_s2(0);
    }
}

void NsaSelectTiling::FillTilingCoreParams()
{
    uint32_t *coreStartIdx = tilingData_->nsaSelectAttentionInferCoreParams.get_coreSidxEnd();
    memcpy_s(coreStartIdx, MAX_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_CORE_NUM * sizeof(uint32_t));
}

void NsaSelectTiling::FillTilingSingleCoreParams() const
{
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_sInnerLoopTimes(sInnerLoopTimes_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_singleProcessSInnerSize(sInnerSize_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_singleProcessSInnerSizeTail(sInnerSizeTail_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_usedCoreNum(usedCoreNum_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_formerCoreNum(formerCoreNum_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_blockSplitBn2Range(blockSplitBn2Range_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_tailSplitedBatchRange(tailSplitedBatchRange_);
    tilingData_->nsaSelectAttentionInferSingleCoreParams.set_groupSplitSize(groupSplitSize_);
}

void NsaSelectTiling::FillTilingSingleCoreTensorSize() const
{
    tilingData_->nsaSelectAttentionInferSingleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_->nsaSelectAttentionInferSingleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

ge::graphStatus NsaSelectTiling::GenTilingKey() const
{
    uint32_t layoutVal = 0U;
    uint32_t inputQVal = 0U;
    context_->tilingKey = layoutVal + inputQVal;
    std::string layout(context_->layOut);
    if (layout == "TND") {
        context_->tilingKey = 1U;
    } 
    OP_LOGD(context_->opName, "Nsa select tilingKey: %lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    auto aicNum = aicNum_;
    auto aivNum = aivNum_;
    aivNum = doubleAiv * usedCoreNum_;
    aicNum = usedCoreNum_;
    context_->blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGD(context_->opName, "Nsa select block dim: %u aiv Num: %u aic Num: %u", context_->blockDim, aivNum, aicNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::ConvertContext(gert::TilingContext &context, NsaSelectAttentionInferContext &nsaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("NsaSelectedAttentionInfer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    nsaContext.opName = context.GetNodeName();
    nsaContext.platformInfo = context.GetPlatformInfo();
    nsaContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    nsaContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    nsaContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    nsaContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_IF((nsaContext.query.shape == nullptr) || (nsaContext.key.shape == nullptr),
            OP_LOGE(context.GetNodeName(), "shape of query or shape of key is null."), return ge::GRAPH_FAILED);

    nsaContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    nsaContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);
    nsaContext.topkIndices.desc = context.GetInputDesc(TOPK_INDICES_INPUT_INDEX);
    nsaContext.topkIndices.shape = context.GetInputShape(TOPK_INDICES_INPUT_INDEX);

    nsaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
    nsaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);

    nsaContext.blockTable.desc = context.GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
    nsaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);

    nsaContext.actualQSeqLengths.desc = context.GetOptionalInputDesc(ACT_Q_SEQ_LEN_INPUT_INDEX);
    OP_CHECK_IF(nsaContext.actualQSeqLengths.desc == nullptr,
            OP_LOGE(context.GetNodeName(), "desc of QLISE  is null."), return ge::GRAPH_FAILED);
    nsaContext.actualQSeqLengths.tensor = context.GetOptionalInputTensor(ACT_Q_SEQ_LEN_INPUT_INDEX);

    nsaContext.actualKVSeqLengths.desc = context.GetOptionalInputDesc(ACT_KV_SEQ_LEN_INPUT_INDEX);
    OP_CHECK_IF(nsaContext.actualKVSeqLengths.desc == nullptr,
            OP_LOGE(context.GetNodeName(), "desc of KVLISE  is null."), return ge::GRAPH_FAILED);
    nsaContext.actualKVSeqLengths.tensor = context.GetOptionalInputTensor(ACT_KV_SEQ_LEN_INPUT_INDEX);

    nsaContext.attenOut.desc = context.GetOutputDesc(OUTPUT_INDEX);
    nsaContext.attenOut.shape = context.GetOutputShape(OUTPUT_INDEX);
    

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from GE is nullptr"),
            return ge::GRAPH_FAILED);
    nsaContext.layOut = attrs->GetStr(LAYOUT_ATTR_INDEX);
    nsaContext.numHeads = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);
    nsaContext.kvHeadNums = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
    nsaContext.selectedBlockSize = attrs->GetAttrPointer<int64_t>(SELECTED_BLOCK_SIZE_ATTR_INDEX);
    nsaContext.selectedBlockCount = attrs->GetAttrPointer<int64_t>(SELECTED_BLOCK_COUNT_ATTR_INDEX);
    nsaContext.blockSize = attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_ATTR_INDEX);
    nsaContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    nsaContext.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR("NsaSelectedAttentionInfer", "workSpaceSize got from GE is nullptr"),
            return ge::GRAPH_FAILED);
    nsaContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NsaSelectTiling::RunBigKernelTiling(NsaSelectAttentionInferContext &context,
                                            NsaSelectAttentionInferTilingDataV2 &tilingData, bool isWorkspace)
{
    this->context_ = &context;
    this->tilingData_ = &tilingData.tilingBase;
    this->isWorkspace_ = isWorkspace;
    if ((this->context_->actualKVSeqLengths.tensor && this->context_->actualKVSeqLengths.tensor->GetData<int64_t>()[0] == -1) &&
        (this->context_->actualQSeqLengths.tensor && this->context_->actualQSeqLengths.tensor->GetData<int64_t>()[0] == -1)) {
        this->isWorkspace_ = true;
        OP_LOGI(context_->opName, "NSA select attention infer aclgraph mode activate.");
    }
    if ((GetNpuInfo() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    if ((PreProcess() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    // user prompt tiling
    if ((ZeroTensorProcess() != ge::GRAPH_SUCCESS) ||
        (Split() != ge::GRAPH_SUCCESS) ||
        (FillTiling() != ge::GRAPH_SUCCESS) ||
        (CalcWorkSpace() != ge::GRAPH_SUCCESS) ||
        (CalcBlockDim() != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
        
    }
    return GenTilingKey();
}

ge::graphStatus NsaSelectTiling::NsaSelectAttentionInferSetTilingData(gert::TilingContext &context,
                                                            NsaSelectAttentionInferTilingDataV2 &tilingData) const
{
    OP_CHECK_IF(context.GetRawTilingData() == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR("NsaSelectedAttentionInfer", "RawTilingData got from GE context is nullptr."),
            return GRAPH_FAILED);
    tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingNsaSelectAttentionInferAdapter(gert::TilingContext *context, NsaSelectAttentionInferContext &nsaContext,
                                                NsaSelectAttentionInferTilingDataV2 &nsaTilingData)
{
    NsaSelectTiling nsaTilingNew;
    if (nsaTilingNew.RunBigKernelTiling(nsaContext, nsaTilingData) == ge::SUCCESS) {
        context->SetTilingKey(nsaContext.tilingKey);
        context->SetBlockDim(nsaContext.blockDim);
        nsaTilingNew.NsaSelectAttentionInferSetTilingData(*context, nsaTilingData);
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_FAILED;
}

Nsa_EXTERN_C ge::graphStatus TilingNsaSelectAttentionInfer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("NsaSelectedAttentionInfer", "Context is nullptr."),
            return ge::GRAPH_FAILED);
    NsaSelectAttentionInferTilingDataV2 tilingData;
    NsaSelectAttentionInferContext nsaContext{.opName = nullptr,
                                        .platformInfo = nullptr,
                                        .query = {nullptr, nullptr},
                                        .key = {nullptr, nullptr},
                                        .value = {nullptr, nullptr},
                                        .topkIndices = {nullptr, nullptr},
                                        .attenMask = {nullptr, nullptr},
                                        .blockTable = {nullptr, nullptr},
                                        .actualQSeqLengths = {nullptr, nullptr},
                                        .actualKVSeqLengths = {nullptr, nullptr},
                                        .attenOut = {nullptr, nullptr},
                                        .layOut = nullptr,
                                        .numHeads = nullptr,
                                        .numKVHeads = nullptr,
                                        .kvHeadNums = nullptr,
                                        .selectedBlockSize = nullptr,
                                        .selectedBlockCount = nullptr,
                                        .blockSize = nullptr,
                                        .scaleValue = nullptr,
                                        .sparseMode = nullptr,
                                        .workSpaces = nullptr,
                                        .tilingKey = 0,
                                        .blockDim = 0};
    if (NsaSelectTiling::ConvertContext(*context, nsaContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to nsa select context");
        return ge::GRAPH_FAILED;
    }
    return TilingNsaSelectAttentionInferAdapter(context, nsaContext, tilingData);
}
} // namespace optiling