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
 * \file nsa_compress_attention_infer_tiling.cpp
 * \brief
 */

#include "nsa_compress_attention_infer_tiling.h"
#include "nsa_compress_attention_infer_tiling_utils.h"
#include "register/op_def_registry.h"
#include "log/log.h"

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include <vector>
#include "op_host/data_copy_transpose_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {

static void ReportTilingFailure(const char* reason)
{
    constexpr const char* opName = "NsaCompressAttentionInfer";
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason);
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

ge::graphStatus NCAITiling::GetNpuInfo()
{
    OP_CHECK_IF(ncaiContext_->platformInfo == nullptr, ReportTilingFailure("Platform info is null."),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(this->ncaiContext_->platformInfo);
    OP_CHECK_IF(ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1009", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{ncaiContext_->opName,
                                              "Atlas300I (Ascend310P) is unsupported; use Atlas A2 or A3."})),
                return ge::GRAPH_FAILED);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckQkv()
{
    auto qDtype = ncaiContext_->query.desc->GetDataType();
    auto kDtype = ncaiContext_->key.desc->GetDataType();
    auto vDtype = ncaiContext_->value.desc->GetDataType();
    auto qDimNum = ncaiContext_->query.shape->GetStorageShape().GetDimNum();
    auto kDimNum = ncaiContext_->key.shape->GetStorageShape().GetDimNum();
    auto vDimNum = ncaiContext_->value.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(qDtype != ge::DataType::DT_FLOAT16 && qDtype != ge::DataType::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ncaiContext_->opName, "query",
                                                      ge::TypeUtils::DataTypeToSerialString(qDtype),
                                                      "Only float16 and bfloat16 are supported."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDtype != qDtype || vDtype != qDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ncaiContext_->opName, "query, key, value",
                                                       ge::TypeUtils::DataTypeToSerialString(qDtype) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(kDtype) + ", " +
                                                           ge::TypeUtils::DataTypeToSerialString(vDtype),
                                                       "Data types must match."),
                return ge::GRAPH_FAILED);
    if (strcmp(ncaiContext_->layOut, "TND") == 0) {
        OP_CHECK_IF(qDimNum != 3U,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ncaiContext_->opName, "query", std::to_string(qDimNum),
                                                             "TND layout requires 3 dimensions."),
                    return ge::GRAPH_FAILED);
    } else if (strcmp(ncaiContext_->layOut, "BSND") == 0) {
        OP_CHECK_IF(qDimNum != 4U,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ncaiContext_->opName, "query", std::to_string(qDimNum),
                                                             "BSND layout requires 4 dimensions."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "layoutOptional", ncaiContext_->layOut,
                                              "Only TND and BSND are supported.");
        return ge::GRAPH_FAILED;
    }
    if (pagedAttentionFlag_ == true) {
        OP_CHECK_IF(kDimNum != 3U || vDimNum != 3U,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        ncaiContext_->opName, "key, value", std::to_string(kDimNum) + ", " + std::to_string(vDimNum),
                        "Paged attention requires 3 dimensions for both tensors."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ncaiContext_->opName, "blockTableOptional",
                                                 "Only paged KV cache is supported.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckQkvShape()
{
    // query、blockTable、actualCmpKvSeqLen的第一维是否相同
    uint32_t queryDim0 = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_0);
    uint64_t batchSizeInBlockTable = ncaiContext_->blockTable.tensor->GetStorageShape().GetDim(DIM_0);
    uint64_t batchSizeInCmpKvSeqLen = ncaiContext_->actualCmpKvSeqLengths.tensor->GetStorageShape().GetDim(DIM_0);
    if (this->isWorkspace_) {
        OP_CHECK_IF(queryDim0 != batchSizeInBlockTable,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ncaiContext_->opName, "query.shape[0], blockTableOptional.shape[0]",
                        std::to_string(queryDim0) + ", " + std::to_string(batchSizeInBlockTable),
                        "Batch sizes must match in max workspace mode."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qSeqSize_ == 1 && queryDim0 != batchSizeInBlockTable,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ncaiContext_->opName, "query.shape[0], blockTableOptional.shape[0]",
                        std::to_string(queryDim0) + ", " + std::to_string(batchSizeInBlockTable),
                        "Batch sizes must match when query sequence length is 1."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(batchSizeInBlockTable != batchSizeInCmpKvSeqLen,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ncaiContext_->opName, "blockTableOptional.shape[0], actualCmpKvSeqLenOptional.shape[0]",
                        std::to_string(batchSizeInBlockTable) + ", " + std::to_string(batchSizeInCmpKvSeqLen),
                        "Batch sizes must match."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(batchSizeInBlockTable <= 0U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "blockTableOptional.shape[0]",
                                                      std::to_string(batchSizeInBlockTable),
                                                      "Batch size must be greater than 0."),
                return ge::GRAPH_FAILED);

    // query第二维是否等于 attr中的qHeadNum_
    uint32_t headNumInQ = 0;
    if (strcmp(ncaiContext_->layOut, "TND") == 0) {
        headNumInQ = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(
            qSeqSize_ > 1 && queryDim0 != qSeqLenCumSum_,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(ncaiContext_->opName, "query.shape[0]", std::to_string(queryDim0),
                                                      "TND token count must equal the sum of actualQSeqLenOptional "
                                                      "when the maximum query sequence length exceeds 1."),
            return ge::GRAPH_FAILED);
    } else if (strcmp(ncaiContext_->layOut, "BSND") == 0) {
        headNumInQ = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_2);
        uint32_t qSeqLen = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(qSeqLen != qSeqSize_,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        ncaiContext_->opName, "query.shape[1]", std::to_string(qSeqLen),
                        "BSND sequence size must equal the maximum query sequence length."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            qSeqSize_ > 1 && queryDim0 != batchSizeInBlockTable,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                ncaiContext_->opName, "query.shape[0]", std::to_string(queryDim0),
                "BSND batch size must match blockTableOptional when the maximum query sequence length exceeds 1."),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(headNumInQ != qHeadNum_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "query head count, numHeads",
                                                      std::to_string(headNumInQ) + ", " + std::to_string(qHeadNum_),
                                                      "Head counts must match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headNumInQ <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "query head count",
                                                      std::to_string(headNumInQ), "Head count must be greater than 0."),
                return ge::GRAPH_FAILED);

    // key的第一维 和 value的第一维是否相同
    uint32_t numBlocksInK = ncaiContext_->key.shape->GetStorageShape().GetDim(DIM_0);
    uint32_t numBlocksInV = ncaiContext_->value.shape->GetStorageShape().GetDim(DIM_0);
    OP_CHECK_IF(numBlocksInK != numBlocksInV,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ncaiContext_->opName, "key.shape[0], value.shape[0]",
                    std::to_string(numBlocksInK) + ", " + std::to_string(numBlocksInV), "Block counts must match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numBlocksInK <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "key.shape[0]", std::to_string(numBlocksInK),
                                              "Block count must be greater than 0."),
        return ge::GRAPH_FAILED);

    // key的第2维 和 value的第2维 和 参数中的pageBlockSize是否相同
    uint32_t blockSizeInK = ncaiContext_->key.shape->GetStorageShape().GetDim(DIM_1);
    uint32_t blockSizeInV = ncaiContext_->value.shape->GetStorageShape().GetDim(DIM_1);
    OP_CHECK_IF(
        blockSizeInK != blockSizeInV || blockSizeInV != static_cast<uint32_t>(*ncaiContext_->blockSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "key.shape[1], value.shape[1], pageBlockSize",
                                              std::to_string(blockSizeInK) + ", " + std::to_string(blockSizeInV) +
                                                  ", " + std::to_string(*ncaiContext_->blockSize),
                                              "Page block sizes must match."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckTopK()
{
    uint32_t selectKvSeqlenMax = (sMax_ - ONE) * compStrideD_ + compSizeL_;
    uint32_t selectBlockNumMax = (selectKvSeqlenMax + selectSize_ - ONE) / selectSize_;
    uint32_t selectKvSeqlenMin = (sMin_ - ONE) * compStrideD_ + compSizeL_;
    uint32_t selectBlockNumMin = (selectKvSeqlenMin + selectSize_ - ONE) / selectSize_;
    OP_CHECK_IF(selectBlockNumMax > 4096U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ncaiContext_->opName, "candidate block count", std::to_string(selectBlockNumMax),
                    "The maximum number of candidate blocks computed from the inputs must not exceed 4096."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectBlockNumMin < *ncaiContext_->selectNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ncaiContext_->opName, "selectBlockCount", std::to_string(*ncaiContext_->selectNum),
                    "Must not exceed the minimum candidate block count across batches."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckAttr()
{
    OP_CHECK_IF(compSizeL_ % 16U != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressBlockSize",
                                                      std::to_string(compSizeL_), "Must be a multiple of 16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compStrideD_ % 16U != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressStride",
                                                      std::to_string(compStrideD_), "Must be a multiple of 16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectSize_ % 16U != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "selectBlockSize",
                                                      std::to_string(selectSize_), "Must be a multiple of 16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compSizeL_ < compStrideD_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressStride, compressBlockSize",
                                              std::to_string(compStrideD_) + ", " + std::to_string(compSizeL_),
                                              "compressStride must be less than or equal to compressBlockSize."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compSizeL_ > selectSize_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressBlockSize, selectBlockSize",
                                              std::to_string(compSizeL_) + ", " + std::to_string(selectSize_),
                                              "compressBlockSize must be less than or equal to selectBlockSize."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(selectSize_ % compStrideD_ != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "selectBlockSize, compressStride",
                                                      std::to_string(selectSize_) + ", " + std::to_string(compStrideD_),
                                                      "selectBlockSize must be divisible by compressStride."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        compSizeL_ < 16U || compSizeL_ > 128U || compSizeL_ % 16U != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressBlockSize", std::to_string(compSizeL_),
                                              "Must be a multiple of 16 in [16, 128]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(compStrideD_ < 16U || compStrideD_ > 64U || compStrideD_ % 16U != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "compressStride",
                                                      std::to_string(compStrideD_), "Must be one of 16, 32, 48 or 64."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        selectSize_ < 16U || selectSize_ > 128U || selectSize_ % 16U != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "selectBlockSize", std::to_string(selectSize_),
                                              "Must be a multiple of 16 in [16, 128]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<uint32_t>(*ncaiContext_->blockSize) <= 0 || static_cast<uint32_t>(*ncaiContext_->blockSize) > 128U,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "pageBlockSize",
                                              std::to_string(*ncaiContext_->blockSize), "Must be in (0, 128]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        static_cast<uint32_t>(*ncaiContext_->blockSize) % 16U != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "pageBlockSize",
                                              std::to_string(*ncaiContext_->blockSize), "Must be a multiple of 16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*this->ncaiContext_->numHeads % *this->ncaiContext_->kvHeadNums != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ncaiContext_->opName, "numHeads, numKeyValueHeads",
                    std::to_string(*ncaiContext_->numHeads) + ", " + std::to_string(*ncaiContext_->kvHeadNums),
                    "numHeads must be divisible by numKeyValueHeads."),
                return ge::GRAPH_FAILED);
    if (CheckTopK() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckAttenMask()
{
    if (ncaiContext_->attenMask.tensor != nullptr) {
        attenMaskFlag_ = 1U;
        return ge::GRAPH_SUCCESS;
    }
    attenMaskFlag_ = 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::ParamsPostCheck()
{
    groupSize_ = qHeadNum_ / kvHeadNum_;
    OP_CHECK_IF(groupSize_ <= 0 || groupSize_ > 128U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "numHeads / numKeyValueHeads",
                                                      std::to_string(groupSize_), "The ratio must be in (0, 128]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(128 % groupSize_ != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "numHeads / numKeyValueHeads",
                                                      std::to_string(groupSize_), "The ratio must divide 128."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockSize_ <= 0 || blockSize_ > 128U,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "value.shape[1]", std::to_string(blockSize_),
                                              "Page block size must be in (0, 128]."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockSize_ % 16U != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "value.shape[1]", std::to_string(blockSize_),
                                              "Page block size must be a multiple of 16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(headSizeQk_ <= 0 || headSizeQk_ > 192U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "query head dimension",
                                                      std::to_string(headSizeQk_), "Head size must be in (0, 192]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headSizeVo_ <= 0 || headSizeVo_ > 128U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "value head dimension",
                                                      std::to_string(headSizeVo_), "Head size must be in (0, 128]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headSizeVo_ > headSizeQk_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "value, query head dimensions",
                                                      std::to_string(headSizeVo_) + ", " + std::to_string(headSizeQk_),
                                                      "Value head size must be less than or equal to query head size."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sMax_ <= 0 || sMax_ > SEQLEN_LIMIT,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ncaiContext_->opName, "actualCmpKvSeqLenOptional maximum",
                                              std::to_string(sMax_), "Maximum sequence length must be in (0, 8192]."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::ProcessQkv()
{
    if (CheckQkv() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    batchSize_ = ncaiContext_->blockTable.tensor->GetStorageShape().GetDim(DIM_0);
    qHeadNum_ = static_cast<uint32_t>(*this->ncaiContext_->numHeads);
    kvHeadNum_ = static_cast<uint32_t>(*this->ncaiContext_->kvHeadNums);
    scaleValue_ = *this->ncaiContext_->scaleValue;
    qSeqSize_ = 1U;
    if (pagedAttentionFlag_ == true) {
        if (strcmp(ncaiContext_->layOut, "TND") == 0) {
            headSizeQk_ = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_2);
        } else if (strcmp(ncaiContext_->layOut, "BSND") == 0) {
            headSizeQk_ = ncaiContext_->query.shape->GetStorageShape().GetDim(DIM_3);
        }
        headSizeVo_ = (ncaiContext_->value.shape->GetStorageShape().GetDim(DIM_2)) / kvHeadNum_;
        blockSize_ = ncaiContext_->value.shape->GetStorageShape().GetDim(DIM_1);
        maxBlockNumPerBatch_ = ncaiContext_->blockTable.tensor->GetStorageShape().GetDim(DIM_1);
        OP_CHECK_IF(GetMaxMinSeqlen() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(ncaiContext_->opName, "get max/min kv seqlen failed"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(GetMaxQSeqlen() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(ncaiContext_->opName, "get max/min q seqlen failed"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckSelKvSeqlen() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(ncaiContext_->opName, "Check selectKvSeqlen failed"),
                    return ge::GRAPH_FAILED);
    } else {
        return ge::GRAPH_FAILED;
    }
    if (CheckQkvShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::ProcessEpilogue()
{
    selectSize_ = static_cast<uint32_t>(*this->ncaiContext_->selectSize);
    selectNum_ = static_cast<uint32_t>(*this->ncaiContext_->selectNum);
    compSizeL_ = static_cast<uint32_t>(*this->ncaiContext_->compSizeL);
    compStrideD_ = static_cast<uint32_t>(*this->ncaiContext_->compStrideD);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::ProcessInput()
{
    if (ProcessEpilogue() != ge::GRAPH_SUCCESS || ProcessQkv() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttenMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::SplitBN()
{
    uint32_t corePerBatch = (aicNum_ + batchSize_ - ONE) / batchSize_;
    uint32_t kvHeadSplitSize = (kvHeadNum_ + corePerBatch - ONE) / corePerBatch;
    groupSize_ = qHeadNum_ / kvHeadNum_;
    uint32_t kvHeadSplitSizeLimit = ROW_LIMIT_QP / groupSize_;
    kvHeadSplitSize_ = (kvHeadSplitSize > kvHeadSplitSizeLimit) ? kvHeadSplitSizeLimit : kvHeadSplitSize;
    kvHeadSplitNum_ = (kvHeadNum_ + kvHeadSplitSize_ - ONE) / kvHeadSplitSize_;

    uint32_t qSeqLenSplitSizeLimit = kvHeadSplitSizeLimit / kvHeadSplitSize_;
    uint32_t totalQSplitNum = 0;
    int64_t actQSeqLen = 0;
    for (uint32_t i = 0; i < batchSize_; i++) {
        if (ncaiContext_->actualQSeqLengths.tensor->GetData<int64_t>() != nullptr && qSeqSize_ != 1U) {
            const int64_t* actualQSeqLenData = ncaiContext_->actualQSeqLengths.tensor->GetData<int64_t>();
            actQSeqLen = actualQSeqLenData[i];
        } else {
            actQSeqLen = 1;
        }
        uint32_t qSeqLenSplitNum =
            (static_cast<uint32_t>(actQSeqLen) + qSeqLenSplitSizeLimit - ONE) / qSeqLenSplitSizeLimit;
        if (qSeqLenSplitNum > 1U) {
            qSeqLenSplitSize_ = qSeqLenSplitSizeLimit;
        } else {
            qSeqLenSplitSize_ = (static_cast<uint32_t>(actQSeqLen) > qSeqLenSplitSize_) ?
                                    static_cast<uint32_t>(actQSeqLen) :
                                    qSeqLenSplitSize_;
        }
        totalQSplitNum += qSeqLenSplitNum;
    }
    processNum_ = kvHeadSplitNum_ * totalQSplitNum;
    processPerBatch_ = kvHeadSplitNum_; // 调测使用，需要删掉
    coreNumUsed_ = std::max(std::min(aicNum_, processNum_), ONE);
    ncaiContext_->blockDim = coreNumUsed_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::Split()
{
    return SplitBN();
}

ge::graphStatus NCAITiling::FillTilingData()
{
    tilingData_->baseParams.set_batchSize(batchSize_);
    tilingData_->baseParams.set_seqSize(sMax_);
    tilingData_->baseParams.set_qSeqSize(qSeqSize_);
    tilingData_->baseParams.set_qHeadNum(qHeadNum_);
    tilingData_->baseParams.set_kvHeadNum(kvHeadNum_);
    tilingData_->baseParams.set_headSizeQk(headSizeQk_);
    tilingData_->baseParams.set_headSizeVo(headSizeVo_);
    tilingData_->baseParams.set_blockSize(blockSize_);
    tilingData_->baseParams.set_groupSize(groupSize_);
    tilingData_->baseParams.set_maxBlockNumPerBatch(maxBlockNumPerBatch_);
    tilingData_->baseParams.set_scaleValue(scaleValue_);
    tilingData_->baseParams.set_attenMaskFlag(attenMaskFlag_);
    tilingData_->baseParams.set_selectSize(selectSize_);
    tilingData_->baseParams.set_selectNum(selectNum_);
    tilingData_->baseParams.set_compSizeL(compSizeL_);
    tilingData_->baseParams.set_compStrideD(compStrideD_);
    tilingData_->baseParams.set_ubSize(ubSize_);
    tilingData_->baseParams.set_workSpaceElemNum(workSpaceElemNum_);
    tilingData_->baseParams.set_mm1ResWorkSpaceSize(mm1ResWorkSpaceSize_);
    tilingData_->baseParams.set_mm2InWorkSpaceSize(mm2InWorkSpaceSize_);
    tilingData_->baseParams.set_scoreInWorkSpaceSize(scoreInWorkSpaceSize_);
    tilingData_->baseParams.set_topKInWorkSpaceSize(topKInWorkSpaceSize_);

    tilingData_->splitBNParams.set_coreNumUsed(coreNumUsed_);
    tilingData_->splitBNParams.set_kvHeadSplitSize(kvHeadSplitSize_);
    tilingData_->splitBNParams.set_kvHeadSplitNum(kvHeadSplitNum_);
    tilingData_->splitBNParams.set_qSeqLenSplitSize(qSeqLenSplitSize_);
    tilingData_->splitBNParams.set_processPerBatch(processPerBatch_);
    tilingData_->splitBNParams.set_processNum(processNum_);

    tilingData_->splitBNParams.set_bnTotalSize(bnTotalSize_);
    tilingData_->splitBNParams.set_headCoreNum(headCoreNum_);
    tilingData_->splitBNParams.set_tailCoreNum(tailCoreNum_);
    tilingData_->splitBNParams.set_bnPerHeadCore(bnPerHeadCore_);
    tilingData_->splitBNParams.set_bnPerTailCore(bnPerTailCore_);

    tilingData_->splitBNParams.set_rowLenPerHeadCore(rowLenPerHeadCore_);
    tilingData_->splitBNParams.set_rowLenPerTailCore(rowLenPerTailCore_);
    tilingData_->splitBNParams.set_baseRowLenHeadCore(baseRowLenHeadCore_);
    tilingData_->splitBNParams.set_baseRowLenTailCore(baseRowLenTailCore_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CalcWorkSpace()
{
    // workspace总共分为四部分：S(fp32, db), P(fp32), P(fp16/bf16)
    groupSize_ = qHeadNum_ / kvHeadNum_;
    uint32_t vectorNumPerCube = 2;
    uint32_t kvHeadqSeqLenSplitSizeAlign =
        (kvHeadSplitSize_ * qSeqLenSplitSize_ + vectorNumPerCube - ONE) / vectorNumPerCube * vectorNumPerCube;
    uint32_t rowNumSp = groupSize_ * kvHeadqSeqLenSplitSizeAlign;
    uint32_t colNumSp = (sMax_ + BLOCK_SIZE - ONE) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t mm1ResElemSize = 4U;
    uint32_t mm2InElemSize = 2U;
    uint32_t scoreInElemSize = 4U;
    uint32_t bufferNum = 2U;
    workSpaceSize_ = libapiSize_;
    blockDim_ = coreNumUsed_;
    workSpaceElemNum_ = rowNumSp * colNumSp;
    mm1ResWorkSpaceSize_ = workSpaceElemNum_ * mm1ResElemSize * bufferNum * blockDim_;

    mm2InWorkSpaceSize_ = workSpaceElemNum_ * mm2InElemSize * bufferNum * blockDim_; // mm2In开db
    scoreInWorkSpaceSize_ = workSpaceElemNum_ * scoreInElemSize * blockDim_;
    topKInWorkSpaceSize_ = workSpaceElemNum_ * scoreInElemSize * blockDim_ / groupSize_;

    workSpaceSize_ += mm1ResWorkSpaceSize_ + mm2InWorkSpaceSize_ + scoreInWorkSpaceSize_ + topKInWorkSpaceSize_;
    if (ncaiContext_->workSpaces) {
        ncaiContext_->workSpaces[0] = workSpaceSize_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::SoftmaxTiling()
{
    ge::Shape srcShape;
    uint32_t alignedColLen =
        (sMax_ + COLLEN_ALIGNED_REQUIRED - ONE) / COLLEN_ALIGNED_REQUIRED * COLLEN_ALIGNED_REQUIRED;
    uint32_t used_ubSize = static_cast<uint32_t>(ubSize_) / SINGLE_UB_SIZE_BFLOAT16;
    uint32_t dataNumSingleUb = used_ubSize / SOFTMAX_INPUT_DATA_BYTE;
    uint32_t ubAvail = dataNumSingleUb / alignedColLen;
    uint32_t groupNumPerHeadCore = (kvHeadSplitSize_ + NUM_VECTOR_PER_CUBE - ONE) / NUM_VECTOR_PER_CUBE;
    uint32_t groupNumPerTailCore = kvHeadSplitSize_ / NUM_VECTOR_PER_CUBE;
    rowLenPerHeadCore_ = groupNumPerHeadCore * groupSize_;
    rowLenPerTailCore_ = groupNumPerTailCore * groupSize_;
    baseRowLenHeadCore_ = std::min(std::min(ubAvail, rowLenPerHeadCore_), COMPARE_INT);
    baseRowLenTailCore_ = std::min(std::min(ubAvail, rowLenPerTailCore_), COMPARE_INT);

    std::vector<int64_t> shapeVec = {std::max(baseRowLenHeadCore_, baseRowLenTailCore_), alignedColLen};
    srcShape = ge::Shape(shapeVec);
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, SOFTMAX_INPUT_DATA_BYTE, false);
    AscendC::SoftMaxTilingFunc(srcShape, SOFTMAX_INPUT_DATA_BYTE, localWorkSpaceSize, tilingData_->softmaxTilingData);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::TopKTiling()
{
    // 调用topk接口
    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    uint32_t dtypesize = 4; // float类型
    uint32_t selectKvSeqlen = (sMax_ - ONE) * compStrideD_ + compSizeL_;
    uint32_t selectBlockNum = (selectKvSeqlen + selectSize_ - ONE) / selectSize_;
    uint32_t alignedTopkIn = (selectBlockNum + ALIGNED_32 - ONE) / ALIGNED_32 * ALIGNED_32;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(this->ncaiContext_->platformInfo);
    AscendC::TopKTilingFunc(ascendcPlatform, alignedTopkIn, 1, selectNum_, dtypesize, true,
                            AscendC::TopKMode::TOPK_NORMAL, true, tilingData_->topkTilingData);
    AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, alignedTopkIn, 1, false, true, AscendC::TopKMode::TOPK_NORMAL, true,
                                  dtypesize, maxsize, minsize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CalcTilingKey()
{
    auto qDtype = ncaiContext_->query.desc->GetDataType();
    auto kDtype = ncaiContext_->key.desc->GetDataType();

    switch (qDtype) {
        case ge::DataType::DT_FLOAT16:
            break;
        case ge::DataType::DT_BF16:
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ncaiContext_->opName, "query",
                                                  ge::TypeUtils::DataTypeToSerialString(qDtype),
                                                  "Only float16 and bfloat16 are supported.");
            return ge::GRAPH_FAILED;
    }
    switch (kDtype) {
        case ge::DataType::DT_FLOAT16:
            break;
        case ge::DataType::DT_BF16:
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ncaiContext_->opName, "key",
                                                  ge::TypeUtils::DataTypeToSerialString(kDtype),
                                                  "Only float16 and bfloat16 are supported.");
            return ge::GRAPH_FAILED;
    }
    uint32_t tilingKey = 0;
    if (strcmp(ncaiContext_->layOut, "BSND") == 0) {
        tilingKey = 1U;
    }
    ncaiContext_->tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::ConvertContext(gert::TilingContext& context, NsaCompressAttentionInferContext& ncaiContext)
{
    if (context.GetNodeName() == nullptr) {
        ReportTilingFailure("Tiling context node name is null.");
        return ge::GRAPH_FAILED;
    }
    ncaiContext.opName = context.GetNodeName();
    ncaiContext.platformInfo = context.GetPlatformInfo();
    ncaiContext.query.desc = context.GetInputDesc(QUERY_INPUT_INDEX);
    ncaiContext.query.shape = context.GetInputShape(QUERY_INPUT_INDEX);
    ncaiContext.key.desc = context.GetInputDesc(KEY_INPUT_INDEX);
    ncaiContext.key.shape = context.GetInputShape(KEY_INPUT_INDEX);
    ncaiContext.value.desc = context.GetInputDesc(VALUE_INPUT_INDEX);
    ncaiContext.value.shape = context.GetInputShape(VALUE_INPUT_INDEX);

    ncaiContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INPUT_INDEX);
    ncaiContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INPUT_INDEX);
    ncaiContext.blockTable.desc = context.GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
    ncaiContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    ncaiContext.actualQSeqLengths.desc = context.GetOptionalInputDesc(ACT_Q_SEQ_LEN_INPUT_INDEX);
    ncaiContext.actualQSeqLengths.tensor = context.GetOptionalInputTensor(ACT_Q_SEQ_LEN_INPUT_INDEX);
    ncaiContext.actualCmpKvSeqLengths.desc = context.GetOptionalInputDesc(ACT_CMP_KV_SEQ_LEN_INPUT_INDEX);
    ncaiContext.actualCmpKvSeqLengths.tensor = context.GetOptionalInputTensor(ACT_CMP_KV_SEQ_LEN_INPUT_INDEX);
    ncaiContext.actualSelKvSeqLengths.desc = context.GetOptionalInputDesc(ACT_SEL_KV_SEQ_LEN_INPUT_INDEX);
    ncaiContext.actualSelKvSeqLengths.tensor = context.GetOptionalInputTensor(ACT_SEL_KV_SEQ_LEN_INPUT_INDEX);
    ncaiContext.topkMask.desc = context.GetOptionalInputDesc(TOPK_MASK_INPUT_INDEX);
    ncaiContext.topkMask.tensor = context.GetOptionalInputTensor(TOPK_MASK_INPUT_INDEX);

    ncaiContext.attenOut.desc = context.GetOutputDesc(ATTEN_OUTPUT_INDEX);
    ncaiContext.attenOut.shape = context.GetOutputShape(ATTEN_OUTPUT_INDEX);
    ncaiContext.selectOut.desc = context.GetOutputDesc(SELECT_OUTPUT_INDEX);
    ncaiContext.selectOut.shape = context.GetOutputShape(SELECT_OUTPUT_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportTilingFailure("Tiling context attributes are null."), return ge::GRAPH_FAILED);
    ncaiContext.numHeads = attrs->GetAttrPointer<int64_t>(NUM_HEADS_ATTR_INDEX);
    ncaiContext.kvHeadNums = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
    ncaiContext.selectSize = attrs->GetAttrPointer<int64_t>(SELECT_SIZE_ATTR_INDEX);
    ncaiContext.selectNum = attrs->GetAttrPointer<int64_t>(SELECT_NUM_ATTR_INDEX);
    ncaiContext.compSizeL = attrs->GetAttrPointer<int64_t>(COMP_SIZE_ATTR_INDEX);
    ncaiContext.compStrideD = attrs->GetAttrPointer<int64_t>(COMP_STRIDE_ATTR_INDEX);
    ncaiContext.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    ncaiContext.layOut = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    ncaiContext.blockSize = attrs->GetAttrPointer<int64_t>(BLOCK_SIZE_ATTR_INDEX);
    ncaiContext.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                ReportTilingFailure("Tiling context workspace sizes are null."), return ge::GRAPH_FAILED);
    ncaiContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::GetNCAITiling(NsaCompressAttentionInferContext& ncaiContext,
                                          NsaCompressAttentionInferTilingData& tilingData, bool isWorkspace)
{
    this->ncaiContext_ = &ncaiContext;
    this->tilingData_ = &tilingData;
    this->isWorkspace_ = isWorkspace;

    // 这里加上判断actualCmpKvSeqLengths如果有tensor但没值，isworkspace置为true
    if (this->ncaiContext_->actualCmpKvSeqLengths.tensor &&
        this->ncaiContext_->actualCmpKvSeqLengths.tensor->GetData<int64_t>()[0] == -1) {
        this->isWorkspace_ = true;
        OP_LOGI(ncaiContext_->opName, "NSA compress attention infer aclgraph mode activate.");
    }

    if (GetNpuInfo() != ge::GRAPH_SUCCESS || ProcessInput() != ge::GRAPH_SUCCESS || Split() != ge::GRAPH_SUCCESS ||
        ParamsPostCheck() != ge::GRAPH_SUCCESS || SoftmaxTiling() != ge::GRAPH_SUCCESS ||
        TopKTiling() != ge::GRAPH_SUCCESS || CalcWorkSpace() != ge::GRAPH_SUCCESS ||
        FillTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CalcTilingKey();
}

ge::graphStatus NCAITiling::GetMaxMinSeqlen()
{
    if (batchSize_ == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            ncaiContext_->opName, "blockTableOptional.shape[0]", std::to_string(batchSize_),
            "Batch size must be greater than 0 to read the compressed KV sequence lengths.");
        return ge::GRAPH_FAILED;
    }
    // isworkspace为true，手动赋值
    if (this->isWorkspace_) {
        sMax_ = SEQLEN_LIMIT;
        sMin_ = SEQLEN_LIMIT;
    } else {
        const int64_t* actualLenData = ncaiContext_->actualCmpKvSeqLengths.tensor->GetData<int64_t>();
        sMin_ = static_cast<uint32_t>(actualLenData[0]);
        for (uint32_t i = 0; i < batchSize_; i++) {
            int64_t actLen = actualLenData[i];
            sMax_ = sMax_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : sMax_;
            sMin_ = sMin_ > static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : sMin_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::GetMaxQSeqlen()
{
    if (ncaiContext_->actualQSeqLengths.tensor->GetData<int64_t>() != nullptr) {
        const int64_t* actualQSeqLenData = ncaiContext_->actualQSeqLengths.tensor->GetData<int64_t>();
        int64_t batchSizeInQSeqLen = ncaiContext_->actualQSeqLengths.tensor->GetStorageShape().GetDim(DIM_0);
        if (ncaiContext_->blockTable.tensor->GetStorageShape().GetDim(DIM_0) != batchSizeInQSeqLen) {
            OP_LOGW(ncaiContext_->opName, "batchSize in blockTable, actualQSeqLengths must be equal");
            qSeqLenCumSum_ = batchSize_;
            qSeqSize_ = 1U;
            return ge::GRAPH_SUCCESS;
        }
        for (uint32_t i = 0; i < batchSize_; i++) {
            int64_t actQSeqLen = actualQSeqLenData[i];
            if (actQSeqLen < MIN_ACTUALQSEQLEN || actQSeqLen > MAX_ACTUALQSEQLEN) {
                OP_LOGW(ncaiContext_->opName, "qSeqLen only supports [1, 4]");
                qSeqLenCumSum_ = batchSize_;
                qSeqSize_ = 1U;
                return ge::GRAPH_SUCCESS;
            }
            qSeqSize_ = qSeqSize_ < static_cast<uint32_t>(actQSeqLen) ? static_cast<uint32_t>(actQSeqLen) : qSeqSize_;
            qSeqLenCumSum_ += static_cast<uint32_t>(actQSeqLen);
        }
    } else {
        qSeqLenCumSum_ = batchSize_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::CheckSelKvSeqlen()
{
    if (qSeqSize_ == 1U) {
        return ge::GRAPH_SUCCESS;
    }
    if (ncaiContext_->actualCmpKvSeqLengths.tensor != nullptr &&
        ncaiContext_->actualSelKvSeqLengths.tensor != nullptr) {
        int64_t batchSizeInSelKvSeqLen = ncaiContext_->actualSelKvSeqLengths.tensor->GetStorageShape().GetDim(DIM_0);
        if (ncaiContext_->blockTable.tensor->GetStorageShape().GetDim(DIM_0) != batchSizeInSelKvSeqLen) {
            OP_LOGW(ncaiContext_->opName, "batchSize in blockTable, batchSizeInSelKvSeqLen must be equal");
            qSeqLenCumSum_ = batchSize_;
            qSeqSize_ = 1U;
            return ge::GRAPH_SUCCESS;
        }
        const int64_t* actualKvSeqLenData = ncaiContext_->actualCmpKvSeqLengths.tensor->GetData<int64_t>();
        const int64_t* actualSelKvSeqLenData = ncaiContext_->actualSelKvSeqLengths.tensor->GetData<int64_t>();
        for (uint32_t i = 0; i < batchSize_; i++) {
            int64_t actKvSeqLen = actualKvSeqLenData[i];
            int64_t actSelKvSeqLen = actualSelKvSeqLenData[i];
            int64_t validKvSeqLen =
                (actSelKvSeqLen - static_cast<int64_t>(compSizeL_)) / static_cast<int64_t>(compStrideD_) +
                static_cast<int64_t>(ONE);
            if (actKvSeqLen != validKvSeqLen) {
                OP_LOGW(
                    ncaiContext_->opName,
                    "actualCmpKvSeqLen must equal '(actualSelKvSeqLen - compressBlockSize)/compressBlockStride + 1'");
                qSeqLenCumSum_ = batchSize_;
                qSeqSize_ = 1U;
                return ge::GRAPH_SUCCESS;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NCAITiling::NSICASetTilingData(gert::TilingContext& context,
                                               NsaCompressAttentionInferTilingData& tilingData)
{
    OP_CHECK_IF(context.GetRawTilingData() == nullptr, ReportTilingFailure("Raw tiling data is null."),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNsaCompressAttentionInfer(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, ReportTilingFailure("Tiling context is null."), return ge::GRAPH_FAILED);
    NsaCompressAttentionInferTilingData tilingData;
    NsaCompressAttentionInferContext ncaiContext{.opName = nullptr,
                                                 .platformInfo = nullptr,
                                                 .query = {nullptr, nullptr},
                                                 .key = {nullptr, nullptr},
                                                 .value = {nullptr, nullptr},
                                                 .attenMask = {nullptr, nullptr},
                                                 .blockTable = {nullptr, nullptr},
                                                 .actualQSeqLengths = {nullptr, nullptr},
                                                 .actualCmpKvSeqLengths = {nullptr, nullptr},
                                                 .actualSelKvSeqLengths = {nullptr, nullptr},
                                                 .topkMask = {nullptr, nullptr},
                                                 .attenOut = {nullptr, nullptr},
                                                 .selectOut = {nullptr, nullptr},
                                                 .numHeads = nullptr,
                                                 .kvHeadNums = nullptr,
                                                 .selectSize = nullptr,
                                                 .selectNum = nullptr,
                                                 .compSizeL = nullptr,
                                                 .compStrideD = nullptr,
                                                 .scaleValue = nullptr,
                                                 .layOut = nullptr,
                                                 .blockSize = nullptr,
                                                 .sparseMode = nullptr,
                                                 .workSpaces = nullptr,
                                                 .tilingKey = 0,
                                                 .blockDim = 1};
    if (NCAITiling::ConvertContext(*context, ncaiContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Error occurred while converting tilingContext to ncai context");
        return ge::GRAPH_FAILED;
    }
    NCAITiling ncaiTiling;
    if (ncaiTiling.GetNCAITiling(ncaiContext, tilingData) == ge::SUCCESS) {
        context->SetTilingKey(ncaiContext.tilingKey);
        context->SetBlockDim(ncaiContext.blockDim);
        ncaiTiling.NSICASetTilingData(*context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNsaCompressAttentionInfer(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NsaCompressAttentionInfer)
    .Tiling(TilingNsaCompressAttentionInfer)
    .TilingInputsDataDependency({5, 6, 7},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .TilingParse<NsaCompressAttentionInferCompileInfo>(
        TilingPrepareForNsaCompressAttentionInfer); // 向框架注册入口函数;
} // namespace optiling
