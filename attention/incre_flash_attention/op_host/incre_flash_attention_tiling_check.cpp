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
 * \file incre_flash_attention_tiling_check.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "incre_flash_attention_tiling_impl.h"
#include "incre_flash_attention_tiling_base.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "incre_flash_attention_tiling_check_mla_gqa.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {
const std::map<ge::Format, std::string> kFormatToStringMap = {
    {ge::FORMAT_NCHW, "NCHW"},
    {ge::FORMAT_NHWC, "NHWC"},
    {ge::FORMAT_ND, "ND"},
    {ge::FORMAT_NCDHW, "NCDHW"},
};

// 设备侧下沉库不提供 ge::TypeUtils::FormatToSerialString，直接引用会形成强未定义符号，
// 在 -Wl,-z,now 链接的下沉库中会导致 dlopen 整库加载失败，此处使用库内等价实现。
inline std::string FormatToSerialString(const ge::Format format)
{
    const auto it = kFormatToStringMap.find(static_cast<ge::Format>(ge::GetPrimaryFormat(format)));
    if (it != kFormatToStringMap.end()) {
        if (ge::HasSubFormat(format)) {
            return it->second + ":" + std::to_string(ge::GetSubFormat(format));
        }
        return it->second;
    }
    OP_LOGW("IncreFlashAttention", "[Check][Param] Format not support %d", format);
    return "RESERVED";
}
} // namespace

ge::graphStatus IFATiling::CheckPABlockSize() const
{
    OP_CHECK_IF(
        blockSize_ == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                              "When Page Attention is enabled, blockSize can not be 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockSize_ > MAX_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
            ("When Page Attention is enabled, blockSize can not be larger than " + std::to_string(MAX_BLOCK_SIZE) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                    "When Page Attention is enabled, if kv cache dtype is int8, blockSize should be 32 aligned."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputKvType_ == ge::DT_FLOAT16) || (inputKvType_ == ge::DT_BF16)) && (blockSize_ % 16 != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                    "When Page Attention is enabled, if kv cache dtype is float16/bfloat16, blockSize should be 16 "
                    "aligned."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckBaseInputsNull() const
{
    // Check base input tensors
    OP_CHECK_IF(ifaContext_->query.shape == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Shape of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->query.shape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "query", "query is empty cause shapesize is 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->query.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Desc of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.shape == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Shape of tensor key is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Desc of tensor key is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.shape == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Shape of tensor value is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Desc of tensor value is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Desc of tensor output is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->attenOut.shape == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Shape of tensor output is nullptr"),
                return ge::GRAPH_FAILED);

    // Check base input attrs
    OP_CHECK_IF(ifaContext_->numHeads == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "numHeads"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->scaleValue == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "scaleValue"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvHeadNums == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "kvHeadNums"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->layOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "layOut"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->blockSize == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "blockSize"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::IsSupportFormat(const ge::Format format) const
{
    if (format == ge::FORMAT_ND) {
        return true;
    }
    if (format == ge::FORMAT_NCHW) {
        return true;
    }
    if (format == ge::FORMAT_NHWC) {
        return true;
    }
    if (format == ge::FORMAT_NCDHW) {
        return true;
    }
    return false;
}

bool IFATiling::IsZeroDimTensor(const gert::Shape shape) const
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

std::string IFATiling::GetTensorDimString(const gert::Shape shape) const
{
    std::stringstream ss;
    ss << "[";
    bool isFirst = true;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (!isFirst) {
            ss << ", ";
        }
        ss << std::to_string(shape.GetDim(i));
        isFirst = false;
    }
    ss << "]";
    return ss.str();
}

ge::graphStatus IFATiling::CheckInputParameterFormat() const
{
    auto qFormat = ifaContext_->query.desc->GetOriginFormat();
    auto kFormat = ifaContext_->key.desc->GetOriginFormat();
    auto vFormat = ifaContext_->value.desc->GetOriginFormat();

    OP_CHECK_IF(
        (!IsSupportFormat(qFormat)),
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "query", FormatToSerialString(qFormat).c_str(),
                                               "the format of query should be ND/NCHW/NHWC/NCDHW."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!IsSupportFormat(kFormat)),
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "key", FormatToSerialString(kFormat).c_str(),
                                               "the format of key should be ND/NCHW/NHWC/NCDHW."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!IsSupportFormat(vFormat)),
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "value", FormatToSerialString(vFormat).c_str(),
                                               "the format of value should be ND/NCHW/NHWC/NCDHW."),
        return ge::GRAPH_FAILED);
    if (ifaContext_->attenMask.desc != nullptr) {
        auto mFormat = ifaContext_->attenMask.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(mFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "attenMask",
                                                           FormatToSerialString(mFormat).c_str(),
                                                           "the format of attenMask should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->kvPaddingSize.desc != nullptr) {
        auto kvPaddingFormat = ifaContext_->kvPaddingSize.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kvPaddingFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "kvPaddingSize",
                                                           FormatToSerialString(kvPaddingFormat).c_str(),
                                                           "the format of kvPaddingSize should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->keySharedPrefix.desc != nullptr) {
        auto kPrefixFormat = ifaContext_->keySharedPrefix.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kPrefixFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix", FormatToSerialString(kPrefixFormat).c_str(),
                        "the format of keySharedPrefix should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->valueSharedPrefix.desc != nullptr) {
        auto vPrefixFormat = ifaContext_->valueSharedPrefix.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vPrefixFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "valueSharedPrefix", FormatToSerialString(vPrefixFormat).c_str(),
                        "the format of valueSharedPrefix should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputAntiquantFormat() const
{
    if (ifaContext_->antiquantScale.desc != nullptr) {
        auto aScaleFormat = ifaContext_->antiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(aScaleFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "antiquantScale", FormatToSerialString(aScaleFormat).c_str(),
                        "the format of antiquantScale should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->antiquantOffset.desc != nullptr) {
        auto aOffsetFormat = ifaContext_->antiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(aOffsetFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "antiquantOffset", FormatToSerialString(aOffsetFormat).c_str(),
                        "the format of antiquantOffset should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->keyAntiquantScale.desc != nullptr) {
        auto kScaleFormat = ifaContext_->keyAntiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kScaleFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "keyAntiquantScale", FormatToSerialString(kScaleFormat).c_str(),
                        "the format of the key's dequant scale should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->keyAntiquantOffset.desc != nullptr) {
        auto kOffsetFormat = ifaContext_->keyAntiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(kOffsetFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "keyAntiquantOffset", FormatToSerialString(kOffsetFormat).c_str(),
                        "the format of the key's dequant offset should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->valueAntiquantScale.desc != nullptr) {
        auto vScaleFormat = ifaContext_->valueAntiquantScale.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vScaleFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "valueAntiquantScale", FormatToSerialString(vScaleFormat).c_str(),
                        "the format of the value's dequant scale should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    if (ifaContext_->valueAntiquantOffset.desc != nullptr) {
        auto vOffsetFormat = ifaContext_->valueAntiquantOffset.desc->GetOriginFormat();
        OP_CHECK_IF((!IsSupportFormat(vOffsetFormat)),
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "valueAntiquantOffset", FormatToSerialString(vOffsetFormat).c_str(),
                        "the format of the value's dequant offset should be ND/NCHW/NHWC/NCDHW."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static const size_t KV_CACHE_DIM_NUM = 4;
static const size_t BLOCK_TABLE_DIM_NUM = 2;

ge::graphStatus IFATiling::ProcessCheckATBInputWithPage() const
{
    auto blockTablesShape = ifaContext_->blockTable.tensor->GetStorageShape();
    ge::DataType inputBlockTableType_ = ifaContext_->blockTable.desc->GetDataType();
    uint64_t taskNumI64 = static_cast<uint64_t>(numHeads_) * batchSize_;
    OP_CHECK_IF((inputBlockTableType_ != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "blockTable",
                                                      DataTypeToSerialString(inputKvType_).c_str(),
                                                      "blockTable dtype should be int32."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((blockTablesShape.GetDimNum() != BLOCK_TABLE_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    ifaContext_->opName, "blockTable", std::to_string(blockTablesShape.GetDimNum()).c_str(),
                    ("the dim num of blockTable should be " + std::to_string(BLOCK_TABLE_DIM_NUM) + ".").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((taskNumI64 > UINT32_MAX),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "taskNum", std::to_string(taskNumI64).c_str(),
                    ("taskNum exceeds UINT32_MAX(" + std::to_string(UINT32_MAX) + ").").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (blockSize_ <= 0 || blockSize_ % 16U != 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                              "blockSize should be positive and a multiple of 16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (static_cast<uint64_t>(headDim_) * blockSize_ > 128U * 128U),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ifaContext_->opName, "headDim, blockSize",
                                               (std::to_string(headDim_) + ", " + std::to_string(blockSize_)).c_str(),
                                               "headDim multiplied by blockSize should not be greater than 128 * 128."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyShapeInput() const
{
    const gert::StorageShape* keyShape = ifaContext_->kCache[0];
    OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    ifaContext_->opName, "key", std::to_string(keyShape->GetStorageShape().GetDimNum()).c_str(),
                    ("the dim num of key should be " + std::to_string(KV_CACHE_DIM_NUM) + ".").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((keyShape->GetStorageShape().GetDim(3) != 16),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "key",
                                                       std::to_string(keyShape->GetStorageShape().GetDim(3)).c_str(),
                                                       "K_cache Shape should be in nz format."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckValueShapeInput() const
{
    const gert::StorageShape* valueShape = ifaContext_->vCache[0];
    OP_CHECK_IF((valueShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    ifaContext_->opName, "value", std::to_string(valueShape->GetStorageShape().GetDimNum()).c_str(),
                    ("the dim num of value should be " + std::to_string(KV_CACHE_DIM_NUM) + ".").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((valueShape->GetStorageShape().GetDim(3) != 16),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(ifaContext_->opName, "value",
                                                       std::to_string(valueShape->GetStorageShape().GetDim(3)).c_str(),
                                                       "V_cache Shape should be in nz format."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessCheckATBInput()
{
    OP_CHECK_IF(
        (numHeads_ <= 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "numHeads", std::to_string(numHeads_).c_str(),
                                              "numHeads should be greater than 0."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((numKvHeads_ <= 0 || numKvHeads_ > numHeads_ || numHeads_ % numKvHeads_ != 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    ifaContext_->opName, "numKvHeads, numHeads",
                    (std::to_string(numKvHeads_) + ", " + std::to_string(numHeads_)).c_str(),
                    "numKvHeads should be in (0, numHeads] and numHeads should be divisible by numKvHeads."),
                return ge::GRAPH_FAILED);

    static const size_t Q_CACHE_DIM_NUM = 4;
    const gert::StorageShape* queryShape = ifaContext_->query.shape;

    OP_CHECK_IF(
        (inputQType_ != ge::DT_FLOAT16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "query", DataTypeToSerialString(inputQType_).c_str(),
                                              "query dtype should be float16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((inputKvType_ != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ifaContext_->opName, "key, value",
                                                       DataTypeToSerialString(inputKvType_).c_str(),
                                                       "the dtype of key and value should be float16."),
                return ge::GRAPH_FAILED);
    if (pageAttentionFlag_) {
        if (ProcessCheckATBInputWithPage() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF((ifaContext_->actualSeqLengths.tensor == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "actualSeqLengths"), return ge::GRAPH_FAILED);

    OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != Q_CACHE_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    ifaContext_->opName, "query", std::to_string(queryShape->GetStorageShape().GetDimNum()).c_str(),
                    ("the dim num of query should be " + std::to_string(Q_CACHE_DIM_NUM) + ".").c_str()),
                return ge::GRAPH_FAILED);

    if (CheckKeyShapeInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckValueShapeInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF((headDim_ > 256 || headDim_ <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDim", std::to_string(headDim_).c_str(),
                                                      "headDim should be in (0, 256]."),
                return ge::GRAPH_FAILED);

    int32_t numTokens = queryShape->GetStorageShape().GetDim(2);

    OP_CHECK_IF(
        (numTokens <= 0 || batchSize_ <= 0 || numHeads_ <= 0),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            ifaContext_->opName, "numTokens, batchSize, numHeads",
            (std::to_string(numTokens) + ", " + std::to_string(batchSize_) + ", " + std::to_string(numHeads_)).c_str(),
            "numTokens, batchSize and numHeads should be greater than 0."),
        return ge::GRAPH_FAILED);

    if (ifaContext_->pseShift.tensor != nullptr) {
        attenMaskFlag_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputQKVTypeMatch() const
{
    OP_CHECK_IF((!ropeFlag_) && (nNumOfQInOneGroup_ > 64),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "numHeads", std::to_string(nNumOfQInOneGroup_).c_str(),
                    "the query's heads num divided by the key/value's heads num cannot be greater than 64."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ((!ropeFlag_) && inputQType_ == ge::DT_INT8 && inputKvType_ == ge::DT_INT8),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "query, key, value",
                                                 ("When QueryRope/KeyRope is null and Qs(" + std::to_string(qSeqSize_) +
                                                  ") is in [1, 32], qkv datatype all int8 is not supported.")
                                                     .c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(((inputQType_ == ge::DT_FLOAT16) &&
                 (inputKvType_ != ge::DT_FLOAT16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    ifaContext_->opName, "key, value", DataTypeToSerialString(inputKvType_).c_str(),
                    "when input Q type is fp16, KV type should be fp16, int8 or int4."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(((inputQType_ == ge::DT_BF16) &&
                 (inputKvType_ != ge::DT_BF16 && inputKvType_ != ge::DT_INT8 && inputKvType_ != ge::DT_INT4)),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    ifaContext_->opName, "key, value", DataTypeToSerialString(inputKvType_).c_str(),
                    "when input Q type is bf16, KV type should be bf16, int8 or int4."),
                return ge::GRAPH_FAILED);

    if (pageAttentionFlag_) {
        OP_CHECK_IF(
            (inputKvType_ == ge::DT_FLOAT16 || inputKvType_ == ge::DT_BF16) && (blockSize_ % 16 != 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                                  "blockSize needs to align to 16 when kv dtype is fp16/bf16."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputKvType_ == ge::DT_INT8) && (blockSize_ % 32 != 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                                  "blockSize needs to align to 32 when kv dtype is int8."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckInputFormatAndLimits()
{
    if (CheckInputParameterFormat() != ge::GRAPH_SUCCESS || CheckInputAntiquantFormat() != ge::GRAPH_SUCCESS ||
        CheckInputQKVTypeMatch() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        if (atbRunFlag_) {
            return ProcessCheckATBInput();
        }
        std::string layOutStr = ifaContext_->layOut;
        OP_CHECK_IF(
            (headDim_ > 128U || numHeads_ * qSeqSize_ / numKvHeads_ > 32) && numHeads_ * qSeqSize_ != numKvHeads_,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                ifaContext_->opName, "headDim",
                "On 300I DUO, when gqa > 1 or qlen > 1, gqa * qlen should be <= 32 and headDim should be <= 128."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            pageAttentionFlag_ && layOutStr != "BSH" && layOutStr != "BSND" && numHeads_ * qSeqSize_ != numKvHeads_,
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                ifaContext_->opName, "inputLayout", layOutStr.c_str(),
                "On 300I DUO, when gqa > 1 or qlen > 1, the input layout should be BSH/BSND with page attention "
                "enabled."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((!pageAttentionFlag_) && layOutStr != "BNSD" && numHeads_ * qSeqSize_ != numKvHeads_,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "inputLayout", layOutStr.c_str(),
                        "On 300I DUO, when gqa > 1 or qlen > 1, the input layout should be BNSD without page "
                        "attention enabled."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (batchSize_ > 256),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "batchSize", std::to_string(batchSize_).c_str(),
                                                  "batchSize cannot be greater than 256 when 310P."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((sMax_ > 65536),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "sMax", std::to_string(sMax_).c_str(),
                                                          "sMax cannot be greater than 65536 when 310P."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (headDim_ % 16 != 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDim", std::to_string(headDim_).c_str(),
                                                  "in 310P, headDim needs to align to 16."),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            (antiQuantFlag_ && (headDim_ % 32 != 0)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDim", std::to_string(headDim_).c_str(),
                                                  "in 310P, headDim needs to align to 32 when kv dtype is int8."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (batchSize_ > 65536),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "batchSize", std::to_string(batchSize_).c_str(),
                                                  "batchSize cannot be greater than 65536."),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((!ropeFlag_) && (headDim_ > 512U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDim", std::to_string(headDim_).c_str(),
                                                      "headDim cannot be greater than 512."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (!ropeFlag_) && (numKvHeads_ > 256U),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "numHeads", std::to_string(numKvHeads_).c_str(),
                                              "numHead of key and value cannot be greater than 256."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVHeadNum(const gert::StorageShape* inputShape) const
{
    uint32_t tmpNumHeads = 0;
    std::string layOutStr = ifaContext_->layOut;
    constexpr uint32_t dimIndex2 = 2;
    if (layOutStr == "BSH") {
        auto H = inputShape->GetStorageShape().GetDim(dimIndex2);
        tmpNumHeads = H / headDim_;
    } else if (layOutStr == "TND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BNSD") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(1);
    } else if (layOutStr == "BSND") {
        tmpNumHeads = inputShape->GetStorageShape().GetDim(dimIndex2);
    }
    OP_CHECK_IF(tmpNumHeads != numKvHeads_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "numHeads", std::to_string(tmpNumHeads).c_str(),
                    ("the head num of tensor in list should be " + std::to_string(numKvHeads_) + ".").c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVShape(const size_t& size, const gert::StorageShape* keyTensorInList,
                                        const gert::StorageShape* valueTensorInList) const
{
    /* kv not continuous */
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH") {
        OP_CHECK_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BSH) ||
                        (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BSH),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        ifaContext_->opName, "key, value",
                        (std::to_string(keyTensorInList->GetStorageShape().GetDimNum()) + ", " +
                         std::to_string(valueTensorInList->GetStorageShape().GetDimNum()))
                            .c_str(),
                        "the dim num of key/value tensor in list should be 3."),
                    return ge::GRAPH_FAILED);
    }
    if ((layOutStr == "BNSD") || (layOutStr == "BSND")) {
        OP_CHECK_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND) ||
                        (valueTensorInList->GetStorageShape().GetDimNum() != DIM_BNSD_OR_BSND),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        ifaContext_->opName, "key, value",
                        (std::to_string(keyTensorInList->GetStorageShape().GetDimNum()) + ", " +
                         std::to_string(valueTensorInList->GetStorageShape().GetDimNum()))
                            .c_str(),
                        "the dim num of key/value tensor in list should be 4."),
                    return ge::GRAPH_FAILED);
    }
    if (layOutStr == "TND") {
        OP_CHECK_IF((keyTensorInList->GetStorageShape().GetDimNum() != DIM_TND) ||
                        (valueTensorInList->GetStorageShape().GetDimNum() != DIM_TND),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        ifaContext_->opName, "key, value",
                        (std::to_string(keyTensorInList->GetStorageShape().GetDimNum()) + ", " +
                         std::to_string(valueTensorInList->GetStorageShape().GetDimNum()))
                            .c_str(),
                        "the dim num of key/value tensor in list should be 4."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(keyTensorInList->GetStorageShape().GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "batch", std::to_string(keyTensorInList->GetStorageShape().GetDim(0)).c_str(),
                    ("b of tensor in tensor list should be 1, list index: " + std::to_string(size) + ".").c_str()),
                return ge::GRAPH_FAILED);
    if (CheckKVHeadNum(keyTensorInList) != ge::GRAPH_SUCCESS ||
        CheckKVHeadNum(valueTensorInList) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeBsh(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
                                              const gert::StorageShape* queryShape,
                                              const gert::StorageShape* keyShape) const
{
    const std::string inputLayoutStr = ifaContext_->layOut;
    auto outExpectDim = inputLayoutStr == "BSH" ? DIM_BSH : DIM_BNSD;
    OP_CHECK_IF((dimOfQ != DIM_BSH) || (dimOfK != DIM_BSH) || (dimOfOut != outExpectDim),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "query, key, attenOut",
                    (std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut)).c_str(),
                    ("when input layout is BSH/BSH_NBSD, the dim num of query/key should be 3 and attenOut should be " +
                     std::to_string(outExpectDim) + ".")
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(1) != 1 && !ropeFlag_ && !gqaMtpFlag_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "query",
                                                      std::to_string(queryShape->GetStorageShape().GetDim(1)).c_str(),
                                                      "when input layout is BSH, the 2nd dim of query should be 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetStorageShape().GetDim(2) / numHeads_ != keyShape->GetStorageShape().GetDim(2) / numKvHeads_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ifaContext_->opName, "query, key",
            (std::to_string(queryShape->GetStorageShape().GetDim(2) / numHeads_) + ", " +
             std::to_string(keyShape->GetStorageShape().GetDim(2) / numKvHeads_))
                .c_str(),
            "when input layout is BSH, the 3rd dimOfQ/numHeads should be equal to the 3rd dimOfK/numKvHeads."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeTnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
                                              const gert::StorageShape* queryShape,
                                              const gert::StorageShape* keyShape) const
{
    OP_CHECK_IF((dimOfQ != DIM_TND) || (dimOfK != DIM_TND) || (dimOfOut != DIM_TND),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "query, key, attenOut",
                    (std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut)).c_str(),
                    "when input layout is TND, the dim num of query/key/attenOut should be 3."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        queryShape->GetStorageShape().GetDim(1) / numHeads_ != keyShape->GetStorageShape().GetDim(1) / numKvHeads_,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ifaContext_->opName, "query, key",
            (std::to_string(queryShape->GetStorageShape().GetDim(1) / numHeads_) + ", " +
             std::to_string(keyShape->GetStorageShape().GetDim(1) / numKvHeads_))
                .c_str(),
            "when input layout is TND, the 2nd dimOfQ/numHeads should be equal to the 2nd dimOfK/numKvHeads."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShapeBnsdOrBsnd(size_t dimOfQ, size_t dimOfK, size_t dimOfOut,
                                                     const gert::StorageShape* queryShape,
                                                     const gert::StorageShape* keyShape) const
{
    OP_CHECK_IF((dimOfQ != DIM_BNSD_OR_BSND) || (dimOfK != DIM_BNSD_OR_BSND) || (dimOfOut != DIM_BNSD_OR_BSND),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "query, key, attenOut",
                    (std::to_string(dimOfQ) + ", " + std::to_string(dimOfK) + ", " + std::to_string(dimOfOut)).c_str(),
                    "when input layout is BNSD/BSND, the dim num of query/key/attenOut should be 4."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(3) != keyShape->GetStorageShape().GetDim(3),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "query, key",
                    (std::to_string(queryShape->GetStorageShape().GetDim(3)) + ", " +
                     std::to_string(keyShape->GetStorageShape().GetDim(3)))
                        .c_str(),
                    "when input layout is BNSD/BSND, the 4th dimOfQ should be equal to the 4th dimOfK."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckQKOutShape() const
{
    if (pageAttentionFlag_) { // page_attention don't check this place
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape* queryShape = ifaContext_->query.shape;
    const gert::StorageShape* keyShape = ifaContext_->kCache[0];
    const std::string inputLayoutStr = ifaContext_->layOut;

    auto dimOfQ = queryShape->GetStorageShape().GetDimNum();
    auto dimOfK = keyShape->GetStorageShape().GetDimNum();
    auto dimOfOut = ifaContext_->attenOut.shape->GetStorageShape().GetDimNum();
    if (inputLayoutStr == "BSH" || inputLayoutStr == "BSH_NBSD") {
        return CheckQKOutShapeBsh(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        return CheckQKOutShapeTnd(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    } else {
        return CheckQKOutShapeBnsdOrBsnd(dimOfQ, dimOfK, dimOfOut, queryShape, keyShape);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKeyShapeTensor(const gert::Shape& aShape) const
{
    auto firstKeyShape = ifaContext_->kCache[0];
    std::string layOutStr = ifaContext_->layOut;
    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (((layOutStr == "BNSD") && (idx == 2)) || // BNSD s index is 2
            ((layOutStr == "BSND") && (idx == 1)) || // BSND s index is 1
            ((layOutStr == "BSH") && (idx == 1))) {  // BSH s index is 1
            continue;                                // s can be different
        }
        OP_CHECK_IF(
            firstKeyShape->GetStorageShape().GetDim(idx) != aShape.GetDim(idx),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ifaContext_->opName, "key",
                (std::to_string(firstKeyShape->GetStorageShape().GetDim(idx)) + ", " +
                 std::to_string(aShape.GetDim(idx)))
                    .c_str(),
                ("expect tensor in keyShape except S must be the same, index: " + std::to_string(idx) + ".").c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool IFATiling::CheckIfRollBack() const
{
    if (sMax_ == 0U) {
        return false; // 空tensor由新模板处理
    }

    if (socVersion_ != IfaSocVersion::SOC_ASCEND_310P) {
        // 1 page attention
        if (ifaContext_->blockTable.tensor != nullptr) {
            return false;
        }
    }

    // 4 D>=1024
    if (headDim_ >= 1024U) {
        return true;
    }

    if (CanChangeToNew()) {
        return false;
    }

    return true;
}

bool IFATiling::ShapeEqual(const gert::Shape& aShape, const gert::Shape& bShape) const
{
    if (aShape.GetDimNum() != bShape.GetDimNum()) {
        return false;
    }

    for (size_t idx = 0; idx < aShape.GetDimNum(); idx++) {
        if (aShape.GetDim(idx) != bShape.GetDim(idx)) {
            return false;
        }
    }

    return true;
}

bool IFATiling::CanChangeToNew() const
{
    if (inOutMode_ == TilingInOutMode::BF16_BF16) {
        return true;
    }
    if (inOutMode_ == TilingInOutMode::BF16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::FP16_FP16 || inOutMode_ == TilingInOutMode::FP16_INT8) {
        return true;
    }

    if (inOutMode_ == TilingInOutMode::INT8_BF16 || inOutMode_ == TilingInOutMode::INT8_FP16) {
        return true;
    }
    return false;
}

static bool IsQuant2ShapeValid(const gert::Shape& inputParaShape, uint32_t headnum, uint32_t headsize)
{
    gert::Shape expectParamShapeBNSD = gert::Shape({1, headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_2 = gert::Shape({headnum, 1, headsize});
    gert::Shape expectParamShapeBNSD_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBSND = gert::Shape({1, 1, headnum, headsize});
    gert::Shape expectParamShapeBSND_2 = gert::Shape({1, headnum, headsize});
    gert::Shape expectParamShapeBSND_3 = gert::Shape({headnum, headsize});
    gert::Shape expectParamShapeBH = gert::Shape({1, headnum * headsize});
    gert::Shape expectParamShapeBH_2 = gert::Shape({1, 1, headnum * headsize});
    gert::Shape expectParamShapeBH_3 = gert::Shape({headnum * headsize});

    bool validShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSND) ||
                      (inputParaShape == expectParamShapeBH) || (inputParaShape == expectParamShapeBNSD_2) ||
                      (inputParaShape == expectParamShapeBSND_2) || (inputParaShape == expectParamShapeBH_2) ||
                      (inputParaShape == expectParamShapeBNSD_3) || (inputParaShape == expectParamShapeBSND_3) ||
                      (inputParaShape == expectParamShapeBH_3);

    return validShape;
}

ge::graphStatus IFATiling::CheckQuant2Shape(const gert::Shape& inputParaShape) const
{
    auto headsize = headDim_; // D
    auto headnum = numHeads_; // Q's N
    bool validShape = IsQuant2ShapeValid(inputParaShape, headnum, headsize);
    if (!validShape && inputParaShape.GetDimNum() == DIM_BNSD) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ifaContext_->opName, "postquant",
            (std::to_string(inputParaShape.GetDim(BNSD_B_IDX)) + ", " +
             std::to_string(inputParaShape.GetDim(BNSD_N_IDX)) + ", " +
             std::to_string(inputParaShape.GetDim(BNSD_S_IDX)) + ", " +
             std::to_string(inputParaShape.GetDim(BNSD_D_IDX)))
                .c_str(),
            ("expect [1, " + std::to_string(headnum) + ", 1, " + std::to_string(headsize) + "] or [1, 1, " +
             std::to_string(headnum) + ", " + std::to_string(headsize) + "].")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 3) { // dim is 3
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ifaContext_->opName, "postquant",
            (std::to_string(inputParaShape.GetDim(BNSD_B_IDX)) + ", " +
             std::to_string(inputParaShape.GetDim(BNSD_N_IDX)) + ", " +
             std::to_string(inputParaShape.GetDim(BNSD_S_IDX)))
                .c_str(),
            ("expect [" + std::to_string(headnum) + ", 1, " + std::to_string(headsize) + "], [1, " +
             std::to_string(headnum) + ", " + std::to_string(headsize) + "] or [1, 1, " +
             std::to_string(headnum * headsize) + "].")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == DIM_BH) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            ifaContext_->opName, "postquant",
            (std::to_string(inputParaShape.GetDim(BH_B_IDX)) + ", " + std::to_string(inputParaShape.GetDim(BH_H_IDX)))
                .c_str(),
            ("expect [1, " + std::to_string(headnum * headsize) + "] or [" + std::to_string(headnum) + ", " +
             std::to_string(headsize) + "].")
                .c_str());
        return ge::GRAPH_FAILED;
    }

    if (!validShape && inputParaShape.GetDimNum() == 1) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ifaContext_->opName, "postquant",
                                               std::to_string(inputParaShape.GetDim(BH_B_IDX)).c_str(),
                                               ("expect [" + std::to_string(headnum * headsize) + "].").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerHead(const gert::Shape& inputParaShape) const
{
    if (antiquantMode_ == PER_TOKEN_MODE) {            // per-token head
        OP_CHECK_IF((inputParaShape.GetDimNum() != 3), // 3: Dim of BGS is 3
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDimNum()).c_str(),
                        "the dim num of antiquant should be 3 when per_token_head mode."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(0) != batchSize_),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDim(0)).c_str(),
                ("the 1st dim of antiquant should be " + std::to_string(batchSize_) + " when per_token_head mode.")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (inputParaShape.GetDim(1) != numKvHeads_),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDim(1)).c_str(),
                ("the 2nd dim of antiquant should be " + std::to_string(numKvHeads_) + " when per_token_head mode.")
                    .c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputParaShape.GetDim(2) < seqSize_),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDim(2)).c_str(),
                        ("the 3rd dim of antiquant should be greater than or equal to " + std::to_string(seqSize_) +
                         " when per_token_head mode.")
                            .c_str()), // 2 : BGS S index is 2
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    } else { // per-tensor head
        gert::Shape expectParamShape = gert::Shape({numKvHeads_});
        OP_CHECK_IF((inputParaShape != expectParamShape),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDim(0)).c_str(),
                        ("expect [" + std::to_string(numKvHeads_) + "] when per_tensor_head mode.").c_str()),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus IFATiling::CheckKVAntiQuantPerChannel(const gert::Shape& inputParaShape) const
{
    std::string layOutStr = ifaContext_->layOut;
    if (gqaKvNZFlag_ && kvAntiParamSplitFlag_) {
        return CheckGqaAntiQuantPerChannel(inputParaShape);
    } else {
        bool validOffsetShape = false;
        gert::Shape expectParamShapeBNSD = gert::Shape({antiquantNum_, numKvHeads_, 1, headDim_});
        gert::Shape expectParamShapeBSNDType1 = gert::Shape({antiquantNum_, 1, numKvHeads_, headDim_});
        gert::Shape expectParamShapeBSNDType2 = gert::Shape({antiquantNum_, numKvHeads_, headDim_});
        gert::Shape expectParamShapeBH = gert::Shape({antiquantNum_, numKvHeads_ * headDim_});
        validOffsetShape = (inputParaShape == expectParamShapeBNSD) || (inputParaShape == expectParamShapeBSNDType1) ||
                           (inputParaShape == expectParamShapeBSNDType2) || (inputParaShape == expectParamShapeBH);
        OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BNSD),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant",
                        (std::to_string(inputParaShape.GetDim(BNSD_B_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BNSD_N_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BNSD_S_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BNSD_D_IDX)))
                            .c_str(),
                        ("expect [" + std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_) + ", 1, " +
                         std::to_string(headDim_) + "] or [" + std::to_string(antiquantNum_) + ", " +
                         std::to_string(numKvHeads_) + ", " + std::to_string(headDim_) + "] or [" +
                         std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_ * headDim_) +
                         "] when per_channel mode.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_PER_CHANNEL_BSND),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant",
                        (std::to_string(inputParaShape.GetDim(BND_B_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BND_N_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BND_D_IDX)))
                            .c_str(),
                        ("expect [" + std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_) + ", 1, " +
                         std::to_string(headDim_) + "] or [" + std::to_string(antiquantNum_) + ", " +
                         std::to_string(numKvHeads_) + ", " + std::to_string(headDim_) + "] or [" +
                         std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_ * headDim_) +
                         "] when per_channel mode.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((!validOffsetShape && inputParaShape.GetDimNum() == DIM_BH),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant",
                        (std::to_string(inputParaShape.GetDim(BH_B_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BH_H_IDX)))
                            .c_str(),
                        ("expect [" + std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_) + ", 1, " +
                         std::to_string(headDim_) + "] or [" + std::to_string(antiquantNum_) + ", " +
                         std::to_string(numKvHeads_) + ", " + std::to_string(headDim_) + "] or [" +
                         std::to_string(antiquantNum_) + ", " + std::to_string(numKvHeads_ * headDim_) +
                         "] when per_channel mode.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiQuantPerChannel(const gert::Shape& inputParaShape) const
{
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BNSD") {
        OP_CHECK_IF((inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BNSD),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDimNum()).c_str(),
                        ("the shape dim of antiquant parameter should be " + std::to_string(DIM_PER_CHANNEL_KVNZ_BNSD) +
                         " when per_channel mode in GQA KV NZ and layout is BNSD.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBNSD = gert::Shape({numKvHeads_, 1, headDim_});
        OP_CHECK_IF((inputParaShape != expectParamShapeBNSD),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant",
                        (std::to_string(inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_N_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_S_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BNSD_NZ_ANTIQUANT_D_IDX)))
                            .c_str(),
                        ("the shape of antiquant parameter should be [" + std::to_string(numKvHeads_) + ", 1, " +
                         std::to_string(headDim_) + "] when per_channel mode in GQA KV NZ and layout is BNSD.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    } else if (layOutStr == "BSND" || layOutStr == "TND") {
        OP_CHECK_IF((inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BSND),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDimNum()).c_str(),
                        ("the shape dim of antiquant parameter should be " + std::to_string(DIM_PER_CHANNEL_KVNZ_BSND) +
                         " when per_channel mode in GQA KV NZ and layout is BSND or TND.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBSND = gert::Shape({numKvHeads_, headDim_});
        OP_CHECK_IF((inputParaShape != expectParamShapeBSND),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "antiquant",
                        (std::to_string(inputParaShape.GetDim(BSND_NZ_ANTIQUANT_N_IDX)) + ", " +
                         std::to_string(inputParaShape.GetDim(BSND_NZ_ANTIQUANT_D_IDX)))
                            .c_str(),
                        ("the shape of antiquant parameter should be [" + std::to_string(numKvHeads_) + ", " +
                         std::to_string(headDim_) + "] when per_channel mode in GQA KV NZ and layout is BSND.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    } else if (layOutStr == "BSH") {
        OP_CHECK_IF((inputParaShape.GetDimNum() != DIM_PER_CHANNEL_KVNZ_BSH),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDimNum()).c_str(),
                        ("the shape dim of antiquant parameter should be " + std::to_string(DIM_PER_CHANNEL_KVNZ_BSH) +
                         " when per_channel mode in GQA KV NZ and layout is BSH.")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        gert::Shape expectParamShapeBH = gert::Shape({numKvHeads_ * headDim_});
        OP_CHECK_IF(
            (inputParaShape != expectParamShapeBH),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ifaContext_->opName, "antiquant", std::to_string(inputParaShape.GetDim(BSH_NZ_ANTIQUANT_H_IDX)).c_str(),
                ("the shape of antiquant parameter should be [" + std::to_string(numKvHeads_ * headDim_) +
                 "] when per_channel mode in GQA KV NZ and layout is BSH.")
                    .c_str()),
            return ge::GRAPH_FAILED);
    } // 其他无效layout场景已校验过，不需要再校验
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiQuantParam(const gert::Tensor* antiquantScaleTensor,
                                               const gert::Tensor* antiquantOffsetTensor,
                                               const gert::CompileTimeTensorDesc* antiquantScaleDesc,
                                               const gert::CompileTimeTensorDesc* antiquantOffsetDesc)
{
    OP_CHECK_IF(
        (antiquantMode_ != 0) && (antiquantMode_ != 1) && (antiquantMode_ != 2), // unseparated antiquant need this
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "antiquantMode",
                                              std::to_string(antiquantMode_).c_str(),
                                              "antiquantMode is invalid, it should be 0 or 1 or 2."),
        return ge::GRAPH_FAILED);
    if (antiquantScaleTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "antiquantScale");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr &&
        antiquantScaleTensor->GetStorageShape().GetDimNum() != antiquantOffsetTensor->GetStorageShape().GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            ifaContext_->opName, "antiquantScale, antiquantOffset",
            (std::to_string(antiquantScaleTensor->GetStorageShape().GetDimNum()) + ", " +
             std::to_string(antiquantOffsetTensor->GetStorageShape().GetDimNum()))
                .c_str(),
            "KV antiquant is enabled, but antiquant scale and antiquant offset have different dims.");
        return ge::GRAPH_FAILED;
    }
    auto tmpAntiquantScale = antiquantScaleTensor->GetStorageShape();
    if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantScale) == ge::GRAPH_FAILED) {
        // 形状非法已在 CheckKVAntiQuantParaShapeLegal 中上报，此处仅透传失败结果，不再重复上报内部错误码
        OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "illegal shape of antiquant scale.");
        return ge::GRAPH_FAILED;
    }
    if (antiquantOffsetTensor != nullptr) {
        auto tmpAntiquantOffset = antiquantOffsetTensor->GetStorageShape();
        if (CheckKVAntiQuantParaShapeLegal(tmpAntiquantOffset) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    }

    ge::DataType antiquantScaleType = antiquantScaleDesc->GetDataType();
    if (antiquantMode_ == DEQUANT_PER_CHANNEL_MODE) {
        if (antiquantScaleType != inputQType_) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                ifaContext_->opName, "antiquantScale", DataTypeToSerialString(antiquantScaleType).c_str(),
                "the dtype of antiquant scale should be the same as the input query dtype.");
            return ge::GRAPH_FAILED;
        }
    }
    if (antiquantMode_ == DEQUANT_PER_TOKEN_MODE) {
        if (antiquantScaleType != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                ifaContext_->opName, "antiquantScale", DataTypeToSerialString(antiquantScaleType).c_str(),
                "per-token mode is enabled, dtype of antiquant scale should be float32.");
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckAntiquantOffsetType(antiquantOffsetTensor, antiquantOffsetDesc, antiquantScaleType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAntiquantOffsetType(const gert::Tensor* antiquantOffsetTensor,
                                                    const gert::CompileTimeTensorDesc* antiquantOffsetDesc,
                                                    ge::DataType antiquantScaleType) const
{
    if (antiquantOffsetTensor != nullptr && antiquantOffsetDesc != nullptr) {
        ge::DataType antiquantOffsetType = antiquantOffsetDesc->GetDataType();
        if (antiquantScaleType != antiquantOffsetType) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                ifaContext_->opName, "antiquantScale, antiquantOffset",
                (DataTypeToSerialString(antiquantScaleType) + ", " + DataTypeToSerialString(antiquantOffsetType))
                    .c_str(),
                "the dtype of antiquant scale and antiquant offset should be the same.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckSupportKVLeftPadding()
{
    if (inputKvType_ == ge::DT_INT4) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            ifaContext_->opName, "kvLeftPadding",
            "the dtype of input kv is int4 or int32, KvLeftPadding is not supported currently.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ropeFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "kvPadding",
                                                         "kvPadding is not supported in MLA."),
                return ge::GRAPH_FAILED);
    if (!batchContinuousFlag_ || !actualSeqLenFlag_ || pageAttentionFlag_) {
        OP_LOGD(ifaContext_->opName, "KVLeftPadding illegal condition:  \
      pagedAttention scene: %d, not isBatchContinues: %d, actualSeqLen not exist: %d.",
                pageAttentionFlag_, !batchContinuousFlag_, !actualSeqLenFlag_);
        return ge::GRAPH_SUCCESS;
    }
    kvPaddingSizeFlag_ = true;
    OP_LOGD(ifaContext_->opName, "KVLeftPadding starts to be used.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckBasic()
{
    OP_CHECK_IF(ifaContext_->keySharedPrefix.tensor == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "tensor of key_shared_prefix is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "desc of key_shared_prefix is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.tensor == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "tensor of value_shared_prefix is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "desc of value_shared_prefix is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc->GetDataType() != inputKvType_,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    ifaContext_->opName, "keySharedPrefix, key",
                    (DataTypeToSerialString(ifaContext_->keySharedPrefix.desc->GetDataType()) + ", " +
                     DataTypeToSerialString(inputKvType_))
                        .c_str(),
                    "the dtype of key_shared_prefix should be the same as the dtype of key."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc->GetDataType() != inputKvType_,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    ifaContext_->opName, "valueSharedPrefix, value",
                    (DataTypeToSerialString(ifaContext_->valueSharedPrefix.desc->GetDataType()) + ", " +
                     DataTypeToSerialString(inputKvType_))
                        .c_str(),
                    "the dtype of value_shared_prefix should be the same as the dtype of value."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(pageAttentionFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keySharedPrefix, valueSharedPrefix",
                                                         "shared prefix with page attention is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keySharedPrefix, valueSharedPrefix",
                                                         "shared prefix with kv padding is not supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(socVersion_ == IfaSocVersion::SOC_ASCEND_310P,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keySharedPrefix, valueSharedPrefix",
                                                         "shared prefix is not supported on 310p."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::SharedPrefixCheckShapes(const gert::Shape& keyShape, const gert::Shape& valueShape) const
{
    OP_CHECK_IF(!ShapeEqual(keyShape, valueShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ifaContext_->opName, "keySharedPrefix, valueSharedPrefix",
                                                       "keySharedPrefix and valueSharedPrefix have different shapes",
                                                       "the tensor shape of key_shared_prefix should be equal to the "
                                                       "tensor shape of value_shared_prefix."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.GetDimNum() != ifaContext_->query.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "keySharedPrefix, query",
                    (std::to_string(keyShape.GetDimNum()) + ", " +
                     std::to_string(ifaContext_->query.shape->GetStorageShape().GetDimNum()))
                        .c_str(),
                    "the dim num of key_shared_prefix should be compatible with the dim num of query."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "keySharedPrefix",
                                                      std::to_string(keyShape.GetDim(0)).c_str(),
                                                      "the batch of key_shared_prefix must be 1."),
                return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        OP_CHECK_IF(keyShape.GetDimNum() == 3 && keyShape.GetDim(2) != numKvHeads_ * headDim_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix, key",
                        (std::to_string(keyShape.GetDim(2)) + ", " + std::to_string(numKvHeads_ * headDim_)).c_str(),
                        "the H of key_shared_prefix should be equal to the H of key."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(keyShape.GetDimNum() == 4 && keyShape.GetDim(2) != numKvHeads_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix, key",
                        (std::to_string(keyShape.GetDim(2)) + ", " + std::to_string(numKvHeads_)).c_str(),
                        "the N of key_shared_prefix should be equal to the N of key."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(keyShape.GetDimNum() == 4 && keyShape.GetDim(3) != headDim_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix, key",
                        (std::to_string(keyShape.GetDim(3)) + ", " + std::to_string(headDim_)).c_str(),
                        "the D of key_shared_prefix should be equal to the D of key."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(keyShape.GetDim(1) != numKvHeads_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix, key",
                        (std::to_string(keyShape.GetDim(1)) + ", " + std::to_string(numKvHeads_)).c_str(),
                        "the N of key_shared_prefix should be equal to the N of key."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(keyShape.GetDim(3) != headDim_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "keySharedPrefix, key",
                        (std::to_string(keyShape.GetDim(3)) + ", " + std::to_string(headDim_)).c_str(),
                        "the D of key_shared_prefix should be equal to the D of key."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckUbSpace()
{
    if (!CalcUbBmm() || !CalcUbSoftMax() || !CalcUbAttenMask()) {
        return false;
    }
    return true;
}

bool IFATiling::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_->opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }
    if (balanceModeFlag_) {
        uint32_t tndFDCoreArrLen = tilingDataMla_.tndSplitCoreParams.get_tndFDCoreArrLen();
        return tndFDCoreArrLen > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}

bool IFATiling::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMM() const
{
    if (!CheckConstraints()) {
        return false;
    }

    if (!CheckDataCopyNd2Nz()) {
        return false;
    }

    uint32_t s2Loop = (maxActualseq_ + 2048U - 1U) / 2048U;
    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ * s2Loop <= 128U) { // 128: 性能收益临界值
        return false;
    }

    if (antiQuantFlag_) {
        if (inputKvType_ == ge::DT_INT4) {
            return false;
        }
    }
    std::string layOutStr = ifaContext_->layOut;
    if (pageAttentionFlag_ && layOutStr == "BNSD") {
        bool quantFlag = antiQuantFlag_ && (antiquantMode_ == PER_CHANNEL_MODE) &&
                         (inputQType_ == ge::DT_FLOAT16 || inputQType_ == ge::DT_BF16) && (inputKvType_ == ge::DT_INT8);
        bool noQuantFlag = (inputQType_ == ge::DT_FLOAT16 && inputKvType_ == ge::DT_FLOAT16) ||
                           (inputQType_ == ge::DT_BF16 && inputKvType_ == ge::DT_BF16);
        if (quantFlag || noQuantFlag) {
            return true;
        }
    }

    return false;
}

bool IFATiling::CheckDataCopyNd2Nz() const
{
    if (innerPrecise_ == IFA_HIGH_PRECISION) {
        return false;
    }
    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    return true;
}

bool IFATiling::CheckConstraints() const
{
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 16U) { // 16: group size
        return false;
    }
    if (headDim_ != 128U) { // 128: D dim
        return false;
    }
    return true;
}

bool IFATiling::EnableCubeViewMMDD() const
{
    if (!CheckConstraints()) {
        return false;
    }
    if (innerPrecise_ == IFA_HIGH_PRECISION) {
        return false;
    }
    if (softmaxLseFlag_) {
        return false;
    }

    std::string layOutStr = ifaContext_->layOut;
    if (layOutStr == "BSH" || layOutStr == "BSND") {
        // 拷贝KV时, DataCopy ND2NZ指令不支持行Stride大于等于65536
        if (static_cast<uint64_t>(headDim_) * numKvHeads_ >= 65536U) {
            return false;
        }
    }
    if (layOutStr == "TND" || layOutStr == "TND_NTD") {
        return false;
    }
    if (ifaContext_->antiquantOffset.desc != nullptr || ifaContext_->keyAntiquantOffset.desc != nullptr ||
        ifaContext_->valueAntiquantOffset.desc != nullptr) {
        return false;
    }
    if (gqaKvNZFlag_ && pageAttentionFlag_ && antiQuantFlag_ &&
        (antiquantMode_ == PER_CHANNEL_MODE || antiquantMode_ == PER_TOKEN_MODE) && (inputQType_ == ge::DT_BF16) &&
        (inputKvType_ == ge::DT_INT8)) {
        return true;
    }
    return false;
}

bool IFATiling::DealSameSeqEachBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return isSameActualseq_;
        } else {
            return isSameSeqAllKVTensor_;
        }
    } else {
        return isSameActualseq_;
    }
}

bool IFATiling::HasZeroSeqBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return hasZeroActualseq_;
        } else {
            return hasZeroSeqKVTensor_;
        }
    } else {
        return hasZeroActualseq_;
    }
}

bool IFATiling::IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
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

bool IFATiling::EnableGQA310P()
{
    if (IsFlashDecode(aicNum_, IfaPerfMode::BMM_ALL_BY_VEC)) {
        return false;
    }
    if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
        return true;
    }
    if (pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) &&
        inputLayout_ == IfaLayout::BSH_BSND) {
        return true;
    }
    if (!pageAttentionFlag_ && nNumOfQInOneGroup_ * qSeqSize_ == 1U && (seqSize_ >= 256U) &&
        inputLayout_ == IfaLayout::BNSD) {
        return true;
    }
    return false;
}

bool IFATiling::EnableCubeViewMMFullLoad()
{
    if (!CheckConstraints()) {
        return false;
    }
    if (IsFlashDecode(aicNum_, IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD)) {
        return false;
    }
    if (softmaxLseFlag_) {
        return false;
    }
    if (!CheckDataCopyNd2Nz()) {
        return false;
    }
    if (!DealSameSeqEachBatch()) {
        return false;
    }
    if (HasZeroSeqBatch()) {
        return false;
    }
    if (maxActualseq_ > 2048U || maxActualseq_ == 0U) {
        return false;
    }

    if (static_cast<uint64_t>(batchSize_) * numKvHeads_ <= 128U) { // 128: 性能收益临界值
        return false;
    }

    if (antiQuantFlag_) {
        if (inputKvType_ == ge::DT_INT4) {
            return false;
        }
        return false;
    } else {
        return false;
    }
}

bool IFATiling::EnableAllVec() const
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        return true;
    }
    if (pageAttentionFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    if (nNumOfQInOneGroup_ > 1U) {
        return false;
    }
    if (headDim_ > 512U) { // 全VEC模板仅支持headDim_不大于512
        return false;
    }
    return (inputQType_ == ge::DT_FLOAT16) && (inputKvType_ == ge::DT_FLOAT16) && (outputType_ == ge::DT_FLOAT16);
}

bool IFATiling::EnableC1V1() const
{
    if (splitKVFlag_) {
        return false;
    }
    if (sysPrefixFlag_) {
        return false;
    }
    // 2:核数不超过vector总核数一半，可以按1:1启动cube和vector
    return (perfMode_ == IfaPerfMode::NORMAL) && (batchSize_ * numKvHeads_ * 2U <= aivNum_);
}

ge::graphStatus IFATiling::CheckPseShiftDataType() const
{
    auto pseShiftDataType = ifaContext_->pseShift.desc->GetDataType();
    if (pseShiftDataType != ge::DT_FLOAT16 && pseShiftDataType != DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "pseShift",
                                              DataTypeToSerialString(pseShiftDataType).c_str(),
                                              "the dtype of pse shift is not supported, only support float16 and "
                                              "bfloat16.");
        return ge::GRAPH_FAILED;
    }

    switch (pseShiftDataType) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            OP_CHECK_IF(
                (inputQType_ != ge::DT_INT8) && (inputQType_ != pseShiftDataType),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    ifaContext_->opName, "pseShift, query",
                    (DataTypeToSerialString(pseShiftDataType) + ", " + DataTypeToSerialString(inputQType_)).c_str(),
                    "the dtype of pse should match the dtype of query or when query is int8."),
                return ge::GRAPH_FAILED);
            break;
        default:
            // 非法 dtype 已在入口拦截，此处为不可达防御分支，仅打印日志不再上报内部错误码
            OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Data type of pse is not currently supported.");
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::ProcessPseShift()
{
    // get pse shift data
    auto pseShiftInput = ifaContext_->pseShift.tensor;
    if (pseShiftInput == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(gqaKvNZFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "pseShift",
                                                         "PseShift is not supported for IFA GQA with KV NZ."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->pseShift.desc == nullptr,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Desc of pse shift tensor is null."),
                return ge::GRAPH_FAILED);

    if (CheckPseShiftDataType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // check pse shift shape (B/1, N, 1, Si)
    const gert::Shape pseShiftShape = pseShiftInput->GetStorageShape();
    uint32_t pseShiftDimNum = pseShiftShape.GetDimNum();
    OP_CHECK_IF(pseShiftDimNum != 4,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ifaContext_->opName, "pseShift",
                                                         std::to_string(pseShiftDimNum).c_str(),
                                                         "the input shape of pse shift must have 4 dims."),
                return GRAPH_FAILED);
    pseShiftBatch_ = pseShiftShape.GetDim(PSE_SHIFT_B);
    uint32_t pseShiftN = pseShiftShape.GetDim(PSE_SHIFT_N);
    uint32_t pseShiftS0 = pseShiftShape.GetDim(PSE_SHIFT_S0);
    pseShiftS1_ = pseShiftShape.GetDim(PSE_SHIFT_S1);

    OP_CHECK_IF(
        (pseShiftBatch_ != 1 && pseShiftBatch_ != batchSize_) || (pseShiftN != numHeads_) || (pseShiftS0 != 1),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ifaContext_->opName, "pseShift",
                                               (std::to_string(pseShiftBatch_) + ", " + std::to_string(pseShiftN) +
                                                ", " + std::to_string(pseShiftS0) + ", " + std::to_string(pseShiftS1_))
                                                   .c_str(),
                                               "the shape of pse shift should match (B, N, 1, S) or (1, N, 1, S)."),
        return ge::GRAPH_FAILED);

    if (pseShiftS1_ < seqSize_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            ifaContext_->opName, "pseShift",
            (std::to_string(pseShiftBatch_) + ", " + std::to_string(pseShiftN) + ", " + std::to_string(pseShiftS0) +
             ", " + std::to_string(pseShiftS1_))
                .c_str(),
            ("the 3rd dim S[" + std::to_string(pseShiftS1_) + "] of pse shift shouldn't be less than sMax[" +
             std::to_string(seqSize_) + "]. When Page Attention is enabled, sMax is maxBlockNumPerBatch * blockSize.")
                .c_str());
        return GRAPH_FAILED;
    }

    pseShiftFlag_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTreeSparseMask()
{
    // 仅支持MLA全量化，且要求s2 >= s1
    OP_CHECK_IF(socVersion_ != IfaSocVersion::SOC_ASCEND_910B,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "sparseMode",
                                                         "tree sparse9 is only supported in ASCEND_910B."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!(ropeFlag_ && quantFlag_),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "sparseMode",
                                                         "tree sparse9 is only supported in MLA full quant situation."),
                return ge::GRAPH_FAILED);

    // tiling下沉场景，获取不到actualseqlen,不进行校验
    if (isWorkspace_) {
        return ge::GRAPH_SUCCESS;
    }

    // 补充s2 >= s1的拦截，在MLA全量化场景，必须开启PA(NZ)，所以actualSeqKV不进行累加
    if (inputLayout_ == IfaLayout::TND) {
        const int64_t* actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        const int64_t* actualSeqKVTnd = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        int64_t qActSize = 0;
        int64_t kvActSize = 0;

        // s2=0 的 batch（入图padding或空tensor场景）不校验 s1<=s2
        for (int32_t b = 0; b < actualLenQDims_; b++) {
            kvActSize = actualSeqKVTnd[b];
            if (kvActSize == 0) {
                continue;
            }
            qActSize = (b == 0) ? actualSeqQTnd[0] : (actualSeqQTnd[b] - actualSeqQTnd[b - 1]);
            OP_CHECK_IF(qActSize > kvActSize,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            ifaContext_->opName, "query, key",
                            (std::to_string(qActSize) + ", " + std::to_string(kvActSize)).c_str(),
                            "in MLA full quant situation, when sparse is " + std::to_string(sparseMode_) +
                                ", qSize should be less than or equal to kvSize."),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(qSeqSize_ > seqSize_,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        ifaContext_->opName, "query, key",
                        (std::to_string(qSeqSize_) + ", " + std::to_string(seqSize_)).c_str(),
                        "in MLA full quant situation, when sparse is " + std::to_string(sparseMode_) +
                            ", qSize should be less than or equal to kvSize."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTreeSparseMaskShape()
{
    // sparse9时需要传入Mask，在TND场景，传入∑s1²，一维矩阵
    // 非TND场景传入[B, S1, S1]
    auto maskShape = ifaContext_->attenMask.tensor;
    if (inputLayout_ == IfaLayout::TND) {
        // sparse9时需要传入Mask，在TND场景，传入∑s1²，一维矩阵
        OP_CHECK_IF(maskShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "attenMask"),
                    return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 1U,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "attenMask", std::to_string(shape.GetDimNum()).c_str(),
                        "in TND/TND_NTD sparse 9, the atten_mask tensor shape must be 1 "
                        "dimension."),
                    return ge::GRAPH_FAILED);

        if (!isWorkspace_) {
            OP_CHECK_IF(shape.GetDim(0) != qSeqSquareSum_,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            ifaContext_->opName, "attenMask",
                            (std::to_string(shape.GetDim(0)) + ", " + std::to_string(qSeqSquareSum_)).c_str(),
                            "in TND/TND_NTD sparse 9, the atten_mask tensor shape must be ∑s1²."),
                        return ge::GRAPH_FAILED);
        }
        attenMaskFlag_ = true;
    } else {
        OP_CHECK_IF(maskShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "attenMask"),
                    return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 3U || shape.GetDim(0) != batchSize_ || shape.GetDim(1) != qSeqSize_ ||
                        shape.GetDim(2) != qSeqSize_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "attenMask",
                        (std::to_string(shape.GetDimNum()) + ", " + std::to_string(shape.GetDim(0)) + ", " +
                         std::to_string(shape.GetDim(1)) + ", " + std::to_string(shape.GetDim(2)))
                            .c_str(),
                        "in BSH sparse 9, the atten_mask tensor shape must be [B, S1, S1]."),
                    return ge::GRAPH_FAILED);
        attenMaskFlag_ = true;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckTndMaskShapeWithSparseMode()
{
    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    if (sparseMode_ == 0U) {                        // sparse 0 模式只支持无mask
        OP_CHECK_IF(maskShape != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "attenMask",
                                                             "in TND/TND_NTD sparse 0, only noMask is supported."),
                    return ge::GRAPH_FAILED);
        attenMaskFlag_ = false;
    } else if (sparseMode_ == 3U) { // sparse 3 模式需要attenMask
        OP_CHECK_IF(maskShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "attenMask"),
                    return ge::GRAPH_FAILED);

        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != 2U || shape.GetDim(0) != 2048 || shape.GetDim(1) != 2048, // 2， 只支持2维shape
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "attenMask",
                        (std::to_string(shape.GetDimNum()) + ", " + std::to_string(shape.GetDim(0)) + ", " +
                         std::to_string(shape.GetDim(1)))
                            .c_str(),
                        "in TND/TND_NTD sparse 3, the atten_mask tensor shape must be [2048, 2048]."),
                    return ge::GRAPH_FAILED);
        attenMaskFlag_ = true;
    } else if (sparseMode_ == 9U) {
        if (ge::GRAPH_SUCCESS == CheckTreeSparseMask() && ge::GRAPH_SUCCESS == CheckTreeSparseMaskShape()) {
            attenMaskFlag_ = true;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "sparseMode", std::to_string(sparseMode_).c_str(),
                                              "TND/TND_NTD only support sparse = 0/3/9.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMaskShapeWithQSeq() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U &&
                     (static_cast<int32_t>(sparseMode_) != 3 && static_cast<int32_t>(sparseMode_) != 9)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS > 1, sparseMode only supports 3/9 in MLA when full quant situation."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && static_cast<int32_t>(sparseMode_) != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS = 1, sparseMode only supports 0 in MLA when full quant situation."),
                    return ge::GRAPH_FAILED);
    } else if (antiQuantFlag_) {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U && static_cast<int32_t>(sparseMode_) != 3),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS > 1, sparseMode only supports 3 in MLA when antiquant situation."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U && static_cast<int32_t>(sparseMode_) != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS = 1, sparseMode only supports 0 in MLA when antiquant situation."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ > 1U &&
                     (static_cast<int32_t>(sparseMode_) != 3 &&
                      static_cast<int32_t>(sparseMode_) != 4)), // 3 : rightDown, 4 : band
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS > 1, sparseMode supports 3 or 4 in MLA when no quant situation."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((ropeFlag_ && qSeqSize_ == 1U &&
                     (static_cast<int32_t>(sparseMode_) != 0 && static_cast<int32_t>(sparseMode_) != 4)),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "when queryS = 1, sparseMode supports 0 or 4 in MLA when no quant situation."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((ropeFlag_ && (qSeqSize_ > 1U || static_cast<int32_t>(sparseMode_) == 4) &&
                 ifaContext_->attenMask.tensor == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "attenMask"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ropeFlag_ && qSeqSize_ == 1U && ifaContext_->attenMask.tensor != nullptr &&
         static_cast<int32_t>(sparseMode_) != 4),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "attenMask",
                                                 "when queryS = 1 and sparseMode is not 4, atten_mask tensor must be "
                                                 "null in MLA."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckAttenMaskShape()
{
    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    if (maskShape == nullptr) {
        attenMaskFlag_ = false;
        return ge::GRAPH_SUCCESS;
    }

    if (maskShape->GetStorageShape().GetShapeSize() == 0) {
        attenMaskFlag_ = false;
        OP_LOGW(ifaContext_->opName, "atten_mask tensor exist, but atten_mask shape size is 0.");
        return ge::GRAPH_SUCCESS;
    }

    if (sparseMode_ == 9U && CheckTreeSparseMaskShape() == ge::GRAPH_SUCCESS) {
        attenMaskFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus IFATiling::ProcessAttenMask()
{
    // 与pfa保持一致，先判断sparsemode
    sparseMode_ = ifaContext_->sparseMode != nullptr ? *ifaContext_->sparseMode : 0;

    if (inputLayout_ == IfaLayout::TND) {
        return CheckTndMaskShapeWithSparseMode();
    }

    if (CheckMaskShapeWithQSeq() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (sparseMode_ == 9U && CheckTreeSparseMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckAttenMaskShape() == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }

    ge::DataType attenMaskType = ifaContext_->attenMask.desc->GetDataType();
    if (attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_INT8 && attenMaskType != ge::DT_UINT8) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "attenMask",
                                              DataTypeToSerialString(attenMaskType).c_str(),
                                              "the dtype of atten_mask is not supported, only support bool, int8 and "
                                              "uint8.");
        return ge::GRAPH_FAILED;
    }

    if ((sparseMode_ == 3 || sparseMode_ == 4 || gqaMtpFlag_) &&
        (socVersion_ != IfaSocVersion::SOC_ASCEND_310P)) { // 3 : rightDown, 4 : band
        auto shape = ifaContext_->attenMask.tensor->GetStorageShape();
        OP_CHECK_IF((sparseMode_ != 0 && sparseMode_ != 1) && (shape.GetDimNum() != 2U || shape.GetDim(0) != 2048 ||
                                                               shape.GetDim(1) != 2048), // 2， 只支持2维shape
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "attenMask",
                        (std::to_string(shape.GetDimNum()) + ", " + std::to_string(shape.GetDim(0)) + ", " +
                         std::to_string(shape.GetDim(1)))
                            .c_str(),
                        "when sparse = " + std::to_string(static_cast<int32_t>(sparseMode_)) +
                            ", the atten_mask tensor shape must be [2048, 2048] in MLA."),
                    return ge::GRAPH_FAILED);

        attenMaskFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }

    auto maskShape = ifaContext_->attenMask.tensor; // input shape = 4
    uint32_t batchSizeOfMask = static_cast<uint32_t>(maskShape->GetStorageShape().GetDim(0));
    if (batchSizeOfMask != batchSize_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            ifaContext_->opName, "attenMask, query",
            (std::to_string(batchSizeOfMask) + ", " + std::to_string(batchSize_)).c_str(),
            "the batchSize of atten_mask must be equal to the batchSize of query.");
        return ge::GRAPH_FAILED;
    }

    auto dimNumOfMask = maskShape->GetStorageShape().GetDimNum();
    attenMaskSize_ = static_cast<uint32_t>(maskShape->GetStorageShape().GetDim(dimNumOfMask - 1));
    uint32_t minAttenMaskSize = pageAttentionFlag_ ? sMax_ : maxActualseq_;
    if (attenMaskSize_ < minAttenMaskSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "attenMask", std::to_string(attenMaskSize_).c_str(),
                                              "the s Size of atten_mask must be greater than or equal to sMax[" +
                                                  std::to_string(minAttenMaskSize) + "].");
        return ge::GRAPH_FAILED;
    }

    attenMaskFlag_ = true;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
