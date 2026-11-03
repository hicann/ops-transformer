/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_tiling_check_mla_gqa.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CHECK_MLA_GQA_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CHECK_MLA_GQA_H_

#include <numeric>
#include <graph/utils/type_utils.h>
#include "incre_flash_attention_tiling_impl.h"
#include "incre_flash_attention_tiling_base.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

ge::graphStatus IFATiling::CheckMlaQueryRopeDesc() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() != ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        ifaContext_->opName, "queryRope",
                        DataTypeToSerialString(ifaContext_->queryRope.desc->GetDataType()).c_str(),
                        "when the dtype of query is int8, queryRope must be bfloat16 in MLA."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() != ifaContext_->query.desc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        ifaContext_->opName, "queryRope, query",
                        (DataTypeToSerialString(ifaContext_->queryRope.desc->GetDataType()) + ", " +
                         DataTypeToSerialString(ifaContext_->query.desc->GetDataType()))
                            .c_str(),
                        "queryRope and query must have the same dType."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBsndLayout(const gert::Shape& qRopeShape, const gert::Shape& qShape)
{
    OP_CHECK_IF(qRopeShape.GetDim(0) != qShape.GetDim(0),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDim(0)) + ", " + std::to_string(qShape.GetDim(0))).c_str(),
                    "queryRope and query must have the same 'B' (batch) dimension."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qRopeShape.GetDim(1) != qShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDim(1)) + ", " + std::to_string(qShape.GetDim(1))).c_str(),
                    "queryRope and query must have the same 'S' (seq length) dimension."),
                return ge::GRAPH_FAILED);

    if (qRopeShape.GetDimNum() != 4U) {
        headDimRope_ = qRopeShape.GetDim(2) / numHeads_; // 2: H
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(qRopeShape.GetDim(2) != qShape.GetDim(2), // 2: N
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDim(2)) + ", " + std::to_string(qShape.GetDim(2))).c_str(),
                    "queryRope and query must have the same 'N' (head num) dimension."),
                return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBnsdLayout(const gert::Shape& qRopeShape, const gert::Shape& qShape)
{
    OP_CHECK_IF((qRopeShape.GetDim(0) != qShape.GetDim(0)) || (qRopeShape.GetDim(1) != qShape.GetDim(1)) ||
                    (qRopeShape.GetDim(2) != qShape.GetDim(2)), // 2: S
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDim(0)) + ", " + std::to_string(qRopeShape.GetDim(1)) + ", " +
                     std::to_string(qRopeShape.GetDim(2)) + ", " + std::to_string(qRopeShape.GetDim(3)) + ", " +
                     std::to_string(qShape.GetDim(0)) + ", " + std::to_string(qShape.GetDim(1)) + ", " +
                     std::to_string(qShape.GetDim(2)) + ", " + std::to_string(qShape.GetDim(3)))
                        .c_str(),
                    "queryRope and query must have the same 'B', 'N' and 'S' dimensions."),
                return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeNzLayout(const gert::Shape& qRopeShape, const gert::Shape& qShape)
{
    OP_CHECK_IF((qRopeShape.GetDim(0) != qShape.GetDim(0)) || (qRopeShape.GetDim(1) != qShape.GetDim(1)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDim(0)) + ", " + std::to_string(qRopeShape.GetDim(1)) + ", " +
                     std::to_string(qRopeShape.GetDim(2)) + ", " + std::to_string(qShape.GetDim(0)) + ", " +
                     std::to_string(qShape.GetDim(1)) + ", " + std::to_string(qShape.GetDim(2)))
                        .c_str(),
                    "queryRope and query must have the same 'T' and 'N' dimensions."),
                return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(2); // 2:D
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRope()
{
    if (CheckMlaQueryRopeDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto qRopeShape = ifaContext_->queryRope.tensor->GetStorageShape();

    auto qShape = ifaContext_->query.shape->GetStorageShape();
    OP_CHECK_IF(qRopeShape.GetDimNum() != qShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "queryRope, query",
                    (std::to_string(qRopeShape.GetDimNum()) + ", " + std::to_string(qShape.GetDimNum())).c_str(),
                    "queryRope and query must have the same number of dimensions."),
                return ge::GRAPH_FAILED);

    if (inputLayout_ == IfaLayout::BSH_BSND) {
        return CheckMlaQueryRopeBsndLayout(qRopeShape, qShape);
    } else if (inputLayout_ == IfaLayout::BNSD) {
        return CheckMlaQueryRopeBnsdLayout(qRopeShape, qShape);
    } else if (inputLayout_ == IfaLayout::TND) {
        return CheckMlaQueryRopeNzLayout(qRopeShape, qShape);
    }

    return ge::GRAPH_SUCCESS; // never here
}

ge::graphStatus IFATiling::CheckMlaAttrs() const
{
    OP_CHECK_IF(numKvHeads_ != 1U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "numKeyValueHeads",
                                                      std::to_string(numKvHeads_).c_str(),
                                                      "the key/value's heads num only support 1 in MLA."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(nNumOfQInOneGroup_ != 1U && nNumOfQInOneGroup_ != 2U && nNumOfQInOneGroup_ != 4U &&
                    nNumOfQInOneGroup_ != 8U && nNumOfQInOneGroup_ != 16U && nNumOfQInOneGroup_ != 32U &&
                    nNumOfQInOneGroup_ != 64U && nNumOfQInOneGroup_ != 128U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "numHeads", std::to_string(nNumOfQInOneGroup_).c_str(),
                    "the query's heads num divided by the key/value's heads num in MLA only support {1, 2, 4, 8, 16, "
                    "32, 64, 128}."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDim_ != 512U,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "queryD", std::to_string(headDim_).c_str(),
                                                      "queryD only support 512 in MLA."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        headDimRope_ != 64U,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDimRope", std::to_string(headDimRope_).c_str(),
                                              "headDimRope only support 64 in MLA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputKvLayout_ == IfaLayout::NZ && inputLayout_ != IfaLayout::BSH_BSND && inputLayout_ != IfaLayout::TND),
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            ifaContext_->opName, "inputLayout", std::string(ifaContext_->layOut).c_str(),
            "When kv is NZ, layout only support {BSH, BSND, TND, BSH_NBSD, BSND_NBSD, TND_NTD}."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeDesc() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ifaContext_->keyRope.desc->GetDataType() != ifaContext_->queryRope.desc->GetDataType()),
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        ifaContext_->opName, "keyRope, queryRope",
                        (DataTypeToSerialString(ifaContext_->keyRope.desc->GetDataType()) + ", " +
                         DataTypeToSerialString(ifaContext_->queryRope.desc->GetDataType()))
                            .c_str(),
                        "when the dtype of query is int8, keyRope and queryRope must have the same dType."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (ifaContext_->keyRope.desc->GetDataType() != ifaContext_->key.desc->GetDataType()),
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ifaContext_->opName, "keyRope, key",
                                                   (DataTypeToSerialString(ifaContext_->keyRope.desc->GetDataType()) +
                                                    ", " + DataTypeToSerialString(ifaContext_->key.desc->GetDataType()))
                                                       .c_str(),
                                                   "keyRope and key must have the same dType."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeShapeMatch(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyRopeShape.GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keyRope",
                                                         "empty keyRope Tensor is not supported in MLA."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "key",
                                                         "empty key Tensor is not supported in MLA."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(0) != keyShape.GetDim(0),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "keyRope, key",
                    (std::to_string(keyRopeShape.GetDim(0)) + ", " + std::to_string(keyShape.GetDim(0))).c_str(),
                    "KeyRope and Key must have the same BlockNum."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyRopeShape.GetDimNum() != keyShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    ifaContext_->opName, "keyRope, key",
                    (std::to_string(keyRopeShape.GetDimNum()) + ", " + std::to_string(keyShape.GetDimNum())).c_str(),
                    "KeyRope and Key must have the same number of dimensions."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBsndLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyShape.GetDim(1) != blockSize_ || keyShape.GetDim(2) != (numKvHeads_ * headDim_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "key",
                    (std::to_string(keyShape.GetDim(0)) + ", " + std::to_string(keyShape.GetDim(1)) + ", " +
                     std::to_string(keyShape.GetDim(2)))
                        .c_str(),
                    ("The dim of KeyShape is 3, KeyShape must be [" + std::to_string(keyShape.GetDim(0)) + ", " +
                     std::to_string(blockSize_) + ", " + std::to_string(numKvHeads_ * headDim_) + "] in MLA.")
                        .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(1) != blockSize_ || keyRopeShape.GetDim(2) != (numKvHeads_ * headDimRope_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "keyRope",
                    (std::to_string(keyRopeShape.GetDim(0)) + ", " + std::to_string(keyRopeShape.GetDim(1)) + ", " +
                     std::to_string(keyRopeShape.GetDim(2)))
                        .c_str(),
                    ("The dim of KeyShape is 3, KeyRopeShape must be [" + std::to_string(keyRopeShape.GetDim(0)) +
                     ", " + std::to_string(blockSize_) + ", " + std::to_string(numKvHeads_ * headDimRope_) + "].")
                        .c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBnsdLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyShape.GetDim(1) != numKvHeads_ || keyShape.GetDim(2) != blockSize_ || keyShape.GetDim(3) != headDim_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "key",
                    (std::to_string(keyShape.GetDim(0)) + ", " + std::to_string(keyShape.GetDim(1)) + ", " +
                     std::to_string(keyShape.GetDim(2)) + ", " + std::to_string(keyShape.GetDim(3)))
                        .c_str(),
                    ("The dim of KeyShape is 4, KeyShape must be [" + std::to_string(keyShape.GetDim(0)) + ", " +
                     std::to_string(numKvHeads_) + ", " + std::to_string(blockSize_) + ", " + std::to_string(headDim_) +
                     "] in MLA.")
                        .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(1) != numKvHeads_ || keyRopeShape.GetDim(2) != blockSize_ ||
                    keyRopeShape.GetDim(3) != headDimRope_,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "keyRope",
                    (std::to_string(keyRopeShape.GetDim(0)) + ", " + std::to_string(keyRopeShape.GetDim(1)) + ", " +
                     std::to_string(keyRopeShape.GetDim(2)) + ", " + std::to_string(keyRopeShape.GetDim(3)))
                        .c_str(),
                    ("The dim of KeyShape is 4, KeyRopeShape must be [" + std::to_string(keyRopeShape.GetDim(0)) +
                     ", " + std::to_string(numKvHeads_) + ", " + std::to_string(blockSize_) + ", " +
                     std::to_string(headDimRope_) + "].")
                        .c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeNzLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    size_t kvTypeSize = static_cast<size_t>(GetTypeSize(ifaContext_->key.desc->GetDataType()));
    size_t kvRopeTypeSize = static_cast<size_t>(GetTypeSize(ifaContext_->keyRope.desc->GetDataType()));
    OP_CHECK_IF(static_cast<size_t>(keyShape.GetDim(1)) != numKvHeads_ ||
                    static_cast<size_t>(keyShape.GetDim(2)) != headDim_ / (32 / kvTypeSize) ||
                    static_cast<size_t>(keyShape.GetDim(3)) != blockSize_ ||
                    static_cast<size_t>(keyShape.GetDim(4)) != 32 / kvTypeSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "key",
                    (std::to_string(keyShape.GetDim(0)) + ", " + std::to_string(keyShape.GetDim(1)) + ", " +
                     std::to_string(keyShape.GetDim(2)) + ", " + std::to_string(keyShape.GetDim(3)) + ", " +
                     std::to_string(keyShape.GetDim(4)))
                        .c_str(),
                    ("KvLayout is NZ and KvDtype is " + DataTypeToSerialString(ifaContext_->key.desc->GetDataType()) +
                     ", keyShape must be [" + std::to_string(keyShape.GetDim(0)) + ", " + std::to_string(numKvHeads_) +
                     ", " + std::to_string(headDim_ / (32 / kvTypeSize)) + ", " + std::to_string(blockSize_) + ", " +
                     std::to_string(32 / kvTypeSize) + "] in MLA.")
                        .c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(static_cast<size_t>(keyRopeShape.GetDim(1)) != numKvHeads_ ||
                    static_cast<size_t>(keyRopeShape.GetDim(2)) != headDimRope_ / (32U / kvRopeTypeSize) ||
                    static_cast<size_t>(keyRopeShape.GetDim(3)) != blockSize_ ||
                    static_cast<size_t>(keyRopeShape.GetDim(4)) != 32U / kvRopeTypeSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    ifaContext_->opName, "keyRope",
                    (std::to_string(keyRopeShape.GetDim(0)) + ", " + std::to_string(keyRopeShape.GetDim(1)) + ", " +
                     std::to_string(keyRopeShape.GetDim(2)) + ", " + std::to_string(keyRopeShape.GetDim(3)) + ", " +
                     std::to_string(keyRopeShape.GetDim(4)))
                        .c_str(),
                    ("KvLayout is NZ, keyRopeShape must be [" + std::to_string(keyRopeShape.GetDim(0)) + ", " +
                     std::to_string(numKvHeads_) + ", " + std::to_string(headDimRope_ / (32U / kvRopeTypeSize)) + ", " +
                     std::to_string(blockSize_) + ", " + std::to_string(32U / kvRopeTypeSize) + "].")
                        .c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRope() const
{
    if (!pageAttentionFlag_) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "pageAttention",
                                                 "only PageAttention KvCache is supported in MLA mode.");
        return ge::GRAPH_FAILED;
    }
    if (CheckMlaKeyRopeDesc() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape keyRopeShape = ifaContext_->keyRope.tensor->GetStorageShape();
    const gert::Shape keyShape = ifaContext_->key.shape->GetStorageShape();
    if (CheckMlaKeyRopeShapeMatch(keyRopeShape, keyShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (inputKvLayout_ == IfaLayout::BSH_BSND) { // PA Dims=3
        return CheckMlaKeyRopeBsndLayout(keyRopeShape, keyShape);
    } else if (inputKvLayout_ == IfaLayout::BNSD) { // Dims = 4
        return CheckMlaKeyRopeBnsdLayout(keyRopeShape, keyShape);
    } else if (inputKvLayout_ ==
               IfaLayout::NZ) { // Dims = 5   [B， N， headDim / (32 / sizeof(KV)), block_size 32/sizeof(KV)]
        return CheckMlaKeyRopeNzLayout(keyRopeShape, keyShape);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaMisc() const
{
    OP_CHECK_IF(antiQuantFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "antiquant",
                                                         "antiquant is not supported in MLA."),
                return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("MLA") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (antiQuantFlag_) {
        OP_CHECK_IF(inputQType_ != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        ifaContext_->opName, "query", DataTypeToSerialString(inputQType_).c_str(),
                        "when antiquant is enabled, input Q dtype must be bf16 in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputLayout_ == IfaLayout::BNSD,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        ifaContext_->opName, "inputLayout", std::string(ifaContext_->layOut).c_str(),
                        "when antiquant is enabled, BNSD or BNSD_NBSD input layout are not supported in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!pageAttentionFlag_,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        ifaContext_->opName, "pageAttention",
                        "when antiquant is enabled, Page Attention must be enabled in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            inputKvLayout_ != IfaLayout::NZ,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "inputKvLayout",
                                                     "when antiquant is enabled, KvCache layout must be NZ in MLA."),
            return ge::GRAPH_FAILED);
    }

    // Kv NZ blocksize：128
    OP_CHECK_IF(
        inputKvLayout_ == IfaLayout::NZ && blockSize_ != 128U,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                              "MLA only support {128} when KvCache layout is NZ."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        blockSize_ != 16U && blockSize_ != 128U,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                              "MLA only support {16, 128} when KvCache layout is ND."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckDefaultMisc(std::string scene) const
{
    OP_CHECK_IF(pseShiftFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "pseShift",
                                                         ("PseShift is not supported in " + scene + ".").c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!batchContinuousFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "kvcache",
                                                         ("Kvcache must be continuous in " + scene + ".").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(sysPrefixFlag_,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "sysPrefix",
                                                         ("SysPrefix is not supported in " + scene + ".").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputType_ == ge::DT_INT8,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "postQuant",
                                                         ("PostQuant is not supported in " + scene + ".").c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        kvPaddingSizeFlag_,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "kvPaddingSize",
                                                 ("kvPaddingSizeFlag_ is not supported in " + scene + ".").c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaTensor() const
{
    if (CheckGqaIOTensor() != ge::GRAPH_SUCCESS || CheckGqaAntiquantTensor() != ge::GRAPH_SUCCESS ||
        CheckGqaBlockTable() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // other tensors should be empty
    return CheckGqaTensorEmpty();
}

ge::graphStatus IFATiling::CheckGqaIOTensor() const
{
    if (!antiQuantFlag_) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "antiquant",
                                                 "IFA GQA with KV NZ only support antiquant!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        ifaContext_->query.desc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "query",
                                              DataTypeToSerialString(ifaContext_->query.desc->GetDataType()).c_str(),
                                              "In IFA GQA with KV NZ antiquant, query type should be BFLOAT16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.desc->GetDataType() != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    ifaContext_->opName, "key", DataTypeToSerialString(ifaContext_->key.desc->GetDataType()).c_str(),
                    "In IFA GQA with KV NZ antiquant, key type should be INT8."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->value.desc->GetDataType() != ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "value",
                                              DataTypeToSerialString(ifaContext_->value.desc->GetDataType()).c_str(),
                                              "In IFA GQA with KV NZ antiquant, value type should be INT8."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->attenOut.desc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(ifaContext_->opName, "attenOut",
                                              DataTypeToSerialString(ifaContext_->attenOut.desc->GetDataType()).c_str(),
                                              "In IFA GQA with KV NZ antiquant, attenOut type should be BFLOAT16."),
        return ge::GRAPH_FAILED);
    constexpr uint32_t kvNzDimNum = 5U;
    if (ifaContext_->key.shape->GetStorageShape().GetDimNum() == kvNzDimNum) { // NZ
        std::string kShapeStr = GetTensorDimString(ifaContext_->key.shape->GetStorageShape());
        std::string vShapeStr = GetTensorDimString(ifaContext_->value.shape->GetStorageShape());
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->key.shape->GetStorageShape()),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ifaContext_->opName, "key", kShapeStr.c_str(),
                                                           "The shape of K should not have 0 dim in GQA with KV NZ!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsZeroDimTensor(ifaContext_->value.shape->GetStorageShape()),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ifaContext_->opName, "value", vShapeStr.c_str(),
                                                           "The shape of V should not have 0 dim in GQA with KV NZ!"),
                    return ge::GRAPH_FAILED);
        uint32_t expected3rdDim = headDim_ / 32U;
        uint32_t expected5thDim = 32U; // 32U / sizeof(int8)
        bool isK3rdDimValid = expected3rdDim == ifaContext_->key.shape->GetStorageShape()[2];
        bool isK5thDimValid = expected5thDim == ifaContext_->key.shape->GetStorageShape()[4];
        OP_CHECK_IF(
            !isK3rdDimValid || !isK5thDimValid,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ifaContext_->opName, "key", kShapeStr.c_str(),
                ("The shape of K should be [" + std::to_string(ifaContext_->key.shape->GetStorageShape()[0]) + ", " +
                 std::to_string(ifaContext_->key.shape->GetStorageShape()[1]) + ", " + std::to_string(expected3rdDim) +
                 ", " + std::to_string(ifaContext_->key.shape->GetStorageShape()[3]) + ", " +
                 std::to_string(expected5thDim) + "] in GQA with KV NZ!")
                    .c_str()),
            return ge::GRAPH_FAILED);
        bool isV3rdDimValid = expected3rdDim == ifaContext_->value.shape->GetStorageShape()[2];
        bool isV5thDimValid = expected5thDim == ifaContext_->value.shape->GetStorageShape()[4];
        OP_CHECK_IF(!isV3rdDimValid || !isV5thDimValid,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "value", kShapeStr.c_str(),
                        ("The shape of V should be [" + std::to_string(ifaContext_->value.shape->GetStorageShape()[0]) +
                         ", " + std::to_string(ifaContext_->value.shape->GetStorageShape()[1]) + ", " +
                         std::to_string(expected3rdDim) + ", " +
                         std::to_string(ifaContext_->value.shape->GetStorageShape()[3]) + ", " +
                         std::to_string(expected5thDim) + "] in GQA with KV NZ!")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiquantTensor() const
{
    OP_CHECK_IF((ifaContext_->keyAntiquantScale.desc == nullptr || ifaContext_->valueAntiquantScale.desc == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "keyAntiquantScale, valueAntiquantScale"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ifaContext_->keyAntiquantScale.tensor == nullptr || ifaContext_->valueAntiquantScale.tensor == nullptr),
        OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "keyAntiquantScale, valueAntiquantScale"),
        return ge::GRAPH_FAILED);

    if (CheckGqaAntiquantType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiquantType() const
{
    // check type
    if (antiquantMode_ == PER_CHANNEL_MODE) {
        OP_CHECK_IF(ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        ifaContext_->opName, "keyAntiquantScale",
                        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str(),
                        "In IFA GQA with KV NZ perchannel antiquant, the key's dequant scale type should be BFLOAT16."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_BF16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                ifaContext_->opName, "valueAntiquantScale",
                DataTypeToSerialString(ifaContext_->valueAntiquantScale.desc->GetDataType()).c_str(),
                "In IFA GQA with KV NZ perchannel antiquant, the value's dequant scale type should be BFLOAT16."),
            return ge::GRAPH_FAILED);
    } else if (antiquantMode_ == PER_TOKEN_MODE) {
        OP_CHECK_IF(ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        ifaContext_->opName, "keyAntiquantScale",
                        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str(),
                        "In IFA GQA with KV NZ pertoken antiquant, the key's dequant scale type should be DT_FLOAT."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        ifaContext_->opName, "valueAntiquantScale",
                        DataTypeToSerialString(ifaContext_->valueAntiquantScale.desc->GetDataType()).c_str(),
                        "In IFA GQA with KV NZ pertoken antiquant, the value's dequant scale type should be DT_FLOAT."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaBlockTable() const
{
    OP_CHECK_IF(!pageAttentionFlag_, OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "blockTable"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaTensorEmpty() const
{
    OP_CHECK_IF(ifaContext_->pseShift.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "pseShift",
                                                         "PseShift not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSeqLengthsQ.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "actualSeqLengthsQ",
                                                         "The query's actual sequence lengths not support for IFA GQA "
                                                         "with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "deqScale1",
                                                         "deqScale1 not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "quantScale1",
                                                         "quantScale1 not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "deqScale2",
                                                         "deqScale2 not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->quantScale2.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "quantScale2",
                                                 "the output's dequant scale not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->quantOffset2.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "quantOffset2",
                                                 "the output's dequant offset not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "antiquantScale",
                                                         "antiquantScale not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantOffset.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "antiquantOffset",
                                                         "antiquantOffset not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryPaddingSize.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "queryPaddingSize",
                                                         "queryPaddingSize not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "kvPaddingSize",
                                                         "kvPaddingSize not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->keyAntiquantOffset.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keyAntiquantOffset",
                                                 "the key's dequant offset not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->valueAntiquantOffset.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "valueAntiquantOffset",
                                                 "the value's dequant offset not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keySharedPrefix",
                                                         "keySharedPrefix not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "valueSharedPrefix",
                                                         "valueSharedPrefix not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSharedPrefixLen.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "actualSharedPrefixLen",
                                                         "actualSharedPrefixLen not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryRope.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "queryRope",
                                                         "queryRope not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRope.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keyRope",
                                                         "keyRope not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->keyRopeAntiquantScale.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "keyRopeAntiquantScale",
                                                 "the key_rope's dequant scale not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->dequantScaleQuery.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "dequantScaleQuery",
                                                 "The query's dequant scale not support for IFA GQA with KV NZ!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->lseOut.desc != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(ifaContext_->opName, "lseOut",
                                                         "lseOut not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaSeqSize() const
{
    std::string layout(ifaContext_->layOut);
    if (layout == "TND" && isWorkspace_) { // tiling下沉场景
        return ge::GRAPH_SUCCESS;
    }
    if (qSeqSize_ == 1U) {
        if (layout == "TND") { // TND MTP场景qSeqSize有可能为1，也需要支持sparseMode3
            OP_CHECK_IF(
                (sparseMode_ != 0 && sparseMode_ != 3),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                    "SparseMode only support 0 or 3 in IFA GQA with KV NZ when query is 1 and layout is TND."),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (sparseMode_ != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                    "SparseMode only support 0 in IFA GQA with KV NZ when query is 1."),
                return ge::GRAPH_FAILED);
        }
    } else if (qSeqSize_ > 1U) {
        OP_CHECK_IF((sparseMode_ != 3), // when qs bigger than 1, sparse mode only support 3
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        ifaContext_->opName, "sparseMode", std::to_string(static_cast<int32_t>(sparseMode_)).c_str(),
                        "SparseMode only support 3 in IFA GQA with KV NZ when query S is bigger than 1."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "queryS", std::to_string(qSeqSize_).c_str(),
                                              "Invalid query S for IFA GQA with KV NZ.");
        return ge::GRAPH_FAILED;
    }
    if (sparseMode_ == 0) {
        OP_CHECK_IF(
            ifaContext_->attenMask.desc != nullptr || ifaContext_->attenMask.tensor != nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                ifaContext_->opName, "attenMask", "attenMask should be null for IFA GQA with KV NZ when sparseMode 0!"),
            return ge::GRAPH_FAILED);
    } else {
        auto attenMaskShape = ifaContext_->attenMask.desc;
        auto attenMaskTensor = ifaContext_->attenMask.tensor;
        OP_CHECK_IF(attenMaskShape == nullptr || attenMaskTensor == nullptr, // mask shape: 2048*2048
                    OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "attenMask"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDimNum() != 2U, // mask shape: 2048*2048
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        ifaContext_->opName, "attenMask",
                        std::to_string(attenMaskTensor->GetStorageShape().GetDimNum()).c_str(),
                        "The dim of attenMask is not expected. Expect 2 when sparseMode 3 in GQA KV NZ."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDim(0) != 2048U ||
                        attenMaskTensor->GetStorageShape().GetDim(1) != 2048U, // mask shape: 2048*2048
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        ifaContext_->opName, "attenMask",
                        (std::to_string(attenMaskTensor->GetStorageShape().GetDim(0)) + ", " +
                         std::to_string(attenMaskTensor->GetStorageShape().GetDim(1)))
                            .c_str(),
                        "The shape of attenMask is not expected. Expect [2048, 2048] when sparseMode 3 in GQA KV NZ."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAttribute() const
{
    OP_CHECK_IF(headDim_ != 128U, // 128: D dim
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "headDim", std::to_string(headDim_).c_str(),
                                                      "IFA GQA with KV NZ only support 128."),
                return ge::GRAPH_FAILED);

    std::string layout(ifaContext_->layOut);
    OP_CHECK_IF(layout != "BSH" && layout != "BSND" && layout != "BNSD" && layout != "TND",
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    ifaContext_->opName, "inputLayout", layout.c_str(),
                    "In IFA GQA with KV NZ antiquant, only BSH, BSND, BNSD and TND layout are supported."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ifaContext_->keyAntiquantMode == nullptr || ifaContext_->valueAntiquantMode == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(ifaContext_->opName, "keyAntiquantMode, valueAntiquantMode"),
                return ge::GRAPH_FAILED);
    int64_t keyAntiquantMode = ifaContext_->keyAntiquantMode != nullptr ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueAntiquantMode = ifaContext_->valueAntiquantMode != nullptr ? *ifaContext_->valueAntiquantMode : 0;
    OP_CHECK_IF(antiquantMode_ == PER_CHANNEL_MODE && (keyAntiquantMode != 0 || valueAntiquantMode != 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    ifaContext_->opName, "keyAntiquantMode, valueAntiquantMode",
                    (std::to_string(keyAntiquantMode) + ", " + std::to_string(valueAntiquantMode)).c_str(),
                    "the key's quant mode and the value's quant mode should be 0 in GQA perchannel antiquant."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(antiquantMode_ == PER_TOKEN_MODE && (keyAntiquantMode != 1 || valueAntiquantMode != 1),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    ifaContext_->opName, "keyAntiquantMode, valueAntiquantMode",
                    (std::to_string(keyAntiquantMode) + ", " + std::to_string(valueAntiquantMode)).c_str(),
                    "the key's quant mode and the value's quant mode should be 1 in GQA pertoken antiquant."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        blockSize_ != 128U && blockSize_ != 512U, // block size only support 128 and 512 in GQA
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ifaContext_->opName, "blockSize", std::to_string(blockSize_).c_str(),
                                              "only support 128 and 512 in GQA."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((innerPrecise_ != IFA_HIGH_PERFORMANCE),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "innerPrecise", std::to_string(innerPrecise_).c_str(),
                    "precision mode only support 1(high performance) in GQA antiquant with KV NZ."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckGqaSeqSize() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(ifaContext_->opName, "Invalid query S %u with sparseMode %d and attenMask",
                                       qSeqSize_, static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("IFA GQA with KV NZ") != ge::GRAPH_SUCCESS || CheckGqaDefault() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaDefault() const
{
    int64_t queryQuantMode = ifaContext_->queryQuantMode != nullptr ? *ifaContext_->queryQuantMode : 0;
    OP_CHECK_IF(queryQuantMode != DEQUANT_PER_CHANNEL_MODE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    ifaContext_->opName, "queryQuantMode", std::to_string(queryQuantMode).c_str(),
                    ("The query's quant mode for IFA GQA with KV NZ should be default value " +
                     std::to_string(DEQUANT_PER_CHANNEL_MODE) + "!")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CHECK_MLA_GQA_H_
