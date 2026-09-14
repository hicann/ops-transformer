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
                    OP_LOGE(ifaContext_->opName, "when the dtype of query is int8, queryRope [%d] must be bfloat16.",
                            ifaContext_->queryRope.desc->GetDataType()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ifaContext_->queryRope.desc->GetDataType() != ifaContext_->query.desc->GetDataType()),
                    OP_LOGE(ifaContext_->opName, "queryRope [%d] and query [%d] must have same dType",
                            ifaContext_->queryRope.desc->GetDataType(), ifaContext_->query.desc->GetDataType()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBsndLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF(qRopeShape.GetDim(0) != qShape.GetDim(0),
                OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'B'", qRopeShape.GetDim(0),
                        qShape.GetDim(0)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qRopeShape.GetDim(1) != qShape.GetDim(1),
                OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'S'", qRopeShape.GetDim(1),
                        qShape.GetDim(1)),
                return ge::GRAPH_FAILED);

    if (qRopeShape.GetDimNum() != 4U) {
        headDimRope_ = qRopeShape.GetDim(2) / numHeads_; // 2: H
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(qRopeShape.GetDim(2) != qShape.GetDim(2), // 2: N
                OP_LOGE(ifaContext_->opName, "queryRope [%ld] and query [%ld] must have same 'N'", qRopeShape.GetDim(2),
                        qShape.GetDim(2)),
                return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeBnsdLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF((qRopeShape.GetDim(0) != qShape.GetDim(0)) || (qRopeShape.GetDim(1) != qShape.GetDim(1)) ||
                    (qRopeShape.GetDim(2) != qShape.GetDim(2)), // 2: S
                OP_LOGE(ifaContext_->opName,
                        "queryRope [%ld, %ld, %ld, %ld] and query [%ld, %ld, %ld, %ld] must have same 'B', 'N' and 'S'",
                        qRopeShape.GetDim(0), qRopeShape.GetDim(1), qRopeShape.GetDim(2), qRopeShape.GetDim(3),
                        qShape.GetDim(0), qShape.GetDim(1), qShape.GetDim(2), qShape.GetDim(3)),
                return ge::GRAPH_FAILED);

    headDimRope_ = qRopeShape.GetDim(3); // 3:D
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaQueryRopeNzLayout(const gert::Shape &qRopeShape, const gert::Shape &qShape)
{
    OP_CHECK_IF(
        (qRopeShape.GetDim(0) != qShape.GetDim(0)) || (qRopeShape.GetDim(1) != qShape.GetDim(1)),
        OP_LOGE(ifaContext_->opName, "queryRope [%ld, %ld, %ld] and query [%ld, %ld, %ld] must have same 'T' and 'N'.",
                qRopeShape.GetDim(0), qRopeShape.GetDim(1), qRopeShape.GetDim(2), qShape.GetDim(0), qShape.GetDim(1),
                qShape.GetDim(2)),
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
                OP_LOGE(ifaContext_->opName, "queryRope(%lu) and query(%lu) dimensions must be the same.",
                        qRopeShape.GetDimNum(), qShape.GetDimNum()),
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
                OP_LOGE(ifaContext_->opName, "the key/value's heads num(%u) only support 1 in MLA.", numKvHeads_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(nNumOfQInOneGroup_ != 1U && nNumOfQInOneGroup_ != 2U && nNumOfQInOneGroup_ != 4U &&
                    nNumOfQInOneGroup_ != 8U && nNumOfQInOneGroup_ != 16U && nNumOfQInOneGroup_ != 32U &&
                    nNumOfQInOneGroup_ != 64U && nNumOfQInOneGroup_ != 128U,
                OP_LOGE(ifaContext_->opName,
                        "the query's heads num divided by the key/value's heads num = %u, MLA only support {1, 2, 4, "
                        "8, 16, 32, 64, 128}.",
                        nNumOfQInOneGroup_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDim_ != 512U, OP_LOGE(ifaContext_->opName, "queryD(%u) only support 512 in MLA.", headDim_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(headDimRope_ != 64U,
                OP_LOGE(ifaContext_->opName, "headDimRope(%u) only support 64 in MLA.", headDimRope_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (inputKvLayout_ == IfaLayout::NZ && inputLayout_ != IfaLayout::BSH_BSND && inputLayout_ != IfaLayout::TND),
        OP_LOGE(ifaContext_->opName,
                "When kv is NZ, layout only support {BSH, BSND, TND, BSH_NBSD, BSND_NBSD, TND_NTD}."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeDesc() const
{
    if (quantFlag_) {
        OP_CHECK_IF((ifaContext_->keyRope.desc->GetDataType() != ifaContext_->queryRope.desc->GetDataType()),
                    OP_LOGE(ifaContext_->opName,
                            "when the dtype of query is int8, keyRope [%d] and queryRope [%d] must have same dType.",
                            ifaContext_->keyRope.desc->GetDataType(), ifaContext_->queryRope.desc->GetDataType()),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((ifaContext_->keyRope.desc->GetDataType() != ifaContext_->key.desc->GetDataType()),
                    OP_LOGE(ifaContext_->opName, "keyRope [%d] and key [%d] must have same dType",
                            ifaContext_->keyRope.desc->GetDataType(), ifaContext_->key.desc->GetDataType()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeShapeMatch(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyRopeShape.GetShapeSize() == 0,
                OP_LOGE(ifaContext_->opName, "empty keyRope Tensor is not supported in MLA."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyShape.GetShapeSize() == 0, OP_LOGE(ifaContext_->opName, "empty key Tensor is not supported in MLA."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(0) != keyShape.GetDim(0),
                OP_LOGE(ifaContext_->opName, "KeyRope(%ld) and Key(%ld) must have same BlockNum",
                        keyRopeShape.GetDim(0), keyShape.GetDim(0)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyRopeShape.GetDimNum() != keyShape.GetDimNum(),
                OP_LOGE(ifaContext_->opName, "KeyRope(%lu) and Key(%lu) dimensions must be the same.",
                        keyRopeShape.GetDimNum(), keyShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBsndLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(
        keyShape.GetDim(1) != blockSize_ || keyShape.GetDim(2) != (numKvHeads_ * headDim_),
        OP_LOGE(ifaContext_->opName, "The dim of KeyShape is 3, KeyShape [%ld, %ld, %ld] must be [%ld, %u, %u] in MLA.",
                keyShape.GetDim(0), keyShape.GetDim(1), keyShape.GetDim(2), keyShape.GetDim(0), blockSize_,
                (numKvHeads_ * headDim_)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        keyRopeShape.GetDim(1) != blockSize_ || keyRopeShape.GetDim(2) != (numKvHeads_ * headDimRope_),
        OP_LOGE(ifaContext_->opName, "The dim of KeyShape is 3, KeyRopeShape [%ld, %ld, %ld] must be [%ld, %u, %u].",
                keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2), keyRopeShape.GetDim(0),
                blockSize_, (numKvHeads_ * headDimRope_)),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRopeBnsdLayout(const gert::Shape keyRopeShape, const gert::Shape keyShape) const
{
    OP_CHECK_IF(keyShape.GetDim(1) != numKvHeads_ || keyShape.GetDim(2) != blockSize_ || keyShape.GetDim(3) != headDim_,
                OP_LOGE(ifaContext_->opName,
                        "The dim of KeyShape is 4, KeyShape [%ld, %ld, %ld, %ld] must be [%ld, %u, %u, %u] in MLA.",
                        keyShape.GetDim(0), keyShape.GetDim(1), keyShape.GetDim(2), keyShape.GetDim(3),
                        keyShape.GetDim(0), numKvHeads_, blockSize_, headDim_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyRopeShape.GetDim(1) != numKvHeads_ || keyRopeShape.GetDim(2) != blockSize_ ||
                    keyRopeShape.GetDim(3) != headDimRope_,
                OP_LOGE(ifaContext_->opName,
                        "The dim of KeyShape is 4, KeyRopeShape [%ld, %ld, %ld, %ld] must be [%ld, %u, %u, %u].",
                        keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2), keyRopeShape.GetDim(3),
                        keyRopeShape.GetDim(0), numKvHeads_, blockSize_, headDimRope_),
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
                OP_LOGE(ifaContext_->opName,
                        "KvLayout is NZ and KvDtype is %s, keyShape [%ld, %ld, %ld, %ld, %ld] must be [%ld, %u, %lu, "
                        "%u, %lu] in MLA.",
                        DataTypeToSerialString(ifaContext_->key.desc->GetDataType()).c_str(), keyShape.GetDim(0),
                        keyShape.GetDim(1), keyShape.GetDim(2), keyShape.GetDim(3), keyShape.GetDim(4),
                        keyShape.GetDim(0), numKvHeads_, (headDim_ / (32 / kvTypeSize)), blockSize_, (32 / kvTypeSize)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(static_cast<size_t>(keyRopeShape.GetDim(1)) != numKvHeads_ ||
                    static_cast<size_t>(keyRopeShape.GetDim(2)) != headDimRope_ / (32U / kvRopeTypeSize) ||
                    static_cast<size_t>(keyRopeShape.GetDim(3)) != blockSize_ ||
                    static_cast<size_t>(keyRopeShape.GetDim(4)) != 32U / kvRopeTypeSize,
                OP_LOGE(ifaContext_->opName,
                        "KvLayout is NZ, keyRopeShape [%ld, %ld, %ld, %ld, %ld] must be [%ld, %u, %lu, %u, %lu].",
                        keyRopeShape.GetDim(0), keyRopeShape.GetDim(1), keyRopeShape.GetDim(2), keyRopeShape.GetDim(3),
                        keyRopeShape.GetDim(4), keyRopeShape.GetDim(0), numKvHeads_,
                        (headDimRope_ / (32U / kvRopeTypeSize)), blockSize_, (32U / kvRopeTypeSize)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckMlaKeyRope() const
{
    if (!pageAttentionFlag_) {
        OP_LOGE(ifaContext_->opName, "only PageAttention KvCache is supported in MLA mode");
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
    OP_CHECK_IF(antiQuantFlag_, OP_LOGE(ifaContext_->opName, "antiquant is not supported in MLA."),
                return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("MLA") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (antiQuantFlag_) {
        OP_CHECK_IF(inputQType_ != ge::DT_BF16,
                    OP_LOGE(ifaContext_->opName, "when antiquant is enabled, input Q dtype must be bf16 in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputLayout_ == IfaLayout::BNSD,
                    OP_LOGE(ifaContext_->opName,
                            "when antiquant is enabled, BNSD or BNSD_NBSD input layout are not supported in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!pageAttentionFlag_,
                    OP_LOGE(ifaContext_->opName, "when antiquant is enabled, Page Attention must be enabled in MLA."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputKvLayout_ != IfaLayout::NZ,
                    OP_LOGE(ifaContext_->opName, "when antiquant is enabled, KvCache layout must be NZ in MLA."),
                    return ge::GRAPH_FAILED);
    }

    // Kv NZ blocksize：128
    OP_CHECK_IF(
        inputKvLayout_ == IfaLayout::NZ && blockSize_ != 128U,
        OP_LOGE(ifaContext_->opName, "blockSize(%u), MLA only support {128} when KvCache layout is NZ.", blockSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ != 16U && blockSize_ != 128U,
                OP_LOGE(ifaContext_->opName, "blockSize(%u), MLA only support {16, 128} when KvCache layout is ND.",
                        blockSize_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckDefaultMisc(std::string scene) const
{
    OP_CHECK_IF(pseShiftFlag_, OP_LOGE(ifaContext_->opName, "PseShift is not supported in %s.", scene.c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!batchContinuousFlag_, OP_LOGE(ifaContext_->opName, "Kvcache must be continuous in %s.", scene.c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(sysPrefixFlag_, OP_LOGE(ifaContext_->opName, "SysPrefix is not supported in %s.", scene.c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputType_ == ge::DT_INT8,
                OP_LOGE(ifaContext_->opName, "PostQuant is not supported in %s.", scene.c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvPaddingSizeFlag_,
                OP_LOGE(ifaContext_->opName, "kvPaddingSizeFlag_ is not supported in %s.", scene.c_str()),
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
        OP_LOGE(ifaContext_->opName, "IFA GQA with KV NZ only support antiquant!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        ifaContext_->query.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                                    "In IFA GQA with KV NZ antiquant, query type should be BFLOAT16, but now it's %d!",
                                    ifaContext_->query.desc->GetDataType()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->key.desc->GetDataType() != ge::DT_INT8,
                OPS_REPORT_VECTOR_INNER_ERR(
                    ifaContext_->opName, "In IFA GQA with KV NZ antiquant, key type should be INT8, but now it's %d!",
                    ifaContext_->key.desc->GetDataType()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->value.desc->GetDataType() != ge::DT_INT8,
                OPS_REPORT_VECTOR_INNER_ERR(
                    ifaContext_->opName, "In IFA GQA with KV NZ antiquant, value type should be INT8, but now it's %d!",
                    ifaContext_->value.desc->GetDataType()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ifaContext_->attenOut.desc->GetDataType() != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(
            ifaContext_->opName, "In IFA GQA with KV NZ antiquant, attenOut type should be BFLOAT16, but now it's %d!",
            ifaContext_->attenOut.desc->GetDataType()),
        return ge::GRAPH_FAILED);
    constexpr uint32_t kvNzDimNum = 5U;
    if (ifaContext_->key.shape->GetStorageShape().GetDimNum() == kvNzDimNum) { // NZ
        std::string kShapeStr = GetTensorDimString(ifaContext_->key.shape->GetStorageShape());
        std::string vShapeStr = GetTensorDimString(ifaContext_->value.shape->GetStorageShape());
        OP_CHECK_IF(
            IsZeroDimTensor(ifaContext_->key.shape->GetStorageShape()),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                                        "The shape of K%s should not have 0 dim in GQA with KV NZ!", kShapeStr.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            IsZeroDimTensor(ifaContext_->value.shape->GetStorageShape()),
            OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                                        "The shape of V%s should not have 0 dim in GQA with KV NZ!", vShapeStr.c_str()),
            return ge::GRAPH_FAILED);
        uint32_t expected3rdDim = headDim_ / 32U;
        uint32_t expected5thDim = 32U; // 32U / sizeof(int8)
        bool isK3rdDimValid = expected3rdDim == ifaContext_->key.shape->GetStorageShape()[2];
        bool isK5thDimValid = expected5thDim == ifaContext_->key.shape->GetStorageShape()[4];
        OP_CHECK_IF(!isK3rdDimValid || !isK5thDimValid,
                    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                                                "The shape of K%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                                                kShapeStr.c_str(), ifaContext_->key.shape->GetStorageShape()[0],
                                                ifaContext_->key.shape->GetStorageShape()[1], expected3rdDim,
                                                ifaContext_->key.shape->GetStorageShape()[3], expected5thDim),
                    return ge::GRAPH_FAILED);
        bool isV3rdDimValid = expected3rdDim == ifaContext_->value.shape->GetStorageShape()[2];
        bool isV5thDimValid = expected5thDim == ifaContext_->value.shape->GetStorageShape()[4];
        OP_CHECK_IF(!isV3rdDimValid || !isV5thDimValid,
                    OPS_REPORT_VECTOR_INNER_ERR(ifaContext_->opName,
                                                "The shape of V%s should be [%ld, %ld, %u, %ld, %u] in GQA with KV NZ!",
                                                kShapeStr.c_str(), ifaContext_->value.shape->GetStorageShape()[0],
                                                ifaContext_->value.shape->GetStorageShape()[1], expected3rdDim,
                                                ifaContext_->value.shape->GetStorageShape()[3], expected5thDim),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAntiquantTensor() const
{
    OP_CHECK_IF((ifaContext_->keyAntiquantScale.desc == nullptr || ifaContext_->valueAntiquantScale.desc == nullptr),
                OP_LOGE(ifaContext_->opName,
                        "In IFA GQA with KV NZ antiquant, the key/value's dequant scale desc should not be null!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ifaContext_->keyAntiquantScale.tensor == nullptr || ifaContext_->valueAntiquantScale.tensor == nullptr),
        OP_LOGE(ifaContext_->opName,
                "In IFA GQA with KV NZ antiquant, the key/value's dequant scale tensor should not be null!"),
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
                    OPS_REPORT_VECTOR_INNER_ERR(
                        ifaContext_->opName,
                        "In IFA GQA with KV NZ perchannel antiquant, the key's dequant scale type should be BFLOAT16, "
                        "but now it's %s!",
                        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_BF16,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        ifaContext_->opName,
                        "In IFA GQA with KV NZ perchannel antiquant, the value's dequant scale type should be "
                        "BFLOAT16, but now it's %s!",
                        DataTypeToSerialString(ifaContext_->valueAntiquantScale.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (antiquantMode_ == PER_TOKEN_MODE) {
        OP_CHECK_IF(ifaContext_->keyAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        ifaContext_->opName,
                        "In IFA GQA with KV NZ pertoken antiquant, the key's dequant scale type should be DT_FLOAT, "
                        "but now it's %s!",
                        DataTypeToSerialString(ifaContext_->keyAntiquantScale.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(ifaContext_->valueAntiquantScale.desc->GetDataType() != ge::DT_FLOAT,
                    OPS_REPORT_VECTOR_INNER_ERR(
                        ifaContext_->opName,
                        "In IFA GQA with KV NZ pertoken antiquant, the value's dequant scale type should be DT_FLOAT, "
                        "but now it's %s!",
                        DataTypeToSerialString(ifaContext_->valueAntiquantScale.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaBlockTable() const
{
    OP_CHECK_IF(!pageAttentionFlag_,
                OP_LOGE(ifaContext_->opName, "In IFA GQA with KV NZ antiquant, blocktable should not be null!"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaTensorEmpty() const
{
    OP_CHECK_IF(ifaContext_->pseShift.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "PseShift not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSeqLengthsQ.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "The query's actual sequence lengths not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale1.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "deqScale1 not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale1.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "quantScale1 not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->deqScale2.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "deqScale2 not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantScale2.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "the output's dequant scale not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->quantOffset2.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "the output's dequant offset not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantScale.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "antiquantScale not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->antiquantOffset.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "antiquantOffset not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryPaddingSize.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "queryPaddingSize not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->kvPaddingSize.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "kvPaddingSize not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyAntiquantOffset.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "the key's dequant offset not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueAntiquantOffset.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "the value's dequant offset not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keySharedPrefix.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "keySharedPrefix not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->valueSharedPrefix.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "valueSharedPrefix not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->actualSharedPrefixLen.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "actualSharedPrefixLen not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->queryRope.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "queryRope not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRope.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "keyRope not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->keyRopeAntiquantScale.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "the key_rope's dequant scale not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->dequantScaleQuery.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "The query's dequant scale not support for IFA GQA with KV NZ!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ifaContext_->lseOut.desc != nullptr,
                OP_LOGE(ifaContext_->opName, "lseOut not support for IFA GQA with KV NZ!"), return ge::GRAPH_FAILED);
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
                OP_LOGE(ifaContext_->opName,
                        "SparseMode[%d] only support 0 or 3 in IFA GQA with KV NZ when query is 1 and layout is TND",
                        static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                (sparseMode_ != 0),
                OP_LOGE(ifaContext_->opName, "SparseMode[%d] only support 0 in IFA GQA with KV NZ when query is 1",
                        static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);
        }
    } else if (qSeqSize_ > 1U) {
        OP_CHECK_IF((sparseMode_ != 3), // when qs bigger than 1, sparse mode only support 3
                    OP_LOGE(ifaContext_->opName,
                            "SparseMode[%d] only support 3 in IFA GQA with KV NZ when query S is bigger than 1",
                            static_cast<int32_t>(sparseMode_)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(ifaContext_->opName, "Invalid query S %u", qSeqSize_);
        return ge::GRAPH_FAILED;
    }
    if (sparseMode_ == 0) {
        OP_CHECK_IF(ifaContext_->attenMask.desc != nullptr || ifaContext_->attenMask.tensor != nullptr,
                    OP_LOGE(ifaContext_->opName, "attenMask should be null for IFA GQA with KV NZ when sparseMode 0!"),
                    return ge::GRAPH_FAILED);
    } else {
        auto attenMaskShape = ifaContext_->attenMask.desc;
        auto attenMaskTensor = ifaContext_->attenMask.tensor;
        OP_CHECK_IF(attenMaskShape == nullptr || attenMaskTensor == nullptr, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "When sparseMode 3, attenMask shape for IFA GQA with KV NZ should not be null!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDimNum() != 2U, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "The dim of attenMask shape[%lu] is not expected. "
                            "Expect 2 when sparseMode 3 in GQA KV NZ.",
                            attenMaskTensor->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(attenMaskTensor->GetStorageShape().GetDim(0) != 2048U ||
                        attenMaskTensor->GetStorageShape().GetDim(1) != 2048U, // mask shape: 2048*2048
                    OP_LOGE(ifaContext_->opName,
                            "The shape of attenMask shape[%ld, %ld] is not expected. "
                            "Expect [2048, 2048] when sparseMode 3 in GQA KV NZ.",
                            attenMaskTensor->GetStorageShape().GetDim(0), attenMaskTensor->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaAttribute() const
{
    OP_CHECK_IF(headDim_ != 128U, // 128: D dim
                OP_LOGE(ifaContext_->opName, "headDim = %u, IFA GQA with KV NZ only support 128.", headDim_),
                return ge::GRAPH_FAILED);

    std::string layout(ifaContext_->layOut);
    OP_CHECK_IF(
        layout != "BSH" && layout != "BSND" && layout != "BNSD" && layout != "TND",
        OP_LOGE(ifaContext_->opName,
                "In IFA GQA with KV NZ antiquant, only BSH, BSND, BNSD and TND layout are supported, but now it's %s",
                layout.c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ifaContext_->keyAntiquantMode == nullptr || ifaContext_->valueAntiquantMode == nullptr,
                OP_LOGE(ifaContext_->opName, "the key's quant mode or the value's quant mode is null!"),
                return ge::GRAPH_FAILED);
    int64_t keyAntiquantMode = ifaContext_->keyAntiquantMode != nullptr ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueAntiquantMode = ifaContext_->valueAntiquantMode != nullptr ? *ifaContext_->valueAntiquantMode : 0;
    OP_CHECK_IF(
        antiquantMode_ == PER_CHANNEL_MODE && (keyAntiquantMode != 0 || valueAntiquantMode != 0),
        OP_LOGE(ifaContext_->opName,
                "the key's quant mode(%ld) and the value's quant mode(%ld) should be 0 in GQA perchannel antiquant.",
                keyAntiquantMode, valueAntiquantMode),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        antiquantMode_ == PER_TOKEN_MODE && (keyAntiquantMode != 1 || valueAntiquantMode != 1),
        OP_LOGE(ifaContext_->opName,
                "the key's quant mode(%ld) and the value's quant mode(%ld) should be 1 in GQA pertoken antiquant.",
                keyAntiquantMode, valueAntiquantMode),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ != 128U && blockSize_ != 512U, // block size only support 128 and 512 in GQA
                OP_LOGE(ifaContext_->opName, "blockSize(%u) only support 128 and 512 in GQA.", blockSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (innerPrecise_ != IFA_HIGH_PERFORMANCE),
        OP_LOGE(ifaContext_->opName, "precision mode[%u] only support 1(high performance) in GQA antiquant with KV NZ",
                innerPrecise_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckGqaSeqSize() != ge::GRAPH_SUCCESS,
                OP_LOGE(ifaContext_->opName, "Invalid query S %u with sparseMode %d and attenMask", qSeqSize_,
                        static_cast<int32_t>(sparseMode_)),
                return ge::GRAPH_FAILED);

    if (CheckDefaultMisc("IFA GQA with KV NZ") != ge::GRAPH_SUCCESS || CheckGqaDefault() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IFATiling::CheckGqaDefault() const
{
    int64_t queryQuantMode = ifaContext_->queryQuantMode != nullptr ? *ifaContext_->queryQuantMode : 0;
    OP_CHECK_IF(
        queryQuantMode != DEQUANT_PER_CHANNEL_MODE,
        OP_LOGE(ifaContext_->opName, "The query's quant mode(%ld) for IFA GQA with KV NZ should be default value %u!",
                queryQuantMode, DEQUANT_PER_CHANNEL_MODE),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_CHECK_MLA_GQA_H_
