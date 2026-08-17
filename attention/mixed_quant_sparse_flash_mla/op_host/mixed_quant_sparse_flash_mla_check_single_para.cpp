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
 * \file mixed_quant_sparse_flash_mla_check_single_para.cpp
 * \brief
 */

#include "mixed_quant_sparse_flash_mla_check.h"
#include "../op_kernel/mixed_quant_sparse_flash_mla_metadata.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;
namespace optiling {

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_BF16}},
    {ORI_KV_NAME, {ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8}},
    {CMP_KV_NAME, {ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8}},
    {ORI_SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {ATTEN_OUT_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {CMP_SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {ORI_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CMP_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_ORI_KV_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_CMP_KV_NAME, {ge::DT_INT32}},
    {SEQUSED_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_ORI_KV_NAME, {ge::DT_INT32}},
    {SEQUSED_CMP_KV_NAME, {ge::DT_INT32}},
    {CMP_RESIDUAL_KV_NAME, {ge::DT_INT32}},
    {ORI_TOPK_LENGTH_NAME, {ge::DT_INT32}},
    {CMP_TOPK_LENGTH_NAME, {ge::DT_INT32}},
    {SINKS_NAME, {ge::DT_FLOAT}},
    {METADATA_NAME, {ge::DT_INT32}},
};

const std::map<std::string, std::vector<MQSMLALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {MQSMLALayout::BSND, MQSMLALayout::TND}},
    {ORI_KV_NAME, {MQSMLALayout::BSND, MQSMLALayout::PA_BBND, MQSMLALayout::TND}},
    {CMP_KV_NAME, {MQSMLALayout::BSND, MQSMLALayout::PA_BBND, MQSMLALayout::TND}},
    {ATTEN_OUT_NAME, {MQSMLALayout::BSND, MQSMLALayout::TND}},
};

template <typename T>
void MQSMLATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                              const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus MQSMLATilingCheck::CheckDimNumInLayoutSupport(const MQSMLALayout &layout,
                                                              const gert::StorageShape *shape,
                                                              const std::string &name) const
{
    const auto &dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, name.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                    "When layout is " + MQSMLALayoutToSerialString(layout) + ", the shape dim of " + name +
                        " should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
                                                      const std::vector<size_t> &expectDimNumList,
                                                      const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
        expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckShapeNumSupport(const gert::StorageShape *shape,
                                                        const std::vector<int64_t> &expectShapeNumList,
                                                        const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectShapeNumList.begin(), expectShapeNumList.end(), shape->GetStorageShape().GetShapeSize()) ==
        expectShapeNumList.end()) {
        LogErrorDimNumSupport(expectShapeNumList, shape->GetStorageShape().GetShapeSize(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void MQSMLATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                             const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << MQSMLADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, name.c_str(), MQSMLADataTypeToSerialString(actualDtype).c_str(),
                                          "The dtype of " + name + " only supports " + oss.str());
}

ge::graphStatus MQSMLATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                     const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void MQSMLATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                              const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s %s only support %s, but got %s", name.c_str(), subName.c_str(), oss.str().c_str(),
            std::to_string(actualValue).c_str());
}

void MQSMLATilingCheck::LogErrorLayoutSupport(const std::vector<MQSMLALayout> &expectLayoutList,
                                              const MQSMLALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << MQSMLALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, name.c_str(), MQSMLALayoutToSerialString(actualLayout).c_str(),
                                          "Tensor " + name + " only supports layout " + oss.str());
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaQuery() const
{
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "Q is required, but got nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.q.shape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "q", std::to_string(opParamInfo_.q.shape->GetStorageShape().GetShapeSize()).c_str(),
                    "Any dim of q cannot be 0"),
                return ge::GRAPH_FAILED);

    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_kv", "The tensor of ori_kv is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor->GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "ori_kv", std::to_string(opParamInfo_.oriKv.tensor->GetShapeSize()).c_str(),
                    "Any dim of ori_kv cannot be 0"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), keyDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ == MQSMLALayout::PA_BBND) {
        OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > 1024,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "When page attention is enabled, ori_block_size(" + std::to_string(oriBlockSize_) +
                            ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]"),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cmp_kv", std::to_string(opParamInfo_.cmpKv.tensor->GetShapeSize()).c_str(),
                        "Any dim of cmp_kv cannot be 0"),
                    return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), keyDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }

        uint32_t cmpKvN2Size_ = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), MQSMLAAxis::N, kvLayout_);
        OP_CHECK_IF(cmpKvN2Size_ != n2Size_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "cmp_kv and ori_kv",
                        Ops::Base::ToString(opParamInfo_.cmpKv.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of ori_kv(" + std::to_string(cmpKvN2Size_) +
                            ") should be equal to the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);

        if (kvLayout_ == MQSMLALayout::PA_BBND) {
            OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > 1024,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_kv", ToStringRaw(opParamInfo_.cmpKv.tensor->GetStorageShape()).c_str(),
                            "When page attention is enabled, cmp_block_size(" + std::to_string(cmpBlockSize_) +
                                ") should be in range (0, " + std::to_string(MAX_BLOCK_SIZE) + "]"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckLayoutSupport(const MQSMLALayout &actualLayout, const std::string &name) const
{
    const auto &it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
                LogErrorLayoutSupport(expectLayoutList, actualLayout, name), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaNumHeads() const
{
    OP_CHECK_IF(n1Size_ < 1 || n1Size_ > 128,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "q",
                                                      ToStringRaw(opParamInfo_.q.shape->GetStorageShape()).c_str(),
                                                      "The head num of q should be in [1, 128]"
                                                      ", but got " +
                                                          std::to_string(n1Size_)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaKvHeadNums() const
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        OP_CHECK_IF(n2Size_ != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "The head num of ori_kv should be 1, but got " + std::to_string(n2Size_)),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(n2Size_ != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_kv", ToStringRaw(opParamInfo_.cmpKv.tensor->GetStorageShape()).c_str(),
                        "The head num of cmp_kv should be 1, but got " + std::to_string(n2Size_)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF(oriMaskMode_ != 0 && oriMaskMode_ != 3 && oriMaskMode_ != 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_mask_mode", std::to_string(oriMaskMode_).c_str(),
                                                      "Ori_mask_mode only supports 0, 3 and 4"),
                return ge::GRAPH_FAILED);
    if (perfMode_ == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(
            cmpMaskMode_ != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_mask_mode", std::to_string(cmpMaskMode_).c_str(),
                                                  "Cmp_mask_mode only supports 0 in SWA mode"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            cmpMaskMode_ != 0 && cmpMaskMode_ != 3,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_mask_mode", std::to_string(cmpMaskMode_).c_str(),
                                                  "Cmp_mask_mode only supports 0 and 3 in non-SWA mode"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF(sparseBlockSize_ != 1,
                OP_LOGE(opName_, "sparseBlockSize_ only support 1, but got %u", sparseBlockSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCmpResidualKv() const
{
    OP_CHECK_IF(perfMode_ != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                    perfMode_ != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE && cmpMaskMode_ != 0 &&
                    opParamInfo_.cmpResidualKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_residual_kv",
                                                         "Cmp_residual_kv is required in CSA and HCA mode"),
                return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpResidualKv.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.cmpResidualKv.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_residual_kv", std::to_string(opParamInfo_.cmpResidualKv.tensor->GetShapeSize()).c_str(),
                "Any dim of cmp_residual_kv cannot be 0"),
            return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpResidualKv.desc, CMP_RESIDUAL_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpResidualKv.tensor->GetShape(), expectDimNumList,
                                                    CMP_RESIDUAL_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCmpSparseIndices() const
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cmp_sparse_indices",
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetShapeSize()).c_str(),
                        "Any dim of cmp_sparse_indices cannot be 0"),
                    return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES_NAME)) {
            return ge::GRAPH_FAILED;
        }

        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(cmpSparseShape.GetDimNum() != DIM_NUM_FOUR,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_sparse_indices", ToStringRaw(cmpSparseShape).c_str(),
                            "When layout_q is BSND, the dim num of cmp_sparse_indices should be 4, but got " +
                                std::to_string(cmpSparseShape.GetDimNum())),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(cmpSparseShape.GetDimNum() != DIM_NUM_THREE,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_sparse_indices", ToStringRaw(cmpSparseShape).c_str(),
                            "When q layout is TND, cmp_sparse_indices should be 3, but got " +
                                std::to_string(cmpSparseShape.GetDimNum())),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaBlockTable() const
{
    OP_CHECK_IF(kvLayout_ == MQSMLALayout::PA_BBND && opParamInfo_.oriBlockTable.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_block_table",
                                                         "Ori_block_table must not be empty when layout_kv is PA_BBND"),
                return ge::GRAPH_FAILED);

    const std::vector<size_t> BlockTableDimNumList = {DIM_NUM_TWO};
    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.oriBlockTable.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "ori_block_table", std::to_string(opParamInfo_.oriBlockTable.tensor->GetShapeSize()).c_str(),
                "Any dim of ori_block_table cannot be 0"),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(),
                                                    BlockTableDimNumList, ORI_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.cmpBlockTable.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_block_table", std::to_string(opParamInfo_.cmpBlockTable.tensor->GetShapeSize()).c_str(),
                "Any dim of cmp_block_table cannot be 0"),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(),
                                                    BlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCuSeqLensQ() const
{
    if (qLayout_ == MQSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((qLayout_ == MQSMLALayout::TND && opParamInfo_.cuSeqLensQ.tensor == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                         "Cu_seqlens_q can't be nullptr when layout_q is TND"),
                return ge::GRAPH_FAILED);

    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_q", std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                        "Any dim of cu_seqlens_q cannot be 0"),
                    return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensQ.desc, CU_SEQLENS_Q_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.cuSeqLensQ.tensor->GetShape(), expectDimNumList, CU_SEQLENS_Q_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSequsedKv() const
{
    if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
        perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    } else {
        OP_CHECK_IF(
            opParamInfo_.sequsedOriKv.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "seqused_ori_kv", "Seqused_ori_kv can not be nullptr in non-BSND layout, but it's empty"),
            return ge::GRAPH_FAILED);
    }
    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedOriKv.desc, SEQUSED_ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.sequsedOriKv.tensor->GetShape(), expectDimNumList, SEQUSED_ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "seqused_ori_kv", std::to_string(opParamInfo_.sequsedOriKv.tensor->GetShapeSize()).c_str(),
                    "The shape size of seqused_ori_kv is not equal to B:" + std::to_string(bSize_)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == MQSMLALayout::PA_BBND && opParamInfo_.sequsedOriKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "seqused_ori_kv",
                                                         "Seqused_ori_kv must not be empty when kvLayout is PA_BBND"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSinks() const
{
    OP_CHECK_IF(opParamInfo_.sinks.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sinks", "Sinks is nullptr, which is not supported"),
                return ge::GRAPH_FAILED);

    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sinks.desc, SINKS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.sinks.tensor->GetShape(), expectDimNumList, SINKS_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sinks.tensor->GetShapeSize() != n1Size_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "sinks", std::to_string(opParamInfo_.sinks.tensor->GetShapeSize()).c_str(),
                    "The shape size of sinks is not equal to the head num of q:" + std::to_string(n1Size_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaMetadata() const
{
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "metadata", "Metadata is required, but got nullptr"),
                return ge::GRAPH_FAILED);

    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.metadata.desc, METADATA_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.metadata.tensor->GetShape(), expectDimNumList, METADATA_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "seqused_cmp_kv", std::to_string(opParamInfo_.metadata.tensor->GetShapeSize()),
                    "The shape size of metadata should be " + std::to_string(METADATA_SIZE)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaOriSparseIndices() const
{
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "ori_sparse_indices",
                        std::to_string(opParamInfo_.oriSparseIndices.tensor->GetShapeSize()).c_str(),
                        "Any dim of ori_sparse_indices cannot be 0"),
                    return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriSparseIndices.desc, ORI_SPARSE_INDICES_NAME)) {
            return ge::GRAPH_FAILED;
        }

        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == MQSMLALayout::BSND) {
            OP_CHECK_IF(oriSparseShape.GetDimNum() != DIM_NUM_FOUR,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "ori_sparse_indices", ToStringRaw(oriSparseShape).c_str(),
                            "When layout_q is BSND, the dim num of ori_sparse_indices should be 4, but got " +
                                std::to_string(oriSparseShape.GetDimNum())),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(oriSparseShape.GetDimNum() != DIM_NUM_THREE,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "ori_sparse_indices", ToStringRaw(oriSparseShape).c_str(),
                            "When layout_q is TND, the dim num of ori_sparse_indices should be 3, but got " +
                                std::to_string(oriSparseShape.GetDimNum())),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCuSeqLensOriKv() const
{
    if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
        perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        opParamInfo_.oriKv.tensor != nullptr && kvLayout_ == MQSMLALayout::TND &&
            opParamInfo_.cuSeqLensOriKv.tensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, "cu_seqlens_ori_kv", "Cu_seqlens_ori_kv is required when ori_kv is provided and layout_kv is TND"),
        return ge::GRAPH_FAILED);

    if (opParamInfo_.cuSeqLensOriKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_ori_kv",
                        std::to_string(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize()).c_str(),
                        "Any dim of cu_seqlens_ori_kv cannot be 0"),
                    return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensOriKv.desc, CU_SEQLENS_ORI_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cuSeqLensOriKv.tensor->GetShape(), expectDimNumList,
                                                    CU_SEQLENS_ORI_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCuSeqLensCmpKv() const
{
    if (opParamInfo_.cuSeqLensCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "cu_seqlens_cmp_kv",
                        std::to_string(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize()).c_str(),
                        "Any dim of cu_seqlens_cmp_kv cannot be 0"),
                    return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensCmpKv.desc, CU_SEQLENS_CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cuSeqLensCmpKv.tensor->GetShape(), expectDimNumList,
                                                    CU_SEQLENS_CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSequsedQ() const
{
    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor->GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "seqused_q", std::to_string(opParamInfo_.seqUsedQ.tensor->GetShapeSize()).c_str(),
                        "Any dim of seqused_q cannot be 0"),
                    return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.seqUsedQ.desc, SEQUSED_Q_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.seqUsedQ.tensor->GetShape(), expectDimNumList, SEQUSED_Q_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaSequsedCmpKv() const
{
    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "seqused_cmp_kv", std::to_string(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize()).c_str(),
                "Any dim of seqused_cmp_kv cannot be 0"),
            return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedCmpKv.desc, SEQUSED_CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.sequsedCmpKv.tensor->GetShape(), expectDimNumList,
                                                    SEQUSED_CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(
            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "seqused_cmp_kv", ToStringRaw(opParamInfo_.seqUsedQ.tensor->GetStorageShape()),
                "The shape size of seqused_cmp_kv(" + std::to_string(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize()) +
                    ") should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaOriTopkLength() const
{
    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.oriTopkLength.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "ori_topk_length", std::to_string(opParamInfo_.oriTopkLength.tensor->GetShapeSize()).c_str(),
                "Any dim of ori_topk_length cannot be 0"),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriTopkLength.desc, ORI_TOPK_LENGTH_NAME)) {
            return ge::GRAPH_FAILED;
        }

        if (qLayout_ == MQSMLALayout::TND) {
            const std::vector<size_t> expectDimNumList = {DIM_NUM_TWO};
            if (ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriTopkLength.tensor->GetShape(),
                                                        expectDimNumList, ORI_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
        } else {
            const std::vector<size_t> expectDimNumList = {DIM_NUM_THREE};
            if (ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriTopkLength.tensor->GetShape(),
                                                        expectDimNumList, ORI_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaCmpTopkLength() const
{
    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        OP_CHECK_IF(
            opParamInfo_.cmpTopkLength.tensor->GetShapeSize() == 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_topk_length", std::to_string(opParamInfo_.cmpTopkLength.tensor->GetShapeSize()).c_str(),
                "Any dim of cmp_topk_length cannot be 0"),
            return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpTopkLength.desc, CMP_TOPK_LENGTH_NAME)) {
            return ge::GRAPH_FAILED;
        }

        if (qLayout_ == MQSMLALayout::TND) {
            const std::vector<size_t> expectDimNumList = {DIM_NUM_TWO};
            if (ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpTopkLength.tensor->GetShape(),
                                                        expectDimNumList, CMP_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
        } else {
            const std::vector<size_t> expectDimNumList = {DIM_NUM_THREE};
            if (ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpTopkLength.tensor->GetShape(),
                                                        expectDimNumList, CMP_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSingleParaTopkValueMode() const
{
    if (perfMode_ == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(
            topkValueMode_ != 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk_value_mode", std::to_string(topkValueMode_).c_str(),
                                                  "Topk_value_mode must be 1 in SWA mode"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            topkValueMode_ != 1 && topkValueMode_ != 2,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk_value_mode", std::to_string(topkValueMode_).c_str(),
                                                  "Topk_value_mode only support 1 and 2"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MQSMLATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() || ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpSparseIndices() || ge::GRAPH_SUCCESS != CheckSingleParaCmpResidualKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() || ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() || ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaBlockTable() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensOriKv() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSequsedQ() || ge::GRAPH_SUCCESS != CheckSingleParaSequsedKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSequsedCmpKv() || ge::GRAPH_SUCCESS != CheckSingleParaOriTopkLength() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpTopkLength() || ge::GRAPH_SUCCESS != CheckSingleParaMetadata() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSinks() || ge::GRAPH_SUCCESS != CheckSingleParaTopkValueMode()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
