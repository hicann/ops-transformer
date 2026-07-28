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
 * \file quant_sparse_flash_mla_check_single_para.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"
#include "../op_kernel/quant_sparse_flash_mla_metadata.h"

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
    {QUERY_NAME, {ge::DT_HIFLOAT8}},          {ORI_KV_NAME, {ge::DT_HIFLOAT8}},
    {CMP_KV_NAME, {ge::DT_HIFLOAT8}},         {ORI_SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {ATTEN_OUT_NAME, {ge::DT_BF16}},          {CMP_SPARSE_INDICES_NAME, {ge::DT_INT32}},
    {ORI_BLOCK_TABLE_NAME, {ge::DT_INT32}},   {CMP_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME, {ge::DT_INT32}},      {CU_SEQLENS_ORI_KV_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_CMP_KV_NAME, {ge::DT_INT32}}, {SEQUSED_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_ORI_KV_NAME, {ge::DT_INT32}},    {SEQUSED_CMP_KV_NAME, {ge::DT_INT32}},
    {CMP_RESIDUAL_KV_NAME, {ge::DT_INT32}},   {ORI_TOPK_LENGTH_NAME, {ge::DT_INT32}},
    {CMP_TOPK_LENGTH_NAME, {ge::DT_INT32}},   {SINKS_NAME, {ge::DT_FLOAT}},
    {METADATA_NAME, {ge::DT_INT32}},
};

const std::map<std::string, std::vector<QSMLALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {QSMLALayout::BSND, QSMLALayout::TND}},
    {ORI_KV_NAME, {QSMLALayout::BSND, QSMLALayout::PA_BBND, QSMLALayout::TND}},
    {CMP_KV_NAME, {QSMLALayout::BSND, QSMLALayout::PA_BBND, QSMLALayout::TND}},
    {ATTEN_OUT_NAME, {QSMLALayout::BSND, QSMLALayout::TND}},
};

template <typename T>
void QSMLATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                             const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus QSMLATilingCheck::CheckDimNumInLayoutSupport(const QSMLALayout &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto &dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
                        QSMLALayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
                        shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
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

ge::graphStatus QSMLATilingCheck::CheckShapeNumSupport(const gert::StorageShape *shape,
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

void QSMLATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                            const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << QSMLADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only support dtype %s, but got %s", name.c_str(), oss.str().c_str(),
            QSMLADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus QSMLATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
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
void QSMLATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
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

void QSMLATilingCheck::LogErrorLayoutSupport(const std::vector<QSMLALayout> &expectLayoutList,
                                             const QSMLALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << QSMLALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only support layout %s, but got %s", name.c_str(), oss.str().c_str(),
            QSMLALayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaQuery() const
{
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr, OP_LOGE(opName_, "Input q is required, but got nullptr."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.q.shape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(opName_, "Any dim of input q cannot be 0 "), return ge::GRAPH_FAILED);

    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr,
                OP_LOGE(opName_, "input oriKv can not be nullptr, but it's empty"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor->GetShapeSize() == 0, OP_LOGE(opName_, "Any dim of input oriKv cannot be 0 "),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), keyDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ == QSMLALayout::PA_BBND) {
        OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > 1024,
                    OP_LOGE(opName_, "when page attention is enabled, ori_block_size(%u) should be in range (0, %u].",
                            oriBlockSize_, MAX_BLOCK_SIZE),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(dSizeOriKvInput_ != 512, // 512: 当前不支持泛化
                OP_LOGE(opName_, "Dimension of OriKv only support 512, but got %u", dSizeOriKvInput_),
                return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cmpKv cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), keyDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(dSizeCmpKvInput_ != 512, // 512: 当前不支持泛化
                    OP_LOGE(opName_, "Dimension of CmpKv only support 512, but got %u", dSizeCmpKvInput_),
                    return ge::GRAPH_FAILED);

        uint32_t cmpKvN2Size_ = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), QSMLAAxis::N, kvLayout_);
        OP_CHECK_IF(cmpKvN2Size_ != n2Size_, OP_LOGE(opName_, "N2 size check failed! Expected cmpKvN2 == oriKvN2."),
                    return ge::GRAPH_FAILED);

        if (kvLayout_ == QSMLALayout::PA_BBND) {
            OP_CHECK_IF(
                cmpBlockSize_ <= 0 || cmpBlockSize_ > 1024,
                OP_LOGE(opName_, "when page attention is enabled, cmp_block_size(%u) should be in range (0, %u].",
                        cmpBlockSize_, MAX_BLOCK_SIZE),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckLayoutSupport(const QSMLALayout &actualLayout, const std::string &name) const
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

ge::graphStatus QSMLATilingCheck::CheckSingleParaNumHeads() const
{
    OP_CHECK_IF(n1Size_ < 1 || n1Size_ > 128,
                OP_LOGE(opName_, "n1Size_ only should be in range [1, 128], but got %u.", n1Size_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaKvHeadNums() const
{
    OP_CHECK_IF(n2Size_ != 1, OP_LOGE(opName_, "n2Size_ only support 1 now, but got %u.", n2Size_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF(oriMaskMode_ != 0 && oriMaskMode_ != 3 && oriMaskMode_ != 4,
                OP_LOGE(opName_, "oriMaskMode only support 0, 3 and 4, but got %u.", oriMaskMode_),
                return ge::GRAPH_FAILED);
    if (perfMode_ == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(cmpMaskMode_ != 0,
                    OP_LOGE(opName_, "cmpMaskMode only support 0 in SWA mode, but got %u.", cmpMaskMode_),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(cmpMaskMode_ != 0 && cmpMaskMode_ != 3,
                    OP_LOGE(opName_, "cmpMaskMode only support 0 and 3 in non-SWA mode, but got %u.", cmpMaskMode_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF(sparseBlockSize_ != 1,
                OP_LOGE(opName_, "sparseBlockSize_ only support 1, but got %u", sparseBlockSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaCmpResidualKv() const
{
    OP_CHECK_IF(perfMode_ != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                    perfMode_ != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE && cmpMaskMode_ != 0 &&
                    opParamInfo_.cmpResidualKv.tensor == nullptr,
                OP_LOGE(opName_, "cmp_residual_kv is required in CSA and HCA mode "), return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpResidualKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpResidualKv.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cmpResidualKv cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpResidualKv.desc, CMP_RESIDUAL_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpResidualKv.tensor->GetShape(), expectDimNumList,
                                                    CMP_RESIDUAL_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaCmpSparseIndices() const
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cmpSparseIndices cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES_NAME)) {
            return ge::GRAPH_FAILED;
        }

        const auto &cmpSparseShape = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(cmpSparseShape.GetDimNum() != DIM_NUM_FOUR,
                        OP_LOGE(opName_, "When q layout is BSND, cmpSparseIndices should be 4D, but got %zuD.",
                                cmpSparseShape.GetDimNum()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(cmpSparseShape.GetDimNum() != DIM_NUM_THREE,
                        OP_LOGE(opName_, "When q layout is TND, cmpSparseIndices should be 3D, but got %zuD.",
                                cmpSparseShape.GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaBlockTable() const
{
    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.oriBlockTable.tensor == nullptr,
                OP_LOGE(opName_, "oriBlockTable must not be empty when kvLayout is PA_BBND."), return ge::GRAPH_FAILED);

    const std::vector<size_t> BlockTableDimNumList = {DIM_NUM_TWO};
    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input oriBlockTable cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(),
                                                    BlockTableDimNumList, ORI_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cmpBlockTable cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(),
                                                    BlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaCuSeqLensQ() const
{
    if (qLayout_ == QSMLALayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((qLayout_ == QSMLALayout::TND && opParamInfo_.cuSeqLensQ.tensor == nullptr),
                OP_LOGE(opName_, "cuSeqLensQ can't be nullptr when layoutQ is TND"), return ge::GRAPH_FAILED);

    if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cuSeqLensQ cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensQ.desc, CU_SEQLENS_Q_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.cuSeqLensQ.tensor->GetShape(), expectDimNumList, CU_SEQLENS_Q_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSequsedKv() const
{
    if (perfMode_ == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
        perfMode_ == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.sequsedOriKv.tensor == nullptr,
                OP_LOGE(opName_, "input sequsedOriKv can not be nullptr in non-BSND layout, but it's empty"),
                return ge::GRAPH_FAILED);
    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedOriKv.desc, SEQUSED_ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.sequsedOriKv.tensor->GetShape(), expectDimNumList, SEQUSED_ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_,
                OP_LOGE(opName_, "input sequsedOriKv's shapeSize is not equal to B: %u, it is %ld", bSize_,
                        opParamInfo_.sequsedOriKv.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(kvLayout_ == QSMLALayout::PA_BBND && opParamInfo_.sequsedOriKv.tensor == nullptr,
                OP_LOGE(opName_, "sequsedOriKv must not be empty when kvLayout is PA_BBND."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSinks() const
{
    OP_CHECK_IF(opParamInfo_.sinks.tensor == nullptr,
                OP_LOGE(opName_, "Input sinks is nullptr, which is not supported"), return ge::GRAPH_FAILED);

    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sinks.desc, SINKS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.sinks.tensor->GetShape(), expectDimNumList, SINKS_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.sinks.tensor->GetShapeSize() != n1Size_,
                OP_LOGE(opName_, "Input sinks's shapeSize is not equal to n1: %u, it is %ld.", n1Size_,
                        opParamInfo_.sinks.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaMetadata() const
{
    OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
                OP_LOGE(opName_, "Input metadata is required, but got nullptr."), return ge::GRAPH_FAILED);

    const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.metadata.desc, METADATA_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.metadata.tensor->GetShape(), expectDimNumList, METADATA_NAME)) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_SIZE,
                OP_LOGE(opName_, "Input metadata's shapeSize should be %u, but got %ld.", METADATA_SIZE,
                        opParamInfo_.metadata.tensor->GetShapeSize()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaOriSparseIndices() const
{
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input oriSparseIndices cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriSparseIndices.desc, ORI_SPARSE_INDICES_NAME)) {
            return ge::GRAPH_FAILED;
        }

        const auto &oriSparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        if (qLayout_ == QSMLALayout::BSND) {
            OP_CHECK_IF(oriSparseShape.GetDimNum() != DIM_NUM_FOUR,
                        OP_LOGE(opName_, "When q layout is BSND, oriSparseIndices should be 4D, but got %zuD.",
                                oriSparseShape.GetDimNum()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(oriSparseShape.GetDimNum() != DIM_NUM_THREE,
                        OP_LOGE(opName_, "When q layout is TND, oriSparseIndices should be 3D, but got %zuD.",
                                oriSparseShape.GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaCuSeqLensOriKv() const
{
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && kvLayout_ == QSMLALayout::TND &&
                    opParamInfo_.cuSeqLensOriKv.tensor == nullptr,
                OP_LOGE(opName_, "cuSeqLensOriKv is required when oriKv is provided and kvLayout is TND."),
                return ge::GRAPH_FAILED);

    if (opParamInfo_.cuSeqLensOriKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cuSeqLensOriKv cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensOriKv.desc, CU_SEQLENS_ORI_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cuSeqLensOriKv.tensor->GetShape(), expectDimNumList,
                                                    CU_SEQLENS_ORI_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaCuSeqLensCmpKv() const
{
    if (opParamInfo_.cuSeqLensCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cuSeqLensCmpKv cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensCmpKv.desc, CU_SEQLENS_CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cuSeqLensCmpKv.tensor->GetShape(), expectDimNumList,
                                                    CU_SEQLENS_CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSequsedQ() const
{
    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input sequsedQ cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.seqUsedQ.desc, SEQUSED_Q_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.seqUsedQ.tensor->GetShape(), expectDimNumList, SEQUSED_Q_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaSequsedCmpKv() const
{
    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input sequsedCmpKv cannot be 0 "), return ge::GRAPH_FAILED);

        const std::vector<size_t> expectDimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedCmpKv.desc, SEQUSED_CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.sequsedCmpKv.tensor->GetShape(), expectDimNumList,
                                                    SEQUSED_CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_,
                    OP_LOGE(opName_, "sequsedCmpKv's shapeSize(%ld) should be equal to B(%u).",
                            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize(), bSize_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSingleParaOriTopkLength() const
{
    if (opParamInfo_.oriTopkLength.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.oriTopkLength.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input oriTopkLength cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriTopkLength.desc, ORI_TOPK_LENGTH_NAME)) {
            return ge::GRAPH_FAILED;
        }

        if (qLayout_ == QSMLALayout::TND) {
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

ge::graphStatus QSMLATilingCheck::CheckSingleParaCmpTopkLength() const
{
    if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpTopkLength.tensor->GetShapeSize() == 0,
                    OP_LOGE(opName_, "Any dim of input cmpTopkLength cannot be 0 "), return ge::GRAPH_FAILED);

        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpTopkLength.desc, CMP_TOPK_LENGTH_NAME)) {
            return ge::GRAPH_FAILED;
        }

        if (qLayout_ == QSMLALayout::TND) {
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

ge::graphStatus QSMLATilingCheck::CheckSingleParaTopkValueMode() const
{
    if (perfMode_ == QSMLATemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(topkValueMode_ != 1,
                    OP_LOGE(opName_, "topkValueMode must be 1 in SWA mode, but got %ld.", topkValueMode_),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(topkValueMode_ != 1 && topkValueMode_ != 2,
                    OP_LOGE(opName_, "topkValueMode only support 1 and 2, but got %ld.", topkValueMode_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckSinglePara() const
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
