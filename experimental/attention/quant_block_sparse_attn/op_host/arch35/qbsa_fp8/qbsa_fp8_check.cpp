/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <string>

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "qbsa_fp8_tiling.h"
#include "../../tiling_common/qbsa_check_utils.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"

namespace optiling {
using namespace qbsa_check;
namespace {
constexpr const char* kOpName = "QuantBlockSparseAttn";
constexpr uint32_t kFp8SparseBlockSize = 128U;
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_4 = 4U;
} // namespace

QbsaFp8Arch35Tiling::QbsaFp8Arch35Tiling(gert::TilingContext* context)
    : QbsaTilingStrategy(context)
{}

ge::graphStatus QbsaFp8Arch35Tiling::CheckAttributes() const
{
    if (CheckFp8Attrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFp8BlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLayouts() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckSoftmaxScale();
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckTensorMetadata() const
{
    if (CheckFp8Inputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckPaInputs() != ge::GRAPH_SUCCESS || CheckMetadata() != ge::GRAPH_SUCCESS ||
        CheckReservedInputs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckFp8ScaleDtype();
}

ge::graphStatus QbsaFp8Arch35Tiling::ParseShapeFacts(QbsaTilingContextParser& parser)
{
    if (parser.ParsePaShapeFacts(tilingInfo_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ResolveHeadGrouping();
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckShapeRelations() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckKeyValueShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckActualSeqLen() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNoneOrCausalMask() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFp8QuantShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckFp8PaStride();
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8Attrs() const
{
    if (tilingInfo_.layoutOutStr != "TND" || (tilingInfo_.layoutQStr != "TND" && tilingInfo_.layoutQStr != "NTD")) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "layout_out", tilingInfo_.layoutOutStr,
                                              "Must be TND in quant_mode=1 FP8 scenario");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.opParamInfo.maskMode != nullptr &&
        *tilingInfo_.opParamInfo.maskMode != static_cast<int64_t>(QBSA_MASK_MODE_NONE) &&
        *tilingInfo_.opParamInfo.maskMode != static_cast<int64_t>(QBSA_MASK_MODE_CAUSAL)) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "mask_mode", std::to_string(*tilingInfo_.opParamInfo.maskMode), "0 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8Inputs() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.key.stride == nullptr || opParamInfo.value.stride == nullptr ||
        opParamInfo.key.stride->GetDimNum() == 0U || opParamInfo.value.stride->GetDimNum() == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "key/value stride[0]", "nullptr",
            "The stride values of key and value must be provided when quant_mode is 1");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.key.stride->GetStride(0U) != opParamInfo.value.stride->GetStride(0U)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "key/value stride[0]",
            std::to_string(opParamInfo.key.stride->GetStride(0U)) + "/" +
                std::to_string(opParamInfo.value.stride->GetStride(0U)),
            "The first stride values of key and value must be equal when quant_mode is 1");
        return ge::GRAPH_FAILED;
    }
    const QBSAOptionalParaInfo* optionalInputs[] = {
        &opParamInfo.blockTable,
        &opParamInfo.cuSeqlensQ,
        &opParamInfo.seqUsedKV,
        &opParamInfo.metadata,
    };
    const char* optionalInputNames[] = {
        "block_table",
        "cu_seqlens_q",
        "seqused_kv",
        "metadata",
    };
    for (size_t i = 0U; i < sizeof(optionalInputs) / sizeof(optionalInputs[0]); ++i) {
        if (optionalInputs[i]->desc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, optionalInputNames[i], "nullptr",
                                                  "The input tensor must be provided when quant_mode is 1");
            return ge::GRAPH_FAILED;
        }
        if (optionalInputs[i]->shape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, optionalInputNames[i], "empty",
                                                  "The input tensor cannot be empty when quant_mode is 1");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8PaStride() const
{
    if (!tilingInfo_.hasViewStride) {
        return ge::GRAPH_SUCCESS;
    }
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    const auto& keyShape = opParamInfo.key.shape->GetStorageShape();
    const auto& valueShape = opParamInfo.value.shape->GetStorageShape();
    const auto& kDescaleShape = opParamInfo.kDescale.shape->GetStorageShape();
    const uint64_t keyBlockBytes =
        static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
        static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
        static_cast<uint64_t>(keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)));
    const uint64_t valueBlockBytes =
        static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
        static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
        static_cast<uint64_t>(valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)));
    const uint64_t kDescaleBlockBytes =
        static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N))) *
        static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) *
        static_cast<uint64_t>(kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM))) *
        sizeof(float);
    const uint64_t expectedPaBlockStride = keyBlockBytes + valueBlockBytes + kDescaleBlockBytes;
    if (tilingInfo_.paBlockStrideVal != expectedPaBlockStride) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "key stride[0]", std::to_string(tilingInfo_.paBlockStrideVal),
            "Must be equal to K/V/k_descale concatenated physical block size " + std::to_string(expectedPaBlockStride));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8ScaleDtype() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (CheckInputDtype(opParamInfo.qDescale, "q_descale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.kDescale, "k_descale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        CheckInputDtype(opParamInfo.vDescale, "v_descale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS ||
        (opParamInfo.pScale.shape != nullptr && opParamInfo.pScale.shape->GetStorageShape().GetShapeSize() > 0 &&
         CheckInputDtype(opParamInfo.pScale, "p_scale", ge::DT_FLOAT, "FLOAT") != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8BlockSize() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.qBlockSize == nullptr || opParamInfo.kvBlockSize == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "sparse_q_block_size/sparse_kv_block_size", "nullptr",
                                              "Both attributes are required in quant_mode=1");
        return ge::GRAPH_FAILED;
    }
    const int64_t qBlockSize = *opParamInfo.qBlockSize;
    const int64_t kvBlockSize = *opParamInfo.kvBlockSize;
    if (qBlockSize != static_cast<int64_t>(kFp8SparseBlockSize) ||
        kvBlockSize != static_cast<int64_t>(kFp8SparseBlockSize)) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "sparse_q_block_size/sparse_kv_block_size",
                                  std::to_string(qBlockSize) + "/" + std::to_string(kvBlockSize),
                                  std::to_string(kFp8SparseBlockSize));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFp8QuantShape() const
{
    if (tilingInfo_.maxBlockNumPerBatch < tilingInfo_.sparseCount) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "block_table.shape[1]", std::to_string(tilingInfo_.maxBlockNumPerBatch),
            "The second dimension of block_table must be greater than or equal to max_Kb of sparse_indices (" +
                std::to_string(tilingInfo_.sparseCount) + ") when quant_mode is 1");
        return ge::GRAPH_FAILED;
    }

    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.qDescale.shape != nullptr) {
        const gert::Shape& qDescaleShape = opParamInfo.qDescale.shape->GetStorageShape();
        if (qDescaleShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "q_descale", std::to_string(qDescaleShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        const QBSALayout qLayout = static_cast<QBSALayout>(tilingInfo_.layoutQValue);
        const int64_t expectedDim0 = static_cast<int64_t>(tilingInfo_.qTokenNum);
        const int64_t expectedDim1 = static_cast<int64_t>(tilingInfo_.n1Size);
        if (qDescaleShape.GetDim(QBSAGetAxisIdx(qLayout, QBSAAxis::T)) != expectedDim0 ||
            qDescaleShape.GetDim(QBSAGetAxisIdx(qLayout, QBSAAxis::N)) != expectedDim1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "q_descale", Ops::Base::ToString(qDescaleShape),
                                                  "shape does not match query layout");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.kDescale.shape != nullptr) {
        const gert::Shape& kDescaleShape = opParamInfo.kDescale.shape->GetStorageShape();
        if (kDescaleShape.GetDimNum() == DIM_NUM_4) {
            if (kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM)) !=
                    static_cast<int64_t>(tilingInfo_.paBlockNumSum) ||
                kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N)) !=
                    static_cast<int64_t>(tilingInfo_.n2Size) ||
                kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE)) !=
                    static_cast<int64_t>(tilingInfo_.kvBlockSizeVal) ||
                kDescaleShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)) != 1) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "k_descale", Ops::Base::ToString(kDescaleShape),
                                                      "must be (" + std::to_string(tilingInfo_.paBlockNumSum) + ", " +
                                                          std::to_string(tilingInfo_.n2Size) + ", " +
                                                          std::to_string(tilingInfo_.kvBlockSizeVal) + ", 1)");
                return ge::GRAPH_FAILED;
            }
            if (tilingInfo_.paBlockStrideVal % sizeof(float) != 0U) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    kOpName, "key stride[0]", std::to_string(tilingInfo_.paBlockStrideVal),
                    "Must be divisible by sizeof(float) to align k_descale block stride");
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(
                    opParamInfo.kDescale.stride, "k_descale", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM),
                    tilingInfo_.paBlockStrideVal / sizeof(float), tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale",
                               QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N), tilingInfo_.kvBlockSizeVal,
                               tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale",
                               QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE), 1U,
                               tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            if (CheckStrideDim(opParamInfo.kDescale.stride, "k_descale",
                               QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM), 1U,
                               tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "k_descale",
                                                     std::to_string(kDescaleShape.GetDimNum()) + "D",
                                                     "4D [blockNum, kvHeadNum, blockSize, 1]");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.vDescale.shape != nullptr) {
        const gert::Shape& vDescaleShape = opParamInfo.vDescale.shape->GetStorageShape();
        if (vDescaleShape.GetDimNum() != DIM_NUM_1 ||
            vDescaleShape.GetDim(0U) != static_cast<int64_t>(tilingInfo_.n2Size)) { // dim0 = n2Size
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "v_descale", Ops::Base::ToString(vDescaleShape),
                                                  "must be [n2Size]");
            return ge::GRAPH_FAILED;
        }
    }

    if (opParamInfo.pScale.shape != nullptr) {
        const gert::Shape& pScaleShape = opParamInfo.pScale.shape->GetStorageShape();
        if (pScaleShape.GetShapeSize() != 0 && (pScaleShape.GetDimNum() != DIM_NUM_1 || pScaleShape.GetDim(0U) != 1)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "p_scale", Ops::Base::ToString(pScaleShape),
                                                  "p_scale must be 1D [1] or empty in quant_mode=1");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
