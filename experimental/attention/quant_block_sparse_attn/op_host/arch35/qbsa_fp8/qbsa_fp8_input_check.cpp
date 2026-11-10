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
constexpr size_t DIM_NUM_1 = 1U;
constexpr size_t DIM_NUM_2 = 2U;
constexpr size_t DIM_NUM_3 = 3U;
constexpr size_t DIM_NUM_4 = 4U;
constexpr int64_t QBSA_ATTEN_MASK_DIM_VALUE = 2048;
// These limits belong to this architecture and quantization mode only.
constexpr uint32_t kMaxBatchSize = 65536U;
constexpr uint32_t kMaxQueryHeads = 128U;
constexpr uint32_t kMaxKvHeads = 8U;
constexpr uint32_t kMaxGroupSize = 16U;
constexpr uint32_t kHeadDim = 128U;
constexpr uint32_t kValueHeadDim = 128U;
} // namespace

ge::graphStatus QbsaFp8Arch35Tiling::ResolveHeadGrouping()
{
    if (tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.n1Size % tilingInfo_.n2Size != 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "n1Size (query head num)", std::to_string(tilingInfo_.n1Size),
            "Must be divisible by n2Size (kv head num) " + std::to_string(tilingInfo_.n2Size));
        return ge::GRAPH_FAILED;
    }

    tilingInfo_.gSize = tilingInfo_.n1Size / tilingInfo_.n2Size;
    tilingInfo_.isGqa = (tilingInfo_.gSize > 1U);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckLayouts() const
{
    constexpr const char* kvLayout = "PA_BNBD";
    constexpr const char* sparseLayout = "B_N_Qb_Kb";
    if (tilingInfo_.layoutKVStr != kvLayout || tilingInfo_.layoutSparseIndicesStr != sparseLayout) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "layout_kv/layout_sparse_indices",
            tilingInfo_.layoutKVStr + "/" + tilingInfo_.layoutSparseIndicesStr,
            std::string("This feature requires ") + kvLayout + " KV and " + sparseLayout + " sparse indices");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckSoftmaxScale() const
{
    constexpr float minExclusive = 0.0F;
    constexpr float maxInclusive = 1.0F;
    if (!(tilingInfo_.softmaxScaleVal > minExclusive && tilingInfo_.softmaxScaleVal <= maxInclusive)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "softmax_scale", std::to_string(tilingInfo_.softmaxScaleVal),
            "Must be in range (" + std::to_string(minExclusive) + ", " + std::to_string(maxInclusive) + "]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckDtype() const
{
    constexpr ge::DataType qkvDtype = ge::DT_FLOAT8_E4M3FN;
    constexpr ge::DataType outputDtype = ge::DT_BF16;
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.key.desc != nullptr && opParamInfo.value.desc != nullptr &&
        opParamInfo.key.desc->GetDataType() != opParamInfo.value.desc->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.key.desc->GetDataType()) + "/" +
                                      ge::TypeUtils::DataTypeToSerialString(opParamInfo.value.desc->GetDataType()),
                                  "key and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qDtype != qkvDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "query", ge::TypeUtils::DataTypeToSerialString(tilingInfo_.qDtype),
                                  ge::TypeUtils::DataTypeToSerialString(qkvDtype));
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.kvDtype != qkvDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "key/value", ge::TypeUtils::DataTypeToSerialString(tilingInfo_.kvDtype),
                                  ge::TypeUtils::DataTypeToSerialString(qkvDtype));
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.value.desc != nullptr && opParamInfo.value.desc->GetDataType() != qkvDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "value",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.value.desc->GetDataType()),
                                  ge::TypeUtils::DataTypeToSerialString(qkvDtype));
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseIndices.desc != nullptr && opParamInfo.sparseIndices.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_indices",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.sparseIndices.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.sparseSeqLen.desc != nullptr && opParamInfo.sparseSeqLen.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "sparse_seq_len",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.sparseSeqLen.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.attenMask.desc != nullptr && opParamInfo.attenMask.shape != nullptr &&
        opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() > 0 &&
        opParamInfo.attenMask.desc->GetDataType() != ge::DT_UINT8) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "atten_mask",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.attenMask.desc->GetDataType()),
                                  "UINT8");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.metadata.desc != nullptr && opParamInfo.metadata.shape != nullptr &&
        opParamInfo.metadata.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "metadata",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.metadata.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedKV.desc != nullptr && opParamInfo.seqUsedKV.shape != nullptr &&
        opParamInfo.seqUsedKV.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "seqused_kv",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.seqUsedKV.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.blockTable.desc != nullptr && opParamInfo.blockTable.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "block_table",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.blockTable.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensQ.desc != nullptr && opParamInfo.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "cu_seqlens_q",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.cuSeqlensQ.desc->GetDataType()),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.attnOut.desc != nullptr && opParamInfo.attnOut.desc->GetDataType() != outputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "attention_out",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.attnOut.desc->GetDataType()),
                                  ge::TypeUtils::DataTypeToSerialString(outputDtype));
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.lseOut.desc != nullptr && opParamInfo.lseOut.desc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "softmax_lse",
                                  ge::TypeUtils::DataTypeToSerialString(opParamInfo.lseOut.desc->GetDataType()),
                                  "FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckFormat() const
{
    constexpr ge::Format expectedFormat = ge::FORMAT_ND;
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    const gert::CompileTimeTensorDesc* requiredDescs[] = {
        opParamInfo.query.desc,         opParamInfo.key.desc,          opParamInfo.value.desc,
        opParamInfo.qDescale.desc,      opParamInfo.kDescale.desc,     opParamInfo.vDescale.desc,
        opParamInfo.sparseIndices.desc, opParamInfo.sparseSeqLen.desc, opParamInfo.attnOut.desc,
        opParamInfo.lseOut.desc,
    };
    const char* requiredNames[] = {"query",         "key",        "value",          "q_descale",
                                   "k_descale",     "v_descale",  "sparse_indices", "sparse_seq_len",
                                   "attention_out", "softmax_lse"};
    for (size_t i = 0; i < sizeof(requiredDescs) / sizeof(requiredDescs[0]); ++i) {
        if (requiredDescs[i] != nullptr && requiredDescs[i]->GetOriginFormat() != expectedFormat) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, requiredNames[i],
                                       Ops::Base::ToString(requiredDescs[i]->GetOriginFormat()).c_str(),
                                       Ops::Base::ToString(expectedFormat).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    const gert::CompileTimeTensorDesc* optionalDescs[] = {
        opParamInfo.attenMask.desc, opParamInfo.pScale.desc,     opParamInfo.blockTable.desc,
        opParamInfo.metadata.desc,  opParamInfo.cuSeqlensQ.desc, opParamInfo.seqUsedKV.desc,
    };
    const char* optionalNames[] = {"atten_mask", "p_scale", "block_table", "metadata", "cu_seqlens_q", "seqused_kv"};
    for (size_t i = 0; i < sizeof(optionalDescs) / sizeof(optionalDescs[0]); ++i) {
        if (optionalDescs[i] != nullptr && optionalDescs[i]->GetOriginFormat() != expectedFormat) {
            OP_LOGE_FOR_INVALID_FORMAT(kOpName, optionalNames[i],
                                       Ops::Base::ToString(optionalDescs[i]->GetOriginFormat()).c_str(),
                                       Ops::Base::ToString(expectedFormat).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckActualSeqLen() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.cuSeqlensQ.shape != nullptr) {
        const auto* cuSeqlensQShape = opParamInfo.cuSeqlensQ.shape;
        const gert::Shape& shape = cuSeqlensQShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(0U) != static_cast<int64_t>(tilingInfo_.bSize) + 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "cu_seqlens_q", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B+1");
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo.seqUsedKV.shape != nullptr) {
        const auto* seqUsedKVShape = opParamInfo.seqUsedKV.shape;
        const gert::Shape& shape = seqUsedKVShape->GetStorageShape();
        if (shape.GetDimNum() != DIM_NUM_1 || shape.GetDim(0U) != static_cast<int64_t>(tilingInfo_.bSize)) { // dim0 = B
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "seqused_kv", Ops::Base::ToString(shape),
                                                  "must be 1D with dim[0] == B");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckMetadata() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.metadata.desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "metadata", "nullptr",
                                              "The metadata tensor must be provided for this feature");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.metadata.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "metadata", "empty",
                                              "The metadata tensor cannot be empty for this feature");
        return ge::GRAPH_FAILED;
    }

    const auto* metadataShape = opParamInfo.metadata.shape;
    const gert::Shape& metadataStorageShape = metadataShape->GetStorageShape();
    if (metadataStorageShape.GetDimNum() != DIM_NUM_1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "metadata", Ops::Base::ToString(metadataStorageShape),
                                              "The metadata tensor must be one-dimensional for this feature");
        return ge::GRAPH_FAILED;
    }
    if (metadataStorageShape.GetDim(0U) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "metadata", Ops::Base::ToString(metadataStorageShape),
                                              "The metadata tensor cannot be empty for this feature");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckPaInputs() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.blockTable.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "block_table", "nullptr",
                                              "Block_table is required for PA execution path");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.cuSeqlensQ.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "cu_seqlens_q", "nullptr",
                                              "Cu_seqlens_q is required for TND/NTD query layout "
                                              "with PA BNBD KV-cache");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedKV.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "seqused_kv", "nullptr",
                                              "Seqused_kv is required for TND/NTD query layout "
                                              "with PA BNBD KV-cache");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckReservedInputs() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.cuSeqlensKV.shape != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "cu_seqlens_kv", "non-null",
                                              "Cu_seqlens_kv is a reserved parameter and must be passed as null; "
                                              "non-null input is rejected");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo.seqUsedQ.shape != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "seqused_q", "non-null",
                                              "Seqused_q is a reserved parameter and must be passed as null; "
                                              "non-null input is rejected");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckShapeConsistency() const
{
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    const gert::Shape& sparseIndicesShape = opParamInfo.sparseIndices.shape->GetStorageShape();
    const gert::Shape& sparseSeqLenShape = opParamInfo.sparseSeqLen.shape->GetStorageShape();
    if (sparseIndicesShape.GetDimNum() != DIM_NUM_4 || sparseSeqLenShape.GetDimNum() != DIM_NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(kOpName, "sparse_indices, sparse_seq_len",
                                                  std::to_string(sparseIndicesShape.GetDimNum()) + "D, " +
                                                      std::to_string(sparseSeqLenShape.GetDimNum()) + "D",
                                                  "sparse_indices must be 4D and sparse_seq_len must be 3D");
        return ge::GRAPH_FAILED;
    }

    if (sparseIndicesShape.GetDim(0U) != static_cast<int64_t>(tilingInfo_.bSize) ||
        sparseIndicesShape.GetDim(1U) != static_cast<int64_t>(tilingInfo_.n1Size)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "sparse_indices", Ops::Base::ToString(sparseIndicesShape),
                                              "Dim[0] must equal batch size and dim[1] must equal "
                                              "query N1 for this feature");
        return ge::GRAPH_FAILED;
    }
    if (sparseSeqLenShape.GetDim(0U) != static_cast<int64_t>(tilingInfo_.bSize) ||
        sparseSeqLenShape.GetDim(1U) != static_cast<int64_t>(tilingInfo_.n1Size) ||
        sparseSeqLenShape.GetDim(2U) != static_cast<int64_t>(tilingInfo_.qbMax)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "sparse_seq_len", Ops::Base::ToString(sparseSeqLenShape),
            "The shape of sparse_seq_len must be [sparse_indices.B, query.N1, sparse_indices.max_Qb] for this feature");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.bSize == 0U || tilingInfo_.n1Size == 0U || tilingInfo_.n2Size == 0U || tilingInfo_.gSize == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "bSize/n1Size/n2Size/gSize",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize),
            "All dimensions must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (tilingInfo_.qBlockSizeVal == 0U || tilingInfo_.kvBlockSizeVal == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "q_block_size/kv_block_size",
            std::to_string(tilingInfo_.qBlockSizeVal) + "/" + std::to_string(tilingInfo_.kvBlockSizeVal),
            "Block sizes must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (tilingInfo_.qbMax == 0U || tilingInfo_.sparseCount == 0U || tilingInfo_.maxBlockNumPerBatch == 0U) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "qbMax/sparseCount/maxBlockNumPerBatch",
                                              std::to_string(tilingInfo_.qbMax) + "/" +
                                                  std::to_string(tilingInfo_.sparseCount) + "/" +
                                                  std::to_string(tilingInfo_.maxBlockNumPerBatch),
                                              "Block counts and sparse count must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckShapeLimits() const
{
    if (tilingInfo_.bSize > kMaxBatchSize || tilingInfo_.n1Size > kMaxQueryHeads || tilingInfo_.n2Size > kMaxKvHeads ||
        tilingInfo_.gSize > kMaxGroupSize || tilingInfo_.dSize != kHeadDim || tilingInfo_.dSizeV != kValueHeadDim) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            kOpName, "B/N1/N2/G/D/DV",
            std::to_string(tilingInfo_.bSize) + "/" + std::to_string(tilingInfo_.n1Size) + "/" +
                std::to_string(tilingInfo_.n2Size) + "/" + std::to_string(tilingInfo_.gSize) + "/" +
                std::to_string(tilingInfo_.dSize) + "/" + std::to_string(tilingInfo_.dSizeV),
            "Requires B<=" + std::to_string(kMaxBatchSize) + ", N1<=" + std::to_string(kMaxQueryHeads) +
                ", N2<=" + std::to_string(kMaxKvHeads) + ", G<=" + std::to_string(kMaxGroupSize) +
                ", D=" + std::to_string(kHeadDim) + ", DV=" + std::to_string(kValueHeadDim));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckKeyValueShape() const
{
    const uint32_t expectedBlockSize = tilingInfo_.kvBlockSizeVal;
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (opParamInfo.key.shape == nullptr || opParamInfo.value.shape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "key/value", "nullptr", "shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& keyShape = opParamInfo.key.shape->GetStorageShape();
    const gert::Shape& valueShape = opParamInfo.value.shape->GetStorageShape();

    if (keyShape.GetDimNum() != DIM_NUM_4 || valueShape.GetDimNum() != DIM_NUM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            kOpName, "key/value",
            std::to_string(keyShape.GetDimNum()) + "D, " + std::to_string(valueShape.GetDimNum()) + "D",
            "must both be 4D PA BNBD [blockNum, kvHeadNum, blockSize, headDim]");
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM)) !=
            keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM)) ||
        valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N)) !=
            keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N)) ||
        valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE)) !=
            keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE))) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key/value",
                                              Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
                                              "value dim[0..2] must match key dim[0..2] for PA BNBD tensors");
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(2U) != static_cast<int64_t>(expectedBlockSize) ||
        valueShape.GetDim(2U) != static_cast<int64_t>(expectedBlockSize)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            kOpName, "key/value", Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
            "dim[2] must equal the feature's PA block size " + std::to_string(expectedBlockSize));
        return ge::GRAPH_FAILED;
    }
    if (keyShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)) !=
            static_cast<int64_t>(tilingInfo_.dSize) ||
        valueShape.GetDim(QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM)) !=
            static_cast<int64_t>(tilingInfo_.dSizeV)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "key/value",
                                              Ops::Base::ToString(keyShape) + "/" + Ops::Base::ToString(valueShape),
                                              "key dim[3] must match dSize and value dim[3] must match dSizeV");
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM),
                       tilingInfo_.paBlockStrideVal, tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N),
                       tilingInfo_.paBlockSizeVal * tilingInfo_.dSize,
                       tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE),
                       tilingInfo_.dSize, tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.key.stride, "key", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM), 1U,
                       tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_NUM),
                       tilingInfo_.paBlockStrideVal, tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::N),
                       tilingInfo_.paBlockSizeVal * tilingInfo_.dSizeV,
                       tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::BLOCK_SIZE),
                       tilingInfo_.dSizeV, tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckStrideDim(opParamInfo.value.stride, "value", QBSAGetAxisIdx(QBSALayout::PA_BNBD, QBSAAxis::HEAD_DIM), 1U,
                       tilingInfo_.hasViewStride) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch35Tiling::CheckNoneOrCausalMask() const
{
    if (tilingInfo_.maskModeVal != QBSA_MASK_MODE_NONE && tilingInfo_.maskModeVal != QBSA_MASK_MODE_CAUSAL) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "mask_mode", std::to_string(tilingInfo_.maskModeVal), "0 or 3");
        return ge::GRAPH_FAILED;
    }
    const auto& opParamInfo = tilingInfo_.opParamInfo;
    if (tilingInfo_.maskModeVal == QBSA_MASK_MODE_NONE) {
        if (opParamInfo.attenMask.shape != nullptr &&
            opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() > 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "atten_mask", "provided",
                                                  "Atten_mask must not be provided when mask_mode=0");
            return ge::GRAPH_FAILED;
        }
    } else if (tilingInfo_.maskModeVal == QBSA_MASK_MODE_CAUSAL) {
        if (opParamInfo.attenMask.shape == nullptr ||
            opParamInfo.attenMask.shape->GetStorageShape().GetShapeSize() <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "atten_mask", "nullptr",
                                                  "Atten_mask is required when mask_mode=3");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& attenMaskShape = opParamInfo.attenMask.shape->GetStorageShape();
        if (attenMaskShape.GetDimNum() != DIM_NUM_2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "atten_mask", std::to_string(attenMaskShape.GetDimNum()) + "D", "2D");
            return ge::GRAPH_FAILED;
        }
        if (attenMaskShape.GetDim(0U) != QBSA_ATTEN_MASK_DIM_VALUE || // dim0 = rows
            attenMaskShape.GetDim(1U) != QBSA_ATTEN_MASK_DIM_VALUE) { // dim1 = cols
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(kOpName, "atten_mask", Ops::Base::ToString(attenMaskShape),
                                                  "must be (2048, 2048)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
