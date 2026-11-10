/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/arch35/gbsa_mxfp8_constants.h"
#include "generic_block_sparse_attention_tiling.h"
#include "../op_kernel/arch35/gbsa_mxfp8_scale_cache_layout.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SPARSE_BLOCK_IDX_INDEX = 3;
constexpr int SPARSE_BLOCK_COUNT_INDEX = 4;
constexpr int METADATA_INDEX = 5;
constexpr int ATTEN_MASK_INDEX = 6;
constexpr int Q_DEQUANT_SCALE_INDEX = 7;
constexpr int K_DEQUANT_SCALE_INDEX = 8;
constexpr int V_DEQUANT_SCALE_INDEX = 9;
constexpr int P_QUANT_SCALE_INDEX = 10;
constexpr int CU_SEQ_LENGTHS_Q_INDEX = 11;
constexpr int CU_SEQ_LENGTHS_KV_INDEX = 12;
constexpr int SEQUSED_Q_INDEX = 13;
constexpr int SEQUSED_KV_INDEX = 14;
constexpr int BLOCK_TABLE_INDEX = 15;

// Must match METADATA_TOTAL_SIZE in the AICPU / kernel metadata protocol.
constexpr uint32_t GBSA_METADATA_TOTAL_SIZE = 1024U;

constexpr int ATTENTION_OUT_INDEX = 0;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

// layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
constexpr int SPARSE_IDX_DIM_KV_HEAD = 0;
constexpr int SPARSE_IDX_DIM_Q_BLOCK = 1;
constexpr int SPARSE_IDX_DIM_KV_BLOCK = 2;
constexpr int SPARSE_IDX_DIM_NUM = 3;

// layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockCount 2D [N_kv, totalQBlocks]
constexpr int SPARSE_COUNT_DIM_KV_HEAD = 0;
constexpr int SPARSE_COUNT_DIM_Q_BLOCK = 1;
constexpr int SPARSE_COUNT_DIM_NUM = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;
constexpr int BLOCK_TABLE_DIM_NUM = 2;

constexpr int ATTR_BLOCK_SHAPE_INDEX = 0;
constexpr int ATTR_Q_INPUT_LAYOUT_INDEX = 1;
constexpr int ATTR_KV_INPUT_LAYOUT_INDEX = 2;
constexpr int ATTR_LAYOUT_SPARSE_PATTERN_INDEX = 3;
constexpr int ATTR_SCALE_VALUE_INDEX = 4;
constexpr int ATTR_MASK_TYPE_INDEX = 5;
constexpr int ATTR_QUANT_TYPE_INDEX = 6;
constexpr int ATTR_DST_TYPE_MAX_INDEX = 7;
constexpr int ATTR_SOFTMAX_PRECISION_INDEX = 8;
constexpr int ATTR_WIN_LEFT_INDEX = 9;
constexpr int ATTR_WIN_RIGHT_INDEX = 10;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 11;
constexpr int ATTR_RESIDUAL_BLOCK_MODE_INDEX = 12;
constexpr int ATTR_IS_CONSISTENT_TOPK_INDEX = 13;

constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_NONE = 0;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_FORCE = 1;
constexpr uint64_t GBSA_RESIDUAL_BLOCK_MODE_TILING_OFFSET = 1000000000ULL;
constexpr int64_t GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB = 4;

constexpr uint32_t SOC_VER_950_CODE = 4;
constexpr uint32_t GBSA_MAX_GROUP_SIZE = 128U;
constexpr uint32_t GBSA_TILE_SIZE_16 = 16U;
constexpr uint32_t GBSA_TILE_SIZE_32 = 32U;
constexpr uint32_t GBSA_TILE_SIZE_64 = 64U;
constexpr uint32_t GBSA_TILE_SIZE_128 = 128U;
constexpr uint32_t GBSA_TILE_SIZE_256 = 256U;
constexpr uint32_t GBSA_TILE_SIZE_384 = 384U;
constexpr uint32_t GBSA_TILE_SIZE_512 = 512U;
constexpr int64_t GBSA_QUANT_TYPE_NONE = 0;
constexpr int64_t GBSA_QUANT_TYPE_FULL = 5;
constexpr int64_t GBSA_QUANT_TYPE_MXFP8 = 6;
constexpr uint64_t GBSA_MXFP8_TILING_OFFSET = 2000000000ULL;
constexpr uint64_t GBSA_MXFP8_SCALE_PIPELINE_DISPATCH_BASE = 9250002900000020ULL;
constexpr uint64_t GBSA_MXFP8_COALESCED_PIPELINE_DISPATCH_BASE = 9250002900000022ULL;
constexpr uint64_t GBSA_MXFP8_BNSD_PIPELINE_DISPATCH_BASE = 9250002900000032ULL;
constexpr uint64_t GBSA_MXFP8_BNSD_COALESCED_PIPELINE_DISPATCH_BASE = 9250002900000034ULL;
constexpr uint64_t GBSA_MXFP8_BSND_PA_BBND_DISPATCH_BASE = 9250002900000040ULL;
constexpr uint64_t GBSA_MXFP8_BSND_PA_BNBD_DISPATCH_BASE = 9250002900000042ULL;
constexpr uint64_t GBSA_MXFP8_BNSD_PA_BBND_DISPATCH_BASE = 9250002900000044ULL;
constexpr uint64_t GBSA_MXFP8_BNSD_PA_BNBD_DISPATCH_BASE = 9250002900000046ULL;
constexpr uint64_t GBSA_MXFP8_TND_PIPELINE_DISPATCH_BASE = 9250002900000050ULL;
constexpr uint64_t GBSA_MXFP8_TND_PA_BBND_DISPATCH_BASE = 9250002900000052ULL;
constexpr uint64_t GBSA_MXFP8_TND_PA_BNBD_DISPATCH_BASE = 9250002900000054ULL;
constexpr uint32_t GBSA_MXFP8_COALESCED_PREFILL_MIN_TOKENS = 1024U;
constexpr uint32_t GBSA_MXFP8_COALESCED_PREFILL_MAX_TOKENS = 8192U;
constexpr int64_t GBSA_WIN_DISABLED = -1;
constexpr float GBSA_DST_TYPE_MAX_DISABLED = 0.0f;
constexpr uint32_t GSA_FD_MAX_ACTIVE_CORE_NUM = 32U;        // Maximum AIC cores used by FD.
constexpr uint32_t GSA_FD_MAX_COMBINE_TASK_NUM = 32U;       // Maximum FD combine tasks.
constexpr uint32_t GSA_FD_BASE_TASK_GATE_NUMERATOR = 3U;    // Numerator of the FD base-task gate ratio.
constexpr uint32_t GSA_FD_BASE_TASK_GATE_DENOMINATOR = 10U; // Denominator of the FD base-task gate ratio.
constexpr uint64_t GSA_FD_WORKSPACE_ALIGNMENT = 512U;       // FD workspace alignment in bytes.

namespace {
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
} // namespace

namespace optiling {

ge::graphStatus GBSATiling::GetNpuInfo(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    if (aicNum_ == 0) {
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "GetCoreNumAic returned 0, which is derived from "
                                                            "platform info and should be positive.");
        return ge::GRAPH_FAILED;
    }
    // Task schedule is owned by AICPU metadata (saTotalTaskNum). Host only launches
    // all AIC cores; idle cores exit when taskIdx >= metadata saTotalTaskNum.
    blockDim_ = aicNum_;
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    socVer_ = static_cast<uint32_t>(ascendcPlatform.GetSocVersion());
    return ge::GRAPH_SUCCESS;
}

namespace {

ge::graphStatus ParseBlockShapeAttr(gert::TilingContext* context,
                                    const gert::TypedContinuousVector<int64_t>* blockShapeArr, uint32_t& blockShapeX,
                                    uint32_t& blockShapeY)
{
    if (blockShapeArr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (blockShapeArr->GetSize() != 2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockShape",
                                              ("size=" + std::to_string(blockShapeArr->GetSize())).c_str(),
                                              "blockShape must contain two elements [x, y]");
        return ge::GRAPH_FAILED;
    }
    const int64_t* data = blockShapeArr->GetData();
    if (data == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockShape", "nullptr",
                                              "the data of blockShape must not be null");
        return ge::GRAPH_FAILED;
    }
    if (data[0] != 1 || data[1] <= 0 || data[1] > 128) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "blockShape",
            ("elem0=" + std::to_string(data[0]) + ", elem1=" + std::to_string(data[1])).c_str(),
            "blockShape requires X=1 and 1<=Y<=128");
        return ge::GRAPH_FAILED;
    }
    blockShapeX = static_cast<uint32_t>(data[0]);
    blockShapeY = static_cast<uint32_t>(data[1]);
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus GBSATiling::ParseCapabilityAttrs(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Attrs got from ge context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const int64_t* softmaxPrecPtr = attrs->GetInt(ATTR_SOFTMAX_PRECISION_INDEX);
    if (softmaxPrecPtr != nullptr) {
        softmaxPrecision_ = static_cast<uint32_t>(*softmaxPrecPtr);
    }
    if (softmaxPrecision_ != 0 && softmaxPrecision_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "softmaxPrecision",
                                              std::to_string(softmaxPrecision_).c_str(),
                                              "only 0 (fp32 SM) or 1 (half/low SM) are supported");
        return ge::GRAPH_FAILED;
    }

    const int64_t* maskTypePtr = attrs->GetInt(ATTR_MASK_TYPE_INDEX);
    if (maskTypePtr != nullptr) {
        maskType_ = *maskTypePtr;
    }
    if (maskType_ != 0 && maskType_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "maskType", std::to_string(maskType_).c_str(),
                                              "only maskType=0 (non-causal) or 1 (causal) is supported");
        return ge::GRAPH_FAILED;
    }

    const int64_t* quantTypePtr = attrs->GetInt(ATTR_QUANT_TYPE_INDEX);
    if (quantTypePtr != nullptr) {
        quantType_ = *quantTypePtr;
    }
    if (quantType_ != GBSA_QUANT_TYPE_NONE && quantType_ != GBSA_QUANT_TYPE_FULL &&
        quantType_ != GBSA_QUANT_TYPE_MXFP8) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "quantType", std::to_string(quantType_).c_str(),
                                              "only 0 (none), 5 (FP8) or 6 (MXFP8) are supported");
        return ge::GRAPH_FAILED;
    }

    const int64_t* layoutSparsePtr = attrs->GetInt(ATTR_LAYOUT_SPARSE_PATTERN_INDEX);
    if (layoutSparsePtr != nullptr) {
        layoutSparsePattern_ = *layoutSparsePtr;
    }
    if (layoutSparsePattern_ != GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "layout_sparse_pattern",
                                              std::to_string(layoutSparsePattern_).c_str(),
                                              "only 4 (KVN_TotalQB_KB) is supported");
        return ge::GRAPH_FAILED;
    }

    const int64_t* residualModePtr = attrs->GetInt(ATTR_RESIDUAL_BLOCK_MODE_INDEX);
    if (residualModePtr != nullptr) {
        residualBlockMode_ = *residualModePtr;
    }
    const bool residualModeSupported =
        residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_NONE ||
        (residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE && quantType_ == GBSA_QUANT_TYPE_NONE);
    if (!residualModeSupported) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "residual_block_mode",
                                              std::to_string(residualBlockMode_).c_str(),
                                              "mode 1 only supports non-quant currently");
        return ge::GRAPH_FAILED;
    }

    const bool* consistentTopkPtr = attrs->GetBool(ATTR_IS_CONSISTENT_TOPK_INDEX);
    if (consistentTopkPtr != nullptr) {
        isConsistentTopk_ = *consistentTopkPtr;
    }

    const int64_t* lseFlagPtr = attrs->GetInt(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlagPtr != nullptr) {
        softmaxLseFlag_ = *lseFlagPtr;
    }
    if (softmaxLseFlag_ != 0 && softmaxLseFlag_ != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "return_softmax_lse",
                                              std::to_string(softmaxLseFlag_).c_str(), "only 0 or 1 is supported");
        return ge::GRAPH_FAILED;
    }
    returnSoftmaxlse_ = (softmaxLseFlag_ == 1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckReservedAttrs(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Attrs got from ge context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float* dstTypeMaxPtr = attrs->GetFloat(ATTR_DST_TYPE_MAX_INDEX);
    if (dstTypeMaxPtr != nullptr && *dstTypeMaxPtr != GBSA_DST_TYPE_MAX_DISABLED) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dst_type_max",
                                              std::to_string(*dstTypeMaxPtr).c_str(),
                                              "dst_type_max is not supported yet and must be 0");
        return ge::GRAPH_FAILED;
    }
    const int64_t* winLeftPtr = attrs->GetInt(ATTR_WIN_LEFT_INDEX);
    const int64_t* winRightPtr = attrs->GetInt(ATTR_WIN_RIGHT_INDEX);
    const int64_t winLeft = (winLeftPtr != nullptr) ? *winLeftPtr : GBSA_WIN_DISABLED;
    const int64_t winRight = (winRightPtr != nullptr) ? *winRightPtr : GBSA_WIN_DISABLED;
    if (winLeft != GBSA_WIN_DISABLED || winRight != GBSA_WIN_DISABLED) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "win_left and win_right",
            ("win_left=" + std::to_string(winLeft) + ", win_right=" + std::to_string(winRight)).c_str(),
            "windowed atten mask is not supported yet, win_left and win_right must be -1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseAttrs(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Attrs got from ge context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float* scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    if (ParseBlockShapeAttr(context, attrs->GetListInt(ATTR_BLOCK_SHAPE_INDEX), blockShapeX_, blockShapeY_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParseCapabilityAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckReservedAttrs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::GetInputLayout(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Attrs got from ge context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const char* layoutQPtr = attrs->GetStr(ATTR_Q_INPUT_LAYOUT_INDEX);
    if (layoutQPtr != nullptr) {
        layoutQ_ = std::string(layoutQPtr);
    }
    const char* layoutKvPtr = attrs->GetStr(ATTR_KV_INPUT_LAYOUT_INDEX);
    if (layoutKvPtr != nullptr) {
        layoutKv_ = std::string(layoutKvPtr);
    }

    if (layoutQ_ != "TND" && layoutQ_ != "BSND" && layoutQ_ != "BNSD") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "q_input_layout", layoutQ_.c_str(),
                                              "q_input_layout only supports TND/BSND/BNSD");
        return ge::GRAPH_FAILED;
    }
    if (layoutKv_ != "PA_BBND" && layoutKv_ != "PA_BNBD" && layoutKv_ != layoutQ_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "kv_input_layout", layoutKv_.c_str(),
                                              "dense Q/K/V must use the same layout; paged KV must be PA_BBND/PA_BNBD");
        return ge::GRAPH_FAILED;
    }
    isPaBnbd_ = (layoutKv_ == "PA_BNBD");
    // Determine paging before parsing shapes, strides or sequence-input requirements.
    isPaged_ = context->GetOptionalInputShape(BLOCK_TABLE_INDEX) != nullptr;
    const bool layoutIsPa = (layoutKv_ == "PA_BBND" || isPaBnbd_);
    if ((isPaged_ && !layoutIsPa) || (!isPaged_ && layoutKv_ != layoutQ_)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "q_input_layout and kv_input_layout",
                                               ("layout_q=" + layoutQ_ + ", layout_kv=" + layoutKv_).c_str(),
                                               "paged KV requires PA_BBND/PA_BNBD layout; dense Q/K/V must use "
                                               "the same layout");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckAttentionOutDtype(gert::TilingContext* context)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        auto outDesc = context->GetOutputDesc(ATTENTION_OUT_INDEX);
        if (outDesc == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attentionOut", "nullptr",
                                                  "the output desc of attentionOut must be provided");
            return ge::GRAPH_FAILED;
        }
        attentionOutDtype_ = outDesc->GetDataType();
        if (attentionOutDtype_ != ge::DT_FLOAT16 && attentionOutDtype_ != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "attentionOut",
                std::to_string(static_cast<int32_t>(attentionOutDtype_)).c_str(),
                "the supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is all "
                "float8_e4m3fn");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// Validate paged key/value: only dim0 may be non-contiguous. Shape-driven, so it
// covers PA_BBND and PA_BNBD alike.
static ge::graphStatus ValidatePagedBbndDim0OnlyNonContig(gert::TilingContext* context, uint64_t inputIndex,
                                                          const gert::Shape& shape, const char* tensorName)
{
    auto* stride = context->GetRequiredInputStride(inputIndex);
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t expectedStride = 1;
    for (size_t i = shape.GetDimNum() - 1; i >= 1; --i) {
        const int64_t dimSize = shape.GetDim(i);
        // size<=1 axes are not used in addressing; PyTorch often leaves non-canonical strides
        // (e.g. Nkv=1). Skip so dim0-strided KV is not falsely rejected.
        if (dimSize <= 1) {
            continue;
        }
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(i));
        if (actualStride != expectedStride) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), tensorName,
                ("dim" + std::to_string(i) + " stride=" + std::to_string(actualStride) +
                 ", expected=" + std::to_string(expectedStride))
                    .c_str(),
                "only the first axis (dim0) may be non-contiguous for paged KV");
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(dimSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseKvCacheStride0(gert::TilingContext* context)
{
    if (!isPaged_) {
        for (const int index : {QUERY_INDEX, KEY_INDEX, VALUE_INDEX}) {
            const auto& shape = context->GetInputShape(index)->GetStorageShape();
            const size_t rank = layoutQ_ == "TND" ? 3U : 4U;
            if (shape.GetDimNum() != rank) {
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "query, key and value",
                    ("inputIndex=" + std::to_string(index) + ", storageRank=" + std::to_string(shape.GetDimNum()) +
                     ", expectedRank=" + std::to_string(rank))
                        .c_str(),
                    "dense Q/K/V storage shape has an invalid rank");
                return ge::GRAPH_FAILED;
            }
            const auto* stride = context->GetRequiredInputStride(index);
            if (stride == nullptr || stride->GetDimNum() == 0) {
                continue; // No explicit stride: infer a contiguous layout from storage shape.
            }
            if (stride->GetDimNum() != shape.GetDimNum()) {
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "query, key and value",
                    ("inputIndex=" + std::to_string(index) + ", storageRank=" + std::to_string(shape.GetDimNum()) +
                     ", strideRank=" + std::to_string(stride->GetDimNum()))
                        .c_str(),
                    "dense Q/K/V explicit stride rank must equal the storage shape rank");
                return ge::GRAPH_FAILED;
            }
            int64_t expected = 1;
            for (int axis = static_cast<int>(rank) - 1; axis >= 0; --axis) {
                if (shape.GetDim(axis) > 1 && stride->GetStride(axis) != expected) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), "query, key and value",
                        ("inputIndex=" + std::to_string(index) + ", axis=" + std::to_string(axis) + ", stride=" +
                         std::to_string(stride->GetStride(axis)) + ", expected=" + std::to_string(expected))
                            .c_str(),
                        "dense Q/K/V must be contiguous");
                    return ge::GRAPH_FAILED;
                }
                expected *= shape.GetDim(axis);
            }
        }
        kStride0_ = 0;
        vStride0_ = 0;
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t pageElems =
        static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);

    const gert::StorageShape* keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape* valueShape = context->GetInputShape(VALUE_INDEX);
    if (keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "key and value", "shape is nullptr",
                                               "the shapes of key and value must be provided when parsing KV stride0");
        return ge::GRAPH_FAILED;
    }

    if (ValidatePagedBbndDim0OnlyNonContig(context, KEY_INDEX, keyShape->GetOriginShape(), "key") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePagedBbndDim0OnlyNonContig(context, VALUE_INDEX, valueShape->GetOriginShape(), "value") !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto* keyStrides = context->GetRequiredInputStride(KEY_INDEX);
    kStride0_ = (keyStrides != nullptr && keyStrides->GetDimNum() > 0 && keyStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(keyStrides->GetStride(0)) :
                    pageElems;
    auto* valueStrides = context->GetRequiredInputStride(VALUE_INDEX);
    vStride0_ = (valueStrides != nullptr && valueStrides->GetDimNum() > 0 && valueStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(valueStrides->GetStride(0)) :
                    pageElems;

    // Packed MXFP8 shares physical offsets between K/V and KScale. ACLNN
    // makes these caches contiguous; direct GE callers must provide that layout.
    if (quantType_ == GBSA_QUANT_TYPE_MXFP8 && (kStride0_ != pageElems || vStride0_ != pageElems)) {
        OP_LOGE(context->GetNodeName(), "MXFP8 paged K/V require contiguous page storage.");
        return ge::GRAPH_FAILED;
    }

    const uint64_t rowElems = static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);
    if (kStride0_ < pageElems || (rowElems > 0 && (kStride0_ % rowElems) != 0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "key",
            ("dim0 stride=" + std::to_string(kStride0_) + ", pageElems=" + std::to_string(pageElems) +
             ", Nkv*D=" + std::to_string(rowElems))
                .c_str(),
            "the dim0 stride of key is invalid for paged KV: it must be >= pageElems and aligned to Nkv*D");
        return ge::GRAPH_FAILED;
    }
    if (vStride0_ < pageElems || (rowElems > 0 && (vStride0_ % rowElems) != 0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "value",
            ("dim0 stride=" + std::to_string(vStride0_) + ", pageElems=" + std::to_string(pageElems) +
             ", Nkv*D=" + std::to_string(rowElems))
                .c_str(),
            "the dim0 stride of value is invalid for paged KV: it must be >= pageElems and aligned to Nkv*D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQueryKeyShapes(gert::TilingContext* context)
{
    const auto* queryShape = context->GetInputShape(QUERY_INDEX);
    const auto* keyShape = context->GetInputShape(KEY_INDEX);
    const auto* valueShape = context->GetInputShape(VALUE_INDEX);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "query, key and value", "shape is nullptr",
                                               "the shapes of query/key/value must be provided");
        return ge::GRAPH_FAILED;
    }
    const auto& q = queryShape->GetStorageShape();
    const auto& k = keyShape->GetOriginShape();
    const auto& v = valueShape->GetOriginShape();
    const bool isPaged = isPaged_;
    const bool isBatched = layoutQ_ != "TND";
    const size_t qRank = isBatched ? 4U : 3U;
    const size_t qHeadAxis = layoutQ_ == "BSND" ? 2U : 1U;
    const size_t qSeqAxis = layoutQ_ == "BNSD" ? 2U : 1U;
    const size_t kvRank = isPaged ? 4U : qRank;
    // PA_BBND [blockNum, blockSize, Nkv, D]; PA_BNBD swaps the two middle axes.
    const size_t pageAxis = isPaBnbd_ ? 2U : 1U;
    const size_t headAxis = isPaged ? (isPaBnbd_ ? 1U : 2U) : qHeadAxis;
    const size_t dAxis = kvRank - 1U;
    if (q.GetDimNum() != qRank || k.GetDimNum() != kvRank || v.GetDimNum() != kvRank) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "query, key and value",
            ("qRank=" + std::to_string(qRank) + ", kvRank=" + std::to_string(kvRank) +
             ", query dimNum=" + std::to_string(q.GetDimNum()) + ", key dimNum=" + std::to_string(k.GetDimNum()) +
             ", value dimNum=" + std::to_string(v.GetDimNum()))
                .c_str(),
            "invalid Q/K/V ranks for the selected layouts");
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < qRank; ++axis) {
        if (q.GetDim(axis) < 0 || q.GetDim(axis) > std::numeric_limits<int32_t>::max()) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "query",
                ("axis=" + std::to_string(axis) + ", dim=" + std::to_string(q.GetDim(axis))).c_str(),
                "Q dimensions must be nonnegative and fit INT32");
            return ge::GRAPH_FAILED;
        }
    }
    if (q.GetDim(qHeadAxis) <= 0 || q.GetDim(qRank - 1U) <= 0 || q.GetDim(qRank - 1U) > GBSA_TILE_SIZE_512 ||
        (isBatched && ((!isPaged && q.GetDim(0) != k.GetDim(0)) || q.GetDim(qSeqAxis) == 0))) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "query",
            ("qHeads=" + std::to_string(q.GetDim(qHeadAxis)) + ", headDim=" + std::to_string(q.GetDim(qRank - 1U)) +
             ", batch=" + std::to_string(q.GetDim(0)) + ", seq=" + std::to_string(q.GetDim(qSeqAxis)))
                .c_str(),
            "invalid Q heads, headDim, batched capacity or Q/KV batch dimensions");
        return ge::GRAPH_FAILED;
    }
    if (socVer_ != SOC_VER_950_CODE && q.GetDim(qRank - 1U) > GBSA_TILE_SIZE_256) {
        OP_LOGE(context->GetNodeName(), "A2/A3 headDim must be in [1, 256].");
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < kvRank; ++axis) {
        if (k.GetDim(axis) < 0 || k.GetDim(axis) > std::numeric_limits<int32_t>::max() ||
            k.GetDim(axis) != v.GetDim(axis)) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "key and value",
                ("axis=" + std::to_string(axis) + ", key=" + std::to_string(k.GetDim(axis)) +
                 ", value=" + std::to_string(v.GetDim(axis)))
                    .c_str(),
                "K/V shapes must match and fit the supported index range");
            return ge::GRAPH_FAILED;
        }
    }
    numHeads_ = static_cast<uint32_t>(q.GetDim(qHeadAxis));
    embeddingSize_ = static_cast<uint32_t>(q.GetDim(qRank - 1U));
    kvHeads_ = static_cast<uint32_t>(k.GetDim(headAxis));
    if (kvHeads_ == 0U || numHeads_ % kvHeads_ != 0U || k.GetDim(dAxis) != embeddingSize_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "query, key and value",
            ("numHeads=" + std::to_string(numHeads_) + ", kvHeads=" + std::to_string(kvHeads_) +
             ", headDim=" + std::to_string(embeddingSize_) + ", kvD=" + std::to_string(k.GetDim(dAxis)))
                .c_str(),
            "invalid Q/K/V head counts or head dimensions");
        return ge::GRAPH_FAILED;
    }
    groupSize_ = numHeads_ / kvHeads_;
    if (groupSize_ > GBSA_MAX_GROUP_SIZE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "query and key",
            ("groupSize=" + std::to_string(groupSize_) + " (numHeads=" + std::to_string(numHeads_) +
             ", kvHeads=" + std::to_string(kvHeads_) + ")")
                .c_str(),
            ("currently only groupSize <= " + std::to_string(GBSA_MAX_GROUP_SIZE) +
             " is supported to avoid kernel L0C/UB overflow")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    totalQTokens_ = static_cast<uint64_t>(q.GetDim(0));
    totalKvTokens_ = static_cast<uint64_t>(k.GetDim(0));
    if (isBatched) {
        batch_ = static_cast<uint32_t>(q.GetDim(0));
        qSeqCapacity_ = static_cast<uint32_t>(q.GetDim(qSeqAxis));
        totalQTokens_ *= qSeqCapacity_;
        if (!isPaged) {
            kvSeqCapacity_ = static_cast<uint32_t>(k.GetDim(qSeqAxis));
            totalKvTokens_ *= kvSeqCapacity_;
        }
    }
    blockSize_ = 0U;
    if (isPaged) {
        blockSize_ = static_cast<uint32_t>(k.GetDim(pageAxis));
        if (blockSize_ != 128U) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "key",
                                                      ("blockSize=" + std::to_string(blockSize_)).c_str(),
                                                      "PA page size must be 128, independently of blockShapeY");
            return ge::GRAPH_FAILED;
        }
        totalKvTokens_ *= blockSize_;
    }
    if (totalKvTokens_ > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / kvHeads_ / embeddingSize_) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "key and value",
                                               ("totalKvTokens=" + std::to_string(totalKvTokens_)).c_str(),
                                               "KV storage size exceeds the supported address range");
        return ge::GRAPH_FAILED;
    }
    if (totalQTokens_ > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / kvHeads_) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query",
                                               ("totalQTokens=" + std::to_string(totalQTokens_)).c_str(),
                                               "Q task capacity exceeds the metadata index range");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseSparseTensors(gert::TilingContext* context)
{
    // layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
    const gert::StorageShape* sparseIdxShape = context->GetInputShape(SPARSE_BLOCK_IDX_INDEX);
    if (sparseIdxShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparseBlockIdx", "nullptr",
                                              "the shape of sparseBlockIdx must be provided");
        return ge::GRAPH_FAILED;
    }

    if (sparseIdxShape->GetStorageShape().GetDimNum() != SPARSE_IDX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "sparseBlockIdx",
            ("dimNum=" + std::to_string(sparseIdxShape->GetStorageShape().GetDimNum())).c_str(),
            "sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK]");
        return ge::GRAPH_FAILED;
    }
    auto sparseIdxDesc = context->GetInputDesc(SPARSE_BLOCK_IDX_INDEX);
    if (sparseIdxDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparseBlockIdx", "nullptr",
                                              "the desc of sparseBlockIdx must be provided");
        return ge::GRAPH_FAILED;
    }
    if (sparseIdxDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "sparseBlockIdx",
            std::to_string(static_cast<int32_t>(sparseIdxDesc->GetDataType())).c_str(),
            "the dtype of sparseBlockIdx must be INT32");
        return ge::GRAPH_FAILED;
    }

    if (sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD) != kvHeads_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "sparseBlockIdx and key",
            ("N_kv=" + std::to_string(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD)) +
             ", kvHeads=" + std::to_string(kvHeads_))
                .c_str(),
            "the sparse head count must equal the K/V head count");
        return ge::GRAPH_FAILED;
    }
    const int64_t topK = sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_BLOCK);
    qBlockNum_ =
        static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)); // totalQBlocks
    if (sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK) != static_cast<int64_t>(totalQTokens_) ||
        topK <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "sparseBlockIdx",
            ("totalQBlocks=" + std::to_string(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)) +
             ", totalQTokens=" + std::to_string(totalQTokens_) + ", topK=" + std::to_string(topK))
                .c_str(),
            "for blockShapeX=1, totalQBlocks must equal the Q storage token capacity and topK must be positive");
        return ge::GRAPH_FAILED;
    }
    constexpr int64_t GBSA_LEGACY_MAX_TOPK = 256;
    const int64_t maxTopK =
        quantType_ == GBSA_QUANT_TYPE_NONE || (socVer_ == SOC_VER_950_CODE && quantType_ == GBSA_QUANT_TYPE_MXFP8) ?
            std::numeric_limits<int32_t>::max() :
            GBSA_LEGACY_MAX_TOPK;
    if (topK > maxTopK) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "sparseBlockIdx",
            ("topK=" + std::to_string(topK) + ", maxTopK=" + std::to_string(maxTopK)).c_str(),
            "topK exceeds the maximum supported for this configuration");
        return ge::GRAPH_FAILED;
    }
    topK_ = static_cast<uint32_t>(topK);

    // sparseBlockCount 2D: [N_kv, totalQBlocks]
    const gert::StorageShape* sparseCountShape = context->GetInputShape(SPARSE_BLOCK_COUNT_INDEX);
    if (sparseCountShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparseBlockCount", "nullptr",
                                              "the shape of sparseBlockCount must be provided");
        return ge::GRAPH_FAILED;
    }

    if (sparseCountShape->GetStorageShape().GetDimNum() != SPARSE_COUNT_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "sparseBlockCount",
            ("dimNum=" + std::to_string(sparseCountShape->GetStorageShape().GetDimNum())).c_str(),
            "sparseBlockCount must be 2D [N_kv, totalQBlocks]");
        return ge::GRAPH_FAILED;
    }
    auto sparseCountDesc = context->GetInputDesc(SPARSE_BLOCK_COUNT_INDEX);
    if (sparseCountDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sparseBlockCount", "nullptr",
                                              "the desc of sparseBlockCount must be provided");
        return ge::GRAPH_FAILED;
    }
    if (sparseCountDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "sparseBlockCount",
            std::to_string(static_cast<int32_t>(sparseCountDesc->GetDataType())).c_str(),
            "the dtype of sparseBlockCount must be INT32");
        return ge::GRAPH_FAILED;
    }

    const uint32_t sparseCountKvHeads =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_KV_HEAD));
    const uint32_t sparseCountQBlocks =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_Q_BLOCK));
    if (sparseCountKvHeads != kvHeads_ || sparseCountQBlocks != qBlockNum_) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "sparseBlockCount and sparseBlockIdx",
            ("sparseBlockCount shape=[" + std::to_string(sparseCountKvHeads) + "," +
             std::to_string(sparseCountQBlocks) + "], sparseBlockIdx [N_kv,totalQBlocks]=[" + std::to_string(kvHeads_) +
             "," + std::to_string(qBlockNum_) + "]")
                .c_str(),
            "the shape of sparseBlockCount must match sparseBlockIdx");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseBlockTable(gert::TilingContext* context)
{
    maxBlocksPerBatch_ = 0U;
    if (!isPaged_) {
        return ge::GRAPH_SUCCESS;
    }
    // blockTable is OPTIONAL in OpDef — must use GetOptionalInputShape (GetInputShape always nullptr).
    const auto* blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    if (blockTableShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockTable", "nullptr",
                                              "paged KV requires blockTable to be provided");
        return ge::GRAPH_FAILED;
    }
    if (blockTableShape->GetStorageShape().GetDimNum() != 2U ||
        blockTableShape->GetStorageShape().GetDim(0) != batch_ || blockTableShape->GetStorageShape().GetDim(1) <= 0 ||
        blockTableShape->GetStorageShape().GetDim(1) > std::numeric_limits<int32_t>::max()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "blockTable",
            ("dimNum=" + std::to_string(blockTableShape->GetStorageShape().GetDimNum()) +
             ", batch=" + std::to_string(blockTableShape->GetStorageShape().GetDim(0)) +
             ", maxBlocksPerBatch=" + std::to_string(blockTableShape->GetStorageShape().GetDim(1)))
                .c_str(),
            "paged KV requires blockTable [batch, maxBlocksPerBatch]; batch must match the query batch and "
            "maxBlocksPerBatch must be positive");
        return ge::GRAPH_FAILED;
    }
    const auto* blockTableDesc = context->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    if (blockTableDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "blockTable", "nullptr",
                                              "the desc of blockTable must be provided");
        return ge::GRAPH_FAILED;
    }
    if (blockTableDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context->GetNodeName(), "blockTable",
            std::to_string(static_cast<int32_t>(blockTableDesc->GetDataType())).c_str(),
            "the dtype of blockTable must be INT32");
        return ge::GRAPH_FAILED;
    }
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQkvDtype(gert::TilingContext* context)
{
    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }
    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    auto valueDesc = context->GetInputDesc(VALUE_INDEX);
    if (keyDesc == nullptr || valueDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "key and value", "desc is nullptr",
                                               "the descs of key and value must be provided");
        return ge::GRAPH_FAILED;
    }
    if (keyDesc->GetDataType() != dataType_ || valueDesc->GetDataType() != dataType_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context->GetNodeName(), "query, key and value",
            ("query=" + std::to_string(static_cast<int32_t>(dataType_)) +
             ", key=" + std::to_string(static_cast<int32_t>(keyDesc->GetDataType())) +
             ", value=" + std::to_string(static_cast<int32_t>(valueDesc->GetDataType())))
                .c_str(),
            "the dtypes of query/key/value must match");
        return ge::GRAPH_FAILED;
    }
    if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "query",
                                              std::to_string(static_cast<int32_t>(dataType_)).c_str(),
                                              "only float16/bfloat16/float8_e4m3fn are supported for query/key/value");
        return ge::GRAPH_FAILED;
    }
    // The legacy per-tensor FP8 path only instantiates PA_BBND.
    if (isPaBnbd_ && quantType_ == GBSA_QUANT_TYPE_FULL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "kv_input_layout", layoutKv_.c_str(),
                                              "PA_BNBD does not support quantType=5; use PA_BBND");
        return ge::GRAPH_FAILED;
    }

    if (scaleValue_ == 0.0f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }
    if (CheckAttentionOutDtype(context) != ge::GRAPH_SUCCESS || CheckSoftmaxPrecision(context) != ge::GRAPH_SUCCESS ||
        CheckQuantConfig(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseInputTensors(gert::TilingContext* context)
{
    if (ParseQueryKeyShapes(context) != ge::GRAPH_SUCCESS || ParseSparseTensors(context) != ge::GRAPH_SUCCESS ||
        ParseSequenceInputs(context) != ge::GRAPH_SUCCESS || ParseBlockTable(context) != ge::GRAPH_SUCCESS ||
        ParseQkvDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CalculateWorkSpace(gert::TilingContext* context)
{
    uint64_t pipelineWorkspaceSize = 0;
    if (socVer_ != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        pipelineWorkspaceSize = mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_;
    } else {
        // MXFP8 uses descriptor rings instead of an identity index array.
        pipelineWorkspaceSize =
            quantType_ == GBSA_QUANT_TYPE_MXFP8 ? 32U : static_cast<uint64_t>(topK_) * sizeof(int32_t);
        if (IsMxfp8PackedPath()) {
            // Four AIV-produced K/V descriptor slots per AIC.
            // Each slot has a cache-line header plus at most one 16-byte record
            // per sparse block. Match the packed kernel's descriptor lookahead ring.
            const uint64_t descriptorBytes =
                GbsaMxScaleCache::DescriptorTileWords(kvBaseTile_, blockShapeY_) * sizeof(uint64_t);
            constexpr uint64_t descriptorSlots = GbsaMxScaleCache::DESCRIPTOR_RING_SLOTS;
            pipelineWorkspaceSize = blockDim_ * descriptorSlots * descriptorBytes;
            // K/V payloads remain in their input layout and go directly to L1.
            // Only bounded descriptor rings occupy the packed workspace.
            tilingData_->set_kvOffsetInfoOffset(0U);
            tilingData_->set_kvOffsetTileWords(0U);
        }
        if (embeddingSize_ > GBSA_TILE_SIZE_256 && qBaseTile_ > GBSA_TILE_SIZE_64) {
            // Two complete FP32 LO tiles per physical AIC; FD workspace follows this region.
            mm2OutSize_ =
                static_cast<uint64_t>(blockDim_) * 2U * qBaseTile_ * AlignUp(embeddingSize_, 16U) * sizeof(float);
            pipelineWorkspaceSize = std::max(pipelineWorkspaceSize, mm2OutSize_);
        }
    }

    // Four per-AIC slots. Each L1-N group has a 64B header and
    // PA entries hold {K offset, V offset, K meta, V meta}; dense entries
    // pack {offset, meta} but retain the original four-word capacity.
    if (socVer_ == SOC_VER_950_CODE && quantType_ != GBSA_QUANT_TYPE_MXFP8) {
        const uint64_t sparseBlocks = (mm1L1TileN_ + 2ULL * blockShapeY_ - 2U) / blockShapeY_;
        const uint64_t copyBlocks = isPaged_ ? std::min<uint64_t>(mm1L1TileN_, sparseBlocks * 2U) : sparseBlocks;
        const uint64_t recordWords = 4U;
        const uint64_t groupBytes = AlignUp(64U + copyBlocks * recordWords * sizeof(uint64_t), 64U);
        const uint64_t tileWords = ((kvBaseTile_ + mm1L1TileN_ - 1U) / mm1L1TileN_) * groupBytes / sizeof(uint64_t);
        const uint64_t offsetBytes = static_cast<uint64_t>(blockDim_) * 4U * tileWords * sizeof(uint64_t);
        const uint64_t offset = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        if (tileWords > std::numeric_limits<uint32_t>::max() ||
            offset > std::numeric_limits<uint64_t>::max() - offsetBytes) {
            OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "KV offset workspace size overflow.");
            return ge::GRAPH_FAILED;
        }
        tilingData_->set_kvOffsetInfoOffset(offset);
        tilingData_->set_kvOffsetTileWords(static_cast<uint32_t>(tileWords));
        pipelineWorkspaceSize = offset + offsetBytes;
    }
    uint64_t userWorkspaceSize = pipelineWorkspaceSize;
    if (fdStaticEnabled_) {
        // saTotalTaskNum * 10 < physicalAicNum * 3.
        const uint32_t maxNonEmptyBaseTaskNum =
            aicNum_ == 0U ? 0U :
                            std::min(GSA_FD_MAX_COMBINE_TASK_NUM, (aicNum_ * GSA_FD_BASE_TASK_GATE_NUMERATOR - 1U) /
                                                                      GSA_FD_BASE_TASK_GATE_DENOMINATOR);
        const uint32_t maxActiveCoreNum = std::min(aicNum_, GSA_FD_MAX_ACTIVE_CORE_NUM);
        // The base-task intervals and active-core intervals are two continuous
        // partitions of the same flat task range, count is bounded by Bmax + Cmax - 1.
        fdPartialCapacity_ = maxNonEmptyBaseTaskNum == 0U || maxActiveCoreNum == 0U ?
                                 0U :
                                 maxNonEmptyBaseTaskNum + maxActiveCoreNum - 1U;
        // Each FD partial task is split across two Vector sub-blocks. Reserve the larger half of groupSize and
        // align each sub-block's FP32 LSE slice to 8 elements (32 bytes) for GM DataCopy alignment.
        fdLseSubStride_ = ((groupSize_ + 1U) / 2U + 7U) / 8U * 8U;
        fdPartialLseOffset_ = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialLseSize =
            static_cast<uint64_t>(fdPartialCapacity_) * 2U * fdLseSubStride_ * sizeof(float);
        fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + partialLseSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t fdPartialOStride = AlignUp(embeddingSize_, 16U);
        const uint64_t partialOSize =
            static_cast<uint64_t>(fdPartialCapacity_) * groupSize_ * fdPartialOStride * sizeof(float);
        if (fdPartialOOffset_ > std::numeric_limits<uint64_t>::max() - partialOSize) {
            OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Flash Decoding workspace size overflow.");
            return ge::GRAPH_FAILED;
        }
        userWorkspaceSize = fdPartialOOffset_ + partialOSize;
    }
    if (userWorkspaceSize > std::numeric_limits<size_t>::max() - libapiSize_) {
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "workspace size overflow.");
        return ge::GRAPH_FAILED;
    }
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    context->SetBlockDim(blockDim_);
    size_t* workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckMetadata(gert::TilingContext* context)
{
    // Metadata is required: INT32, 1D, fixed size. Content is filled by AICPU and not re-checked here.
    const gert::StorageShape* metadataShape = context->GetOptionalInputShape(METADATA_INDEX);
    if (metadataShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "metadata", "nullptr",
                                              "metadata must be provided");
        return ge::GRAPH_FAILED;
    }
    if (metadataShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "metadata",
            ("dimNum=" + std::to_string(metadataShape->GetStorageShape().GetDimNum())).c_str(), "metadata must be 1D");
        return ge::GRAPH_FAILED;
    }
    const int64_t metadataSize = metadataShape->GetStorageShape().GetDim(0);
    if (metadataSize != static_cast<int64_t>(GBSA_METADATA_TOTAL_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "metadata", ("dim0=" + std::to_string(metadataSize)).c_str(),
            ("the dim 0 of metadata must be " + std::to_string(GBSA_METADATA_TOTAL_SIZE)).c_str());
        return ge::GRAPH_FAILED;
    }
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_INDEX);
    if (metadataDesc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "metadata", "nullptr",
                                              "the desc of metadata must be provided");
        return ge::GRAPH_FAILED;
    }
    if (metadataDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "metadata",
                                              std::to_string(static_cast<int32_t>(metadataDesc->GetDataType())).c_str(),
                                              "the dtype of metadata must be INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckReservedOptionalInputs(gert::TilingContext* context)
{
    // Reserved optional tensors must stay nullptr until the feature is wired.
    if (context->GetOptionalInputTensor(ATTEN_MASK_INDEX) != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "atten_mask", "provided",
                                              "atten_mask is not supported yet and must be nullptr");
        return ge::GRAPH_FAILED;
    }
    if (quantType_ == GBSA_QUANT_TYPE_MXFP8) {
        return CheckMxfp8Scales(context);
    }
    if (context->GetOptionalInputTensor(P_QUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "p_quant_scale", "provided",
                                              "p_quant_scale is not supported yet and must be nullptr");
        return ge::GRAPH_FAILED;
    }
    // FP8 full-quant path does not consume dequant scales yet; reject non-null (tests pass null).
    if (context->GetOptionalInputTensor(Q_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(K_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(V_DEQUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(),
                                               "q_dequant_scale, k_dequant_scale and v_dequant_scale", "provided",
                                               "q/k/v_dequant_scale are not supported yet and must be nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckMxfp8Scales(gert::TilingContext* context)
{
    const auto& qShape = context->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto& kShape = context->GetInputShape(KEY_INDEX)->GetOriginShape();
    // The trailing pair packs two independent group-of-32 E8M0 scales.
    // V is per-channel: the caller repeats each channel scale over the sequence groups.
    const auto checkScale = [&](int index, ge::DataType dtype, const std::vector<int64_t>& dimensions) -> bool {
        const auto* shape = context->GetOptionalInputShape(index);
        const auto* desc = context->GetOptionalInputDesc(index);
        if (shape == nullptr || desc == nullptr || desc->GetDataType() != dtype ||
            shape->GetStorageShape().GetDimNum() != dimensions.size()) {
            OP_LOGE(context->GetNodeName(), "MXFP8 scale input %d is missing or has invalid dtype/rank.", index);
            return false;
        }
        size_t axis = 0;
        for (const int64_t expected : dimensions) {
            if (shape->GetStorageShape().GetDim(axis) != expected) {
                OP_LOGE(context->GetNodeName(), "MXFP8 scale input %d dim %zu must be %ld.", index, axis, expected);
                return false;
            }
            ++axis;
        }
        return true;
    };
    const auto packedDimensions = [this](const gert::Shape& shape) {
        std::vector<int64_t> dimensions;
        for (size_t axis = 0; axis + 1U < shape.GetDimNum(); ++axis) {
            dimensions.push_back(shape.GetDim(axis));
        }
        dimensions.push_back((embeddingSize_ + 63U) / 64U);
        dimensions.push_back(2);
        return dimensions;
    };
    // Dense V uses one repeated scale pair per sequence/head/channel, independent
    // of its data layout. Paged V retains the existing group-of-64 token axis.
    const std::vector<int64_t> vDimensions =
        !isPaged_ ? std::vector<int64_t>{batch_, kvHeads_, embeddingSize_, 2} :
                    (isPaBnbd_ ? std::vector<int64_t>{kShape.GetDim(0), kvHeads_, 2, embeddingSize_, 2} :
                                 std::vector<int64_t>{kShape.GetDim(0), 2, kvHeads_, embeddingSize_, 2});
    if (!checkScale(Q_DEQUANT_SCALE_INDEX, ge::DT_FLOAT8_E8M0, packedDimensions(qShape)) ||
        !checkScale(K_DEQUANT_SCALE_INDEX, ge::DT_FLOAT8_E8M0, packedDimensions(kShape)) ||
        !checkScale(V_DEQUANT_SCALE_INDEX, ge::DT_FLOAT8_E8M0, vDimensions)) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputShape(P_QUANT_SCALE_INDEX) != nullptr &&
        !checkScale(P_QUANT_SCALE_INDEX, ge::DT_FLOAT, {1})) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

namespace {
const char* SequenceInputName(int index)
{
    switch (index) {
        case CU_SEQ_LENGTHS_Q_INDEX:
            return "cuSeqLengthsQ";
        case CU_SEQ_LENGTHS_KV_INDEX:
            return "cuSeqLengthsKv";
        case SEQUSED_Q_INDEX:
            return "sequsedQ";
        default:
            return "sequsedKv";
    }
}
} // namespace

ge::graphStatus GBSATiling::ParseSequenceInputs(gert::TilingContext* context)
{
    const bool isPaged = isPaged_;
    // TND derives B from cuQ; batched layouts already obtained B from Q shape.
    const int indices[] = {CU_SEQ_LENGTHS_Q_INDEX, CU_SEQ_LENGTHS_KV_INDEX, SEQUSED_Q_INDEX, SEQUSED_KV_INDEX};
    for (const int index : indices) {
        const bool isCu = index == CU_SEQ_LENGTHS_Q_INDEX || index == CU_SEQ_LENGTHS_KV_INDEX;
        const bool required = (index == CU_SEQ_LENGTHS_Q_INDEX && layoutQ_ == "TND") ||
                              (index == CU_SEQ_LENGTHS_KV_INDEX && layoutKv_ == "TND") ||
                              (index == SEQUSED_Q_INDEX && layoutQ_ != "TND") ||
                              (index == SEQUSED_KV_INDEX && (isPaged || layoutQ_ != "TND"));
        const auto* shape = context->GetOptionalInputShape(index);
        if (shape == nullptr) {
            if (required) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), SequenceInputName(index), "nullptr",
                                                      "this sequence input is required for the selected layouts");
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (isPaged && index == CU_SEQ_LENGTHS_KV_INDEX) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "cuSeqLengthsKv", "provided",
                                                  "cuSeqLengthsKv must be absent for paged KV");
            return ge::GRAPH_FAILED;
        }
        const auto* desc = context->GetOptionalInputDesc(index);
        if (desc == nullptr || desc->GetDataType() != (isCu ? ge::DT_INT64 : ge::DT_INT32) ||
            shape->GetStorageShape().GetDimNum() != 1U) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), SequenceInputName(index),
                ("dtype=" +
                 std::to_string(static_cast<int32_t>(desc != nullptr ? desc->GetDataType() : ge::DT_UNDEFINED)) +
                 ", dimNum=" + std::to_string(shape->GetStorageShape().GetDimNum()))
                    .c_str(),
                isCu ? "cu-sequence inputs must be INT64 1D" : "seqused inputs must be INT32 1D");
            return ge::GRAPH_FAILED;
        }
        const int64_t length = shape->GetStorageShape().GetDim(0);
        if (index == CU_SEQ_LENGTHS_Q_INDEX && layoutQ_ == "TND") {
            if (length < 1 || length - 1 > std::numeric_limits<int32_t>::max()) {
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "cuSeqLengthsQ",
                                                          ("size=" + std::to_string(length)).c_str(),
                                                          "TND Q requires cuSeqLengthsQ[B+1] with B within INT32");
                return ge::GRAPH_FAILED;
            }
            batch_ = static_cast<uint32_t>(length - 1);
        } else if (length != static_cast<int64_t>(batch_) + (isCu ? 1 : 0)) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), SequenceInputName(index),
                ("size=" + std::to_string(length) + ", batch=" + std::to_string(batch_)).c_str(),
                "the length of the sequence input must match the batch");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckSoftmaxPrecision(gert::TilingContext* context)
{
    if (socVer_ == SOC_VER_950_CODE) {
        if (quantType_ == GBSA_QUANT_TYPE_FULL && softmaxPrecision_ != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "softmaxPrecision",
                                                  std::to_string(softmaxPrecision_).c_str(),
                                                  "on chip 950, FP8 full-quant only supports softmaxPrecision=1");
            return ge::GRAPH_FAILED;
        }
    } else if (dataType_ == ge::DT_BF16 && softmaxPrecision_ == 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "softmaxPrecision", std::to_string(softmaxPrecision_).c_str(),
            "on chip 910 & 910_93, when query dtype is bfloat16, only softmaxPrecision=0 is supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckQuantConfig(gert::TilingContext* context)
{
    // Preserve arch22 non-quant capabilities while adding arch35 MXFP8.
    const bool isMxfp8 = quantType_ == GBSA_QUANT_TYPE_MXFP8;
    const bool isQuant = quantType_ != GBSA_QUANT_TYPE_NONE;
    const bool generalizedMxfp8 = socVer_ == SOC_VER_950_CODE && isMxfp8 && dataType_ == ge::DT_FLOAT8_E4M3FN;
    if ((!isPaged_ || layoutQ_ != "TND") && !generalizedMxfp8 &&
        (quantType_ != 0 || (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "quantType and query/key/value dtype",
            ("quantType=" + std::to_string(quantType_) + ", dtype=" + std::to_string(static_cast<int32_t>(dataType_)))
                .c_str(),
            "dense non-paged KV requires FP16/BF16 quantType=0 or arch35 MXFP8");
        return ge::GRAPH_FAILED;
    }
    const bool isFp8 = (dataType_ == ge::DT_FLOAT8_E4M3FN);
    const bool isQuant5 = (quantType_ == GBSA_QUANT_TYPE_FULL);
    if (isFp8 != isQuant) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "quantType and query/key/value dtype",
            ("quantType=" + std::to_string(quantType_) + ", dtype=" + std::to_string(static_cast<int32_t>(dataType_)))
                .c_str(),
            "FP8 requires quantType=5 or 6 with FLOAT8_E4M3FN query/key/value");
        return ge::GRAPH_FAILED;
    }
    if (maskType_ == 0 && isQuant5) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "mask_type", std::to_string(maskType_).c_str(),
                                              "non-causal attention (mask_type=0) only supports FP16/BF16 non-quant");
        return ge::GRAPH_FAILED;
    }
    if (isQuant && socVer_ != SOC_VER_950_CODE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "quantType", std::to_string(quantType_).c_str(),
                                              "FP8 full-quant is only supported on chip 950");
        return ge::GRAPH_FAILED;
    }
    if (returnSoftmaxlse_ && isQuant5) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "return_softmax_lse and quantType",
            ("return_softmax_lse=1, quantType=" + std::to_string(quantType_)).c_str(),
            "returnSoftmaxlse=1 is not supported for the FP8 full-quant path");
        return ge::GRAPH_FAILED;
    }
    if (blockShapeY_ != 128U) {
        const bool arch35NonQuant =
            socVer_ == SOC_VER_950_CODE && quantType_ == 0 && (dataType_ == ge::DT_FLOAT16 || dataType_ == ge::DT_BF16);
        const bool arch22NonQuant =
            socVer_ != SOC_VER_950_CODE && quantType_ == 0 && (dataType_ == ge::DT_FLOAT16 || dataType_ == ge::DT_BF16);
        const bool supportedArch22DenseY =
            blockShapeY_ >= 1U && blockShapeY_ <= 128U && (blockShapeY_ & (blockShapeY_ - 1U)) == 0U;
        const bool supportedArch22PagedY =
            blockShapeY_ >= 1U && blockShapeY_ <= 128U && (blockShapeY_ & (blockShapeY_ - 1U)) == 0U;
        const bool arch22DenseNonQuant = arch22NonQuant && !isPaged_ && supportedArch22DenseY;
        const bool arch22PagedNonQuant =
            arch22NonQuant && isPaged_ && blockSize_ == 128U && supportedArch22PagedY && (128U % blockShapeY_) == 0U;
        if (!arch35NonQuant && !generalizedMxfp8 && !arch22DenseNonQuant && !arch22PagedNonQuant) {
            OP_LOGE(context->GetNodeName(),
                    "Generalized blockShapeY=%u requires arch35 quantType=0 or 6, or arch22 "
                    "FP16/BF16 quantType=0 with power-of-two Y in [1, 128] (PA page size 128).",
                    blockShapeY_);
            return ge::GRAPH_FAILED;
        }
    }
    if (isMxfp8 && embeddingSize_ != 192U && embeddingSize_ != GBSA_TILE_SIZE_256) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "embedding_size",
                                              std::to_string(embeddingSize_).c_str(),
                                              "MXFP8 only supports headDim 192 or 256; D128 is not supported");
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE && !isQuant && embeddingSize_ > GBSA_TILE_SIZE_256) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "embedding_size",
                                              std::to_string(embeddingSize_).c_str(),
                                              "on chip 950 FP16/BF16 non-quant currently supports headDim up to 256");
        return ge::GRAPH_FAILED;
    }
    const bool fdHeadDimSupported =
        socVer_ == SOC_VER_950_CODE ?
            (embeddingSize_ == GBSA_TILE_SIZE_128 || ((!isQuant || isMxfp8) && embeddingSize_ <= GBSA_TILE_SIZE_256)) :
            (embeddingSize_ >= 1U && embeddingSize_ <= GBSA_TILE_SIZE_256);
    fdStaticEnabled_ = topK_ >= 12U && fdHeadDimSupported;
    return ge::GRAPH_SUCCESS;
}

bool GBSATiling::IsMxfp8PackedPath() const
{
    // Q layout controls output strides; KV layout independently controls Raw
    // scale packing. Paged inputs retain 128-token pages and descriptor groups.
    const bool qHeadMajor = layoutQ_ == "BNSD";
    const bool headMajor = layoutKv_ == "BNSD" || layoutKv_ == "PA_BNBD";
    const bool packedLayout =
        (layoutQ_ == "TND" || layoutQ_ == "BSND" || qHeadMajor) &&
        (isPaged_ ? blockSize_ == 128U && (layoutKv_ == "PA_BBND" || layoutKv_ == "PA_BNBD") : layoutKv_ == layoutQ_);
    // TND has no fixed per-batch capacity in the shape. Bound addresses by
    // total storage tokens; the kernel uses cuKv and optional sequsedKv per batch.
    const uint64_t kvCapacity = isPaged_           ? static_cast<uint64_t>(maxBlocksPerBatch_) * blockSize_ :
                                layoutKv_ == "TND" ? totalKvTokens_ :
                                                     kvSeqCapacity_;
    const uint64_t kScaleRowBytes = static_cast<uint64_t>(kvHeads_) * (embeddingSize_ / 32U);
    const uint32_t qTile = GbsaMxScaleCache::QueryTileRows(groupSize_);
    const uint32_t tile = GbsaMxScaleCache::KvTileTokens(groupSize_);
    return socVer_ == SOC_VER_950_CODE && quantType_ == GBSA_QUANT_TYPE_MXFP8 && batch_ != 0U && packedLayout &&
           (embeddingSize_ == 192U || embeddingSize_ == GBSA_TILE_SIZE_256) &&
           (blockShapeY_ == 4U || blockShapeY_ == 8U) &&
           // Both accepted precision values use FP32 scores. Value 1 remains
           // a compatibility alias; ParseAttrs rejects values outside 0/1.
           // D192 and D256 share both the four head-group tiers and D256 buffer
           // capacity. Input/output and scale strides still use the actual D.
           groupSize_ <= 128U && !returnSoftmaxlse_ && kvCapacity != 0U && kScaleRowBytes != 0U &&
           kvCapacity * kvHeads_ <= UINT32_MAX / embeddingSize_ &&
           // Output DataCopyPad's GM head stride is a uint32_t byte count.
           (!qHeadMajor || static_cast<uint64_t>(qSeqCapacity_) * embeddingSize_ * sizeof(uint16_t) <= UINT32_MAX) &&
           GbsaMxScaleCache::StreamFits(kvHeads_, embeddingSize_, GbsaMxScaleCache::RawTileTokens(tile), blockShapeY_,
                                        headMajor) &&
           GbsaMxScaleCache::PipelineL1End(embeddingSize_, qTile, tile) <= GbsaMxScaleCache::L1_DATA_CAPACITY &&
           GbsaMxScaleCache::PipelineUbEnd(kvSeqCapacity_, kvHeads_, embeddingSize_, blockShapeY_, tile, headMajor,
                                           qTile) <= GbsaMxScaleCache::UB_CAPACITY;
}

uint64_t GBSATiling::GenerateTilingKey()
{
    /**
     * BSA-style decimal bitfields (must match op_kernel tilingkey.h):
     * - [0-1]   Q Layout           2=TND, 3=BNSD, 4=BSND
     * - [2-4]   Mask Type          maskType_ * 1000 (0=non-causal, 1=causal)
     * - [5-7]   Softmax Precision  0=Float, 1=Half/BF16 (*100000)
     * - [8-10]  PagedCache         0=Dense, 1=WithCache (*1000000)
     * - [11-13] KV Layout          10=BNSD, 20=BSND, 30=TND, 70=PA_BBND, 80=PA_BNBD (*1000000)
     * - dtype                    FP16=+0, BF16=+22220, FP8 out FP16=+10 / BF16=+20
     * - LSE                      +100000000
     * - op+arch                  920=aicore220, 925=aicore310
     */
    const bool isArch35 = (socVer_ == SOC_VER_950_CODE);
    uint64_t tilingKey = isArch35 ? GBSA_OP_ARCH35_BASE : GBSA_OP_ARCH22_BASE;

    // dtype / attentionOut dtype additives
    if (dataType_ == ge::DT_FLOAT16) {
        // +0
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += GbsaTilingKeyEncoding::BF16_DTYPE_FIELD;
    } else if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (attentionOutDtype_ == ge::DT_FLOAT16) {
            tilingKey += GbsaTilingKeyEncoding::FP8_OUT_FP16_FIELD;
        } else if (attentionOutDtype_ == ge::DT_BF16) {
            tilingKey += GbsaTilingKeyEncoding::FP8_OUT_BF16_FIELD;
        }
    }

    // Arch22 shares its existing dense/paged dispatch across layouts. Layout strides
    // are carried in tiling data, including when static compilation folds the data.
    // Arch35 retains its layout-specific template dispatch.
    tilingKey += socVer_ != SOC_VER_950_CODE ?
                     (isPaged_ ? GbsaTilingKeyEncoding::KV_PA_BBND_FIELD : GbsaTilingKeyEncoding::KV_TND_FIELD) :
                 layoutKv_ == "PA_BBND" ? GbsaTilingKeyEncoding::KV_PA_BBND_FIELD :
                 layoutKv_ == "PA_BNBD" ? GbsaTilingKeyEncoding::KV_PA_BNBD_FIELD :
                 layoutKv_ == "BNSD"    ? GbsaTilingKeyEncoding::KV_BNSD_FIELD :
                 layoutKv_ == "BSND"    ? GbsaTilingKeyEncoding::KV_BSND_FIELD :
                                          GbsaTilingKeyEncoding::KV_TND_FIELD;
    if (isPaged_) {
        tilingKey += GbsaTilingKeyEncoding::PAGED_FIELD;
    }

    // Softmax precision: 0=FP32 softmax, 1=FP16/BF16 softmax.
    if (softmaxPrecision_ == 1) {
        tilingKey += GbsaTilingKeyEncoding::LOW_PRECISION_SOFTMAX_FIELD;
    }

    // maskType: 0=non-causal, 1=causal.
    tilingKey += static_cast<uint64_t>(maskType_) * GbsaTilingKeyEncoding::MASK_FIELD;

    // Q layout: TND = 2, BNSD = 3, BSND = 4.
    tilingKey += socVer_ != SOC_VER_950_CODE ? GbsaTilingKeyEncoding::Q_TND_FIELD :
                 layoutQ_ == "BNSD"          ? GbsaTilingKeyEncoding::Q_BNSD_FIELD :
                 layoutQ_ == "BSND"          ? GbsaTilingKeyEncoding::Q_BSND_FIELD :
                                               GbsaTilingKeyEncoding::Q_TND_FIELD;

    if (returnSoftmaxlse_) {
        tilingKey += GBSA_LSE_OUT_OFFSET;
    }
    if (residualBlockMode_ == GBSA_RESIDUAL_BLOCK_MODE_FORCE) {
        tilingKey += GBSA_RESIDUAL_BLOCK_MODE_TILING_OFFSET;
    }
    if (quantType_ == GBSA_QUANT_TYPE_MXFP8) {
        tilingKey += GBSA_MXFP8_TILING_OFFSET;
    }
    return tilingKey;
}

void GBSATiling::CalcBaseTileTilingParams950()
{
    if (groupSize_ <= GBSA_TILE_SIZE_16) {
        qBaseTile_ = GBSA_TILE_SIZE_16;
    } else if (groupSize_ <= GBSA_TILE_SIZE_32) {
        qBaseTile_ = GBSA_TILE_SIZE_32;
    } else if (groupSize_ <= GBSA_TILE_SIZE_64) {
        qBaseTile_ = GBSA_TILE_SIZE_64;
    } else {
        qBaseTile_ = GBSA_TILE_SIZE_128;
    }

    if (embeddingSize_ <= GBSA_TILE_SIZE_256) {
        kvBaseTile_ = qBaseTile_ <= GBSA_TILE_SIZE_32 ? GBSA_TILE_SIZE_512 :
                      qBaseTile_ <= GBSA_TILE_SIZE_64 ? GBSA_TILE_SIZE_256 :
                                                        GBSA_TILE_SIZE_128;
    } else {
        kvBaseTile_ = qBaseTile_ <= GBSA_TILE_SIZE_16 ? GBSA_TILE_SIZE_384 :
                      qBaseTile_ <= GBSA_TILE_SIZE_32 ? GBSA_TILE_SIZE_256 :
                                                        GBSA_TILE_SIZE_128;
    }

    // High-precision softmax supports up to 512 columns. Low precision still
    // supports at most 256, including the 384-column tile used by D > 256.
    if (softmaxPrecision_ == 1U && kvBaseTile_ > GBSA_TILE_SIZE_256) {
        kvBaseTile_ = GBSA_TILE_SIZE_256;
    }
}

void GBSATiling::CalcMatmulPhaseL1TileInfo950()
{
    const uint32_t embeddingSizeAligned = embeddingSize_ <= GBSA_TILE_SIZE_128 ? GBSA_TILE_SIZE_128 :
                                          embeddingSize_ <= GBSA_TILE_SIZE_256 ? GBSA_TILE_SIZE_256 :
                                                                                 GBSA_TILE_SIZE_512;

    // Q is resident in L1. K is the only matrix further split in L1 and uses
    // double buffering. P and V hold a complete attention base tile.
    mm1L1TileM_ = qBaseTile_;
    mm1L1TileKLeft_ = embeddingSizeAligned;
    qL1BufNum_ = 1U;
    kL1BufNum_ = 2U;

    mm2L1TileM_ = qBaseTile_;
    mm2L1TileN_ = embeddingSizeAligned;
    mm2L1TileKLeft_ = kvBaseTile_;
    mm2L1TileKRight_ = kvBaseTile_;
    pL1BufNum_ = 3U;

    if (embeddingSize_ <= GBSA_TILE_SIZE_128) {
        mm1L1TileN_ = qBaseTile_ <= GBSA_TILE_SIZE_64 ? GBSA_TILE_SIZE_256 : GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_128;
        vL1BufNum_ = 2U;
    } else if (embeddingSize_ <= GBSA_TILE_SIZE_256) {
        mm1L1TileN_ = GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_256;
        vL1BufNum_ = qBaseTile_ <= GBSA_TILE_SIZE_32 ? 1U : 2U;
    } else {
        mm1L1TileN_ = GBSA_TILE_SIZE_128;
        mm1L1TileKRight_ = GBSA_TILE_SIZE_128;
        vL1BufNum_ = (qBaseTile_ == GBSA_TILE_SIZE_64) ? 2U : 1U;
    }
}

ge::graphStatus GBSATiling::FillTilingData(gert::TilingContext* context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_qBlockNum(qBlockNum_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_groupSize(groupSize_);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    const bool packed = IsMxfp8PackedPath();
    // Both precision values select the same FP32-S grouped kernel. Preserve
    // the requested precision in metadata/tiling validation and retain the keys.
    const bool coalescedPipeline = packed && !isPaged_ && embeddingSize_ == GBSA_TILE_SIZE_256 && blockShapeY_ == 4U &&
                                   maskType_ == 1U && qSeqCapacity_ == kvSeqCapacity_ &&
                                   qSeqCapacity_ >= GBSA_MXFP8_COALESCED_PREFILL_MIN_TOKENS &&
                                   qSeqCapacity_ <= GBSA_MXFP8_COALESCED_PREFILL_MAX_TOKENS;
    const bool headMajor = layoutQ_ == "BNSD";
    uint64_t pipelineDispatch = headMajor ? (coalescedPipeline ? GBSA_MXFP8_BNSD_COALESCED_PIPELINE_DISPATCH_BASE :
                                                                 GBSA_MXFP8_BNSD_PIPELINE_DISPATCH_BASE) :
                                            (coalescedPipeline ? GBSA_MXFP8_COALESCED_PIPELINE_DISPATCH_BASE :
                                                                 GBSA_MXFP8_SCALE_PIPELINE_DISPATCH_BASE);
    if (layoutQ_ == "TND") {
        pipelineDispatch =
            isPaged_ ? (isPaBnbd_ ? GBSA_MXFP8_TND_PA_BNBD_DISPATCH_BASE : GBSA_MXFP8_TND_PA_BBND_DISPATCH_BASE) :
                       GBSA_MXFP8_TND_PIPELINE_DISPATCH_BASE;
    } else if (isPaged_) {
        pipelineDispatch =
            headMajor ? (isPaBnbd_ ? GBSA_MXFP8_BNSD_PA_BNBD_DISPATCH_BASE : GBSA_MXFP8_BNSD_PA_BBND_DISPATCH_BASE) :
                        (isPaBnbd_ ? GBSA_MXFP8_BSND_PA_BNBD_DISPATCH_BASE : GBSA_MXFP8_BSND_PA_BBND_DISPATCH_BASE);
    }
    const uint64_t dispatchKey =
        packed ? pipelineDispatch + (attentionOutDtype_ == ge::DT_FLOAT16 ? 0U : 1U) : tilingKey;
    context->SetTilingKey(dispatchKey);

    tilingData_->baseTileInfo.set_qBaseTile(qBaseTile_);
    tilingData_->baseTileInfo.set_kvBaseTile(kvBaseTile_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileM(mm1L1TileM_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileN(mm1L1TileN_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileKLeft(mm1L1TileKLeft_);
    tilingData_->mmPhaseL1TileInfo.set_mm1L1TileKRight(mm1L1TileKRight_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileM(mm2L1TileM_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileN(mm2L1TileN_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileKLeft(mm2L1TileKLeft_);
    tilingData_->mmPhaseL1TileInfo.set_mm2L1TileKRight(mm2L1TileKRight_);
    tilingData_->mmPhaseL1TileInfo.set_qL1BufNum(qL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_kL1BufNum(kL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_vL1BufNum(vL1BufNum_);
    tilingData_->mmPhaseL1TileInfo.set_pL1BufNum(pL1BufNum_);
    tilingData_->set_kStride0(kStride0_);
    tilingData_->set_vStride0(vStride0_);
    tilingData_->set_fdStaticEnabled(fdStaticEnabled_ ? 1U : 0U);
    tilingData_->set_fdLseSubStride(fdLseSubStride_);
    tilingData_->set_fdPartialCapacity(fdPartialCapacity_);
    tilingData_->set_fdPartialLseOffset(fdPartialLseOffset_);
    tilingData_->set_fdPartialOOffset(fdPartialOOffset_);
    tilingData_->set_residualBlockMode(static_cast<uint32_t>(residualBlockMode_));
    tilingData_->set_isConsistentTopk(static_cast<uint32_t>(isConsistentTopk_ ? 1 : 0));
    tilingData_->set_qSeqCapacity(qSeqCapacity_);
    tilingData_->set_kvSeqCapacity(kvSeqCapacity_);
    tilingData_->set_qLayout(layoutQ_ == "TND" ? 0U : layoutQ_ == "BSND" ? 1U : 2U);
    tilingData_->set_kvLayout(isPaBnbd_ ? 3U : layoutKv_ == "BNSD" ? 2U : layoutKv_ == "BSND" ? 1U : 0U);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_totalKvTokens(totalKvTokens_);
    tilingData_->set_maskType(static_cast<uint32_t>(maskType_));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::GetTiling(gert::TilingContext* context, GenericBlockSparseAttentionTilingData& tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (GetInputLayout(context) != ge::GRAPH_SUCCESS || ParseAttrs(context) != ge::GRAPH_SUCCESS ||
        ParseInputTensors(context) != ge::GRAPH_SUCCESS || ParseKvCacheStride0(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMetadata(context) != ge::GRAPH_SUCCESS || CheckReservedOptionalInputs(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (quantType_ == GBSA_QUANT_TYPE_MXFP8 && !IsMxfp8PackedPath()) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported MXFP8 configuration: only grouped packed execution is supported; "
                "Fragment fallback has been removed. Requires nonempty B/KV, TND/BSND/BNSD Q with matching dense KV "
                "or PA_BBND/PA_BNBD KV (128-token contiguous pages), "
                "TND cuSeqLengths for storage prefixes (optional seqused for actual lengths), "
                "D192/256 groupSize<=128, softmaxPrecision=0/1 (FP32 scores), "
                "returnSoftmaxlse=false, and blockY=4/8. "
                "Address and UB capacity limits must also be satisfied. "
                "Got B=%u, Sq=%u, Skv=%u, D=%u, blockY=%u, groupSize=%u, layouts=%s/%s, precision=%u, LSE=%u.",
                batch_, qSeqCapacity_, kvSeqCapacity_, embeddingSize_, blockShapeY_, groupSize_, layoutQ_.c_str(),
                layoutKv_.c_str(), static_cast<uint32_t>(softmaxPrecision_), static_cast<uint32_t>(returnSoftmaxlse_));
        return ge::GRAPH_FAILED;
    }
    if (socVer_ == SOC_VER_950_CODE && quantType_ != GBSA_QUANT_TYPE_MXFP8) {
        CalcBaseTileTilingParams950();
        CalcMatmulPhaseL1TileInfo950();
    } else if (socVer_ != SOC_VER_950_CODE) {
        kvBaseTile_ = GBSA_TILE_SIZE_256;
    } else if (quantType_ == GBSA_QUANT_TYPE_MXFP8) {
        qBaseTile_ = GbsaMxScaleCache::QueryTileRows(groupSize_);
        kvBaseTile_ = GbsaMxScaleCache::KvTileTokens(groupSize_);
        mm1L1TileM_ = qBaseTile_;
        // QK streams 1/2/4 K groups through three physical L1 slots.
        // P/V retain the full 128/256/512-token logical tile selected by the head group.
        mm1L1TileN_ = GbsaMxScaleCache::GROUP_TOKENS;
        mm1L1TileKLeft_ = GbsaMxScaleCache::BufferDim(embeddingSize_);
        mm1L1TileKRight_ = mm1L1TileKLeft_;
        mm2L1TileM_ = qBaseTile_;
        mm2L1TileN_ = GbsaMxScaleCache::BufferDim(embeddingSize_);
        mm2L1TileKLeft_ = kvBaseTile_;
        mm2L1TileKRight_ = kvBaseTile_;
        kL1BufNum_ = 3U;
        vL1BufNum_ = 2U;
    } else {
        const bool smallM = groupSize_ <= GBSA_TILE_SIZE_16 && blockShapeX_ == 1U;
        kvBaseTile_ = smallM && embeddingSize_ <= GBSA_TILE_SIZE_256 ? GBSA_TILE_SIZE_512 : GBSA_TILE_SIZE_256;
    }
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::SetTilingData(gert::TilingContext* context,
                                          GenericBlockSparseAttentionTilingData& tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    GenericBlockSparseAttentionTilingData tilingData;
    GBSATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        if (tiling.SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    } else {
        // 下层校验函数已通过 ErrorManager 首报具体错误码,此处仅透传失败状态,避免重复首报
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGenericBlockSparseAttention(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttention)
    .Tiling(TilingGenericBlockSparseAttention)
    .TilingParse<GenericBlockSparseAttentionCompileInfo>(TilingPrepareForGenericBlockSparseAttention);

} // namespace optiling
