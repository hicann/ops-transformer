/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
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

// TND + isPackedGQA=1 sparseBlockIdx 3D: [N_kv, totalQBlocks, topK]
constexpr int SPARSE_IDX_DIM_KV_HEAD = 0;
constexpr int SPARSE_IDX_DIM_Q_BLOCK = 1;
constexpr int SPARSE_IDX_DIM_KV_BLOCK = 2;
constexpr int SPARSE_IDX_DIM_NUM = 3;

// TND + isPackedGQA=1 sparseBlockCount 2D: [N_kv, totalQBlocks]
constexpr int SPARSE_COUNT_DIM_KV_HEAD = 0;
constexpr int SPARSE_COUNT_DIM_Q_BLOCK = 1;
constexpr int SPARSE_COUNT_DIM_NUM = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_BLOCK_SHAPE_INDEX = 0;
constexpr int ATTR_IS_PACKED_GQA_INDEX = 1;
constexpr int ATTR_Q_INPUT_LAYOUT_INDEX = 2;
constexpr int ATTR_KV_INPUT_LAYOUT_INDEX = 3;
constexpr int ATTR_SCALE_VALUE_INDEX = 4;
constexpr int ATTR_MASK_TYPE_INDEX = 5;
constexpr int ATTR_QUANT_TYPE_INDEX = 6;
constexpr int ATTR_DST_TYPE_MAX_INDEX = 7;
constexpr int ATTR_SOFTMAX_PRECISION_INDEX = 8;
constexpr int ATTR_WIN_LEFT_INDEX = 9;
constexpr int ATTR_WIN_RIGHT_INDEX = 10;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 11;

constexpr uint32_t SOC_VER_950_CODE = 4;
constexpr uint32_t GBSA_MAX_GROUP_SIZE = 128U;
constexpr int64_t GBSA_QUANT_TYPE_NONE = 0;
constexpr int64_t GBSA_QUANT_TYPE_FULL = 5;
constexpr int64_t GBSA_WIN_DISABLED = -1;
constexpr float GBSA_DST_TYPE_MAX_DISABLED = 0.0f;

namespace optiling {

ge::graphStatus GBSATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    if (aicNum_ == 0) {
        OP_LOGE(context->GetNodeName(), "GetCoreNumAic returned 0.");
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

ge::graphStatus ParseBlockShapeAttr(gert::TilingContext *context,
                                    const gert::TypedContinuousVector<int64_t> *blockShapeArr, uint32_t &blockShapeX,
                                    uint32_t &blockShapeY)
{
    if (blockShapeArr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (blockShapeArr->GetSize() != 2) {
        OP_LOGE(context->GetNodeName(), "blockShape must contain two elements [x, y], got size %zu.",
                blockShapeArr->GetSize());
        return ge::GRAPH_FAILED;
    }
    const int64_t *data = blockShapeArr->GetData();
    if (data == nullptr) {
        OP_LOGE(context->GetNodeName(), "blockShape data is null.");
        return ge::GRAPH_FAILED;
    }
    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(context->GetNodeName(), "blockShape values must be positive, got [%ld, %ld].", data[0], data[1]);
        return ge::GRAPH_FAILED;
    }
    blockShapeX = static_cast<uint32_t>(data[0]);
    blockShapeY = static_cast<uint32_t>(data[1]);
    if (blockShapeX != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported blockShapeX=%u, currently only blockShapeX=1 is supported.",
                blockShapeX);
        return ge::GRAPH_FAILED;
    }
    if (blockShapeY != 128) {
        OP_LOGE(context->GetNodeName(), "Unsupported blockShapeY=%u, currently only blockShapeY=128 is supported.",
                blockShapeY);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus GBSATiling::ParseCapabilityAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const int64_t *softmaxPrecPtr = attrs->GetInt(ATTR_SOFTMAX_PRECISION_INDEX);
    if (softmaxPrecPtr != nullptr) {
        softmaxPrecision_ = static_cast<uint32_t>(*softmaxPrecPtr);
    }
    if (softmaxPrecision_ != 0 && softmaxPrecision_ != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported softmaxPrecision=%u, only 0 (fp32 SM) or 1 (half/low SM) are supported.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *maskTypePtr = attrs->GetInt(ATTR_MASK_TYPE_INDEX);
    if (maskTypePtr != nullptr) {
        maskType_ = *maskTypePtr;
    }
    if (maskType_ != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported maskType=%ld, only maskType=1 is supported.", maskType_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *quantTypePtr = attrs->GetInt(ATTR_QUANT_TYPE_INDEX);
    if (quantTypePtr != nullptr) {
        quantType_ = *quantTypePtr;
    }
    if (quantType_ != GBSA_QUANT_TYPE_NONE && quantType_ != GBSA_QUANT_TYPE_FULL) {
        OP_LOGE(context->GetNodeName(), "Unsupported quantType=%ld, only 0 (none) or 5 (full-quant) are supported.",
                quantType_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *isPackedGqaPtr = attrs->GetInt(ATTR_IS_PACKED_GQA_INDEX);
    if (isPackedGqaPtr != nullptr) {
        isPackedGQA_ = *isPackedGqaPtr;
    }
    if (isPackedGQA_ != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported isPackedGQA=%ld, only 1 (packed GQA) is supported.", isPackedGQA_);
        return ge::GRAPH_FAILED;
    }

    const int64_t *lseFlagPtr = attrs->GetInt(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    if (lseFlagPtr != nullptr) {
        softmaxLseFlag_ = *lseFlagPtr;
    }
    if (softmaxLseFlag_ != 0) {
        OP_LOGE(context->GetNodeName(), "Unsupported returnSoftmaxlse=%ld, only 0 is supported.", softmaxLseFlag_);
        return ge::GRAPH_FAILED;
    }
    returnSoftmaxlse_ = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckReservedAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float *dstTypeMaxPtr = attrs->GetFloat(ATTR_DST_TYPE_MAX_INDEX);
    if (dstTypeMaxPtr != nullptr && *dstTypeMaxPtr != GBSA_DST_TYPE_MAX_DISABLED) {
        OP_LOGE(context->GetNodeName(), "Since dst_type_max is not yet supported, it must be 0, but got %f.",
                *dstTypeMaxPtr);
        return ge::GRAPH_FAILED;
    }
    const int64_t *winLeftPtr = attrs->GetInt(ATTR_WIN_LEFT_INDEX);
    const int64_t *winRightPtr = attrs->GetInt(ATTR_WIN_RIGHT_INDEX);
    const int64_t winLeft = (winLeftPtr != nullptr) ? *winLeftPtr : GBSA_WIN_DISABLED;
    const int64_t winRight = (winRightPtr != nullptr) ? *winRightPtr : GBSA_WIN_DISABLED;
    if (winLeft != GBSA_WIN_DISABLED || winRight != GBSA_WIN_DISABLED) {
        OP_LOGE(context->GetNodeName(),
                "Since windowed atten mask is not yet supported, "
                "win_left & win_right must be -1, but got %ld, %ld.",
                winLeft, winRight);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
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

ge::graphStatus GBSATiling::GetInputLayout(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "GetAttrs returned nullptr.");
        return ge::GRAPH_FAILED;
    }

    const char *layoutQPtr = attrs->GetStr(ATTR_Q_INPUT_LAYOUT_INDEX);
    if (layoutQPtr != nullptr) {
        layoutQ_ = std::string(layoutQPtr);
    }
    const char *layoutKvPtr = attrs->GetStr(ATTR_KV_INPUT_LAYOUT_INDEX);
    if (layoutKvPtr != nullptr) {
        layoutKv_ = std::string(layoutKvPtr);
    }

    if (layoutQ_ != "TND") {
        OP_LOGE(context->GetNodeName(), "layoutQ only supports TND, got %s.", layoutQ_.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutKv_ != "PAGED_BBND") {
        OP_LOGE(context->GetNodeName(), "layoutKv only supports PAGED_BBND, got %s.", layoutKv_.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckAttentionOutDtype(gert::TilingContext *context)
{
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        auto outDesc = context->GetOutputDesc(ATTENTION_OUT_INDEX);
        if (outDesc == nullptr) {
            OP_LOGE(context->GetNodeName(), "attentionOut desc is nullptr.");
            return ge::GRAPH_FAILED;
        }
        attentionOutDtype_ = outDesc->GetDataType();
        if (attentionOutDtype_ != ge::DT_FLOAT16 && attentionOutDtype_ != ge::DT_BF16) {
            OP_LOGE(context->GetNodeName(),
                    "The supported dtype of attentionOut is float16 or bfloat16 when the dtype of query/key/value is "
                    "all float8_e4m3fn, but now it is %d.",
                    attentionOutDtype_);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// Validate PAGED_BBND key/value: only dim0 may be non-contiguous.
static ge::graphStatus ValidatePagedBbndDim0OnlyNonContig(gert::TilingContext *context, uint64_t inputIndex,
                                                          const gert::Shape &shape, const char *tensorName)
{
    auto *stride = context->GetRequiredInputStride(inputIndex);
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t expectedStride = 1;
    for (size_t i = shape.GetDimNum() - 1; i >= 1; --i) {
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(i));
        if (actualStride != expectedStride) {
            OP_LOGE(context->GetNodeName(),
                    "Tensor %s dim%zu is non-contiguous: actual stride=%llu, expected=%llu. "
                    "Only the first axis (dim0) may be non-contiguous for PAGED_BBND.",
                    tensorName, i, static_cast<unsigned long long>(actualStride),
                    static_cast<unsigned long long>(expectedStride));
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(shape.GetDim(i));
        if (i == 1) {
            break;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseKvCacheStride0(gert::TilingContext *context)
{
    const uint64_t pageElems =
        static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    if (keyShape == nullptr || valueShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value shape is nullptr when parsing KV stride0.");
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

    auto *keyStrides = context->GetRequiredInputStride(KEY_INDEX);
    kStride0_ = (keyStrides != nullptr && keyStrides->GetDimNum() > 0 && keyStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(keyStrides->GetStride(0)) :
                    pageElems;
    auto *valueStrides = context->GetRequiredInputStride(VALUE_INDEX);
    vStride0_ = (valueStrides != nullptr && valueStrides->GetDimNum() > 0 && valueStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(valueStrides->GetStride(0)) :
                    pageElems;

    const uint64_t rowElems = static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);
    if (kStride0_ < pageElems || (rowElems > 0 && (kStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "key dim0 stride (%llu) invalid for PAGED_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(kStride0_), static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    if (vStride0_ < pageElems || (rowElems > 0 && (vStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "value dim0 stride (%llu) invalid for PAGED_BBND: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(vStride0_), static_cast<unsigned long long>(pageElems),
                static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQueryKeyShapes(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(QUERY_INDEX);
    if (queryShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Query shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));
    if (embeddingSize_ != 128) {
        OP_LOGE(context->GetNodeName(), "Unsupported embeddingSize=%u, currently only D=128 is supported.",
                embeddingSize_);
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    if (keyShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "Key shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    // ND PA cache: use origin shape [blockNum, blockSize, Nkv, D].
    // Dim0-strided views can collapse storage shape, so do not use storage shape here.
    const gert::Shape &keyOrigin = keyShape->GetOriginShape();
    blockSize_ = static_cast<uint32_t>(keyOrigin.GetDim(BLOCKED_KV_DIM_BLOCK_SIZE));
    if (blockSize_ != blockShapeY_) {
        OP_LOGE(context->GetNodeName(), "KV page blockSize=%u must equal blockShapeY=%u.", blockSize_, blockShapeY_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseSparseTensors(gert::TilingContext *context)
{
    // TND + isPackedGQA=1: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
    const gert::StorageShape *sparseIdxShape = context->GetInputShape(SPARSE_BLOCK_IDX_INDEX);
    if (sparseIdxShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "sparseBlockIdx shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (sparseIdxShape->GetStorageShape().GetDimNum() != SPARSE_IDX_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK] for TND, but got %zu dims.",
                sparseIdxShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kvHeads_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD));
    qBlockNum_ =
        static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)); // totalQBlocks
    topK_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_BLOCK));
    if (kvHeads_ == 0 || numHeads_ % kvHeads_ != 0) {
        OP_LOGE(context->GetNodeName(), "numHeads=%u must be divisible by kvHeads=%u (and kvHeads > 0).", numHeads_,
                kvHeads_);
        return ge::GRAPH_FAILED;
    }
    const uint32_t groupSize = numHeads_ / kvHeads_;
    if (groupSize > GBSA_MAX_GROUP_SIZE) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported groupSize=%u (numHeads=%u, kvHeads=%u), currently only groupSize <= %u is supported to "
                "avoid kernel L0C/UB overflow.",
                groupSize, numHeads_, kvHeads_, GBSA_MAX_GROUP_SIZE);
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t GBSA_MAX_TOPK = 256U;
    if (topK_ > GBSA_MAX_TOPK) {
        OP_LOGE(context->GetNodeName(), "Unsupported topK=%u, only topK<=%u is supported on this build.", topK_,
                GBSA_MAX_TOPK);
        return ge::GRAPH_FAILED;
    }

    // sparseBlockCount 2D: [N_kv, totalQBlocks]
    const gert::StorageShape *sparseCountShape = context->GetInputShape(SPARSE_BLOCK_COUNT_INDEX);
    if (sparseCountShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "sparseBlockCount shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (sparseCountShape->GetStorageShape().GetDimNum() != SPARSE_COUNT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "sparseBlockCount must be 2D [N_kv, totalQBlocks] for TND, but got %zu dims.",
                sparseCountShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const uint32_t sparseCountKvHeads =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_KV_HEAD));
    const uint32_t sparseCountQBlocks =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_Q_BLOCK));
    if (sparseCountKvHeads != kvHeads_ || sparseCountQBlocks != qBlockNum_) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockCount shape [%u,%u] must match sparseBlockIdx [N_kv,totalQBlocks]=[%u,%u].",
                sparseCountKvHeads, sparseCountQBlocks, kvHeads_, qBlockNum_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseBlockTable(gert::TilingContext *context)
{
    // blockTable is OPTIONAL in OpDef — must use GetOptionalInputShape (GetInputShape always nullptr).
    const gert::StorageShape *blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    if (blockTableShape == nullptr) {
        blockTablePresent_ = false;
        OP_LOGE(context->GetNodeName(),
                "Stage 1 requires blockTable for PAGED_BBND layout, but blockTableOptional is nullptr.");
        return ge::GRAPH_FAILED;
    }
    blockTablePresent_ = true;
    batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
    maxBlocksPerBatch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::ParseQkvDtype(gert::TilingContext *context)
{
    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }
    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    auto valueDesc = context->GetInputDesc(VALUE_INDEX);
    if (keyDesc == nullptr || valueDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (keyDesc->GetDataType() != dataType_ || valueDesc->GetDataType() != dataType_) {
        OP_LOGE(context->GetNodeName(), "query/key/value dtypes must match, got query=%d key=%d value=%d.",
                static_cast<int32_t>(dataType_), static_cast<int32_t>(keyDesc->GetDataType()),
                static_cast<int32_t>(valueDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context->GetNodeName(), "Unsupported query/key/value dtype=%d, only float16/bfloat16/float8_e4m3fn.",
                static_cast<int32_t>(dataType_));
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

ge::graphStatus GBSATiling::ParseInputTensors(gert::TilingContext *context)
{
    if (ParseQueryKeyShapes(context) != ge::GRAPH_SUCCESS || ParseSparseTensors(context) != ge::GRAPH_SUCCESS ||
        ParseBlockTable(context) != ge::GRAPH_SUCCESS || ParseQkvDtype(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CalculateWorkSpace(gert::TilingContext *context)
{
    if (socVer_ != SOC_VER_950_CODE) {
        constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
        constexpr uint32_t NUM3 = 3;
        // Identity reserved after S/P/O buffers (must match kernel layout).
        mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(uint16_t) * NUM3;
        mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        workSpaceSize_ = libapiSize_ + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ + identityIdxSize;
    } else {
        uint32_t dtypeSize = (dataType_ == ge::DT_FLOAT8_E4M3FN) ? 1 : 2;
        uint64_t perTaskWorkspace = static_cast<uint64_t>(topK_) * blockShapeY_ * embeddingSize_ * dtypeSize * 2;
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        workSpaceSize_ = libapiSize_ + identityIdxSize + static_cast<uint64_t>(blockDim_) * perTaskWorkspace;
    }

    context->SetBlockDim(blockDim_);
    size_t *workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckMetadata(gert::TilingContext *context)
{
    // Metadata is required: INT32, 1D, fixed size. Content is filled by AICPU and not re-checked here.
    const gert::StorageShape *metadataShape = context->GetOptionalInputShape(METADATA_INDEX);
    if (metadataShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata must be provided.");
        return ge::GRAPH_FAILED;
    }
    if (metadataShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "metadata dim num must be 1, but got %zu.",
                metadataShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const int64_t metadataSize = metadataShape->GetStorageShape().GetDim(0);
    if (metadataSize != static_cast<int64_t>(GBSA_METADATA_TOTAL_SIZE)) {
        OP_LOGE(context->GetNodeName(), "metadata dim 0 must be %u, but got %ld.", GBSA_METADATA_TOTAL_SIZE,
                metadataSize);
        return ge::GRAPH_FAILED;
    }
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_INDEX);
    if (metadataDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (metadataDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "metadata dtype must be DT_INT32, but got %d.",
                static_cast<int32_t>(metadataDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckReservedOptionalInputs(gert::TilingContext *context)
{
    // Reserved optional tensors must stay nullptr until the feature is wired.
    if (context->GetOptionalInputTensor(ATTEN_MASK_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "atten_mask is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputTensor(P_QUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "p_quant_scale is NOT YET supported.");
        return ge::GRAPH_FAILED;
    }
    // Full-quant kernel does not consume dequant scales yet; reject non-null (tests pass null).
    if (context->GetOptionalInputTensor(Q_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(K_DEQUANT_SCALE_INDEX) != nullptr ||
        context->GetOptionalInputTensor(V_DEQUANT_SCALE_INDEX) != nullptr) {
        OP_LOGE(context->GetNodeName(), "q/k/v_dequant_scale are NOT YET supported and must be nullptr.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckCuSeqLengths(gert::TilingContext *context)
{
    // layoutQ is fixed to TND and layoutKv is fixed to PAGED_BBND (validated in GetInputLayout),
    // so cuSeqLengthsQ/Kv are required to locate per-batch token ranges on device.
    if (context->GetOptionalInputTensor(CU_SEQ_LENGTHS_Q_INDEX) == nullptr) {
        OP_LOGE(context->GetNodeName(), "cuSeqLengthsQ cannot be empty when layoutQ is TND.");
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputTensor(CU_SEQ_LENGTHS_KV_INDEX) == nullptr) {
        OP_LOGE(context->GetNodeName(), "cuSeqLengthsKv cannot be empty when layoutKv is PAGED_BBND.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckSoftmaxPrecision(gert::TilingContext *context)
{
    if (socVer_ == SOC_VER_950_CODE) {
        if (softmaxPrecision_ != 1) {
            OP_LOGE(context->GetNodeName(), "On chip 950, only softmaxPrecision=1 is supported, but got %u.",
                    softmaxPrecision_);
            return ge::GRAPH_FAILED;
        }
    } else if (dataType_ == ge::DT_BF16 && softmaxPrecision_ == 1) {
        OP_LOGE(context->GetNodeName(),
                "On chip 910 & 910_93, when query dtype is bfloat16, "
                "only softmaxPrecision=0 is supported, but got %u.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::CheckQuantConfig(gert::TilingContext *context)
{
    // Full-quant contract: quantType=5 iff Q/K/V dtype is FLOAT8_E4M3FN; FP8 only on 950.
    const bool isFp8 = (dataType_ == ge::DT_FLOAT8_E4M3FN);
    const bool isQuant5 = (quantType_ == GBSA_QUANT_TYPE_FULL);
    if (isFp8 != isQuant5) {
        OP_LOGE(context->GetNodeName(),
                "FP8 full-quant requires quantType=5 with FLOAT8_E4M3FN Q/K/V, "
                "got quantType=%ld dtype=%d.",
                quantType_, static_cast<int32_t>(dataType_));
        return ge::GRAPH_FAILED;
    }
    if (isQuant5 && socVer_ != SOC_VER_950_CODE) {
        OP_LOGE(context->GetNodeName(), "FP8 full-quant is only supported on chip 950, but socVer=%u.", socVer_);
        return ge::GRAPH_FAILED;
    }
    if (returnSoftmaxlse_ && isQuant5) {
        OP_LOGE(context->GetNodeName(), "returnSoftmaxlse=1 is not supported for FP8 full-quant path.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GBSATiling::GenerateTilingKey()
{
    /**
     * BSA-style decimal bitfields (must match op_kernel tilingkey.h):
     * - [0-1]   Q Layout           2=TND
     * - [2-4]   Mask Type          maskType_ * 1000 (current path: 1)
     * - [5-7]   Softmax Precision  0=Float, 1=Half (*100000); arch35 keys keep 0
     * - [8-10]  PagedCache         1=WithCache (*1000000)
     * - [11-13] KV Layout          70=PAGED_BBND (*1000000 → 70000000)
     * - dtype                    FP16=+0, BF16=+22220, FP8 out FP16=+10 / BF16=+20
     * - LSE                      +100000000
     * - op+arch                  920=aicore220, 925=aicore310
     */
    const bool isArch35 = (socVer_ == SOC_VER_950_CODE);
    uint64_t tilingKey = isArch35 ? GBSA_OP_ARCH35_BASE : GBSA_OP_ARCH22_BASE;

    // dtype / FP8 out-dtype (same additives as BSA)
    if (dataType_ == ge::DT_FLOAT16) {
        // +0
    } else if (dataType_ == ge::DT_BF16) {
        tilingKey += 22220ULL;
    } else if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (attentionOutDtype_ == ge::DT_FLOAT16) {
            tilingKey += 10ULL;
        } else if (attentionOutDtype_ == ge::DT_BF16) {
            tilingKey += 20ULL;
        }
    }

    // KV layout: PAGED_BBND = 70
    tilingKey += 70000000ULL;
    // Paged cache required on current path
    tilingKey += 1000000ULL;

    // Softmax precision: only discriminates kernels on arch22
    if (!isArch35 && softmaxPrecision_ == 1) {
        tilingKey += 100000ULL;
    }

    // maskType (current support: 1)
    tilingKey += static_cast<uint64_t>(maskType_) * 1000ULL;

    // Q layout: TND = 2
    tilingKey += 2ULL;

    if (returnSoftmaxlse_) {
        tilingKey += GBSA_LSE_OUT_OFFSET;
    }
    return tilingKey;
}

ge::graphStatus GBSATiling::FillTilingData(gert::TilingContext *context)
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
    uint32_t groupSize = numHeads_ / kvHeads_;
    tilingData_->set_groupSize(groupSize);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    // BaseTileInfo
    uint32_t qBaseTile = (embeddingSize_ <= 128) ? 128 : 64;
    uint32_t kvBaseTile = blockShapeY_;
    tilingData_->set_qBaseTile(qBaseTile);
    tilingData_->set_kvBaseTile(kvBaseTile);

    // MmPhaseL1TileInfo: QK matmul L1 tile = [qBaseTile, kvBaseTile, embed]
    tilingData_->set_mm1L1TileM(qBaseTile);
    tilingData_->set_mm1L1TileN(kvBaseTile);
    tilingData_->set_mm1L1TileKLeft(embeddingSize_);
    tilingData_->set_mm1L1TileKRight(embeddingSize_);
    // PV matmul L1 tile = [qBaseTile, embed, kvBaseTile]
    tilingData_->set_mm2L1TileM(qBaseTile);
    tilingData_->set_mm2L1TileN(embeddingSize_);
    tilingData_->set_mm2L1TileKLeft(kvBaseTile);
    tilingData_->set_mm2L1TileKRight(kvBaseTile);
    // Buffer counts
    tilingData_->set_qL1BufNum(1);
    tilingData_->set_kL1BufNum(1);
    tilingData_->set_vL1BufNum(1);
    tilingData_->set_pL1BufNum(3); // PRE_LAUNCH + 1
    tilingData_->set_kStride0(kStride0_);
    tilingData_->set_vStride0(vStride0_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::GetTiling(gert::TilingContext *context, GenericBlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;
    ge::graphStatus ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GetNpuInfo failed");
        return ret;
    }
    if (GetInputLayout(context) != ge::GRAPH_SUCCESS || ParseAttrs(context) != ge::GRAPH_SUCCESS ||
        ParseInputTensors(context) != ge::GRAPH_SUCCESS || ParseKvCacheStride0(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Parse attrs/inputs failed");
        return ge::GRAPH_FAILED;
    }
    if (CheckMetadata(context) != ge::GRAPH_SUCCESS || CheckReservedOptionalInputs(context) != ge::GRAPH_SUCCESS ||
        CheckCuSeqLengths(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Validate config failed");
        return ge::GRAPH_FAILED;
    }
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "CalculateWorkSpace failed");
        return ret;
    }
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "FillTilingData failed");
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GBSATiling::SetTilingData(gert::TilingContext *context,
                                          GenericBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttention(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    GenericBlockSparseAttentionTilingData tilingData;
    GBSATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGenericBlockSparseAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttention)
    .Tiling(TilingGenericBlockSparseAttention)
    .TilingParse<GenericBlockSparseAttentionCompileInfo>(TilingPrepareForGenericBlockSparseAttention);

} // namespace optiling
