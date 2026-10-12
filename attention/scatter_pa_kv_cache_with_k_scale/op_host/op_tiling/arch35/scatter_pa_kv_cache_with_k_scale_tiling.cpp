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
 * \file scatter_pa_kv_cache_with_k_scale_tiling.cpp
 * \brief Tiling implementation for scatter_pa_kv_cache_with_k_scale
 */

#include "platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "version/metadef_version.h"
#include "../../../op_kernel/arch35/scatter_pa_kv_cache_with_k_scale_tiling_data.h"
#include "../../../op_kernel/arch35/scatter_pa_kv_cache_with_k_scale_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

// ==================== 常量定义 ====================

// 平台相关常量
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

// 维度数量常量
constexpr int64_t KEY_DIM_NUM = 3;
constexpr int64_t KEY_CACHE_DIM_NUM = 4;
constexpr int64_t VALUE_CACHE_DIM_NUM = 4;
constexpr int64_t SLOT_MAPPING_DIM_NUM = 1;
constexpr int64_t KEY_SCALE_DIM_NUM = 2;
constexpr int64_t KEY_SCALE_CACHE_DIM_NUM = 4;

// 输入tensor索引常量
constexpr size_t INPUT_KEY_IDX = 0;
constexpr size_t INPUT_VALUE_IDX = 1;
constexpr size_t INPUT_KEY_CACHE_IDX = 2;
constexpr size_t INPUT_VALUE_CACHE_IDX = 3;
constexpr size_t INPUT_SLOT_MAPPING_IDX = 4;
constexpr size_t INPUT_KEY_SCALE_IDX = 5;
constexpr size_t INPUT_KEY_SCALE_CACHE_IDX = 6;

// 输出tensor索引常量
constexpr size_t OUTPUT_KEY_CACHE_IDX = 0;
constexpr size_t OUTPUT_VALUE_CACHE_IDX = 1;
constexpr size_t OUTPUT_KEY_SCALE_CACHE_IDX = 2;

// shape维度索引常量
constexpr int64_t DIM_TOKEN = 0;
constexpr int64_t DIM_HEAD = 1;
constexpr int64_t DIM_HEAD_SIZE = 2;
constexpr int64_t DIM_BLOCK = 0;
constexpr int64_t DIM_BLOCK_SIZE = 2;
constexpr int64_t DIM_CACHE_HEAD = 1;
constexpr int64_t DIM_CACHE_HEAD_SIZE = 3;
constexpr int64_t DIM_SLOT_TOKEN = 0;
constexpr int64_t DIM_SCALE_TOKEN = 0;
constexpr int64_t DIM_SCALE_HEAD = 1;
constexpr int64_t DIM_SCALE_CACHE_BLOCK = 0;
constexpr int64_t DIM_SCALE_CACHE_HEAD = 1;
constexpr int64_t DIM_SCALE_CACHE_BLOCK_SIZE = 2;
constexpr int64_t DIM_SCALE_CACHE_SIZE = 3;

// 核心数相关常量
constexpr int64_t MIN_CORE_NUM = 1;

// 字节大小常量
constexpr size_t SIZEOF_32BIT = 4;
constexpr size_t SIZEOF_64BIT = 8;

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

struct ScatterPaKvCacheWithKScaleCompileInfo {};

// ==================== tensor info (shape + stride, align gather_elements) ====================

static void GetTensorInfo(gert::TilingContext* ctx, size_t idx, int64_t* shape, int64_t* stride, int64_t& dimNum)
{
    auto* shapeRef = ctx->GetInputShape(idx);

    bool isView = ctx->InputIsView(idx);
    if (isView) {
        auto* s = ctx->GetInputStride(idx);
        if (s != nullptr && s->GetDimNum() > 0) {
            auto viewShape = shapeRef->GetShape();
            dimNum = viewShape.GetDimNum();
            for (int64_t i = 0; i < dimNum; ++i) {
                shape[i] = viewShape.GetDim(i);
                stride[i] = s->GetStride(i);
            }
            return;
        }
    }

    auto storageShape = EnsureNotScalar(shapeRef->GetStorageShape());
    dimNum = storageShape.GetDimNum();
    int64_t st = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        shape[i] = storageShape.GetDim(i);
        stride[i] = st;
        st *= storageShape.GetDim(i);
    }
}

// ==================== platform / workspace ====================

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ScatterPaKvCacheWithKScale", "coreNum", "0", "coreNum must be > 0");
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ScatterPaKvCacheWithKScale", "ubSize", "0", "ubSize must be > 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    int64_t userWorkspaceSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// ==================== dtype validation ====================

static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto keyDesc = context->GetInputDesc(INPUT_KEY_IDX);
    auto valueDesc = context->GetInputDesc(INPUT_VALUE_IDX);
    auto kcDesc = context->GetInputDesc(INPUT_KEY_CACHE_IDX);
    auto vcDesc = context->GetInputDesc(INPUT_VALUE_CACHE_IDX);
    auto smDesc = context->GetInputDesc(INPUT_SLOT_MAPPING_IDX);
    auto ksDesc = context->GetInputDesc(INPUT_KEY_SCALE_IDX);
    auto kscDesc = context->GetInputDesc(INPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, smDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kscDesc);

    auto keyDtype = keyDesc->GetDataType();
    auto valueDtype = valueDesc->GetDataType();
    auto kcDtype = kcDesc->GetDataType();
    auto vcDtype = vcDesc->GetDataType();
    auto smDtype = smDesc->GetDataType();
    auto ksDtype = ksDesc->GetDataType();
    auto kscDtype = kscDesc->GetDataType();

    if (keyDtype != ge::DT_FLOAT8_E5M2 && keyDtype != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key",
                                              Ops::Base::ToString(keyDtype).c_str(),
                                              "dtype must be DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN");
        return ge::GRAPH_FAILED;
    }
    if (keyDtype != valueDtype || keyDtype != kcDtype || keyDtype != vcDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "ScatterPaKvCacheWithKScale", "key, value, key_cache, value_cache",
            (std::string(Ops::Base::ToString(keyDtype)) + ", " + Ops::Base::ToString(valueDtype) + ", " +
             Ops::Base::ToString(kcDtype) + ", " + Ops::Base::ToString(vcDtype))
                .c_str(),
            "dtype of key, value, key_cache, and value_cache must be the same");
        return ge::GRAPH_FAILED;
    }
    if (smDtype != ge::DT_INT32 && smDtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("ScatterPaKvCacheWithKScale", "slot_mapping",
                                              Ops::Base::ToString(smDtype).c_str(),
                                              "dtype must be DT_INT32 or DT_INT64");
        return ge::GRAPH_FAILED;
    }
    if (ksDtype != ge::DT_FLOAT || kscDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "ScatterPaKvCacheWithKScale", "key_scale, key_scale_cache",
            (std::string(Ops::Base::ToString(ksDtype)) + ", " + Ops::Base::ToString(kscDtype)).c_str(),
            "dtype must be DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ==================== shape validation ====================

static gert::Shape GetTensorShape(gert::TilingContext* context, size_t idx)
{
    auto* shapeRef = context->GetInputShape(idx);
    if (shapeRef == nullptr) {
        return gert::Shape();
    }

    bool isView = context->InputIsView(idx);
    if (isView) {
        return shapeRef->GetShape();
    } else {
        return EnsureNotScalar(shapeRef->GetStorageShape());
    }
}

static ge::graphStatus ValidateShape(gert::TilingContext* context, int64_t numTokens, int64_t numHead,
                                     int64_t kHeadSize, int64_t vHeadSize, int64_t numBlocks, int64_t blockSize)
{
    auto valueShape = GetTensorShape(context, INPUT_VALUE_IDX);
    if (valueShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "value", "null", "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDim(DIM_HEAD) != numHead) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "value",
                                              std::to_string(valueShape.GetDim(DIM_HEAD)).c_str(),
                                              "shape[1] must be numHead");
        return ge::GRAPH_FAILED;
    }

    auto kcShape = GetTensorShape(context, INPUT_KEY_CACHE_IDX);
    if (kcShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_cache", "null", "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (kcShape.GetDimNum() != KEY_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "key_cache",
                                     std::to_string(kcShape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    if (kcShape.GetDim(DIM_CACHE_HEAD_SIZE) == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_cache",
                                                  std::to_string(kcShape.GetDim(DIM_CACHE_HEAD_SIZE)).c_str(),
                                                  "shape[3] cannot be 0");
        return ge::GRAPH_FAILED;
    }
    if (kcShape.GetDim(DIM_CACHE_HEAD) != numHead) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_cache",
                                              std::to_string(kcShape.GetDim(DIM_CACHE_HEAD)).c_str(),
                                              "shape[1] must be numHead");
        return ge::GRAPH_FAILED;
    }
    if (kcShape.GetDim(DIM_CACHE_HEAD_SIZE) != kHeadSize) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_cache",
                                              std::to_string(kcShape.GetDim(DIM_CACHE_HEAD_SIZE)).c_str(),
                                              "shape[3] must be kHeadSize");
        return ge::GRAPH_FAILED;
    }

    auto vcShape = GetTensorShape(context, INPUT_VALUE_CACHE_IDX);
    if (vcShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "value_cache", "null", "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (vcShape.GetDimNum() != VALUE_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "value_cache",
                                     std::to_string(vcShape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    if (vcShape.GetDim(DIM_CACHE_HEAD_SIZE) == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("ScatterPaKvCacheWithKScale", "value_cache", "value_cache_head_size",
                                                  "shape[3] cannot be 0");
        return ge::GRAPH_FAILED;
    }
    if (vcShape.GetDim(DIM_CACHE_HEAD) != numHead) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "value_cache",
                                              std::to_string(vcShape.GetDim(DIM_CACHE_HEAD)).c_str(),
                                              "shape[1] must be numHead");
        return ge::GRAPH_FAILED;
    }
    if (vcShape.GetDim(DIM_CACHE_HEAD_SIZE) != vHeadSize) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "value_cache",
                                              std::to_string(vcShape.GetDim(DIM_CACHE_HEAD_SIZE)).c_str(),
                                              "shape[3] must be vHeadSize");
        return ge::GRAPH_FAILED;
    }

    auto smShape = GetTensorShape(context, INPUT_SLOT_MAPPING_IDX);
    if (smShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "slot_mapping", "null", "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (smShape.GetDim(DIM_SLOT_TOKEN) != numTokens) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("ScatterPaKvCacheWithKScale", "slot_mapping", "slot_mapping_shape_0",
                                                  "shape[0] must be numTokens");
        return ge::GRAPH_FAILED;
    }

    auto ksShape = GetTensorShape(context, INPUT_KEY_SCALE_IDX);
    if (ksShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_scale", "null", "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (ksShape.GetDimNum() != KEY_SCALE_DIM_NUM || ksShape.GetDim(DIM_SCALE_TOKEN) != numTokens ||
        ksShape.GetDim(DIM_SCALE_HEAD) != numHead) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "ScatterPaKvCacheWithKScale", "key_scale", "key_scale_shape",
            "shape[0] must be numTokens, shape[1] must be numHead, and num dimensions must be 2");
        return ge::GRAPH_FAILED;
    }

    auto kscShape = GetTensorShape(context, INPUT_KEY_SCALE_CACHE_IDX);
    if (kscShape.GetDimNum() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ScatterPaKvCacheWithKScale", "key_scale_cache", "null",
                                              "shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (kscShape.GetDimNum() != KEY_SCALE_CACHE_DIM_NUM || kscShape.GetDim(DIM_SCALE_CACHE_BLOCK) != numBlocks ||
        kscShape.GetDim(DIM_SCALE_CACHE_HEAD) != numHead || kscShape.GetDim(DIM_SCALE_CACHE_BLOCK_SIZE) != blockSize ||
        kscShape.GetDim(DIM_SCALE_CACHE_SIZE) != 1) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "ScatterPaKvCacheWithKScale", "key_scale_cache", "key_scale_cache_shape",
            "shape[0] must be numBlocks, shape[1] must be numHead, shape[2] must be blockSize, shape[3] must be 1, "
            "and num dimensions must be 4");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ==================== stride copy helpers ====================

template <size_t N>
static void CopyStrides(int64_t* dst, const int64_t* src)
{
    for (size_t i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

// ==================== extract tensor parameters ====================

struct TensorParams {
    int64_t numTokens;
    int64_t numHead;
    int64_t kHeadSize;
    int64_t vHeadSize;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t maxSlot;
    int64_t keyStride[KEY_DIM_NUM];
    int64_t valueStride[KEY_DIM_NUM];
    int64_t keyCacheStride[KEY_CACHE_DIM_NUM];
    int64_t valueCacheStride[VALUE_CACHE_DIM_NUM];
    int64_t slotMappingStride[SLOT_MAPPING_DIM_NUM];
    int64_t keyScaleStride[KEY_SCALE_DIM_NUM];
    int64_t keyScaleCacheStride[KEY_SCALE_CACHE_DIM_NUM];
};

static ge::graphStatus ExtractTensorParams(gert::TilingContext* context, TensorParams& params)
{
    int64_t keyShapeArr[KEY_DIM_NUM];
    int64_t keyStrideArr[KEY_DIM_NUM];
    int64_t keyDim;
    GetTensorInfo(context, INPUT_KEY_IDX, keyShapeArr, keyStrideArr, keyDim);
    if (keyDim != KEY_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "key", std::to_string(keyDim).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    params.numTokens = keyShapeArr[DIM_TOKEN];
    params.numHead = keyShapeArr[DIM_HEAD];
    params.kHeadSize = keyShapeArr[DIM_HEAD_SIZE];
    CopyStrides<KEY_DIM_NUM>(params.keyStride, keyStrideArr);

    int64_t valueShapeArr[KEY_DIM_NUM];
    int64_t valueStrideArr[KEY_DIM_NUM];
    int64_t valueDim;
    GetTensorInfo(context, INPUT_VALUE_IDX, valueShapeArr, valueStrideArr, valueDim);
    if (valueDim != KEY_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "value", std::to_string(valueDim).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    params.vHeadSize = valueShapeArr[DIM_HEAD_SIZE];
    CopyStrides<KEY_DIM_NUM>(params.valueStride, valueStrideArr);

    int64_t kcShapeArr[KEY_CACHE_DIM_NUM];
    int64_t kcStrideArr[KEY_CACHE_DIM_NUM];
    int64_t kcDim;
    GetTensorInfo(context, INPUT_KEY_CACHE_IDX, kcShapeArr, kcStrideArr, kcDim);
    if (kcDim != KEY_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "key_cache", std::to_string(kcDim).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    params.numBlocks = kcShapeArr[DIM_BLOCK];
    params.blockSize = kcShapeArr[DIM_BLOCK_SIZE];
    CopyStrides<KEY_CACHE_DIM_NUM>(params.keyCacheStride, kcStrideArr);

    int64_t vcShapeArr[VALUE_CACHE_DIM_NUM];
    int64_t vcStrideArr[VALUE_CACHE_DIM_NUM];
    int64_t vcDim;
    GetTensorInfo(context, INPUT_VALUE_CACHE_IDX, vcShapeArr, vcStrideArr, vcDim);
    if (vcDim != VALUE_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "value_cache", std::to_string(vcDim).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    CopyStrides<VALUE_CACHE_DIM_NUM>(params.valueCacheStride, vcStrideArr);

    int64_t smShapeArr[SLOT_MAPPING_DIM_NUM];
    int64_t smStrideArr[SLOT_MAPPING_DIM_NUM];
    int64_t smDim;
    GetTensorInfo(context, INPUT_SLOT_MAPPING_IDX, smShapeArr, smStrideArr, smDim);
    if (smDim != SLOT_MAPPING_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "slot_mapping", std::to_string(smDim).c_str(), "1");
        return ge::GRAPH_FAILED;
    }
    CopyStrides<SLOT_MAPPING_DIM_NUM>(params.slotMappingStride, smStrideArr);

    int64_t ksShapeArr[KEY_SCALE_DIM_NUM];
    int64_t ksStrideArr[KEY_SCALE_DIM_NUM];
    int64_t ksDim;
    GetTensorInfo(context, INPUT_KEY_SCALE_IDX, ksShapeArr, ksStrideArr, ksDim);
    if (ksDim != KEY_SCALE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "key_scale", std::to_string(ksDim).c_str(), "2");
        return ge::GRAPH_FAILED;
    }
    CopyStrides<KEY_SCALE_DIM_NUM>(params.keyScaleStride, ksStrideArr);

    int64_t kscShapeArr[KEY_SCALE_CACHE_DIM_NUM];
    int64_t kscStrideArr[KEY_SCALE_CACHE_DIM_NUM];
    int64_t kscDim;
    GetTensorInfo(context, INPUT_KEY_SCALE_CACHE_IDX, kscShapeArr, kscStrideArr, kscDim);
    if (kscDim != KEY_SCALE_CACHE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("ScatterPaKvCacheWithKScale", "key_scale_cache", std::to_string(kscDim).c_str(),
                                     "4");
        return ge::GRAPH_FAILED;
    }
    CopyStrides<KEY_SCALE_CACHE_DIM_NUM>(params.keyScaleCacheStride, kscStrideArr);

    params.maxSlot = params.numBlocks * params.blockSize;

    return ge::GRAPH_SUCCESS;
}

// ==================== fill tiling data ====================

static void FillTilingData(ScatterPaKvCacheWithKScaleTilingData* tiling, const TensorParams& params, int64_t coreNum,
                           uint64_t ubSize, int64_t tplMode)
{
    tiling->numTokens = params.numTokens;
    tiling->numHead = params.numHead;
    tiling->kHeadSize = params.kHeadSize;
    tiling->vHeadSize = params.vHeadSize;
    tiling->numBlocks = params.numBlocks;
    tiling->blockSize = params.blockSize;
    tiling->maxSlot = params.maxSlot;

    CopyStrides<KEY_DIM_NUM>(tiling->keyStride, params.keyStride);
    CopyStrides<KEY_DIM_NUM>(tiling->valueStride, params.valueStride);
    CopyStrides<KEY_CACHE_DIM_NUM>(tiling->keyCacheStride, params.keyCacheStride);
    CopyStrides<VALUE_CACHE_DIM_NUM>(tiling->valueCacheStride, params.valueCacheStride);
    CopyStrides<SLOT_MAPPING_DIM_NUM>(tiling->slotMappingStride, params.slotMappingStride);
    CopyStrides<KEY_SCALE_DIM_NUM>(tiling->keyScaleStride, params.keyScaleStride);
    CopyStrides<KEY_SCALE_CACHE_DIM_NUM>(tiling->keyScaleCacheStride, params.keyScaleCacheStride);

    // SIMT: VF 网格循环所需核数
    int64_t needCoreNum = std::max(MIN_CORE_NUM, std::min(params.numTokens, coreNum));
    tiling->needCoreNum = needCoreNum;

    // SIMD: 按 num_tokens 分核（参考 TemplateNormal）
    int64_t blockFactor = (params.numTokens > 0) ? CeilDiv(params.numTokens, coreNum) : 0;
    int64_t usedCoreNum =
        (params.numTokens > 0) ? std::min(CeilDiv(params.numTokens, blockFactor), coreNum) : MIN_CORE_NUM;
    int64_t tailBlockFactor = params.numTokens - blockFactor * (usedCoreNum - 1);
    tiling->blockFactor = blockFactor;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tailBlockFactor = tailBlockFactor;

    // SIMD kernel 的 TPipe 可用 UB：纯 AIV kernel，无需预留 DCACHE
    if (tplMode == SCATTER_KV_CACHE_TPL_SIMD) {
        tiling->ubSize = static_cast<int64_t>(ubSize) - STATIC_UB_ESTIMATE;
    } else {
        tiling->ubSize = static_cast<int64_t>(ubSize) - DCACHE_SIZE - STATIC_UB_ESTIMATE;
    }
}

// ==================== dump tiling info ====================

static void DumpTilingInfo(gert::TilingContext* context, const ScatterPaKvCacheWithKScaleTilingData* tiling,
                           int64_t tplMode)
{
    OP_LOGI(context, "==================== ScatterPaKvCacheWithKScale Tiling Info ====================");
    OP_LOGI(context, "Basic Parameters:");
    OP_LOGI(context, "  numTokens      : %lld", tiling->numTokens);
    OP_LOGI(context, "  numHead        : %lld", tiling->numHead);
    OP_LOGI(context, "  kHeadSize      : %lld", tiling->kHeadSize);
    OP_LOGI(context, "  vHeadSize      : %lld", tiling->vHeadSize);
    OP_LOGI(context, "  numBlocks      : %lld", tiling->numBlocks);
    OP_LOGI(context, "  blockSize      : %lld", tiling->blockSize);
    OP_LOGI(context, "  maxSlot        : %lld", tiling->maxSlot);
    OP_LOGI(context, "  needCoreNum    : %lld", tiling->needCoreNum);
    OP_LOGI(context, "  usedCoreNum    : %lld", tiling->usedCoreNum);
    OP_LOGI(context, "  blockFactor    : %lld", tiling->blockFactor);
    OP_LOGI(context, "  tailBlockFactor: %lld", tiling->tailBlockFactor);
    OP_LOGI(context, "  ubSize         : %lld", tiling->ubSize);

    OP_LOGI(context, "Strides:");
    OP_LOGI(context, "  keyStride      : [%lld, %lld, %lld]", tiling->keyStride[0], tiling->keyStride[1],
            tiling->keyStride[2]);
    OP_LOGI(context, "  valueStride    : [%lld, %lld, %lld]", tiling->valueStride[0], tiling->valueStride[1],
            tiling->valueStride[2]);
    OP_LOGI(context, "  keyCacheStride : [%lld, %lld, %lld, %lld]", tiling->keyCacheStride[0],
            tiling->keyCacheStride[1], tiling->keyCacheStride[2], tiling->keyCacheStride[3]);
    OP_LOGI(context, "  valueCacheStride: [%lld, %lld, %lld, %lld]", tiling->valueCacheStride[0],
            tiling->valueCacheStride[1], tiling->valueCacheStride[2], tiling->valueCacheStride[3]);
    OP_LOGI(context, "  slotMappingStride: [%lld]", tiling->slotMappingStride[0]);
    OP_LOGI(context, "  keyScaleStride : [%lld, %lld]", tiling->keyScaleStride[0], tiling->keyScaleStride[1]);
    OP_LOGI(context, "  keyScaleCacheStride: [%lld, %lld, %lld, %lld]", tiling->keyScaleCacheStride[0],
            tiling->keyScaleCacheStride[1], tiling->keyScaleCacheStride[2], tiling->keyScaleCacheStride[3]);

    OP_LOGI(context, "Scene Mode: %s",
            (tiling->kHeadSize == tiling->vHeadSize) ? "SPECIALIZED (kHeadSize == vHeadSize)" :
                                                       "GENERALIZED (kHeadSize != vHeadSize)");
    OP_LOGI(context, "Template Mode: %s",
            (tplMode == SCATTER_KV_CACHE_TPL_SIMD) ? "SIMD (last axis contiguous)" : "SIMT");
    OP_LOGI(context, "===============================================================================");
}

// ==================== template mode selection ====================

// SIMD 模板 DataCopyPad 多行搬运要求 headSize 不小于 128B
constexpr int64_t SIMD_MIN_HEAD_SIZE_BYTES = 128;

// 算子仅支持 FP8（1 字节）key/value，返回 0 表示未知 dtype（走 SIMT 兜底）
static int64_t GetKeyDtypeSize(gert::TilingContext* context)
{
    auto keyDesc = context->GetInputDesc(INPUT_KEY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    ge::DataType keyDtype = keyDesc->GetDataType();
    if (keyDtype == ge::DT_FLOAT8_E5M2 || keyDtype == ge::DT_FLOAT8_E4M3FN) {
        return 1;
    }
    return 0;
}

// SIMD 模板使用 DataCopyPad 多行搬运，要求 headSize 不小于 128B、key/value/cache 尾轴连续且头间无重叠，
// key_scale 走 DataCopyPad 多块拷贝，要求尾轴 stride 为正；
// PCIe through 场景下 tensor 布局由框架保证，直接选择 simd 模板
static bool IsSimdCapable(gert::TilingContext* context, const TensorParams& params, int64_t dtypeSize)
{
    if (dtypeSize <= 0) {
        return false;
    }
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    if (context->GetPcieThroughFlag()) {
        return true;
    }
#endif
    bool headSizeValid = params.kHeadSize * dtypeSize >= SIMD_MIN_HEAD_SIZE_BYTES &&
                         params.vHeadSize * dtypeSize >= SIMD_MIN_HEAD_SIZE_BYTES;
    bool lastAxisContiguous = params.keyStride[DIM_HEAD_SIZE] == 1 && params.valueStride[DIM_HEAD_SIZE] == 1 &&
                              params.keyCacheStride[DIM_CACHE_HEAD_SIZE] == 1 &&
                              params.valueCacheStride[DIM_CACHE_HEAD_SIZE] == 1;
    bool headStrideValid = params.keyStride[DIM_HEAD] >= params.kHeadSize &&
                           params.valueStride[DIM_HEAD] >= params.vHeadSize &&
                           params.keyCacheStride[DIM_CACHE_HEAD] >= params.kHeadSize &&
                           params.valueCacheStride[DIM_CACHE_HEAD] >= params.vHeadSize;
    bool scaleStrideValid =
        params.keyScaleStride[DIM_SCALE_HEAD] >= 1 && params.keyScaleCacheStride[DIM_SCALE_CACHE_HEAD] >= 1;
    return headSizeValid && lastAxisContiguous && headStrideValid && scaleStrideValid;
}

// ==================== set tiling key ====================

static void SetTilingKeyByScene(gert::TilingContext* context, int64_t kHeadSize, int64_t vHeadSize, int64_t tplMode)
{
    uint64_t schMode =
        (kHeadSize == vHeadSize) ? SCATTER_KV_CACHE_SCENE_SPECIALIZED : SCATTER_KV_CACHE_SCENE_GENERALIZED;
    uint64_t tilingKey = GET_TPL_TILING_KEY(schMode, static_cast<uint64_t>(tplMode));
    context->SetTilingKey(tilingKey);
}

// ==================== main tiling ====================

static ge::graphStatus ScatterPaKvCacheWithKScaleTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetPlatformInfo error");
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetWorkspaceSize error");
        return ge::GRAPH_FAILED;
    }
    if (ValidateDtype(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "ValidateDtype error");
        return ge::GRAPH_FAILED;
    }

    TensorParams params;
    if (ExtractTensorParams(context, params) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "ExtractTensorParams error");
        return ge::GRAPH_FAILED;
    }

    if (ValidateShape(context, params.numTokens, params.numHead, params.kHeadSize, params.vHeadSize, params.numBlocks,
                      params.blockSize) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "ValidateShape error");
        return ge::GRAPH_FAILED;
    }

    int64_t keyDtypeSize = GetKeyDtypeSize(context);
    int64_t tplMode =
        IsSimdCapable(context, params, keyDtypeSize) ? SCATTER_KV_CACHE_TPL_SIMD : SCATTER_KV_CACHE_TPL_SIMT;

    ScatterPaKvCacheWithKScaleTilingData* tiling = context->GetTilingData<ScatterPaKvCacheWithKScaleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (memset_s(tiling, sizeof(ScatterPaKvCacheWithKScaleTilingData), 0,
                 sizeof(ScatterPaKvCacheWithKScaleTilingData)) != EOK) {
        OP_LOGE(context, "memset tiling data error");
        return ge::GRAPH_FAILED;
    }

    FillTilingData(tiling, params, coreNum, ubSize, tplMode);
    context->SetBlockDim((tplMode == SCATTER_KV_CACHE_TPL_SIMD) ? tiling->usedCoreNum : tiling->needCoreNum);

    DumpTilingInfo(context, tiling, tplMode);

    if (ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ScatterPaKvCacheWithKScale", "ubSize", std::to_string(ubSize).c_str(),
                                              "ubSize must be > DCACHE_SIZE + STATIC_UB_ESTIMATE");
        return ge::GRAPH_FAILED;
    }
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "SetLocalMemorySize failed");
        return ge::GRAPH_FAILED;
    }

    SetTilingKeyByScene(context, params.kHeadSize, params.vHeadSize, tplMode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForScatterPaKvCacheWithKScale([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterPaKvCacheWithKScale)
    .Tiling(ScatterPaKvCacheWithKScaleTilingFunc)
    .TilingParse<ScatterPaKvCacheWithKScaleCompileInfo>(TilingParseForScatterPaKvCacheWithKScale);

} // namespace optiling
