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
 * \file gather_pa_kv_cache_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include <array>
#include "log/log.h"
#include "platform/platform_info_def.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "gather_pa_kv_cache_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr int64_t INDEX_ATTR_CACHE_MODE = 0;
constexpr int64_t INDEX_ATTR_IS_SEQ_LENS_CUMSUM = 1;
constexpr int64_t INDEX_INPUT_KEY_CACHE = 0;
constexpr int64_t INDEX_INPUT_VALUE_CACHE = 1;
constexpr int64_t INDEX_INPUT_BLOCK_TABLES = 2;
constexpr int64_t INDEX_INPUT_SEQ_LENS = 3;
constexpr int64_t INDEX_INPUT_KEY = 4;
constexpr int64_t INDEX_INPUT_VALUE = 5;
constexpr int64_t INDEX_OPT_INPUT_SEQ_OFFSETS = 6;
constexpr int64_t INDEX_OUTPUT_KEY = 0;
constexpr int64_t INDEX_OUTPUT_VALUE = 1;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t WORKSPACE_SIZE = 32;
constexpr uint32_t UB_REVERSE = 1024;
constexpr uint32_t SEQ_LEN_ACC_SPACE = 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;

static const std::map<ge::DataType, uint32_t> tilingDataTypeByteTable = {
    {ge::DT_INT64, 8},    {ge::DT_INT32, 4},       {ge::DT_UINT32, 4},        {ge::DT_FLOAT, 4}, {ge::DT_INT16, 2},
    {ge::DT_UINT16, 2},   {ge::DT_FLOAT16, 2},     {ge::DT_BF16, 2},          {ge::DT_INT8, 1},  {ge::DT_UINT8, 1},
    {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E5M2, 1}, {ge::DT_FLOAT8_E4M3FN, 1},
};

static const std::set<ge::DataType> KV_SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT16, ge::DT_UINT16,   ge::DT_INT8,        ge::DT_UINT8};

static const std::set<ge::DataType> INDEX_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_INT64};

#define FORMAT_KEY_VALUE_NOT_SUPPORTED \
    "%s dtype only support [float32, float16, bf16, hf8, fp8_e5m2, fp8_e4m3fn, int32, uint32, int16, uint16, int8, " \
    "uint8], please check."

ge::graphStatus GatherPaKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const GatherPaKvCacheCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetAttrs()
{
    auto* attrs = context_->GetAttrs();
    cacheMode_ = std::string(attrs->GetAttrPointer<char>(INDEX_ATTR_CACHE_MODE));
    if (cacheMode_.empty()) {
        cacheMode_ = "Norm";
    } else {
        std::set<std::string> checkList = {"Norm", "PA_NZ"};
        OP_CHECK_IF(checkList.find(cacheMode_) == checkList.end(),
                    OP_LOGE(context_, "[attr]cache_mode only support ['Norm', 'PA_NZ'], please check."),
                    return ge::GRAPH_FAILED);
    }
    isCacheModeNorm_ = (cacheMode_ == "Norm");

    const bool* isSeqLensCumsumPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_IS_SEQ_LENS_CUMSUM);
    if (isSeqLensCumsumPtr == nullptr) {
        isSeqLenCumSum_ = true;
    } else {
        isSeqLenCumSum_ = *isSeqLensCumsumPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputKeyCache()
{
    auto kCacheDesc = context_->GetInputDesc(INDEX_INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    ge::DataType kCacheDType = kCacheDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(kCacheDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, FORMAT_KEY_VALUE_NOT_SUPPORTED, "key_cache"), return ge::GRAPH_FAILED);

    uint32_t kCacheDTypeByteSize = tilingDataTypeByteTable.find(kCacheDType)->second;
    keyByteSize_ = kCacheDTypeByteSize;

    // 非连续场景下输入可能为view，必须通过GetTensorInfo获取逻辑shape与stride。
    // 不能直接用GetStorageShape：view的物理shape会退化(如降为1维)，导致维度校验误判。
    OP_CHECK_IF(GetTensorInfo(kCacheShape_, kCacheStride_, INDEX_INPUT_KEY_CACHE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "get key_cache tensor info failed."), return ge::GRAPH_FAILED);
    kCacheShape_ = EnsureNotScalar(kCacheShape_);
    kCacheDimNum_ = kCacheShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(kCacheDimNum_ != 4,
                OP_LOGE(context_, "key_cache dimension must be 4, but got %zu. Please check.", kCacheDimNum_),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < kCacheDimNum_; i++) {
        OP_CHECK_IF(kCacheShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "key_cache.shape[%zu] must be positive, Please check.", i),
                    return ge::GRAPH_FAILED);
    }

    numBlocks_ = kCacheShape_.GetDim(0);
    blockSize_ = kCacheShape_.GetDim(1);
    // 当数据格式为NZ时
    if (!isCacheModeNorm_) {
        OP_CHECK_IF(kCacheShape_.GetDim(kCacheDimNum_ - 1) * keyByteSize_ != BLOCK_SIZE,
                    OP_LOGE(context_, "key_cache.shape[3](%ld) must align and equal to 32B, please check.",
                            kCacheShape_.GetDim(kCacheDimNum_ - 1)),
                    return ge::GRAPH_FAILED);
        // 2 is blockSize_ dim
        blockSize_ = kCacheShape_.GetDim(2);
    }

    // ===== 非连续支持: stride信息已在前面GetTensorInfo中获取 =====
    if (kCacheStride_.GetDimNum() > 0) {
        // 判断连续性 (size为1的轴stride不影响连续性)
        int64_t contigStride = 1;
        isKCacheContiguous_ = true;
        for (int i = static_cast<int>(kCacheDimNum_) - 1; i >= 0; i--) {
            if (kCacheShape_.GetDim(i) != 1 && kCacheStride_.GetStride(i) != contigStride) {
                isKCacheContiguous_ = false;
            }
            contigStride *= kCacheShape_.GetDim(i);
        }

        if (!isKCacheContiguous_) {
            if (isCacheModeNorm_) {
                // ND模式: 判断具体哪些轴非连续
                int64_t expectedStride1 = kCacheShape_.GetDim(2) * kCacheShape_.GetDim(3);
                int64_t expectedStride2 = kCacheShape_.GetDim(3);
                isKCacheSlotNonContig_ = (kCacheStride_.GetStride(1) != expectedStride1);
                isKCacheHeadNonContig_ = (kCacheStride_.GetStride(2) != expectedStride2);
            } else {
                // NZ模式: 仅支持dim0(blockNum)非连续
                int64_t expectedStride1 = kCacheShape_.GetDim(2) * kCacheShape_.GetDim(3);
                int64_t expectedStride2 = kCacheShape_.GetDim(3);
            }
        }
    } else {
        isKCacheContiguous_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputValueCache()
{
    auto vCacheDesc = context_->GetInputDesc(INDEX_INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    ge::DataType vCacheDType = vCacheDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(vCacheDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, FORMAT_KEY_VALUE_NOT_SUPPORTED, "value_cache"), return ge::GRAPH_FAILED);
    uint32_t vCacheDTypeByteSize = tilingDataTypeByteTable.find(vCacheDType)->second;
    valueByteSize_ = vCacheDTypeByteSize;

    // 非连续场景下value_cache可能为view，必须通过GetTensorInfo获取逻辑shape与stride。
    OP_CHECK_IF(GetTensorInfo(vCacheShape_, vCacheStride_, INDEX_INPUT_VALUE_CACHE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "get value_cache tensor info failed."), return ge::GRAPH_FAILED);
    vCacheShape_ = EnsureNotScalar(vCacheShape_);
    vCacheDimNum_ = vCacheShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(vCacheDimNum_ != 4,
                OP_LOGE(context_, "value_cache dimension must be 4, but got %zu. Please check.", vCacheDimNum_),
                return ge::GRAPH_FAILED);

    // 当数据格式为NZ时，需要检查尾轴是否与32B对齐。kcache和vcache除第1维，其他轴必须相等。
    // 当数据格式为ND时，kcache和vcache的shape的非尾轴必须相等。
    if (!isCacheModeNorm_) {
        OP_CHECK_IF(vCacheShape_.GetDim(vCacheDimNum_ - 1) * valueByteSize_ != BLOCK_SIZE,
                    OP_LOGE(context_, "value_cache last dimension must align and equal to 32B, please check."),
                    return ge::GRAPH_FAILED);
    }
    // key_cache与value_cache的num_blocks(dim0)数值无需相等: kernel用同一blockId分别乘
    // 各自的stride索引, 不依赖dim0相等(与scatter Norm模式CheckNormal行为对齐)。
    // 此处仅校验非尾轴中除dim0外的维度(ND: dim1/dim2; NZ: dim2, 跳过dim1)。
    for (size_t i = 1; i < vCacheDimNum_ - 1; i++) {
        if (!isCacheModeNorm_ && i == DIM_ONE) {
            continue;
        }
        OP_CHECK_IF(vCacheShape_.GetDim(i) != kCacheShape_.GetDim(i),
                    OP_LOGE(context_,
                            "value_cache.shape[%zu] %ld is not equal to key_cache.shape[%zu] %ld, "
                            "please check.",
                            i, vCacheShape_.GetDim(i), i, kCacheShape_.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    // ===== 非连续支持: valueCache stride信息已在前面GetTensorInfo中获取 =====
    if (vCacheStride_.GetDimNum() > 0) {
        int64_t contigStride = 1;
        isVCacheContiguous_ = true;
        for (int i = static_cast<int>(vCacheDimNum_) - 1; i >= 0; i--) {
            if (vCacheShape_.GetDim(i) != 1 && vCacheStride_.GetStride(i) != contigStride) {
                isVCacheContiguous_ = false;
            }
            contigStride *= vCacheShape_.GetDim(i);
        }

        if (!isVCacheContiguous_) {
            if (isCacheModeNorm_) {
                int64_t expectedStride1 = vCacheShape_.GetDim(2) * vCacheShape_.GetDim(3);
                int64_t expectedStride2 = vCacheShape_.GetDim(3);
                isVCacheSlotNonContig_ = (vCacheStride_.GetStride(1) != expectedStride1);
                isVCacheHeadNonContig_ = (vCacheStride_.GetStride(2) != expectedStride2);
            } else {
                // NZ模式: 仅支持dim0非连续
                int64_t expectedStride1 = vCacheShape_.GetDim(2) * vCacheShape_.GetDim(3);
                int64_t expectedStride2 = vCacheShape_.GetDim(3);
            }
        }
    } else {
        isVCacheContiguous_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputBlockTables()
{
    auto blockTablesDesc = context_->GetInputDesc(INDEX_INPUT_BLOCK_TABLES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTablesDesc);
    ge::DataType blockTablesDType = blockTablesDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(blockTablesDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "block_tables dtype only support [int32, int64], please check."),
                return ge::GRAPH_FAILED);
    indexByteSize_ = tilingDataTypeByteTable.find(blockTablesDType)->second;

    auto blockTableStoreShape = context_->GetInputShape(INDEX_INPUT_BLOCK_TABLES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTableStoreShape);
    blockTableShape_ = EnsureNotScalar(blockTableStoreShape->GetStorageShape());
    size_t blockTableDimNum = blockTableShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(blockTableDimNum != 2,
                OP_LOGE(context_, "block_tables dimension must be 2, but got %zu. Please check.", blockTableDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < blockTableDimNum; i++) {
        OP_CHECK_IF(blockTableShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "block_tables.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }
    batchCount_ = blockTableShape_.GetDim(0);
    blockTableWidth_ = blockTableShape_.GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputSeqLens()
{
    auto seqLensDesc = context_->GetInputDesc(INDEX_INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLensDesc);
    ge::DataType seqLensDType = seqLensDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(seqLensDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "seq_lens dtype only support [int32, int64], please check."),
                return ge::GRAPH_FAILED);

    auto seqLenStoreShape = context_->GetInputShape(INDEX_INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLenStoreShape);
    seqLenShape_ = EnsureNotScalar(seqLenStoreShape->GetStorageShape());
    uint32_t seqLenDimNum = seqLenShape_.GetDimNum();

    // 检查形状是否合法
    OP_CHECK_IF(seqLenDimNum != 1,
                OP_LOGE(context_, "seq_lens dimension must be 1, but got %u. Please check.", seqLenDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < seqLenDimNum; i++) {
        OP_CHECK_IF(seqLenShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "seq_lens.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }
    // seq_lens的第0维要和block_tables的第0维相等
    if (isSeqLenCumSum_) {
        OP_CHECK_IF(seqLenShape_.GetDim(0) != (blockTableShape_.GetDim(0) + 1),
                    OP_LOGE(context_, "When [attr]is_seq_lens_cumsum is true, seq_lens.shape[0] must equal "
                                      "block_tables.shape[0] + 1, please check."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(seqLenShape_.GetDim(0) != blockTableShape_.GetDim(0),
                    OP_LOGE(context_, "When [attr]is_seq_lens_cumsum is false, seq_lens.shape[0] must equal "
                                      "block_tables.shape[0], please check."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputOutputKey()
{
    auto keyDesc = context_->GetInputDesc(INDEX_INPUT_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    ge::DataType keyDType = keyDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(keyDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, FORMAT_KEY_VALUE_NOT_SUPPORTED, "key"), return ge::GRAPH_FAILED);

    // 非连续场景下key可能为view，必须通过GetTensorInfo获取逻辑shape与stride。
    OP_CHECK_IF(GetTensorInfo(keyShape_, keyOutStride_, INDEX_INPUT_KEY) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "get key tensor info failed."), return ge::GRAPH_FAILED);
    keyShape_ = EnsureNotScalar(keyShape_);
    size_t keyDimNum = keyShape_.GetDimNum();

    uint32_t keyDimExpect = (isCacheModeNorm_) ? uint32_t(DIM_THREE) : uint32_t(DIM_TWO);
    OP_CHECK_IF(keyDimNum != keyDimExpect,
                OP_LOGE(context_, "key dimension must be %u, but got %zu. Please check.", keyDimExpect, keyDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < keyDimNum; i++) {
        OP_CHECK_IF(keyShape_.GetDim(i) <= 0, OP_LOGE(context_, "key.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }

    numTokens_ = keyShape_.GetDim(0);

    if (isCacheModeNorm_) {
        // ND
        hiddenSizeK_ = keyShape_.GetDim(DIM_ONE) * keyShape_.GetDim(DIM_TWO) * keyByteSize_;
        numHeadsK_ = keyShape_.GetDim(DIM_ONE);
        for (size_t i = 1; i < keyDimNum; i++) {
            OP_CHECK_IF(keyShape_.GetDim(i) != kCacheShape_.GetDim(i + 1),
                        OP_LOGE(context_,
                                "key.shape[%zu] %ld is not equal to key_cache.shape[%zu] %ld, "
                                "please check.",
                                i, keyShape_.GetDim(i), i + 1, kCacheShape_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // NZ
        hiddenSizeK_ = keyShape_.GetDim(DIM_ONE) * keyByteSize_;
        numHeadsK_ = 0; // NZ模式不需要numHeads
        uint64_t kCacheShape1 = kCacheShape_.GetDim(DIM_ONE);
        uint64_t kCacheShape3 = kCacheShape_.GetDim(DIM_THREE) * keyByteSize_;
        uint64_t hiddenSizeKCache = kCacheShape1 * kCacheShape3;
        OP_CHECK_IF(hiddenSizeK_ != hiddenSizeKCache,
                    OP_LOGE(context_,
                            "key.shape[1] (%lu) is not equal to "
                            "the product of key_cache.shape[1] and key_cache.shape[3] (%lu * %lu = %lu), "
                            "please check.",
                            hiddenSizeK_, kCacheShape1, kCacheShape3, hiddenSizeKCache),
                    return ge::GRAPH_FAILED);
    }

    // ===== 非连续支持: key输出stride已在前面GetTensorInfo中获取 (ND和NZ模式都支持) =====
    if (keyOutStride_.GetDimNum() > 0) {
        // 判断连续性 (size为1的轴stride不影响连续性)
        int64_t contigStride = 1;
        isKeyOutContiguous_ = true;
        for (int i = static_cast<int>(keyDimNum) - 1; i >= 0; i--) {
            if (keyShape_.GetDim(i) != 1 && keyOutStride_.GetStride(i) != contigStride) {
                isKeyOutContiguous_ = false;
            }
            contigStride *= keyShape_.GetDim(i);
        }
        // ND模式(3维): 单独判断head(dim1)轴是否非连续, 用于kernel决定是否按head拆分散写。
        // dim1连续stride = head_size = keyShape[2]。
        if (isCacheModeNorm_ && keyDimNum == uint32_t(DIM_THREE) && keyShape_.GetDim(DIM_ONE) != 1) {
            isKeyOutHeadNonContig_ = (keyOutStride_.GetStride(DIM_ONE) != keyShape_.GetDim(DIM_TWO));
        }
    } else {
        isKeyOutContiguous_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputOutputValue()
{
    auto valueDesc = context_->GetInputDesc(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    ge::DataType valueDType = valueDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(valueDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, FORMAT_KEY_VALUE_NOT_SUPPORTED, "value"), return ge::GRAPH_FAILED);

    // 非连续场景下value可能为view，必须通过GetTensorInfo获取逻辑shape与stride。
    OP_CHECK_IF(GetTensorInfo(valueShape_, valueOutStride_, INDEX_INPUT_VALUE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "get value tensor info failed."), return ge::GRAPH_FAILED);
    valueShape_ = EnsureNotScalar(valueShape_);
    size_t valueDimNum = valueShape_.GetDimNum();

    // 检查形状是否合法
    uint32_t valueDimExpect = (isCacheModeNorm_) ? uint32_t(DIM_THREE) : uint32_t(DIM_TWO);
    OP_CHECK_IF(
        valueDimNum != valueDimExpect,
        OP_LOGE(context_, "value dimension must be %u, but got %zu. Please check.", valueDimExpect, valueDimNum),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < valueDimNum - 1; i++) {
        OP_CHECK_IF(valueShape_.GetDim(i) != keyShape_.GetDim(i),
                    OP_LOGE(context_, "value.shape[%zu] %ld is not equal to key.shape[%zu] %ld, please check.", i,
                            valueShape_.GetDim(i), i, keyShape_.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    if (isCacheModeNorm_) {
        // ND
        hiddenSizeV_ = valueShape_.GetDim(DIM_ONE) * valueShape_.GetDim(DIM_TWO) * valueByteSize_;
        numHeadsV_ = valueShape_.GetDim(DIM_ONE);
        for (size_t i = 1; i < valueDimNum; i++) {
            OP_CHECK_IF(valueShape_.GetDim(i) != vCacheShape_.GetDim(i + 1),
                        OP_LOGE(context_,
                                "value.shape[%zu] %ld is not equal to value_cache.shape[%zu] %ld, "
                                "please check.",
                                i, valueShape_.GetDim(i), i + 1, vCacheShape_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // NZ
        hiddenSizeV_ = valueShape_.GetDim(DIM_ONE) * valueByteSize_;
        numHeadsV_ = 0;
        uint64_t vCacheShape1 = vCacheShape_.GetDim(DIM_ONE);
        uint64_t vCacheShape3 = vCacheShape_.GetDim(DIM_THREE) * valueByteSize_;
        uint64_t hiddenSizeVCache = vCacheShape1 * vCacheShape3;
        OP_CHECK_IF(hiddenSizeV_ != hiddenSizeVCache,
                    OP_LOGE(context_,
                            "value.shape[1] (%lu) is not equal to "
                            "the product of value_cache.shape[1] and value_cache.shape[3] (%lu * %lu = %lu), "
                            "please check.",
                            hiddenSizeV_, vCacheShape1, vCacheShape3, hiddenSizeVCache),
                    return ge::GRAPH_FAILED);
    }

    // ===== 非连续支持: value输出stride已在前面GetTensorInfo中获取 (ND和NZ模式都支持) =====
    if (valueOutStride_.GetDimNum() > 0) {
        int64_t contigStride = 1;
        isValueOutContiguous_ = true;
        for (int i = static_cast<int>(valueDimNum) - 1; i >= 0; i--) {
            if (valueShape_.GetDim(i) != 1 && valueOutStride_.GetStride(i) != contigStride) {
                isValueOutContiguous_ = false;
            }
            contigStride *= valueShape_.GetDim(i);
        }
        // ND模式(3维): 单独判断head(dim1)轴是否非连续。dim1连续stride = head_size = valueShape[2]。
        if (isCacheModeNorm_ && valueDimNum == uint32_t(DIM_THREE) && valueShape_.GetDim(DIM_ONE) != 1) {
            isValueOutHeadNonContig_ = (valueOutStride_.GetStride(DIM_ONE) != valueShape_.GetDim(DIM_TWO));
        }
    } else {
        isValueOutContiguous_ = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputSeqOffset()
{
    auto seqOffsetDesc = context_->GetOptionalInputDesc(INDEX_OPT_INPUT_SEQ_OFFSETS);
    if (seqOffsetDesc == nullptr) {
        OP_LOGD(context_, "[attr]seq_offset is not been set.");
        hasSeqOffset_ = false;
        return ge::GRAPH_SUCCESS;
    }
    hasSeqOffset_ = true;
    ge::DataType seqOffsetDType = seqOffsetDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(seqOffsetDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "seq_offset dtype only support [int64, int32], please check."),
                return ge::GRAPH_FAILED);

    auto seqOffsetStoreShape = context_->GetOptionalInputShape(INDEX_OPT_INPUT_SEQ_OFFSETS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqOffsetStoreShape);
    seqOffsetShape_ = EnsureNotScalar(seqOffsetStoreShape->GetStorageShape());
    uint32_t seqOffsetDimNum = seqOffsetShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(seqOffsetDimNum != 1,
                OP_LOGE(context_, "seq_offset dimension must be 1, but got %u. Please check.", seqOffsetDimNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(seqOffsetShape_.GetDim(0) != blockTableShape_.GetDim(0),
                OP_LOGE(context_, "seqOffset.shape[%zu] %ld is not equal to block_tables.shape[%zu] %ld, please check.",
                        size_t(0), seqOffsetShape_.GetDim(0), size_t(0), blockTableShape_.GetDim(0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("GatherPaKvCacheTiling", "context is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputKeyCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input key_cache failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputValueCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input value_cache failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputBlockTables() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input block_tables failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqLens() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input seq_lens failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputKey() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output key failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output value failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqOffset() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get optional input seq_offset failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool GatherPaKvCacheTiling::IsCapable()
{
    return true;
}

ge::graphStatus GatherPaKvCacheTiling::DoOpTiling()
{
    // 分核策略(按连续性隔离, 连续路径零改动):
    //   ND非连续(cache任一轴 或 ref任一轴非连续): 按"全局block轮询"分核, kernel用 globalBlk % needCoreNum 分配,
    //             故直接用满所有核。解决batch少(如batchCount=3/6)时按batch分核只用极少核的瓶颈。
    //   其余(ND连续 / NZ): 维持按batch分核, 行为完全不变。
    bool isNdNonContig = isCacheModeNorm_ && (!isKCacheContiguous_ || !isVCacheContiguous_ || !isKeyOutContiguous_ ||
                                              !isValueOutContiguous_);
    int64_t batchPerCore = Ops::Base::CeilDiv(batchCount_, coreNum_);
    if (isNdNonContig) {
        needCoreNum_ = static_cast<uint32_t>(coreNum_);
    } else {
        needCoreNum_ = static_cast<uint32_t>(std::min(Ops::Base::CeilDiv(batchCount_, batchPerCore), coreNum_));
    }
    // uint32_t batchTail = batchCount_ - batchPerCore * (needCoreNum - 1);
    uint32_t tileBase = BLOCK_SIZE;

    // 计算UB最大能放下的KV Cache大小
    uint32_t seqLenAccumSize = 1024;
    uint32_t factor = (ubSize_ - UB_REVERSE - seqLenAccumSize * DOUBLE_BUFFER * indexByteSize_ - BLOCK_SIZE) /
                      (tileBase * DOUBLE_BUFFER);
    uint64_t cacheBlockK = static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(hiddenSizeK_);
    uint64_t cacheBlockV = static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(hiddenSizeV_);
    uint64_t maxUbHiddenSizeK =
        std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockK); // 最大不超过1个cacheBlock
    uint64_t maxUbHiddenSizeV =
        std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockV); // 最大不超过1个cacheBlock
    uint64_t maxUbHiddenSize = std::max(maxUbHiddenSizeK, maxUbHiddenSizeV);
    maxUbHiddenSize =
        Ops::Base::CeilAlign(maxUbHiddenSize, static_cast<uint64_t>(tileBase)); // 保证maxUbHiddenSize和32B对齐

    // 动态调整: 如果有多余空间，就用于累加和的计算
    if (maxUbHiddenSizeK == cacheBlockK || maxUbHiddenSizeV == cacheBlockV) {
        uint32_t spareBuffer = ubSize_ - UB_REVERSE - maxUbHiddenSize * DOUBLE_BUFFER - BLOCK_SIZE;
        seqLenAccumSize = Ops::Base::CeilDiv(spareBuffer / DOUBLE_BUFFER, BLOCK_SIZE) * BLOCK_SIZE / indexByteSize_;
    }

    hiddenSizeK_ /= keyByteSize_;
    hiddenSizeV_ /= valueByteSize_;
    maxUbHiddenSizeK /= keyByteSize_;
    maxUbHiddenSizeV /= valueByteSize_;
    maxUbHiddenSize /= keyByteSize_;
    // 配置tilingdata
    tilingData_.set_batchCount(batchCount_);
    tilingData_.set_batchPerCore(batchPerCore);
    tilingData_.set_needCoreNum(needCoreNum_);
    tilingData_.set_seqLenAccumSize(seqLenAccumSize);
    tilingData_.set_blockTableWidth(blockTableWidth_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_hiddenSizeK(hiddenSizeK_);
    tilingData_.set_hiddenSizeV(hiddenSizeV_);
    tilingData_.set_numTokens(numTokens_);
    tilingData_.set_maxUbHiddenSizeK(maxUbHiddenSizeK);
    tilingData_.set_maxUbHiddenSizeV(maxUbHiddenSizeV);
    tilingData_.set_maxUbHiddenSize(maxUbHiddenSize);
    tilingData_.set_kvCacheBlockSize(blockSize_);

    // ===== 非连续支持: 设置stride字段到TilingData =====
    uint32_t nonContigFlag = 0;
    if (!isKCacheContiguous_) {
        nonContigFlag |= (1 << 0);
        if (isKCacheSlotNonContig_)
            nonContigFlag |= (1 << 4);
        if (isKCacheHeadNonContig_)
            nonContigFlag |= (1 << 5);
    }
    if (!isVCacheContiguous_) {
        nonContigFlag |= (1 << 1);
        if (isVCacheSlotNonContig_)
            nonContigFlag |= (1 << 6);
        if (isVCacheHeadNonContig_)
            nonContigFlag |= (1 << 7);
    }
    if (!isKeyOutContiguous_)
        nonContigFlag |= (1 << 2);
    if (!isValueOutContiguous_)
        nonContigFlag |= (1 << 3);
    if (isKeyOutHeadNonContig_)
        nonContigFlag |= (1 << 8);
    if (isValueOutHeadNonContig_)
        nonContigFlag |= (1 << 9);

    tilingData_.set_nonContiguousFlag(nonContigFlag);

    // keyCache stride (元素粒度)
    if (!isKCacheContiguous_) {
        tilingData_.set_kCacheStride0(kCacheStride_.GetStride(0));
        tilingData_.set_kCacheStride1(kCacheStride_.GetStride(1));
        tilingData_.set_kCacheStride2(kCacheStride_.GetStride(2));
    } else {
        // 连续时填充连续stride值
        tilingData_.set_kCacheStride0(static_cast<int64_t>(blockSize_) * hiddenSizeK_);
        tilingData_.set_kCacheStride1(hiddenSizeK_);
        // 连续时dim2(N轴)步长=尾轴长度: ND为head_size, NZ为elenum_aligned, 均取GetDim(3)。
        tilingData_.set_kCacheStride2(kCacheShape_.GetDim(3));
    }

    // valueCache stride
    if (!isVCacheContiguous_) {
        tilingData_.set_vCacheStride0(vCacheStride_.GetStride(0));
        tilingData_.set_vCacheStride1(vCacheStride_.GetStride(1));
        tilingData_.set_vCacheStride2(vCacheStride_.GetStride(2));
    } else {
        tilingData_.set_vCacheStride0(static_cast<int64_t>(blockSize_) * hiddenSizeV_);
        tilingData_.set_vCacheStride1(hiddenSizeV_);
        // 连续时dim2(N轴)步长=尾轴长度: ND为head_size, NZ为elenum_aligned, 均取GetDim(3)。
        tilingData_.set_vCacheStride2(vCacheShape_.GetDim(3));
    }

    // key/value输出stride (ND和NZ模式都支持)
    // stride0=token(dim0)步长, stride1=head(dim1)步长。连续时填稠密值:
    //   ND: stride0=hiddenSize(=num_heads*head_size), stride1=head_size; NZ: 无dim1,
    //   stride1填hiddenSize(不被kernel使用)。
    if (!isKeyOutContiguous_) {
        tilingData_.set_keyOutStride0(keyOutStride_.GetStride(0));
        tilingData_.set_keyOutStride1(keyOutStride_.GetStride(1));
    } else {
        tilingData_.set_keyOutStride0(
            static_cast<int64_t>(keyShape_.GetDim(0)) > 0 ? static_cast<int64_t>(hiddenSizeK_) : 0);
        tilingData_.set_keyOutStride1((isCacheModeNorm_ && numHeadsK_ > 0) ?
                                          static_cast<int64_t>(hiddenSizeK_ / numHeadsK_) :
                                          static_cast<int64_t>(hiddenSizeK_));
    }

    if (!isValueOutContiguous_) {
        tilingData_.set_valueOutStride0(valueOutStride_.GetStride(0));
        tilingData_.set_valueOutStride1(valueOutStride_.GetStride(1));
    } else {
        tilingData_.set_valueOutStride0(
            static_cast<int64_t>(valueShape_.GetDim(0)) > 0 ? static_cast<int64_t>(hiddenSizeV_) : 0);
        tilingData_.set_valueOutStride1((isCacheModeNorm_ && numHeadsV_ > 0) ?
                                            static_cast<int64_t>(hiddenSizeV_ / numHeadsV_) :
                                            static_cast<int64_t>(hiddenSizeV_));
    }

    tilingData_.set_numHeadsK(numHeadsK_);
    tilingData_.set_numHeadsV(numHeadsV_);

    // 根据属性设置tilingkey
    for (const auto& item : tilingKeyTable) {
        if (item.isCacheModeNorm == isCacheModeNorm_ && item.isSeqLenCumSum == isSeqLenCumSum_ &&
            item.hasSeqOffset == hasSeqOffset_) {
            tilingKey_ = item.tilingKey;
            break;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GatherPaKvCacheTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GatherPaKvCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(needCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetContiguousTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx)
{
    auto xStorageShape = context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    shape = xStorageShape->GetStorageShape();
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1;
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx)
{
    bool isView = context_->InputIsView(idx);
    if (isView) {
        auto* inputStride = context_->GetInputStride(idx);
        if (inputStride == nullptr || inputStride->GetDimNum() == 0) {
            return GetContiguousTensorInfo(shape, stride, idx);
        } else {
            stride = *inputStride;
            shape = context_->GetInputShape(idx)->GetShape();
        }
    } else {
        return GetContiguousTensorInfo(shape, stride, idx);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGatherPaKvCache(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForGatherPaKvCache", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingForGatherPaKvCache enter.");

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context, "Tiling4GatherPaKvCache enter.");
        return Tiling4GatherPaKvCache(context);
    }

    GatherPaKvCacheTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForGatherPaKvCache(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForGatherPaKvCache", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForGatherPaKvCache enter");

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        return TilingPrepare4GatherPaKvCache(context);
    }

    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GatherPaKvCache op.
IMPL_OP_OPTILING(GatherPaKvCache)
    .Tiling(TilingForGatherPaKvCache)
    .TilingParse<GatherPaKvCacheCompileInfo>(TilingPrepareForGatherPaKvCache);

} // namespace optiling
