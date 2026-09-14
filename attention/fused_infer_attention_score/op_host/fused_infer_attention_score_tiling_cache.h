/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_cache.h
 * \brief FIA 手写路径 tiling 结果缓存
 *
 * 背景：手写框架路径（tiling sink）下，GWS（GetWorkspaceSize）与 Run 两个阶段会各自触发一次
 * tiling，导致同一 (shape, attr) 组合的 tiling 被重复执行，host 侧耗时显著增加。
 *
 * 方案：以影响 tiling 结果的全部输入信息为键，缓存 tiling 输出
 * （tiling data blob / tilingKey / blockDim / workspaceSize / scheduleMode），
 * 命中时直接恢复，跳过 tiling 计算。
 *
 * 键内容（影响 tiling 分支与结果的全部要素）：
 *   - npu 架构、缓存格式版本号
 *   - 全部 31 个输入的 storage/origin shape、dtype、format、stride（含动态 kv tensor 展开与
 *     非连续 kv cache stride、keyRope 的 view 属性）
 *   - 全部 16 个属性（含 outDtype）
 *   - 输出 shape/dtype
 *   - tiling 过程中会读取内容的 host tensor：actualSeqLengths/actualSeqLengthsKv/actualSharedPrefixLen
 *
 * 生效范围：仅 arch22 路由（else 分支）启用缓存，
 * arch35（DAV_3510）与 arch38（DAV_5102）路径不启用，
 * 行为与引入缓存前完全一致。
 *
 * 逃生开关：环境变量 FIA_TILING_CACHE_DISABLE=1 可关闭缓存（退化回原有行为）。
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_CACHE_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_CACHE_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "exe_graph/runtime/tiling_context.h"

#include "fused_infer_attention_score_tiling_index.h"
#include "../../common/op_host/fia_tiling_schedule_recorder.h"

namespace optiling {

// outDtype 属性不在 tiling_index.h 中定义，与 infershape 中保持一致
constexpr uint32_t FIA_TILING_CACHE_ATTR_OUT_DTYPE_INDEX = 15;

class FusedInferAttentionScoreTilingCache {
public:
    struct TilingResult {
        std::string tilingDataBlob; // 序列化后的 tiling data
        uint64_t tilingKey = 0;
        uint32_t blockDim = 0;
        uint64_t workspaceSize = 0;
        bool scheduleModeSet = false; // 本次 tiling 是否设置过 schedule mode
        uint32_t scheduleMode = 0;
    };

    static FusedInferAttentionScoreTilingCache &GetInstance()
    {
        static FusedInferAttentionScoreTilingCache instance;
        return instance;
    }

    static bool IsCacheDisabled()
    {
        static const bool disabled = []() {
            const char *env = std::getenv("FIA_TILING_CACHE_DISABLE");
            return (env != nullptr) && (env[0] != '\0') && (std::strcmp(env, "0") != 0);
        }();
        return disabled;
    }

    // 由 tiling context 构造缓存键；返回 false 表示无法构造（调用方应跳过缓存逻辑）
    static bool BuildKey(gert::TilingContext *context, const int32_t npuArch, std::string &key)
    {
        if (context == nullptr || context->GetAttrs() == nullptr) {
            return false;
        }
        key.clear();
        Append(key, kCacheKeyVersion);
        Append(key, npuArch);

        auto *attrs = context->GetAttrs();
        AppendAttr<int64_t>(key, attrs, ATTR_N_INDEX);
        AppendAttr<float>(key, attrs, ATTR_SCALE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_PRE_TOKEN_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_NEXT_TOKEN_INDEX);
        AppendAttrStr(key, attrs, ATTR_INPUT_LAYOUT_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_NUM_KV_HEADS_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_SPARSE_MODE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_INNER_PRECISE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_BLOCK_SIZE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_ANTIQUANT_MODE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_SOFTMAX_LSE_FLAG_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_KEY_ANTIQUANT_MODE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_VALUE_ANTIQUANT_MODE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_QUERY_QUANT_MODE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_PSE_TYPE_INDEX);
        AppendAttr<int64_t>(key, attrs, FIA_TILING_CACHE_ATTR_OUT_DTYPE_INDEX);

        // 全部静态输入（含 optional 输入）：shape / dtype / format / stride
        for (uint32_t inputIndex = 0; inputIndex <= KV_START_IDX_INDEX; ++inputIndex) {
            AppendInput(key, context, inputIndex);
        }
        // 动态 kv tensor 展开：shape + stride
        AppendDynamicInputs(key, context, KEY_INDEX);
        AppendDynamicInputs(key, context, VALUE_INDEX);
        // keyRope/动态 kv 的 view 属性参与非连续 cache 判定
        key.push_back(context->InputIsView(KEY_ROPE_INDEX) ? 1 : 0);
        key.push_back(context->InputIsView(KEY_INDEX) ? 1 : 0);
        key.push_back(context->InputIsView(VALUE_INDEX) ? 1 : 0);
        AppendStride(key, context->GetOptionalInputStride(KEY_ROPE_INDEX));

        // tiling 过程中会读取内容的 host tensor
        AppendHostTensorData(key, context, ACTUAL_SEQ_Q_INDEX);
        AppendHostTensorData(key, context, ACTUAL_SEQ_KV_INDEX);
        AppendHostTensorData(key, context, ACTUAL_SHARED_PREFIX_LEN_INDEX);

        // 输出
        for (uint32_t outputIndex = 0; outputIndex <= SOFTMAX_LSE_INDEX; ++outputIndex) {
            AppendOutput(key, context, outputIndex);
        }
        return true;
    }

    // 在真实 tiling 完成后，从 context 快照 tiling 结果
    static bool Snapshot(gert::TilingContext *context, TilingResult &result)
    {
        if (context == nullptr) {
            return false;
        }
        auto *rawTilingData = context->GetRawTilingData();
        if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
            return false;
        }
        size_t *workSpaces = context->GetWorkspaceSizes(1);
        if (workSpaces == nullptr) {
            return false;
        }
        const size_t dataSize = rawTilingData->GetDataSize();
        result.tilingDataBlob.resize(dataSize);
        if (dataSize != 0) {
            auto *src = reinterpret_cast<const char *>(rawTilingData->GetData());
            std::memcpy(result.tilingDataBlob.data(), src, dataSize);
        }
        result.tilingKey = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        result.workspaceSize = workSpaces[0];
        result.scheduleModeSet = FiaTilingScheduleRecorder::Get(result.scheduleMode);
        return true;
    }

    // 将缓存结果写回 context；失败时调用方应回退执行真实 tiling
    static ge::graphStatus Restore(gert::TilingContext *context, const TilingResult &result)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto *rawTilingData = context->GetRawTilingData();
        size_t *workSpaces = context->GetWorkspaceSizes(1);
        if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
            rawTilingData->GetCapacity() < result.tilingDataBlob.size() || workSpaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (!result.tilingDataBlob.empty()) {
            auto *dst = reinterpret_cast<char *>(rawTilingData->GetData());
            std::memcpy(dst, result.tilingDataBlob.data(), result.tilingDataBlob.size());
        }
        rawTilingData->SetDataSize(result.tilingDataBlob.size());
        context->SetTilingKey(result.tilingKey);
        context->SetBlockDim(result.blockDim);
        workSpaces[0] = result.workspaceSize;
        if (result.scheduleModeSet) {
            context->SetScheduleMode(result.scheduleMode);
        }
        return ge::GRAPH_SUCCESS;
    }

    bool Get(const std::string &key, TilingResult &result)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = cache_.find(key);
        if (iter == cache_.end()) {
            return false;
        }
        result = iter->second;
        return true;
    }

    void Add(std::string key, const TilingResult &result)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cache_.size() >= kMaxCacheEntryNum) {
            return;
        }
        cache_[std::move(key)] = result;
    }

private:
    static constexpr uint32_t kCacheKeyVersion = 1; // 键序列化格式变化时递增
    static constexpr size_t kMaxCacheEntryNum = 128;
    static constexpr uint32_t kMaxDynamicTensorNum = 1024; // 动态输入数量防御上限
    static constexpr uint64_t kMaxHostTensorElems = 65536; // host tensor 元素数防御上限

    std::mutex mutex_;
    std::map<std::string, TilingResult> cache_;

    template <typename T>
    static void Append(std::string &key, const T &value)
    {
        key.append(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    template <typename T>
    static void AppendAttr(std::string &key, const gert::RuntimeAttrs *attrs, const uint32_t index)
    {
        const T *value = attrs->GetAttrPointer<T>(index);
        if (value == nullptr) {
            key.push_back(0);
            return;
        }
        key.push_back(1);
        Append(key, *value);
    }

    static void AppendAttrStr(std::string &key, const gert::RuntimeAttrs *attrs, const uint32_t index)
    {
        const char *value = attrs->GetAttrPointer<char>(index);
        if (value == nullptr) {
            key.push_back(0);
            return;
        }
        key.push_back(1);
        const uint32_t len = static_cast<uint32_t>(std::strlen(value));
        Append(key, len);
        key.append(value, len);
    }

    static void AppendShape(std::string &key, const gert::Shape &shape)
    {
        Append<uint32_t>(key, static_cast<uint32_t>(shape.GetDimNum()));
        for (size_t dim = 0; dim < shape.GetDimNum(); ++dim) {
            Append<int64_t>(key, shape.GetDim(dim));
        }
    }

    static void AppendStride(std::string &key, const gert::Stride *stride)
    {
        if (stride == nullptr || stride->GetDimNum() == 0) {
            key.push_back(0);
            return;
        }
        key.push_back(1);
        Append<uint32_t>(key, static_cast<uint32_t>(stride->GetDimNum()));
        for (size_t dim = 0; dim < stride->GetDimNum(); ++dim) {
            Append<int64_t>(key, stride->GetStride(dim));
        }
    }

    static const gert::CompileTimeTensorDesc *GetInputDescSafe(gert::TilingContext *context, const uint32_t index)
    {
        const auto *desc = context->GetInputDesc(index);
        if (desc == nullptr) {
            desc = context->GetOptionalInputDesc(index);
        }
        return desc;
    }

    static void AppendDesc(std::string &key, const gert::CompileTimeTensorDesc *desc)
    {
        if (desc == nullptr) {
            key.push_back(0);
            return;
        }
        key.push_back(1);
        Append<int32_t>(key, static_cast<int32_t>(desc->GetDataType()));
        Append<int32_t>(key, static_cast<int32_t>(desc->GetStorageFormat()));
    }

    static void AppendInput(std::string &key, gert::TilingContext *context, const uint32_t index)
    {
        // required 输入走 GetInputShape；optional 输入 GetInputShape 可能返回 nullptr，需回退
        const auto *shape = context->GetInputShape(index);
        if (shape == nullptr) {
            shape = context->GetOptionalInputShape(index);
        }
        if (shape == nullptr) {
            key.push_back(0);
        } else {
            key.push_back(1);
            AppendShape(key, shape->GetStorageShape());
            AppendShape(key, shape->GetOriginShape());
        }
        AppendDesc(key, GetInputDescSafe(context, index));
        AppendStride(key, context->GetInputStride(index));
    }

    static void AppendDynamicInputs(std::string &key, gert::TilingContext *context, const uint32_t inputIndex)
    {
        uint32_t tensorNum = 0;
        while (tensorNum < kMaxDynamicTensorNum) {
            const auto *shape = context->GetDynamicInputShape(inputIndex, tensorNum);
            if (shape == nullptr) {
                break;
            }
            AppendShape(key, shape->GetStorageShape());
            AppendStride(key, context->GetDynamicInputStride(inputIndex, tensorNum));
            ++tensorNum;
        }
        Append<uint32_t>(key, tensorNum);
    }

    static void AppendHostTensorData(std::string &key, gert::TilingContext *context, const uint32_t index)
    {
        const auto *tensor = context->GetOptionalInputTensor(index);
        const auto *data = (tensor != nullptr) ? tensor->GetData<int64_t>() : nullptr;
        if (data == nullptr) {
            key.push_back(0);
            return;
        }
        key.push_back(1);
        const int64_t shapeSize = tensor->GetShapeSize();
        if (shapeSize <= 0 || static_cast<uint64_t>(shapeSize) > kMaxHostTensorElems) {
            // 异常 shapeSize：仅记录存在性标记，避免读取越界（此处不缓存内容，退化为以
            // nullptr/非 nullptr 区分；该输入通常为 (batch+1) 或标量，不会触发此分支）
            Append<uint64_t>(key, 0);
            return;
        }
        const uint64_t byteSize = static_cast<uint64_t>(shapeSize) * sizeof(int64_t);
        Append<uint64_t>(key, byteSize);
        key.append(reinterpret_cast<const char *>(data), static_cast<size_t>(byteSize));
    }

    static void AppendOutput(std::string &key, gert::TilingContext *context, const uint32_t index)
    {
        const auto *shape = context->GetOutputShape(index);
        if (shape == nullptr) {
            key.push_back(0);
        } else {
            key.push_back(1);
            AppendShape(key, shape->GetStorageShape());
        }
        const auto *desc = context->GetOutputDesc(index);
        AppendDesc(key, desc);
    }
};
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_CACHE_H
