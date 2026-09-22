/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_CACHE_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_CACHE_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "tiling/platform/platform_ascendc.h"
#include "fused_infer_attention_score_tiling_index.h"
#include "../../common/op_host/fia_tiling_schedule_recorder.h"

namespace optiling {
// Cross-invocation HOST tiling cache for A2/A3. Entries become usable immediately
// after successful tiling; no executor or device-task replay is stored here.
// opbase still owns workspace allocation, address refresh and kernel launch.
class FusedInferAttentionScoreTilingCache {
public:
    struct TilingResult {
        std::string tilingDataBlob;
        uint64_t tilingKey = 0;
        uint32_t blockDim = 0;
        size_t workspaceSize = 0;
        bool scheduleModeSet = false;
        uint32_t scheduleMode = 0;
    };
    using ResultPtr = std::shared_ptr<const TilingResult>;

    explicit FusedInferAttentionScoreTilingCache(size_t entries = 128, size_t bytes = 16 * 1024 * 1024)
        : maxEntries_(entries),
          maxBytes_(bytes)
    {}

    static FusedInferAttentionScoreTilingCache &GetInstance()
    {
        static FusedInferAttentionScoreTilingCache instance;
        return instance;
    }

    static bool IsCacheDisabled()
    {
        static const bool disabled = [] {
            const char *value = std::getenv("FIA_TILING_CACHE_DISABLE");
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        }();
        return disabled;
    }

    static bool BuildKey(gert::TilingContext *context, std::string &key)
    {
        if (context == nullptr || context->GetAttrs() == nullptr || context->GetPlatformInfo() == nullptr) {
            return false;
        }
        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        if (platform.GetCurNpuArch() != NpuArch::DAV_2201) {
            return false;
        }
        key.clear();
        Append<uint32_t>(key, 2); // serialization version
        Append<int32_t>(key, static_cast<int32_t>(platform.GetCurNpuArch()));
        Append<int32_t>(key, static_cast<int32_t>(platform.GetSocVersion()));
        Append<uint32_t>(key, platform.GetCoreNumAic());
        Append<uint32_t>(key, platform.GetCoreNumAiv());
        Append<uint64_t>(key, platform.GetLibApiWorkSpaceSize());
        for (auto type : {platform_ascendc::CoreMemType::UB, platform_ascendc::CoreMemType::L1,
                          platform_ascendc::CoreMemType::L0_A, platform_ascendc::CoreMemType::L0_B,
                          platform_ascendc::CoreMemType::L0_C, platform_ascendc::CoreMemType::L2}) {
            uint64_t size = 0;
            platform.GetCoreMemSize(type, size);
            Append(key, size);
        }
        Append<int32_t>(key, context->GetDeterministic());

        const auto *attrs = context->GetAttrs();
        AppendAttr<int64_t>(key, attrs, ATTR_N_INDEX);
        AppendAttr<float>(key, attrs, ATTR_SCALE_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_PRE_TOKEN_INDEX);
        AppendAttr<int64_t>(key, attrs, ATTR_NEXT_TOKEN_INDEX);
        const char *layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
        if (layout == nullptr) {
            return false;
        }
        const size_t layoutSize = std::strlen(layout);
        if (layoutSize > 64) {
            return false;
        }
        Append<uint32_t>(key, static_cast<uint32_t>(layoutSize));
        key.append(layout, layoutSize);
        for (uint32_t index = ATTR_NUM_KV_HEADS_INDEX; index <= 15; ++index) {
            if (index == ATTR_SOFTMAX_LSE_FLAG_INDEX) {
                AppendAttr<bool>(key, attrs, index);
            } else {
                AppendAttr<int64_t>(key, attrs, index);
            }
        }

        // Logical IR indices must not be confused with flattened tensor indices.
        AppendTensorMeta(key, context->GetInputShape(QUERY_INDEX), context->GetInputDesc(QUERY_INDEX),
                         context->GetInputStride(QUERY_INDEX));
        if (!AppendDynamicInputs(key, context, KEY_INDEX) || !AppendDynamicInputs(key, context, VALUE_INDEX)) {
            return false;
        }
        for (uint32_t index = PSE_SHIFT_INDEX; index <= KV_START_IDX_INDEX; ++index) {
            AppendTensorMeta(key, context->GetOptionalInputShape(index), context->GetOptionalInputDesc(index),
                             context->GetOptionalInputStride(index));
            key.push_back(context->GetOptionalInputTensor(index) != nullptr);
        }
        // Capture every view flag consumed by the FIA parser.
        for (uint32_t index : {KEY_INDEX, VALUE_INDEX, KEY_ROPE_INDEX}) {
            key.push_back(context->InputIsView(index));
        }
        // Only host-visible, complete value-dependent inputs may participate.
        for (uint32_t index : {ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX, ACTUAL_SHARED_PREFIX_LEN_INDEX,
                               Q_START_IDX_INDEX, KV_START_IDX_INDEX}) {
            if (!AppendHostTensorData(key, context, index)) {
                return false;
            }
        }
        for (uint32_t index = 0; index <= SOFTMAX_LSE_INDEX; ++index) {
            AppendTensorMeta(key, context->GetOutputShape(index), context->GetOutputDesc(index), nullptr);
        }
        return key.size() <= kMaxKeyBytes;
    }

    static bool Snapshot(gert::TilingContext *context, const FiaTilingScheduleRecorder::Scope &scope,
                         TilingResult &result)
    {
        if (context == nullptr || context->GetWorkspaceNum() != 1) {
            return false;
        }
        auto *data = context->GetRawTilingData();
        auto *workspace = context->GetWorkspaceSizes(1); // all cached FIA routes produce exactly one workspace
        if (data == nullptr || data->GetData() == nullptr || workspace == nullptr ||
            data->GetDataSize() > data->GetCapacity() || data->GetDataSize() > kMaxBlobBytes) {
            return false;
        }
        result.tilingDataBlob.assign(reinterpret_cast<const char *>(data->GetData()), data->GetDataSize());
        result.tilingKey = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        result.workspaceSize = workspace[0];
        result.scheduleModeSet = scope.Get(result.scheduleMode);
        return true;
    }

    static ge::graphStatus Restore(gert::TilingContext *context, const TilingResult &result)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto *data = context->GetRawTilingData();
        auto *workspace = context->GetWorkspaceSizes(1);
        if (data == nullptr || data->GetData() == nullptr || workspace == nullptr ||
            data->GetCapacity() < result.tilingDataBlob.size()) {
            return ge::GRAPH_FAILED;
        }
        if (result.scheduleModeSet &&
            FiaTilingScheduleRecorder::Set(context, result.scheduleMode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        std::memcpy(data->GetData(), result.tilingDataBlob.data(), result.tilingDataBlob.size());
        data->SetDataSize(result.tilingDataBlob.size());
        context->SetTilingKey(result.tilingKey);
        context->SetBlockDim(result.blockDim);
        workspace[0] = result.workspaceSize;
        return ge::GRAPH_SUCCESS;
    }

    ResultPtr Get(const std::string &key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter == entries_.end()) {
            return nullptr;
        }
        recent_.splice(recent_.begin(), recent_, iter->second.position);
        return iter->second.result; // immutable shared ownership; restore/copy happens outside the lock
    }

    bool Add(const std::string &key, const TilingResult &result)
    {
        if (key.size() > kMaxKeyBytes || result.tilingDataBlob.size() > kMaxBlobBytes) {
            return false;
        }
        const size_t bytes = key.size() * 2 + result.tilingDataBlob.size() + sizeof(Entry);
        if (maxEntries_ == 0 || bytes > maxBytes_) {
            return false;
        }
        auto copy = std::make_shared<const TilingResult>(result);
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            recent_.splice(recent_.begin(), recent_, iter->second.position);
            return true;
        }
        while (!recent_.empty() && (entries_.size() >= maxEntries_ || usedBytes_ + bytes > maxBytes_)) {
            const auto old = entries_.find(recent_.back());
            usedBytes_ -= old->second.bytes;
            entries_.erase(old);
            recent_.pop_back();
        }
        recent_.push_front(key);
        try {
            entries_.emplace(key, Entry{std::move(copy), recent_.begin(), bytes});
        } catch (...) {
            recent_.pop_front();
            throw;
        }
        usedBytes_ += bytes;
        return true;
    }

private:
    static constexpr size_t kMaxKeyBytes = 1024 * 1024;
    static constexpr size_t kMaxBlobBytes = 1024 * 1024;
    static constexpr uint32_t kMaxDynamicTensors = 1024;
    static constexpr int64_t kMaxHostElements = 65536;
    struct Entry {
        ResultPtr result;
        std::list<std::string>::iterator position;
        size_t bytes;
    };
    size_t maxEntries_;
    size_t maxBytes_;
    size_t usedBytes_ = 0;
    std::mutex mutex_;
    std::list<std::string> recent_;
    std::unordered_map<std::string, Entry> entries_;

    template <typename T>
    static void Append(std::string &key, const T &value)
    {
        key.append(reinterpret_cast<const char *>(&value), sizeof(T));
    }
    template <typename T>
    static void AppendAttr(std::string &key, const gert::RuntimeAttrs *attrs, uint32_t index)
    {
        const T *value = attrs->GetAttrPointer<T>(index);
        key.push_back(value != nullptr);
        if (value != nullptr) {
            Append(key, *value);
        }
    }
    static void AppendShape(std::string &key, const gert::Shape &shape)
    {
        Append<uint32_t>(key, static_cast<uint32_t>(shape.GetDimNum()));
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            Append<int64_t>(key, shape.GetDim(i));
        }
    }
    static void AppendTensorMeta(std::string &key, const gert::StorageShape *shape,
                                 const gert::CompileTimeTensorDesc *desc, const gert::Stride *stride)
    {
        key.push_back(shape != nullptr);
        if (shape != nullptr) {
            AppendShape(key, shape->GetStorageShape());
            AppendShape(key, shape->GetOriginShape());
        }
        key.push_back(desc != nullptr);
        if (desc != nullptr) {
            Append<int32_t>(key, static_cast<int32_t>(desc->GetDataType()));
            Append<int32_t>(key, static_cast<int32_t>(desc->GetStorageFormat()));
            Append<int32_t>(key, static_cast<int32_t>(desc->GetOriginFormat()));
        }
        key.push_back(stride != nullptr);
        if (stride != nullptr) {
            Append<uint32_t>(key, static_cast<uint32_t>(stride->GetDimNum()));
            for (size_t i = 0; i < stride->GetDimNum(); ++i) {
                Append<int64_t>(key, stride->GetStride(i));
            }
        }
    }
    static bool AppendDynamicInputs(std::string &key, gert::TilingContext *context, uint32_t index)
    {
        for (uint32_t i = 0; i < kMaxDynamicTensors; ++i) {
            const auto *shape = context->GetDynamicInputShape(index, i);
            key.push_back(shape != nullptr);
            if (shape == nullptr) {
                return true;
            }
            AppendTensorMeta(key, shape, context->GetDynamicInputDesc(index, i),
                             context->GetDynamicInputStride(index, i));
        }
        // Do not silently truncate: entries beyond the bound could change tiling.
        if (context->GetDynamicInputShape(index, kMaxDynamicTensors) != nullptr) {
            return false;
        }
        key.push_back(0);
        return true;
    }
    static bool AppendHostTensorData(std::string &key, gert::TilingContext *context, uint32_t index)
    {
        const auto *tensor = context->GetOptionalInputTensor(index);
        key.push_back(tensor != nullptr);
        if (tensor == nullptr) {
            return true;
        }
        const int64_t count = tensor->GetShapeSize();
        if (tensor->GetDataType() != ge::DT_INT64 || count < 0 || count > kMaxHostElements) {
            return false;
        }
        Append(key, count);
        if (count == 0) {
            return true;
        }
        if (tensor->GetPlacement() != gert::kOnHost && tensor->GetPlacement() != gert::kFollowing) {
            return false; // never dereference device contents or cache an incomplete value key
        }
        const auto *data = tensor->GetData<int64_t>();
        if (data == nullptr) {
            return false;
        }
        key.append(reinterpret_cast<const char *>(data), static_cast<size_t>(count) * sizeof(int64_t));
        return key.size() <= kMaxKeyBytes;
    }
};
} // namespace optiling
#endif
