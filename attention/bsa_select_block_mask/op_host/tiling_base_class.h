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
 * \file tiling_base_class.h
 * \brief
 */
#ifndef __TILING_BASE_CLASS_H
#define __TILING_BASE_CLASS_H

#include <numeric>
#include <algorithm>
#include <sstream>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "err/ops_err.h"

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

using namespace ge;
using namespace AscendC;

namespace optiling {

static const int64_t MAX_VAR_LEN_SEQ_LEN = 4096L;
static const int64_t GM_ALIGN = 512;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;

static constexpr uint32_t D_SIZE_128 = 128;
static constexpr uint32_t BLOCK_SHAPE_ALIGN = 8;
static constexpr float SPARSITY_MIN = 0.0f;
static constexpr float SPARSITY_MAX = 1.0f;
static constexpr uint32_t HEAD_NUM_MIN = 1;
static constexpr uint32_t HEAD_NUM_MAX = 40;
static constexpr uint16_t Q_CHUNK_SIZE = 128;
static constexpr uint16_t K_CHUNK_SIZE = 128;
static constexpr uint8_t SCHEDULE_MODE_ALL_CORE = 1;

template <typename T>
static auto BSACeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline auto BSAMax(T a, T b) -> T
{
    return (a < b) ? (b) : (a);
}

class TilingBaseClass {
public:
    explicit TilingBaseClass(gert::TilingContext *context)
        : context_(context)
    {}

    virtual ~TilingBaseClass() = default;

    ge::graphStatus DoTiling()
    {
        auto ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetPlatformInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = GetWorkspaceSize();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        context_->SetTilingKey(GetTilingKey());
        DumpTilingInfo();
        return ge::GRAPH_SUCCESS;
    }

    virtual void Reset(gert::TilingContext *context)
    {
        context_ = context;
    }

protected:
    virtual bool IsCapable() = 0;
    virtual ge::graphStatus GetPlatformInfo() = 0;
    virtual ge::graphStatus GetShapeAttrsInfo() = 0;
    virtual ge::graphStatus DoOpTiling() = 0;
    [[nodiscard]] virtual uint64_t GetTilingKey() const = 0;
    virtual ge::graphStatus GetWorkspaceSize() = 0;
    virtual void DumpTilingInfo()
    {
        int32_t enable = 1;
        if (enable != 1) {
            return;
        }
        auto buf = (uint32_t *)context_->GetRawTilingData()->GetData();
        auto bufLen = context_->GetRawTilingData()->GetDataSize();
        std::ostringstream oss;
        oss << "Start to dump tiling info. tilingkey:" << context_->GetTilingKey() << ", tiling data size:" << bufLen
            << ", content:";
        for (size_t i = 0; i < bufLen / sizeof(uint32_t); i++) {
            oss << *(buf + i) << ",";
            if (oss.str().length() > 640) {
                OP_LOGD(context_, "%s", oss.str().c_str());
                oss.str("");
            }
        }
        OP_LOGD(context_, "%s", oss.str().c_str());
    }

protected:
    gert::TilingContext *context_ = nullptr;
    std::unique_ptr<platform_ascendc::PlatformAscendC> ascendcPlatform_{nullptr};
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};
};

} // namespace optiling
#endif
