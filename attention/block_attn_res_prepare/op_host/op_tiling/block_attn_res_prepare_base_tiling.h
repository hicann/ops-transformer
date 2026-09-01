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
 * \file block_attn_res_prepare_base_tiling.h
 * \brief Base tiling workflow for BlockAttnResPrepare.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_BASE_TILING_H
#define BLOCK_ATTN_RES_PREPARE_BASE_TILING_H

#include <cstddef>
#include <cstdint>

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/block_attn_res_prepare_apt_tiling_key.h"

namespace optiling {
using namespace BlockAttnResPrepareTilingKey;

constexpr float BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS = 1e-6F;

struct BlockAttnResPrepareCompileInfo {
    uint64_t aicCoreNum = 0;
    uint64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t systemWorkspaceSize = 0;
};

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext *context);
ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext *context);

// Owns the template-independent tiling workflow. A new template derives directly from this class and only provides
// its capability check, tile calculation, tiling key, data view and diagnostic dump.
class BlockAttnResPrepareBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttnResPrepareBaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~BlockAttnResPrepareBaseTiling() override = default;

protected:
    static constexpr uint64_t AIV_CORE_NUM_PER_AIC = 2UL;

    template <typename T>
    static constexpr T CeilDiv(T value, T factor)
    {
        return (value + factor - 1) / factor;
    }

    template <typename T>
    static constexpr T AlignUp(T value, T factor)
    {
        return CeilDiv(value, factor) * factor;
    }

    struct WorkDistribution {
        uint32_t usedCoreNum = 0;
        uint32_t bigCoreNum = 0;
        uint32_t blockFactor = 0;
        uint32_t tailBlockFactor = 0;
    };

    struct TilingDataView {
        const void *data = nullptr;
        size_t size = 0;
        const char *templateName = nullptr;
    };

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    virtual TilingDataView GetTilingDataView() const = 0;

    ge::graphStatus CheckInputShapes();
    ge::graphStatus CheckInputDtypes() const;
    ge::graphStatus CalculateWorkDistribution(uint64_t totalWorkUnits, uint64_t maxCoreNum, const char *templateName,
                                              WorkDistribution &distribution) const;

    uint64_t totalT_ = 0;
    uint64_t totalN_ = 0;
    uint64_t totalS_ = 0;
    uint64_t totalD_ = 0;
    float eps_ = BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS;

    uint64_t aicCoreNum_ = 0;
    uint64_t aivCoreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t systemWorkspaceSize_ = 0;
    uint64_t workspaceSize_ = 0;
    uint32_t usedCoreNum_ = 0;
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_BASE_TILING_H
