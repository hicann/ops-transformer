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

#include "exe_graph/runtime/tiling_parse_context.h"
#include "graph/types.h"
#include "op_host/tiling_base.h"

namespace optiling {

constexpr float BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS = 1e-6F;

// Platform information is queried directly during tiling.
struct BlockAttnResPrepareCompileInfo {};

ge::graphStatus TilingBlockAttnResPrepare(gert::TilingContext* context);
ge::graphStatus TilingPrepareBlockAttnResPrepare(gert::TilingParseContext* context);

// Owns the template-independent tiling workflow. Each template binds its tiling data buffer and provides
// its capability check, tile calculation and tiling key.
class BlockAttnResPrepareBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttnResPrepareBaseTiling(gert::TilingContext* context)
        : TilingBaseClass(context)
    {}
    ~BlockAttnResPrepareBaseTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus CheckInputShapes();
    ge::graphStatus CheckInputDtypes();

    const void* tilingDataPtr_ = nullptr;
    size_t tilingDataSize_ = 0;

    uint64_t totalT_ = 0;
    uint64_t totalN_ = 0;
    uint64_t totalS_ = 0;
    uint64_t totalD_ = 0;

    uint64_t aicCoreNum_ = 0;
    uint64_t aivCoreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t systemWorkspaceSize_ = 0;

    ge::DataType blockResDtype_ = ge::DT_UNDEFINED;
    float eps_ = BLOCK_ATTN_RES_PREPARE_DEFAULT_EPS;
    uint32_t usedCoreNum_ = 0;
};

} // namespace optiling

#endif // BLOCK_ATTN_RES_PREPARE_BASE_TILING_H
