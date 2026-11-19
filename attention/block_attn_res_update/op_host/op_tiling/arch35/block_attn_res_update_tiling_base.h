/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TILING_BASE_H
#define BLOCK_ATTN_RES_UPDATE_TILING_BASE_H

#include <cstdint>
#include "graph/types.h"
#include "op_host/tiling_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

ge::graphStatus Tiling4BlockAttnResUpdate(gert::TilingContext* context);
ge::graphStatus TilingPrepare4BlockAttnResUpdate(gert::TilingParseContext* context);

namespace BlockAttnResUpdate {

class BlockAttnResUpdateTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttnResUpdateTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {}
    ~BlockAttnResUpdateTilingBase() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus CheckContext();

    uint64_t tSize_ = 0;
    uint64_t ubSize_ = 0;
    const char* opName_ = nullptr;
    uint32_t dSize_ = 0;
    uint32_t aivNum_ = 0;
    ge::DataType outputNormWeightDtype_ = ge::DT_UNDEFINED;
    float eps_ = 1e-6F;
    float outputNormEps_ = 1e-6F;
    bool applyOutputNorm_ = false;
};

} // namespace BlockAttnResUpdate
} // namespace optiling

#endif // BLOCK_ATTN_RES_UPDATE_TILING_BASE_H
