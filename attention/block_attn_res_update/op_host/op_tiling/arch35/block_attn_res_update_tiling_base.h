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
#include "op_host/tiling_base.h"

namespace optiling {
namespace block_attn_res_update {

class BlockAttnResUpdateTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockAttnResUpdateTilingBase(gert::TilingContext *context)
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
    uint32_t dSize_ = 0;
    float eps_ = 0.0F;
    uint64_t ubSize_ = 0;
    uint32_t aivNum_ = 0;
    const char *opName_ = nullptr;
};

} // namespace block_attn_res_update
} // namespace optiling

#endif // BLOCK_ATTN_RES_UPDATE_TILING_BASE_H
