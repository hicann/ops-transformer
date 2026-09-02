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
 * \file moe_re_routing_membase_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_MEMBASE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_MEMBASE_H_

#include "moe_re_routing_tiling.h"

namespace optiling {

class MoeReRoutingTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeReRoutingTiling(gert::TilingContext *contextReRouting)
        : TilingBaseClass(contextReRouting)
    {}
    ~MoeReRoutingTiling() override = default;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t tilingKey_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::RESERVED_VERSION;
    MoeReRoutingTilingData tilingData_;
    ParamsMoeReRouting commonParams_;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_MEMBASE_H_
