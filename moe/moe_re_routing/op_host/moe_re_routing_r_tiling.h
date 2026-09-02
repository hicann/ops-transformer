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
 * \file moe_re_routing_r_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_R_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_R_TILING_H_

#include "moe_re_routing_tiling_base.h"
#include "op_host/tiling_util.h"

namespace optiling {

class MoeReRoutingRTiling : public MoeReRoutingTilingBase {
public:
    explicit MoeReRoutingRTiling(gert::TilingContext *contextReRoutingR)
        : MoeReRoutingTilingBase(contextReRoutingR)
    {}
    ~MoeReRoutingRTiling() override = default;

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }

private:
    void PrintTilingData();
    void SplitCore();
    ge::graphStatus SplitUb();

private:
    int64_t blockNum_ = {0};
    int64_t blockFactor_ = {0};
    MoeReRoutingRTilingData tilingData_;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_R_TILING_H_
