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
 * \file moe_re_routing_re_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_RE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_RE_TILING_H_

#include "moe_re_routing_tiling_base.h"

namespace optiling {

class MoeReRoutingReTiling : public MoeReRoutingTilingBase {
public:
    explicit MoeReRoutingReTiling(gert::TilingContext *contextReRoutingReTiling)
        : MoeReRoutingTilingBase(contextReRoutingReTiling)
    {}
    ~MoeReRoutingReTiling() override = default;

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override
    {
        return (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) &&
               (rankNums_ < MIN_RANK_NUM && expertNum_ != 1);
    }

private:
    void PrintTilingData();
    void SplitCore();
    ge::graphStatus SplitUb();

private:
    int64_t blockNumE_ = {0};
    int64_t blockFactorE_ = {0};
    int64_t blockNumR_ = {0};
    int64_t blockFactorR_ = {0};
    static constexpr int64_t MIN_RANK_NUM = 5;
    MoeReRoutingReTilingData tilingData_;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_RE_TILING_H_
