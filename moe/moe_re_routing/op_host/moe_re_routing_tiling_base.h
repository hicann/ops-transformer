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
 * \file moe_re_routing_tiling_base.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_BASE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_BASE_H_

#include "moe_re_routing_tiling.h"

namespace optiling {

class MoeReRoutingTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeReRoutingTilingBase(gert::TilingContext *contextReRouting)
        : TilingBaseClass(contextReRouting)
    {}
    ~MoeReRoutingTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus CheckParam();

private:
    ge::graphStatus CheckDtypeAndAttr() const;
    ge::graphStatus CheckOutputShape() const;
    ge::graphStatus CheckNullptr() const;

protected:
    int64_t coreNum_{0};
    int64_t ubSize_ = {0};
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::RESERVED_VERSION;
    static constexpr int64_t INDEX_UB_SIZE = 256;
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr size_t RESERVE_WORKSPACE_BYTE = static_cast<size_t>(16) * 1024 * 1024;
    uint64_t tilingKey_ = {0};
    int64_t tokenSum_ = {0};
    int64_t tokenSize_ = {0};
    int64_t scaleSize_ = {0};
    int64_t rankNums_ = {0};
    int64_t ubFactor_ = {0};
    int64_t blockNum_ = {0};
    int64_t expertNum_ = {0};
    int64_t idxType_ = {0};
    bool hasScale_{false};
    bool hasTopkWeight_{false};
    ge::DataType tokenDtype_ = ge::DT_UNDEFINED;
    ge::DataType expertDtype_ = ge::DT_UNDEFINED;
    ge::DataType scaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType topkWeightDtype_ = ge::DT_UNDEFINED;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MOE_RE_ROUTING_BASE_H_
