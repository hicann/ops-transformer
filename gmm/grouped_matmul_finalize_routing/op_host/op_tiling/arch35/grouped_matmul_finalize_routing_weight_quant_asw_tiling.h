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
 * \file grouped_matmul_finalize_routing_weight_quant_asw_tiling.h
 * \brief weight quant tiling class
 */

#pragma once

#include <vector>
#include <string>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "grouped_matmul_finalize_routing_tiling_tool.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace GMMFinalizeRoutingArch35Tiling;
using namespace GMMFRTilingTools;

constexpr int32_t GMMFR_WEIGHT_QUANT_ASW_TILING = 3;

class GMMFRWeightQuantAswTiling : public TilingBaseClass {
public:
    explicit GMMFRWeightQuantAswTiling(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~GMMFRWeightQuantAswTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void Reset();

private:
    void SetQuantMode();

    GQmmInputInfo inputParams_;
    GMMFRExtraInfo frInfo_;
    GMMFinalizeRoutingTilingData tilingData_;
};

} // namespace optiling
