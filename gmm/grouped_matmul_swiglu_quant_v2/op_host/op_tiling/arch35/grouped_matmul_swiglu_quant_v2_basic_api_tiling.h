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
 * \file grouped_matmul_swiglu_quant_v2_basic_api_tiling.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_API_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_API_TILING_H

#include <cstddef>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "../../grouped_matmul_swiglu_quant_v2_host_utils.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"

namespace optiling {
class GroupedMatmulSwigluQuantV2BasicApiTiling950 : public GroupedQmmBasicApiTiling {
public:
    explicit GroupedMatmulSwigluQuantV2BasicApiTiling950(gert::TilingContext *context)
        : GroupedQmmBasicApiTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2BasicApiTiling950() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    const char *GetOpType() const override
    {
        return "GroupedMatmulSwigluQuantV2";
    }

    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset() override;

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckTensorApiShapes() const;
    bool CheckTensorApiScaleShapes() const;
    bool IsFp8(ge::DataType dtype) const;
    bool IsSupportedFormat(ge::Format format) const;
    size_t GetDynamicInputCount(uint32_t inputIndex) const;

    GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData tilingData_;
    ge::DataType xScaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType quantDtype_ = ge::DT_UNDEFINED;
    ge::Format xScaleFormat_ = static_cast<ge::Format>(-1);
    ge::Format weightScaleFormat_ = static_cast<ge::Format>(-1);
    ge::Format yScaleFormat_ = static_cast<ge::Format>(-1);
    uint32_t aivNum_ = 0U;
    bool platformMemoryReady_ = false;
};
} // namespace optiling

#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_API_TILING_H
