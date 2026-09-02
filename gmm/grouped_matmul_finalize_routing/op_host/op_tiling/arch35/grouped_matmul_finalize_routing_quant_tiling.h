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
 * \file grouped_matmul_finalize_routing_quant_tiling.h
 * \brief
 */

#ifndef ARCH35_GROUPED_MATMUL_FINALIZE_ROUTING_QUANT_TILING_H
#define ARCH35_GROUPED_MATMUL_FINALIZE_ROUTING_QUANT_TILING_H

#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "../../grouped_matmul_finalize_routing_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace GroupedMatmulFinalizeRoutingArch35TilingConstant {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t W_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t BIAS_INDEX = 3;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 4;
constexpr uint32_t GROUPLIST_INDEX = 5;
constexpr uint32_t SHARE_INPUT_INDEX = 6;
constexpr uint32_t LOGIT_INDEX = 7;
constexpr uint32_t ROW_INDEX_INDEX = 8;
constexpr uint32_t OFFSET_INDEX = 9;

constexpr uint32_t Y_INDEX = 0;

constexpr uint32_t ATTR_INDEX_DTYPE = 0;
constexpr uint32_t ATTR_INDEX_SHARE_INPUT_WEIGHT = 1;
constexpr uint32_t ATTR_INDEX_SHARE_INPUT_OFFSET = 2;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_X = 3;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_W = 4;
constexpr uint32_t ATTR_INDEX_OUTPUT_BS = 5;
constexpr uint32_t ATTR_INDEX_GROUP_LIST_TYPE = 6;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 7;

constexpr uint32_t DIM_NUM_X = 2;
constexpr uint32_t DIM_NUM_PERTOKENSCALE = 1;
constexpr uint32_t DIM_NUM_MX_PERTOKENSCALE = 3;
constexpr uint32_t DIM_NUM_WEIGHT = 3;
constexpr uint32_t DIM_NUM_SCALE = 3;
constexpr uint32_t DIM_NUM_MX_SCALE = 4;
constexpr uint32_t DIM_NUM_Y = 2;
constexpr uint32_t OUT_DTYPE_BF16_INDEX = 2;
constexpr int32_t SPLIT_M = 0;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024U;
constexpr float DETER_WORKSPACE_RATIO = 0.8f; // 减去系统开销后，确定性workspace占比
} // namespace GroupedMatmulFinalizeRoutingArch35TilingConstant

class GroupedMatmulFinalizeRoutingQuantTiling : public GroupedQmmTiling {
public:
    explicit GroupedMatmulFinalizeRoutingQuantTiling(gert::TilingContext *context)
        : GroupedQmmTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulFinalizeRoutingQuantTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmTiling::Reset(context);
        Reset();
    }

protected:
    const char *GetOpType() const override
    {
        return "GroupedMatmulFinalizeRouting";
    }
    // 0、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 1、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 2、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 3、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 4、保存Tiling数据
    ge::graphStatus PostTiling() override;
    // 5、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    void Reset() override;
    ge::graphStatus DeterministicTilingProcess();

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    int64_t LogQuantParams() const;
    int64_t LogMatmulParams() const;
    bool SetQuantModeForGMMFinalizeRouting();

    bool CheckOptionalAttr();
    bool CheckDtype() const;
    bool CheckOptional(uint32_t index, const char *paramName, ge::DataType targetDtype) const;
    bool IsFp4Dtype(ge::DataType dtype) const;
    bool IsFp8Dtype(ge::DataType dtype) const;
    bool CheckInputsShape(const gert::Shape &xShape, const gert::StorageShape *wStorageShape,
                          const gert::StorageShape *pertokenScaleStorageShape, const gert::Shape &scaleShape,
                          const gert::Shape &yShape) const;
    bool CheckOptionalInputsShape();
    bool CheckDim(const gert::Shape &xShape, const gert::Shape &wShape,
                  const gert::StorageShape *pertokenScaleStorageShape, const gert::Shape &scaleShape,
                  const gert::Shape &yShape) const;
    bool CheckFp4Shape(const gert::Shape &xShape, const gert::Shape &wShape) const;
    bool CheckCoreNum() const override;
    bool LoadInputDescsForRouting();
    bool CheckOptionalInputsForRouting();
    bool CheckScaleAndPerTokenDims(const gert::StorageShape *pertokenScaleStorageShape,
                                   const gert::Shape &scaleShape) const;

    GMMFinalizeRoutingTilingData tilingData_;
    uint64_t sharedInputLen_ = 0;
    uint64_t sharedInputOffset_ = 0;
    uint64_t rowIndex_ = 0;
    float sharedInputWeight_ = 1.0;
    uint64_t outputBs_ = 0;
    int8_t scaleType_ = 0;
    int8_t rowIndexType_ = 0;
    uint32_t deterWorkspaceSize_ = 0;
};
} // namespace optiling

#endif
