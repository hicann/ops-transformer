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
 * \file grouped_matmul_finalize_routing_weight_quant_checker.h
 * \brief weight quant tiling checker
 */

#pragma once

#include <vector>
#include <string>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/types.h>
#include "register/op_impl_registry.h"
#include "grouped_matmul_finalize_routing_tiling_tool.h"

namespace {
static constexpr const char* OP_NAME = "GroupedMatmulFinalizeRouting";
}

namespace optiling {
namespace GMMFinalizeRoutingWeightQuantChecker {

using namespace GMMFRTilingTools;
using namespace GMMFRTilingConstants;

constexpr uint64_t AIC_AIV_CORE_RATIO_2 = 2;
constexpr uint64_t AIC_AIV_CORE_RATIO_1 = 1;
constexpr uint64_t ALIGN_SIZE_K = 32;
constexpr uint64_t ALIGN_SIZE_N = 32;
constexpr int64_t C0_SIZE_32 = 32;
constexpr int64_t C0_SIZE_64 = 64;
constexpr int64_t CUBE_BLOCK = 16;

class GMMFinalizeRoutingWeightQuantChecker {
public:
    GMMFinalizeRoutingWeightQuantChecker(gert::TilingContext* context, GQmmInputInfo& inputParams,
                                         GMMFRExtraInfo& frInfo)
        : context_(context),
          inputParams_(inputParams),
          frInfo_(frInfo)
    {}
    ~GMMFinalizeRoutingWeightQuantChecker() = default;

    ge::graphStatus CheckGroupedMatmulFinalizeRoutingWeightQuant(NpuArch npuArch, uint64_t aicNum, uint64_t aivNum);

private:
    bool CheckAttrs();
    bool CheckDtypeValid();
    bool CheckInputsDtype();
    bool CheckWeightFormat();
    bool CheckShapeValid();
    bool CheckYShape();
    bool CheckBiasShape();
    int64_t GetC0Size(ge::DataType dType) const;
    bool CheckWeightShape();
    bool CheckMxScaleShape();
    bool CheckKNAlign();
    bool CheckCoreNum(uint64_t aicNum, uint64_t aivNum);

    gert::TilingContext* context_;
    GQmmInputInfo& inputParams_;
    GMMFRExtraInfo& frInfo_;
    NpuArch npuArch_ = NpuArch::DAV_9201;
};

} // namespace GMMFinalizeRoutingWeightQuantChecker
} // namespace optiling
