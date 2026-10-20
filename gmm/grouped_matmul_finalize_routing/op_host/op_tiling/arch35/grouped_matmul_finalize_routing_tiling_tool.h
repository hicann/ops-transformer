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
 * \file grouped_matmul_finalize_routing_tiling_tool.h
 * \brief 输入输出信息提取与公共校验工具
 */

#pragma once

#include <cstdint>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include "op_host/tiling_base.h"
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace optiling {

namespace GMMFRTilingConstants {
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

constexpr int32_t SPLIT_M = 0;
constexpr size_t DIM_NUM_2 = 2;
constexpr size_t DIM_NUM_3 = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr uint32_t DIM_NUM_ROW_INDEX = 1;
constexpr uint32_t DIM_NUM_LOGIT = 1;
constexpr uint32_t OUT_DTYPE_BF16_INDEX = 2;
constexpr int64_t MX_SCALE_BLOCK_SIZE = 64;
constexpr int64_t MX_SCALE_LAST_DIM = 2;
constexpr int64_t MX_HIF4_SCALE_LAST_DIM = 1;
} // namespace GMMFRTilingConstants

namespace GMMFRTilingTools {

using namespace GMMFRTilingConstants;

struct GMMFRExtraInfo {
    int8_t scaleType = 0;    // 0=float8e8m0, 1=fp32, 2=bf16
    int8_t rowIndexType = 0; // 0=int64, 1=int32
    int64_t outputDtype = 0;
    int64_t outputBs = 0;
    int64_t sharedInputOffset = 0;
    int64_t sharedInputLen = 0;
    float sharedInputWeight = 1.0f;
    bool hasSharedInput = false;
};

// 场景识别
bool IsMxFp8Fp4(const GQmmInputInfo& inputParams);
bool IsMxFp8Hif4(const GQmmInputInfo& inputParams);

// 前置非空校验，最先调用，后续函数不需要重复判空
ge::graphStatus CheckDescNotNull(gert::TilingContext* context);
ge::graphStatus CheckShapeNotNull(gert::TilingContext* context);

// 信息提取
ge::graphStatus GetDtype(gert::TilingContext* context, GQmmInputInfo& inputParams);
ge::graphStatus GetAttrs(gert::TilingContext* context, GQmmInputInfo& inputParams);
ge::graphStatus GetInputs(gert::TilingContext* context, GQmmInputInfo& inputParams);
ge::graphStatus GetFrInfo(gert::TilingContext* context, const GQmmInputInfo& inputParams, GMMFRExtraInfo& frInfo);

// 输入信息综合提取：前置非空校验 + dtype/attr/input/frInfo 信息提取
ge::graphStatus GetInputsInfo(gert::TilingContext* context, GQmmInputInfo& inputParams, GMMFRExtraInfo& frInfo);

// 通用 shape 校验工具
bool CheckInputTensorShape(const char* name, const gert::Shape& shape, const std::vector<int64_t>& expectedShape);

// 公共校验
bool CheckCommonInputsDtype(const GMMFRExtraInfo& frInfo, gert::TilingContext* context);
bool CheckFinalizeRoutingInputsShape(const GQmmInputInfo& inputParams, const GMMFRExtraInfo& frInfo,
                                     gert::TilingContext* context);
bool CheckSharedInput(const GQmmInputInfo& inputParams, const GMMFRExtraInfo& frInfo, gert::TilingContext* context);

} // namespace GMMFRTilingTools
} // namespace optiling
