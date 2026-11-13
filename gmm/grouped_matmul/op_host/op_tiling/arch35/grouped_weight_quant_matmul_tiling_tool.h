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
 * \file grouped_weight_quant_matmul_tiling_tool.h
 * \brief
 */
#pragma once

#include "log/log.h"
#include "../grouped_matmul_tiling.h"

namespace optiling {
namespace GMMWeightQuantConstants {
constexpr size_t X_INDEX = 0;
constexpr size_t WEIGHT_INDEX = 1;
constexpr size_t BIAS_INDEX = 2;
constexpr size_t SCALE_INDEX = 3;
constexpr size_t OFFSET_INDEX = 4;
constexpr size_t ANTIQUANT_SCALE_INDEX = 5;
constexpr size_t ANTIQUANT_OFFSET_INDEX = 6;
constexpr size_t GROUPLIST_INDEX = 7;
constexpr size_t PER_TOKEN_SCALE_INDEX = 8;
constexpr size_t Y_INDEX = 0;
constexpr size_t ATTR_INDEX_SPLIT_ITEM = 0;
constexpr size_t ATTR_INDEX_TRANS_W = 2;
constexpr size_t ATTR_INDEX_TRANS_X = 3;
constexpr size_t ATTR_INDEX_GROUPTYPE = 4;
constexpr size_t ATTR_INDEX_GROUP_LIST_TYPE = 5;
constexpr size_t ATTR_INDEX_ACT_TYPE = 6;
constexpr size_t ATTR_INDEX_TUNING_CONFIG = 7;

const std::string X = "x";
const std::string WEIGHT = "weight";
const std::string BIAS = "bias";
const std::string ANTIQUANT_SCALE = "antiquantScale";
const std::string ANTIQUANT_OFFSET = "antiquantOffset";
const std::string PER_TOKEN_SCALE = "perTokenScale";

constexpr uint16_t MAX_TENSOR_CONT = 128;

constexpr int64_t NO_SPLIT = -1L;
constexpr int64_t SPLIT_M = 0L;

constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256UL;
constexpr uint64_t BASIC_BLOCK_SIZE_240 = 240UL;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BASIC_BLOCK_SIZE_112 = 112UL;
constexpr uint64_t STEP_K_4 = 4UL;

constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32UL;
constexpr uint64_t L1_ALIGN_SIZE = 32UL;
constexpr uint64_t KB_UNIT = 1024UL;

constexpr size_t DIM_NUM_1 = 1;
constexpr size_t DIM_NUM_2 = 2;
constexpr size_t DIM_NUM_3 = 3;
constexpr size_t DIM_NUM_4 = 4;
constexpr size_t DIM_NUM_5 = 5;
constexpr size_t MIN_X_DIM = 2;
constexpr size_t MAX_X_DIM = 6;
constexpr size_t MIN_ND_DIM = 2;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;

constexpr int64_t N_K_ALIGN_VALUE_4BIT = 64L;
constexpr int64_t C0_64 = 64L;

constexpr int64_t MX_GROUP_SIZE_32 = 32L;
constexpr int64_t MX_GROUP_SIZE_64 = 64L;
constexpr int64_t MX_SCALE_LAST_DIM_1 = 1L;
constexpr int64_t MX_SCALE_LAST_DIM_2 = 2L;

constexpr uint64_t BYTE_BITS = 8UL;

} // namespace GMMWeightQuantConstants

namespace GMMWeightQuantTilingTool {

enum class AntiQuantMode : uint8_t {
    DEFAULT = 0,
    PERTENSOR_MODE = 1,
    PERCHANNEL_MODE = 2,
    PERGROUP_MODE = 3,
    MX_PERGROUP_MODE = 4,
};

struct GMMWeightQuantBasicTiling {
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
};

struct GMMWeightQuantInputInfo {
    const char* opName = nullptr;

    uint64_t mSize = 0UL;
    uint64_t kSize = 0UL;
    uint64_t nSize = 0UL;
    uint64_t groupNum = 0UL;

    AntiQuantMode weightQuantMode = AntiQuantMode::DEFAULT;

    int64_t groupType = GMMWeightQuantConstants::NO_SPLIT;
    int64_t splitItem = 0;
    int64_t groupListType = 0;
    uint16_t groupSize = 0;

    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType weightDtype = ge::DT_UNDEFINED;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DT_UNDEFINED;
    ge::DataType antiquantScaleDtype = ge::DT_UNDEFINED;
    ge::DataType antiquantOffsetDtype = ge::DT_UNDEFINED;
    ge::DataType perTokenScaleDtype = ge::DT_UNDEFINED;

    ge::Format weightFormat = ge::FORMAT_ND;

    bool enableMxShapeGeneralization = false; // 960DT GroupedMatmul only
    bool transX = false;
    bool transWeight = false;
    bool hasBias = false;
    bool hasAntiquantOffset = false;
    bool isSingleX = false;
    bool isSingleW = false;
    bool isSingleY = false;
    bool isWeightNz = false;
};

struct GMMWeightQuantInputNum {
    uint16_t numX = 0;
    uint16_t numWeight = 0;
    uint16_t numBias = 0;
    uint16_t numAntiquantScale = 0;
    uint16_t numAntiquantOffset = 0;
};

struct MKNDimInfo {
    int64_t gDim = 0L;
    int64_t mDim = 0L;
    int64_t xKDim = 0L;
    int64_t weightKDim = 0L;
    int64_t nDim = 0L;
};

uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype);

bool AnalyzeInputs(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                   GMMWeightQuantInputNum& inputNum);
void GetNumOfInputs(const gert::TilingContext* context, GMMWeightQuantInputNum& inputNum);
uint16_t GetTensorListSize(const gert::TilingContext* context, size_t inputIdx);
bool AnalyzeAttrs(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                  GMMWeightQuantInputNum& inputNum);
bool AnalyzeInputParams(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams,
                        GMMWeightQuantInputNum& inputNum);
bool AnalyzeGroupList(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams);
bool AnalyzeQuantInputParams(const gert::TilingContext* context, GMMWeightQuantInputInfo& inputParams);

bool IsMxA8W4(const GMMWeightQuantInputInfo& inputParams);
bool IsMxA8HiF4(const GMMWeightQuantInputInfo& inputParams);
bool IsA16MxFp4(const GMMWeightQuantInputInfo& inputParams);
bool IsA16MxHiF4(const GMMWeightQuantInputInfo& inputParams);

} // namespace GMMWeightQuantTilingTool

} // namespace optiling
