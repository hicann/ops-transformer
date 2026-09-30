/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_H
#define OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_H

#include <cstddef>
#include <cstdint>

#include "opdev/op_executor.h"

namespace gmm_swiglu_quant_v3 {
constexpr float DEFAULT_CLAMP_LIMIT = 7.0F;
constexpr float DEFAULT_GLU_ALPHA = 1.702F;
constexpr float DEFAULT_GLU_BIAS = 1.0F;
constexpr size_t SINGLE_TENSOR_SIZE = 1;
constexpr size_t FIRST_TENSOR_INDEX = 0;
constexpr size_t X_DIM_NUM = 2;
constexpr size_t X_SCALE_DIM_NUM = 3;
constexpr size_t WEIGHT_VIEW_DIM_NUM = 3;
constexpr size_t WEIGHT_SCALE_DIM_NUM = 4;
constexpr size_t GROUP_LIST_DIM_NUM = 1;
constexpr size_t OUTPUT_DIM_NUM = 2;
constexpr size_t OUTPUT_SCALE_DIM_NUM = 3;
constexpr size_t WEIGHT_STORAGE_DIM_NUM = 5;

constexpr int64_t MX_GROUP_SIZE = 64;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr int64_t NZ_INNER_SIZE = 16;
constexpr int64_t NZ_C0_SIZE = 32;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr int64_t MXFP8_N_ALIGN = 64;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr int64_t QUANT_MODE_MX = 2;
constexpr int64_t DEQUANT_DTYPE_FLOAT = 0;
constexpr int64_t SCALE_ALG_OCP = 0;
constexpr int64_t SCALE_ALG_CUBLAS = 1;
constexpr double DEFAULT_DST_TYPE_MAX = 0.0;
constexpr char DEFAULT_ROUND_MODE[] = "rint";
} // namespace gmm_swiglu_quant_v3

namespace l0op {

const std::tuple<aclTensor*, aclTensor*> GroupedMatmulSwigluQuantV2(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale, const aclTensor* xScale,
    const aclTensorList* weightAssistanceMatrix, const aclTensor* bias, const aclTensor* smoothScale,
    const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t quantDtype,
    bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfigOptional, aclOpExecutor* executor);

const std::tuple<aclTensor*, aclTensor*> GroupedMatmulSwigluQuantV2(
    const aclTensor* x, const aclTensorList* weight, const aclTensorList* weightScale, const aclTensor* xScale,
    const aclTensorList* weightAssistanceMatrix, const aclTensor* bias, const aclTensor* smoothScale,
    const aclTensor* groupList, int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t quantDtype,
    bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfigOptional, int64_t swigluMode,
    float clampLimit, float gluAlpha, float gluBias, const char* roundMode, int64_t scaleAlg, float dstTypeMax,
    aclOpExecutor* executor);
} // namespace l0op

#endif
