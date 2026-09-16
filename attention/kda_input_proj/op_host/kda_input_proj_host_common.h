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
 * \file kda_input_proj_host_common.h
 * \brief Shared IR indices/names for KdaInputProj host (def / InferShape / tiling).
 */

#ifndef KDA_INPUT_PROJ_HOST_COMMON_H
#define KDA_INPUT_PROJ_HOST_COMMON_H

#include <cstdint>

namespace kda_input_proj {
// Inputs
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_QKV_INDEX = 1;
constexpr uint32_t WEIGHT_BETA_INDEX = 2;
constexpr uint32_t WEIGHT_GATE_INDEX = 3;
constexpr uint32_t WEIGHT_G_INDEX = 4;
constexpr uint32_t WEIGHT_QKV_SCALE_INDEX = 5;

// Outputs
constexpr uint32_t QKV_INDEX = 0;
constexpr uint32_t BETA_INDEX = 1;
constexpr uint32_t GATE_INDEX = 2;
constexpr uint32_t G_INDEX = 3;

// Attrs
constexpr uint32_t ATTR_TRANS_WEIGHT_QKV_INDEX = 0;
constexpr uint32_t ATTR_TRANS_WEIGHT_BETA_INDEX = 1;
constexpr uint32_t ATTR_TRANS_WEIGHT_GATE_INDEX = 2;
constexpr uint32_t ATTR_TRANS_WEIGHT_G_INDEX = 3;

// IR names (must match OpDef)
inline constexpr const char *X_NAME = "x";
inline constexpr const char *WEIGHT_QKV_NAME = "weight_qkv";
inline constexpr const char *WEIGHT_BETA_NAME = "weight_beta";
inline constexpr const char *WEIGHT_GATE_NAME = "weight_gate";
inline constexpr const char *WEIGHT_G_NAME = "weight_g";
inline constexpr const char *WEIGHT_QKV_SCALE_NAME = "weight_qkv_scale";

inline constexpr const char *QKV_NAME = "qkv";
inline constexpr const char *BETA_NAME = "beta";
inline constexpr const char *GATE_NAME = "gate";
inline constexpr const char *G_NAME = "g";

inline constexpr const char *ATTR_TRANS_WEIGHT_QKV_NAME = "trans_weight_qkv";
inline constexpr const char *ATTR_TRANS_WEIGHT_BETA_NAME = "trans_weight_beta";
inline constexpr const char *ATTR_TRANS_WEIGHT_GATE_NAME = "trans_weight_gate";
inline constexpr const char *ATTR_TRANS_WEIGHT_G_NAME = "trans_weight_g";
} // namespace kda_input_proj

#endif // KDA_INPUT_PROJ_HOST_COMMON_H
