/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_test_constants.h
 * \brief Shared tensor indices and dimensions for TextGatedResidualGrad tests.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace HcGradTest {
enum TensorIndex : size_t {
    GRAD_MIXED,
    GRAD_INJECT,
    X,
    W_DOWN,
    W_UP,
    W_GATE,
    GAMMA,
    INV_RMS,
    INJECT,
    DOWN_PRE,
    UP,
    DX,
    DW_DOWN,
    DW_UP,
    DW_GATE,
    DGAMMA,
    TENSOR_COUNT
};
constexpr int64_t CHANNELS = 2;
constexpr int64_t HIDDEN = 16;
constexpr int64_t FEATURES = CHANNELS * HIDDEN;
constexpr int64_t PROJECTION = 16;
constexpr uint32_t GATE_GROUP_SIZE = 4;
constexpr uint32_t COMPLETE_GATE_MASK = 15;
constexpr uint32_t GAMMA_INPUT_MASK = 1;
constexpr uint32_t GAMMA_OUTPUT_MASK = 2;
constexpr uint32_t COMPLETE_GAMMA_MASK = 3;
} // namespace HcGradTest
