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
 * \file text_gated_residual_grad_constants.h
 * \brief Shared TextGatedResidualGrad architecture, tiling and buffer constants.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_CONSTANTS_H
#define TEXT_GATED_RESIDUAL_GRAD_CONSTANTS_H

#include <cstdint>

// Architecture IDs must also be available to preprocessor branches.
#define HC_GRAD_ARCH_A2 2201
#define HC_GRAD_ARCH_A5 3510
#define HC_GRAD_CCE_A5 310
#ifndef HC_GRAD_NPU_ARCH
#if defined(__NPU_ARCH__)
#define HC_GRAD_NPU_ARCH __NPU_ARCH__
#else
#define HC_GRAD_NPU_ARCH 0
#endif
#endif

namespace TextGatedResidualGrad {

// Shared host/kernel geometry and workspace alignment.
constexpr uint32_t HC_GRAD_TASK_ROWS = 1024U;
constexpr uint32_t HC_GRAD_GATE_ALIGN = 16U;
constexpr uint32_t HC_GRAD_MAX_AIV = 64U;
constexpr uint32_t HC_GRAD_RING_SLOTS = 3U;
constexpr uint64_t HC_GRAD_WS_ALIGN = 512U;
constexpr uint32_t HC_GRAD_NO_TASK = 0xFFFFFFFFU;
// All supported architectures read dup directly as ColumnMajor by default.
constexpr bool HC_GRAD_USE_DUP_T = false;
constexpr uint32_t HC_GRAD_CUBE_TILE_M_A2 = 128U;
constexpr uint32_t HC_GRAD_CUBE_TILE_M_A5 = 256U;
constexpr uint32_t HC_GRAD_C1A_TILE_N = 128U;
constexpr uint32_t HC_GRAD_CUBE_TILE_K = 64U;
constexpr uint32_t HC_GRAD_CUBE_TILE_N_A5 = 256U;
constexpr uint32_t HC_GRAD_SWIZZLE_OFFSET = 3U;
constexpr uint32_t HC_GRAD_SWIZZLE_M_FIRST = 0U;
constexpr uint32_t HC_GRAD_SWIZZLE_N_FIRST = 1U;
constexpr uint32_t HC_GRAD_L1_STAGES = 4U;
constexpr uint32_t HC_GRAD_L0_STAGES = 2U;
constexpr uint32_t HC_GRAD_L0C_STAGES = 1U;
constexpr uint32_t HC_GRAD_A2_CUBE_UB_BYTES = 64U * 1024U;

// Scheduler capacity and conservative termination bounds, in beats.
constexpr uint32_t HC_GRAD_VECTOR_OPS_PER_BEAT = 3U;
constexpr uint32_t HC_GRAD_PAIRED_REDUCTION_TASKS = 2U;
constexpr uint32_t HC_GRAD_DIRECT_BEATS_PER_TASK = 14U;
constexpr uint32_t HC_GRAD_TRANSPOSE_BEATS_PER_TASK = 20U;
constexpr uint32_t HC_GRAD_DRAIN_BEATS = 32U;

// UB layouts: serial, V1 pipeline and V6 pipeline share one buffer pool.
constexpr uint32_t HC_GRAD_DATA_BLOCK_BYTES = 32U;
constexpr uint32_t HC_GRAD_VECTOR_ALIGN = 16U;
constexpr uint32_t HC_GRAD_VECTOR_ALIGN_MASK = HC_GRAD_VECTOR_ALIGN - 1U;
constexpr uint32_t HC_GRAD_DATA_BLOCK_MASK = HC_GRAD_DATA_BLOCK_BYTES - 1U;
constexpr uint32_t HC_GRAD_VECTOR_UB_A2 = 192U * 1024U;
constexpr uint32_t HC_GRAD_VECTOR_UB_FALLBACK = 248U * 1024U;
constexpr uint32_t HC_GRAD_VECTOR_EVENT = 0U;
constexpr uint32_t HC_GRAD_MAX_ROW_BLOCK = 128U;
// Input/output ping-pong; outputs reuse each stage's consumed input slots.
constexpr uint32_t HC_GRAD_PIPE_MAX_STAGES = 2U;
// Event 0 belongs to serial copies. Pipeline flags survive across iterations
// and must not be consumed by a serial SetFlag/WaitFlag pair.
constexpr int32_t HC_GRAD_PIPE_EVENT_BASE = 1;
constexpr uint32_t HC_GRAD_PIPE_STAGE_MASK = HC_GRAD_PIPE_MAX_STAGES - 1U;
constexpr uint32_t HC_GRAD_SERIAL_FLOAT_SLOTS = 6U;
constexpr uint32_t HC_GRAD_SERIAL_HALF_SLOTS = 3U;
constexpr uint32_t HC_GRAD_V1_SCRATCH_SLOTS = 3U;
constexpr uint32_t HC_GRAD_V1_FLOAT_SLOTS = 5U;
constexpr uint32_t HC_GRAD_V1_INPUT_SLOTS = 3U;
constexpr uint32_t HC_GRAD_V6_INPUT_SLOTS = 2U;
constexpr uint32_t HC_GRAD_V6_X_SLOT = 4U;
constexpr uint32_t HC_GRAD_V6_FLOAT_SLOTS = 5U;
constexpr uint32_t HC_GRAD_V6_ROW_SLOTS = 4U;
constexpr uint32_t HC_GRAD_SERIAL_POOL_BYTES =
    HC_GRAD_SERIAL_FLOAT_SLOTS * sizeof(float) + HC_GRAD_SERIAL_HALF_SLOTS * sizeof(uint16_t);
constexpr uint32_t HC_GRAD_V1_POOL_BYTES =
    HC_GRAD_V1_FLOAT_SLOTS * sizeof(float) + HC_GRAD_PIPE_MAX_STAGES * HC_GRAD_V1_INPUT_SLOTS * sizeof(uint16_t);
constexpr uint32_t HC_GRAD_V6_POOL_BYTES =
    HC_GRAD_V6_FLOAT_SLOTS * sizeof(float) + HC_GRAD_PIPE_MAX_STAGES * sizeof(uint16_t);
constexpr uint32_t HC_GRAD_V6_ROW_BYTES = HC_GRAD_V6_ROW_SLOTS * sizeof(float);
constexpr uint32_t HC_GRAD_SCALAR_SLOTS = 2U; // Sum and centered residual, each aligned to one data block.
constexpr uint32_t HC_GRAD_DGAMMA_SLOTS = 2U; // Kahan sum and compensation, per AIV.
constexpr uint32_t HC_GRAD_GAMMA_CACHE_MIN_TASKS = 2U;
constexpr uint32_t HC_GRAD_GAMMA_RESIDENT_BYTES = (HC_GRAD_DGAMMA_SLOTS + 1U) * sizeof(float);
constexpr uint32_t HC_GRAD_KAHAN_RESIDENT_BYTES = HC_GRAD_DGAMMA_SLOTS * sizeof(float);
constexpr uint32_t HC_GRAD_TRANSPOSE_TILE = 16U;
constexpr uint32_t HC_GRAD_TRANSPOSE_TILE_BUFFERS = 2U;
constexpr uint32_t HC_GRAD_TRANSPOSE_SCRATCH =
    HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TASK_ROWS * sizeof(uint16_t) +
    HC_GRAD_TRANSPOSE_TILE_BUFFERS * HC_GRAD_TRANSPOSE_TILE * HC_GRAD_TRANSPOSE_TILE * sizeof(uint16_t);
constexpr uint32_t HC_GRAD_SIGMOID_CHUNK = 512U;
constexpr uint32_t HC_GRAD_SIGMOID_BUFFERS = 2U; // Ones and denominator.
constexpr uint32_t HC_GRAD_SIGMOID_SCRATCH = HC_GRAD_SIGMOID_BUFFERS * HC_GRAD_SIGMOID_CHUNK * sizeof(float);
constexpr uint32_t HC_GRAD_DGAMMA_COPY_CHUNK = 32768U;
constexpr uint32_t HC_GRAD_WEIGHT_COPY_CHUNK = 1024U;
constexpr uint32_t HC_GRAD_DGAMMA_REDUCE_CHUNK = 1024U;
constexpr uint32_t HC_GRAD_ZERO_CHUNK = 8192U;
constexpr uint32_t HC_GRAD_DACT_ACCUM_SLOTS = 2U;
constexpr uint32_t HC_GRAD_DACT_REDUCE_ROWS = 64U;
constexpr uint32_t HC_GRAD_DGAMMA_FLUSH_BLOCKS = 4U;
constexpr float HC_GRAD_MAX_FINITE_FLOAT = 3.402823466e38F;
constexpr float HC_GRAD_INJECT_TO_SIGMOID = 0.5f;
constexpr float HC_GRAD_SIGMOID_TO_INJECT = 2.0f;

// Buffer indices describe lifetime sharing, rather than tensor dimensions.
constexpr uint32_t HC_GRAD_V1_X = 0U;
constexpr uint32_t HC_GRAD_V1_UP = 1U;
constexpr uint32_t HC_GRAD_V1_GRAD_MIXED = 2U;
constexpr uint32_t HC_GRAD_V1_XN = 3U;
constexpr uint32_t HC_GRAD_V1_GATE = 4U;
constexpr uint32_t HC_GRAD_V1_TEMP = 5U;
constexpr uint32_t HC_GRAD_HALF_INPUT = 0U;
constexpr uint32_t HC_GRAD_HALF_SECOND_INPUT = 1U;
// Shared by grad_mixed input and sequentially produced half outputs.
constexpr uint32_t HC_GRAD_HALF_IO_SCRATCH = 2U;
constexpr uint32_t HC_GRAD_V1_PIPE_X = 0U;
constexpr uint32_t HC_GRAD_V1_PIPE_GATE = 1U;
constexpr uint32_t HC_GRAD_V1_PIPE_TEMP = 2U;
constexpr uint32_t HC_GRAD_ACT_DOWN = 0U;
constexpr uint32_t HC_GRAD_ACT_SIGMOID = 1U;
constexpr uint32_t HC_GRAD_ACT_OUTPUT = 2U;
constexpr uint32_t HC_GRAD_V3_DACT = 0U;
constexpr uint32_t HC_GRAD_V3_DOWN = 1U;
constexpr uint32_t HC_GRAD_V3_SIGMOID = 2U;
constexpr uint32_t HC_GRAD_V3_DERIVATIVE = 3U;
constexpr uint32_t HC_GRAD_GATE_GRAD_INJECT = 0U;
constexpr uint32_t HC_GRAD_GATE_INJECT = 1U;
constexpr uint32_t HC_GRAD_GATE_TEMP = 2U;
constexpr uint32_t HC_GRAD_V6_DIRECT_INPUT = 0U;
constexpr uint32_t HC_GRAD_V6_DX2_INPUT = 1U;
constexpr uint32_t HC_GRAD_V6_SERIAL_TOTAL = 0U;
constexpr uint32_t HC_GRAD_V6_SERIAL_X = 1U;
constexpr uint32_t HC_GRAD_V6_SERIAL_GN = 2U;
constexpr uint32_t HC_GRAD_V6_SERIAL_PRODUCT = 3U;
constexpr uint32_t HC_GRAD_V6_SERIAL_CORRECTION = 4U;
constexpr uint32_t HC_GRAD_V6_SERIAL_REDUCE_WORK = 5U;
constexpr uint32_t HC_GRAD_V6_ROW_GN = 0U;
constexpr uint32_t HC_GRAD_V6_ROW_PRODUCT = 1U;
constexpr uint32_t HC_GRAD_V6_ROW_PENDING = 2U;
constexpr uint32_t HC_GRAD_V6_ROW_REDUCE_WORK = 3U;
constexpr uint32_t HC_GRAD_REDUCE_ACCUM = 0U;
constexpr uint32_t HC_GRAD_REDUCE_SLICE = 1U;
constexpr uint32_t HC_GRAD_REDUCE_COMP = 2U;
constexpr uint32_t HC_GRAD_REDUCE_TMP = 3U;
constexpr uint32_t HC_GRAD_REDUCE_TMP_SUM = 4U;
constexpr uint32_t HC_GRAD_DACT_REDUCE_SLICE = 4U;
constexpr uint32_t HC_GRAD_SCRATCH_BASE = 0U;

} // namespace TextGatedResidualGrad
#endif // TEXT_GATED_RESIDUAL_GRAD_CONSTANTS_H
