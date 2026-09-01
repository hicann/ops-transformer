/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_

#include "kernel_operator.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h"

#ifndef QKV_K_SCALE_LAYOUT_NTD
#define QKV_K_SCALE_LAYOUT_NTD 0
#endif
#ifndef QKV_K_SCALE_LAYOUT_TND
#define QKV_K_SCALE_LAYOUT_TND 1
#endif
#ifndef QKV_K_SCALE_ROPE_MODE_ROPE
#define QKV_K_SCALE_ROPE_MODE_ROPE 0
#endif
#ifndef QKV_K_SCALE_ROPE_MODE_MROPE
#define QKV_K_SCALE_ROPE_MODE_MROPE 1
#endif
#ifndef QKV_K_SCALE_K_QUANT_MODE_FP8
#define QKV_K_SCALE_K_QUANT_MODE_FP8 0
#endif
#ifndef QKV_K_SCALE_K_QUANT_MODE_INT8
#define QKV_K_SCALE_K_QUANT_MODE_INT8 1
#endif
#ifndef QKV_K_SCALE_CACHE_DTYPE_FP8_E4M3FN
#define QKV_K_SCALE_CACHE_DTYPE_FP8_E4M3FN QKV_K_SCALE_K_QUANT_MODE_FP8
#endif
#ifndef QKV_K_SCALE_CACHE_DTYPE_INT8
#define QKV_K_SCALE_CACHE_DTYPE_INT8 QKV_K_SCALE_K_QUANT_MODE_INT8
#endif
#ifndef QKV_K_SCALE_Q_QUANT_MODE_PER_TOKEN_PER_HEAD
#define QKV_K_SCALE_Q_QUANT_MODE_PER_TOKEN_PER_HEAD 0
#endif
#ifndef QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT
#define QKV_K_SCALE_Q_QUANT_MODE_NO_QUANT 1
#endif
#ifndef QKV_K_SCALE_Q_QUANT_MODE_MX
#define QKV_K_SCALE_Q_QUANT_MODE_MX 2
#endif
#ifndef QKV_K_SCALE_K_QUANT_MODE_PER_TOKEN_PER_HEAD
#define QKV_K_SCALE_K_QUANT_MODE_PER_TOKEN_PER_HEAD 0
#endif
#ifndef QKV_K_SCALE_K_QUANT_MODE_MX
#define QKV_K_SCALE_K_QUANT_MODE_MX 1
#endif
namespace QkvRmsNormRopeCacheWithKScale {
using AscendC::CO2Layout;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::Fixpipe;
using AscendC::FixpipeConfig;
using AscendC::FixpipeParamsC310;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LoadData;
using AscendC::LoadData2DParamsV2;
using AscendC::LocalTensor;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Nd2NzParams;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint32_t QKV_K_SCALE_MIX_AIV_PER_AIC = 2U;
constexpr uint32_t QKV_K_SCALE_HEAD_DIM_D128 = 128U;
constexpr uint32_t QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV = 8U;
constexpr uint32_t QKV_K_SCALE_NZ_C0 = 16U;
constexpr uint32_t QKV_K_SCALE_QK_PREPROCESS_UB_NZ_STRIDE_ALIGN = QKV_K_SCALE_NZ_C0;
constexpr uint32_t QKV_K_SCALE_BLOCK_BYTES = 32U;
constexpr uint32_t QKV_K_SCALE_KIB = 1024U;

constexpr uint8_t QKV_K_SCALE_CROSS_CORE_SYNC_MODE = 4U;
constexpr uint64_t QKV_K_SCALE_AIV1_FLAG_OFFSET = 16U;
constexpr uint64_t SYNC_A_READY = 0U;
constexpr uint64_t SYNC_FIX_OUTPUT_READY = 1U;
constexpr uint64_t SYNC_FIX_OUTPUT_CONSUMED = 2U;
constexpr uint64_t SYNC_A_CONSUMED = 3U;
constexpr uint64_t SYNC_MROPE_K_CONSUMED = 4U;

constexpr uint32_t QKV_K_SCALE_ROPE_INPUT_ONE_BUFFER_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_OUTPUT_ONE_BUFFER_BYTES = 64U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_INPUT_POOL_BYTES = 80U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_BYTES = 128U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_ROPE_INPUT_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_ROPE_INPUT_ONE_BUFFER_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_INPUT_POOL_ELEMENTS = QKV_K_SCALE_INPUT_POOL_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS = QKV_K_SCALE_OUTPUT_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS = QKV_K_SCALE_OUTPUT_DB_POOL_BYTES / sizeof(float);

constexpr uint32_t QKV_K_SCALE_ROTATION_ONE_L1_BYTES =
    QKV_K_SCALE_HEAD_DIM_D128 * QKV_K_SCALE_HEAD_DIM_D128 * sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_ROTATION_L1_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_ROTATION_RESERVED_L1_BYTES = 2U * QKV_K_SCALE_ROTATION_ONE_L1_BYTES;
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_POOL_OFFSET = QKV_K_SCALE_ROTATION_RESERVED_L1_BYTES;
constexpr uint32_t QKV_K_SCALE_ROTATION_ONE_L1_ELEMENTS = QKV_K_SCALE_ROTATION_ONE_L1_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS =
    QKV_K_SCALE_MIX_AIV_PER_AIC * 64U * QKV_K_SCALE_KIB / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS = 2U * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS;

constexpr uint32_t QKV_K_SCALE_RESERVE_UB_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_GAMMA_UB_BYTES = 1U * QKV_K_SCALE_KIB;
// RoPE consumes one D-wide row per token.  M-RoPE keeps the raw T/H/W rows
// in the same ping-pong slot, so reserve three rows per token without
// changing the top-level UB partition.
constexpr uint32_t QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES = 6U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES = 512U;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_BYTES = 512U;
constexpr uint32_t QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS = QKV_K_SCALE_BLOCK_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS = QKV_K_SCALE_HEAD_DIM_D128;
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES =
    2U * QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS * sizeof(uint16_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_ONE_BUFFER_BYTES = 10U * QKV_K_SCALE_KIB;
// Keep the scene-local position slot valid for the generic eight-token AIV
// tile limit: 8 tokens * 3 axes * sizeof(int32_t), rounded to a stable 128 B
// transfer slot.  Current host tiling uses a smaller cap, but the layout must
// remain safe if that cap is raised later.
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_BYTES = 128U;
constexpr uint32_t QKV_K_SCALE_MROPE_GATHER_INDEX_BYTES = (QKV_K_SCALE_HEAD_DIM_D128 / 2U) * sizeof(uint32_t);

constexpr uint32_t QKV_K_SCALE_GAMMA_UB_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET = QKV_K_SCALE_GAMMA_UB_OFFSET + QKV_K_SCALE_GAMMA_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET =
    QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET + 2U * QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_OFFSET =
    QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET + 2U * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET =
    QKV_K_SCALE_V_SCALE_UB_OFFSET + QKV_K_SCALE_V_SCALE_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_V_OUT_DB_POOL_OFFSET =
    QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET + QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_DB_POOL_OFFSET =
    QKV_K_SCALE_V_OUT_DB_POOL_OFFSET + 2U * QKV_K_SCALE_V_OUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_GATHER_INDEX_OFFSET =
    QKV_K_SCALE_MROPE_POSITION_DB_POOL_OFFSET + 2U * QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_GAMMA_UB_ELEMENTS = QKV_K_SCALE_GAMMA_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS = QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_COS_SIN_DB_POOL_ELEMENTS = 2U * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_DB_POOL_ELEMENTS = 2U * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_ELEMENTS = QKV_K_SCALE_V_SCALE_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_ELEMENTS =
    QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES / sizeof(uint16_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS = QKV_K_SCALE_V_OUT_ONE_BUFFER_BYTES / sizeof(fp8_e4m3fn_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_DB_POOL_ELEMENTS = 2U * QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_DB_POOL_ELEMENTS = 2U * QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_ELEMENTS;
static_assert(QKV_K_SCALE_MROPE_POSITION_ONE_BUFFER_BYTES >= QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV * 3U * sizeof(int32_t),
              "M-RoPE position slot is smaller than the maximum AIV token slice");
constexpr uint32_t QKV_K_SCALE_MROPE_GATHER_INDEX_ELEMENTS = QKV_K_SCALE_MROPE_GATHER_INDEX_BYTES / sizeof(uint32_t);
static_assert(QKV_K_SCALE_MROPE_GATHER_INDEX_OFFSET + QKV_K_SCALE_MROPE_GATHER_INDEX_BYTES <=
                  QKV_K_SCALE_RESERVE_UB_BYTES,
              "M-RoPE reserve layout exceeds the 40 KiB UB budget");

constexpr uint32_t QKV_K_SCALE_INPUT_POOL_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET = QKV_K_SCALE_INPUT_POOL_OFFSET + QKV_K_SCALE_INPUT_POOL_BYTES;
constexpr uint32_t QKV_K_SCALE_RESERVE_UB_OFFSET = QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET + QKV_K_SCALE_OUTPUT_DB_POOL_BYTES;

// Compact direct-M-RoPE layout.  Valid M-RoPE selectors use this layout;
// host tiling reduces tokenTile when its per-tile resource limits are not met.
// The ordinary 248 KiB layout above remains the RoPE layout, not an M-RoPE
// fallback selector.
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_MAX_K_ROWS = 14U;
// The host contract requires Nq = 8 * Nk, Nk = Nv, and Nq <= 64.
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_MAX_KV_HEADS = 8U;
// Keep the existing 80 KiB input pool, but use four input and raw cos/sin
// slots for the compact M-RoPE path. Slot mapping and output pools remain
// double buffered; M-RoPE positions use the persistent window below.
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM = 4U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_INPUT_ONE_BUFFER_BYTES =
    QKV_K_SCALE_INPUT_POOL_BYTES / QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_INPUT_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_INPUT_ONE_BUFFER_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_BYTES =
    2U * QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_OUTPUT_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_FLOAT_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_BYTES / sizeof(float);

constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_MAX_K_ROWS * QKV_K_SCALE_HEAD_DIM_D128 * sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_BYTES =
    2U * QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_FLOAT_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_FLOAT_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_BYTES / sizeof(float);

constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_OFFSET = 0x0000U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_BYTES = 0x0400U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_OFFSET = 0x0400U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_MAX_KV_HEADS * QKV_K_SCALE_HEAD_DIM_D128 * sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_OFFSET = 0x1400U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_BYTES = 0x0200U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_GATHER_INDEX_UB_OFFSET = 0x1600U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_GATHER_INDEX_UB_BYTES = 0x0100U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_POSITION_DB_POOL_OFFSET = 0x1800U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_POSITION_DB_POOL_BYTES = 0x0100U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_POSITION_ONE_BUFFER_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_POSITION_DB_POOL_BYTES / 2U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_DB_POOL_OFFSET = 0x1900U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_DB_POOL_BYTES = 0x0040U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_ONE_BUFFER_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_DB_POOL_BYTES / 2U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_OFFSET = 0x1A00U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_BYTES = 0x6000U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_OFFSET = 0x7A00U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_BYTES = 0x0E00U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_OUT_ONE_BUFFER_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_BYTES / 2U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_OFFSET = 0x8800U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_BYTES = 0x1200U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_BYTES / 2U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_K_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_MAX_K_ROWS * QKV_K_SCALE_HEAD_DIM_D128 * sizeof(int8_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_SCALE_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_MAX_K_ROWS * QKV_K_SCALE_BLOCK_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RESERVE_END = 0xA000U;

constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_INPUT_POOL_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_OFFSET =
    QKV_K_SCALE_MROPE_COMPACT_INPUT_POOL_OFFSET + QKV_K_SCALE_INPUT_POOL_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET =
    QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_OFFSET + QKV_K_SCALE_MROPE_COMPACT_OUTPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_OFFSET =
    QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_OFFSET + QKV_K_SCALE_MROPE_COMPACT_K_ROTATION_DB_POOL_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_BASE_UB_BYTES =
    QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_OFFSET + QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_BYTES;

// Cache M-RoPE positions in bounded token windows. A single window is
// persistent while its tiles run, so position consumers need one MTE2_S
// boundary per window instead of one per tile. Larger core ranges are
// refilled in batches.
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_WINDOW_TOKEN_CAPACITY = 1024U;
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_CACHE_BYTES =
    QKV_K_SCALE_MROPE_POSITION_WINDOW_TOKEN_CAPACITY * 3U * sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_CACHE_OFFSET = QKV_K_SCALE_MROPE_COMPACT_BASE_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_MROPE_POSITION_CACHE_ELEMENTS = QKV_K_SCALE_MROPE_POSITION_CACHE_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_TOTAL_UB_BYTES =
    QKV_K_SCALE_MROPE_POSITION_CACHE_OFFSET + QKV_K_SCALE_MROPE_POSITION_CACHE_BYTES;

constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_GAMMA_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_BYTES / sizeof(uint16_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_POSITION_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_POSITION_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_POSITION_DB_POOL_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_POSITION_DB_POOL_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_DB_POOL_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_DB_POOL_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_OUT_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_V_OUT_ONE_BUFFER_BYTES / sizeof(fp8_e4m3fn_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_V_OUT_DB_POOL_BYTES / sizeof(fp8_e4m3fn_t);
constexpr uint32_t QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_ELEMENTS =
    QKV_K_SCALE_MROPE_COMPACT_K_QUANT_DB_POOL_BYTES / sizeof(int8_t);

static_assert(QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_K_BYTES +
                      QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_SCALE_BYTES <=
                  QKV_K_SCALE_MROPE_COMPACT_K_QUANT_ONE_BUFFER_BYTES,
              "compact M-RoPE K quant slot is too small");
static_assert(QKV_K_SCALE_INPUT_POOL_BYTES % QKV_K_SCALE_MROPE_COMPACT_INPUT_BUFFER_NUM == 0U,
              "compact M-RoPE input pool must divide evenly into four slots");
static_assert(QKV_K_SCALE_MROPE_COMPACT_RAW_COS_SIN_DB_POOL_BYTES % QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES == 0U,
              "compact M-RoPE raw cos/sin pool must contain whole slots");
static_assert(QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_OFFSET + QKV_K_SCALE_MROPE_COMPACT_V_SCALE_UB_BYTES <=
                  QKV_K_SCALE_MROPE_COMPACT_QK_NZ_SCATTER_INDEX_UB_OFFSET,
              "compact M-RoPE V scale overlaps the Q/K scatter index");
static_assert(QKV_K_SCALE_MROPE_COMPACT_SLOT_MAPPING_ONE_BUFFER_ELEMENTS >= QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV,
              "compact M-RoPE slot mapping slot is smaller than the maximum AIV token slice");
static_assert(QKV_K_SCALE_MROPE_COMPACT_RESERVE_END <= QKV_K_SCALE_MROPE_COMPACT_RESERVE_UB_BYTES,
              "compact M-RoPE reserve layout exceeds 40 KiB");
static_assert(QKV_K_SCALE_MROPE_POSITION_CACHE_OFFSET % QKV_K_SCALE_BLOCK_BYTES == 0U,
              "compact M-RoPE position cache must be 32-byte aligned");
static_assert(QKV_K_SCALE_MROPE_COMPACT_TOTAL_UB_BYTES <= 256U * QKV_K_SCALE_KIB,
              "compact M-RoPE layout exceeds physical UB capacity");

struct TileParam {
    uint64_t tokenOffset;
    uint64_t tokenSize;
    uint64_t cubeTokenSize;
    uint64_t cubeHalfTokenSize;
    uint64_t aivTokenOffset;
    uint64_t aivTokenSize;
    uint64_t aivBlockTokenOffset;
    uint64_t vHeadSize;
    uint64_t cacheBaseOffset[QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV];
    uint64_t scaleCacheBaseOffset[QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV];
};

struct GlobalTensors {
    GM_ADDR qkv;
    GM_ADDR qGamma;
    GM_ADDR kGamma;
    GM_ADDR cosSin;
    GM_ADDR slotMapping;
    GM_ADDR kCache;
    GM_ADDR vCache;
    GM_ADDR kScaleCache;
    GM_ADDR queryStartLoc;
    GM_ADDR seqLens;
    GM_ADDR rotation;
    GM_ADDR vScale;
    GM_ADDR mropePosition;
    GM_ADDR qOut;
    GM_ADDR qScale;
    GM_ADDR kCacheOut;
    GM_ADDR vCacheOut;
    GM_ADDR kScaleCacheOut;
    GM_ADDR workspace;
};

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t factor)
{
    if (factor == 0U) {
        return 0U;
    }
    return (value + factor - 1U) / factor;
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        return value;
    }
    return ((value + align - 1U) / align) * align;
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t MaxU64(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint64_t NzMatrixElements(uint64_t rowCount)
{
    return (QKV_K_SCALE_HEAD_DIM_D128 / QKV_K_SCALE_NZ_C0) * AlignUp(rowCount, QKV_K_SCALE_NZ_C0) * QKV_K_SCALE_NZ_C0;
}

template <typename T>
__aicore__ inline void DataCopyGmToUb2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint64_t rowCount,
                                        uint64_t colCount, uint64_t srcStride)
{
    if (rowCount == 0U || colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rowCount);
    params.blockLen = static_cast<uint32_t>(colCount * sizeof(T));
    params.srcStride = static_cast<decltype(params.srcStride)>((srcStride - colCount) * sizeof(T));
    params.dstStride = 0U;
    params.rsv = 0U;
    DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
    const uint32_t blockElements = QKV_K_SCALE_BLOCK_BYTES / sizeof(T);
    const uint64_t alignedColCount = AlignUp(colCount, blockElements);
    if (alignedColCount != colCount) {
        padParams.isPad = true;
        padParams.rightPadding = static_cast<uint8_t>(alignedColCount - colCount);
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyUbToGm2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint64_t rowCount,
                                        uint64_t colCount, uint64_t srcStride, uint64_t dstStride)
{
    if (rowCount == 0U || colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rowCount);
    params.blockLen = static_cast<uint32_t>(colCount * sizeof(T));
    params.srcStride =
        static_cast<decltype(params.srcStride)>((srcStride - colCount) * sizeof(T) / QKV_K_SCALE_BLOCK_BYTES);
    params.dstStride = static_cast<decltype(params.dstStride)>((dstStride - colCount) * sizeof(T));
    params.rsv = 0U;
    DataCopyPad(dst, src, params);
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
