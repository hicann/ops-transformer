/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_TILING_DATA_INT8_H_
#define QUANT_FLASH_ATTN_TILING_DATA_INT8_H_

#include <cstdint>

// These values are passed to the AscendC template registry as integer macros.
// Keep them in sync with the kernel's compile-time dispatch and the tiling ABI.
#define QFA_INT8_QUANT_MODE 12U
#define QFA_INT8_TPL_BNSD 1U
#define QFA_INT8_TPL_TND 2U
#define QFA_INT8_TPL_BSND_OR_MIXED 3U
#define QFA_INT8_TPL_KV_LAYOUT 0U
#define QFA_INT8_CFG_PIPELINE_MASK 255U
#define QFA_INT8_CFG_WINDOW_FLAG 256U
#define QFA_INT8_CFG_LSE_FLAG 512U
#define QFA_INT8_CFG_WINDOW_LSE 768U
#define QFA_INT8_CFG_SHARED_QP 0U
#define QFA_INT8_CFG_SHARED_QP_LSE 512U
#define QFA_INT8_CFG_RESIDENT_Q 1U
#define QFA_INT8_CFG_RESIDENT_Q_LSE 513U
#define QFA_INT8_CFG_D64_M512 2U
#define QFA_INT8_CFG_D64_M512_LSE 514U
#define QFA_INT8_CFG_D128_M512 3U
#define QFA_INT8_CFG_D128_M512_LSE 515U
#define QFA_INT8_CFG_GENERIC_M512 4U
#define QFA_INT8_CFG_GENERIC_M512_LSE 516U
#define QFA_INT8_CFG_ISOLATED_SHARED_QP 9U
#define QFA_INT8_CFG_ISOLATED_SHARED_QP_LSE 521U
#define QFA_INT8_CFG_D96_M512_FIXPIPE 13U
#define QFA_INT8_CFG_D96_M512_FIXPIPE_LSE 525U
#define QFA_INT8_CFG_D64_M512_TASK_MAP 17U
#define QFA_INT8_CFG_D64_M512_TASK_MAP_LSE 529U
#define QFA_INT8_CFG_DYNAMIC_MHA 21U
#define QFA_INT8_CFG_DYNAMIC_MHA_LSE 533U
#define QFA_INT8_CFG_D96_M256_CAUSAL_MHA 22U
#define QFA_INT8_CFG_D96_M256_CAUSAL_MHA_LSE 534U
#define QFA_INT8_CFG_D128_M512_TASK_MAP 23U
#define QFA_INT8_CFG_D128_M512_TASK_MAP_LSE 535U

static_assert((QFA_INT8_CFG_SHARED_QP | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_SHARED_QP_LSE);
static_assert((QFA_INT8_CFG_RESIDENT_Q | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_RESIDENT_Q_LSE);
static_assert((QFA_INT8_CFG_D64_M512 | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D64_M512_LSE);
static_assert((QFA_INT8_CFG_D128_M512 | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D128_M512_LSE);
static_assert((QFA_INT8_CFG_GENERIC_M512 | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_GENERIC_M512_LSE);
static_assert((QFA_INT8_CFG_ISOLATED_SHARED_QP | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_ISOLATED_SHARED_QP_LSE);
static_assert((QFA_INT8_CFG_D96_M512_FIXPIPE | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D96_M512_FIXPIPE_LSE);
static_assert((QFA_INT8_CFG_D64_M512_TASK_MAP | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D64_M512_TASK_MAP_LSE);
static_assert((QFA_INT8_CFG_DYNAMIC_MHA | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_DYNAMIC_MHA_LSE);
static_assert((QFA_INT8_CFG_D96_M256_CAUSAL_MHA | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D96_M256_CAUSAL_MHA_LSE);
static_assert((QFA_INT8_CFG_D128_M512_TASK_MAP | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_D128_M512_TASK_MAP_LSE);
static_assert((QFA_INT8_CFG_WINDOW_FLAG | QFA_INT8_CFG_LSE_FLAG) == QFA_INT8_CFG_WINDOW_LSE);

namespace BaseApi::Int8GqaTnd {
constexpr int32_t QFA_LAYOUT_BNSD = 0;
constexpr int32_t QFA_LAYOUT_BSND = 1;
constexpr int32_t QFA_LAYOUT_TND = 2;
constexpr int32_t QFA_INVALID_LAYOUT = -1;
constexpr int64_t QFA_MASK_NONE = 0;
constexpr int64_t QFA_MASK_CAUSAL_ATTR = 3;
constexpr uint32_t QFA_MASK_CAUSAL_TILING = 1U;
constexpr int32_t QFA_UNBOUNDED_WINDOW = -1;
// Head sizes with dedicated dispatch or data-movement policies.
constexpr uint32_t QFA_HEAD_DIM_64 = 64U;
constexpr uint32_t QFA_HEAD_DIM_72 = 72U;
constexpr uint32_t QFA_HEAD_DIM_96 = 96U;
constexpr uint32_t QFA_HEAD_DIM_128 = 128U;
constexpr uint32_t QFA_HEAD_DIM_256 = 256U;
constexpr uint32_t QFA_MAX_HEAD_DIM = 256U;
constexpr uint32_t QFA_HEAD_DIM_ALIGNMENT = 32U;
constexpr uint32_t QFA_COMPACT_OUTPUT_ALIGNMENT = 8U;
constexpr uint32_t QFA_COMPACT_OUTPUT_MIN_DIM = 64U;
constexpr uint32_t QFA_MAX_AIC_CORES = 128U;
constexpr uint32_t QFA_INVALID_INDEX = UINT32_MAX;
constexpr uint32_t QFA_VECTOR_CORES_PER_CUBE = 2U;
constexpr uint32_t QFA_MIRRORED_TASK_PAIR_SIZE = 2U;
constexpr uint32_t QFA_CORE_WORK_SETUP_STAGES = 2U;
constexpr uint32_t QFA_CORE_RANGES_NONE = 0U;
constexpr uint32_t QFA_CORE_RANGES_CONTIGUOUS = 1U;
constexpr uint32_t QFA_CORE_RANGES_TASK_MAP = 2U;
constexpr uint32_t QFA_MIN_QUERY_TILE_ROWS = 64U;
constexpr uint32_t QFA_DEFAULT_QUERY_TILE_ROWS = 128U;
constexpr uint32_t QFA_LARGE_QUERY_TILE_ROWS = 512U;
constexpr uint32_t QFA_SMALL_KV_TILE_TOKENS = 128U;
constexpr uint32_t QFA_MEDIUM_KV_TILE_TOKENS = 256U;
constexpr uint32_t QFA_DEFAULT_KV_TILE_TOKENS = 512U;
constexpr uint32_t QFA_PV_OVERLAY_MIN_QUERY_TILE = 256U;
constexpr uint32_t QFA_LARGE_FP32_HEAD_DIM_THRESHOLD = 128U;
constexpr uint32_t QFA_LARGE_FP32_ACCUMULATOR_BYTES = 128U * 1024U;
constexpr uint32_t QFA_Q_L1_TILE_BYTES = 64U * 1024U;
constexpr uint32_t QFA_PV_OVERLAY_SLOTS = 2U;
constexpr uint32_t QFA_PIPELINE_SLOTS = 3U;
constexpr uint32_t QFA_K_SCALE_CACHE_TOKENS = 8192U;
constexpr uint32_t QFA_K_SCALE_PREFIX_PANELS = 32U;
constexpr uint16_t QFA_VEC_SCORE_PREFETCH = 1U << 0U;
constexpr uint16_t QFA_VEC_DIRECT_FIRST_PV = 1U << 1U;
constexpr uint16_t QFA_VEC_OVERLAP_COEFF = 1U << 2U;
constexpr uint16_t QFA_VEC_P_BROADCAST = 1U << 4U;
constexpr uint16_t QFA_VEC_D64_TASK_MAP_FEATURES =
    QFA_VEC_SCORE_PREFETCH | QFA_VEC_DIRECT_FIRST_PV | QFA_VEC_OVERLAP_COEFF;
constexpr uint16_t QFA_VEC_PREFILL_FEATURES = QFA_VEC_D64_TASK_MAP_FEATURES | QFA_VEC_P_BROADCAST;
} // namespace BaseApi::Int8GqaTnd

// A2/A3 mode 12. One task owns a grouped query M tile and scans all KV tiles. Each active AIC is paired with two
// AIV subcores and bounded FP16/P8 slots.
struct QuantFlashAttnTilingDataInt8 {
    uint32_t batchSize, numHeadsQ, numHeadsKv;
    uint32_t seqLenQ, seqLenKv, headDim, headDimAlign, groupSize;
    uint32_t blockM, blockN, usedCoreNum, totalTilesM, totalTasks;
    uint32_t scoreSlotElems, pSlotElems, pvSlotElems, coreWorkspaceBytes;
    float softmaxScale;
    uint32_t layoutQ, layoutKv, layoutOut; // 0 BNSD, 1 BSND, 2 TND
    uint32_t maskMode;                     // 0 dense, 1 bottom-right causal
    uint32_t qTokenStart;                  // populated per task for packed TND
    uint32_t kvTokenStart;
    uint32_t totalTokensQ;
    uint32_t totalTokensKv;
    // Internal eager alignment adapter may store only this leading dimension. Appended so all existing
    // configuration field offsets remain unchanged.
    uint32_t outputHeadDim;
    int32_t winLeft;
    int32_t winRight;
};

// Fixed tiling ABI: 28 integer words and one FP32 softmax scale.
static_assert(sizeof(QuantFlashAttnTilingDataInt8) == 28 * sizeof(uint32_t) + sizeof(float),
              "Unexpected INT8 GQA tiling ABI");

// Keep the optional core table outside the configuration copied into each Cube/Vector object. Its storage is
// constant and does not grow with sequence.
struct QuantFlashAttnTilingPayloadInt8 {
    QuantFlashAttnTilingDataInt8 config;
    uint32_t useCoreRanges;
    uint32_t coreTaskBegin[BaseApi::Int8GqaTnd::QFA_MAX_AIC_CORES + 1U];
};

constexpr uint32_t QFA_Q_BLOCK_TASK_CAPACITY = 512U;

// The task map keeps complete Q tasks on one core. Keys that use this payload append a bounded task-ID table.
struct QuantFlashAttnTilingQBlockPayloadInt8 {
    QuantFlashAttnTilingPayloadInt8 base;
    uint32_t taskIds[QFA_Q_BLOCK_TASK_CAPACITY];
};

static_assert(sizeof(QuantFlashAttnTilingQBlockPayloadInt8) ==
                  sizeof(QuantFlashAttnTilingPayloadInt8) + QFA_Q_BLOCK_TASK_CAPACITY * sizeof(uint32_t),
              "Unexpected INT8 GQA Q-block tiling ABI");

#endif // QUANT_FLASH_ATTN_TILING_DATA_INT8_H_
