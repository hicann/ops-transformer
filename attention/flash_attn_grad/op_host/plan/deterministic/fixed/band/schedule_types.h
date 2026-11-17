/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_BAND_SCHEDULE_TYPES_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_BAND_SCHEDULE_TYPES_H_

// BAND candidate/index contracts; no dependency on Host plan state.
#include <cstdint>

namespace optiling {
namespace fag_pto_arch35 {
enum class DeterBandScheduleMode : uint8_t {
    DISABLED = 0,
    CAUSAL = 1,
    DENSE = 2,
    BAND = 3,
    CAUSAL_UPPER = 4,
    GROUPED = 5
};

struct DeterBandScheduleParams {
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

constexpr int64_t GQA_BAND_MAX_PHASES = 5;
// Wire format: full/tail phase arrays followed by scalar schedule fields.
constexpr int64_t GQA_BAND_PHASE_ARRAY_COUNT = 2;
constexpr int64_t GQA_BAND_PHASE_FIELD_COUNT = 7;
constexpr int64_t GQA_BAND_SCHEDULE_SCALAR_COUNT = 9;
// Distinct template selectors isolate the new scheduling arithmetic from legacy BAND kernels.
constexpr uint8_t GQA_BAND_COLUMN_OWNER_TEMPLATE = 5;
constexpr uint8_t GQA_BAND_BALANCED_OWNER_TEMPLATE = 6;
constexpr uint8_t GQA_BAND_TRANSLATED_OWNER_TEMPLATE = 7;
constexpr uint8_t GQA_BAND_FOLDED_OWNER_TEMPLATE = 8;
constexpr uint8_t GQA_BAND_OVERLAP_TEMPLATE = 9;

// Internal kernel dispatch values, distinct from tiling-key selectors and phase tags.
// Keep uint8_t values so existing kernel template instantiations remain unchanged.
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_GENERIC = 0;
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_COLUMN = 1;
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_BALANCED = 2;
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_TRANSLATED = 3;
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_FOLDED = 4;
constexpr uint8_t GQA_BAND_OWNER_DISPATCH_OVERLAP = 5;

// All serialized fields are explicitly eight bytes, including the phase tag.
enum class GqaBandPhaseKind : int64_t {
    RECTANGLE = 0,
    LOWER_TRIANGLE = 1,
    UPPER_TRIANGLE = 2,
    DIAGONAL = 3,
    EDGE_PAIR = 4,
    EDGE_PAIR_Q_BOUND = 5,
    EDGE_PAIR_P_BOUND = 6,
    RECTANGLE_BATCH_MAJOR = 7,
    DIAGONAL_COLUMN_OWNER = 8,
    // Whole-band column owner: rowOffset/colOffset carry normalized p/q, not translations.
    BAND_COLUMN_OWNER = 9,
    // Tag 10 was retired; preserve the remaining serialized tag values.
    LOWER_TRIANGLE_COLUMN_OWNER = 11,
    // Descending column-height order, folded across lanes; p/q use the offset fields.
    BAND_BALANCED_COLUMN_OWNER = 12,
    // Fold physical columns modulo m. Dimensions are normalized physical m/n.
    // Upper: rowOffset holds q (p=1), colOffset is a column translation.
    FOLDED_UPPER_COLUMN_OWNER = 13,
    // Lower: rowOffset is a row translation, colOffset holds p (q=1).
    FOLDED_LOWER_COLUMN_OWNER = 14,
    // Complete rectangle plus adjacent triangle. m is the rectangle height,
    // n is its width; the triangle side is m-1. Offsets are physical translations.
    OVERLAP_LOWER = 15,
    // Mirror the complete m-by-(n+m-1) lower domain in both axes before translation.
    OVERLAP_UPPER = 16,
    // Whole-band folded owner; rowOffset/colOffset carry normalized p/q.
    BAND_FOLDED_COLUMN_OWNER = 17
};

// Normalized calculation domain, not part of the serialized tiling layout.
// p/q are band bounds; rowOrigin/colOrigin translate local block coordinates.
struct GqaBandDomain {
    int64_t m;
    int64_t n;
    int64_t p;
    int64_t q;
    int64_t rowOrigin;
    int64_t colOrigin;
};

struct GqaBandPhase {
    GqaBandPhaseKind kind = GqaBandPhaseKind::RECTANGLE;
    int64_t m = 0;
    int64_t n = 0;
    int64_t rowOffset = 0;
    int64_t colOffset = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
};

// Final host-selected schedule. The device consumes it without rebuilding geometry.
// Unused phases are zeroed by the host. Zero tailPhaseCount reuses phases[].tailRounds.
struct GqaBandSchedule {
    GqaBandPhase phases[GQA_BAND_MAX_PHASES];
    GqaBandPhase tailPhases[GQA_BAND_MAX_PHASES];
    int64_t phaseCount = 0;
    int64_t tailPhaseCount = 0;
    int64_t k = 0;
    int64_t b = 0;
    int64_t g = 0;
    int64_t groupsPerChunk = 0;
    int64_t fullRounds = 0;
    int64_t tailRounds = 0;
    int64_t totalRounds = 0;
};

struct DeterBandScheduleResult {
    DeterBandScheduleMode mode = DeterBandScheduleMode::DISABLED;
    int64_t maxRound = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
    GqaBandSchedule groupedSchedule{};
};

static_assert(sizeof(GqaBandPhase) == GQA_BAND_PHASE_FIELD_COUNT * sizeof(int64_t),
              "GQA BAND phase wire layout must not contain padding");
static_assert(sizeof(GqaBandSchedule) ==
                  (GQA_BAND_PHASE_ARRAY_COUNT * GQA_BAND_MAX_PHASES * GQA_BAND_PHASE_FIELD_COUNT +
                   GQA_BAND_SCHEDULE_SCALAR_COUNT) *
                      sizeof(int64_t),
              "GQA BAND schedule wire layout must not contain padding");
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_BAND_SCHEDULE_TYPES_H_
