/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_GRAD_DETER_FIXED_CANDIDATE_TYPES_H_
#define FLASH_ATTN_GRAD_DETER_FIXED_CANDIDATE_TYPES_H_

// Fixed candidate state; independent of Host context and final plan writers.
#include <cstdint>
#include "band/schedule_types.h"

namespace optiling {
namespace fag_pto_arch35 {
constexpr int64_t MAX_CORE_NUM = 36;
constexpr int64_t INPUT_FORMAT_TND = 4;

enum class DeterGqaBandScheduleMode : uint8_t {
    ORIGINAL = 0,
    PIECEWISE = 1
};

enum class DeterSparseType : uint32_t {
    DETER_DENSE = 2,
    DETER_CAUSAL = 3,
    DETER_BAND = 4
};

enum class SparseMode : uint32_t {
    NO_MASK = 0,
    LEFT_UP_CAUSAL = 2,
    RIGHT_DOWN_CAUSAL = 3,
    BAND = 4
};

enum class SparseType : uint8_t {
    UNSUPPORTED = 255
};

enum class SplitAxisEnum : uint8_t {
    BN2GS1S2 = 0
};

enum class ConstAxisTemplateNum : int64_t {
    NUM128 = 128
};

struct FixedCandidateState {
    int64_t aicNum = 0;
    int64_t d = 0;
    int64_t dv = 0;
    int64_t inputBytes = 2;
    uint64_t l2CacheSize = 0;
    bool hasAttenMask = false;
    int64_t b = 0;
    int64_t coreNum = 0;
    int64_t deterGqaBandGroups = 0;
    GqaBandSchedule deterGqaBandSchedule{};
    DeterGqaBandScheduleMode deterGqaBandScheduleMode = DeterGqaBandScheduleMode::ORIGINAL;
    int64_t deterGqaCausalRoundsPerGroup = 0;
    int64_t deterGqaCausalGroups = 0;
    int64_t deterMaxRound = 0;
    int64_t deterSparseType = 0;
    int64_t enableSwizzle = 0;
    int64_t g = 0;
    int64_t isDeterministic = 0;
    int64_t isSwizzle = 0;
    int64_t layoutType = 0;
    int64_t n2 = 0;
    int64_t s1 = 0;
    int64_t s1CvRatio = 0;
    int64_t s1Inner = 0;
    int64_t s1Outer = 0;
    int64_t s1Token = 0;
    int64_t s2 = 0;
    int64_t s2CvRatio = 0;
    int64_t s2Inner = 0;
    int64_t s2Outer = 0;
    int64_t s2Token = 0;
    int64_t sparseMode = 0;
    int64_t sparseType = 0;
    SplitAxisEnum splitAxis = SplitAxisEnum::BN2GS1S2;
    int64_t tailZeroCount = 0;
};

struct BlockScheduleResult {
    bool isSwizzle = false;
    DeterBandScheduleResult deterBandSchedule;
};

struct GQADenseScheduleResult {
    int64_t baseRound = 0;
    int64_t selectedRound = 0;
    int64_t basePeriod = 0;
    int64_t selectedPeriod = 0;
};
} // namespace fag_pto_arch35
} // namespace optiling

#endif // FLASH_ATTN_GRAD_DETER_FIXED_CANDIDATE_TYPES_H_
