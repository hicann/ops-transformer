/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_KEY_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_KEY_H

#include <cstdint>
#include "ascendc/host_api/tiling/template_argument.h"

namespace MegaMhc {
/** Compile-time transport selection shared by host planning and the scheduling templates. */
enum class ProjectionTransport : uint32_t {
    GmStage = 0U,
    UbToL1 = 1U
};
constexpr uint32_t kProjectionColumns = 24U;
constexpr uint32_t kProjectionSlots = 2U;
/** Maximum simultaneously live M accumulators; L0C and FIX_M provide at most eight slots. */
constexpr uint32_t kMaxProjectionMStages = 8U;
constexpr ProjectionTransport kNoMidProjectionTransport = ProjectionTransport::UbToL1;
constexpr ProjectionTransport kWithMidProjectionTransport = ProjectionTransport::UbToL1;
} // namespace MegaMhc

#define MEGA_MHC_SPLIT_M_K 0
#define MEGA_MHC_SPLIT_BS 2

// Append SKIP_RMS0 so calls supplying gamma0 retain their existing tiling keys.
ASCENDC_TPL_ARGS_DECL(MegaMhc,
                      ASCENDC_TPL_UINT_DECL(TILING_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, MEGA_MHC_SPLIT_M_K,
                                            MEGA_MHC_SPLIT_BS),
                      ASCENDC_TPL_BOOL_DECL(WITH_MID, 0, 1), ASCENDC_TPL_BOOL_DECL(USE_HF32, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(SKIP_RMS0, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MEGA_MHC_SPLIT_M_K,
                                                          MEGA_MHC_SPLIT_BS),
                                     ASCENDC_TPL_BOOL_SEL(WITH_MID, 0, 1), ASCENDC_TPL_BOOL_SEL(USE_HF32, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(SKIP_RMS0, 0, 1)));

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_TILING_KEY_H
