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
 * \file mhc_pre_tiling_key.h
 * \brief
 */

#ifndef __OP_KERNEL_MHC_PRE_TILING_KEY_H__
#define __OP_KERNEL_MHC_PRE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MHC_PRE_SPLIT_BS 0
#define MHC_PRE_SPLIT_ND 1
#define MHC_PRE_SPLIT_M_K 2
#define MHC_PRE_M_K_WEIGHT_REUSE 3
#define MHC_PRE_HAS_RESI 0
#define MHC_PRE_NO_RESI 1

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
#define MHC_PRE_TPL_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_1
#else
#define MHC_PRE_TPL_KERNEL_TYPE ASCENDC_TPL_MIX_AIC_1_2
#endif

constexpr unsigned int MHC_PRE_BS_WAVE_D = 5120U;
constexpr unsigned int MHC_PRE_MK_SHARED_X_STAGE = 3U;
constexpr unsigned int MHC_PRE_MK_WEIGHT_REUSE_WAVE_STAGE = 4U;
constexpr unsigned int MHC_PRE_BS_X_QUEUE_MAX_BYTES = 80U * 1024U;
constexpr unsigned int MHC_PRE_BS_OUT_QUEUE_BYTES = 20U * 1024U;
constexpr unsigned int MHC_PRE_BS_DEFAULT_STAGE_ROWS = 32U;
constexpr unsigned int MHC_PRE_BS_EXTENDED_STAGE_ROWS = 48U;

// Preserve the original FP32 wave range when expanding the two-stage MK path.
constexpr unsigned int MHC_PRE_MK_WAVE_MAX_TOTAL_LENGTH = 49152U;
constexpr unsigned int MHC_PRE_MK_WAVE_PROJECTION_N = 32U;
constexpr unsigned int MHC_PRE_MK_PART2_ROWS = 32U;
constexpr unsigned int MHC_PRE_MK_PART2_INPUT_BYTES = 80U * 1024U;
constexpr unsigned int MHC_PRE_MK_PART2_OUTPUT_BYTES = 20U * 1024U;
constexpr unsigned int MHC_PRE_MK_PART2_TEMP_BYTES = 40U * 1024U;
constexpr unsigned int MHC_PRE_MK_FINE_PROJECTION_K = 256U;
constexpr unsigned int MHC_PRE_MK_FINE_FP32_ORDINARY_K = 128U;
constexpr unsigned int MHC_PRE_MK_FINE_REUSE_MAX_M = 4096U;
constexpr unsigned int MHC_PRE_MK_FINE_ORDINARY_MAX_M = 512U;
// Keep one row of finer projection partials inside the Part2 UB for the widest fusion layout.
constexpr unsigned int MHC_PRE_MK_FINE_ORDINARY_MAX_GROUPS = 128U;

ASCENDC_TPL_ARGS_DECL(MhcPre,
                      ASCENDC_TPL_UINT_DECL(TILING_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_BS,
                                            MHC_PRE_SPLIT_ND, MHC_PRE_SPLIT_M_K, MHC_PRE_M_K_WEIGHT_REUSE),
                      ASCENDC_TPL_UINT_DECL(HAS_RESI, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI,
                                            MHC_PRE_NO_RESI));

#if defined(MHC_PRE_ENABLE_ASCEND950)
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_BS),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_BS),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_M_K),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_M_K),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_M_K_WEIGHT_REUSE),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_M_K_WEIGHT_REUSE),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)));
#else
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_ND),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_ND),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_BS),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_BS),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_M_K),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_HAS_RESI)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(MHC_PRE_TPL_KERNEL_TYPE),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_SPLIT_M_K),
                                     ASCENDC_TPL_UINT_SEL(HAS_RESI, ASCENDC_TPL_UI_LIST, MHC_PRE_NO_RESI)));
#endif

#endif
