/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2_apt.cpp
 * \brief
 */
#include "arch35/moe_finalize_routing_v2_tiling_struct.h"
#include "arch35/moe_finalize_routing_v2_h_full_load.h"
#include "arch35/moe_finalize_routing_v2_h_partial_load.h"
#include "arch35/moe_finalize_routing_v2_row_k_h_full_load.h"
#include "arch35/moe_finalize_routing_v2_k_h_full_load.h"
// hif4（hifloat4）反量化仅 960（9201/9202）工具链支持，其余平台（如 950/3510）不编译
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
#include "arch35/moe_finalize_routing_v2_hif4_k_h_full_load.h"
#include "arch35/moe_finalize_routing_v2_hif4_h_full_load.h"
#include "arch35/moe_finalize_routing_v2_hif4_h_partial_load.h"
#include "arch35/moe_finalize_routing_v2_hif4_row_k_h_full_load.h"
#endif

#define TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_FP32 10000
#define TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_FP16 10001
#define TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_BF16 10002
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_FP32 10010
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_FP16 10011
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_BF16 10012
#define TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_FP32 10020
#define TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_FP16 10021
#define TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_BF16 10022
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_FP32 10030
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_FP16 10031
#define TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_BF16 10032
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_FP32 20000
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_FP16 20001
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_BF16 20002
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_FP32 20010
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_FP16 20011
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_BF16 20012
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_FP32 20020
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_FP16 20021
#define TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_BF16 20022
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_FP32 20030
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_FP16 20031
#define TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_BF16 20032
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_FP32 30000
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_FP16 30001
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_BF16 30002
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP32 30010
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP16 30011
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_BF16 30012
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_FP32 30020
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_FP16 30021
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_BF16 30022
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_FP32 30030
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_FP16 30031
#define TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_BF16 30032
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_FP32 40000
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_FP16 40001
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_BF16 40002
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP32 40010
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP16 40011
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_BF16 40012
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_FP32 40020
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_FP16 40021
#define TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_BF16 40022
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_FP32 40030
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_FP16 40031
#define TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_BF16 40032
#define TILING_KEY_HIF4_K_H_FULL_LOAD_DROPLESS_COLUMN 50000
#define TILING_KEY_HIF4_K_H_FULL_LOAD_DROP_PAD_COLUMN 50010
#define TILING_KEY_HIF4_K_H_FULL_LOAD_DROPLESS_ROW 50020
#define TILING_KEY_HIF4_K_H_FULL_LOAD_DROP_PAD_ROW 50030
#define TILING_KEY_HIF4_H_FULL_LOAD_DROPLESS_COLUMN 60000
#define TILING_KEY_HIF4_H_FULL_LOAD_DROP_PAD_COLUMN 60010
#define TILING_KEY_HIF4_H_FULL_LOAD_DROPLESS_ROW 60020
#define TILING_KEY_HIF4_H_FULL_LOAD_DROP_PAD_ROW 60030
#define TILING_KEY_HIF4_H_PARTIAL_LOAD_DROPLESS_COLUMN 70000
#define TILING_KEY_HIF4_H_PARTIAL_LOAD_DROP_PAD_COLUMN 70010
#define TILING_KEY_HIF4_H_PARTIAL_LOAD_DROPLESS_ROW 70020
#define TILING_KEY_HIF4_H_PARTIAL_LOAD_DROP_PAD_ROW 70030
#define TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN 80000
#define TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN 80010
#define TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROPLESS_ROW 80020
#define TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROP_PAD_ROW 80030

#define TILING_KEY_MX4_K_BATCH_DROPLESS_COLUMN 130000
#define TILING_KEY_MX4_K_BATCH_DROP_PAD_COLUMN 130010
#define TILING_KEY_MX4_K_BATCH_DROPLESS_ROW 130020
#define TILING_KEY_MX4_K_BATCH_DROP_PAD_ROW 130030
#define TILING_KEY_MX4_H_PARTIAL_LOAD_DROPLESS_COLUMN 140000
#define TILING_KEY_MX4_H_PARTIAL_LOAD_DROP_PAD_COLUMN 140010
#define TILING_KEY_MX4_H_PARTIAL_LOAD_DROPLESS_ROW 140020
#define TILING_KEY_MX4_H_PARTIAL_LOAD_DROP_PAD_ROW 140030
#define TILING_KEY_MX4_K_H_FULL_LOAD_DROPLESS_COLUMN 150000
#define TILING_KEY_MX4_K_H_FULL_LOAD_DROP_PAD_COLUMN 150010
#define TILING_KEY_MX4_K_H_FULL_LOAD_DROPLESS_ROW 150020
#define TILING_KEY_MX4_K_H_FULL_LOAD_DROP_PAD_ROW 150030
#define TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN 160000
#define TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN 160010
#define TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROPLESS_ROW 160020
#define TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROP_PAD_ROW 160030

using namespace MoeFinalizeRoutingV2Regbase;

#define MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HFullLoad<INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE> op; \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, alpha1, alpha2, v, y, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HPartialLoad<INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE> op; \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, alpha1, alpha2, v, y, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2RowKHFullLoad<INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE> op; \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, alpha1, alpha2, v, y, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2KHFullLoad<INPUT_TYPE, SCALE_DTYPE, DROP_PAD_MODE> op; \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, alpha1, alpha2, v, y, userWS, \
                &tilingData, &tPipe); \
        op.Process(); \
    } while (0)

// hif4暂不适配空专家特性
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
#define MOE_FINALIZE_ROUTING_V2_HIF4_KH_FULLLOAD_IMPL(DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2Hif4KHFullLoad<DROP_PAD_MODE> op(&tPipe, &tilingData); \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, y, userWS); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_HIF4_H_FULLLOAD_IMPL(DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2Hif4HFullLoad<DROP_PAD_MODE> op(&tPipe, &tilingData); \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, y, userWS); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_HIF4_H_PARTIALLOAD_IMPL(DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2Hif4HPartialLoad<DROP_PAD_MODE> op(&tPipe, &tilingData); \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, y, userWS); \
        op.Process(); \
    } while (0)

#define MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULLLOAD_IMPL(DROP_PAD_MODE) \
    do { \
        MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2Hif4RowKHFullLoad<DROP_PAD_MODE> op(&tPipe, &tilingData); \
        op.Init(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, y, userWS); \
        op.Process(); \
    } while (0)
#endif

extern "C" __global__ __aicore__ void moe_finalize_routing_v2(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1,
                                                              GM_ADDR x2, GM_ADDR bias, GM_ADDR scales,
                                                              GM_ADDR expertIdx, GM_ADDR x, GM_ADDR alpha1,
                                                              GM_ADDR alpha2, GM_ADDR v, GM_ADDR expandedXScale,
                                                              GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = nullptr;
    // MXFP4 uses only UB buffers; zero workspace is valid for this path.
    if constexpr (!IsSameType<DTYPE_EXPANDED_X, fp4x2_e2m1_t>::value) {
        userWS = GetUserWorkspace(workspace);
        if (userWS == nullptr) {
            return;
        }
    }

    REGISTER_TILING_DEFAULT(MoeFinalizeRoutingV2RegbaseTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe tPipe;
#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP4
    if constexpr (IsSameType<DTYPE_EXPANDED_X, fp4x2_e2m1_t>::value) {
        if (TILING_KEY_IS(TILING_KEY_MX4_K_BATCH_DROPLESS_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_COLUMN> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_BATCH_DROP_PAD_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_COLUMN> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_BATCH_DROPLESS_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_BATCH_DROP_PAD_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_H_PARTIAL_LOAD_DROPLESS_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HPartialLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_H_PARTIAL_LOAD_DROP_PAD_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HPartialLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_H_PARTIAL_LOAD_DROPLESS_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HPartialLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_H_PARTIAL_LOAD_DROP_PAD_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2HPartialLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_H_FULL_LOAD_DROPLESS_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2KHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_H_FULL_LOAD_DROP_PAD_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2KHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_H_FULL_LOAD_DROPLESS_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2KHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_K_H_FULL_LOAD_DROP_PAD_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2KHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_ROW> op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2RowKHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2RowKHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_COLUMN>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROPLESS_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2RowKHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROPLESS_ROW>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_MX4_ROW_K_H_FULL_LOAD_DROP_PAD_ROW)) {
            MoeFinalizeRoutingV2Regbase::MoeFinalizeRoutingV2RowKHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, DROP_PAD_ROW>
                op;
            op.Init(expandedX, expandedRowIdx, x1, x2, bias, expandedXScale, scales, expertIdx, y, userWS, &tilingData,
                    &tPipe);
            op.Process();
        }
        return;
    }
#endif
    // 适配hif4：部分函数模板无法实例化fp4类型，因此需要区分编译
    if constexpr (IsSameType<DTYPE_EXPANDED_X, float>::value || IsSameType<DTYPE_EXPANDED_X, half>::value ||
                  IsSameType<DTYPE_EXPANDED_X, bfloat16_t>::value) {
        if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROPLESS_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_FULL_LOAD_DROP_PAD_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROPLESS_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_H_PARTIAL_LOAD_DROP_PAD_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_H_PARTIALLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROPLESS_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_ROW_K_H_FULL_LOAD_DROP_PAD_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_ROWKH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP32)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_FP16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_COLUMN_BF16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROPLESS_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_FP32)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, float, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_FP16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, half, DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_K_H_FULL_LOAD_DROP_PAD_ROW_BF16)) {
            MOE_FINALIZE_ROUTING_V2_KH_FULLLOAD_IMPL(DTYPE_EXPANDED_X, bfloat16_t, DROP_PAD_ROW);
        }
    }
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
    else if constexpr (IsSameType<DTYPE_EXPANDED_X, fp4x2_e1m2_t>::value) {
        if (TILING_KEY_IS(TILING_KEY_HIF4_K_H_FULL_LOAD_DROPLESS_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_KH_FULLLOAD_IMPL(DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_K_H_FULL_LOAD_DROP_PAD_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_KH_FULLLOAD_IMPL(DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_K_H_FULL_LOAD_DROPLESS_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_KH_FULLLOAD_IMPL(DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_K_H_FULL_LOAD_DROP_PAD_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_KH_FULLLOAD_IMPL(DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_FULL_LOAD_DROPLESS_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_FULLLOAD_IMPL(DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_FULL_LOAD_DROP_PAD_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_FULLLOAD_IMPL(DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_FULL_LOAD_DROPLESS_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_FULLLOAD_IMPL(DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_FULL_LOAD_DROP_PAD_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_FULLLOAD_IMPL(DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_PARTIAL_LOAD_DROPLESS_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_PARTIALLOAD_IMPL(DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_PARTIAL_LOAD_DROP_PAD_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_PARTIALLOAD_IMPL(DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_PARTIAL_LOAD_DROPLESS_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_PARTIALLOAD_IMPL(DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_H_PARTIAL_LOAD_DROP_PAD_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_H_PARTIALLOAD_IMPL(DROP_PAD_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROPLESS_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULLLOAD_IMPL(DROPLESS_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROP_PAD_COLUMN)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULLLOAD_IMPL(DROP_PAD_COLUMN);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROPLESS_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULLLOAD_IMPL(DROPLESS_ROW);
        } else if (TILING_KEY_IS(TILING_KEY_HIF4_ROW_K_H_FULL_LOAD_DROP_PAD_ROW)) {
            MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULLLOAD_IMPL(DROP_PAD_ROW);
        }
    }
#endif
    tPipe.Destroy();
}
