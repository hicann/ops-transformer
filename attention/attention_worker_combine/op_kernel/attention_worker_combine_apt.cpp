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
 * \file attention_worker_combine_apt.cpp
 * \brief Ascend950 kernel entry for AttentionWorkerCombine
 */

#include "arch35/attention_worker_combine_mxfp_dequant.h"
#include "arch35/attention_worker_combine.h"
#include "arch35/attention_worker_combine_tiling_struct.h"
#include "kernel_operator.h"

#define TILING_KEY_DIVIDE_BS_FP16 10000UL
#define TILING_KEY_DIVIDE_BS_BF16 10001UL
#define TILING_KEY_DIVIDE_H_FP16 10010UL
#define TILING_KEY_DIVIDE_H_BF16 10011UL
#define TILING_KEY_DIVIDE_K_FP16 10020UL
#define TILING_KEY_DIVIDE_K_BF16 10021UL

#define TILING_KEY_DIVIDE_BS_MXFP8_E5M2 11002UL
#define TILING_KEY_DIVIDE_BS_MXFP8_E4M3 11003UL
#define TILING_KEY_DIVIDE_BS_MXFP4_E2M1 11004UL
#define TILING_KEY_DIVIDE_H_MXFP8_E5M2 11012UL
#define TILING_KEY_DIVIDE_H_MXFP8_E4M3 11013UL
#define TILING_KEY_DIVIDE_H_MXFP4_E2M1 11014UL
#define TILING_KEY_DIVIDE_K_MXFP8_E5M2 11022UL
#define TILING_KEY_DIVIDE_K_MXFP8_E4M3 11023UL
#define TILING_KEY_DIVIDE_K_MXFP4_E2M1 11024UL

extern "C" __global__ __aicore__ void attention_worker_combine(GM_ADDR schedule_context, GM_ADDR expert_scales,
                                                               GM_ADDR layer_id, GM_ADDR y, GM_ADDR next_layer_id,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::AttentionWorkerCombineRegbaseTilingData);
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_MXFP8_E5M2)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e5m2_t, false, AttentionWorkerCombineRegbase::MxfpSplit::BS>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_MXFP8_E4M3)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e4m3fn_t, false, AttentionWorkerCombineRegbase::MxfpSplit::BS>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_MXFP4_E2M1)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            uint8_t, true, AttentionWorkerCombineRegbase::MxfpSplit::BS>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_MXFP8_E5M2)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e5m2_t, false, AttentionWorkerCombineRegbase::MxfpSplit::K>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_MXFP8_E4M3)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e4m3fn_t, false, AttentionWorkerCombineRegbase::MxfpSplit::K>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_MXFP4_E2M1)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            uint8_t, true, AttentionWorkerCombineRegbase::MxfpSplit::K>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_MXFP8_E5M2)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e5m2_t, false, AttentionWorkerCombineRegbase::MxfpSplit::H>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_MXFP8_E4M3)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            fp8_e4m3fn_t, false, AttentionWorkerCombineRegbase::MxfpSplit::H>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_MXFP4_E2M1)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombineMxfpDequant<
            uint8_t, true, AttentionWorkerCombineRegbase::MxfpSplit::H>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_FP16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<half,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::BS>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_BS_BF16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<bfloat16_t,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::BS>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_FP16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<half,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::H>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_H_BF16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<bfloat16_t,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::H>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_FP16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<half,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::K>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIVIDE_K_BF16)) {
        AttentionWorkerCombineRegbase::KernelAttentionWorkerCombine<bfloat16_t,
                                                                    AttentionWorkerCombineRegbase::NonquantSplit::K>
            op;
        op.Init(schedule_context, expert_scales, layer_id, y, next_layer_id, workspace, &tiling_data, &pipe);
        op.Process();
    }
}
