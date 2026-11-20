/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "arch35/engram_gate_tiling_data.h"
#include "arch35/engram_gate_arch35.h"
#include "arch35/engram_gate_pure_arch35.h"
#include "arch35/engram_gate_chunked_arch35.h"
extern "C" __global__ __aicore__ void engram_gate(GM_ADDR hidden_states, GM_ADDR kv, GM_ADDR weight_fused,
                                                  GM_ADDR output, GM_ADDR dot, GM_ADDR gate_score, GM_ADDR rstd_x,
                                                  GM_ADDR rstd_k, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(EngramGateTilingData);
    GET_TILING_DATA_WITH_STRUCT(EngramGateTilingData, data, tiling);
    if (data.rows == 0) {
        return;
    }
    if (data.pureSlots != 0) {
        constexpr uint8_t normalStore = AscendC::GetCacheModeBits<AscendC::CacheMode::CACHE_MODE_NORMAL>();
        constexpr uint8_t bypassStore = AscendC::GetCacheModeBits<AscendC::CacheMode::CACHE_MODE_DISABLE>();
        if (data.saveStats) {
            if (data.pureBypass) {
                EngramGatePure::EngramGateArch35<DTYPE_HIDDEN_STATES, true, bypassStore> op;
                op.Init(hidden_states, kv, weight_fused, output, dot, gate_score, rstd_x, rstd_k, &data);
                op.Process();
            } else {
                EngramGatePure::EngramGateArch35<DTYPE_HIDDEN_STATES, true, normalStore> op;
                op.Init(hidden_states, kv, weight_fused, output, dot, gate_score, rstd_x, rstd_k, &data);
                op.Process();
            }
        } else {
            if (data.pureBypass) {
                EngramGatePure::EngramGateArch35<DTYPE_HIDDEN_STATES, false, bypassStore> op;
                op.Init(hidden_states, kv, weight_fused, output, dot, gate_score, rstd_x, rstd_k, &data);
                op.Process();
            } else {
                EngramGatePure::EngramGateArch35<DTYPE_HIDDEN_STATES, false, normalStore> op;
                op.Init(hidden_states, kv, weight_fused, output, dot, gate_score, rstd_x, rstd_k, &data);
                op.Process();
            }
        }
    } else if (data.chunkedEnabled != 0) {
        EngramGateOps::EngramGateArch35Chunked<DTYPE_HIDDEN_STATES> op;
        op.Init(hidden_states, kv, weight_fused, output, dot, gate_score, rstd_x, rstd_k, &data);
        op.Process();
    }
}
