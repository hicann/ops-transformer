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
 * \file qkv_rms_norm_rope_cache_apt.cpp
 * \brief QkvRmsNormRopeCache arch35(Ascend950)内核入口
 *
 * 由 op_host/qkv_rms_norm_rope_cache_def.cpp 的 ascend950 config 里
 * ExtendCfgInfo("opFile.value", "qkv_rms_norm_rope_cache_apt") 选中;
 * ascend910b / ascend910_93 继续使用默认入口 op_kernel/qkv_rms_norm_rope_cache.cpp。
 *
 * 注意:opFile.value 选的是【源文件】,而 opc 校验的内核函数名仍按算子名推导
 * (origin_func_name = 算子蛇形名),所以这里的 extern "C" 函数必须叫
 * qkv_rms_norm_rope_cache —— 与 A2 入口同名,但两者按 soc 互斥编译,不会撞符号。
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/qkv_rms_norm_rope_cache_regbase.h"

using namespace AscendC;
using namespace QkvRmsNormRopeCache;

#define QKV_RMS_NORM_ROPE_CACHE_REGBASE_PA_NZ 10000

extern "C" __global__ __aicore__ void qkv_rms_norm_rope_cache(
    GM_ADDR qkv, GM_ADDR q_gamma, GM_ADDR k_gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR q_out,
    GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset,
    GM_ADDR q_out_out, GM_ADDR k_cache_out, GM_ADDR v_cache_out, GM_ADDR q_out_proto, GM_ADDR k_cache_proto,
    GM_ADDR v_cache_proto, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (TILING_KEY_IS(QKV_RMS_NORM_ROPE_CACHE_REGBASE_PA_NZ)) {
        GET_TILING_DATA_WITH_STRUCT(QkvRmsNormRopeCacheRegbaseTilingData, tiling_data_in, tiling);
        const QkvRmsNormRopeCacheRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        QkvRmsNormRopeCacheRegbase<DTYPE_QKV, DTYPE_K_CACHE, DTYPE_V_CACHE> op(&pipe, tilingData);
        op.Init(qkv, q_gamma, k_gamma, cos, sin, index, q_out, k_cache, v_cache, k_scale, v_scale, k_offset, v_offset,
                q_out_out, k_cache_out, v_cache_out, q_out_proto, k_cache_proto, v_cache_proto);
        op.Process();
    }
}
