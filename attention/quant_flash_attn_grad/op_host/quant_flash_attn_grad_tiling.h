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
 * \file quant_flash_attn_grad_tiling.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_GRAD_TILING_H_
#define QUANT_FLASH_ATTN_GRAD_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QuantFlashAttnGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, g)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, n1)
TILING_DATA_FIELD_DEF(int64_t, n2)
TILING_DATA_FIELD_DEF(int64_t, t1)
TILING_DATA_FIELD_DEF(int64_t, t2)
TILING_DATA_FIELD_DEF(int64_t, s1_outer)
TILING_DATA_FIELD_DEF(int64_t, s2_outer)
TILING_DATA_FIELD_DEF(int64_t, s1_tail)
TILING_DATA_FIELD_DEF(int64_t, s2_tail)
TILING_DATA_FIELD_DEF(float, softmax_scale)
TILING_DATA_FIELD_DEF(bool, has_seq_used_q)
TILING_DATA_FIELD_DEF(bool, has_seq_used_k)

TILING_DATA_FIELD_DEF(int64_t, dq_work_space_offset)
TILING_DATA_FIELD_DEF(int64_t, dk_work_space_offset)
TILING_DATA_FIELD_DEF(int64_t, dv_work_space_offset)
TILING_DATA_FIELD_DEF(int64_t, sfmg_work_space_offset)
TILING_DATA_FIELD_DEF(int64_t, q_pre_block_factor)
TILING_DATA_FIELD_DEF(int64_t, q_pre_block_total)
TILING_DATA_FIELD_DEF(int64_t, q_pre_block_tail)
TILING_DATA_FIELD_DEF(int64_t, k_pre_block_factor)
TILING_DATA_FIELD_DEF(int64_t, k_pre_block_total)
TILING_DATA_FIELD_DEF(int64_t, k_pre_block_tail)
TILING_DATA_FIELD_DEF(int64_t, v_pre_block_factor)
TILING_DATA_FIELD_DEF(int64_t, v_pre_block_total)
TILING_DATA_FIELD_DEF(int64_t, v_pre_block_tail)
TILING_DATA_FIELD_DEF(int64_t, metadata_len)

TILING_DATA_FIELD_DEF(int64_t, sfmg_used_core_num)
TILING_DATA_FIELD_DEF(int64_t, sfmg_dy_buffer_len)
TILING_DATA_FIELD_DEF(int64_t, sfmg_y_buffer_len)
TILING_DATA_FIELD_DEF(int64_t, sfmg_output_buffer_len)
TILING_DATA_FIELD_DEF(int64_t, single_loop_nburst_num)
TILING_DATA_FIELD_DEF(int64_t, normal_core_loop_times)
TILING_DATA_FIELD_DEF(int64_t, tail_core_loop_times)
TILING_DATA_FIELD_DEF(int64_t, normal_core_last_loop_nburst_num)
TILING_DATA_FIELD_DEF(int64_t, tail_core_last_loop_nburst_num)
TILING_DATA_FIELD_DEF(int64_t, normal_core_nburst_nums)
TILING_DATA_FIELD_DEF(int64_t, tail_core_nburst_nums)
TILING_DATA_FIELD_DEF(int64_t, normal_axis_size)

TILING_DATA_FIELD_DEF(int64_t, q_post_block_factor)
TILING_DATA_FIELD_DEF(int64_t, q_post_block_total)
TILING_DATA_FIELD_DEF(int64_t, q_post_base_num)
TILING_DATA_FIELD_DEF(int64_t, q_post_tail_num)
TILING_DATA_FIELD_DEF(int64_t, k_post_block_factor)
TILING_DATA_FIELD_DEF(int64_t, k_post_block_total)
TILING_DATA_FIELD_DEF(int64_t, k_post_base_num)
TILING_DATA_FIELD_DEF(int64_t, k_post_tail_num)
TILING_DATA_FIELD_DEF(int64_t, v_post_block_factor)
TILING_DATA_FIELD_DEF(int64_t, v_post_block_total)
TILING_DATA_FIELD_DEF(int64_t, v_post_base_num)
TILING_DATA_FIELD_DEF(int64_t, v_post_tail_num)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantFlashAttnGrad, QuantFlashAttnGradTilingData)

struct QuantFlashAttnGradCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};
} // namespace optiling
#endif // QUANT_FLASH_ATTN_GRAD_TILING_H_
