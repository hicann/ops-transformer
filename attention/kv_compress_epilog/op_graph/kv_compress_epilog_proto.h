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
 * \file kv_compress_epilog_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_KvCompressEpilog_H_
#define OPS_OP_PROTO_INC_KvCompressEpilog_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Compresses cached key/value data in the KV-cache epilog stage by applying
 *        group-wise dynamic quantization (FP8 group / E8M0) with scale factors. \n
 *        The quantized rows are scattered into cache for each valid slot. \n

 * @par Inputs:
 * Inputs including:
 * @li cache: The existing key/value cache tensor to be updated in-place. \n
 *   - Data types: uint8.
 *   - format: ND
 * @li x: Input activation tensor to be quantized and written into the cache.
 *   - Data types: bfloat16.
 *   - format: ND
 * @li slot_mapping: Token index mapping indicating which cache slot each token maps to.
 *   - Data types: int32, int64.
 *   - format: ND

 * @par Attributes:
 * @li quant_group_size: An optional int attribute. Quant group size. Defaults to 64.
 *   For quant_mode 2 only 16/32/64 are supported and (d-64) must be divisible by it.
 * @li quant_mode: An optional int attribute. Quantization mode.
 *   - 0: nope per-group(64) FP8(E4M3) dynamic quant, scale stored as bfloat16, rope kept as bf16
 *   - 1: same as 0 but scale stored as float8_e8m0 (default)
 *   - 2: rope static hifloat8 quant (scaled by x_scale), nope per-group FLOAT4_E2M1 dynamic quant
 *        (scale = group amax / 6, stored as bfloat16). Row layout (bytes):
 *        [rope hifloat8 64B][nope fp4 (d-64)/2 B][nope bf16 scale nGroup*2 B][pad to 128]
 * @li round_scale: An optional bool attribute. Whether to round the group scale value for
 *   quant_mode 0/1. Defaults to true. Ignored when quant_mode==2.
 * @li x_scale: An optional float attribute. For quant_mode 2 it is the rope hifloat8 static
 *   quant scale; reserved/unused for quant_mode 0/1. Defaults to 1.0.

 * @par Outputs:
 * @li cache: Updated key/value cache after in-place compression.

 * @attention Constraints:
 * @code{.c}
 *  - cache is both input and output (in-place update).
 *  - slot_mapping dimensions should equal x dimensions minus 1.
 *  - The last dimension (d) of x must be 64-aligned (d % 64 == 0) and greater than 64.
 *  - For quant_mode==2, quant_group_size must be 16/32/64 and (d-64) divisible by it.
 *  - Ascend 950PR&950DT products.
 * @endcode
 */
REG_OP(KvCompressEpilog)
    .INPUT(cache, TensorType({DT_UINT8}))
    .INPUT(x, TensorType({DT_BF16}))
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .OUTPUT(cache, TensorType({DT_UINT8}))
    .ATTR(quant_group_size, Int, 64)
    .ATTR(quant_mode, Int, 1)
    .ATTR(round_scale, Bool, true)
    .ATTR(x_scale, Float, 1.0)
    .OP_END_FACTORY_REG(KvCompressEpilog)
} // namespace ge

#endif // OPS_OP_PROTO_INC_KvCompressEpilog_H_
