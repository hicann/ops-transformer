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
 * \file indexer_quant_cache_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IndexerQuantCache_H_
#define OPS_OP_PROTO_INC_IndexerQuantCache_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Compresses cached key/value data in the Indexer attention epilog stage
 *        by applying block-wise dynamic quantization (FP8/INT8/MX-FP4) with scale factors. \n
 *        Supports per-block quantization on the last dimension with optional MX-FP8/MX-FP4 scale format. \n

 * @par Inputs:
 * Inputs including:
 * @li cache: The existing key/value cache tensor to be updated in-place. \n
 *   - Data types: float8_e4m3fn, float8_e5m2, uint8, float4_e2m1, float4_e1m2.
 *   - format: ND
 * @li cache_scale: Scale factors for per-block quantization.
 *   - Data types: float32, float8_e8m0.
 *   - format: ND
 * @li x: Input activation tensor to be quantized and written into the cache.
 *   - Data types: float16, bfloat16.
 *   - format: ND
 * @li slot_mapping: Token index mapping indicating which cache slot each token maps to.
 *   - Data types: int32.
 *   - format: ND

 * @par Attributes:
 * @li quant_mode: An optional int attribute. Quantization mode.
 *   - 0: MX-FP8 quantization (scale stored as float8_e8m0)
 *   - 1: Normal block-wise quantization (default, scale stored as float32)
 *   - 2: HiFloat8 quantization (scale attribute applied)
 *   - 3: MX-FP4 quantization (cache packed as uint8, scale stored as float8_e8m0)
 * @li round_scale: An optional bool attribute. Whether to round the MX-FP8 scale value. Defaults to true.
 * @li x_scale: An optional float attribute. Global scale multiplier for HiFloat8 mode. Defaults to 1.0.

 * @par Outputs:
 * @li cache: Updated key/value cache after in-place compression.
 * @li cache_scale: Updated scale factors after in-place compression.

 * @attention Constraints:
 * @code{.c}
 *  - cache and cache_scale are both input and output (in-place update).
 *  - slot_mapping dimensions should equal x dimensions minus 1.
 *  - The last dimension (d) of x is quantized per 128 elements per block;
 *    MX-FP4 (quant_mode=3) quantizes per 32 elements (standard MX block).
 *  - Ascend 950PR&950DT products.
 * @endcode
 */
REG_OP(IndexerQuantCache)
    .INPUT(cache, "T1")
    .INPUT(cache_scale, "T2")
    .INPUT(x, "T0")
    .INPUT(slot_mapping, TensorType::IndexNumberType())
    .OUTPUT(cache, "T1")
    .OUTPUT(cache_scale, "T2")
    .ATTR(quant_mode, Int, 1)
    .ATTR(round_scale, Bool, true)
    .ATTR(x_scale, Float, 1.0)
    .DATATYPE(T0, TensorType({DT_FLOAT16, DT_BF16}))
    .DATATYPE(T1, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_UINT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(IndexerQuantCache)
} // namespace ge

#endif // OPS_OP_PROTO_INC_IndexerQuantCache_H_
