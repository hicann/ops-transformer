/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v4_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MOE_INIT_ROUTING_V4_OPS_H_
#define OPS_OP_PROTO_INC_MOE_INIT_ROUTING_V4_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief compute init routing for moe (v4, ascend950 only).
* @par Inputs:
* @li x: A 2D tensor. Shape is: (B*S, H). Type is:Int8, BFloat16, Float16, Float32, Hif8, FLOAT8_E5M2, FLOAT8_E4M3FN
         or FLOAT4_E2M1. Format support ND.
* @li expert_idx: A 2D tensor. Shape is: (B*S, K). Type is:Int32. Expert index. Format support ND.
* @li scale: A 1D or 2D tensor. Shape is: (B*S) or (expert_end-expert_start, H) when quant_mode is 1.
             When quant_mode is 13, scale can be omitted or shape is (1, H). Type is:Float32 or Float8_E8M0.
             Format support ND.
* @li offset: A 2D tensor. Shape is: (expert_end - expert_start, 1) or (expert_end - expert_start, H).
*             Type is:Float32. Format support ND.
* @li active_num: A scalar tensor. Type is:Int64. Identify activate scenario. When provided, the value
                  should equal B*S*K. Default: not provided (equivalent to B*S*K). Format support ND.
* @li topk_weight: A 2D tensor. Shape is: (B*S, K). Type is:Float32. TopK routing weights. Format support ND.
* @par Outputs:
* @li expanded_x: A 2D tensor. Shape is: (B*S*K, H). When quant_mode is -1, type is:Int8, BFloat16, Float16, Float32,
                  Hif8, FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT4_E2M1.
                  The data type must be the same as that of x.
                  When quant_mode is 0, type is Int8.
                  When quant_mode is 1, type is Int8.
                  When quant_mode in [2, 4, 14], type is FLOAT8_E5M2.
                  When quant_mode in [3, 5, 15], type is FLOAT8_E4M3FN.
                  When quant_mode in [6, 7, 8], type is Hif8.
                  When quant_mode is 9, type is FLOAT4_E2M1.
                  When quant_mode in [11, 12], type is [FLOAT8_E5M2, FLOAT8_E4M3FN].
                  When quant_mode is 13, type is Int4.
                  When quant_mode in [16, 17], type is [FLOAT8_E5M2, FLOAT8_E4M3FN].
                  Format support ND.
* @li expanded_row_idx: A 1D tensor. Shape is: (B*S*K). Type is:Int32. Format support ND.
* @li expert_tokens_count_or_cumsum: A 1D tensor. represents the number of tokens processed by each expert and the
                                       cumulative value. The value is controlled by expert_tokens_num_flag to output.
                                       Type is:Int64. shape is (expert_end - expert_start, ). Format support ND.
* @li expanded_scale: A 1D tensor when quant_mode in [-1, 0, 1, 13]. Shape is: (B*S*K).
                      Type is:Float32. The data type must be the same as that of scale.
                      A 2D tensor when quant_mode in [2, 3, 16, 17]. Shape is: (B*S*K, M),
                      in which M is CeilAlign(CeilDiv(H, 32), 2). Type is: Float8_E8M0.
                      A 2D tensor when quant_mode in [4, 5, 14, 15]. Shape is: (B*S*K, CeilDiv(H, 128)).
                      Type is: Float32.
                      A 3D tensor when quant_mode in [11, 12] (FP8 PerBlock量化，BlockSize=128).
                      Shape is: (B*S*K, M, 2), in which M is CeilDiv(H, 256).
                      Type is: Float32 (FP32 Scale).
                      A 3D tensor when quant_mode is -1 and dtype of input x equals FLOAT4_E2M1 or quant_mode is 9.
                      Shape is: (B*S*K, M, 2), in which M is CeilAlign(CeilDiv(H, 32), 2).
                      Type is: Float8_E8M0.
                      Format support ND.
* @li expanded_topk_weight: A 2D tensor. Shape is: (B*S*K, 1) when topk_weight is provided and drop_pad_mode is 0,
                            or (expert_num*expert_capacity, 1) when drop_pad_mode is 1.
                            Shape is (0,) when topk_weight is not provided. Type is:Float32. Format support ND.
* @par Attributes:
* @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: -1.
* @li expert_num: Optional parameter. Type is:Int32. Number of experts. Default: -1.
* @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
* @li expert_tokens_num_type: Optional parameter. Type is:Int32. The value is 0(compute tokens cumsum) or
                              1(compute tokens count), which in dropPad scenario. Default: 0.
* @li expert_tokens_num_flag: Optional parameter. Type is:Bool. The value is true (compute tokens) or
                              false(do not compute tokens), which in dropPad scenario. Default: false.
* @li quant_mode: Optional parameter. Type is:Int. Valid values: -1(unquant), 0(static), 1(dynamic), 2(fp8_e5m2),
                  3(fp8_e4m3fn), 4(fp8_group_e5m2), 5(fp8_group_e4m3fn), 6(hif8 cast), 7(hif8_pertensor),
                  8(hif8_pertoken), 9(fp4_e2m1), 11(fp8_perblock_e5m2), 12(fp8_perblock_e4m3fn), 13(int4_dynamic),
                  14(fp8_group_amax_e5m2), 15(fp8_group_amax_e4m3fn),
                  16(mxfp8_roundscale_amax_e5m2), 17(mxfp8_roundscale_amax_e4m3fn). Default: -1.
* @li active_expert_range: Optional parameter. Type is:ListInt. Like [expert_start, expert_end].
                           expert_start must be greater than or equal to 0, expert_end must be less than or equal to
                           10240, expert_start must be less than expert_end. Default: [].
* @li row_idx_type: Optional parameter. Type is:Int. The value is 0(gather) or 1(scatter). Default: 0.
*/
REG_OP(MoeInitRoutingV4)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(expert_idx, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(active_num, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(topk_weight, TensorType({DT_FLOAT}))
    .OUTPUT(expanded_x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16, DT_FLOAT8_E5M2,
                                    DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_INT4}))
    .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))
    .OUTPUT(expanded_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(expanded_topk_weight, TensorType({DT_FLOAT}))
    .ATTR(expert_capacity, Int, -1)
    .ATTR(expert_num, Int, -1)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_num_type, Int, 0)
    .ATTR(expert_tokens_num_flag, Bool, false)
    .ATTR(quant_mode, Int, -1)
    .ATTR(active_expert_range, ListInt, {})
    .ATTR(row_idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingV4)
} // namespace ge

#endif
