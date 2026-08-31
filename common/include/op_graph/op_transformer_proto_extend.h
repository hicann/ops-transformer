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
 * \file op_transformer_proto_extend.h
 * \brief
 */
#ifndef OPS_OP_MATH_PROTO_EXTEND_H_
#define OPS_OP_MATH_PROTO_EXTEND_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief swin_transformer model specific structure.Operator only supports swin_transformer.

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be (B*W, N, S1, S2) or (B, W, N, S1, S2).
* @li atten_mask: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
                  the shape should be (W, S1, S2) or (W, 1, S1, S2) or (1, W, 1, S1, S2)
* @li relative_pos_bias: An ND Tensor. Must be one of the following types: float16, float, bfloat16.
                         the shape sholud be (N, S1, S2) or (1, N, S1, S2) or (1, 1, N, S1, S2)

* @par Attributes:
* @li scale_value: A optional attribute, the type is float. Defaults to 1.0.
* @li inner_precision_mode: A optional attribute, the type is int. Defaults to 0, reserved field.

* @par Outputs:
* One output, including:
* @li y: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be same with x.
*/
REG_OP(MaskedSoftmaxWithRelPosBias)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(relative_pos_bias, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(inner_precision_mode, Int, 0)
    .OP_END_FACTORY_REG(MaskedSoftmaxWithRelPosBias)

/**
* @brief AttentionScore's forward calculation.

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li key: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li value: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li padding_mask: A matrix Tensor. The type only support float16. Enter a 4D Tensor.
* @li scale: A scalar. The type only support float16. Enter a 4D Tensor.
* @li drop_mask: A matrix Tensor. An optional input parameter. The type only support uint8. Enter a 4D Tensor.

* @par Attributes:
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li query_transpose: A bool. If True, changes the shape of "query" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li key_transpose: A bool. If True, changes the shape of "key" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li bmm_score_transpose_a: A bool. If True, changes the shape of "mid_data" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li bmm_score_transpose_b: A bool. If True, changes the shape of "value" from [B, N, S, D] to [B, N, D, S].
* Default: false.
* @li softmax_axes: A list of int. The dimension softmax would be performed on. Defaults to "[-1]".

* @par Outputs:
* attention_score: The result matrix Tensor. The type only support float16. The output shape is the same as query.
* softmax_output: The result matrix Tensor. The type only support float16. The output shape is the same as query.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
#ifndef OPS_PROTO_DEF_ATTENTIONSCORE
#define OPS_PROTO_DEF_ATTENTIONSCORE
        REG_OP(AttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16}))
    .INPUT(key, TensorType({DT_FLOAT16}))
    .INPUT(value, TensorType({DT_FLOAT16}))
    .INPUT(padding_mask, TensorType({DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_INT8}))
    .OUTPUT(attention_score, TensorType({DT_FLOAT16}))
    .OUTPUT(softmax_output, TensorType({DT_FLOAT16}))
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(query_transpose, Bool, false)
    .ATTR(key_transpose, Bool, false)
    .ATTR(bmm_score_transpose_a, Bool, false)
    .ATTR(bmm_score_transpose_b, Bool, false)
    .ATTR(softmax_axes, ListInt, {-1})
    .OP_END_FACTORY_REG(AttentionScore)
#endif

    /**
     * @brief Tutel dispatch function in moe.
     *
     * @par Inputs:
     * @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Attributes:
     * capacity: expert capacity.
     *
     * @par Outputs:
     * y: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */

    REG_OP(MoeTutelDispatch)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(capacity, Int)
    .OP_END_FACTORY_REG(MoeTutelDispatch)

    /**
     * @brief Tutel combine function in moe.
     *
     * @par Inputs:
     * @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li gates: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Outputs:
     * @li x_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */
    REG_OP(MoeTutelCombineX)
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineX)

    /**
     * @brief Tutel combine function in moe.
     *
     * @par Inputs:
     * @li x: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li y_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.
     * @li indices: A mutable Tensor of the type DT_INT32, for topk's k size.
     * @li locations: A mutable Tensor of the type DT_INT32, for token size.
     *
     * @par Outputs:
     * gates_grad: A mutable Tensor of the type DT_FLOAT, DT_FLOAT16, DT_BF16.\n
     */
    REG_OP(MoeTutelCombineGates)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(locations, TensorType({DT_INT32}))
    .OUTPUT(gates_grad, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(MoeTutelCombineGates)

    /**
    * @brief Fusion op for FFN. This op supports to compute MoeFFN(Mixture-of-Experts) or FFN.
    * @par Inputs:
    * fourteen inputs, including:
    * @li x: A matrix Tensor. The type support int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. Shape supports at least 2 dimensions (M,K1), and at most 8 dimensions.
    * @li weight1: A matrix Tensor for weight of the first matmul. The type support int4, int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K1, N1)/(K1, N1).
    * @li weight2: A matrix Tensor for weight of the second matmul. The type support int4, int8, float16, bfloat16.
    * Format support ND, FRACTAL_NZ. When having experts/having no expert, shape should be (E, K2, N2)/(K2, N2).
    * @li expert_tokens: A matrix Tensor. Indicating num of tokens in each of experts. If having experts, expert_tokens
    should be passed; if having no experts, expert_tokens should not be passed.
    * The type support int64. Format support ND. If not null, shape should be (E) and should satisfy E <= 256.
    * @li bias1: A matrix Tensor for bias of the first matmul. The type support int32, float16, float32. Format support
    ND.
    * When having experts/having no expert, shape should be (E, N1)/(N1).
    * @li bias2: A matrix Tensor for bias of the secend matmul. The type support int32, float16, float32. Format support
    ND.
    * When having experts/having no expert, shape should be (E, N2)/(N2).
    * @li scale: A matrix Tensor. Indicating scaling factor of quantization parameter.
    * The type support float32. Format support ND. In per-tensor quantization cases, when having experts/having no
    expert, shape should be (E)/(1).
    * In per-channel quantization cases, when having experts/having no expert, shape should be (E, N1)/(N1).
    * @li offset: A matrix Tensor. Indicating the offset of the quantization parameter.
    * The type support float32. Format support ND. When having experts/having no expert, shape should be (E)/(1).
    * @li deq_scale1: A matrix Tensor. Indicating scaling factor of dequantization parameter for the first matmul.
    * The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape
    should be (E, N1)/(N1).
    * @li deq_scale2: A matrix Tensor. Indicating scaling factor of dequantization parameter for the second matmul.
    * The type support uint64, int64, float32, bfloat16. Format support ND. When having experts/having no expert, shape
    should be (E, N2)/(N2).
    * @li antiquant_scale1: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the
    first matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N1)/(N1).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1,
    N1).
    * @li antiquant_scale2: A matrix Tensor. Indicating the scaling factor of the fake-quantization parameter for the
    second matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N1)/(N1).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2,
    N2).
    * @li antiquant_offset1: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the first
    matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N2)/(N2).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G1, N1)/(G1,
    N1).
    * @li antiquant_offset2: A matrix Tensor. Indicating the offset of the fake-quantization parameter for the second
    matmul.
    * The type support float16, bfloat16. Format support ND. In per-channel fake-quantization cases, when having
    experts/having no expert, shape should be (E, N2)/(N2).
    * In per-in-group fake-quantization cases, when having experts/having no expert, shape should be (E, G2, N2)/(G2,
    N2).

    * @par Attributes:
    * @li activation: A string. The type of activation. Support fastgelu, gelu, relu, silu, geglu, swiglu and reglu.
    * @li inner_precise: An int. 0, fp16 high precision. 1, high performance. Default value: 0
    * @li output_dtype: An int. -1, output data type is float16. 0, output data type is float16. 1, output data type is
    bfloat16. Default -1.
    * @li tokens_index_flag: A bool. false, values in expert_tokens are values. true, values in expert_tokens are
    indices. Default value: false
    *
    * @par Outputs:
    * y: A matrix Tensor. The type support float16, bfloat16.
    * Format support ND, FRACTAL_NZ. Num of dimension should be same as x.
    *\n
    *\n
    * The following are the supported data formats and data types (for Atlas A2 Training Series Product/Atlas 800I A2
    Inference Product/A200I A2 Box Heterogeneous Component):
    *\n
    | Tensor    | x       | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 |
    antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y       | | :-------: | :-----: |
    :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: |
    :---------------------------------: | :-----: | | Format1   | ND      | ND              | ND          | ND | ND | ND
    | ND                                  | ND      | | Data Type | float16 | float16         | float16     | - | - | -
    | -                                   | float16 | |           | bfloat16| bfloat16        | float32     | - | - | -
    | -                                   | bfloat16| |           | int8    | int8            | int32       | float32 |
    uint64                | -                                  | -                                   | float16 | | |
    int8    | int8            | int32       | float32      | bfloat16              | - | - | bfloat16| |           |
    int8    | int8            | int32       | float32      | int64                 | - | - | float16 | |           |
    int8    | int8            | int32       | float32      | float32               | - | - | float16 | |           |
    float16 | int8            | float16     | -            | -                     | float16 | float16 | float16 | | |
    bfloat16| int8            | float32     | -            | -                     | bfloat16 | bfloat16 | bfloat16| |
    | float16 | int4            | float16     | -            | -                     | float16 | float16 | float16 | |
    | bfloat16| int4            | float32     | -            | -                     | bfloat16 | bfloat16 | bfloat16|
    *\n
    * The following are the supported data formats and data types (for Atlas Inference Series Product):
    *\n
    | Tensor    | x          | weight1/weight2 | bias1/bias2 | scale/offset | deq_scale1/deq_scale2 |
    antiquant_scale1/antiquant_scale2  | antiquant_offset1/antiquant_offset2 | y          | | :-------: | :--------: |
    :-------------: | :---------: | :----------: | :-------------------: | :--------------------------------: |
    :---------------------------------: | :--------: | | Format1   | FRACTAL_NZ | FRACTAL_NZ      | ND          | ND |
    ND                    | ND                                 | ND                                  | FRACTAL_NZ | |
    Data Type | float16    | float16         | float16     | -            | -                     | - | - | float16    |
    *\n
    * @attention Constraints:
    * @li Atlas Inference Series Product only support non-quantization high performance no-expert cases; x and y must
    have two dimensions; activation only supports gelu/fastgelu/relu/silu.
    * @li If expert_tokens is passed, when tokens_index_flag is true, it must be a non-negative monotone non-decreasing
    array; when tokens_index_flag is false, it must be a non-negative array.
    * @li If expert_tokens is passed, when tokens_index_flag is false, sum of expert_tokens should be equal to the first
    dim M of x; when tokens_index_flag is true, the last value in expert_tokens should be equal to the first dim M of x.
    * @li If activation is geglu/swiglu/reglu, only supporting float16 (data type of required inputs are all float16)
    high performance no-expert cases, and should satisfy N1=2*K2.
    * @li If activation is gelu/fastgelu/relu/silu, should satisfy N1=K2.
    * @li All cases should satisfy K1=N2, K1<65536, K2<65536, and M should be less than 2147483547 after aligning to 32
    byte.
    * @li Non-quantization cases should not pass quantization or fake-quantization related optional inputs, quantization
    cases should not pass fake-quantization related optional inputs, fake-quantization cases should not pass
    quantization related optional inputs.
    * @li Per-tensor quantization cases support data type template with deq_scale1/deq_scale2 float32, while per-channel
    quantization cases do not support this data type template.
    * @li If data type of weight1 and weight2 is int4, the last dimension of weight1 and weight2 must be even.
    * @li In per-in-group fake-quantization cases, group num G1 of antiquant_scale1 and antiquant_offset1 must be
    divisible by K1, group num G2 of antiquant_scale2 and antiquant_offset2 must be divisible by K2.
    * @li Attr inner_precise is only valid in non-quantization cases. In non-quantization cases, if data type of
    required inputs are all bfloat16, inner_precise only supports 0; if data type of required inputs are all float16,
    inner_precise can pass 0 or 1.
    * @li Attr output_dtype is only valid in quantization cases.
    */
    REG_OP(FFN)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .INPUT(weight1, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .INPUT(weight2, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_INT4}))
    .OPTIONAL_INPUT(expert_tokens, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale1, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(deq_scale2, TensorType({DT_UINT64, DT_BF16, DT_INT64, DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(activation, String)
    .ATTR(inner_precise, Int, 0)
    .ATTR(output_dtype, Int, -1)
    .ATTR(tokens_index_flag, Bool, false)
    .OP_END_FACTORY_REG(FFN)

} // namespace ge
#endif
