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
 * \file grouped_matmul_proto.h
 * \brief Graph IR definition for GroupedMatmul.
 */
#ifndef OPS_TRANSFORMER_GROUPED_MATMUL_PROTO_H_
#define OPS_TRANSFORMER_GROUPED_MATMUL_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
// clang-format off
/**
* @brief Function GroupedMatmul. This op computes groups of matmuls.

* @par Inputs:
* @li x: A Tensor List. Format supports ND. The type support float16, bfloat16, int8, float32, int4, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2(only weight NZ supported).
* Maximum length of x is 128.
* @li weight: A Tensor List of weight. Format supports ND/NZ. The type support float16, bfloat16, int8, float32, int4, int32, hifloat8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2(only weight NZ supported).
* Maximum length of weight is 128. When type is float32/int32 in fake-quantization modes which only supported in Ascend950PR/Ascend950DT, the input is float4_e2m1/int4-packed data.
* @li bias: A Tensor List of bias. Format supports ND. The type support float16, float32, int32, bfloat16.
* Length of bias must be the same as weight.
* @li scale: A Tensor List of scale. Indicating scaling factor of quantization parameter.
* Format supports ND. The type support uint64, int64, bfloat16, float32, float8_e8m0. Length of scale must be the same as weight.
* @li offset: A Tensor List of offset. Indicating the offset of the quantization parameter.
* Format supports ND. The type support float32. Length of offset must be the same as weight.
* @li antiquant_scale: A Tensor List of antiquant_scale. Indicating the scaling factor of the fake-quantization parameter.
* Format supports ND. The type support float16, bfloat16, float8_e8m0. Length of antiquant_scale must be the same as weight.
* @li antiquant_offset: A Tensor List of antiquant_offset. Indicating the offset of the fake-quantization parameter.
* Format supports ND. The type support float16, bfloat16. Length of antiquant_offset must be the same as weight.
* @li group_list: a Tensor. Indicating the matmul size distribution along separated dimension.
* Format supports ND. The type support int64.
* @li per_token_scale: A Tensor of per_token_scale.
* Indicating the scaling factor of the quantization parameter, introduced by x quantization.
* Format supports ND. The type support float32, float8_e8m0.

* @par Attributes:
* @li split_item: An int. Indicate whether required separated y. Default: 0.
* @li dtype: An int. only invalid for quant case. -1, output data type is int8.
* 0, output data type is float16. 1, output data type is bfloat16. 2, output data type is int32. 3, output data type is float32. Default: 0.
* @li transpose_weight: A bool. Reserved parameter,
* indicate wether input weight is transposed, not enabled. Default: false.
* @li transpose_x: A bool. Reserved parameter,
* indicate wether input x is transposed, not enabled. Default: false.
* @li group_type: An int. Indicates the grouped dimension in group_list. -1, group_list is null.
* 0, grouped dimension is M. 1, grouped dimension is N, not supported currently. 2, grouped dimension is K. Default: -1.
* @li group_list_type: An int. Indicates whether the value in group_list is cumsum, count or sparse.
* 0, value in group_list is cumsum. 1, value in group_list is count. 2, value in group_list is sparse (group_idx, size) pairs. Default: 0.
* @li act_type: An int. Indicate activation function type. Value range 0-5. Default: 0.
* 0, no activation. 1, relu. 2, gelu_tanh. 3, gelu_err_func, not supported currently. 4, fastgelu. 5, silu.
* @li tuning_config: A ListInt. The first element in the array indicates the expected number of tokens processed by each expert.
* The operator tiling performs optimal tiling based on the first element in the array. Default: {0}.

* @par Outputs:
* y: A Tensor List. Format supports ND.
* The type support float16, bfloat16, int8, float32, int32.
*\n
*\n
* - The following are the supported data formats and corresponding data types (for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component):
*\n
*\n
| Tensor    | x       | weight    | bias    | scale   | offset  | antiquant_scale | antiquant_offset | per_token_scale | y       |
| :-------: | :-----: | :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Format1   | ND      | ND        | ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | float16 | float16   | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | float32 | float32   | float32 | uint64  | float32 | float16 | float16 |    -    | float32 |
|           | bfloat16| bfloat16  | float32 | uint64  | float32 | float16 | float16 |    -    | bfloat16|
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int8    |
|           | int8    | int8      | int32   | bfloat16| float32 | float16 | float16 | float32 | bfloat16|
|           | int8    | int8      | int32   | float32 | float32 | float16 | float16 | float32 | float16 |
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int32   |
|           | float16 | int8      | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | bfloat16| int8      | float32 | uint64  | float32 | bfloat16| bfloat16|    -    | bfloat16|
|           | float16 | int4      | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
|           | bfloat16| int4      | float32 | uint64  | float32 | bfloat16| bfloat16|    -    | bfloat16|
| Format2   | ND      | FRACTAL_NZ| ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | int8    | int8      | int32   | bfloat16| float32 | float16 | float16 | float32 | bfloat16|
|           | int8    | int8      | int32   | float32 | float32 | float16 | float16 | float32 | float16 |
|           | int8    | int8      | int32   | uint64  | float32 | float16 | float16 |    -    | int32   |
*\n
*\n
* - The following are the supported data formats and data types (for Atlas Inference Series Product):
*\n
*\n
| Tensor    | x       | weight    | bias    | scale   | offset  | antiquant_scale | antiquant_offset | per_token_scale | y       |
| :-------: | :-----: | :-------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Format2   | ND      | FRACTAL_NZ| ND      | ND      | ND      | ND      | ND      | ND      | ND      |
| Data Type | float16 | float16   | float16 | uint64  | float32 | float16 | float16 |    -    | float16 |
*\n
*\n
* - The following are the supported data formats and data types (for Ascend950PR/Ascend950DT):
*\n
| Tensor    | x                         | weight                                  | bias                   | scale            | offset  | antiquant_scale | antiquant_offset | per_token_scale | y   |
| :-------: | :-----------------------: | :-------------------------------------: | :--------------------: | :--------------: | :-----: | :---------: | :-----: | :---------: | :----------------------: |
| Format1   | ND                        | ND                                      | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/int8    |
|           | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | int32                    |
|           | int8                      | int8                                    | int32/float16/float32  | float32          | float32 | float16     | float16 | -/float32   | float16                  |
|           | int8                      | int8                                    | int32/bfloat16/float32 | bfloat16/float32 | float32 | float16     | float16 | -/float32   | bfloat16                 |
|           | hifloat8                  | hifloat8                                | float32                | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/float32 |
|           | hifloat8                  | hifloat8                                | float32                | float32          | float32 | float16     | float16 | -/float32   | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | float32          | float32 | float16     | float16 | -/float32   | float16/bfloat16/float32 |
|           | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2               | float32                | float8_e8m0      | float32 | float16     | float16 | float8_e8m0 | float16/bfloat16/float32 |
|           | float4_e2m1               | float4_e2m1                             | float32                | float8_e8m0      | float32 | float16     | float16 | float8_e8m0 | float16/bfloat16/float32 |
|           | float16                   | float16                                 | float16/float32        |       -          |    -    |      -      |    -    |      -      | float16                  |
|           | bfloat16                  | bfloat16                                | bfloat16/float32       |       -          |    -    |      -      |    -    |      -      | bfloat16                 |
|           | float16                   | int8/float8_e4m3fn/float8_e5m2/hifloat8 | float16                | uint64           | float32 | float16     | float16 |      -      | float16                  |
|           | bfloat16                  | int8/float8_e4m3fn/float8_e5m2/hifloat8 | bfloat16/float32       | uint64           | float32 | bfloat16    | bfloat16|      -      | bfloat16                 |
|           | float16                   | int4                                    | float16                | uint64           | float32 | float16     | float16 |      -      | float16                  |
|           | bfloat16                  | int4                                    | bfloat16/float32       | uint64           | float32 | bfloat16    | bfloat16|      -      | bfloat16                 |
|           | int4                      | int4                                    | -                      | uint64           | -       | -           | -       | float32     | float16/bfloat16         |
| Format2   | ND                        | FRACTAL_NZ                              | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | int8                      | int8                                    | int32                  | uint64/int64     | float32 | float16     | float16 |      -      | float16/bfloat16         |
|           | int8                      | int8                                    | int32/float16/float32  | float32          | float32 | float16     | float16 | -/float32   | float16                  |
|           | int8                      | int8                                    | int32/bfloat16/float32 | bfloat16/float32 | float32 | float16     | float16 | -/float32   | bfloat16                 |
|           | int8                      | int4                                    | float32                | float32          | float32 | float16     | float16 | float32     | float16/bfloat16         |
|           | int4                      | int4                                    | -                      | uint64           | -       | -           | -       | float32     | float16/bfloat16         |
|           | float16                   | float4_e2m1/float32                     | float16                | uint64           | float32 | float8_e8m0 | float16 |      -      | float16                  |
|           | bfloat16                  | float4_e2m1/float32                     | bfloat16/float32       | uint64           | float32 | float8_e8m0 | bfloat16|      -      | bfloat16                 |
|           | float8_e4m3fn             | float4_e2m1/float4_e1m2/float32         | float16                | uint64           | float32 | float8_e8m0 | float16 | float8_e8m0 | float16                  |
|           | float8_e4m3fn             | float4_e2m1/float4_e1m2/float32         | bfloat16               | uint64           | float32 | float8_e8m0 | bfloat16| float8_e8m0 | bfloat16                 |
|           | float8_e4m3fn             | float8_e4m3fn                           | -                      | float8_e8m0      | -       | -           | -       | float8_e8m0 | float16/bfloat16/float32 |
|           | float4_e2m1/float4_e1m2   | float4_e2m1/float4_e1m2                 | float32                | float8_e8m0      | float32 | float8_e8m0 | float16 | float8_e8m0 | float16/bfloat16/float32 |
| Format3   | ND                        | FRACTAL_NZ_C0_16                        | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | float16                   | float4_e2m1                             | float16                | uint64           | float32 | float8_e8m0 | float16 |      -      | float16                  |
|           | bfloat16                  | float4_e2m1                             | bfloat16/float32       | uint64           | float32 | float8_e8m0 | float16 |      -      | bfloat16                 |
| Format4   | ND                        | FRACTAL_NZ_C0_32                        | ND                     | ND               | ND      | ND          | ND      | ND          | ND                       |
| Data Type | float8_e4m3fn             | float4_e2m1/float4_e1m2                 | float16                | uint64           | float32 | float8_e8m0 | float16 | float8_e8m0 | float16                  |
|           | float8_e4m3fn             | float4_e2m1/float4_e1m2                 | bfloat16               | uint64           | float32 | float8_e8m0 | bfloat16| float8_e8m0 | bfloat16                 |
|           | int8                      | int32                                   | float32                | float32          | float32 | float16     | float16 | float32     | bfloat16/float16                |
*\n
* - The following are the supported data types and no quantization modes(for Ascend950PR/Ascend950DT):
| x                         | weight                    | bias                   |   y                      | group_type |
| :-----------------------: | :-----------------------: | :--------------------: | :----------------------: | :--------: |
| float16                   | float16                   | float16/float32        | float16                  |-1/0/2      |
| bfloat16                  | bfloat16                  | bfloat16/float32       | bfloat16                 |-1/0/2      |
| float32                   | float32                   | float32/float32        | float32                  |-1/0/2      |
*\n
* - The following are the supported data types and quantization modes(for Ascend950PR/Ascend950DT):
*\n
pertensor-perchannel && pertensor-pertensor:
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | int8                      | int8                      | -                    | uint64/int64     | 0          |
  | int8                      | int8                      | -                    | float32/bfloat16 | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | -                    | uint64/int64     | 0          |
  | hifloat8                  | hifloat8                  | -                    | uint64/int64     | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | -/float32            | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | -/float32            | float32          | 0/2        |
*\n
pertoken-perchannel && pertoken-pertensor
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | int8                      | int8                      | float32              | float32/bfloat16 | 0          |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | float32              | float32          | 0/2        |
*\n
pergroup-perblock(group_size_m = 1, group_size_k = 128, group_size_n = 128):
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float32              | float32          | 0/2        |
  | hifloat8                  | hifloat8                  | float32              | float32          | 0/2        |
*\n
mx quant(group_size_k = 32)：
*\n
  | x type                    | weight type               | per_token_scale type | scale type       | group_type |
  | ------------------------- | ------------------------- | -------------------- | ---------------- | ---------- |
  | float8_e4m3fn/float8_e5m2 | float8_e4m3fn/float8_e5m2 | float8_e8m0          | float8_e8m0      | 0/2        |
  | float4_e2m1               | float4_e2m1               | float8_e8m0          | float8_e8m0      | 0          |
  | float4_e2m1/float4_e1m2   | float4_e2m1/float4_e1m2(weight NZ)| float8_e8m0  | float8_e8m0      | 0          |
*\n
perchannel：
*\n
  | x type                    | weight type                             | antiquant_scale type | antiquant_offset type | group_type |
  | ------------------------- | --------------------------------------- | -------------------- | --------------------- | ---------- |
  | float16                   | int8/float8_e4m3fn/float8_e5m2/hifloat8 | float16              | float16               | 0          |
  | bfloat16                  | int8/float8_e4m3fn/float8_e5m2/hifloat8 | bfloat16             | bfloat16              | 0          |
  | float16                   | int4                                    | float16              | float16               | 0          |
  | bfloat16                  | int4                                    | bfloat16             | bfloat16              | 0          |
*\n
pertoken-(perchannel & pergroup)：
*\n
  | x type                    | weight type               | per_token_scale type | scale type            | antiquant_scale type | group_type |
  | ------------------------- | ------------------------- | -------------------- | --------------------- | -------------------- | ---------- |
  | int8                      | int4/int32                | float32              | float32               | float16              | 0          |
  | int4                      | int4                      | float32              | uint64                | -                    | 0          |
*\n
mx fake-quant(group_size_k = 32)：
*\n
  | x type                    | weight type                     | antiquant_scale type | antiquant_offset type | per_token_scale type | group_type |
  | ------------------------- | ------------------------------- | -------------------- | --------------------- | -------------------- | ---------- |
  | float16                   | float4_e2m1/float32             | float8_e8m0          | float16               |           -          | 0          |
  | bfloat16                  | float4_e2m1/float32             | float8_e8m0          | bfloat16              |           -          | 0          |
  | float8_e4m3fn             | float4_e2m1/float4_e1m2/float32 | float8_e8m0          | float16/bfloat16      | float8_e8m0          | 0          |
*\n
* When x is float8_e4m3fn and weight is float4_e2m1, float4_e1m2, or float32, group_size_k must be 32,
* x must not be transposed, and weight must be transposed. When weight is float4_e2m1 or float4_e1m2
* with shape (N,K), the SMS scenario is supported; the tensor list lengths of antiquant_scale,
* antiquant_offset, and bias must be the same as that of weight.
*\n
* - The following are the supported quantization modes, data types and shapes(for Ascend950PR/Ascend950DT):
*\n
| quantization         | group_type | x type                             | scale type                      | x shape   | weight shape        | y shape   | scale shape                               | per_token_scale shape | bias shape |
|----------------------|------------|------------------------------------|-------------------------------  |-----------|---------------------|-----------|-------------------------------------------|-----------------------|------------|
| pertensor-pertensor  | 0          | int8                               | -/uint64/int64/float32/bfloat16 | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | -/[(B,)]/[(B,1)]                          | -                     | -/[(B,N)]  |
| pertensor-pertensor  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | uint64/int64                    | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | -                     | -          |
| pertensor-pertensor  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | -/(B,)/(B,1)          | -          |
| pertensor-pertensor  | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,)]/[(B,1)]                            | -/(B,)/(B,1)          | -          |
| pertensor-perchannel | 0          | int8                               | -/uint64/int64/float32/bfloat16 | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | -/[(B,N)]                                 | -                     | -/[(B,N)]  |
| pertensor-perchannel | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | uint64/int64                    | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | -                     | -          |
| pertensor-perchannel | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | -/(B,)/(B,1)          | -          |
| pertensor-perchannel | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,N)]                                   | -/(B,)/(B,1)          | -          |
| pertoken-pertensor   | 0          | int8                               | float32/bfloat16                | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | (M,)                  | -/[(B,N)]  |
| pertoken-pertensor   | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,)]/[(B,1)]                            | (M,)                  | -          |
| pertoken-pertensor   | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,)]/[(B,1)]                            | (B,M)                 | -          |
| pertoken-perchannel  | 0          | int8                               | float32/bfloat16                | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | (M,)                  | -/[(B,N)]  |
| pertoken-perchannel  | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N)]                                   | (M,)                  | -          |
| pertoken-perchannel  | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(B,N)]                                   | (B,M)                 | -          |
| pertoken-perchannel  | 0          | int4                               | uint64                          | [(M,K)]   | [(B,K,N)]           | [(M,N)]   | [(B,N,)]                                  | (M,)                  | -          |
| pertoken-pergroup    | 0          | int4                               | uint64                          | [(M,K)]   | [(B,K,N)]           | [(M,N)]   | [(B,ceil(K/group_size),N,)]               | (M,)                  | -          |
| pergroup-perblock    | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,N,K)]           | [(M,N)]   | [(B,ceil(N/128),ceil(K/128))]             | (M,ceil(K/128))       | -          |
| pergroup-perblock    | 0          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(M,K)]   | [(B,K,N)]           | [(M,N)]   | [(B,ceil(K/128),ceil(N/128))]             | (M,ceil(K/128))       | -          |
| pergroup-perblock    | 2          | float8_e4m3fn/float8_e5m2/hifloat8 | float32                         | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(K/128+B, ceil(N/128))]                  | (K/128+B,M)           | -          |
| mx                   | 0          | float8_e4m3fn/float8_e5m2          | float8_e8m0                     | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N,ceil(K/64),2)]/[(B,ceil(K/64),N,2)] | (M,ceil(K/64),2)      | -          |
| mx                   | 2          | float8_e4m3fn/float8_e5m2          | float8_e8m0                     | [(K,M)]   | [(K,N)]             | [(B,M,N)] | [(K/64+B,N,2)]                            | (K/64+B,M,2)          | -          |
| mx                   | 0          | float4_e2m1                        | float8_e8m0                     | [(M,K)]   | [(B,N,K)]/[(B,K,N)] | [(M,N)]   | [(B,N,ceil(K/64),2)]/[(B,ceil(K/64),N,2)] | (M,ceil(K/64),2)      | -/[(B,N)]  |
| mx                 | 0     | float4_e2m1/float4_e1m2 | float8_e8m0 | [(M,K)] | [(B,ceil(K/64),ceil(N/16),16,64)]/[(B,ceil(N/64),ceil(K/16),16,64)] | [(M,N)] | [(B,N,ceil(K/64),2)]/[(B,ceil(K/64),N,2)] | (M,ceil(K/64),2) | -/[(B,N)] |

| quantization                      | group_type | x type        | weight type                        | per_token_scale type | scale type | antiquant_scale type | antiquant_offset type | x shape | weight shape                      | y shape | per_token_scale shape | scale shape | antiquant_scale shape       | antiquant_offset shape |
|---------------------------------- |------------|---------------|------------------------------------|----------------------|------------|----------------------|-----------------------|---------------------------------------------|---------|-----------------------|-------------|-----------------------------|------------------------|
| perchannel                        | 0          | float16       | int8                               | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | 0          | bfloat16      | int8                               | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | -1         | float16       | int8                               | -                    | uint64     | float16              | float16               | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | -1         | bfloat16      | int8                               | -                    | uint64     | bfloat16             | bfloat16              | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | 0          | float16       | float8_e4m3fn/float8_e5m2/hifloat8 | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]                         | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]                  |
| perchannel                        | 0          | bfloat16      | float8_e4m3fn/float8_e5m2/hifloat8 | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]                         | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]                  |
| perchannel                        | 0          | float16       | int4                               | -                    | uint64     | float16              | float16               | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | 0          | bfloat16      | int4                               | -                    | uint64     | bfloat16             | bfloat16              | [(M,K)] | [(B,N,K)]/[(B,K,N)]               | [(M,N)] | -                     | [(0)]       | [(B,N,)]                    | [(0)]/[(B,N,)]         |
| perchannel                        | -1         | float16       | int4                               | -                    | uint64     | float16              | float16               | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| perchannel                        | -1         | bfloat16      | int4                               | -                    | uint64     | bfloat16             | bfloat16              | [(M1,K1),(M2,K2),...] | [(N1,K1),(N2,K2),...]/[(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | - | [(0)] | [(N1,),(N2,)...] | [(0)]/[(N1,),(N2,)...] |
| pertoken-(perchannel & pergroup)  | 0          | int8          | int4                               | float32              | float32    | float16              | float16               | [(M,K)] | [(B,ceil(N/32),ceil(K/16),16,32)] | [(M,N)] | (M,)                  | [(B,N,)]    | [(B,ceil(K/group_size),N,)] | [(0)]                  |
| pertoken-perchannel               | 0          | int4          | int4                               | float32              | uint64     | -                    | -                     | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)]/[(B,ceil(N/32),ceil(K/16),16,32)] | [(M,N)] | (M,) | [(B,N,)]                    | -                   | -                      |
| pertoken-pergroup                 | 0          | int4          | int4                               | float32              | uint64     | -                    | -                     | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)]/[(B,ceil(N/32),ceil(K/16),16,32)] | [(M,N)] | (M,) | [(B,ceil(K/group_size),N,)] | -                   | -                      |
| mx                                | 0          | float16       | float4_e2m1                        | -                    | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,16)] | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | bfloat16      | float4_e2m1                        | -                    | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,16)] | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | float16       | float32                            | -                    | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,2)]  | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | bfloat16      | float32                            | -                    | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(N/16),ceil(K/16),16,2)]  | [(M,N)] | -                     | [(0)]       | [(B,ceil(K/32),N,)]         | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float4_e2m1/float4_e1m2            | float8_e8m0          | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)] | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float4_e2m1/float4_e1m2            | float8_e8m0          | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,32)] | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float32                            | float8_e8m0          | uint64     | float8_e8m0          | float16               | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,4)]  | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
| mx                                | 0          | float8_e4m3fn | float32                            | float8_e8m0          | uint64     | float8_e8m0          | bfloat16              | [(M,K)] | [(B,ceil(K/32),ceil(N/16),16,4)]  | [(M,N)] | (M,ceil(K/64)*2)      | [(0)]       | [(B,N,ceil(K/64)*2,)]       | [(0)]                  |
*\n
* @attention Constraints
* @li single-tensor: for tensor list type of input and output, tensors for different groups are not separated, and only one tensor in the tensor list.
* @li multi-tensor: for tensor list type of input and output, tensors for different groups are separated, and multi separated tensors in the tensor list.
* @li If group_list is passed, when group_list_type is 0, it must be a non-negative monotone non-decreasing array; when group_list_type is 1, it must be a non-negative array; when group_list_type is 2, it must be non-negative (group_idx, size) pairs, with non-zero groups placed first and zero groups placed last.
* @li Quantization and fake-quantization are supported only when group_type is set to -1 or 0 for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and Atlas Inference Series Product.
* Quantization is supported only when group_type is set to 0 or 2 for Ascend950PR/Ascend950DT.
* @li Quantization with y data type bfloat16 or float16 or float32 is only supported in single-tensor x, single-tensor weight, single-tensor y cases.
* @li Each axis for tensors in x and weight for each group of matmul should be less or equal to 2147483647 (maximum of data type int32) after aligning to 32 byte.
* @li When the dtype of the weight is int4, the K axis size should be even if the weight is transposed, otherwise, the N axis size should be even.
* @li In following tables, "S" stands for single-tensor, and "M" stands for multi-tensor, expressed in the sequence of x, weight, y.
* For example, "SMS" indicates single-tensor x, multi-tensor weight, and single-tensor y;
* "optional-dynamic inputs" stands for DYNAMIC_INPUT not needed in all scenarios, which includes bias/scale/antiquant_scale/antiquant_offset. Offset is not needed in all scenarios.
*\n
* The following are the supported shapes and constrains for different scenarios(for Atlas A2 Training Series Product/Atlas 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and Atlas Inference Series Product):
*\n
| group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |    constrains    |
| :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: | :--------------: |
| -1         |        MMM         | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |1) K1,K2,... < 65536.<br>2) N1,N2,... < 65536.<br>3) B <= 128.|
| 0          |        SSS         | [(M,K)] | [(B,K,N)] | [(M,N)] | [(B,N)] | (B) | (M) |1) K < 65536.<br>2) N < 65536.<br>3) B <= 1024.|
| 0          |        SMS         | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) K < 65536.<br>2) N < 65536.<br>3) B <= 128.|
| 0          |        MMS         | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) K1,K2,... < 65536.<br>2) N < 65536.<br>3) B <= 128.|
*\n
*\n
* The following are the supported shapes and constrains for different scenarios(for Ascend950PR/Ascend950DT):
*\n
| group_type | supported scenario | x shape | weight shape | y shape | optional-dynamic inputs shape if needed | group_list shape if passed | per_token_scale shape if passed |    constrains    |
| :--------: | :----------------: | :-----: | :----------: | :-----: | :-------------------------------------: | :------------------------: | :-----------------------------: | :--------------: |
| -1         |        MMM         | [(M1,K1),(M2,K2),...] | [(K1,N1),(K2,N2),...] | [(M1,N1),(M2,N2),...] | [(N1),(N2),...] | not support | not support |1) B <= 128.|
| 0          |        SSS         | [(M,K)] | [(B,K,N)]/[(B,N,K)] | [(M,N)] | See the constrains below | (B) | See the constrains below |1) B <= 1024.|
| 0          |        SMS         | [(M,K)] | [(K,N),(K,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) B <= 128.|
| 0          |        MMS         | [(M1,K1),(M2,K2),...] | [(K1,N),(K2,N),...] | [(M,N)] | [(N),(N),...] | (B) | not support |1) B <= 128.|
| 2          |        SSS         | [(K,M)] | [(K,N)] | [(B,M,N)] | See the constrains below | (B) | See the constrains below |1) B <= 1024.|
*\n
* The constrains of inputs' shape in quantization(for Ascend950PR/Ascend950DT) can be found in the table "The following are the supported quantization modes, data types and shapes(for Ascend950PR/Ascend950DT)" above.
* @li Shape of offset and not needed optional-dynamic inputs is [(0)].
* @li Shape of weight indicated in above table corresponds to data format ND. Weight with format NZ supports the single-tensor x, single-tensor weight, single-tensor y case and, in MX quantization, the single-tensor x, multi-tensor weight, single-tensor y case.
* @li When weight has format NZ, N axis should align to 32 bytes, i.e. if weight has data type int8 , N axis align to 32; if weight has data type float16 , N axis align to 16.
* @li In the MXFP4 scenario, the constraint that k must be even and not equal 2. Under non-transposed weight, n must also be even.
*\n
*\n
* @li The following are the supported group_type and constrains for different scenarios:
| group_type | supported scenario |    constrains    |
| :--------: | :----------------: | :--------------: |
| -1         |        MMM         |1) tensors in x support dim num 2-6, tensors in weight support dim num 2, tensors in y should have the same dim num with tensor in x.<br>2) group_list must be passed as null.|
| 0          |        SSS         |1) tensor in x and y should have dim num 2, tensor in weight should have dim num 3.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.|
| 0          |        SMS         |1) tensors in x, weight and y should have dim num 2.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.<br>3) The K axis and N axis of each tensor in weight must be the same.|
| 0          |        MMS         |1) tensors in x, weight and y should have dim num 2.<br>2) if group_list is passed, when group_list_type equals to 0, difference between two adjacent value in group_list should be consistent with the first dimension of each tensor in x; when group_list_type equals to 1, values in group_list should be consistent with the first dimension of each tensor in x.<br>3) The N axis of each tensor in weight must be the same.|
| 2          |        SSS         |1) tensor in x and weight should have dim num 2, tensor in y should have dim num 3.<br>2) group_list must be passed, if group_list_type equals to 0, the last value in group_list must be no greater than the first dimension of tensor in x; if group_list_type equals to 1, sum of values in group_list must be no greater than the first dimension of tensor in x.|
*\n
* @li Atlas Inference Series Product Constraints: \n
* - Atlas Inference Series Product only supports single-tensor x, single-tensor weight, single-tensor y, and group_type 0 cases. \n
* - Atlas Inference Series Product only supports x, weight and y have data type float16, and N axis should align to 16. \n
*/
// clang-format on
REG_OP(GroupedMatmul)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                  DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT4, DT_FLOAT8_E5M2,
                                       DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .DYNAMIC_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(scale, TensorType({DT_UINT64, DT_INT64, DT_BF16, DT_FLOAT32, DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(offset, TensorType({DT_FLOAT32}))
    .DYNAMIC_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(per_token_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT, DT_INT32}))
    .ATTR(split_item, Int, 0)
    .ATTR(dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(transpose_x, Bool, false)
    .ATTR(group_type, Int, -1)
    .ATTR(group_list_type, Int, 0)
    .ATTR(act_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .OP_END_FACTORY_REG(GroupedMatmul)

} // namespace ge

#endif
