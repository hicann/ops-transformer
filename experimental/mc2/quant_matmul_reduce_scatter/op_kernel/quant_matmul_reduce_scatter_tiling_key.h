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
 * \file quant_matmul_reduce_scatter_tiling_key.h
 * \brief TilingKey 模板实参（TPL）声明：TRANSB 决定 x2(B矩阵) 布局 → kernel LayoutB
 */

#ifndef QUANT_MATMUL_REDUCE_SCATTER_TILING_KEY_H
#define QUANT_MATMUL_REDUCE_SCATTER_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// TRANSB: x2 是否转置存储（attr transpose_x2）。true=[N,K]（转置 B），false=[K,N]（非转置 B）
// x2 format（ND/NZ）不进 tiling key：kernel 侧由 opc 注入的 FORMAT_X2 宏编译期判断
// TRANSB 决定 B 布局，format（ND/NZ）由 FORMAT_X2 宏区分
ASCENDC_TPL_ARGS_DECL(QuantMatmulReduceScatter, // 算子OpType
                      ASCENDC_TPL_BOOL_DECL(TRANSB, 0, 1), );

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
// 两种转置语义均为真实运行路径：SEL 展开为两份 kernel 二进制（tiling key 0/1）
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TRANSB, 0, 1)));

#endif // QUANT_MATMUL_REDUCE_SCATTER_TILING_KEY_H
