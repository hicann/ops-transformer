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
 * \file quant_grouped_matmul_inplace_add_tiling_utils.h
 * \brief
 */

#ifndef QUANT_GROUPED_MATMUL_INPLACE_ADD_TILING_UTILS_H
#define QUANT_GROUPED_MATMUL_INPLACE_ADD_TILING_UTILS_H

#include <exe_graph/runtime/tiling_context.h>

#include <cstdint>

#include "../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"

namespace QuantGroupedMatmulInplaceAdd {

// QuantGroupedInplaceAddTiling 和 QGMMInplaceAddBasicApiTiling 共用的校验/解析函数。
// 抽成自由函数避免两条继承路径复制逻辑，同时保持两边继承链独立（前者继承 GroupedQmmTiling，
// 后者继承 GroupedQmmBasicApiTiling）。

bool AnalyzeAttrsForInplaceAdd(const gert::TilingContext *context, optiling::GQmmInputInfo &inputParams);

// 只读取 GQmmInputInfo 的 dtype 字段，合法性校验由 CheckDtypeForInplaceAdd 负责
bool AnalyzeDtypeForInplaceAdd(const gert::TilingContext *context, optiling::GQmmInputInfo &inputParams);

// 校验 HIFLOAT8+HIFLOAT8 (T-T/T-C) 或 FP8_E4M3FN/E5M2+FP8_E4M3FN/E5M2 (MX) 组合
bool CheckDtypeForInplaceAdd(const optiling::GQmmInputInfo &inputParams);

bool CheckCoreNumForInplaceAdd(const gert::TilingContext *context, const optiling::GQmmInputInfo &inputParams);

// HIFLOAT8 量化 (T-T / T-C 合并)：
//   scale1: (g,) 或 (g, 1)
//   scale2: (g,)、(g, 1) 或 (g, n)，末轴 1 走 T-T，n 走 T-C
// 与 aclnn 的 CheckHif8QuantParamsShape 语义对齐。
bool CheckShapeForHif8Quant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                            const optiling::GQmmInputInfo &inputParams);

// MX 量化：scale1/scale2 均为 3D，末轴 = MXFP_MULTI_BASE_SIZE
bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                          const optiling::GQmmInputInfo &inputParams);

// 由 (transB, transA, kernelType) 三元组生成 tiling key，BasicApi 与 普通路径共用
uint64_t GetTilingKeyForInplaceAdd(const optiling::GQmmInputInfo &inputParams);

} // namespace QuantGroupedMatmulInplaceAdd

#endif // QUANT_GROUPED_MATMUL_INPLACE_ADD_TILING_UTILS_H
