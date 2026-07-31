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
 * \file dense_lightning_indexer_softmax_lse_v2_tiling.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_H_
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DenseLISoftmaxLseV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, b)
TILING_DATA_FIELD_DEF(int64_t, s1)
TILING_DATA_FIELD_DEF(int64_t, s2)
TILING_DATA_FIELD_DEF(int64_t, n1)
TILING_DATA_FIELD_DEF(int64_t, d)
TILING_DATA_FIELD_DEF(int64_t, cmp_ratio)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(DenseLightningIndexerSoftmaxLseV2, DenseLISoftmaxLseV2TilingData)

struct DenseLISoftmaxLseV2CompileInfo {
    static ge::graphStatus ParamCheck(gert::TilingContext *context, int64_t layout, int64_t maskMode, int64_t cmpRatio,
                                      const std::string &layoutQStr, const std::string &layoutKStr, int64_t bSize,
                                      int64_t s1Size, int64_t s2Size, int64_t n1Size, int64_t n2Size, int64_t dSize,
                                      int64_t keyB, int64_t keyD, int64_t weightB, int64_t weightS1, int64_t weightN1,
                                      int64_t outDim0, int64_t outDim1, int64_t outDim2);
    static ge::graphStatus CheckShapeDims(gert::TilingContext *context, int64_t layout,
                                          const gert::StorageShape *queryShape, const gert::StorageShape *keyShape,
                                          const gert::StorageShape *weightShape);
};

} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_TILING_H_
