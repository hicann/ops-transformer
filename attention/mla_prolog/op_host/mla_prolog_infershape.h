/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_infershape.h
 * \brief
 */

#ifndef MLA_PROLOG_INFERSHAPE_H
#define MLA_PROLOG_INFERSHAPE_H

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
//input
constexpr uint32_t TOKEN_X_INDEX = 0;
constexpr uint32_t WEIGHT_UK_INDEX = 3;
constexpr uint32_t ROPE_SIN_INDEX = 7;
constexpr uint32_t KV_CACHE_INDEX = 10;
constexpr uint32_t KR_CACHE_INDEX = 11;
constexpr uint32_t KV_CACHE_INDEX_V3 = 9;
constexpr uint32_t KR_CACHE_INDEX_V3 = 10;
// output
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t QUERY_ROPE_INDEX = 1;
constexpr uint32_t KV_CACHE_OUT_INDEX = 2;
constexpr uint32_t KR_CACHE_OUT_INDEX = 3;
// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

struct MlaProlgoProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t N { 0 };
    int64_t Hckv { 0 };
    int64_t He { 0 };
    int64_t Dr { 0 };
};

ge::graphStatus GetMlaPrologShapeDim(const gert::InferShapeContext* context, MlaProlgoProtoShapeParam &shapeParam);
ge::graphStatus SetMlaPrologShapeDim(const MlaProlgoProtoShapeParam &shapeParam, gert::InferShapeContext* context);
ge::graphStatus InferShapeMlaProlog(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeMlaProlog(gert::InferDataTypeContext* context);
}  // namespace ops

#endif // MLA_PROLOG_INFERSHAPE_H