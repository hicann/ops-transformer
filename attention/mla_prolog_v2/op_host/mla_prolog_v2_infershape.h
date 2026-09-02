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
 * \file mla_prolog_v2_infershape.h
 * \brief
 */

#ifndef MLA_PROLOG_V2_INFERSHAPE_H
#define MLA_PROLOG_V2_INFERSHAPE_H

#include "../../mla_prolog/op_host/mla_prolog_infershape.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
// OUTPUT
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;

ge::graphStatus SetMlaPrologV2ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context);

ge::graphStatus InferShapeMlaPrologV2(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeMlaPrologV2(gert::InferDataTypeContext *context);

} // namespace ops

#endif // MLA_PROLOG_V2_INFERSHAPE_H
