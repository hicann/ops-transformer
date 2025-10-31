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
 * \file mla_prolog_v3_infershape.h
 * \brief
 */

#ifndef MLA_PROLOG_V3_INFERSHAPE_H
#define MLA_PROLOG_V3_INFERSHAPE_H

#include "../../mla_prolog/op_host/mla_prolog_infershape.h"

using namespace ge;

namespace ops {
// INPUT
constexpr uint32_t WEIGHT_UQ_QR_INDEX = 2;
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
// OUTPUT
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;
constexpr uint32_t QUERY_NORM_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_Q_NORM_INDEX = 6;

ge::graphStatus SetMlaPrologV3ShapeDim(const MlaProlgoProtoShapeParam &shapeParam, gert::InferShapeContext* context);

ge::graphStatus InferShapeMlaPrologV3(gert::InferShapeContext* context);
ge::graphStatus InferDataTypeMlaPrologV3(gert::InferDataTypeContext* context);


}  // namespace ops

#endif // MLA_PROLOG_V3_INFERSHAPE_H