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
 * \file moe_masked_scatter_def.cpp
 * \brief MoeMaskedScatter ophost
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16
};

static const std::vector<ge::DataType> MASK_SUPPORT_DTYPE = {
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL
};

static const std::vector<ge::Format> SUPPORT_FORMAT = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
class MoeMaskedScatter : public OpDef {
 public:
  explicit MoeMaskedScatter(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Input("mask")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(MASK_SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Input("updates")
        .ParamType(REQUIRED)
        .AutoContiguous()
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(SUPPORT_DTYPE)
        .Format(SUPPORT_FORMAT)
        .UnknownShapeFormat(SUPPORT_FORMAT);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true);
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};

OP_ADD(MoeMaskedScatter);
}  // namespace ops
