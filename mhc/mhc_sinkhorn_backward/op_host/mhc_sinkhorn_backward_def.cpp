/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_backward_def.cpp
 * \brief mhc_sinkhorn_backward_def
 */
#include "register/op_def_registry.h"

namespace ops
{
class MhcSinkhornBackward: public OpDef {
public:
    explicit MhcSinkhornBackward(const char* name) : OpDef(name)
    {
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        this->Input("sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mhc_sinkhorn_backward_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(MhcSinkhornBackward);
}  // namespace ops
