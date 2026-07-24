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
 * \file masked_causal_conv1d_def.cpp
 * \brief MaskedCausalConv1d operator definition
 */

#include "register/op_def_registry.h"

namespace ops {

class MaskedCausalConv1d : public OpDef {
public:
    explicit MaskedCausalConv1d(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});

        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "masked_causal_conv1d_apt");
        this->AICore().AddConfig("ascend950", config_950);
    }
};

OP_ADD(MaskedCausalConv1d);

} // namespace ops
