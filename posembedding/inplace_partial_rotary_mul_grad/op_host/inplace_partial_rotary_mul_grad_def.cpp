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
 * \file inplace_partial_rotary_mul_grad_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> DyDtype = {{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16}};

static const std::vector<ge::DataType> cosDtype = {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16,
                                                    ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                    ge::DT_BF16}};
static const std::vector<ge::Format> formatList = {{ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                    ge::FORMAT_ND}};
class InplacePartialRotaryMulGrad : public OpDef {
public:
    explicit InplacePartialRotaryMulGrad(const char *name) : OpDef(name)
    {
        this->Input("dy").ParamType(REQUIRED).DataType(DyDtype).Format(formatList);
        this->Input("cos").ParamType(REQUIRED).DataType(cosDtype).Format(formatList);
        this->Input("sin").ParamType(REQUIRED).DataType(cosDtype).Format(formatList);
        this->Output("dy").ParamType(REQUIRED).DataType(DyDtype).Format(formatList);
        this->Attr("rotary_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("partial_slice").AttrType(OPTIONAL).ListInt({0, 0});

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "inplace_partial_rotary_mul_grad_apt");
        this->AICore().AddConfig("ascend950", config950);
        this->AICore().AddConfig("ascend960dt", config950);
    }
};

OP_ADD(InplacePartialRotaryMulGrad);
} // namespace ops
