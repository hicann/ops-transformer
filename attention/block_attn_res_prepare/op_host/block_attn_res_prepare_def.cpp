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
 * \file block_attn_res_prepare_def.cpp
 * \brief BlockAttnResPrepare operator definition.
 */

#include "register/op_def_registry.h"

namespace ops {
namespace {

constexpr float DEFAULT_EPS = 1e-6F;

} // namespace

class BlockAttnResPrepare : public OpDef {
public:
    explicit BlockAttnResPrepare(const char *name)
        : OpDef(name)
    {
        this->Input("block_res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("valid_blocks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pseudo_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("numerator")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("logit_max")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("exp_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("eps").AttrType(OPTIONAL).Float(DEFAULT_EPS);

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "block_attn_res_prepare_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(BlockAttnResPrepare);

} // namespace ops
