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
 * \file kda_input_proj_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

#include "kda_input_proj_host_common.h"

namespace ops {
class KdaInputProj : public OpDef {
public:
    explicit KdaInputProj(const char *name) : OpDef(name)
    {
        this->Input(kda_input_proj::X_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input(kda_input_proj::WEIGHT_QKV_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input(kda_input_proj::WEIGHT_BETA_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input(kda_input_proj::WEIGHT_GATE_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input(kda_input_proj::WEIGHT_G_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input(kda_input_proj::WEIGHT_QKV_SCALE_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output(kda_input_proj::QKV_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output(kda_input_proj::BETA_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output(kda_input_proj::GATE_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output(kda_input_proj::G_NAME)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr(kda_input_proj::ATTR_TRANS_WEIGHT_QKV_NAME).AttrType(OPTIONAL).Bool(true);
        this->Attr(kda_input_proj::ATTR_TRANS_WEIGHT_BETA_NAME).AttrType(OPTIONAL).Bool(true);
        this->Attr(kda_input_proj::ATTR_TRANS_WEIGHT_GATE_NAME).AttrType(OPTIONAL).Bool(true);
        this->Attr(kda_input_proj::ATTR_TRANS_WEIGHT_G_NAME).AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(KdaInputProj);
} // namespace ops
