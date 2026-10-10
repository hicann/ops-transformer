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
 * \file mhc_sinkhorn_def.cpp
 * \brief mhc_sinkhorn_def
 */
#include "register/op_def_registry.h"

namespace ops
{
class MhcSinkhorn: public OpDef {
public:
    explicit MhcSinkhorn(const char* name) : OpDef(name)
    {
        this->Input("h_res")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        this->Output("norm_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        this->Output("sum_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
           .Format({ge::FORMAT_ND})
           .UnknownShapeFormat({ge::FORMAT_ND})
           .AutoContiguous();
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-06);
        this->Attr("num_iters").AttrType(OPTIONAL).Int(20);
        this->Attr("out_flag").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "mhc_sinkhorn_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
        this->AICore().AddConfig("ascend350", aicore_config);
    }
};

OP_ADD(MhcSinkhorn);
}  // namespace ops
