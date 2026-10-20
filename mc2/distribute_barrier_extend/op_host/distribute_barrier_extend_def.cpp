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
 * \file distribute_barrier_extend_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class DistributeBarrierExtend : public OpDef {
public:
    explicit DistributeBarrierExtend(const char *name) : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BOOL, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT8, ge::DT_INT4})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("time_out")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("elastic_info")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("x_ref")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BOOL, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32,
                       ge::DT_INT64, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT8, ge::DT_INT4})
            .FormatList({ge::FORMAT_ND});

        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("world_size").AttrType(REQUIRED).Int();

        // _apt entry (arch35/distribute_barrier_extend_apt.cpp).
        OpAICoreConfig aicore_config_apt;
        aicore_config_apt.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "distribute_barrier_extend_apt");

        this->AICore().AddConfig("ascend950", aicore_config_apt);
        this->AICore().AddConfig("ascend960dt", aicore_config_apt);
        this->AICore().AddConfig("ascend960pr", aicore_config_apt);
    }
};

OP_ADD(DistributeBarrierExtend);

} // namespace ops
