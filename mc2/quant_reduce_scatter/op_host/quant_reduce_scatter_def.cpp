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
 * \file quant_reduce_scatter.cpp
 * \brief 算子信息库定义
 */
#include <register/op_def_registry.h>

namespace ops {
class QuantReduceScatter : public OpDef {
public:
    explicit QuantReduceScatter(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT4_E1M2,
                       ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,
                       ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT4_E1M2})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC});
        this->Input("scales")
            .ParamType(REQUIRED)
            .DataType(
                {ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT,       ge::DT_FLOAT,
                 ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0,
                 ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC});

        this->Output("out_put")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC});

        this->Attr("hccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("reduce_op").AttrType(OPTIONAL).String("sum");
        this->Attr("output_dtype")
            .AttrType(OPTIONAL)
            .Int(static_cast<int64_t>(ge::DT_BF16)); // 默认值为bf16，check一下对应的枚举值
        this->Attr("world_size").AttrType(REQUIRED).Int();

        // ascend950 AI处理器定义OpAICoreConfig变量，定制化配置参数
        OpAICoreConfig aicore_config_950;
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape，复用二进制，后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};

OP_ADD(QuantReduceScatter);

} // namespace ops
