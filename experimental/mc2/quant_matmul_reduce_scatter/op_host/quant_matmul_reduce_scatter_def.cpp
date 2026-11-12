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
 * \file quant_matmul_reduce_scatter.cpp
 * \brief 算子信息库定义
 */
#include <register/op_def_registry.h>

namespace ops {
class QuantMatmulReduceScatter : public OpDef {
public:
    explicit QuantMatmulReduceScatter(const char* name) : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // x1/x2 dtype 配对按等长列表 index 配对口径，共 9 组组合：
        //   idx 0-3 : x1=e4m3 + x2=e4m3（y bf16/fp16 × x2 ND/NZ）
        //   idx 4-7 : x1=e5m2 + x2=e5m2（同上 4 条）
        //   idx 8-11: x1=e4m3 + x2=e5m2（同上 4 条）
        //   idx 12-15: x1=e5m2 + x2=e4m3（同上 4 条）
        //   idx 16-17: x1=fp4  + x2=fp4 （fp4 仅支持 ND，x2 不配 NZ）
        // fp8 类型可任意配对，fp4 只能与 fp4 配对；y 取 bf16/fp16，scale 固定 e8m0。
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("x1_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("hccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("rank_size").AttrType(OPTIONAL).Int(0);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_mode").AttrType(OPTIONAL).String("ubmem");

        OpAICoreConfig aicore_config_950;
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true") // 动态shape，复用二进制，后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};
OP_ADD(QuantMatmulReduceScatter);
}  // namespace ops
