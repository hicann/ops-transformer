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
 * \file matmul_reduce_scatter_v2.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
// x1_scale/x2_scale 公共 dtype 列表（950 场景：前段为 fp32，后段为 fp8_e8m0）
static const std::vector<ge::DataType> scaleDtypes950 = {
    ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
    ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
    ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT,
    ge::DT_FLOAT,       ge::DT_FLOAT,       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};
// 910b mm 输入 x1/x2 公共 dtype 列表
static const std::vector<ge::DataType> mmDtypes910b = {
    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8,   ge::DT_INT8,
    ge::DT_INT8, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_INT8, ge::DT_INT8,   ge::DT_INT8,
    ge::DT_INT8, ge::DT_INT8,    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16};
// 910b 纯 fp32 输入公共 dtype 列表（x1_scale/quant_scale/amax_out）
static const std::vector<ge::DataType> floatDtypes910b = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
// 910b ND 布局列表
static const std::vector<ge::Format> ndFormats910b(mmDtypes910b.size(), ge::FORMAT_ND);
// 910b x2 输入 ND/NZ 混合布局列表（Format 与 UnknownShapeFormat 相同）
static const std::vector<ge::Format> nzFormats910b = {
    ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ,
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
    ge::FORMAT_ND,         ge::FORMAT_ND,         ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND,         ge::FORMAT_ND};
} // namespace

class MatmulReduceScatterV2 : public OpDef {
public:
    explicit MatmulReduceScatterV2(const char *name)
        : OpDef(name)
    {
        DefineRequiredInputs();
        DefineOptionalInputs();
        DefineOutputs();
        DefineAttributes();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape,复用二进制,后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "matmul_reduce_scatter_v2_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
        this->MC2().HcclGroup("group");

        OpAICoreConfig aicore_config_910b;
        DefineAicoreConfig910bRequiredInputs(aicore_config_910b);
        DefineAicoreConfig910bOptionalInputs(aicore_config_910b);
        DefineAicoreConfig910bOutputs(aicore_config_910b);

        aicore_config_910b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape,复用二进制,后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config_910b);
        this->AICore().AddConfig("ascend910_93", aicore_config_910b);
        this->MC2().HcclGroup("group");
    }

private:
    void DefineRequiredInputs()
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,          ge::DT_FLOAT16,       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
    }

    void DefineOptionalInputs()
    {
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Input("x1_scale").ParamType(OPTIONAL).DataType(scaleDtypes950).FormatList({ge::FORMAT_ND});

        this->Input("x2_scale").ParamType(OPTIONAL).DataType(scaleDtypes950).FormatList({ge::FORMAT_ND});
        this->Input("quant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    }

    void DefineOutputs()
    {
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Output("amax_out").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    }

    void DefineAttributes()
    {
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("reduce_op").AttrType(REQUIRED).String("sum");
        this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
        this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
        this->Attr("comm_turn").AttrType(OPTIONAL).Int(0);
        this->Attr("rank_size").AttrType(OPTIONAL).Int(0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("is_amax_out").AttrType(OPTIONAL).Bool(false);
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int>(ge::DT_UNDEFINED));
        this->Attr("comm_mode").AttrType(REQUIRED).String("ai_cpu");
    }

    void DefineAicoreConfig910bRequiredInputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Input("x1")
            .ParamType(REQUIRED)
            .DataType(mmDtypes910b)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b)
            .AutoContiguous();
        aicore_config_910b.Input("x2")
            .ParamType(REQUIRED)
            .DataType(mmDtypes910b)
            .Format(nzFormats910b)
            .UnknownShapeFormat(nzFormats910b)
            .IgnoreContiguous();
    }

    void DefineAicoreConfig910bOptionalInputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b)
            .AutoContiguous();
        aicore_config_910b.Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType(floatDtypes910b)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b)
            .AutoContiguous();
        aicore_config_910b.Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b)
            .AutoContiguous();
        aicore_config_910b.Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType(floatDtypes910b)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b)
            .AutoContiguous();
    }

    void DefineAicoreConfig910bOutputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
                       ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b);
        aicore_config_910b.Output("amax_out")
            .ParamType(OPTIONAL)
            .DataType(floatDtypes910b)
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat(ndFormats910b);
    }
};

OP_ADD(MatmulReduceScatterV2);
} // namespace ops
