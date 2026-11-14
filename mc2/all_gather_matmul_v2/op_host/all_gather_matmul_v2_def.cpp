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
 * \file all_gather_matmul_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "../../common/op_host/mc2_matmul_v2_op_def_common.h"

// 旧版 CANN(如 9.2.0)的 op_build 生成器不支持 DT_HIFLOAT4/Ascend960,
// 通过 devkit 版本宏守卫整段 960 配置: CANN >= 9.3 才注册(参照 allto_all_matmul 同型守卫)
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4
#endif
#endif

namespace ops {
class AllGatherMatmulV2 : public OpDef {
public:
    explicit AllGatherMatmulV2(const char *name)
        : OpDef(name)
    {
        DefineRequiredInputs();
        DefineOptionalInputs();
        DefineOutputs();
        DefineAttributes();

        OpAICoreConfig aicore_config;
        Mc2OpDef::ConfigureMatmulV2DynamicAicore(aicore_config)
            .ExtendCfgInfo("opFile.value", "all_gather_matmul_v2_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
        this->MC2().HcclGroup("group");
#ifdef ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4
        // A6(ascend960dt)：在主线 32 dtype 组合基础上追加 HiFP4 组合（单配置注册，超集覆盖基础组合；
        // 结构对齐 allto_all_matmul_def.cpp 的编译态守卫，工具链不支持时关闭该宏即可隔离 960 配置）
        OpAICoreConfig aicore_config_960dt;
        DefineAicoreConfig960dt(aicore_config_960dt);
        Mc2OpDef::ConfigureMatmulV2DynamicAicore(aicore_config_960dt)
            .ExtendCfgInfo("opFile.value", "all_gather_matmul_v2_apt");
        this->AICore().AddConfig("ascend960dt", aicore_config_960dt);
        this->MC2().HcclGroup("group");
#endif

        OpAICoreConfig aicore_config_910b;
        DefineAicoreConfig910bRequiredInputs(aicore_config_910b);
        DefineAicoreConfig910bOptionalInputs(aicore_config_910b);
        DefineAicoreConfig910bOutputs(aicore_config_910b);

        Mc2OpDef::ConfigureMatmulV2DynamicAicore(aicore_config_910b);
        this->AICore().AddConfig("ascend910b", aicore_config_910b);
        this->AICore().AddConfig("ascend910_93", aicore_config_910b);
        this->MC2().HcclGroup("group");
    }

private:
    void DefineRequiredInputs()
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
    }

    void DefineOptionalInputs()
    {
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType(Mc2OpDef::GetMatmulV2ScaleDataTypes())
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType(Mc2OpDef::GetMatmulV2ScaleDataTypes())
            .FormatList({ge::FORMAT_ND});
        this->Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineOutputs()
    {
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Output("gather_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1})
            .FormatList({ge::FORMAT_ND});
        this->Output("amax_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
    }

    void DefineAttributes()
    {
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("is_trans_a").AttrType(OPTIONAL).Bool(false);
        this->Attr("is_trans_b").AttrType(OPTIONAL).Bool(false);
        this->Attr("gather_index").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_turn").AttrType(OPTIONAL).Int(0);
        this->Attr("rank_size").AttrType(OPTIONAL).Int(0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("is_gather_out").AttrType(OPTIONAL).Bool(true);
        this->Attr("is_amax_out").AttrType(OPTIONAL).Bool(false);
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int>(ge::DT_UNDEFINED));
        this->Attr("comm_mode").AttrType(REQUIRED).String("ai_cpu");
    }

    void DefineAicoreConfig910bRequiredInputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                     ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
    }

    void DefineAicoreConfig910bOptionalInputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void DefineAicoreConfig910bOutputs(OpAICoreConfig &aicore_config_910b) const
    {
        aicore_config_910b.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Output("gather_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                       ge::DT_INT8, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT4, ge::DT_INT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
        aicore_config_910b.Output("amax_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND});
    }

    // A6(ascend960dt) 专用配置：在全局 32 dtype 基础上额外开放 DT_HIFLOAT4 / DT_HIFLOAT4_SCALE
    void DefineAicoreConfig960dt(OpAICoreConfig &aicore_config_960dt) const
    {
        aicore_config_960dt.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       Ops::Transformer::DT_HIFLOAT4,      Ops::Transformer::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_960dt.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       Ops::Transformer::DT_HIFLOAT4,      Ops::Transformer::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_960dt.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_960dt.Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       Ops::Transformer::DT_HIFLOAT4_SCALE, Ops::Transformer::DT_HIFLOAT4_SCALE})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_960dt.Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,         ge::DT_FLOAT,
                       ge::DT_FLOAT,         ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,   ge::DT_FLOAT8_E8M0,
                       Ops::Transformer::DT_HIFLOAT4_SCALE, Ops::Transformer::DT_HIFLOAT4_SCALE})
            .FormatList({ge::FORMAT_ND});
        aicore_config_960dt.Input("quant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_960dt.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_960dt.Output("gather_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16,       ge::DT_BF16,          ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                       ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,
                       Ops::Transformer::DT_HIFLOAT4,      Ops::Transformer::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND});
        aicore_config_960dt.Output("amax_out")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
    }
};

OP_ADD(AllGatherMatmulV2);
} // namespace ops
