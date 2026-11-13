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
 * \file quant_lightning_indexer_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "common/include/op_host/op_const_def_transformer.h"

#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ENABLE_DT_HIFLOAT4
#endif
#endif

namespace ops {
class QuantLightningIndexerV2 : public OpDef {
public:
    explicit QuantLightningIndexerV2(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({Ops::Transformer::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType({Ops::Transformer::DT_INT8}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("w").ParamType(REQUIRED).DataType({Ops::Transformer::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("q_descale")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({Ops::Transformer::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_values").ParamType(REQUIRED).DataTypeList({Ops::Transformer::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int(2048);       // 2048: 筛选前2048个作为输出index
        this->Attr("quant_mode").AttrType(REQUIRED).Int(1);    // 1: per-token-head
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1); // -1: 默认值，表示任意可能长度
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0); // 0: 默认值，无mask
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0); //  0: 默认值
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        OpAICoreConfig aicore_config_95;
        // fp8/mxfp8/hif8/mxfp4
        aicore_config_95.Input("q")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_HIFLOAT8, Ops::Transformer::DT_FLOAT4_E2M1, Ops::Transformer::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("k")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_HIFLOAT8, Ops::Transformer::DT_FLOAT4_E2M1, Ops::Transformer::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_95.Input("w")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("q_descale")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("k_descale")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_95.Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_95.Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicore_config_95);
        this->AICore().AddConfig("ascend350", aicore_config_95);

#ifdef ENABLE_DT_HIFLOAT4
        OpAICoreConfig aicore_config_96;
        // fp8/mxfp8/hif8/mxfp4/hif4
        aicore_config_96.Input("q")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_HIFLOAT8, Ops::Transformer::DT_FLOAT4_E2M1, Ops::Transformer::DT_INT8, Ops::Transformer::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("k")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_FLOAT8_E4M3FN, Ops::Transformer::DT_HIFLOAT8, Ops::Transformer::DT_FLOAT4_E2M1, Ops::Transformer::DT_INT8, Ops::Transformer::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_96.Input("w")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT16, Ops::Transformer::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("q_descale")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT16, Ops::Transformer::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("k_descale")
            .ParamType(REQUIRED)
            .DataType({Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT, Ops::Transformer::DT_FLOAT8_E8M0, Ops::Transformer::DT_FLOAT16, Ops::Transformer::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicore_config_96.Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_96.Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        aicore_config_96.Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({Ops::Transformer::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        aicore_config_96.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend960dt", aicore_config_96);
#endif
    }
};
OP_ADD(QuantLightningIndexerV2);
} // namespace ops
