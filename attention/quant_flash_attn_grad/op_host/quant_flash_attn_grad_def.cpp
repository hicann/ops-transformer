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
 * \file quant_flash_attn_grad_def.cpp
 * \brief QuantFlashAttnGrad算子定义（量化注意力反向梯度）
 * 输入q/k/v/do为UINT8（表达fp4_e2m1/hif8/hif4量化），输出dq/dk/dv/dsink固定BF16。
 * 支持BSND/BNSD/TND三种layout。
 */

#include "op_common/op_host/util/op_arch_helper.h"
#include "register/op_def_registry.h"

namespace ops {

class QuantFlashAttnGrad : public OpDef {
public:
    explicit QuantFlashAttnGrad(const char *name)
        : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("do_descale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("p_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ds_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("softmax_lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_kv")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sinks")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dk")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dsink")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("quant_mode").AttrType(REQUIRED).Int();
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("win_left").AttrType(OPTIONAL).Int(-1);
        this->Attr("win_right").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");

        OpAICoreConfig aicore_config_3510;
        aicore_config_3510.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "quant_flash_attn_grad");

        // ascend950 按 NpuArch::DAV_3510 注册：opbase 的 g_archToSocNames 映射
        // DAV_3510 -> {"ascend950", "ascend350"}，与逐个 AddConfig 语义等价（对齐 ops-nn matmul 族用法）。
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicore_config_3510);
    }
};

OP_ADD(QuantFlashAttnGrad);

} // namespace ops
