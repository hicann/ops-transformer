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
  * \file mixed_quant_flash_attn_def.cpp
  * \brief MixedQuantFlashAttn算子定义（混合量化，Q不量化KV量化）
  *        Q数据类型仅支持FLOAT16和BFLOAT16。
  *        KV数据类型仅支持FP4_E2M1(MXFP4)。
  *        KV仅支持PA_BBND/PA_BNBD分页注意力layout。
  *        k_descale/v_descale仅支持FLOAT8_E8M0(MX规范缩放因子)。
  *        quant_mode仅支持1(Q不量化KV MXFP4)。
  */


 #include "register/op_def_registry.h"


 namespace ops {


 class MixedQuantFlashAttn : public OpDef {
 public:
     explicit MixedQuantFlashAttn(const char *name) : OpDef(name)
     {
         this->Input("q")
             .ParamType(REQUIRED)
             .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
             .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
         this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
         this->Input("k_descale")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_HIFLOAT4_SCALE, ge::DT_HIFLOAT4_SCALE})
             .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("v_descale")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_HIFLOAT4_SCALE, ge::DT_HIFLOAT4_SCALE})
             .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("block_table")
             .ParamType(OPTIONAL)
             .DataTypeList({ge::DT_INT32})
             .FormatList({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("cu_seqlens_q")
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
         this->Output("attn_out")
             .ParamType(REQUIRED)
             .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
             .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .AutoContiguous();
         this->Output("softmax_lse")
             .ParamType(OPTIONAL)
             .DataTypeList({ge::DT_FLOAT})
             .FormatList({ge::FORMAT_ND})
             .AutoContiguous();
         this->Attr("quant_compute_mode")
             .AttrType(REQUIRED)
             .Int(1);
         this->Attr("softmax_scale")
             .AttrType(OPTIONAL)
             .Float(0.0f);
         this->Attr("mask_mode")
             .AttrType(OPTIONAL)
             .Int(0);
         this->Attr("win_left")
             .AttrType(OPTIONAL)
             .Int(-1);
         this->Attr("win_right")
             .AttrType(OPTIONAL)
             .Int(-1);
         this->Attr("max_seqlen_q")
             .AttrType(OPTIONAL)
             .Int(-1);
         this->Attr("max_seqlen_kv")
             .AttrType(OPTIONAL)
             .Int(-1);
         this->Attr("layout_q")
             .AttrType(OPTIONAL)
             .String("BSND");
         this->Attr("layout_kv")
             .AttrType(OPTIONAL)
             .String("PA_BBND");
         this->Attr("layout_attn_out")
             .AttrType(OPTIONAL)
             .String("BSND");
         this->Attr("return_softmax_lse")
             .AttrType(OPTIONAL)
             .Bool(false);


         OpAICoreConfig aicore_config_96;
         aicore_config_96.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(true)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("prebuildPattern.value", "Opaque")
             .ExtendCfgInfo("coreType.value", "AiCore")
             .ExtendCfgInfo("opFile.value", "mixed_quant_flash_attn")
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");   // set value of aclnn support

         this->AICore().AddConfig("ascend960dt", aicore_config_96);
     }
 };


 OP_ADD(MixedQuantFlashAttn);


 }  // namespace ops
