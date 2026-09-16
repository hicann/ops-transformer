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
 * \file moe_distribute_combine_teardown_def.cpp
 * \brief 算子信息库定义
 */

#include "register/op_def_registry.h"

namespace ops {

class MoeDistributeCombineTeardown : public OpDef {
public:
    explicit MoeDistributeCombineTeardown(const char *name)
        : OpDef(name)
    {
        DefineRequiredInputs();
        DefineOptionalInputs();
        this->Output("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        DefineAttributes();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        this->AICore().AddConfig("ascend950", aicore_config);
        this->MC2().HcclGroup("group_ep");
    }

private:
    void DefineRequiredInputs()
    {
        this->Input("expand_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("quant_expand_x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expand_idx")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_scales")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("comm_cmd_info")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineOptionalInputs()
    {
        this->Input("x_active_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("shared_expert_x")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
    }

    void DefineAttributes()
    {
        this->Attr("group_ep").AttrType(REQUIRED).String();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ep_rank_id").AttrType(REQUIRED).Int();
        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        this->Attr("expert_shard_type").AttrType(OPTIONAL).Int(0);
        this->Attr("shared_expert_num").AttrType(OPTIONAL).Int(1);
        this->Attr("shared_expert_rank_num").AttrType(OPTIONAL).Int(0);
        this->Attr("global_bs").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_type").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
    }
};

OP_ADD(MoeDistributeCombineTeardown);

} // namespace ops
