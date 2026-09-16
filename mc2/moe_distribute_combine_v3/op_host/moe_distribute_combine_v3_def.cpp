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
 * \file moe_distribute_combine_v3_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "../../common/op_host/mc2_moe_input_registration.h"

namespace ops {
class MoeDistributeCombineV3 : public OpDef {
public:
    explicit MoeDistributeCombineV3(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        Mc2OpDef::DefineCombineRequiredInputs(*this);

        Mc2OpDef::DefineCombineOptionalInputs(*this);

        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        DefineAttributes();

        // A3 (arch22): _a3 entry (opFile -> arch22/..._a3.cpp). Reuses the class-level I/O.
        OpAICoreConfig aicore_config_a3;
        aicore_config_a3.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "moe_distribute_combine_v3_a3");

        // A5 (arch35): dedicated kernel entry (opFile -> arch35/..._apt.cpp).
        OpAICoreConfig aicore_config_apt;
        aicore_config_apt.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "moe_distribute_combine_v3_apt");

        this->AICore().AddConfig("ascend910_93", aicore_config_a3);
        this->AICore().AddConfig("ascend950", aicore_config_apt);
    }

private:
    void DefineAttributes()
    {
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ep_rank_id").AttrType(REQUIRED).Int();
        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("tp_world_size").AttrType(OPTIONAL).Int(0);
        this->Attr("tp_rank_id").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_shard_type").AttrType(OPTIONAL).Int(0);
        this->Attr("shared_expert_num").AttrType(OPTIONAL).Int(1);
        this->Attr("shared_expert_rank_num").AttrType(OPTIONAL).Int(0);
        this->Attr("global_bs").AttrType(OPTIONAL).Int(0);
        this->Attr("out_dtype").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        this->Attr("zero_expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("copy_expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("const_expert_num").AttrType(OPTIONAL).Int(0);
    }
};

OP_ADD(MoeDistributeCombineV3);

} // namespace ops
