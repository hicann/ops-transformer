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
 * \file all_gather_add_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AllGatherAdd : public OpDef {
public:
    explicit AllGatherAdd(const char *name) : OpDef(name)
    {
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});

        this->Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("gather_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});

        this->Attr("group").AttrType(REQUIRED).String(); // 通算融合算子属性，表示通信域名称
        this->Attr("rank_size").AttrType(REQUIRED).Int(0);

        // arch22(A2/A3) 入口: op_kernel/all_gather_add_a3.cpp，AICPU 高层 Hccl。
        OpAICoreConfig aicore_config_arch22;
        aicore_config_arch22.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "all_gather_add_a3");

        // arch35(A5) 入口: op_kernel/all_gather_add_apt.cpp，MTE 裸窗口直写。
        OpAICoreConfig aicore_config_arch35;
        aicore_config_arch35.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "all_gather_add_apt");

        this->AICore().AddConfig("ascend910b", aicore_config_arch22);
        this->AICore().AddConfig("ascend910_93", aicore_config_arch22);
        this->AICore().AddConfig("ascend950", aicore_config_arch35);
        this->MC2().HcclGroup("group"); // group 属性配置为该算子的通信域名称
    }
};

OP_ADD(AllGatherAdd);
} // namespace ops
