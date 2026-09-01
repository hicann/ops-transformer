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
 * \file ffn_to_attention_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class FFNToAttentionV2 : public OpDef {
public:
    explicit FFNToAttentionV2(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("session_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("micro_batch_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("token_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_offsets")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_token_num")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_rank_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("world_size").AttrType(REQUIRED).Int();
        this->Attr("token_info_table_shape").AttrType(REQUIRED).ListInt();
        this->Attr("token_data_shape").AttrType(REQUIRED).ListInt();
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config);
        // this->MC2().HcclGroup({"group"});
    }
};

OP_ADD(FFNToAttentionV2);
} // namespace ops
