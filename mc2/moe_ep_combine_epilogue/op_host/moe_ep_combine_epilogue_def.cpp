/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

class MoeEpCombineEpilogue : public OpDef {
public:
    explicit MoeEpCombineEpilogue(const char *name)
        : OpDef(name)
    {
        this->Input("context")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("topk_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("combined_x")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("combined_topk_weights").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ep_rank_id").AttrType(REQUIRED).Int();
        this->Attr("num_experts").AttrType(REQUIRED).Int();
        this->Attr("num_max_tokens_per_rank").AttrType(REQUIRED).Int();
        this->Attr("ccl_buffer_size").AttrType(REQUIRED).Int();
        this->Attr("has_topk_weights").AttrType(OPTIONAL).Bool(true);
        this->Attr("topo_type").AttrType(OPTIONAL).Int(0);
        this->Attr("rank_num_per_server").AttrType(OPTIONAL).Int(1);

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
    }
};
OP_ADD(MoeEpCombineEpilogue);

} // namespace ops
