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

class BlockAttnResUpdate : public OpDef {
public:
    explicit BlockAttnResUpdate(const char *name)
        : OpDef(name)
    {
        this->Input("partial_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("delta").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("pseudo_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("numerator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("logit_max").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("exp_sum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        // The same input/output name declares partial_block as a reference tensor for ACLNNGraph.
        this->Output("partial_block")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("h").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        // eps is serialized into BlockAttnResUpdateTilingData; it is not a kernel tensor argument.
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-6F);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "block_attn_res_update");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BlockAttnResUpdate);
} // namespace ops
