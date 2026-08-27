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
 * \file fused_causal_conv1d_def.cpp
 * \brief FusedCausalConv1d ascendc impl
 */

#include "register/op_def_registry.h"

namespace ops {
class FusedCausalConv1d : public OpDef {
public:
    explicit FusedCausalConv1d(const char *name)
        : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("conv_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("query_start_loc").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("cache_indices").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("initial_state_mode").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("bias").ParamType(OPTIONAL).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Input("num_accepted_tokens").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("num_computed_tokens").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("block_idx_first_scheduled_token")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_idx_last_scheduled_token")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("initial_state_idx").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("conv_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Attr("activation_mode").AttrType(REQUIRED).Int(); // 0: None  1: silu  2: swish
        this->Attr("pad_slot_id").AttrType(REQUIRED).Int();     // skip invalid batch
        this->Attr("run_mode").AttrType(REQUIRED).Int();        // 0: prefill   1: decode
        this->Attr("max_query_len").AttrType(REQUIRED).Int();
        this->Attr("residual_connection").AttrType(REQUIRED).Int(); // 0: no residual  1: with residual
        this->Attr("block_size").AttrType(REQUIRED).Int();          // when APC enabled, it can not be set to 0
        this->Attr("conv_mode").AttrType(REQUIRED).Int();           // 0: for Qwen and Pangu7B  1: for Pangu
        this->Attr("max_draft_tokens").AttrType(OPTIONAL).Int(7);
        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "fused_causal_conv1d_apt");
        this->AICore().AddConfig("ascend950", config_950);
    }
};

OP_ADD(FusedCausalConv1d);
} // namespace ops
