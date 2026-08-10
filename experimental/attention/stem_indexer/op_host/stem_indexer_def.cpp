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
 * \file stem_indexer_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class StemIndexer : public OpDef {
public:
    explicit StemIndexer(const char *name) : OpDef(name)
    {
        this->Input("qflat").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kflat").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vbias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("q_seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("kv_seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("num_prompt_tokens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_seq_len").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("causal").AttrType(OPTIONAL).Bool(true);
        this->Attr("stem_block_size").AttrType(OPTIONAL).Int(128);
        this->Attr("stem_stride").AttrType(OPTIONAL).Int(16);
        this->Attr("alpha").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("initial_blocks").AttrType(OPTIONAL).Int(4);
        this->Attr("window_size").AttrType(OPTIONAL).Int(4);
        this->Attr("k_block_num_rate_medium").AttrType(OPTIONAL).Float(0.2f);
        this->Attr("k_block_num_bias_medium").AttrType(OPTIONAL).Int(30);
        this->Attr("k_block_num_rate_large").AttrType(OPTIONAL).Float(0.1f);
        this->Attr("k_block_num_bias_large").AttrType(OPTIONAL).Int(30);
        this->Attr("topk_score_precision").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        // 当前仅适配 A5（ascend950），A2/A3 暂未适配，不注册
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(StemIndexer);
} // namespace ops
