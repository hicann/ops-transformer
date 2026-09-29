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
 * \file lightning_indexer_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class LightningIndexerV2 : public OpDef {
public:
    explicit LightningIndexerV2(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        // candidate (two-level topk): source 模式输出候选块索引 (candidate_topk_blocks != -1 时生效)
        this->Output("candidate_topk_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // candidate_block_length: 预留接口, 恒输出空 tensor
        this->Output("candidate_block_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int(2048); // 筛选前2048个作为输出index
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0);
        // candidate (two-level topk): -1=关闭(现网行为), 有效值 (0,2048] 且 64 的倍数时开启 source 模式
        this->Attr("candidate_topk_blocks").AttrType(OPTIONAL).Int(-1);
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(8); // [2,64] 且为 2 的幂
        // 【R4 清理 2026-09-23】ascend950 config 随 arch35 kernel 删除一并移除：experimental 版
        // candidate（两级 TopK）仅支持 910B/910_93，950 的 LIV2 能力由顶层 attention/lightning_indexer_v2 承担
        OpAICoreConfig aicore_config_910b;
        aicore_config_910b.Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config_910b);
        this->AICore().AddConfig("ascend910_93", aicore_config_910b);
    }
};
OP_ADD(LightningIndexerV2);
} // namespace ops
