/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_def.cpp
 * \brief cloned from lightning_indexer_v2 @Phase1, consumer-mode specialization
 * (SLI consumer impl design §3.1)
 */
#include "register/op_def_registry.h"

namespace ops {
class SparseLightningIndexer : public OpDef {
public:
    explicit SparseLightningIndexer(const char *name) : OpDef(name)
    {
        // Inputs 0-10 与 LightningIndexerV2 逐字相同
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
        // candidate (two-level topk) consumer：候选块索引输入（source 算子输出直连；
        // 值域 [0,numBlocks) 或 -1，槽位无序，允许重复/越界/-1 脏数据，跨算子契约见 README）
        this->Input("candidate_topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // candidate_block_length：预留接口，仅接受空 tensor（N4/O13，本期不消费）
        this->Input("candidate_block_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // Outputs：sparse_indices / sparse_values 与 LIV2 相同（本算子无 candidate 输出）
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        // Attrs 0-6 与 LIV2 相同；本算子无 candidate_topk_blocks 属性（candBlocks 由输入
        // shape[-1] 推导，N2），candidate_block_size 为 attr index 7
        this->Attr("topk").AttrType(REQUIRED).Int(2048); // 筛选前2048个作为输出index
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0); // 恒 0（C6，leak 降级污染泄漏槽 value）
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(8); // [2,64] 且为 2 的幂
        // 仅注册 ascend910b / ascend910_93（N6：无 ascend950 config，框架级不支持）
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        aicoreConfig.Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(SparseLightningIndexer);
} // namespace ops
