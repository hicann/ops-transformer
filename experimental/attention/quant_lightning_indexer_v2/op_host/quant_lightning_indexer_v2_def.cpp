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
 * \file quant_lightning_indexer_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantLightningIndexerV2 : public OpDef {
public:
    explicit QuantLightningIndexerV2(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_INT8}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("q_descale")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cmp_residual_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("output_idx_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_values").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("candidate_topk_index_out").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        // candidate_block_length: 预留接口, 恒输出空 tensor
        this->Output("candidate_block_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int(2048);       // 2048: 筛选前2048个作为输出index
        this->Attr("quant_mode").AttrType(REQUIRED).Int(1);    // 1: per-token-head
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1); // -1: 默认值，表示任意可能长度
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0); // 0: 默认值，无mask
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0); //  0: 默认值
        // candidate (两级TopK 第一级): 开关由 candidate_topk_blocks 承载, 无独立 mode 属性 ——
        // -1 = 关闭(现网行为, 默认); 2048 = 开启 source 模式 (当前仅支持这两个取值)
        this->Attr("candidate_topk_blocks").AttrType(OPTIONAL).Int(-1);
        // candidate_block_size: candidate_topk_blocks=-1(关闭) 时必须为 -1; 开启时必须 > 0
        // (当前仅支持 8: 块内归约 BlockReduceMax 以 32B 块即 8 个 fp32 为粒度)
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(-1);
        // 0 轴非连续支持不注册属性: stride 由输入描述符经 GetDynamicInputStride 进入 tiling
        // (对齐上游 PR 12740 / 稳定算子 9.2.0 机制), kernel 侧 keyStride0==0 时兜底紧凑公式
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

        // arch35 (Ascend 950) 配置已移除: 本算子仅支持 arch22 (910B/910_93)
    }
};
OP_ADD(QuantLightningIndexerV2);
} // namespace ops
