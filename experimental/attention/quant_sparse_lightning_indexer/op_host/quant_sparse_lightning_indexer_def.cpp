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
 * \file quant_sparse_lightning_indexer_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class QuantSparseLightningIndexer : public OpDef {
public:
    explicit QuantSparseLightningIndexer(const char *name) : OpDef(name)
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
        this->Input("candidate_topk_index")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // candidate_block_length：预留接口，仅接受空 tensor（N4/O13，本期不消费）
        this->Input("candidate_block_length")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("sparse_values").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int(2048);       // 2048: 筛选前2048个作为输出index
        this->Attr("quant_mode").AttrType(REQUIRED).Int(1);    // def 默认值(主算子语义); 本算子仅支持 2(INT8), tiling 拒绝其他取值
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1); // -1: 默认值，表示任意可能长度
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0); // 0: 默认值，无mask
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int(1);
        this->Attr("return_value").AttrType(OPTIONAL).Int(0); //  0: 默认值
        // candidate_topk_blocks 不再是入参: 候选块个数 = 输入 candidate_topk_index 的末维
        // (tiling 从输入形状推导并校验, 当前仅支持 2048)
        this->Attr("candidate_block_size").AttrType(OPTIONAL).Int(8);        // 候选块大小(位置数), 须 > 0, 当前仅支持 8
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
OP_ADD(QuantSparseLightningIndexer);
} // namespace ops
