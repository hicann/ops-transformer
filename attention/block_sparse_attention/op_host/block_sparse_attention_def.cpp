/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

class BlockSparseAttention : public OpDef {
public:
    explicit BlockSparseAttention(const char *name)
        : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1,
                       ge::DT_FLOAT4_E2M1, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .FormatList({ge::FORMAT_ND});
        this->Input("block_sparse_mask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Input("atten_mask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8}).FormatList({ge::FORMAT_ND});
        this->Input("block_shape").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("actual_seq_lengths").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv").ParamType(OPTIONAL).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Input("q_dequant_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND});
        this->Input("k_dequant_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND});
        this->Input("v_dequant_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND});
        this->Input("p_quant_scale").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse").ParamType(OPTIONAL).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        this->Attr("q_input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("kv_input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(1);
        this->Attr("mask_type").AttrType(OPTIONAL).Int(0);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(0.0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Float(0.0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(BlockSparseAttention);

} // namespace ops
