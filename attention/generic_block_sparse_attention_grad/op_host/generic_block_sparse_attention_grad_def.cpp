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
 * \file generic_block_sparse_attention_grad_def.cpp
 * \brief GE OpDef for GenericBlockSparseAttentionGrad (IR aligned with design / xlsx).
 */

#include "register/op_def_registry.h"

namespace ops {

class GenericBlockSparseAttentionGrad : public OpDef {
public:
    explicit GenericBlockSparseAttentionGrad(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("rsvd_block_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("rsvd_block_count")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("metadata")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BOOL, ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND});
        this->Input("cu_seq_lengths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("cu_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("seqused_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Output("dQuery")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("dKey")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("dValue")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("block_shape").AttrType(OPTIONAL).ListInt({1, 128});
        this->Attr("is_packed_gqa").AttrType(OPTIONAL).Int(1);
        this->Attr("q_input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("kv_input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("mask_type").AttrType(OPTIONAL).Int(1);
        this->Attr("softmax_precision").AttrType(OPTIONAL).Int(0);
        this->Attr("window_size_left").AttrType(OPTIONAL).Int(-1);
        this->Attr("window_size_right").AttrType(OPTIONAL).Int(-1);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(GenericBlockSparseAttentionGrad);

} // namespace ops
