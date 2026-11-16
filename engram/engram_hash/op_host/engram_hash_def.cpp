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
 * \file engram_hash_def.cpp
 * \brief EngramHash 算子信息库定义
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramHash : public OpDef {
public:
    explicit EngramHash(const char *name) : OpDef(name)
    {
        // N-gram token ids, shape 为 [num_tokens, max_ngram_size]
        this->Input("ngram_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 每层 hash 乘子, shape 为 [num_ngram_layers, max_ngram_size]
        this->Input("multipliers")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 每个 n-gram 长度每张 embedding 表的大小, shape 为 [num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram]
        this->Input("vocab_sizes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 每层 embedding 表偏移 (词表前缀和), shape 为 [num_ngram_layers, num_out_cols]
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // hash 后的 embedding 索引, shape 为 [num_ngram_layers, num_tokens, num_out_cols]
        this->Output("hash_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend960dt");
        this->AICore().AddConfig("ascend960pr");
    }
};
OP_ADD(EngramHash);
} // namespace ops
