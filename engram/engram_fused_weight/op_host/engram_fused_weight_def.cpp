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
 * \file engram_fused_weight_def.cpp
 * \brief EngramFusedWeight 算子信息库定义
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramFusedWeight : public OpDef {
public:
    explicit EngramFusedWeight(const char *name) : OpDef(name)
    {
        // Engram 门控的两个 per-head 加权权重 (q_weight / k_weight), shape 为 [hcMult, hiddenSize]
        this->Input("weight_hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight_embed")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 融合权重: fp32(weight_hidden) * fp32(weight_embed), 与输入同 shape
        this->Output("weight_fused")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend960dt");
        this->AICore().AddConfig("ascend960pr");
    }
};
OP_ADD(EngramFusedWeight);
} // namespace ops
