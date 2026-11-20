/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_gate_def.cpp
 * \brief EngramGate 算子信息库定义 (前向)
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramGate : public OpDef {
public:
    explicit EngramGate(const char* name)
        : OpDef(name)
    {
        ConfigureInputs();
        ConfigureOutputs();
        ConfigureExecution();
    }

private:
    void ConfigureInputs()
    {
        // 残差流输入: [num_tokens, hc_mult, hidden_size]
        this->Input("hidden_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // kv 融合输入: [num_tokens, (hc_mult + 1) * hidden_size], 前 hc_mult 份 k + 1 份共享 v
        this->Input("kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 融合权重: fp32(weight_hidden) * fp32(weight_embed), [hc_mult, hidden_size]
        this->Input("weight_fused")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
    }

    void ConfigureOutputs()
    {
        // 门控输出: cast(hidden_states + gate * v)，与 hidden_states 的 shape 和数据类型相同。
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // 反向中间结果（每个 (token, head) 对应一个 FP32 标量）：raw_dot、门控值和两路 RMS 倒数。
        // 声明为 OPTIONAL：save_for_backward=false 时允许 aclnn 传 nullptr。
        this->Output("dot")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gate_score")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd_x")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("rstd_k")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void ConfigureExecution()
    {
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("clamp_value").AttrType(OPTIONAL).Float(1e-6);
        // 是否写出反向中间结果：false 时 4 个中间结果输出为空 Tensor，kernel 跳过写出。
        this->Attr("save_for_backward").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "engram_gate");
        this->AICore().AddConfig("ascend950", regbaseCfg);
        this->AICore().AddConfig("ascend960dt", regbaseCfg);
        this->AICore().AddConfig("ascend960pr", regbaseCfg);
    }
};
OP_ADD(EngramGate);
} // namespace ops
