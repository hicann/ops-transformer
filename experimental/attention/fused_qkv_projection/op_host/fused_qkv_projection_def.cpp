/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "register/op_def_registry.h"

namespace {
const std::vector<ge::DataType> kFusedQkvDataTypes = {ge::DT_FLOAT, ge::DT_FLOAT16};
const std::vector<ge::Format> kFusedQkvFormats = {ge::FORMAT_ND, ge::FORMAT_ND};
} // namespace

namespace ops {

class FusedQkvProjection : public OpDef {
public:
    explicit FusedQkvProjection(const char *name)
        : OpDef(name)
    {
        this->Input("hidden_states")
            .ParamType(REQUIRED)
            .DataType(kFusedQkvDataTypes)
            .Format(kFusedQkvFormats)
            .AutoContiguous();

        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType(kFusedQkvDataTypes)
            .Format(kFusedQkvFormats)
            .AutoContiguous();

        this->Input("bias").ParamType(OPTIONAL).DataType(kFusedQkvDataTypes).Format(kFusedQkvFormats).AutoContiguous();

        this->Output("query")
            .ParamType(REQUIRED)
            .DataType(kFusedQkvDataTypes)
            .Format(kFusedQkvFormats)
            .AutoContiguous();

        this->Output("key").ParamType(REQUIRED).DataType(kFusedQkvDataTypes).Format(kFusedQkvFormats).AutoContiguous();

        this->Output("value")
            .ParamType(REQUIRED)
            .DataType(kFusedQkvDataTypes)
            .Format(kFusedQkvFormats)
            .AutoContiguous();

        this->Attr("q_output_dim").AttrType(REQUIRED).Int();
        this->Attr("k_output_dim").AttrType(REQUIRED).Int();
        this->Attr("v_output_dim").AttrType(REQUIRED).Int();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true);
        aicoreConfig.DynamicFormatFlag(false);
        aicoreConfig.DynamicRankSupportFlag(true);
        aicoreConfig.DynamicShapeSupportFlag(true);
        aicoreConfig.NeedCheckSupportFlag(false);
        aicoreConfig.PrecisionReduceFlag(true);
        aicoreConfig.ExtendCfgInfo("opFile.value", "fused_qkv_projection");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(FusedQkvProjection);
} // namespace ops
