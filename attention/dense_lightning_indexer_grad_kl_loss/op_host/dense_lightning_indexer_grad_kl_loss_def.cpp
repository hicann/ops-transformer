/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_grad_kl_loss_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> qkDtype = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16
};
static const std::vector<ge::DataType> weightDtype = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT
};

class DenseLightningIndexerGradKLLoss : public OpDef {
public:
    explicit DenseLightningIndexerGradKLLoss(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_index")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_index")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType(weightDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("softmax_max")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("softmax_sum")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("softmax_max_index")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("softmax_sum_index")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_rope")
            .ParamType(OPTIONAL)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_rope")
            .ParamType(OPTIONAL)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Output("d_query_index")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND});
        this->Output("d_key_index")
            .ParamType(REQUIRED)
            .DataType(qkDtype)
            .FormatList({ge::FORMAT_ND});
        this->Output("d_weight")
            .ParamType(REQUIRED)
            .DataType(weightDtype)
            .FormatList({ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        this->Attr("layout").AttrType(OPTIONAL).String("TND");         // BSND, TND
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);            // 3:默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(DenseLightningIndexerGradKLLoss);
} // namespace ops
