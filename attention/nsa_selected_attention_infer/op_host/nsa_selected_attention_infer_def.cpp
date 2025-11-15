/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file nsa_selected_attention_infer_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
    class NsaSelectedAttentionInfer : public OpDef {
    public:
    explicit NsaSelectedAttentionInfer(const char *name) : OpDef(name)
        {
            this->Input("query")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("key")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("value")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("topk_indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("atten_mask")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_UINT8, ge::DT_UINT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("block_table")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("actual_q_seq_lengths")
                .ParamType(OPTIONAL)
                .ValueDepend(OPTIONAL)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("actual_kv_seq_lengths")
                .ParamType(OPTIONAL)
                .ValueDepend(OPTIONAL)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("attention_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("input_layout").AttrType(REQUIRED).String("BSND");
            this->Attr("num_heads").AttrType(REQUIRED).Int(1);
            this->Attr("num_key_value_heads").AttrType(REQUIRED).Int(1);
            this->Attr("selected_block_size").AttrType(REQUIRED).Int(1);
            this->Attr("selected_block_count").AttrType(REQUIRED).Int(1);
            this->Attr("block_size").AttrType(REQUIRED).Int(1);
            this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
            this->Attr("sparse_mode").AttrType(REQUIRED).Int(0);
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
        }
    };
    OP_ADD(NsaSelectedAttentionInfer);
    } // namespace ops