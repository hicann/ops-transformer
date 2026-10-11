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
 * \file swin_transformer_ln_qkv.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class SwinTransformerLnQKV : public OpDef {
public:
    explicit SwinTransformerLnQKV(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("query_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("key_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("value_output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Attr("head_num")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("head_dim")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("seq_length")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("shifts")
            .AttrType(OPTIONAL)
            .ListInt({});
        this->Attr("epsilon")
            .AttrType(OPTIONAL)
            .Float(0.0000001f);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("kirinx90");
        this->AICore().AddConfig("kirin9030");
    }
};

OP_ADD(SwinTransformerLnQKV);
}
