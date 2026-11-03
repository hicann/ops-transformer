/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class KvCacheTurboQuant : public OpDef {
public:
    explicit KvCacheTurboQuant(const char* name) : OpDef(name)
    {
        Input("kv_vectors").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        Input("rotation_matrix")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        Input("qjl_matrix")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        Output("quant_idx").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        Output("quant_qjl").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        Output("quant_norm")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        Output("quant_gamma")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        Attr("mse_bits").Int(3);
        Attr("output_dtype").AttrType(OPTIONAL).Int(0);
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(KvCacheTurboQuant);
} // namespace ops
