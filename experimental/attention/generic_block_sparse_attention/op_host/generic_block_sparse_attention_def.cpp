/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

// W8A8 pseudo-quantization only (arch22 / ascend910b):
// FP16/BF16 query + INT8 KV stored in the PA_NZ cache format.
class GenericBlockSparseAttention : public OpDef {
public:
    explicit GenericBlockSparseAttention(const char* name) : OpDef(name)
    {
        // dtype column mapping:
        //   col0: FP16 query + INT8 KV -> FP16 output (W8A8 antiquant)
        //   col1: BF16 query + INT8 KV -> BF16 output (W8A8 antiquant)
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            // INT8 KV is stored in the PA_NZ cache format.
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .IgnoreContiguous();
        this->Input("sparseBlockIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("sparseBlockCount")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("metaData")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("qDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("kDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("vDequantScale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Input("cuSeqLengthsQOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cuSeqLengthsKvOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sequsedQOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("sequsedKvOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("blockTableOptional")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        // W8A8 path requires blockShapeX=1; keep default aligned with supported config.
        this->Attr("blockShape").AttrType(OPTIONAL).ListInt({1, 128});
        // Kernel task decode and sparse layouts are packed-GQA only (task = T * Nkv).
        this->Attr("isPackedGQA").AttrType(OPTIONAL).Int(1);
        // W8A8 uses layoutQ="TND" and layoutKv="PAGED_NZ" with FRACTAL_NZ key/value.
        this->Attr("layoutQ").AttrType(OPTIONAL).String("TND");
        this->Attr("layoutKv").AttrType(OPTIONAL).String("PAGED_NZ");
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("maskType").AttrType(OPTIONAL).Int(1);
        this->Attr("softmaxPrecision").AttrType(OPTIONAL).Int(0);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GenericBlockSparseAttention);

}  // namespace ops
