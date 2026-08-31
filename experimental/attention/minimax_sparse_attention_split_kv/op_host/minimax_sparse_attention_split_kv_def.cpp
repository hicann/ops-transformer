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

class MinimaxSparseAttentionSplitKv : public OpDef {
public:
    explicit MinimaxSparseAttentionSplitKv(const char *name)
        : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // Optional: present => paged KV cache (4D key/value + block table).
        // Absent  => contiguous dense K/V matching query + inputLayout
        // (TND / BNSD / BSND).
        this->Input("blockTable")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // k2qRowPtr: per-head CSR row pointers [kvHeads, totalKvRows+1], int32. A plain runtime
        // tensor Input (no const-fold) — the kernel reads csrStart/csrEnd from this GM tensor
        // (params.k2qRowPtr) via GetValue; host tiling derives totalKvRows from the Input SHAPE
        // only (no host value-read, so no .tolist / aclIntArray / ConvertToTensor front-end).
        this->Input("k2qRowPtr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k2qQIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k2qSlotIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actualSeqLengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actualSeqLengthsKv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // Same as FusedInferAttentionScore: required IR output, fp32.
        // When softmaxLseFlag is false, infershape is [0] and kernel skips the write.
        this->Output("softmaxLse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("blockSize").AttrType(OPTIONAL).Int(128);
        this->Attr("topK").AttrType(OPTIONAL).Int(8);
        // 0: fp32 softmax + fp32 O_partial; 1: bf16 softmax + bf16 O_partial;
        // 4 (default): bf16 softmax + fp32 O_partial.
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(4);
        this->Attr("softmaxLseFlag").AttrType(OPTIONAL).Bool(false);
        // Q/K/V layout: "TND" [T,N,D], "BNSD" [B,N,S,D], "BSND" [B,S,N,D].
        // Rank must match. Paged KV cache requires TND query.
        this->Attr("inputLayout").AttrType(OPTIONAL).String("TND");

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(MinimaxSparseAttentionSplitKv);

} // namespace ops
