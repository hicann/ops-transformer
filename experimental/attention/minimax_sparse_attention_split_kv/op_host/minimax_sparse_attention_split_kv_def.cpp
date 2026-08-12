/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "register/op_def_registry.h"

namespace ops {

class MinimaxSparseAttentionSplitKv : public OpDef {
public:
    explicit MinimaxSparseAttentionSplitKv(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Input("blockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // k2qRowPtr: per-head CSR row pointers [kvHeads, totalKvRows+1], int32. A plain runtime
        // tensor Input (no const-fold) — the kernel reads csrStart/csrEnd from this GM tensor
        // (params.k2qRowPtr) via GetValue; host tiling derives totalKvRows from the Input SHAPE
        // only (no host value-read, so no .tolist / aclIntArray / ConvertToTensor front-end).
        this->Input("k2qRowPtr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("k2qQIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("k2qSlotIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Input("actualSeqLengthsKv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});

        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        this->Attr("blockSize").AttrType(OPTIONAL).Int(128);
        this->Attr("topK").AttrType(OPTIONAL).Int(8);
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(4);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(MinimaxSparseAttentionSplitKv);

}  // namespace ops
