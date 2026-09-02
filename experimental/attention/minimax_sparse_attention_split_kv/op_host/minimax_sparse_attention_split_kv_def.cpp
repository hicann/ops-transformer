/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

// 算子定义类：描述算子在 GE 图中的输入、输出与属性规格。
class MinimaxSparseAttentionSplitKv : public OpDef {
public:
    explicit MinimaxSparseAttentionSplitKv(const char *name)
        : OpDef(name)
    {
        // query: [total_q_tokens, num_q_heads, D], fp16/bf16/fp8, TND 布局的 Q 矩阵
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // key: [num_physical_blocks, blockSize, kvHeads, D], fp16/bf16/fp8, Paged K Cache（分页存储的 K 缓存）
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // value: [num_physical_blocks, blockSize, kvHeads, D], fp16/bf16/fp8, Paged V Cache（分页存储的 V 缓存）
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // Optional: present => paged KV cache (4D key/value + block table).
        // Absent  => contiguous dense K/V matching query + inputLayout
        // (TND / BNSD / BSND).
        this->Input("blockTable")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // k2qRowPtr: per-head CSR row pointers [kvHeads, totalKvRows+1], int32. A plain runtime
        // tensor Input (no const-fold) — the kernel reads csrStart/csrEnd from this GM tensor
        // (params.k2qRowPtr) via GetValue; host tiling derives totalKvRows from the Input SHAPE
        // only (no host value-read, so no .tolist / aclIntArray / ConvertToTensor front-end).
        this->Input("k2qRowPtr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // k2qQIndices: CSR 格式的列数据，全局 Q token id 列表
        this->Input("k2qQIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // k2qSlotIndices: workspace 中的槽位索引，范围 [0, topK)
        this->Input("k2qSlotIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // actualSeqLengths: 每个 batch 的实际 Q 序列长度
        this->Input("actualSeqLengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // actualSeqLengthsKv: 每个 batch 的实际 KV 序列长度
        this->Input("actualSeqLengthsKv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // attentionOut: [total_q_tokens, num_q_heads, D], 与 query 同 dtype 的注意力计算输出
        this->Output("attentionOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
        // softmaxLse: [total_q_tokens, num_q_heads, 1] fp32 when enabled,
        // otherwise an empty [0] tensor. Required IR output (same as FusedInferAttentionScore).
        this->Output("softmaxLse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});

        // numKeyValueHeads: KV 头数，默认值 1
        this->Attr("numKeyValueHeads").AttrType(OPTIONAL).Int(1);
        // scaleValue: 缩放因子，默认值 0.0（表示使用 1/sqrt(D)）
        this->Attr("scaleValue").AttrType(OPTIONAL).Float(0.0);
        // blockSize: 分页块大小，默认值 128
        this->Attr("blockSize").AttrType(OPTIONAL).Int(128);
        // topK: 稀疏注意力选取的 top-K 数量，默认值 8
        this->Attr("topK").AttrType(OPTIONAL).Int(8);
        // innerPrecise: 内部精度模式，默认值 4
        // 0: fp32 softmax + fp32 O_partial; 1: bf16 softmax + bf16 O_partial;
        // 4 (default): bf16 softmax + fp32 O_partial.
        this->Attr("innerPrecise").AttrType(OPTIONAL).Int(4);
        // softmaxLseFlag: 是否返回 log-sum-exp，默认关闭以保持原路径零开销。
        this->Attr("softmaxLseFlag").AttrType(OPTIONAL).Bool(false);
        // Q/K/V layout: "TND" [T,N,D], "BNSD" [B,N,S,D], "BSND" [B,S,N,D].
        // Rank must match. Paged KV cache requires TND query.
        this->Attr("inputLayout").AttrType(OPTIONAL).String("TND");

        // 声明算子支持的 AICore 型号配置。
        // ascend950: Atlas A5 (arch35, __CCE_AICORE__==310)
        // ascend910b: Atlas A2 (DAV_2201, __CCE_AICORE__==220)
        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

// 将算子定义注册到算子注册表，使其可被 GE 识别与调用。
OP_ADD(MinimaxSparseAttentionSplitKv);

} // namespace ops
