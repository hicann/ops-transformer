/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class SparseFlashMlaGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseFlashMlaGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseFlashMlaGradTiling TearDown" << std::endl;
    }
};

namespace {
struct SparseFlashMlaGradCompileInfo {};

using TensorPara = gert::TilingContextPara::TensorDescription;
using AttrPara = gert::TilingContextPara::OpAttr;

// InputIndex: query,d_out,out,lse,ori_kv,cmp_kv,ori_sparse_indices,cmp_sparse_indices,
// cu_seqlens_*,seqused_*,cmp_residual_kv,ori/cmp_topk_length,sinks,metadata
std::vector<TensorPara> MakeBsndSwaInputs(int64_t s1, int64_t s2, ge::DataType dtype = ge::DT_FLOAT16)
{
    const int64_t b = 1;
    const int64_t n1 = 64;
    const int64_t n2 = 1;
    const int64_t d = 512;
    TensorPara emptyI32{{{}, {}}, ge::DT_INT32, ge::FORMAT_ND};
    TensorPara emptyFp{{{}, {}}, dtype, ge::FORMAT_ND};
    return {
        TensorPara{{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},          // query
        TensorPara{{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},          // d_out
        TensorPara{{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND},          // out
        TensorPara{{{b, n2, s1, n1}, {b, n2, s1, n1}}, ge::DT_FLOAT, ge::FORMAT_ND}, // lse
        TensorPara{{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND},          // ori_kv
        emptyFp,                                                                     // cmp_kv
        emptyI32,                                                                    // ori_sparse_indices
        emptyI32,                                                                    // cmp_sparse_indices
        emptyI32,                                                                    // cu_seqlens_q
        emptyI32,                                                                    // cu_seqlens_ori_kv
        emptyI32,                                                                    // cu_seqlens_cmp_kv
        emptyI32,                                                                    // seqused_q
        emptyI32,                                                                    // seqused_ori_kv
        emptyI32,                                                                    // seqused_cmp_kv
        emptyFp,                                                                     // cmp_residual_kv
        emptyI32,                                                                    // ori_topk_length
        emptyI32,                                                                    // cmp_topk_length
        emptyFp,                                                                     // sinks
        emptyI32,                                                                    // metadata
    };
}

std::vector<TensorPara> MakeBsndOutputs(int64_t s1, int64_t s2, ge::DataType dtype = ge::DT_FLOAT16)
{
    const int64_t b = 1;
    const int64_t n1 = 64;
    const int64_t n2 = 1;
    const int64_t d = 512;
    TensorPara emptyFp{{{}, {}}, dtype, ge::FORMAT_ND};
    TensorPara emptyF32{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND};
    return {
        TensorPara{{{b, s1, n1, d}, {b, s1, n1, d}}, dtype, ge::FORMAT_ND}, // d_query
        TensorPara{{{b, s2, n2, d}, {b, s2, n2, d}}, dtype, ge::FORMAT_ND}, // d_ori_kv
        emptyFp,                                                            // d_cmp_kv
        emptyF32,                                                           // d_sinks
        emptyF32,                                                           // ori_softmax_l1_norm
        emptyF32,                                                           // cmp_softmax_l1_norm
    };
}

std::vector<AttrPara> MakeBsndSwaAttrs()
{
    return {
        AttrPara{"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.0441941738f)},
        AttrPara{"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        AttrPara{"ori_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        AttrPara{"cmp_mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        AttrPara{"ori_win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(127)},
        AttrPara{"ori_win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        AttrPara{"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
        AttrPara{"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
    };
}
} // namespace

// =================== E1: BSND S1=0 (负例) ===================
TEST_F(SparseFlashMlaGradTiling, SparseFlashMlaGrad_910b_tiling_E1_bsnd_s1_zero)
{
    SparseFlashMlaGradCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("SparseFlashMlaGrad", MakeBsndSwaInputs(0, 128), MakeBsndOutputs(0, 128),
                                              MakeBsndSwaAttrs(), &compileInfo, "Ascend910B", 20, 196608, 16384);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
