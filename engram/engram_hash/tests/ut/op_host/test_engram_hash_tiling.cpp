/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_hash_tiling.cpp
 * \brief EngramHash tiling UT
 *
 * 覆盖与 tilelang 输入范围契约对应的场景:
 *   - 成功: 模型默认 (T=4096,N=3,L=2,tables=8) / 小规模 / 单 token / N=4 (C=24) / N=2,L=1 / N=7,L=5 / 空输入 T=0;
 *   - 失败: ids 非 2D / N < 2 / vocab 非 3D / vocab.dim1 != N-1 / L == 0 / tables == 0 / mult shape 不一致 /
 *     offsets shape 不一致 / 输出 shape 不一致 / 1024 线程约束违反 / L 超过核数 / dtype 不匹配。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

class EngramHashTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramHashTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramHashTiling TearDown" << std::endl;
    }
};

// gert::StorageShape 只有 initializer_list 构造, 不能从 std::vector<int64_t> 隐式构造,
// 需通过 SetDim 逐维填充。
static gert::TilingContextPara::TensorDescription TensorDesc(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    gert::Shape s;
    s.SetDimNum(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        s.SetDim(i, shape[i]);
    }
    gert::StorageShape ss;
    ss.MutableOriginShape() = s;
    ss.MutableStorageShape() = s;
    return gert::TilingContextPara::TensorDescription(ss, dtype, ge::FORMAT_ND);
}

static gert::TilingContextPara MakePara(const std::vector<int64_t>& idsShape, const std::vector<int64_t>& multShape,
                                        const std::vector<int64_t>& vocabShape, const std::vector<int64_t>& offsShape,
                                        const std::vector<int64_t>& outShape, ge::DataType idsDtype,
                                        ge::DataType multDtype, ge::DataType vocabDtype, ge::DataType offsDtype,
                                        ge::DataType outDtype)
{
    // builder 要求 compileInfo 非空才能构造 TilingContext (与 add_example UT 一致);
    // tiling 从 PlatformInfo 取核数/UB, 该占位对象不会被解引用。
    struct EngramHashUtCompileInfo {
    } compileInfo;
    return gert::TilingContextPara("EngramHash",
                                   {
                                       TensorDesc(idsShape, idsDtype),
                                       TensorDesc(multShape, multDtype),
                                       TensorDesc(vocabShape, vocabDtype),
                                       TensorDesc(offsShape, offsDtype),
                                   },
                                   {
                                       TensorDesc(outShape, outDtype),
                                   },
                                   {}, &compileInfo, "Ascend950");
}

// 模型默认形状 (DeepSeek-V4.1 engram: T=4096, N=3, L=2, tables=8, C=16)。
TEST_F(EngramHashTiling, engram_hash_tiling_model_shape)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL); // 单一 tiling key 0
}

// 小规模: 尾部 tile 不满。
TEST_F(EngramHashTiling, engram_hash_tiling_small_tokens)
{
    auto para = MakePara({42, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 42, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 单 token。
TEST_F(EngramHashTiling, engram_hash_tiling_single_token)
{
    auto para = MakePara({1, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 1, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// N=4, tables=8 即 C=24 的模型 shape。
TEST_F(EngramHashTiling, engram_hash_tiling_ngram4_c24)
{
    auto para = MakePara({2048, 4}, {2, 4}, {2, 3, 8}, {2, 24}, {2, 2048, 24}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// N=2, L=1, tables=16。
TEST_F(EngramHashTiling, engram_hash_tiling_ngram2_l1)
{
    auto para = MakePara({128, 2}, {1, 2}, {1, 1, 16}, {1, 16}, {1, 128, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// N=7, L=5, tables=3 (C=18)。
TEST_F(EngramHashTiling, engram_hash_tiling_ngram7_l5)
{
    auto para = MakePara({100, 7}, {5, 7}, {5, 6, 3}, {5, 18}, {5, 100, 18}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 空输入: num_tokens = 0, 与 tilelang 一致允许 (kernel 直接返回)。
TEST_F(EngramHashTiling, engram_hash_tiling_empty_tokens)
{
    auto para = MakePara({0, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 0, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 失败: ids 非 2D (1D)。
TEST_F(EngramHashTiling, engram_hash_tiling_ids_1d_fail)
{
    auto para = MakePara({4096}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: ids 非 2D (3D)。
TEST_F(EngramHashTiling, engram_hash_tiling_ids_3d_fail)
{
    auto para = MakePara({2, 4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64,
                         ge::DT_INT32, ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: max_ngram_size = 1 < 2。
TEST_F(EngramHashTiling, engram_hash_tiling_ngram_too_small_fail)
{
    auto para = MakePara({4096, 1}, {2, 1}, {2, 0, 8}, {2, 0}, {2, 4096, 0}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: vocab 非 3D (2D)。
TEST_F(EngramHashTiling, engram_hash_tiling_vocab_2d_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: vocab.dim1 != N - 1。
TEST_F(EngramHashTiling, engram_hash_tiling_vocab_dim1_mismatch_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 3, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: L = 0。
TEST_F(EngramHashTiling, engram_hash_tiling_zero_layers_fail)
{
    auto para = MakePara({4096, 3}, {0, 3}, {0, 2, 8}, {0, 16}, {0, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: tables = 0。
TEST_F(EngramHashTiling, engram_hash_tiling_zero_tables_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 0}, {2, 0}, {2, 4096, 0}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: multipliers shape 与 (L, N) 不一致。
TEST_F(EngramHashTiling, engram_hash_tiling_mult_shape_mismatch_fail)
{
    auto para = MakePara({4096, 3}, {2, 4}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: offsets shape 与 (L, C) 不一致。
TEST_F(EngramHashTiling, engram_hash_tiling_offsets_shape_mismatch_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 8}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 输出 shape 与 [L, T, C] 不一致。
TEST_F(EngramHashTiling, engram_hash_tiling_out_shape_mismatch_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 8}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 1024 线程约束违反 (N=3, tables=200 -> C=400, 1024//400=2 < lcm(8,8)=8)。
// 正例: C=400 (原受 1024 线程约束限制, 现已放开, VF 内 stride 循环覆盖)。
TEST_F(EngramHashTiling, engram_hash_tiling_wide_out_cols)
{
    auto para = MakePara({4, 3}, {2, 3}, {2, 2, 200}, {2, 400}, {2, 4, 400}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 正例: L=100 超过 AIV 核数 (原受核数限制, 现已放开, block 平坦映射 + 持久核循环)。
TEST_F(EngramHashTiling, engram_hash_tiling_many_layers)
{
    auto para = MakePara({4, 3}, {100, 3}, {100, 2, 8}, {100, 16}, {100, 4, 16}, ge::DT_INT32, ge::DT_INT64,
                         ge::DT_INT32, ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 失败: ngram_token_ids dtype 为 INT64。
TEST_F(EngramHashTiling, engram_hash_tiling_bad_ids_dtype_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: multipliers dtype 为 INT32。
TEST_F(EngramHashTiling, engram_hash_tiling_bad_mult_dtype_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT32);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 输出 dtype 非 INT32。
TEST_F(EngramHashTiling, engram_hash_tiling_bad_output_dtype_fail)
{
    auto para = MakePara({4096, 3}, {2, 3}, {2, 2, 8}, {2, 16}, {2, 4096, 16}, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                         ge::DT_INT32, ge::DT_INT64);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
