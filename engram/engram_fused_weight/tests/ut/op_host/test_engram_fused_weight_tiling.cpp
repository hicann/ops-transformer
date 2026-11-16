/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fused_weight_tiling.cpp
 * \brief EngramFusedWeight tiling UT
 *
 * 覆盖与 tilelang 输入范围契约对应的场景:
 *   - 成功: 常规 [4, 5120] / 最小 [2, 128] / 空输入 [0, 512] / 大规模多块;
 *   - 失败: 总元素数不被 256 整除 / 非 2D / 两输入 shape 不一致 / 输入 dtype 非 BF16 / 输出 dtype 非 FP32。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

class EngramFusedWeightTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFusedWeightTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFusedWeightTiling TearDown" << std::endl;
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

static gert::TilingContextPara MakePara(const std::vector<int64_t>& whShape, const std::vector<int64_t>& weShape,
                                        const std::vector<int64_t>& outShape, ge::DataType whDtype,
                                        ge::DataType weDtype, ge::DataType outDtype)
{
    // builder 要求 compileInfo 非空才能构造 TilingContext (与 add_example UT 一致);
    // tiling 从 PlatformInfo 取核数/UB, 该占位对象不会被解引用。
    struct EngramFwUtCompileInfo {
    } compileInfo;
    return gert::TilingContextPara("EngramFusedWeight",
                                   {
                                       TensorDesc(whShape, whDtype),
                                       TensorDesc(weShape, weDtype),
                                   },
                                   {
                                       TensorDesc(outShape, outDtype),
                                   },
                                   {}, &compileInfo, "Ascend950");
}

// 模型默认形状 (DeepSeek-V4.1 engram: hcMult=4, hiddenSize=5120)。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_model_shape)
{
    auto para = MakePara({4, 5120}, {4, 5120}, {4, 5120}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL); // 单一 tiling key 0
}

// 最小合法输入: 总元素数 = 2 * 128 = 256。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_min_total)
{
    auto para = MakePara({2, 128}, {2, 128}, {2, 128}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 空输入: 总元素数 = 0, 与 tilelang 一致允许 (kernel 直接返回)。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_empty)
{
    auto para = MakePara({0, 512}, {0, 512}, {0, 512}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 大规模输入: 总元素数 = 8 * 8192 = 65536, 触发较大的 blockSize 与持久核路径。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_large)
{
    auto para = MakePara({8, 8192}, {8, 8192}, {8, 8192}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0UL);
}

// 失败: 总元素数 3 * 100 = 300, 不被 256 整除。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_bad_align_fail)
{
    auto para = MakePara({3, 100}, {3, 100}, {3, 100}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 非 2D 输入 (1D)。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_1d_fail)
{
    auto para = MakePara({512}, {512}, {512}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 非 2D 输入 (3D)。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_3d_fail)
{
    auto para = MakePara({2, 4, 512}, {2, 4, 512}, {2, 4, 512}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 两输入 shape 不一致。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_shape_mismatch_fail)
{
    auto para = MakePara({4, 5120}, {4, 4096}, {4, 5120}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 输出 shape 与输入不一致。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_out_shape_mismatch_fail)
{
    auto para = MakePara({4, 5120}, {4, 5120}, {4, 4096}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: weight_hidden dtype 为 FP16。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_bad_input_dtype_fail)
{
    auto para = MakePara({4, 5120}, {4, 5120}, {4, 5120}, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}

// 失败: 输出 dtype 非 FP32。
TEST_F(EngramFusedWeightTiling, engram_fused_weight_tiling_bad_output_dtype_fail)
{
    auto para = MakePara({4, 5120}, {4, 5120}, {4, 5120}, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
    ExecuteTestCase(para, ge::GRAPH_FAILED, std::numeric_limits<uint64_t>::max());
}
