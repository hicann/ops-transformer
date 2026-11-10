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
#include <gtest/gtest.h>
#include "../../../op_host/partial_rotary_mul_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class PartialRotaryMulQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PartialRotaryMulQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PartialRotaryMulQuantTiling TearDown" << std::endl;
    }
};

struct PartialRotaryMulQuantCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
    int64_t l2Size;
};

// ==================== 950 HiFloat8 用例 ====================

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp16_hif8_950_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp32_hif8_950_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_bf16_hif8_950_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{2, 256, 8, 64}, {2, 256, 8, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 256, 8, 32}, {2, 256, 8, 32}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

// ==================== 960PR HiFloat8 + HiFloat4 用例 ====================

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp16_hif8_960pr_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960PR");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp16_hif4_960pr_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // y_dtype=1 仅 960 系列（含 960PR）支持 HiFloat4
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend960PR");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp32_hif8_960pr_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960PR");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

// ==================== 950 边界/异常用例 ====================

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_invalid_dtype_950_failed)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  // int32 不支持
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend950");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ==================== 960DT HiFloat8 + HiFloat4 用例 ====================

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp16_hif8_960dt_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960DT");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp16_hif4_960dt_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 128, 4, 128}, {1, 128, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 128, 4, 64}, {1, 128, 4, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  // y_dtype=1 仅 960 系列（含 960DT）支持 HiFloat4
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend960DT");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}

TEST_F(PartialRotaryMulQuantTiling, test_partial_rotary_mul_quant_fp32_hif8_960dt_success)
{
    PartialRotaryMulQuantCompileInfo compileInfo = {40, 196608, 33554432};
    gert::TilingContextPara tilingContextPara("PartialRotaryMulQuant",
                                              {
                                                  {{{1, 64, 2, 64}, {1, 64, 2, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 64, 2, 32}, {1, 64, 2, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                              },
                                              &compileInfo, "Ascend960DT");
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS);
}
