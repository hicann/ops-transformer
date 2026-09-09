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
 * \file test_quant_reduce_scatter_tiling.cpp
 * \brief host侧tiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace QuantReduceScatterUT {

const std::string OP_NAME = "QuantReduceScatter";

struct QuantReduceScatterTestParam {
    std::string caseName;
    // x
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    // scales
    std::initializer_list<int64_t> scalesShape;
    ge::DataType scalesDtype;
    ge::Format scalesFormat;
    // output
    std::initializer_list<int64_t> outputShape;
    ge::DataType outputDtype;
    ge::Format outputFormat;
    // attrs
    int64_t hcclBufferSizeAttr;
    std::string reduceOpAttr;
    int64_t outputDtypeAttr;
    // rank size
    uint64_t rankNum;
    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream &operator<<(std::ostream &os, const QuantReduceScatterTestParam &param)
{
    return os << param.caseName;
}

// 构造ut用例：这里可以按照正常用例/异常用例分开声明
static QuantReduceScatterTestParam g_testCases[] = {
    {"quant_reduce_scatter_abuse_case_tg_10",
     {1024, 5120},
     ge::DT_FLOAT8_E4M3FN,
     ge::FORMAT_ND,
     {1024, 80, 2},
     ge::DT_FLOAT8_E8M0,
     ge::FORMAT_ND,
     {128, 5120},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     314572800,
     "sum",
     ge::DT_FLOAT16,
     8,
     "Ascend910_93",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     0},
};

class QuantReduceScatterArch22TilingTest : public testing::TestWithParam<QuantReduceScatterTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantReduceScatterArch22TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantReduceScatterArch22TilingTest TearDown." << std::endl;
    }
};

static struct QuantReduceScatterCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const QuantReduceScatterTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    // 参数封装
    gert::StorageShape contextShape = {{1}, {1}};
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape scalesShape = {param.scalesShape, param.scalesShape};
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND},
         {xShape, param.xDtype, param.xFormat},
         {scalesShape, param.scalesDtype, param.scalesFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outputShape, param.outputDtype, param.outputFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSizeAttr)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduceOpAttr)},
         {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.outputDtypeAttr))},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.rankNum))}});
    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

static void ThreadFunction(const QuantReduceScatterTestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const QuantReduceScatterTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(QuantReduceScatterArch22TilingTest, GeneralCasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(QuantReduceScatterArch22TilingTest, GeneralCasesMultiThreadTest)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(QuantReduceScatterTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(QuantReduceScatterTilingUT, QuantReduceScatterArch22TilingTest, testing::ValuesIn(g_testCases));

} // namespace QuantReduceScatterUT
