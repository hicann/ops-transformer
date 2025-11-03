/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/op_tiling/quant_matmul_all_reduce_add_rms_norm_tiling.h"
#include "../../../op_host/op_tiling/matmul_all_reduce_add_rms_norm_tiling.h"
#include "../../../op_host/op_tiling/weight_quant_matmul_all_reduce_add_rms_norm_tiling.h"

using namespace std;
namespace MatmulAllReduceAddRmsNormUT {
struct TestParam {
    string caseName;
    uint32_t blockDim;
    uint64_t tilingKey;
};

using WeightQuantTestParam = TestParam;

std::unique_ptr<gert::TilingContextPara::TensorDescription> CreateTensorShape(
    gert::StorageShape shape,
    ge::DataType dtype,
    ge::Format format
) {
    return std::unique_ptr<gert::TilingContextPara::TensorDescription>(
        new gert::TilingContextPara::TensorDescription(
            shape,  // 这里用大括号构造
            dtype,
            format
        )
    );
}

class MatmulAllReduceAddRmsNormTiling : public testing::TestWithParam<TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceAddRmsNormTiling Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceAddRmsNormTiling Test TearDown" << std::endl;
    }
};

static void SplitStr2Vec(const string& input, const string& delimiter, vector<string>& output) {
    auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }

    if (currPos < input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}
map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16}, {"FLOAT", ge::DT_FLOAT}, {"BF16", ge::DT_BF16},
                                      {"INT8", ge::DT_INT8},       {"INT4", ge::DT_INT4},   {"UINT64", ge::DT_UINT64},
                                      {"INT32", ge::DT_INT32}};

struct MatmulAllReduceArnCompileInfo{
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;    
};
static void TestOneParamCase(const WeightQuantTestParam& param) {
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName, "_", testParam);

    size_t idx = 0;
    string model_name = testParam[idx++];
    bool is_valid_case = false;
    if (model_name.find("InValid") != std::string::npos) {
        is_valid_case = true;
    }
    string group_name = testParam[idx++];
    string reduce_op = testParam[idx++];

    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t n = stol(testParam[idx++]);

    int64_t biasFlag = stol(testParam[idx++]);
    int64_t x3Flag = stol(testParam[idx++]);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    int64_t antiquant_offsetExistFlag = stol(testParam[idx++]);
    int64_t antiquant_scaleExistFlag = stol(testParam[idx++]);
    int64_t dequant_scaleExistFlag = stol(testParam[idx++]);
    int64_t antigroupSize = stol(testParam[idx++]);
    float epslion = stof(testParam[idx++]);

    ge::DataType xDtype = dtypeMap[testParam[idx++]];
    ge::DataType weightDtype = dtypeMap[testParam[idx++]];
    ge::DataType biasDtype = dtypeMap[testParam[idx++]];
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    ge::DataType normOutDtype = yDtype;
    ge::DataType quantDtype = yDtype;

    auto xShape = CreateTensorShape({}, xDtype, ge::FORMAT_ND);
    auto weigthShape = CreateTensorShape({}, weightDtype, ge::FORMAT_ND);
    auto biasShape = CreateTensorShape({}, biasDtype, ge::FORMAT_ND);
    auto residualShape = CreateTensorShape({{1, m, n}, {1, m, n}}, yDtype, ge::FORMAT_ND);
    auto gammaShape = CreateTensorShape({{n}, {n}}, yDtype, ge::FORMAT_ND);
    auto antiQuantOffsetShape = CreateTensorShape({}, xDtype, ge::FORMAT_ND);
    auto antiQuantScaleShape = CreateTensorShape({}, xDtype, ge::FORMAT_ND);
    auto quantScaleShape = CreateTensorShape({}, quantDtype, ge::FORMAT_ND);
    auto yShape = CreateTensorShape({{1, m, n}, {1, m, n}}, yDtype, ge::FORMAT_ND);

    auto normOutputShape = CreateTensorShape({{1, m, n}, {1, m, n}}, normOutDtype, ge::FORMAT_ND);

    if (transA) {
        xShape->shape_ = {{k, m}, {k, m}};
    } else {
        xShape->shape_ = {{m, k}, {m, k}};
    }

    if (transB) {
        weigthShape->shape_ = {{n, k}, {n, k}};
    } else {
        weigthShape->shape_ = {{k, n}, {k, n}};
    }

    {
        if (group > 0) {
            int64_t groupNum = (k + group - 1) / group;
            if (transB) {
                antiQuantOffsetShape->shape_ = {{n, groupNum}, {n, groupNum}};
                antiQuantScaleShape->shape_ = {{n, groupNum}, {m, groupNum}};
            } else {
                antiQuantOffsetShape->shape_ = {{groupNum, n}, {groupNum, n}};
                antiQuantScaleShape->shape_ = {{groupNum, n}, {groupNum, n}};
            }
        } else if (group < 0) {
            antiQuantOffsetShape->shape_ = {{n}, {n}};
            antiQuantScaleShape->shape_ = {{n}, {n}};
            quantScaleShape->shape_ = {{n}, {n}};
            if (yDtype != ge::DT_BF16) {
                quantDtype = ge::DT_UINT64;
            }
        } else {
            antiQuantOffsetShape->shape_ = {{1}, {1}};
            antiQuantScaleShape->shape_ = {{1}, {1}};
            quantScaleShape->shape_ = {{1}, {1}};
            if (yDtype != ge::DT_BF16) {
                quantDtype = ge::DT_UINT64;
            }
        }
    }
    quantScaleShape->dtype_ = quantDtype;
    if (biasFlag){
        biasShape->shape_ = {{n}, {n}};
    }
    if (!antiquant_offsetExistFlag){
        antiQuantOffsetShape->shape_ = {};
    }
    if (!antiquant_scaleExistFlag){
        antiQuantScaleShape->shape_ = {};
    }
    if (!dequant_scaleExistFlag){
        quantScaleShape->shape_ = {};
    }
    if (yShape->dtype_ == ge::DT_BF16) {
        printf("the yDtype is BF16\n");
    } else {
        printf("Exist error!\n");
    }
    printf("dataType BF16 is : %d",ge::DataType::DT_BF16);
    MatmulAllReduceArnCompileInfo compileInfo {8, 262144};
    const std::string socVersion = "Ascend910B";
    uint64_t coreNum = 8;
    uint64_t ubSize = 262144;
    uint64_t tilingDataSize = 40960;
    gert::TilingContextPara tilingContextPara("MatmulAllReduceAddRmsNorm",
        {
            // input info
            *xShape,
            *weigthShape,
            *biasShape,
            *residualShape,
            *gammaShape,
            *antiQuantOffsetShape,
            *antiQuantScaleShape,
            *quantScaleShape,
        },
        {
            //output info
            *yShape,
            *normOutputShape,
        },
        {
            // attr
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(group_name)},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(reduce_op)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(transA)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(transB)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(antigroupSize)},
            {"epslion", Ops::Transformer::AnyValue::CreateFrom<float>(epslion)},
        },
        &compileInfo,
        socVersion,
        coreNum,
        ubSize,
        tilingDataSize);
    uint64_t expectTilingKey = 10000UL;
    std::string expectTilingData = "8 16 64 262144 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    uint64_t mc2TilingDataReservedLen = 42;
    if (is_valid_case) {
        ExecuteTestCase(tilingContextPara);
    } else {
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.tilingKey);
    }
}

TEST_P(MatmulAllReduceAddRmsNormTiling, generalTest) {
TestParam param = GetParam();
TestOneParamCase(param);
}

static TestParam casesParamsQuant[] = {
        // 量化
        {"MODEL0_group_sum_4096_688_4096_1_0_0_0_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16", 8, 0},
        // transB
        {"MODEL0_group_sum_4_4096_11008_1_0_0_1_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16", 8, 1},
        {"MODEL0_group_sum_4_4096_11008_1_0_0_0_-1_0_0_1_10_0.1_INT8_INT8_INT32_BF16", 8, 0},
        // 伪量化
        {"MODEL0_group_sum_4096_688_4096_1_0_0_0_-1_1_1_0_0_0.1_FLOAT16_INT8_FLOAT16_FLOAT16", 8, 311200},
        // transB
        {"MODEL0_group_sum_4_4096_11008_1_0_0_1_-1_1_1_0_0_0.1_FLOAT16_INT4_FLOAT16_FLOAT16", 8, 311210},
        // per group
        {"MODEL0_group_sum_4_4096_11008_1_0_0_0_32_1_1_0_32_0.1_BF16_INT4_BF16_BF16", 8, 311300},
        // 非量化
        {"MODEL0_group_sum_4_4096_11008_1_0_0_0_32_0_0_0_32_0.1_BF16_BF16_BF16_BF16", 8, 65536UL},
        {"MODEL0_group_sum_9471_18_379_1_0_0_0_32_0_0_0_32_0.1_BF16_BF16_BF16_BF16", 8, 65536UL},

};
static TestParam InValidCheckcasesParamsQuant[] = {
        // dequant设置的同时，antiquant信息也设置
        {"InValidMODEL0_group_sum_4_4096_11008_1_0_0_0_-1_1_1_1_10_0.1_INT8_INT8_BF16_BF16", 8, 311300},
        // transA
        {"InValidMODEL0_group_sum_4096_688_4096_1_0_1_0_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16", 8, 0},
        // 伪量化
        // transA
        {"InValidMODEL0_group_sum_4096_688_4096_1_0_1_0_-1_1_1_0_0_0.1_BF16_INT8_BF16_BF16", 8, 311200},
        // per group: groupsize <32
        {"InValidMODEL0_group_sum_4_4096_11008_1_0_0_0_32_1_1_0_30_0.1_BF16_INT4_BF16_BF16", 8, 311300},
        // per group: k - 1 < 32
        {"InValidMODEL0_group_sum_4_2_11008_1_0_0_0_32_1_1_0_32_0.1_BF16_INT4_BF16_BF16", 8, 311300},
        // epsilon
        {"InValidMODEL0_group_sum_4096_688_4096_1_0_0_0_-1_0_0_1_0_0_INT8_INT8_INT32_BF16", 8, 0},
        {"InValidMODEL0_group_sum_4096_688_4096_1_0_0_0_-1_0_0_1_0_1_INT8_INT8_INT32_BF16", 8, 0},
        // reduceOp
        {"InValidMODEL0_group_mul_4096_688_4096_1_0_0_0_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16", 8, 0},
        // commTurn
        {"InValidMODEL0_group_mul_4096_688_4096_1_0_0_0_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16", 8, 0},
};

// 正常用例合集

INSTANTIATE_TEST_CASE_P(MatMulAllReduceAddResNormal, MatmulAllReduceAddRmsNormTiling,
        testing::ValuesIn(casesParamsQuant));

// 异常用例合集

INSTANTIATE_TEST_CASE_P(MatMulAllReduceAddResNormal2, MatmulAllReduceAddRmsNormTiling,
        testing::ValuesIn(InValidCheckcasesParamsQuant));
}//matmul_all_reduce_add_rms_norm_ut