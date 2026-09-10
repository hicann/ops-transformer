/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_all_reduce.cpp
 * \brief aclnn ut
 */

#include <float.h>
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "aclnn_quant_all_reduce.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace QuantAllReduceUT {
class TestAclnnQuantAllReduce : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
        cout << "TestAclnnQuantAllReduce SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnQuantAllReduce TearDown" << endl;
    }
};

struct QuantAllReduceAclnnTestParam {
    string caseName;
    // shape
    vector<int64_t> xShape;
    vector<int64_t> scalesShape;
    vector<int64_t> outputShape;
    // dtype
    aclDataType xDtype;
    aclDataType scalesDtype;
    aclDataType outputDtype;
    // format
    aclFormat xFormat;
    aclFormat scalesFormat;
    aclFormat outputFormat;
    // 返回状态
    aclnnStatus aclnnStatusUt;
};

static QuantAllReduceAclnnTestParam g_dtypeCasesParams[] = {
    // mx正常用例
    {"test_aclnn_quant_all_reduce_mx_BS96_H7168_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_true",
     {96, 7168},
     {96, 112, 2},
     {96, 7168},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_BS96_H7168_FLOAT8E5M2_FLOAT8E8M0_FLOAT16_true",
     {96, 7168},
     {96, 112, 2},
     {96, 7168},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H7168_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_true",
     {1, 96, 7168},
     {1, 96, 112, 2},
     {1, 96, 7168},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H7168_FLOAT8E5M2_FLOAT8E8M0_BF16_true",
     {1, 96, 7168},
     {1, 96, 112, 2},
     {1, 96, 7168},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT8_E8M0,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_true",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E5M2_FLOAT8E8M0_FLOAT16_true",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H4096_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT16_true",
     {1, 96, 4096},
     {1, 96, 64, 2},
     {1, 96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H4096_FLOAT8E5M2_FLOAT8E8M0_FLOAT_true",
     {1, 96, 4096},
     {1, 96, 64, 2},
     {1, 96, 4096},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    // K-G正常用例
    {"test_aclnn_quant_all_reduce_kg_BS96_H7168_INT8_FLOAT_FLOAT_true",
     {96, 7168},
     {96, 56},
     {96, 7168},
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H7168_HIFLOAT8_FLOAT_FLOAT_true",
     {96, 7168},
     {96, 56},
     {96, 7168},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H7168_FLOAT8E4M3FN_FLOAT_FLOAT_true",
     {96, 7168},
     {96, 56},
     {96, 7168},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H7168_FLOAT8E5M2_FLOAT_FLOAT_true",
     {96, 7168},
     {96, 56},
     {96, 7168},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H7168_INT8_FLOAT_BF16_true",
     {1, 96, 7168},
     {1, 96, 56},
     {1, 96, 7168},
     ACL_INT8,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H7168_HIFLOAT8_FLOAT_FLOAT_true",
     {1, 96, 7168},
     {1, 96, 56},
     {1, 96, 7168},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H7168_FLOAT8E4M3FN_FLOAT_BF16_true",
     {1, 96, 7168},
     {1, 96, 56},
     {1, 96, 7168},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H7168_FLOAT8E5M2_FLOAT_BF16_true",
     {1, 96, 7168},
     {1, 96, 56},
     {1, 96, 7168},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_INT8_FLOAT_FLOAT16_true",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_HIFLOAT8_FLOAT_BF16_true",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_FLOAT8E4M3FN_FLOAT_BF16_true",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_FLOAT8E5M2_FLOAT_FLOAT16_true",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_INT8_FLOAT_BF16_true",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_INT8,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_HIFLOAT8_FLOAT_BF16_true",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_FLOAT8E4M3FN_FLOAT_BF16_true",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_FLOAT8E5M2_FLOAT_FLOAT16_true",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_SUCCESS},
    // 异常用例-组合异常与类型异常
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H4096_INT8_FLOAT8E8M0_BF16_false",
     {1, 96, 4096},
     {1, 96, 64, 2},
     {1, 96, 4096},
     ACL_INT8,
     ACL_FLOAT8_E8M0,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // 组合异常mx量化
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_HIFLOAT8_FLOAT8E8M0_FLOAT16_false",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // 组合异常mx量化
    {"test_aclnn_quant_all_reduce_mx_B1_S96_H4096_FLOAT16_FLOAT8E8M0_BF16_false",
     {1, 96, 4096},
     {1, 96, 64, 2},
     {1, 96, 4096},
     ACL_FLOAT16,
     ACL_FLOAT8_E8M0,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // x类型异常mx量化
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E4M3FN_FLOAT16_FLOAT_false",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // scales类型异常mx量化
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT_FLOAT8E8M0_INT8_false",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E5M2,
     ACL_FLOAT8_E8M0,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // output类型异常mx量化
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_FLOAT16_FLOAT_BF16_false",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // x类型异常K-G量化
    {"test_aclnn_quant_all_reduce_kg_B1_S96_H4096_INT8_FLOAT_FLOAT_false",
     {1, 96, 4096},
     {1, 96, 32},
     {1, 96, 4096},
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // scales类型异常K-G量化
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_HIFLOAT8_FLOAT_INT8_false",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID}, // output类型异常K-G量化
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_HIFLOAT8_FLOAT_INT8_false_invalid_format_NZ",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_INT8,
     ACL_FORMAT_FRACTAL_Z,
     ACL_FORMAT_FRACTAL_Z,
     ACL_FORMAT_FRACTAL_Z,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_HIFLOAT8_FLOAT_INT8_false_invalid_format_NCL",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_HIFLOAT8,
     ACL_FLOAT,
     ACL_INT8,
     ACL_FORMAT_NCL,
     ACL_FORMAT_NCL,
     ACL_FORMAT_NCL,
     ACLNN_ERR_PARAM_INVALID},
    // 私有format异常-有效dtype，触发IsPrivateFormat检查
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_INT8_FLOAT_FLOAT_false_private_format_x",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_INT8_FLOAT_FLOAT_false_private_format_scales",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_kg_BS96_H4096_INT8_FLOAT_FLOAT_false_private_format_output",
     {96, 4096},
     {96, 32},
     {96, 4096},
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_false_private_format_x",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_false_private_format_scales",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACL_FORMAT_ND,
     ACLNN_ERR_PARAM_INVALID},
    {"test_aclnn_quant_all_reduce_mx_BS96_H4096_FLOAT8E4M3FN_FLOAT8E8M0_FLOAT_false_private_format_output",
     {96, 4096},
     {96, 64, 2},
     {96, 4096},
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     ACLNN_ERR_PARAM_INVALID},
};

static void TestOneParamCase(const QuantAllReduceAclnnTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    vector<int64_t> xShape = param.xShape;
    vector<int64_t> scalesShape = param.scalesShape;
    vector<int64_t> outputShape = param.outputShape;
    aclDataType xDtype = param.xDtype;
    aclDataType scalesDtype = param.scalesDtype;
    aclDataType outputDtype = param.outputDtype;
    aclFormat xFormat = param.xFormat;
    aclFormat scalesFormat = param.scalesFormat;
    aclFormat outputFormat = param.outputFormat;
    aclnnStatus retStatus = param.aclnnStatusUt;
    // 封装
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc x = TensorDesc(xShape, xDtype, xFormat);
    TensorDesc scales = TensorDesc(scalesShape, scalesDtype, scalesFormat);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    const char *reduceOp = "sum";
    auto ut =
        OP_API_UT(aclnnQuantAllReduce, INPUT(context, x, scales, hcclBufferSize, worldSize, reduceOp), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
}

TEST_F(TestAclnnQuantAllReduce, DtypeCasesParamsTest)
{
    if (std::size(g_dtypeCasesParams) != 0) {
        uint64_t numCases = sizeof(g_dtypeCasesParams) / sizeof(g_dtypeCasesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_dtypeCasesParams[idx]);
        }
    }
}

TEST_F(TestAclnnQuantAllReduce, NullptrContextTest)
{
    auto x = TensorDesc({96, 4096}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto scales = TensorDesc({96, 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantAllReduce,
                        INPUT((aclTensor *)nullptr, x, scales, (int64_t)2097152, (int64_t)2, "sum"), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnQuantAllReduce, NullptrInputTest)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto scales = TensorDesc({96, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    auto ut = OP_API_UT(aclnnQuantAllReduce,
                        INPUT(context, (aclTensor *)nullptr, scales, hcclBufferSize, worldSize, "sum"), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnQuantAllReduce, NullptrScalesTest)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({96, 4096}, ACL_INT8, ACL_FORMAT_ND);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    auto ut = OP_API_UT(aclnnQuantAllReduce, INPUT(context, x, (aclTensor *)nullptr, hcclBufferSize, worldSize, "sum"),
                        OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnQuantAllReduce, ReFormatXTest)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({96, 4096}, ACL_INT8, ACL_FORMAT_NCHW);
    auto scales = TensorDesc({96, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    auto ut =
        OP_API_UT(aclnnQuantAllReduce, INPUT(context, x, scales, hcclBufferSize, worldSize, "sum"), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnQuantAllReduce, ReFormatScalesTest)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({96, 4096}, ACL_INT8, ACL_FORMAT_ND);
    auto scales = TensorDesc({96, 32}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    auto ut =
        OP_API_UT(aclnnQuantAllReduce, INPUT(context, x, scales, hcclBufferSize, worldSize, "sum"), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnQuantAllReduce, ReFormatOutputTest)
{
    TensorDesc context = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({96, 4096}, ACL_INT8, ACL_FORMAT_ND);
    auto scales = TensorDesc({96, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output = TensorDesc({96, 4096}, ACL_FLOAT, ACL_FORMAT_NCHW);
    int64_t hcclBufferSize = 2097152;
    int64_t worldSize = 2;
    auto ut =
        OP_API_UT(aclnnQuantAllReduce, INPUT(context, x, scales, hcclBufferSize, worldSize, "sum"), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

} // namespace QuantAllReduceUT
