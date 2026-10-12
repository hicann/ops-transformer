/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

namespace {
constexpr int64_t kBlockCount = 16;
constexpr int64_t kBlockSize = 64;
constexpr int64_t kHeadDim = 192;
constexpr int64_t kValueHeadDim = 128;
constexpr int64_t kN1 = 4;
constexpr int64_t kN2 = 1;
constexpr int64_t kT1 = 2;
constexpr int64_t kT2 = 2048;

// 11 inputs: q, k, v, attentionOut, attentionOutGrad, softmaxMax, softmaxSum,
//            topkIndices, actualSeqQLen(opt), actualSeqKVLen(opt), attenMsk(opt)
const std::vector<uint32_t> kInputIrInstanceFull{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
const std::vector<uint32_t> kOutputIrInstance{1, 1, 1};

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

std::vector<TensorDesc> MakeInputs(ge::DataType qkvDtype, int64_t t1 = kT1, int64_t t2 = kT2, int64_t n1 = kN1,
                                   int64_t n2 = kN2, int64_t d = kHeadDim, int64_t d2 = kValueHeadDim)
{
    return {
        // query: [t1, n1, d]
        {{{t1, n1, d}, {t1, n1, d}}, qkvDtype, ge::FORMAT_ND},
        // key: [t2, n2, d]
        {{{t2, n2, d}, {t2, n2, d}}, qkvDtype, ge::FORMAT_ND},
        // value: [t2, n2, d2]
        {{{t2, n2, d2}, {t2, n2, d2}}, qkvDtype, ge::FORMAT_ND},
        // attentionOut: [t1, n1, d2]
        {{{t1, n1, d2}, {t1, n1, d2}}, qkvDtype, ge::FORMAT_ND},
        // attentionOutGrad: [t1, n1, d2]
        {{{t1, n1, d2}, {t1, n1, d2}}, qkvDtype, ge::FORMAT_ND},
        // softmaxMax: [t1, n1, 8]
        {{{t1, n1, 8}, {t1, n1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // softmaxSum: [t1, n1, 8]
        {{{t1, n1, 8}, {t1, n1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        // topkIndices: [t1, n2, selectedBlockCount]
        {{{t1, n2, kBlockCount}, {t1, n2, kBlockCount}}, ge::DT_INT32, ge::FORMAT_ND},
        // actualSeqQLen optional - empty
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // actualSeqKVLen optional - empty
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        // attenMsk optional - empty
        {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
    };
}

std::vector<TensorDesc> MakeEmptyOutputs()
{
    return {
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
}

std::vector<OpAttr> MakeAttrs(const std::string& layout, int64_t blockCount = kBlockCount,
                              int64_t blockSize = kBlockSize, int64_t headNum = kN1, int64_t sparseMode = 0,
                              float scaleValue = 0.088388f)
{
    return {
        {"scaleValue", Ops::Transformer::AnyValue::CreateFrom<float>(scaleValue)},
        {"selectedBlockCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockCount)},
        {"selectedBlockSize", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
        {"headNum", Ops::Transformer::AnyValue::CreateFrom<int64_t>(headNum)},
        {"inputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        {"sparseMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(sparseMode)},
    };
}
} // namespace

class NsaSelectedAttentionGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaSelectedAttentionGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaSelectedAttentionGradProto TearDown" << std::endl;
    }
};

// A1: positive, TND layout, fp16, single batch
TEST_F(NsaSelectedAttentionGradProto, A1_TND_fp16_basic)
{
    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_FLOAT16), MakeEmptyOutputs(),
                                     MakeAttrs("TND"), kInputIrInstanceFull, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {kT1, kN1, kHeadDim},      // dq = query shape
        {kT2, kN2, kHeadDim},      // dk = key shape
        {kT2, kN2, kValueHeadDim}, // dv = value shape
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A2: positive, TND layout, bf16
TEST_F(NsaSelectedAttentionGradProto, A2_TND_bf16_basic)
{
    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_BF16), MakeEmptyOutputs(),
                                     MakeAttrs("TND", kBlockCount, kBlockSize, kN1, 2), kInputIrInstanceFull,
                                     kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {kT1, kN1, kHeadDim},
        {kT2, kN2, kHeadDim},
        {kT2, kN2, kValueHeadDim},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A3: positive, lower-case "tnd" should be accepted (case-insensitive layout in infershape)
TEST_F(NsaSelectedAttentionGradProto, A3_tnd_lowercase_layout)
{
    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_FLOAT16), MakeEmptyOutputs(),
                                     MakeAttrs("tnd"), kInputIrInstanceFull, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {kT1, kN1, kHeadDim},
        {kT2, kN2, kHeadDim},
        {kT2, kN2, kValueHeadDim},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// A4: positive, multi-batch TND with different t1/t2 values
TEST_F(NsaSelectedAttentionGradProto, A4_TND_multi_batch)
{
    constexpr int64_t kMultiT1 = 8;
    constexpr int64_t kMultiT2 = 8192;

    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_FLOAT16, kMultiT1, kMultiT2),
                                     MakeEmptyOutputs(), MakeAttrs("TND"), kInputIrInstanceFull, kOutputIrInstance);

    std::vector<std::vector<int64_t>> expectShapes = {
        {kMultiT1, kN1, kHeadDim},
        {kMultiT2, kN2, kHeadDim},
        {kMultiT2, kN2, kValueHeadDim},
    };
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectShapes);
}

// E1: negative, invalid layout BSH
TEST_F(NsaSelectedAttentionGradProto, E1_invalid_layout_BSH)
{
    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_FLOAT16), MakeEmptyOutputs(),
                                     MakeAttrs("BSH"), kInputIrInstanceFull, kOutputIrInstance);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// E2: negative, invalid layout SBH
TEST_F(NsaSelectedAttentionGradProto, E2_invalid_layout_SBH)
{
    gert::InfershapeContextPara para("NsaSelectedAttentionGrad", MakeInputs(ge::DT_FLOAT16), MakeEmptyOutputs(),
                                     MakeAttrs("SBH"), kInputIrInstanceFull, kOutputIrInstance);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
