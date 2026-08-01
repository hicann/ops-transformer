/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "quant_block_sparse_attn_host_ut_param.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

class QuantBlockSparseAttnInferShapeTest : public testing::TestWithParam<QuantBlockSparseAttnInferShapeUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantBlockSparseAttnInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantBlockSparseAttnInferShapeTest TearDown" << std::endl; }
};

TEST_P(QuantBlockSparseAttnInferShapeTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<TensorDesc> inputTensorDesc({
        param.query,
        param.key,
        param.value,
        param.qDescale,
        param.kDescale,
        param.vDescale,
        param.pScale,
        param.cuSeqlensQ,
        param.cuSeqlensKv,
        param.sequsedQ,
        param.sequsedKv,
        param.sparseIndices,
        param.sparseSeqLen,
        param.blockTable,
        param.attenMask,
        param.metadata,
    });

    std::vector<TensorDesc> outputTensorDesc({
        param.attentionOut,
        param.softmaxLse,
    });

    std::vector<OpAttr> attrs({
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_q_block_size)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_kv_block_size)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_sparse_indices)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
    });

    gert::InfershapeContextPara para(OP_NAME, inputTensorDesc, outputTensorDesc, attrs);

    ExecuteTestCase(para, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnInferShape, QuantBlockSparseAttnInferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnInferShapeUtParam>);

} // namespace QuantBlockSparseAttnUT
