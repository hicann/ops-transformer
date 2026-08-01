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

#include "../quant_block_sparse_attn_host_ut_param.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace QuantBlockSparseAttnUT {

static const std::string OP_NAME = "QuantBlockSparseAttn";

struct QuantBlockSparseAttnCompileInfo {
} compileInfo;

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

const TensorDesc EmptyInput(gert::StorageShape({0}, {0}), ge::DT_INT32, ge::FORMAT_ND);

static TensorDesc NormalizeTd(const TensorDesc &td)
{
    if (td.dtype_ == ge::DT_UNDEFINED) {
        return EmptyInput;
    }
    return td;
}

class QuantBlockSparseAttnTilingArch35Test : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {
protected:
    static void SetUpTestCase() { std::cout << "QuantBlockSparseAttnTilingArch35Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantBlockSparseAttnTilingArch35Test TearDown" << std::endl; }
};

TEST_P(QuantBlockSparseAttnTilingArch35Test, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<TensorDesc> inputTensorDesc({
        NormalizeTd(param.query),
        NormalizeTd(param.key),
        NormalizeTd(param.value),
        NormalizeTd(param.qDescale),
        NormalizeTd(param.kDescale),
        NormalizeTd(param.vDescale),
        NormalizeTd(param.pScale),
        NormalizeTd(param.cuSeqlensQ),
        NormalizeTd(param.cuSeqlensKv),
        NormalizeTd(param.sequsedQ),
        NormalizeTd(param.sequsedKv),
        NormalizeTd(param.sparseIndices),
        NormalizeTd(param.sparseSeqLen),
        NormalizeTd(param.blockTable),
        NormalizeTd(param.attenMask),
        NormalizeTd(param.metadata),
    });

    std::vector<TensorDesc> outputTensorDesc({
        NormalizeTd(param.attentionOut),
        NormalizeTd(param.softmaxLse),
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

    gert::TilingContextPara para(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, "Ascend950", 64,
                                 262144, 65536);

    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);

    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ok, true);
        EXPECT_GT(tilingInfo.blockNum, 0U);
        EXPECT_GT(tilingInfo.tilingDataSize, 0U);
    } else {
        EXPECT_EQ(ok, false);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantBlockSparseAttnTiling, QuantBlockSparseAttnTilingArch35Test,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

} // namespace QuantBlockSparseAttnUT
