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
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"

namespace {
using TensorDesc = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

TensorDesc Desc(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    return TensorDesc(gert::StorageShape(dims, dims), dtype, ge::FORMAT_ND);
}

TensorDesc Empty(ge::DataType dtype)
{
    return Desc({}, dtype);
}

gert::InfershapeContextPara MakePara(const std::string &layout, ge::DataType dtype, int64_t batch, int64_t sequence,
                                     int64_t hidden, int64_t headDim, int64_t cmpRatio, int64_t blockSize,
                                     int64_t maxBlocks, bool withNorm = false)
{
    const int64_t tokenCount = layout == "BSH" ? batch * sequence : sequence;
    const int64_t blockNum = std::max<int64_t>(1, batch * maxBlocks);
    std::vector<TensorDesc> inputs;
    if (layout == "BSH") {
        inputs.emplace_back(Desc({batch, sequence, hidden}, dtype));
    } else {
        inputs.emplace_back(Desc({tokenCount, hidden}, dtype));
    }
    inputs.emplace_back(Desc({headDim, hidden}, dtype));
    inputs.emplace_back(Desc({headDim, hidden}, dtype));
    inputs.emplace_back(Desc({cmpRatio, headDim}, ge::DT_FLOAT));
    inputs.emplace_back(Desc({blockNum, blockSize, 2 * headDim}, ge::DT_FLOAT));
    inputs.emplace_back(Desc({batch, maxBlocks}, ge::DT_INT32));
    inputs.emplace_back(Desc({batch}, ge::DT_INT32));
    inputs.emplace_back(withNorm ? Desc({headDim}, ge::DT_FLOAT) : Empty(ge::DT_FLOAT));
    inputs.emplace_back(withNorm ? Desc({headDim}, ge::DT_FLOAT) : Empty(ge::DT_FLOAT));
    inputs.emplace_back(Empty(ge::DT_BF16));
    inputs.emplace_back(Empty(ge::DT_BF16));
    inputs.emplace_back(layout == "TH" ? Desc({batch + 1}, ge::DT_INT32) : Empty(ge::DT_INT32));
    inputs.emplace_back(Empty(ge::DT_INT32));

    std::vector<TensorDesc> outputs{Empty(dtype)};
    std::vector<OpAttr> attrs{
        {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
        {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize * 2 * headDim)},
    };

    gert::InfershapeContextPara para("KeyPool", inputs, outputs, attrs);
    return para;
}
} // namespace

TEST(KeyPoolInfershape, BshBf16D128)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 8, 2048, 128, 4, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 2, 128}});
}

TEST(KeyPoolInfershape, BshFp16D512)
{
    auto para = MakePara("BSH", ge::DT_FLOAT16, 1, 256, 4096, 512, 128, 128, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 2, 512}});
}

TEST(KeyPoolInfershape, BshLayerNorm)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 16, 4096, 512, 8, 8, 2, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 2, 512}});
}

TEST(KeyPoolInfershape, ThUnevenBatch)
{
    auto para = MakePara("TH", ge::DT_BF16, 3, 17, 4096, 512, 4, 4, 4);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{3, 4, 512}});
}

TEST(KeyPoolInfershape, ThFp16)
{
    auto para = MakePara("TH", ge::DT_FLOAT16, 2, 256, 2048, 128, 16, 16, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 2, 128}});
}

TEST(KeyPoolInfershape, EmptyBatch)
{
    auto para = MakePara("BSH", ge::DT_BF16, 0, 8, 4096, 512, 4, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{0, 2, 512}});
}

TEST(KeyPoolInfershape, ZeroCmpRatio)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 8, 2048, 128, 0, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
