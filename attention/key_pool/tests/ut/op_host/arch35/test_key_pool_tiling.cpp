/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../../op_host/arch35/key_pool_tiling.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

const char *kA5SocInfo = "{"
                         "\"hardware_info\":{"
                         "\"BT_SIZE\":0,"
                         "\"load3d_constraints\":\"1\","
                         "\"Intrinsic_fix_pipe_l0c2out\":false,"
                         "\"Intrinsic_data_move_l12ub\":true,"
                         "\"Intrinsic_data_move_l0c2ub\":true,"
                         "\"Intrinsic_data_move_out2l1_nd2nz\":false,"
                         "\"4096\":196608,"
                         "\"L2_SIZE\":201326592,"
                         "\"L1_SIZE\":524288,"
                         "\"L0A_SIZE\":65536,"
                         "\"L0B_SIZE\":65536,"
                         "\"L0C_SIZE\":131072,"
                         "\"vector_core_cnt\":40,"
                         "\"cube_core_cnt\":20,"
                         "\"socVersion\":\"Ascend950\""
                         "}"
                         "}";

TensorDesc Desc(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    return TensorDesc(gert::StorageShape(dims, dims), dtype, ge::FORMAT_ND);
}

TensorDesc Empty(ge::DataType dtype)
{
    return Desc({}, dtype);
}

gert::TilingContextPara MakePara(const std::string &layout, ge::DataType dtype, int64_t batch, int64_t sequence,
                                 int64_t hidden, int64_t headDim, int64_t cmpRatio, int64_t blockSize,
                                 int64_t maxBlocks, bool withNorm = false)
{
    const int64_t tokenCount = layout == "BSH" ? batch * sequence : sequence;
    const int64_t blockNum = std::max<int64_t>(1, batch * maxBlocks);
    std::vector<TensorDesc> inputs;
    inputs.emplace_back(layout == "BSH" ? Desc({batch, sequence, hidden}, dtype) : Desc({tokenCount, hidden}, dtype));
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

    const int64_t outputCapacity = (maxBlocks * blockSize + cmpRatio - 1) / cmpRatio;
    std::vector<TensorDesc> outputs{Desc({batch, outputCapacity, headDim}, dtype)};
    std::vector<OpAttr> attrs{
        {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cmpRatio)},
        {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6F)},
        {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"state_cache_stride_dim0", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize * 2 * headDim)},
    };
    return gert::TilingContextPara("KeyPool", inputs, outputs, attrs, nullptr, "Ascend950", kA5SocInfo, 4096);
}
} // namespace

TEST(KeyPoolTilingArch35, BshBf16)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 8, 4096, 512, 4, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 32);
}

TEST(KeyPoolTilingArch35, BshFp16)
{
    auto para = MakePara("BSH", ge::DT_FLOAT16, 1, 256, 4096, 512, 128, 128, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 34);
}

TEST(KeyPoolTilingArch35, BshLayerNorm)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 16, 2048, 128, 8, 8, 2, true);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 32);
}

TEST(KeyPoolTilingArch35, ThUnevenBatch)
{
    auto para = MakePara("TH", ge::DT_BF16, 3, 17, 4096, 512, 4, 4, 4);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 33);
}

TEST(KeyPoolTilingArch35, EmptyBatch)
{
    auto para = MakePara("BSH", ge::DT_BF16, 0, 8, 4096, 512, 4, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 48);
}

TEST(KeyPoolTilingArch35, RejectsUnsupportedHeadDim)
{
    auto para = MakePara("BSH", ge::DT_BF16, 2, 8, 2048, 256, 4, 4, 2);
    ExecuteTestCase(para, ge::GRAPH_FAILED, 32);
}
