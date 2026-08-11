/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include <iostream>
#include <gtest/gtest.h>
#include "../../../../mla_preprocess/op_host/mla_preprocess_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {
gert::StorageShape MakeStorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    auto &originShape = shape.MutableOriginShape();
    auto &storageShape = shape.MutableStorageShape();
    originShape.SetDimNum(dims.size());
    storageShape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        originShape.SetDim(i, dims[i]);
        storageShape.SetDim(i, dims[i]);
    }
    return shape;
}
} // namespace

class MlaPreprocessTilingV2 : public testing::TestWithParam<int> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MlaPreprocessTilingV2 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MlaPreprocessTilingV2 TearDown" << std::endl;
    }
};

TEST_P(MlaPreprocessTilingV2, mla_preprocess_v2_test_tiling_case0)
{
    // 0: both [tokenNum, 64], 1: both [0], 2: mismatched, 3: token dimension mismatch.
    const int ropeInputMode = GetParam();
    struct MlaPreProcessCompileInfo {
    } compileInfo; // compileInfo对应算子op_host/op_tiling/${OP_NAME}_tiling.cpp中的
                   // struct ${OP_NAME}CompileInfo

    int64_t tokenNum = 8;
    int64_t hiddenNum = 7168;
    int64_t headNum = 32;
    int64_t blockNum = 192;
    int64_t blockSize = 128;

    int64_t wdqDim = 128;
    int64_t qRopeDim = 0;
    int64_t kRopeDim = 0;
    double epsilon = 1e-05;
    int64_t qRotaryCoeff = 2;
    int64_t kRotaryCoeff = 2;
    bool transposeWdq = true;
    bool transposeWuq = true;
    bool transposeWuk = true;
    int64_t cacheMode = 1;
    int64_t quantMode = 0;
    bool doRmsNorm = true;
    bool qDownOutFlag = true;
    int64_t wdkvSplitCount = 1;
    const std::vector<int64_t> normalRopeShape = {tokenNum, 64};
    const std::vector<int64_t> disabledRopeShape = {0};
    const std::vector<int64_t> tokenMismatchShape = {0, 64};
    const std::vector<int64_t> cosShape =
        ropeInputMode == 0 || ropeInputMode == 2
            ? normalRopeShape
            : (ropeInputMode == 1 ? disabledRopeShape : tokenMismatchShape);
    const std::vector<int64_t> sinShape =
        ropeInputMode == 0
            ? normalRopeShape
            : (ropeInputMode == 1 || ropeInputMode == 2 ? disabledRopeShape : tokenMismatchShape);
    gert::TilingContextPara tilingContextPara(
        "MlaPreprocessV2",
        {
            // input info
            {{{tokenNum, hiddenNum}, {tokenNum, hiddenNum}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND}, // input 对应 {{{原始shape}, {运行时shape}}, 数据类型, 数据格式}
            {{{hiddenNum}, {hiddenNum}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // gamma0
            {{{hiddenNum}, {hiddenNum}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // beta0
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                          // quantScale0
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},                                             // quantOffset0
            {{{2112, hiddenNum}, {2112, hiddenNum}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ},         // wdqkv
            {{{2112}, {2112}}, ge::DT_INT64, ge::FORMAT_ND},                                      // deScale0
            {{{2112}, {2112}}, ge::DT_INT32, ge::FORMAT_ND},                                      // bias0
            {{{1536}, {1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                    // gamma1
            {{{1536}, {1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                    // beta1
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                          // quantScale1
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},                                             // quantOffset1
            {{{headNum * 192, 1536}, {headNum * 192, 1536}}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ}, // wuq
            {{{headNum * 192}, {headNum * 192}}, ge::DT_INT64, ge::FORMAT_ND},                    // deScale1
            {{{headNum * 192}, {headNum * 192}}, ge::DT_INT32, ge::FORMAT_ND},                    // bias1
            {{{512}, {512}}, ge::DT_FLOAT16, ge::FORMAT_ND},                                      // gamma2
            {MakeStorageShape(cosShape), ge::DT_FLOAT16, ge::FORMAT_ND},                              // cos
            {MakeStorageShape(sinShape), ge::DT_FLOAT16, ge::FORMAT_ND},                              // sin
            {{{headNum, 128, 512}, {headNum, 128, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},          // wuk
            {{{blockNum, blockSize, 1, 576}, {blockNum, blockSize, 1, 576}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // kvCache
            {{{blockNum, blockSize, 1, 64}, {blockNum, blockSize, 1, 64}},
             ge::DT_FLOAT16,
             ge::FORMAT_ND},                                         // kvCacheRope
            {{{tokenNum}, {tokenNum}}, ge::DT_INT32, ge::FORMAT_ND}, // slotmapping
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // ctkvScale
            {{{headNum}, {headNum}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // qNopeScale
        },
        {
            // output info
            {{{tokenNum, headNum, 512}, {tokenNum, headNum, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{blockNum, blockSize, 1, 512}, {blockNum, blockSize, 1, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, headNum, 64}, {tokenNum, headNum, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{blockNum, blockSize, 1, 64}, {blockNum, blockSize, 1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{tokenNum, 1536}, {tokenNum, 1536}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            // 特别注意，attr是根据顺序匹配的，不是根据字段匹配的！！！不确定attr顺序的话，可以在算子的ophost/xxxx_def.h中看
            {"wdqDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qRopeDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"kRopeDim", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-05f)},
            {"qRotaryCoeff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"kRotaryCoeff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"transposeWdq", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transposeWuq", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"transposeWuk", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"cacheMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"quantMode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"doRmsNorm", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"wdkvSplitCount", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"qDownOutFlag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo);

    uint64_t expectTilingKey = 56UL;
    ExecuteTestCase(tilingContextPara, ropeInputMode >= 2 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS, expectTilingKey);
}

INSTANTIATE_TEST_SUITE_P(RopeInputModes, MlaPreprocessTilingV2, testing::Values(0, 1, 2, 3));
