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
#include <unordered_set>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace std;

namespace DistributeBarrierUT {
constexpr uint64_t g_mc2TilingDataReservedLen = sizeof(Mc2InitTiling) + sizeof(Mc2CcTiling);

template <typename T>
static string ToString(void* buf, size_t size, unordered_set<size_t> mask = {})
{
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += mask.find(i) == mask.end() ? std::to_string(data[i]) : "*";
        result += " ";
    }
    return result;
}

const unordered_set<size_t> g_barrierTilingDataMask = {4};

class DistributeBarrierArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierArch22TilingTest, TestTiling)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 0 0 ";
    // barrier算子没有tilingkey，需要单独搭建与其他mc2算子不同的测试
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierArch22TilingTest, TestTilingWorldSize1)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierArch22TilingTest, TestTilingWorldSize385)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(385)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOut)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 1 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierArch22TilingTest, TestTilingElasticInfo)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 256 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOutElasticInfo)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 257 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

// timeout negative: dim0 != 1
TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOutDim0Invalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// timeout negative: dim num != 1
TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOutDimNumInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1, 1}, {1, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// timeout negative: wrong dtype
TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOutDtypeInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// elasticInfo negative: dim0 != 4 + 2 * worldSize
TEST_F(DistributeBarrierArch22TilingTest, TestTilingElasticInfoDim0Invalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// elasticInfo negative: dim num != 1
TEST_F(DistributeBarrierArch22TilingTest, TestTilingElasticInfoDimNumInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36, 1}, {36, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// elasticInfo negative: wrong dtype
TEST_F(DistributeBarrierArch22TilingTest, TestTilingElasticInfoDtypeInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// timeout negative: format is FRACTAL_NZ
TEST_F(DistributeBarrierArch22TilingTest, TestTilingTimeOutFormatInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

// elasticInfo negative: format is FRACTAL_NZ
TEST_F(DistributeBarrierArch22TilingTest, TestTilingElasticInfoFormatInvalid)
{
    struct DistributeBarrierCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrier",
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "Ascend910_93", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

} // namespace DistributeBarrierUT
