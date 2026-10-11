/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace DistributeBarrierExtendUT {
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

class DistributeBarrierExtendArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierExtendArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierExtendArch35TilingTest TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTiling)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 65536 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingWorldSize1)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingWorldSize1025)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1025)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    TilingInfo tilingInfo;
    ASSERT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingTimeOut)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 65537 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingElasticInfo)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 65792 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

TEST_F(DistributeBarrierExtendArch35TilingTest, TestTilingTimeOutElasticInfo)
{
    struct DistributeBarrierExtendCompileInfo {
    } compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara("DistributeBarrierExtend",
                                              {
                                                  {{{2052}, {2052}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{36}, {36}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{3, 4}, {3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                               {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)}},
                                              &compileInfo, "3510", coreNum, ubSize);
    std::string expectTilingData = "16 0 20 0 * 0 0 0 65793 0 ";
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    auto tilingDataResult =
        ToString<uint32_t>(tilingInfo.tilingData.get() + g_mc2TilingDataReservedLen,
                           tilingInfo.tilingDataSize - g_mc2TilingDataReservedLen, g_barrierTilingDataMask);
    EXPECT_EQ(expectTilingData, tilingDataResult);
}

} // namespace DistributeBarrierExtendUT
