/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef __CCE_KT_TEST__
#include "../../../../op_kernel/block_attn_res_update.cpp"
#endif
#include "../test_block_attn_res_update_utils.h"

namespace {

const BlockAttnResUpdateKernelCsvLoadResult &GetKernelCases()
{
    static const BlockAttnResUpdateKernelCsvLoadResult result =
        BlockAttnResUpdateKernelTestUtils::GetParams("Ascend950", "BLOCK_ATTN_RES_UPDATE_FLOW");
    return result;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResUpdateKernelTestParam> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class BlockAttnResUpdateKernelTest : public testing::TestWithParam<BlockAttnResUpdateKernelTestParam> {
protected:
    static void SetUpTestCase()
    {
#ifdef __CCE_KT_TEST__
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    }
};

TEST(BlockAttnResUpdateKernelCsv, LoadsAscend950Cases)
{
    const auto &result = GetKernelCases();
    for (const auto &error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.params.empty());
}

TEST_P(BlockAttnResUpdateKernelTest, RunKernel)
{
#ifdef __CCE_KT_TEST__
    BlockAttnResUpdateKernelTestUtils::TestOneParamCase950(GetParam());
#else
    GTEST_SKIP() << "Kernel CPU simulator is unavailable";
#endif
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_UPDATE_ASCEND950, BlockAttnResUpdateKernelTest,
                         testing::ValuesIn(GetKernelCases().params), MakeParamName);

} // namespace
