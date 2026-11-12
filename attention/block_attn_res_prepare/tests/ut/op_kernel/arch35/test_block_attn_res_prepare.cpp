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
#include "../../../../op_kernel/block_attn_res_prepare_apt.cpp"
#endif
#include "../test_block_attn_res_prepare_utils.h"

namespace {

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
constexpr const char* KERNEL_NPU_ARCH = "9201";
#else
constexpr const char* KERNEL_NPU_ARCH = "3510";
#endif

const BlockAttnResPrepareKernelCsvLoadResult& GetKernelCases()
{
    static const BlockAttnResPrepareKernelCsvLoadResult result =
        BlockAttnResPrepareKernelTestUtils::GetParams(KERNEL_NPU_ARCH, "BLOCK_ATTN_RES_PREPARE_FLOW");
    return result;
}

std::string MakeParamName(const testing::TestParamInfo<BlockAttnResPrepareKernelTestParam>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class BlockAttnResPrepareKernelTest : public testing::TestWithParam<BlockAttnResPrepareKernelTestParam> {
protected:
    static void SetUpTestCase()
    {
#ifdef __CCE_KT_TEST__
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    }
};

TEST(BlockAttnResPrepareKernelCsv, LoadsSelectedSocCases)
{
    const auto& result = GetKernelCases();
    for (const auto& error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.params.empty());
}

TEST_P(BlockAttnResPrepareKernelTest, RunKernel)
{
#ifdef __CCE_KT_TEST__
    BlockAttnResPrepareKernelTestUtils::TestOneParamCase(GetParam());
#else
    GTEST_SKIP() << "Kernel CPU simulator is unavailable";
#endif
}

INSTANTIATE_TEST_SUITE_P(BLOCK_ATTN_RES_PREPARE_ARCH35, BlockAttnResPrepareKernelTest,
                         testing::ValuesIn(GetKernelCases().params), MakeParamName);

} // namespace
