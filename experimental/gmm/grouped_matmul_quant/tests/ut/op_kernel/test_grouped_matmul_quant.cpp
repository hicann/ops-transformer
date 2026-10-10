/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file test_grouped_matmul_quant.cpp
 *  \brief CPU-kernel smoke tests for GroupedMatmulQuant.
 *
 * The kernel uses Mix mode, Cube instructions and cross-core synchronization.
 * tikicpulib is used here to verify that representative execution paths can be
 * launched without a crash. Numerical accuracy is verified by the Ascend 910B
 * pytest suite in ../../test_grouped_matmul_quant.py.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void grouped_matmul_quant(GM_ADDR x, GM_ADDR quantizedWeight, GM_ADDR weightScale,
                                                           GM_ADDR weightOffset, GM_ADDR groupList, GM_ADDR y,
                                                           GM_ADDR workspace, GM_ADDR tiling);

class GroupedMatmulQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulQuantKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulQuantKernelTest TearDown" << std::endl;
    }
};

namespace {
// tikicpulib runs the two AIV sub-blocks of a Mix kernel serially while
// accounting their source buffers against one 192 KiB UB budget. Give each
// simulated AIV half of that budget. L0A is reduced as well because the
// anti-quant temporary layout in UB is derived from L0ASize.
constexpr uint32_t UB_SIZE = 196608U / 2U;
constexpr uint32_t L1_SIZE = 524288U;
constexpr uint32_t L0A_SIZE = 32768U;
constexpr uint32_t L0B_SIZE = 65536U;
constexpr uint32_t L0C_SIZE = 131072U;
constexpr uint32_t BUFFER_NUM = 8U;

template <typename T>
void FillTensor(T* data, size_t count, float value)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(value);
    }
}

template <>
void FillTensor<bfloat16_t>(bfloat16_t* data, size_t count, float value)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = AscendC::ToBfloat16(value);
    }
}

void FillTiling(GroupedMatmulQuantTilingData& data, uint32_t dtype, uint32_t m, uint32_t k, uint32_t n,
                uint32_t groupNum, uint32_t scaleGroupSize, bool noGroup, bool splitK)
{
    std::memset(&data, 0, sizeof(data));
    data.CoreNum = 1U;
    data.dataType = dtype;
    data.UBSize = UB_SIZE;
    data.L1Size = L1_SIZE;
    data.L0ASize = L0A_SIZE;
    data.L0BSize = L0B_SIZE;
    data.L0CSize = L0C_SIZE;
    data.noGroup = static_cast<uint32_t>(noGroup);
    data.originE = groupNum;
    data.originM = m;
    data.originN = n;
    data.originK = k;
    data.scaleK = k / scaleGroupSize;
    data.scaleGroupSize = scaleGroupSize;
    data.fracN = n / 16U;
    data.fracK = k / 16U;
    data.splitK = static_cast<uint32_t>(splitK);

    data.clearBaseN = UB_SIZE / 4U;
    const uint64_t totalCount = static_cast<uint64_t>(m) * n;
    const uint64_t clearStep = 2UL * data.CoreNum * data.clearBaseN;
    data.clearOutLoop = static_cast<uint32_t>(totalCount / clearStep);
    const uint64_t clearTailCount = totalCount % clearStep;
    const uint64_t clearTailRepeats = (clearTailCount + 127UL) / 128UL;
    data.clearOutTailN = static_cast<uint32_t>(clearTailRepeats / (data.CoreNum * 2UL) * 128UL);
    data.clearOutTailCoreNum = static_cast<uint32_t>(clearTailRepeats % (data.CoreNum * 2UL));

    data.castBaseN = UB_SIZE / 6U / 2U;
    const uint64_t castStep = 2UL * data.CoreNum * data.castBaseN;
    data.castOutLoop = static_cast<uint32_t>(totalCount / castStep);
    const uint64_t castTailCount = totalCount % castStep;
    const uint64_t castTailRepeats = (castTailCount + 255UL) / 256UL;
    data.castOutTailN = static_cast<uint32_t>(castTailRepeats / (data.CoreNum * 2UL) * 256UL);
    data.castOutTailCoreNum = static_cast<uint32_t>(castTailRepeats % (data.CoreNum * 2UL));
}

template <typename T>
void RunKernelSmoke(uint32_t dtype, uint32_t m, uint32_t k, uint32_t n, const std::vector<int64_t>& groupList,
                    uint32_t scaleGroupSize, bool splitK, float offsetValue)
{
    const uint32_t groupNum = groupList.empty() ? 1U : static_cast<uint32_t>(groupList.size());
    const bool noGroup = groupList.empty();
    const size_t xCount = static_cast<size_t>(m) * k;
    const size_t packedWeightCount = static_cast<size_t>(groupNum) * k * n / 8U;
    const size_t scaleCount = static_cast<size_t>(groupNum) * (k / scaleGroupSize) * n;
    const size_t yCount = static_cast<size_t>(m) * n;
    const size_t workspaceBytes =
        static_cast<size_t>(L0A_SIZE) * BUFFER_NUM + (splitK ? yCount * sizeof(float) : 0U) + 1024U;

    auto* x = reinterpret_cast<T*>(AscendC::GmAlloc(xCount * sizeof(T)));
    auto* weight = reinterpret_cast<int32_t*>(AscendC::GmAlloc(packedWeightCount * sizeof(int32_t)));
    auto* scale = reinterpret_cast<T*>(AscendC::GmAlloc(scaleCount * sizeof(T)));
    auto* offset = reinterpret_cast<half*>(AscendC::GmAlloc(scaleCount * sizeof(half)));
    auto* groups = reinterpret_cast<int64_t*>(AscendC::GmAlloc(groupNum * sizeof(int64_t)));
    auto* y = reinterpret_cast<T*>(AscendC::GmAlloc(yCount * sizeof(T)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceBytes));
    auto* tiling =
        reinterpret_cast<GroupedMatmulQuantTilingData*>(AscendC::GmAlloc(sizeof(GroupedMatmulQuantTilingData)));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(weight, nullptr);
    ASSERT_NE(scale, nullptr);
    ASSERT_NE(offset, nullptr);
    ASSERT_NE(groups, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    FillTensor(x, xCount, 1.0F);
    std::memset(weight, 0, packedWeightCount * sizeof(int32_t));
    FillTensor(scale, scaleCount, 0.5F);
    FillTensor(offset, scaleCount, offsetValue);
    FillTensor(y, yCount, 0.0F);
    std::memset(workspace, 0, workspaceBytes);
    if (!noGroup) {
        std::memcpy(groups, groupList.data(), groupNum * sizeof(int64_t));
    }
    FillTiling(*tiling, dtype, m, k, n, groupNum, scaleGroupSize, noGroup, splitK);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(10000001UL);
    ICPU_RUN_KF(grouped_matmul_quant, 1, reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(weight),
                reinterpret_cast<uint8_t*>(scale), reinterpret_cast<uint8_t*>(offset),
                noGroup ? nullptr : reinterpret_cast<uint8_t*>(groups), reinterpret_cast<uint8_t*>(y), workspace,
                reinterpret_cast<uint8_t*>(tiling));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(groups);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

TEST_F(GroupedMatmulQuantKernelTest, fp16_single_group_non_split_k)
{
    RunKernelSmoke<half>(1U, 16U, 32U, 16U, {}, 32U, false, 0.0F);
}

TEST_F(GroupedMatmulQuantKernelTest, bf16_multiple_groups_nonzero_offset)
{
    RunKernelSmoke<bfloat16_t>(27U, 32U, 64U, 16U, {0, 32}, 32U, false, 1.0F);
}

TEST_F(GroupedMatmulQuantKernelTest, fp16_scale_group_64_split_k)
{
    RunKernelSmoke<half>(1U, 17U, 64U, 32U, {}, 64U, true, -0.5F);
}
