/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#define private public
#include "../../../op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.h"
#undef private

namespace aicpu {
namespace {

TEST(MixedQuantSparseFlashMlaMetadata, UsesTurboQuantS2BaseSizeOnAscend910)
{
    MixedQuantSparseFlashMlaMetadataCpuKernel kernel;
    kernel.numHeadsQ_ = 128;
    kernel.numHeadsKv_ = 1;
    kernel.quantMode_ = 3;
    kernel.socVersion_ = "Ascend910B";

    ASSERT_TRUE(kernel.ParamsInit());
    EXPECT_EQ(kernel.s2BaseSize_, 512U);
}

TEST(MixedQuantSparseFlashMlaMetadata, PreservesA5S2BaseSize)
{
    for (int32_t quantMode : {1, 2}) {
        MixedQuantSparseFlashMlaMetadataCpuKernel kernel;
        kernel.numHeadsQ_ = 128;
        kernel.numHeadsKv_ = 1;
        kernel.quantMode_ = quantMode;
        kernel.socVersion_ = "Ascend950";

        ASSERT_TRUE(kernel.ParamsInit());
        EXPECT_EQ(kernel.s2BaseSize_, 128U);
    }
}

} // namespace
} // namespace aicpu
