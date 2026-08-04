/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * NOTE: This is a placeholder test file.
 *
 * The quant_block_sparse_attn kernel is NOT feasible for CPU simulation (tikicpulib)
 * for the following reasons:
 *
 * 1. AIC+AIV coordination: The kernel uses KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2),
 *    which requires simultaneous execution of AIC (Cube) and AIV (Vector) cores.
 *    tikicpulib does not support this mixed-core execution model.
 *
 * 2. Cross-core synchronization: The kernel uses CrossCoreWaitFlag, SetCrossCore,
 *    and SyncAll<false>() for AIC<->AIV communication, which are not available
 *    in CPU simulation.
 *
 * 3. g_coreType dispatch: The kernel selects between QBSABlockCube (real on AIC,
 *    dummy on AIV) and QBSABlockVec (dummy on AIC, real on AIV) based on g_coreType.
 *    CPU simulation cannot emulate both core types simultaneously.
 *
 * 4. RegBase kernel: Uses regbaseutil namespace, ARGS_TRAITS, CHILD_SPEC_TEMPLATE_ARGS
 *    and complex template traits that are not compatible with CPU simulation.
 *
 * 5. Complex tiling data: QuantBlockSparseAttnTilingData has 7 nested structures
 *    (attrParams, paParams, sparseParams, inputParamsRegbase, multiCoreParamsRegbase,
 *    dropmaskParamsRegbase, initOutputParams) with array fields, making manual
 *    construction extremely difficult.
 *
 * 6. GM metadata dependency: ProcessMainLoop reads task scheduling metadata from
 *    GM via GetBsaAttrMetadata(metadataGm, ...), which is prepared by the host
 *    tiling implementation and cannot be easily simulated.
 */

#include <cstdint>
#include <iostream>

#include "gtest/gtest.h"

class quant_block_sparse_attn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "quant_block_sparse_attn_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "quant_block_sparse_attn_test TearDown" << std::endl;
    }
};

TEST_F(quant_block_sparse_attn_test, quant_block_sparse_attn_not_feasible_for_cpu_simulation)
{
    std::cout << "quant_block_sparse_attn kernel uses AIC+AIV coordination "
                 "(KERNEL_TYPE_MIX_AIC_1_2) and RegBase patterns, "
                 "which are not supported by tikicpulib CPU simulation."
              << std::endl;
    SUCCEED();
}
