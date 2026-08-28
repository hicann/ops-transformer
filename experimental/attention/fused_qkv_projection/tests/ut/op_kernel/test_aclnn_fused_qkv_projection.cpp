/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_qkv_projection(GM_ADDR hiddenStates, GM_ADDR weight, GM_ADDR bias,
                                                           GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR workspace,
                                                           GM_ADDR tiling);

class FusedQkvProjectionKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "FusedQkvProjectionKernelTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "FusedQkvProjectionKernelTest TearDown\n" << endl;
    }
};

using DataType = float;

// 基础测试：M=16, hidden=16, fusedDim=32, bias
TEST_F(FusedQkvProjectionKernelTest, test_basic_m16)
{
    int64_t batch = 2;
    int64_t seqLen = 8;
    int64_t hiddenSize = 16;
    int64_t qDim = 16;
    int64_t kDim = 8;
    int64_t vDim = 8;
    int64_t fusedDim = qDim + kDim + vDim; // 32
    int64_t M = batch * seqLen;            // 16

    size_t hsSize = M * hiddenSize * sizeof(DataType);
    size_t wtSize = hiddenSize * fusedDim * sizeof(DataType);
    size_t biasSize = fusedDim * sizeof(DataType);
    size_t qSize = M * qDim * sizeof(DataType);
    size_t kSize = M * kDim * sizeof(DataType);
    size_t vSize = M * vDim * sizeof(DataType);

    uint8_t *hs = (uint8_t *)AscendC::GmAlloc(hsSize);
    uint8_t *wt = (uint8_t *)AscendC::GmAlloc(wtSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasSize);
    uint8_t *q = (uint8_t *)AscendC::GmAlloc(qSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *v = (uint8_t *)AscendC::GmAlloc(vSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(FusedQkvProjectionTilingData));

    // 填入测试数据
    auto *hsData = reinterpret_cast<DataType *>((uintptr_t)hs);
    auto *wtData = reinterpret_cast<DataType *>((uintptr_t)wt);
    auto *biasData = reinterpret_cast<DataType *>((uintptr_t)bias);
    for (int64_t i = 0; i < M * hiddenSize; i++)
        hsData[i] = static_cast<DataType>(sin(i * 0.73f) * 1.5f);
    for (int64_t i = 0; i < hiddenSize * fusedDim; i++)
        wtData[i] = static_cast<DataType>(cos(i * 1.19f));
    for (int64_t i = 0; i < fusedDim; i++)
        biasData[i] = static_cast<DataType>(sin(i * 2.37f) * 0.5f);

    // 配置 tiling 数据
    auto *td = reinterpret_cast<FusedQkvProjectionTilingData *>((uintptr_t)tiling);
    memset(td, 0, sizeof(FusedQkvProjectionTilingData));
    td->M = static_cast<int32_t>(M);
    td->N = static_cast<int32_t>(fusedDim);
    td->K = static_cast<int32_t>(hiddenSize);
    td->singleCoreM = static_cast<int32_t>(M);
    td->singleCoreN = static_cast<int32_t>(fusedDim);
    td->baseM = static_cast<int32_t>(M);
    td->baseN = static_cast<int32_t>(fusedDim);
    td->baseK = static_cast<int32_t>(hiddenSize);
    td->qDim = static_cast<int32_t>(qDim);
    td->kDim = static_cast<int32_t>(kDim);
    td->vDim = static_cast<int32_t>(vDim);
    td->hasBias = true;
    td->blockDim = 1;

    // 调用 kernel
    uint32_t blockDim = 1;
    AscendC::SetSysWorkspace(ws);
    fused_qkv_projection<<<blockDim, nullptr, nullptr>>>(hs, wt, bias, q, k, v, ws, tiling);
    AscendC::Synchronize();

    // 验证 Q 输出
    auto *qData = reinterpret_cast<DataType *>((uintptr_t)q);
    for (int64_t r = 0; r < M; r++) {
        for (int64_t fd = 0; fd < qDim; fd++) {
            float sum = 0.0f;
            for (int64_t hh = 0; hh < hiddenSize; hh++)
                sum += hsData[r * hiddenSize + hh] * wtData[hh * fusedDim + fd];
            sum += biasData[fd];
            EXPECT_NEAR(qData[r * qDim + fd], static_cast<DataType>(sum), 1e-3f)
                << "Q mismatch at r=" << r << " fd=" << fd;
        }
    }

    AscendC::GmFree(hs);
    AscendC::GmFree(wt);
    AscendC::GmFree(bias);
    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    AscendC::GmFree(ws);
    AscendC::GmFree(tiling);
}

// 无 bias 测试
TEST_F(FusedQkvProjectionKernelTest, test_no_bias)
{
    int64_t batch = 1;
    int64_t seqLen = 16;
    int64_t hiddenSize = 64;
    int64_t qDim = 32;
    int64_t kDim = 16;
    int64_t vDim = 16;
    int64_t fusedDim = qDim + kDim + vDim;
    int64_t M = batch * seqLen;

    size_t hsSize = M * hiddenSize * sizeof(DataType);
    size_t wtSize = hiddenSize * fusedDim * sizeof(DataType);
    size_t qSize = M * qDim * sizeof(DataType);
    size_t kSize = M * kDim * sizeof(DataType);
    size_t vSize = M * vDim * sizeof(DataType);

    uint8_t *hs = (uint8_t *)AscendC::GmAlloc(hsSize);
    uint8_t *wt = (uint8_t *)AscendC::GmAlloc(wtSize);
    uint8_t *q = (uint8_t *)AscendC::GmAlloc(qSize);
    uint8_t *k = (uint8_t *)AscendC::GmAlloc(kSize);
    uint8_t *v = (uint8_t *)AscendC::GmAlloc(vSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(FusedQkvProjectionTilingData));

    auto *hsData = reinterpret_cast<DataType *>((uintptr_t)hs);
    auto *wtData = reinterpret_cast<DataType *>((uintptr_t)wt);
    for (int64_t i = 0; i < M * hiddenSize; i++)
        hsData[i] = static_cast<DataType>(sin(i * 0.73f) * 1.5f);
    for (int64_t i = 0; i < hiddenSize * fusedDim; i++)
        wtData[i] = static_cast<DataType>(cos(i * 1.19f));

    auto *td = reinterpret_cast<FusedQkvProjectionTilingData *>((uintptr_t)tiling);
    memset(td, 0, sizeof(FusedQkvProjectionTilingData));
    td->M = static_cast<int32_t>(M);
    td->N = static_cast<int32_t>(fusedDim);
    td->K = static_cast<int32_t>(hiddenSize);
    td->singleCoreM = static_cast<int32_t>(M);
    td->singleCoreN = static_cast<int32_t>(fusedDim);
    td->baseM = static_cast<int32_t>(M);
    td->baseN = static_cast<int32_t>(fusedDim);
    td->baseK = static_cast<int32_t>(hiddenSize);
    td->qDim = static_cast<int32_t>(qDim);
    td->kDim = static_cast<int32_t>(kDim);
    td->vDim = static_cast<int32_t>(vDim);
    td->hasBias = false;
    td->blockDim = 1;

    uint32_t blockDim = 1;
    AscendC::SetSysWorkspace(ws);
    fused_qkv_projection<<<blockDim, nullptr, nullptr>>>(hs, wt, nullptr, q, k, v, ws, tiling);
    AscendC::Synchronize();

    auto *qData = reinterpret_cast<DataType *>((uintptr_t)q);
    for (int64_t r = 0; r < M; r++) {
        for (int64_t fd = 0; fd < qDim; fd++) {
            float sum = 0.0f;
            for (int64_t hh = 0; hh < hiddenSize; hh++)
                sum += hsData[r * hiddenSize + hh] * wtData[hh * fusedDim + fd];
            EXPECT_NEAR(qData[r * qDim + fd], static_cast<DataType>(sum), 1e-3f)
                << "Q mismatch at r=" << r << " fd=" << fd;
        }
    }

    AscendC::GmFree(hs);
    AscendC::GmFree(wt);
    AscendC::GmFree(q);
    AscendC::GmFree(k);
    AscendC::GmFree(v);
    AscendC::GmFree(ws);
    AscendC::GmFree(tiling);
}
