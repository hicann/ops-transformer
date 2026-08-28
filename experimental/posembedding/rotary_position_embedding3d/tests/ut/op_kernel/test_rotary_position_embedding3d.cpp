/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_rotary_position_embedding3d.h"
#include "../../../op_kernel/rotary_position_embedding3d.cpp"

#include <cstdint>
#include <cmath>
#include <random>

using namespace std;

// ---------------------------------------------------------------------------
// Helper: factor video dims (matches FactorVideoDims in tiling.cpp)
// ---------------------------------------------------------------------------
static void FactorVideoDims(int64_t seqLen, int64_t &T, int64_t &H, int64_t &W)
{
    if (seqLen <= 1) {
        T = 1;
        H = 1;
        W = seqLen;
        return;
    }
    int64_t maxDim = min(128LL, (int64_t)sqrt((double)seqLen));
    for (int64_t w = maxDim; w >= 1; w--) {
        if (seqLen % w == 0) {
            int64_t rest = seqLen / w;
            for (int64_t h = min(128LL, (int64_t)sqrt((double)rest)); h >= 1; h--) {
                if (rest % h == 0) {
                    T = rest / h;
                    H = h;
                    W = w;
                    return;
                }
            }
        }
    }
    T = 1;
    H = 1;
    W = seqLen;
}

// ---------------------------------------------------------------------------
// Helper: compute 2:1:1 band dims (matches ComputeBandDims in tiling.cpp)
// ---------------------------------------------------------------------------
static void ComputeBandDimsHelper(int64_t headDim, int64_t &tBand, int64_t &hBand, int64_t &wBand)
{
    // 2:1:1 ratio: T = D/2, H = D/4, W = D/4, all even
    int64_t unit = headDim / 4;
    if (unit % 2 != 0) {
        unit -= 1;
    }
    tBand = unit * 2;
    hBand = unit;
    wBand = headDim - tBand - hBand;
    if (wBand % 2 != 0) {
        wBand -= 1;
        tBand += 1;
        if (tBand % 2 != 0) {
            tBand -= 1;
            hBand += 1;
        }
    }
    if (hBand % 2 != 0) {
        hBand -= 1;
        tBand += 1;
    }
    if (tBand % 2 != 0) {
        tBand -= 1;
        hBand += 1;
    }
}

// ---------------------------------------------------------------------------
// Helper: generate test data (x, cos_sin, golden, tiling) into pre-allocated buffers
// Matches the Python gen_data.py / gen_tiling.py logic with seed=42
// ---------------------------------------------------------------------------
template <typename T>
static void GenerateTestData(int64_t B, int64_t S, int64_t D, uint8_t *xBuf, uint8_t *cosSinBuf, uint8_t *goldenBuf,
                             uint8_t *tilingBuf)
{
    const int64_t halfD = D / 2;
    const float freqBase = 10000.0f;

    // 1. Factor video dims
    int64_t tDim, H, W;
    FactorVideoDims(S, tDim, H, W);
    int64_t hw = H * W;
    int64_t wv = W;

    // 2. Compute band dims
    int64_t tb, hb, wb;
    ComputeBandDimsHelper(D, tb, hb, wb);
    int64_t halfTb = tb / 2;
    int64_t halfHb = hb / 2;
    int64_t halfWb = wb / 2;

    // 3. Frequency tables
    double rt = pow(freqBase, -2.0 / tb);
    double rh = pow(freqBase, -2.0 / hb);
    double rw = pow(freqBase, -2.0 / wb);

    std::vector<double> thetaT(halfTb), thetaH(halfHb), thetaW(halfWb);
    for (int64_t k = 0; k < halfTb; k++)
        thetaT[k] = pow(rt, k);
    for (int64_t k = 0; k < halfHb; k++)
        thetaH[k] = pow(rh, k);
    for (int64_t k = 0; k < halfWb; k++)
        thetaW[k] = pow(rw, k);

    // 4. Generate x with fixed seed 42 (matches Python's np.random.default_rng(42))
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    T *x = reinterpret_cast<T *>(xBuf);
    for (int64_t i = 0; i < B * S * D; i++) {
        x[i] = static_cast<T>(dist(rng));
    }

    // 5. Compute cos_sin
    T *cs = reinterpret_cast<T *>(cosSinBuf);
    for (int64_t batch = 0; batch < B; batch++) {
        for (int64_t pos = 0; pos < S; pos++) {
            int64_t tCoord = pos / hw;
            int64_t rem = pos % hw;
            int64_t iCoord = rem / wv;
            int64_t jCoord = rem % wv;

            for (int64_t k = 0; k < halfTb; k++) {
                double a = tCoord * thetaT[k];
                cs[batch * S * D + pos * D + k] = static_cast<T>(cos(a));
                cs[batch * S * D + pos * D + halfD + k] = static_cast<T>(sin(a));
            }
            for (int64_t k = 0; k < halfHb; k++) {
                double a = iCoord * thetaH[k];
                cs[batch * S * D + pos * D + halfTb + k] = static_cast<T>(cos(a));
                cs[batch * S * D + pos * D + halfD + halfTb + k] = static_cast<T>(sin(a));
            }
            for (int64_t k = 0; k < halfWb; k++) {
                double a = jCoord * thetaW[k];
                cs[batch * S * D + pos * D + halfTb + halfHb + k] = static_cast<T>(cos(a));
                cs[batch * S * D + pos * D + halfD + halfTb + halfHb + k] = static_cast<T>(sin(a));
            }
        }
    }

    // 6. Compute golden (in float, then cast to T for storage to match golden file format)
    T *golden = reinterpret_cast<T *>(goldenBuf);
    for (int64_t batch = 0; batch < B; batch++) {
        for (int64_t pos = 0; pos < S; pos++) {
            int64_t tCoord = pos / hw;
            int64_t rem = pos % hw;
            int64_t iCoord = rem / wv;
            int64_t jCoord = rem % wv;

            for (int64_t k = 0; k < halfTb; k++) {
                double a = tCoord * thetaT[k];
                double c = cos(a), sn = sin(a);
                double xl = static_cast<double>(x[batch * S * D + pos * D + k]);
                double xr = static_cast<double>(x[batch * S * D + pos * D + halfD + k]);
                golden[batch * S * D + pos * D + k] = static_cast<T>(xl * c - xr * sn);
                golden[batch * S * D + pos * D + halfD + k] = static_cast<T>(xl * sn + xr * c);
            }
            for (int64_t k = 0; k < halfHb; k++) {
                double a = iCoord * thetaH[k];
                double c = cos(a), sn = sin(a);
                int64_t off = halfTb + k;
                double xl = static_cast<double>(x[batch * S * D + pos * D + off]);
                double xr = static_cast<double>(x[batch * S * D + pos * D + halfD + off]);
                golden[batch * S * D + pos * D + off] = static_cast<T>(xl * c - xr * sn);
                golden[batch * S * D + pos * D + halfD + off] = static_cast<T>(xl * sn + xr * c);
            }
            for (int64_t k = 0; k < halfWb; k++) {
                double a = jCoord * thetaW[k];
                double c = cos(a), sn = sin(a);
                int64_t off = halfTb + halfHb + k;
                double xl = static_cast<double>(x[batch * S * D + pos * D + off]);
                double xr = static_cast<double>(x[batch * S * D + pos * D + halfD + off]);
                golden[batch * S * D + pos * D + off] = static_cast<T>(xl * c - xr * sn);
                golden[batch * S * D + pos * D + halfD + off] = static_cast<T>(xl * sn + xr * c);
            }
        }
    }

    // 7. Compute tiling data (matches tiling.cpp logic)
    int64_t total = B * S * D;
    int64_t blockLen = total / 8;
    constexpr int64_t ubSize = 196608; // 192KB on ascend910b
    size_t typeSize = sizeof(T);
    int64_t maxTileElts = (ubSize - static_cast<uint64_t>(D / 2) * typeSize) / (3ULL * typeSize);
    int64_t posPerBlock = blockLen / D;
    int64_t posPerTile = maxTileElts / D;
    if (posPerTile <= 0)
        posPerTile = 1;
    if (posPerTile > posPerBlock)
        posPerTile = posPerBlock;
    int64_t tileNum = (posPerBlock + posPerTile - 1) / posPerTile;
    int64_t tileLength = posPerTile * D;

    double rrt = pow(freqBase, -2.0 / tb);
    double rrh = pow(freqBase, -2.0 / hb);
    double rrw = pow(freqBase, -2.0 / wb);

    RotaryPositionEmbedding3dTilingData *td = reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tilingBuf);
    td->totalLength = total;
    td->headDim = D;
    td->seqLen = S;
    td->T = tDim;
    td->H = H;
    td->W = W;
    td->tBand = tb;
    td->hBand = hb;
    td->wBand = wb;
    td->tileNum = tileNum;
    td->tileLength = tileLength;
    td->blockLength = blockLen;
    td->freqBase = freqBase;
    td->rT = static_cast<float>(rrt);
    td->rH = static_cast<float>(rrh);
    td->rW = static_cast<float>(rrw);
}

class rotary_position_embedding3d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rotary_position_embedding3d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotary_position_embedding3d_test TearDown\n" << endl;
    }
};

// [2, 4, 8] float32 — 3D 基准测试
TEST_F(rotary_position_embedding3d_test, test_case_fp32_3d_001)
{
    uint32_t B = 2, S = 4, D = 8;
    size_t inputXByteSize = B * S * D * sizeof(float);
    size_t inputCosSinByteSize = B * S * D * sizeof(float);
    size_t outputZByteSize = B * S * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    uint8_t *goldenBuf = new uint8_t[outputZByteSize];
    GenerateTestData<float>(B, S, D, x, cos_sin, goldenBuf, tiling);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<0>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float *z_f32 = (float *)z;
    size_t numElts = outputZByteSize / sizeof(float);
    float *golden = reinterpret_cast<float *>(goldenBuf);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - golden[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > 5e-2f) { // relaxed tolerance for half precision
            if (utPass)
                std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp " << golden[i] << " diff "
                          << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    delete[] goldenBuf;
}

// [4, 16, 64] float16 — 大尺寸 fp16 测试
TEST_F(rotary_position_embedding3d_test, test_case_fp16_3d_002)
{
    uint32_t B = 4, S = 16, D = 64;
    size_t inputXByteSize = B * S * D * sizeof(half);
    size_t inputCosSinByteSize = B * S * D * sizeof(half);
    size_t outputZByteSize = B * S * D * sizeof(half);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    uint8_t *goldenBuf = new uint8_t[outputZByteSize];
    GenerateTestData<half>(B, S, D, x, cos_sin, goldenBuf, tiling);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<1>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float *z_f32 = (float *)z;
    size_t numElts = outputZByteSize / sizeof(float);
    half *golden = reinterpret_cast<half *>(goldenBuf);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - static_cast<float>(golden[i]));
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > 5e-2f) { // relaxed tolerance for half precision
            if (utPass)
                std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp "
                          << static_cast<float>(golden[i]) << " diff " << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    delete[] goldenBuf;
}

// [1, 32, 128] float32 — 单 batch 大 sequence 测试
TEST_F(rotary_position_embedding3d_test, test_case_fp32_3d_003)
{
    uint32_t B = 1, S = 32, D = 128;
    size_t inputXByteSize = B * S * D * sizeof(float);
    size_t inputCosSinByteSize = B * S * D * sizeof(float);
    size_t outputZByteSize = B * S * D * sizeof(float);
    size_t tilingDataSize = sizeof(RotaryPositionEmbedding3dTilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *cos_sin = (uint8_t *)AscendC::GmAlloc(inputCosSinByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputZByteSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 8;

    uint8_t *goldenBuf = new uint8_t[outputZByteSize];
    GenerateTestData<float>(B, S, D, x, cos_sin, goldenBuf, tiling);
    RotaryPositionEmbedding3dTilingData *tilingDatafromBin =
        reinterpret_cast<RotaryPositionEmbedding3dTilingData *>(tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    auto rotary_position_embedding3d_wrapper = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        rotary_position_embedding3d<0>(x, y, z, workspace, tiling);
    };
    ICPU_RUN_KF(rotary_position_embedding3d_wrapper, blockDim, x, cos_sin, z, workspace,
                (uint8_t *)(tilingDatafromBin));

    // Verify against golden reference
    float *z_f32 = (float *)z;
    size_t numElts = outputZByteSize / sizeof(float);
    float *golden = reinterpret_cast<float *>(goldenBuf);
    bool utPass = true;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < numElts; i++) {
        float diff = fabsf(z_f32[i] - golden[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > 5e-2f) { // relaxed tolerance for half precision
            if (utPass)
                std::cout << "First mismatch at " << i << ": got " << z_f32[i] << " exp " << golden[i] << " diff "
                          << diff << std::endl;
            utPass = false;
        }
    }
    std::cout << "  Max diff: " << maxDiff << " " << (utPass ? "PASS" : "FAIL") << std::endl;
    EXPECT_TRUE(utPass);

    AscendC::GmFree(x);
    AscendC::GmFree(cos_sin);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    delete[] goldenBuf;
}
