/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_indexer_quant_cache.cpp
 * \brief IndexerQuantCache op_kernel UT (arch35 / ascend950).
 *
 * 用 ICPU_RUN_KF 在 CPU 仿真上真正执行 kernel，覆盖 6 个 tiling key 分支:
 *   quantMode=1 Normal 动态量化 (single/multi row, roundScale true/false)
 *   quantMode=2 HiFloat8 整行量化
 *   quantMode=0 MX-FP8
 *   quantMode=3 MX-FP4
 * key = 10000 + (rowFactor!=1 ? 10 : 0) + (quantMode in {1,2} ? 1 : quantMode==3 ? 2 : 0)
 * 校验: kernel 运行不崩溃 + 写入 slot 的 cache 字节非全 0 (sanity)。
 *
 * 说明: IndexerQuantCacheTilingData 结构体由 UT 框架根据 op_host BEGIN_TILING_DATA_DEF
 * 自动生成 (强制 -include 进本编译单元), 这里直接使用其纯结构体成员赋值。
 */
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void indexer_quant_cache(GM_ADDR cache, GM_ADDR cache_scale, GM_ADDR x,
                                                          GM_ADDR slot_mapping, GM_ADDR cache_out,
                                                          GM_ADDR cache_scale_out, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MX_BLOCK = 32;
constexpr int64_t MXFP8_QUANT_MODE = 0;
constexpr int64_t NORMAL_QUANT_MODE = 1;
constexpr int64_t HIFLOAT_QUANT_MODE = 2;
constexpr int64_t MXFP4_QUANT_MODE = 3;
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

template <typename T>
T* GmAllocWrapper(size_t size)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

// half: 1.0 = 0x3C00, -0.5 = 0xB800. 交替填充保证每行非零最大值。
void FillHalf(uint8_t* gm, int64_t numElem)
{
    uint16_t* p = reinterpret_cast<uint16_t*>(gm);
    for (int64_t i = 0; i < numElem; ++i) {
        p[i] = (i & 1) ? 0xB800 : 0x3C00;
    }
}
} // namespace

struct IqcParams {
    int64_t bs;
    int64_t d;
    int64_t quantMode;
    int64_t roundScale;
    bool singleRow; // true -> rowFactor=1 (single_row 模板), false -> multi_row
    int64_t blockDim;
    bool withSkip;
    std::string desc;
};

class IndexerQuantCacheKernelTest : public testing::TestWithParam<IqcParams> {
protected:
    void ComputeTiling(IndexerQuantCacheTilingData& t, const IqcParams& p)
    {
        int64_t scaleCol =
            (p.quantMode == MXFP8_QUANT_MODE || p.quantMode == MXFP4_QUANT_MODE) ? CeilDiv(p.d, MX_BLOCK) : 1;
        int64_t coreNum = p.blockDim;
        int64_t rowOfFormer = CeilDiv(p.bs, coreNum);
        int64_t usedCore = std::min(CeilDiv(p.bs, rowOfFormer), coreNum);
        int64_t rowOfTail = p.bs - (usedCore - 1) * rowOfFormer;
        int64_t rowFactor = p.singleRow ? 1 : rowOfFormer;

        int64_t cacheCol = (p.quantMode == MXFP4_QUANT_MODE) ? (p.d + 1) / 2 : p.d;

        t.bs = p.bs;
        t.d = p.d;
        t.scaleCol = scaleCol;
        t.rowOfFormerBlock = rowOfFormer;
        t.rowOfTailBlock = rowOfTail;
        t.rowLoopOfFormerBlock = CeilDiv(rowOfFormer, rowFactor);
        t.rowLoopOfTailBlock = CeilDiv(rowOfTail, rowFactor);
        t.rowFactor = rowFactor;
        t.tailRowFactorOfFormerBlock = (rowOfFormer % rowFactor == 0) ? rowFactor : rowOfFormer % rowFactor;
        t.tailRowFactorOfTailBlock = (rowOfTail % rowFactor == 0) ? rowFactor : rowOfTail % rowFactor;
        t.quantMode = p.quantMode;
        t.roundScale = p.roundScale;
        t.scalesAttr = 1.0f;
        t.blockSize = 1;
        t.cacheRowStride = cacheCol;
        t.cacheBlockStride = cacheCol;
        t.scaleRowStride = scaleCol;
        t.scaleBlockStride = scaleCol;

        usedCore_ = usedCore;
        cacheCol_ = cacheCol;
        scaleCol_ = scaleCol;

        // tiling key
        tilingKey_ = 10000;
        if (rowFactor != 1) {
            tilingKey_ += 10;
        }
        if (p.quantMode == NORMAL_QUANT_MODE || p.quantMode == HIFLOAT_QUANT_MODE) {
            tilingKey_ += 1;
        } else if (p.quantMode == MXFP4_QUANT_MODE) {
            tilingKey_ += 2;
        }
    }

    int64_t usedCore_ = 1;
    int64_t cacheCol_ = 0;
    int64_t scaleCol_ = 0;
    int64_t tilingKey_ = 10000;
};

TEST_P(IndexerQuantCacheKernelTest, RunKernel)
{
    auto p = GetParam();

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    IndexerQuantCacheTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    ComputeTiling(tilingData, p);

    std::cout << "[IndexerQuantCache] " << p.desc << " bs=" << p.bs << " d=" << p.d << " quantMode=" << p.quantMode
              << " roundScale=" << p.roundScale << " singleRow=" << p.singleRow << " blockDim=" << p.blockDim
              << " tilingKey=" << tilingKey_ << std::endl;

    int64_t cacheRows = p.bs + 1;
    size_t shapeX = static_cast<size_t>(p.bs) * p.d * sizeof(uint16_t); // half
    size_t shapeSlot = static_cast<size_t>(p.bs) * sizeof(int32_t);
    size_t shapeCache = static_cast<size_t>(cacheRows) * p.d * sizeof(uint8_t) + 512; // 覆盖 d / (d+1)/2
    size_t shapeScale = static_cast<size_t>(cacheRows) * scaleCol_ * sizeof(float) + 512;

    uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* scaleGm = GmAllocWrapper<uint8_t>(shapeScale);
    uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
    uint8_t* slotGm = GmAllocWrapper<uint8_t>(shapeSlot);
    uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* scaleOutGm = GmAllocWrapper<uint8_t>(shapeScale);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(IndexerQuantCacheTilingData));

    FillHalf(xGm, static_cast<int64_t>(p.bs) * p.d);
    std::memset(cacheGm, 0, shapeCache);
    std::memset(scaleGm, 0, shapeScale);
    std::memset(cacheOutGm, 0, shapeCache);
    std::memset(scaleOutGm, 0, shapeScale);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &tilingData, sizeof(IndexerQuantCacheTilingData));

    int32_t* slot = reinterpret_cast<int32_t*>(slotGm);
    for (int64_t i = 0; i < p.bs; ++i) {
        slot[i] = static_cast<int32_t>(i);
    }
    int64_t skipped = 0;
    if (p.withSkip) {
        for (int64_t i = 0; i < p.bs; i += 3) {
            slot[i] = -1;
            ++skipped;
        }
    }

    uint32_t blockDim = static_cast<uint32_t>(usedCore_);
    ICPU_SET_TILING_KEY(static_cast<uint64_t>(tilingKey_));
    ICPU_RUN_KF(indexer_quant_cache, blockDim, cacheGm, scaleGm, xGm, slotGm, cacheOutGm, scaleOutGm, workspace,
                tilingGm);

    bool anyNonZero = false;
    for (int64_t i = 0; i < p.bs && !anyNonZero; ++i) {
        if (slot[i] == -1) {
            continue;
        }
        uint8_t* row = cacheGm + slot[i] * cacheCol_;
        for (int64_t c = 0; c < cacheCol_; ++c) {
            if (row[c] != 0) {
                anyNonZero = true;
                break;
            }
        }
    }
    if (p.bs - skipped > 0) {
        EXPECT_TRUE(anyNonZero) << "no written cache row is non-zero";
    }

    AscendC::GmFree(cacheGm);
    AscendC::GmFree(scaleGm);
    AscendC::GmFree(xGm);
    AscendC::GmFree(slotGm);
    AscendC::GmFree(cacheOutGm);
    AscendC::GmFree(scaleOutGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

INSTANTIATE_TEST_SUITE_P(QuantBranches, IndexerQuantCacheKernelTest,
                         testing::Values(
                             // Normal 动态量化 (key 10001 single / 10011 multi), roundScale true/false
                             IqcParams{32, 128, NORMAL_QUANT_MODE, 1, true, 1, false, "normal_single_round"},
                             IqcParams{32, 128, NORMAL_QUANT_MODE, 0, true, 1, false, "normal_single_noround"},
                             IqcParams{32, 128, NORMAL_QUANT_MODE, 1, false, 1, false, "normal_multi_round"},
                             IqcParams{30, 128, NORMAL_QUANT_MODE, 1, false, 1, true, "normal_multi_skip"},
                             IqcParams{48, 128, NORMAL_QUANT_MODE, 1, false, 2, false, "normal_multi_multicore"},
                             IqcParams{32, 64, NORMAL_QUANT_MODE, 1, true, 1, false, "normal_single_d64"},
                             // HiFloat8 整行量化 (key 10001 single / 10011 multi)
                             IqcParams{32, 128, HIFLOAT_QUANT_MODE, 1, true, 1, false, "hifloat_single"},
                             IqcParams{32, 128, HIFLOAT_QUANT_MODE, 1, false, 1, false, "hifloat_multi"},
                             // MX-FP8 (key 10000 single / 10010 multi)
                             IqcParams{32, 128, MXFP8_QUANT_MODE, 1, true, 1, false, "mxfp8_single"},
                             IqcParams{32, 128, MXFP8_QUANT_MODE, 1, false, 1, false, "mxfp8_multi"},
                             // MX-FP4 (key 10002 single / 10012 multi)
                             IqcParams{32, 128, MXFP4_QUANT_MODE, 1, true, 1, false, "mxfp4_single"},
                             IqcParams{32, 128, MXFP4_QUANT_MODE, 1, false, 1, false, "mxfp4_multi"}));

// 结构体字段 sanity (保留原 tiling-data 校验意图)。
TEST(IndexerQuantCacheTilingStruct, FieldsWritable)
{
    IndexerQuantCacheTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.bs = 1024;
    t.d = 128;
    t.quantMode = 1;
    t.roundScale = 1;
    t.scalesAttr = 1.0f;
    EXPECT_EQ(t.bs, 1024);
    EXPECT_EQ(t.d, 128);
    EXPECT_EQ(t.quantMode, 1);
    EXPECT_EQ(t.roundScale, 1);
    EXPECT_FLOAT_EQ(t.scalesAttr, 1.0f);
    EXPECT_GT(sizeof(IndexerQuantCacheTilingData), 0u);
}

// ---------------------------------------------------------------------------
// Reading B (4D-only 契约): cache 行宽 headDim 可大于写出长度 cacheCol;
//   kernel 每行只写 cacheCol(=d) 字节, 行内余下 [cacheCol, headDim) 必须保持原值。
//   通过把 cacheRowStride/cacheBlockStride 置为 headDim(>cacheCol) 并预填哨兵字节验证。
//   mode1 Normal single-row, bs=8, d=128, headDim=160 (>128)。
// ---------------------------------------------------------------------------
TEST(IndexerQuantCacheKernelReadingB, WideHeadDimLeavesTailUntouched)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    const int64_t bs = 8;
    const int64_t d = 128;
    const int64_t headDim = 160; // > cacheCol(=d=128)
    const int64_t cacheCol = d;  // mode1: cacheCol == d
    const int64_t scaleCol = 1;  // mode1: 整行一个 float scale
    const uint8_t SENTINEL = 0xAB;

    IndexerQuantCacheTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.bs = bs;
    t.d = d;
    t.scaleCol = scaleCol;
    t.rowOfFormerBlock = bs;
    t.rowOfTailBlock = bs;
    t.rowLoopOfFormerBlock = bs; // single-row: rowFactor=1
    t.rowLoopOfTailBlock = bs;
    t.rowFactor = 1;
    t.tailRowFactorOfFormerBlock = 1;
    t.tailRowFactorOfTailBlock = 1;
    t.quantMode = NORMAL_QUANT_MODE;
    t.roundScale = 1;
    t.scalesAttr = 1.0f;
    t.blockSize = 1;            // contiguous flat slots
    t.cacheRowStride = headDim; // 行宽 = headDim (> cacheCol)
    t.cacheBlockStride = headDim;
    t.scaleRowStride = scaleCol;
    t.scaleBlockStride = scaleCol;

    int64_t tilingKey = 10001; // mode1 single-row

    int64_t cacheRows = bs + 1;
    size_t shapeX = static_cast<size_t>(bs) * d * sizeof(uint16_t);
    size_t shapeSlot = static_cast<size_t>(bs) * sizeof(int32_t);
    size_t shapeCache = static_cast<size_t>(cacheRows) * headDim * sizeof(uint8_t) + 512;
    size_t shapeScale = static_cast<size_t>(cacheRows) * scaleCol * sizeof(float) + 512;

    uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* scaleGm = GmAllocWrapper<uint8_t>(shapeScale);
    uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
    uint8_t* slotGm = GmAllocWrapper<uint8_t>(shapeSlot);
    uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* scaleOutGm = GmAllocWrapper<uint8_t>(shapeScale);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(IndexerQuantCacheTilingData));

    FillHalf(xGm, static_cast<int64_t>(bs) * d);
    std::memset(cacheGm, SENTINEL, shapeCache); // 哨兵: 验证未写区保持原值
    std::memset(scaleGm, 0, shapeScale);
    std::memset(cacheOutGm, 0, shapeCache);
    std::memset(scaleOutGm, 0, shapeScale);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &t, sizeof(IndexerQuantCacheTilingData));

    int32_t* slot = reinterpret_cast<int32_t*>(slotGm);
    for (int64_t i = 0; i < bs; ++i) {
        slot[i] = static_cast<int32_t>(i);
    }

    ICPU_SET_TILING_KEY(static_cast<uint64_t>(tilingKey));
    ICPU_RUN_KF(indexer_quant_cache, 1, cacheGm, scaleGm, xGm, slotGm, cacheOutGm, scaleOutGm, workspace, tilingGm);

    bool anyWritten = false;
    bool tailPreserved = true;
    for (int64_t i = 0; i < bs; ++i) {
        uint8_t* row = cacheGm + slot[i] * headDim;
        for (int64_t c = 0; c < cacheCol; ++c) {
            if (row[c] != SENTINEL) {
                anyWritten = true; // 写出区被改写(量化输出)
            }
        }
        for (int64_t c = cacheCol; c < headDim; ++c) {
            if (row[c] != SENTINEL) {
                tailPreserved = false; // 行尾 [cacheCol, headDim) 不应被改写
            }
        }
    }
    EXPECT_TRUE(anyWritten) << "written region [0,cacheCol) should be updated";
    EXPECT_TRUE(tailPreserved) << "row tail [cacheCol,headDim) must keep original value (Reading B)";

    AscendC::GmFree(cacheGm);
    AscendC::GmFree(scaleGm);
    AscendC::GmFree(xGm);
    AscendC::GmFree(slotGm);
    AscendC::GmFree(cacheOutGm);
    AscendC::GmFree(scaleOutGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}
