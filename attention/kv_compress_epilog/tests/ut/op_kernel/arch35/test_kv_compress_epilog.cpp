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
 * \file test_kv_compress_epilog.cpp
 * \brief KvCompressEpilog op_kernel UT (arch35 / ascend950).
 *
 * 用 ICPU_RUN_KF 在 CPU 仿真上真正执行 kernel，覆盖两种量化模式:
 *   quantMode=0  group quant + bf16 scale (roundScale true/false)
 *   quantMode=1  group quant + e8m0 scale
 * 校验: kernel 运行不崩溃 + 写入 slot 的 cache 字节非全 0 (sanity)。
 *
 * 说明: KvCompressEpilogTilingData 结构体由 UT 框架根据 op_host BEGIN_TILING_DATA_DEF
 * 自动生成 (强制 -include 进本编译单元), 因此这里直接使用其纯结构体成员赋值, 不再单独定义。
 */
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void kv_compress_epilog(GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping,
                                                         GM_ADDR cache_out, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t SLICE_SIZE = 64;
constexpr int64_t QUANT_GROUP = 128;
constexpr int64_t QUANT_MODE_GROUP_BF16 = 0;
constexpr int64_t QUANT_MODE_GROUP_E8M0 = 1;
constexpr int64_t QUANT_MODE_HIF8_FP4 = 2; // rope hifloat8 + nope FLOAT4_E2M1 (bf16 scale)
constexpr int64_t KV_CACHE_ROW_ALIGN = 128;
constexpr int64_t ROPE_HIF8_BYTES = 64;
constexpr int64_t FP4_SCALE_BYTES = 2;
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0 ? a : (a + b - 1) / b;
}
inline int64_t RoundUp(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
T* GmAllocWrapper(size_t size)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

// 把 bf16 GM 用 1.0 (0x3F80) 与 -0.5 (0xBF00) 交替填充, 保证每行有非零最大值。
void FillBf16(uint8_t* gm, int64_t numElem)
{
    uint16_t* p = reinterpret_cast<uint16_t*>(gm);
    for (int64_t i = 0; i < numElem; ++i) {
        p[i] = (i & 1) ? 0xBF00 /*-0.5*/ : 0x3F80 /*1.0*/;
    }
}
} // namespace

struct KvParams {
    int64_t bs;
    int64_t d;
    int64_t quantMode;
    int64_t roundScale;
    int64_t blockDim;
    bool withSkip; // 部分 slot 置 -1, 验证 skip 分支
    std::string desc;
    int64_t quantGroup = QUANT_GROUP; // mode2 下取 16/32/64; mode0/1 沿用默认 128
    // >0 时强制 rowFactor (每核 UB tile 行数), 用于构造 rowLoopOfFormerBlock>1 的多 tile 流水路径。
    // 默认 0 = 沿用 rowOfFormer(单 tile)。
    int64_t rowFactorOverride = 0;
    // true 时按"每行 rope 段写入行号 marker"填充 x, 并逐 slot 校验 cache rope 字节 == 该 slot 对应 x 行的 marker,
    // 直接验证多 tile 下 x 行↔slot 配对正确(rope 段在 mode0/1 是 bf16 逐字节拷贝, 不受量化影响)。
    bool verifyRope = false;
    // >0 时为 cache 行宽 headDim(=kvCacheRowStride), 模拟 4D cache [.,.,1,headDim] 的行内 pad: 算子每行只写
    // kvCacheCol 字节, 行步长为 headDim(>=kvCacheCol)。默认 0 = headDim==kvCacheCol(行紧密排布)。
    int64_t headDim = 0;
};

class KvCompressEpilogKernelTest : public testing::TestWithParam<KvParams> {
protected:
    void ComputeTiling(KvCompressEpilogTilingData& t, const KvParams& p)
    {
        int64_t scaleCol;
        int64_t concatCol;
        int64_t kvCacheCol;
        int64_t padCol;
        int64_t perGroup;
        if (p.quantMode == QUANT_MODE_HIF8_FP4) {
            // mode2 行布局: [rope hifloat8 64B][nope FLOAT4_E2M1 (d-64)/2 B][nope bf16 scale nGroup*2 B][pad]
            perGroup = p.quantGroup;
            scaleCol = (p.d - SLICE_SIZE) / perGroup; // nGroup
            concatCol = ROPE_HIF8_BYTES + (p.d - SLICE_SIZE) / 2 + scaleCol * FP4_SCALE_BYTES;
            kvCacheCol = RoundUp(concatCol, KV_CACHE_ROW_ALIGN);
            padCol = kvCacheCol - concatCol;
        } else {
            // mode0/1 行布局: [rope bf16 128B][nope fp8 (d-64)B][scale]
            perGroup = QUANT_GROUP;
            scaleCol = CeilDiv(p.d - SLICE_SIZE, SLICE_SIZE);
            int64_t scaleBytes = (p.quantMode == QUANT_MODE_GROUP_BF16) ? 2 : 1;
            concatCol = p.d - SLICE_SIZE + SLICE_SIZE * 2 + scaleCol * scaleBytes;
            kvCacheCol = RoundUp(concatCol, QUANT_GROUP);
            padCol = kvCacheCol - concatCol;
        }
        int64_t coreNum = p.blockDim;
        int64_t rowOfFormer = CeilDiv(p.bs, coreNum);
        int64_t usedCore = std::min(CeilDiv(p.bs, rowOfFormer), coreNum);
        int64_t rowOfTail = p.bs - (usedCore - 1) * rowOfFormer;
        // 默认一次处理整块(单 tile); rowFactorOverride>0 时强制更小的 rowFactor 触发多 tile 流水。
        int64_t rowFactor = (p.rowFactorOverride > 0) ? p.rowFactorOverride : rowOfFormer;

        // 4D cache 契约: 行宽 headDim(=rowStride) 须 >= kvCacheCol; headDim>kvCacheCol 时算子每行
        // 只写 kvCacheCol 字节, 行内余下 headDim-kvCacheCol 字节保持原值。默认 headDim==kvCacheCol。
        int64_t rowStride = (p.headDim > 0) ? p.headDim : kvCacheCol;
        t.bs = p.bs;
        t.d = p.d;
        t.kvCacheCol = kvCacheCol;
        t.kvCacheRowStride = rowStride;
        t.kvCacheBlockSize = 1;
        t.kvCacheBlockStride = rowStride;
        t.scaleCol = scaleCol;
        t.concatCol = concatCol;
        t.padCol = padCol;
        t.quantMode = p.quantMode;
        t.roundScale = p.roundScale;
        t.perGroupSize = perGroup;
        t.rowOfFormerBlock = rowOfFormer;
        t.rowOfTailBlock = rowOfTail;
        t.rowLoopOfFormerBlock = CeilDiv(rowOfFormer, rowFactor);
        t.rowLoopOfTailBlock = CeilDiv(rowOfTail, rowFactor);
        t.rowFactor = rowFactor;
        t.tailRowFactorOfFormerBlock = (rowOfFormer % rowFactor == 0) ? rowFactor : rowOfFormer % rowFactor;
        t.tailRowFactorOfTailBlock = (rowOfTail % rowFactor == 0) ? rowFactor : rowOfTail % rowFactor;
        t.scalesAttr = 1.0f;

        usedCore_ = usedCore;
        kvCacheCol_ = kvCacheCol;
        rowStride_ = rowStride;
    }

    int64_t usedCore_ = 1;
    int64_t kvCacheCol_ = 0;
    int64_t rowStride_ = 0; // cache 行步长(headDim), >= kvCacheCol_
};

TEST_P(KvCompressEpilogKernelTest, RunKernel)
{
    auto p = GetParam();
    std::cout << "[KvCompressEpilog] " << p.desc << " bs=" << p.bs << " d=" << p.d << " quantMode=" << p.quantMode
              << " roundScale=" << p.roundScale << " blockDim=" << p.blockDim << std::endl;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    KvCompressEpilogTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    ComputeTiling(tilingData, p);

    // slot 上限 (= bs-1), cache 大小按 (maxSlot+1)*rowStride(headDim) 预留并留余量。
    int64_t cacheRows = p.bs + 1;
    size_t shapeX = static_cast<size_t>(p.bs) * p.d * sizeof(uint16_t); // bf16
    size_t shapeCache = static_cast<size_t>(cacheRows) * rowStride_ * sizeof(uint8_t) + 512;
    size_t shapeSlot = static_cast<size_t>(p.bs) * sizeof(int32_t);

    uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
    uint8_t* slotGm = GmAllocWrapper<uint8_t>(shapeSlot);
    uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(KvCompressEpilogTilingData));

    FillBf16(xGm, static_cast<int64_t>(p.bs) * p.d);
    // verifyRope: 每行 rope 段(后 64 列)前两个 bf16 写入该行的行号 marker。mode0/1 下 rope 是 bf16
    // 逐字节拷贝到 cache 行首 [0:128], 因此 cache 行首 4 字节可唯一标识"是哪一行 x 落到了该 slot"。
    auto Marker0 = [](int64_t r) -> uint16_t { return static_cast<uint16_t>(r & 0xFFFF); };
    auto Marker1 = [](int64_t r) -> uint16_t { return static_cast<uint16_t>(0xA000 | (r & 0x0FFF)); };
    if (p.verifyRope) {
        uint16_t* xp = reinterpret_cast<uint16_t*>(xGm);
        int64_t ropeBase = p.d - SLICE_SIZE; // rope 段在每行的列偏移
        for (int64_t r = 0; r < p.bs; ++r) {
            xp[r * p.d + ropeBase + 0] = Marker0(r);
            xp[r * p.d + ropeBase + 1] = Marker1(r);
        }
    }
    std::memset(cacheGm, 0, shapeCache);
    std::memset(cacheOutGm, 0, shapeCache);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &tilingData, sizeof(KvCompressEpilogTilingData));

    int32_t* slot = reinterpret_cast<int32_t*>(slotGm);
    for (int64_t i = 0; i < p.bs; ++i) {
        slot[i] = static_cast<int32_t>(i);
    }
    int64_t skippedSlots = 0;
    if (p.withSkip) {
        for (int64_t i = 0; i < p.bs; i += 3) { // 每 3 行 skip 一行
            slot[i] = -1;
            ++skippedSlots;
        }
    }

    uint32_t blockDim = static_cast<uint32_t>(usedCore_);
    // 单一 tiling key 0; mode0/1/2 在 kernel 内按 tilingData.quantMode 运行时分支 (与 host GetTilingKey 一致)
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(kv_compress_epilog, blockDim, cacheGm, xGm, slotGm, cacheOutGm, workspace, tilingGm);

    // sanity: 至少一个有效 slot 的 cache 行被写入了非零字节。
    bool anyNonZero = false;
    for (int64_t i = 0; i < p.bs && !anyNonZero; ++i) {
        if (slot[i] == -1) {
            continue;
        }
        uint8_t* row = cacheGm + slot[i] * rowStride_;
        for (int64_t c = 0; c < kvCacheCol_; ++c) {
            if (row[c] != 0) {
                anyNonZero = true;
                break;
            }
        }
    }
    if (p.bs - skippedSlots > 0) {
        EXPECT_TRUE(anyNonZero) << "no written cache row is non-zero";
    }

    // 多 tile x行↔slot 配对校验: 逐有效 slot 比对 cache 行首 rope marker。
    if (p.verifyRope) {
        int64_t mismatches = 0;
        int64_t firstBad = -1;
        for (int64_t r = 0; r < p.bs; ++r) {
            if (slot[r] == -1) {
                continue;
            }
            const uint16_t* row = reinterpret_cast<const uint16_t*>(cacheGm + slot[r] * rowStride_);
            if (row[0] != Marker0(r) || row[1] != Marker1(r)) {
                if (firstBad < 0) {
                    firstBad = r;
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0) << "rope x-row<->slot mispairing: " << mismatches << "/" << p.bs
                                 << " rows; first bad row=" << firstBad << " (rowFactor=" << p.rowFactorOverride
                                 << ", rowLoop="
                                 << (p.rowFactorOverride > 0 ? CeilDiv(CeilDiv(p.bs, p.blockDim), p.rowFactorOverride) :
                                                               1)
                                 << ")";
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(slotGm);
    AscendC::GmFree(cacheGm);
    AscendC::GmFree(cacheOutGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

INSTANTIATE_TEST_SUITE_P(
    QuantModes, KvCompressEpilogKernelTest,
    testing::Values(
        KvParams{32, 256, QUANT_MODE_GROUP_BF16, 1, 1, false, "group_bf16_round"},
        KvParams{32, 256, QUANT_MODE_GROUP_BF16, 0, 1, false, "group_bf16_noround"},
        KvParams{32, 256, QUANT_MODE_GROUP_E8M0, 1, 1, false, "group_e8m0_round"},
        KvParams{32, 256, QUANT_MODE_GROUP_E8M0, 0, 1, false, "group_e8m0_noround"},
        KvParams{32, 128, QUANT_MODE_GROUP_BF16, 1, 1, false, "group_bf16_d128"},
        KvParams{48, 256, QUANT_MODE_GROUP_E8M0, 1, 2, false, "group_e8m0_multicore"},
        KvParams{30, 256, QUANT_MODE_GROUP_BF16, 1, 1, true, "group_bf16_with_skip"},
        // mode2: rope hifloat8 + nope FLOAT4_E2M1, d=256 (nope=192, 可被 16/32/64 整除)
        KvParams{32, 256, QUANT_MODE_HIF8_FP4, 0, 1, false, "hif8_fp4_group64", 64},
        KvParams{32, 256, QUANT_MODE_HIF8_FP4, 0, 1, false, "hif8_fp4_group32", 32},
        KvParams{32, 256, QUANT_MODE_HIF8_FP4, 0, 1, false, "hif8_fp4_group16", 16},
        KvParams{48, 256, QUANT_MODE_HIF8_FP4, 0, 2, false, "hif8_fp4_group64_multicore", 64},
        KvParams{30, 256, QUANT_MODE_HIF8_FP4, 0, 1, true, "hif8_fp4_group32_with_skip", 32},
        // ===== 多 tile (rowLoopOfFormerBlock>1) x行↔slot 配对回归 (rowFactorOverride>0 + verifyRope) =====
        // 这些用例复现并固化大 bs 数据错乱 bug: 每核需多个 UB tile, 校验每个 slot 落的是正确 x 行的 rope。
        // KvParams{bs, d, mode, round, blockDim, withSkip, desc, quantGroup, rowFactorOverride, verifyRope}
        KvParams{512, 384, QUANT_MODE_GROUP_BF16, 1, 1, false, "multitile_bf16_rf64", QUANT_GROUP, 64, true},
        KvParams{512, 384, QUANT_MODE_GROUP_E8M0, 1, 1, false, "multitile_e8m0_rf64", QUANT_GROUP, 64, true},
        KvParams{200, 384, QUANT_MODE_GROUP_E8M0, 1, 1, false, "multitile_e8m0_tail", QUANT_GROUP, 64, true},
        KvParams{300, 384, QUANT_MODE_GROUP_BF16, 1, 2, false, "multitile_bf16_2core", QUANT_GROUP, 64, true},
        KvParams{260, 384, QUANT_MODE_GROUP_E8M0, 1, 1, true, "multitile_e8m0_skip", QUANT_GROUP, 64, true},
        // mode2 多 tile 路径(共用同一 Process 流水), sanity: 不崩溃且有效 slot 非全 0。
        KvParams{512, 384, QUANT_MODE_HIF8_FP4, 0, 1, false, "multitile_hif8fp4_rf64", 64, 64, false},
        // ===== 4D cache 行内 pad (headDim>kvCacheCol) 回归: 行步长=headDim, 每行只写 kvCacheCol 字节 =====
        // KvParams{bs, d, mode, round, blockDim, withSkip, desc, quantGroup, rowFactorOverride, verifyRope, headDim}
        // mode0 d256 kvCacheCol=352, headDim=512: 验证按行步长 512 寻址、写 352 字节, sanity 非全 0。
        KvParams{32, 256, QUANT_MODE_GROUP_BF16, 1, 1, false, "bf16_intrarow_pad_hd512", QUANT_GROUP, 0, false, 512},
        // e8m0 d384 kvCacheCol=453, headDim=640 + 多 tile + rope 配对: 验证 pad 下 slot*headDim 寻址正确。
        KvParams{512, 384, QUANT_MODE_GROUP_E8M0, 1, 1, false, "multitile_e8m0_pad_hd640", QUANT_GROUP, 64, true,
                 640}));

// 结构体字段 sanity (保留原 tiling-data 校验意图: 字段可写、布局可用)。
TEST(KvCompressEpilogTilingStruct, FieldsWritable)
{
    KvCompressEpilogTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.bs = 1024;
    t.d = 256;
    t.quantMode = 1;
    t.roundScale = 1;
    t.scalesAttr = 1.0f;
    EXPECT_EQ(t.bs, 1024);
    EXPECT_EQ(t.d, 256);
    EXPECT_EQ(t.quantMode, 1);
    EXPECT_EQ(t.roundScale, 1);
    EXPECT_FLOAT_EQ(t.scalesAttr, 1.0f);
    EXPECT_GT(sizeof(KvCompressEpilogTilingData), 0u);
}
