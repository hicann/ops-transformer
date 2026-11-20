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

// ==================== mode 4 (MXFP4 packed) kernel 用例（ICPU CPU 仿真） ====================
// mode4 为 uint8 cache 路径, 可在现有 DTYPE 宏 (uint8_t) 下于 ICPU 执行;
// mode3 (fp8 cache) 的字节级精度由真机 ST (ttk aclnn 9 用例) 覆盖, ICPU 不实例化 fp8 dtype。
// 校验: 有效 slot 的数据段被写入(非哨兵) + 哨兵区(行尾 pad / 块尾 / skip 行)保持原值不写。
namespace {
struct MxfpParams {
    int64_t bs;
    int64_t d;
    int64_t group;
    bool layout2; // false=layout1(2D), true=layout2(4D)
    int64_t blockDim;
    bool withSkip;
    std::string desc;
    // >0 时 layout2 块间留 blockGap 字节间隙(块表跳转的非连续摆位):
    // blockStride = blockSize*cacheCol + blockGap（32B 对齐为建议值，非强制）, 间隙区保持哨兵不写。
    int64_t blockGap = 0;
};

TEST(KvCompressEpilogMxfpKernelTest, RunMxfp4Kernel)
{
    const std::vector<MxfpParams> cases = {
        {32, 256, 32, false, 1, false, "mxfp4_g32_layout1"},
        {32, 256, 16, false, 1, false, "mxfp4_g16_layout1_main"}, // 主场景
        {24, 256, 16, true, 1, true, "mxfp4_g16_layout2_skip"},   // 4D 分页 + skip
        {48, 256, 32, true, 2, false, "mxfp4_g32_layout2_2core"},
        // 非连续块(块表跳转): blockGap=128 -> blockStride=8*192+128=1664, 块间 128B 间隙不写。
        {24, 256, 16, true, 1, true, "mxfp4_g16_layout2_noncontig_skip", 128},
        // 非连续块 + 双核: 多核 slot 散写跨非连续块寻址。blockStride=8*176+96=1504。
        {48, 256, 32, true, 2, false, "mxfp4_g32_layout2_noncontig_2core", 96},
        // 非连续块 + 非 32B 对齐 blockStride: blockGap=24 -> blockStride=8*192+24=1560, 间隙不写。
        {24, 256, 16, true, 1, true, "mxfp4_g16_layout2_noncontig_unaligned_skip", 24},
    };
    for (const auto& p : cases) {
        std::cout << "[KvCompressEpilog][mxfp4] " << p.desc << " bs=" << p.bs << " d=" << p.d << " group=" << p.group
                  << " layout=" << (p.layout2 ? "4D" : "2D") << std::endl;
        AscendC::SetKernelMode(KernelMode::AIV_MODE);

        // mxfp 行格式: dataCol = d/2, scaleCol = d/group, concatCol = dataCol + 2*scaleCol,
        // layout1 无 pad: kvCacheCol = concatCol
        const int64_t dataCol = p.d / 2;
        const int64_t scaleCol = p.d / p.group;
        const int64_t scaleBytes = scaleCol * 2;
        const int64_t concatCol = dataCol + scaleBytes;
        const int64_t kvCacheCol = concatCol;

        // layout1: cache 2D [rows, headDim], headDim=kvCacheCol+32 (加宽行验证行尾不写);
        // layout2: cache 4D [B, S, 1, C], C=tokenBundle+32 (块尾不写), blockSize=S=8
        const int64_t blockSize = p.layout2 ? 8 : 1;
        int64_t blockNum = 0;
        int64_t cacheCol = 0;
        int64_t blockStride = 0;
        int64_t cacheRows = 0;
        if (p.layout2) {
            cacheCol = dataCol + scaleBytes + 32;              // > tokenBundle
            blockNum = (p.bs + blockSize - 1) / blockSize + 1; // 留一个空块
            blockStride = blockSize * cacheCol + p.blockGap;   // 非连续: 块间留 blockGap 间隙
            cacheRows = blockNum * blockSize;
        } else {
            cacheCol = kvCacheCol + 32;
            blockNum = 1;
            blockStride = 0;
            cacheRows = p.bs + 1;
        }

        KvCompressEpilogTilingData t;
        std::memset(&t, 0, sizeof(t));
        int64_t rowOfFormer = CeilDiv(p.bs, p.blockDim);
        int64_t usedCore = std::min(CeilDiv(p.bs, rowOfFormer), p.blockDim);
        int64_t rowOfTail = p.bs - (usedCore - 1) * rowOfFormer;
        int64_t rowFactor = rowOfFormer; // 单 tile 即可 (bs<=48, UB 足够)
        t.bs = p.bs;
        t.d = p.d;
        t.kvCacheCol = kvCacheCol;
        t.kvCacheRowStride = p.layout2 ? cacheCol : cacheCol; // 两布局行步长均为物理列宽
        t.kvCacheBlockSize = blockSize;
        t.kvCacheBlockStride = p.layout2 ? blockStride : cacheCol;
        t.scaleCol = scaleCol;
        t.concatCol = concatCol;
        t.padCol = kvCacheCol - concatCol;
        t.quantMode = 4;
        t.roundScale = 1;
        t.perGroupSize = p.group;
        t.rowOfFormerBlock = rowOfFormer;
        t.rowOfTailBlock = rowOfTail;
        t.rowLoopOfFormerBlock = CeilDiv(rowOfFormer, rowFactor);
        t.rowLoopOfTailBlock = CeilDiv(rowOfTail, rowFactor);
        t.rowFactor = rowFactor;
        t.tailRowFactorOfFormerBlock = rowFactor;
        t.tailRowFactorOfTailBlock = rowFactor;
        t.scalesAttr = 1.0f;
        t.layout = p.layout2 ? 2 : 1;
        t.cacheRows = cacheRows;
        t.dataCol = dataCol;
        t.scaleBytes = scaleBytes;
        t.layout2TokenBundleSize = dataCol + scaleBytes;
        t.layout2CacheCol = p.layout2 ? cacheCol : 0;

        size_t shapeX = static_cast<size_t>(p.bs) * p.d * sizeof(uint16_t);
        size_t shapeCache =
            p.layout2 ? static_cast<size_t>(blockNum) * blockStride : static_cast<size_t>(cacheRows) * cacheCol;
        size_t shapeSlot = static_cast<size_t>(p.bs) * sizeof(int32_t);
        uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
        uint8_t* slotGm = GmAllocWrapper<uint8_t>(shapeSlot);
        uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
        uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
        uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
        uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(KvCompressEpilogTilingData));

        FillBf16(xGm, p.bs * p.d);
        std::memset(cacheGm, 0xA5, shapeCache); // 哨兵: 未写入区必须保持 0xA5
        std::memset(cacheOutGm, 0xA5, shapeCache);
        std::memset(workspace, 0, SYS_WORKSPACE);
        std::memcpy(tilingGm, &t, sizeof(KvCompressEpilogTilingData));

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

        const uint64_t tilingKey = p.layout2 ? (p.group == 16 ? 2102UL : 2101UL) : (p.group == 16 ? 2002UL : 2001UL);
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(kv_compress_epilog, static_cast<uint32_t>(usedCore), cacheGm, xGm, slotGm, cacheOutGm, workspace,
                    tilingGm);

        // 校验 1: 每个有效 slot 的数据段 [0, dataCol) 与 scale 段被写 (不再是全 0xA5)
        int64_t written = 0;
        int64_t untouchedBad = 0;
        const int64_t bundle = dataCol + scaleBytes;
        for (int64_t i = 0; i < p.bs; ++i) {
            if (slot[i] < 0 || slot[i] >= cacheRows) {
                continue;
            }
            uint8_t* row = nullptr;
            if (p.layout2) {
                const int64_t blk = slot[i] / blockSize;
                const int64_t pos = slot[i] % blockSize;
                row = cacheGm + blk * blockStride + pos * bundle;
            } else {
                row = cacheGm + slot[i] * cacheCol;
            }
            bool dataChanged = false;
            for (int64_t c = 0; c < dataCol; ++c) {
                if (row[c] != 0xA5) {
                    dataChanged = true;
                    break;
                }
            }
            if (dataChanged) {
                ++written;
            }
            // 校验 2a (layout1): 行尾 [kvCacheCol, cacheCol) 不写 (行步长=cacheCol, 无跨行重叠)
            if (!p.layout2) {
                for (int64_t c = kvCacheCol; c < cacheCol; ++c) {
                    if (row[c] != 0xA5) {
                        ++untouchedBad;
                        break;
                    }
                }
            }
        }
        // 校验 2b (layout2): bundle 紧凑排列(tokenStride 间隔), 哨兵区 = 块尾 [blockSize*bundle,
        //     blockSize*cacheCol) 与块间间隙 [blockSize*cacheCol, blockStride)(非连续 blockGap>0 时)
        //     —— 所有块的块尾/间隙均不应被写
        if (p.layout2) {
            for (int64_t blk = 0; blk < blockNum; ++blk) {
                uint8_t* tail = cacheGm + blk * blockStride + blockSize * bundle;
                for (int64_t c = 0; c < blockStride - blockSize * bundle; ++c) {
                    if (tail[c] != 0xA5) {
                        ++untouchedBad;
                        break;
                    }
                }
            }
        }
        EXPECT_EQ(written, p.bs - skipped) << p.desc << ": written=" << written << " expected=" << (p.bs - skipped);
        EXPECT_EQ(untouchedBad, 0) << p.desc << ": sentinel region was overwritten";

        AscendC::GmFree(xGm);
        AscendC::GmFree(slotGm);
        AscendC::GmFree(cacheGm);
        AscendC::GmFree(cacheOutGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tilingGm);
    }
}

// ==================== mode3 MXFP8 round_scale=0 多行 UB 行步长回归 ====================
// 基线 VFProcessMxFp8<TCache,false> 必须按 RoundUp<uint8_t>(kvCacheCol) 计算 UB 行起点，
// 与 layout2d/layout4d 的 CopyOutRows 保持一致。若不齐，多行 batch 从第 2 行起会读错 UB：
// ICPU 表现为数据错位，真机则因非 32B 对齐向量写触发 VEC_ERROR。
// 这里逐字节比较“单行独立运行”与“多行 batch 运行”中同一 x 行落到同一 slot 的结果。
namespace {
struct Mxfp8AddrCfg {
    int64_t bs;
    int64_t d;
    int64_t rowStride;
    int64_t roundScale;
    std::string desc;
};

void FillMxfp8Rows(std::vector<uint16_t>& xbuf, int64_t rows, int64_t d)
{
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < d; ++i) {
            // 0.25 的倍数，bf16 可精确表示；行间、列间均不同，便于暴露行错位。
            float v = static_cast<float>(r + 1) * 0.25f + static_cast<float>(i % 16) * 0.125f;
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            xbuf[r * d + i] = static_cast<uint16_t>(bits >> 16);
        }
    }
}

void RunMxfp8Kernel(const Mxfp8AddrCfg& cfg, const std::vector<uint16_t>& xAll, const std::vector<int32_t>& slots,
                    int64_t cacheRows, std::vector<uint8_t>& cache)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    const int64_t groups = cfg.d / 32;
    const int64_t dataCol = cfg.d;
    const int64_t concatCol = dataCol + groups * 2;
    const int64_t kvCacheCol = concatCol;

    KvCompressEpilogTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.bs = cfg.bs;
    t.d = cfg.d;
    t.kvCacheCol = kvCacheCol;
    t.kvCacheRowStride = cfg.rowStride;
    t.kvCacheBlockSize = 1;
    t.kvCacheBlockStride = cfg.rowStride;
    t.scaleCol = groups;
    t.concatCol = concatCol;
    t.padCol = 0;
    t.quantMode = 3;
    t.roundScale = cfg.roundScale;
    t.perGroupSize = 32;
    t.rowOfFormerBlock = cfg.bs;
    t.rowOfTailBlock = cfg.bs;
    t.rowLoopOfFormerBlock = 1;
    t.rowLoopOfTailBlock = 1;
    t.rowFactor = cfg.bs;
    t.tailRowFactorOfFormerBlock = cfg.bs;
    t.tailRowFactorOfTailBlock = cfg.bs;
    t.scalesAttr = 1.0f;
    t.layout = 1;
    t.cacheRows = cacheRows;
    t.dataCol = dataCol;
    t.scaleBytes = groups * 2;
    t.layout2TokenBundleSize = concatCol;
    t.layout2CacheCol = 0;

    size_t shapeX = static_cast<size_t>(cfg.bs) * cfg.d * sizeof(uint16_t);
    size_t shapeCache = static_cast<size_t>(cacheRows) * cfg.rowStride;
    uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
    uint8_t* slotGm = GmAllocWrapper<uint8_t>(cfg.bs * sizeof(int32_t));
    uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(KvCompressEpilogTilingData));

    std::memcpy(xGm, xAll.data(), shapeX);
    std::memcpy(slotGm, slots.data(), cfg.bs * sizeof(int32_t));
    std::memset(cacheGm, 0xCC, shapeCache);
    std::memset(cacheOutGm, 0, shapeCache);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &t, sizeof(KvCompressEpilogTilingData));

    ICPU_SET_TILING_KEY(2000);
    ICPU_RUN_KF(kv_compress_epilog, 1, cacheGm, xGm, slotGm, cacheOutGm, workspace, tilingGm);
    cache.assign(cacheGm, cacheGm + shapeCache);

    AscendC::GmFree(xGm);
    AscendC::GmFree(slotGm);
    AscendC::GmFree(cacheGm);
    AscendC::GmFree(cacheOutGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

int64_t CountMxfp8RowMismatch(const Mxfp8AddrCfg& cfg)
{
    std::vector<uint16_t> xAll(static_cast<size_t>(cfg.bs) * cfg.d);
    FillMxfp8Rows(xAll, cfg.bs, cfg.d);
    std::vector<int32_t> slots(cfg.bs);
    for (int64_t i = 0; i < cfg.bs; ++i) {
        slots[i] = static_cast<int32_t>(i);
    }

    // 逐行单 batch 运行作为参考：同一 x 行、slot=0。
    std::vector<std::vector<uint8_t>> refRows(cfg.bs);
    for (int64_t r = 0; r < cfg.bs; ++r) {
        Mxfp8AddrCfg one = cfg;
        one.bs = 1;
        std::vector<uint16_t> xOne(xAll.begin() + r * cfg.d, xAll.begin() + (r + 1) * cfg.d);
        std::vector<int32_t> slotOne = {0};
        std::vector<uint8_t> oneCache;
        RunMxfp8Kernel(one, xOne, slotOne, 1, oneCache);
        refRows[r].assign(oneCache.begin(), oneCache.begin() + cfg.rowStride);
    }

    std::vector<uint8_t> multiCache;
    RunMxfp8Kernel(cfg, xAll, slots, cfg.bs, multiCache);

    int64_t badRows = 0;
    int64_t firstBadByte = -1;
    for (int64_t r = 0; r < cfg.bs; ++r) {
        const uint8_t* got = multiCache.data() + slots[r] * cfg.rowStride;
        for (int64_t b = 0; b < cfg.rowStride; ++b) {
            if (got[b] != refRows[r][b]) {
                ++badRows;
                if (firstBadByte < 0) {
                    firstBadByte = b;
                }
                break;
            }
        }
    }
    std::cout << "[mxfp8-addr] " << cfg.desc << " d=" << cfg.d << " rowStride=" << cfg.rowStride
              << " rs=" << cfg.roundScale << " mismatchRows=" << badRows << "/" << cfg.bs
              << " firstBadByte=" << firstBadByte << std::endl;
    return badRows;
}
} // namespace

TEST(KvCompressEpilogMxfp8AddrTest, RoundScaleOffMultiRowUbStride)
{
    // d=128: kvCacheCol=136 非 32B 对齐，CopyOutRows 按 160 取行。
    EXPECT_EQ(CountMxfp8RowMismatch(Mxfp8AddrCfg{8, 128, 160, 0, "d128_unaligned_rs0"}), 0)
        << "round_scale=0 multi-row must use aligned UB row stride";
    // d=256: kvCacheCol=272 非 32B 对齐，CopyOutRows 按 288 取行。
    EXPECT_EQ(CountMxfp8RowMismatch(Mxfp8AddrCfg{8, 256, 288, 0, "d256_unaligned_rs0"}), 0)
        << "round_scale=0 multi-row must use aligned UB row stride";
    // 对照：d=512 的 kvCacheCol=544 本身 32B 对齐。
    EXPECT_EQ(CountMxfp8RowMismatch(Mxfp8AddrCfg{8, 512, 544, 0, "d512_aligned_rs0_control"}), 0)
        << "aligned control should match";
}
} // namespace
