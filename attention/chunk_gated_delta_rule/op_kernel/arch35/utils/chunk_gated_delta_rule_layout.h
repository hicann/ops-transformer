/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "chunk_gated_delta_rule_utils_arch35.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// 跨层共同约定：物理 tile、可用容量，以及 AIC/AIV 交换数据时必须一致的地址。
// 各 Compute 的私有暂存区、锁编号和绑定结构放在各自实现内；这里只保留一份通信布局。
struct FusedLayout {
    // Ascend950 物理暂存容量，单位字节。这里直接固化当前 arch35 kernel 的硬件约束；
    // 如后续支持多代硬件，应随 arch 目录一起调整。
    static constexpr uint32_t UB_SIZE = 248 * 1024;  // 248 KiB
    static constexpr uint32_t L1_SIZE = 512 * 1024;  // 512 KiB
    static constexpr uint32_t L0A_SIZE = 64 * 1024;  // 64 KiB
    static constexpr uint32_t L0B_SIZE = 64 * 1024;  // 64 KiB
    static constexpr uint32_t L0C_SIZE = 256 * 1024; // 256 KiB

    // 地址只是无状态的 SDK 张量视图。缓冲区的可复用时点仍由显式 Lock/Unlock 与跨核事件约束。
    struct BufferViews {
        template <typename T>
        __aicore__ static inline LocalTensor<T> Ub(uint32_t byteOffset)
        {
            return At<T, TPosition::VECCALC, UB_SIZE>(byteOffset);
        }
        template <typename T>
        __aicore__ static inline LocalTensor<T> L1(uint32_t byteOffset)
        {
            return At<T, TPosition::A1, L1_SIZE>(byteOffset);
        }
        template <typename T>
        __aicore__ static inline LocalTensor<T> L0A(uint32_t byteOffset)
        {
            return At<T, TPosition::A2, L0A_SIZE>(byteOffset);
        }
        template <typename T>
        __aicore__ static inline LocalTensor<T> L0B(uint32_t byteOffset)
        {
            return At<T, TPosition::B2, L0B_SIZE>(byteOffset);
        }
        template <typename T>
        __aicore__ static inline LocalTensor<T> L0C(uint32_t byteOffset)
        {
            return At<T, TPosition::CO1, L0C_SIZE>(byteOffset);
        }

    private:
        template <typename T, TPosition position, uint32_t capacity>
        __aicore__ static inline LocalTensor<T> At(uint32_t byteOffset)
        {
            return LocalTensor<T>(position, byteOffset, (capacity - byteOffset) / sizeof(T));
        }
    };

    static constexpr uint32_t CHUNK = 64;
    static constexpr uint32_t HALF_CHUNK = 32;
    // Cube 始终按最大物理特征维 128 计算；逻辑 Dk/Dv 只约束有效数据的搬运与写回。
    // 尾 chunk、特征尾部在送入 MMAD 前补零，不能把一次 DMA 的对齐补零误当成整个 tile 已清零。
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t HALF_DIM = 64;
    static constexpr uint32_t STATE_SLICE_ELEMS = HALF_DIM * DIM;
    static constexpr uint32_t SCORE_SLICE_ELEMS = HALF_CHUNK * CHUNK;

    // 使用架构给出的可用 UB 容量，不在布局中再维护一份硬件容量数字。
    static constexpr uint32_t UB_WORKSPACE_BYTES = UB_SIZE;
    static constexpr uint32_t MUTEX_COUNT = 8;
    static_assert(2 * HALF_DIM == DIM && 2 * HALF_CHUNK == CHUNK, "AIV halves must cover one physical tile");

    // 布局表均使用半开区间 [begin,end)，单位 KiB；代码中的偏移/容量单位为字节。
    // L1 总览：[0,160) chunk 槽0 | [160,192) S | [192,256) 未使用；
    //          [256,416) chunk 槽1 | [416,512) 未使用。
    // 优化方案：L1 高位交织，两槽按半个 L1 的跨度对称分配，使对应矩阵落在不同的256KiB半区。
    // 当前槽读与下一槽写可避免同半区竞争；这不消除槽内矩阵、共享 S 的所有 bank 冲突。
    // S 是单份 BF16 NZ [D,D]=32KiB，AIV MTE3 -> AIC MTE1；StateBufferFree/StateReady 控制复用。
    // 槽内矩阵均为 BF16 NZ，C=CHUNK=64、D=DIM=128；物理维度已按16对齐，容量=行*列*2B。
    // 相对槽起点     内容 / shape                         写入流水 -> 读取流水
    // [  0, 16)     Q:[C,D]                               MTE2 -> MTE1
    // [ 16, 32)     K:[C,D]                               MTE2 -> MTE1
    // [ 32, 40)     A:[C,C]，左下块由 -L10 更新为 A10      AIV MTE3 / AIC FIX -> MTE1
    // [ 40, 42)     (-L10)@A00:[C/2,C/2]                  FIX -> MTE1
    // [ 42, 48)     保留空隙6 KiB，当前不存放矩阵
    // [ 48, 64)     rawKcd=-beta*exp(G)*K:[C,D]            AIV MTE3 -> MTE1
    // [ 64, 80)     vBeta=beta*V:[C,D]                     AIV MTE3 -> MTE1
    // [ 80,112)     [Kcd;Q']:[2*C,D]                      FIX / AIV MTE3 -> MTE1
    // [112,128)     R:[C,D]                               AIV MTE3 -> MTE1
    // [128,144)     Kg=exp(G_last-G)*K:[C,D]               AIV MTE3 -> MTE1
    // [144,152)     M=maskedQkt:[C,C]                     AIV MTE3 -> MTE1
    // [152,160)     保留空隙8 KiB，当前不存放矩阵
    // 6/8 KiB 留白沿用已有布局，不能据此推断硬件要求这些大小的对齐或可获得 bank 优化。
    // 优化方案：两槽使当前递推和下一块前缀重叠；rawKcd/vBeta 连续，合并一次 MTE3 搬运。
    // Q/K、逆块临时区的私有起点在 CubeCompute；公共部分只定义两侧需要一致的边界。
    static constexpr uint32_t L1_TILE_BYTES = CHUNK * DIM * sizeof(bfloat16_t); // [C,D] = 16 KiB
    static constexpr uint32_t L1_INVERSE_OFFSET = 2 * L1_TILE_BYTES;            // 紧接 Q、K
    static constexpr uint32_t L1_RAW_KCD_OFFSET = L1_INVERSE_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t) +
                                                  HALF_CHUNK * HALF_CHUNK * sizeof(bfloat16_t) + 6 * 1024;
    static constexpr uint32_t L1_VBETA_OFFSET = L1_RAW_KCD_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_KCD_QPRIME_OFFSET = L1_VBETA_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_RESIDUAL_OFFSET = L1_KCD_QPRIME_OFFSET + 2 * L1_TILE_BYTES;
    static constexpr uint32_t L1_KG_OFFSET = L1_RESIDUAL_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_MASKED_QKT_OFFSET = L1_KG_OFFSET + L1_TILE_BYTES;
    static constexpr uint32_t L1_SLOT_BYTES = L1_MASKED_QKT_OFFSET + CHUNK * CHUNK * sizeof(bfloat16_t) + 8 * 1024;
    // 槽容量和槽间距分开：160KiB 是存储范围，256KiB 是双缓冲的地址跨度。
    static constexpr uint32_t L1_SLOT_STRIDE_BYTES = L1_SIZE / 2;
    static constexpr uint32_t L1_STATE_OFFSET = L1_SLOT_BYTES; // 单份 S 放在槽0之后

    // Chunk 路径：每个 AIV 各有一份 UB；这里的“共享”指 AIC FIX 与目标 AIV 约定相同地址。
    // [  0, 72) Vector 私有：S32 + 留白8 + lower/inverse/identity 各8 + M双槽各4 KiB
    // [ 72, 88) BF16 delta:[D/2,D]，FIX -> V；BF16 状态接口使用
    // [ 88,160) Vector 私有输入/状态副本/标量及留白，详见 VectorCompute 的完整布局
    // [160,168) 槽0 KKT/QKT：两个 BF16 [C/2,C]，FIX -> V
    // [168,176) Vector 私有 inverse 发布双槽，各 BF16 [C/2,C]，V -> MTE3
    // [176,208) Stage1 槽0：32 KiB，分阶段复用（见下）
    // [208,240) Stage1 槽1：32 KiB，分阶段复用（见下）
    // [240,248) 槽1 KKT/QKT：两个 BF16 [C/2,C]，FIX -> V
    // Stage1 同一槽的生命周期：
    //   依次 rawKcd/vBeta/Q'/Kg，四个 BF16 [C/2,D] 矩阵(4*8KiB)，V -> MTE3 -> L1；
    //   然后前16KiB接收 NZ R32:[D/16,C/2,16]，接着8KiB保存 R16，末8KiB空闲，FIX -> V -> MTE3；
    //   最后全部32KiB保存 FP32 delta:[D/2,D]，供状态更新。
    // FP32 接口由 FIX 直接写 delta；BF16 接口从 [72,88) Cast 到此处，舍入语义不变。
    // 跨阶段必须沿用 ResidualReady、StateDeltaReady/BufferFree 等握手，不能因地址相同并行复用。
    // 下方72KiB边界是私有区各容量之和；顶部从可用UB末端向下排，避免硬写160/176/208/240。
    static constexpr uint32_t UB_STATE_DELTA_BF16_OFFSET = STATE_SLICE_ELEMS * sizeof(float) + 8 * 1024 +
                                                           3 * SCORE_SLICE_ELEMS * sizeof(float) +
                                                           2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t UB_PREFIX_DIRECT_BYTES = 2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t);
    // 最大阶段占用：4*BF16[C/2,D] = FP32[D/2,D] = 32 KiB；R32+R16 只需24 KiB。
    static constexpr uint32_t UB_STAGE1_SLOT_BYTES = STATE_SLICE_ELEMS * sizeof(float);
    static constexpr uint32_t UB_PREFIX_DIRECT1_OFFSET = UB_WORKSPACE_BYTES - UB_PREFIX_DIRECT_BYTES;
    static constexpr uint32_t UB_STAGE1_PUB1_OFFSET = UB_PREFIX_DIRECT1_OFFSET - UB_STAGE1_SLOT_BYTES;
    static constexpr uint32_t UB_STAGE1_PUB0_OFFSET = UB_STAGE1_PUB1_OFFSET - UB_STAGE1_SLOT_BYTES;
    static constexpr uint32_t UB_PREFIX_DIRECT0_OFFSET =
        UB_STAGE1_PUB0_OFFSET - UB_PREFIX_DIRECT_BYTES -
        2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t); // 中间还需两个 inverse 发布槽，共8 KiB

    static_assert(L1_SLOT_BYTES <= L1_SLOT_STRIDE_BYTES, "L1 slot crosses its half");
    static_assert(L1_STATE_OFFSET >= L1_SLOT_BYTES &&
                      L1_STATE_OFFSET + DIM * DIM * sizeof(bfloat16_t) <= L1_SLOT_STRIDE_BYTES,
                  "shared L1 state must fit between the two slots");
    static_assert(L1_SLOT_STRIDE_BYTES + L1_SLOT_BYTES <= L1_SIZE, "L1 slot 1 exceeds capacity");
    static_assert(L1_VBETA_OFFSET == L1_RAW_KCD_OFFSET + CHUNK * DIM * sizeof(bfloat16_t),
                  "rawKcd and vBeta must stay contiguous for merged DMA");
    static_assert(UB_STATE_DELTA_BF16_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_PREFIX_DIRECT0_OFFSET,
                  "BF16 state delta overlaps shared prefix scores");
    static_assert(UB_PREFIX_DIRECT0_OFFSET + UB_PREFIX_DIRECT_BYTES <= UB_STAGE1_PUB0_OFFSET,
                  "prefix scores slot 0 overlaps Stage1");
    static_assert(UB_STAGE1_PUB0_OFFSET + STATE_SLICE_ELEMS * sizeof(float) <= UB_STAGE1_PUB1_OFFSET,
                  "FP32 state delta slot 0 overlaps slot 1");
    static_assert(UB_STAGE1_PUB1_OFFSET + UB_STAGE1_SLOT_BYTES <= UB_PREFIX_DIRECT1_OFFSET,
                  "Stage1 slot 1 overlaps prefix scores slot 1");
    static_assert(UB_STAGE1_SLOT_BYTES >= STATE_SLICE_ELEMS * sizeof(float),
                  "shared Stage1 slot cannot hold the FP32 state delta");
    static_assert(UB_PREFIX_DIRECT1_OFFSET + UB_PREFIX_DIRECT_BYTES <= UB_WORKSPACE_BYTES,
                  "shared UB regions exceed workspace capacity");

    __aicore__ static inline uint32_t PrefixDirectOffset(uint32_t slot)
    {
        return slot == 0 ? UB_PREFIX_DIRECT0_OFFSET : UB_PREFIX_DIRECT1_OFFSET;
    }

    __aicore__ static inline uint32_t PrefixTransformsUbOffset(uint32_t slot)
    {
        return slot == 0 ? UB_STAGE1_PUB0_OFFSET : UB_STAGE1_PUB1_OFFSET;
    }
};

} // namespace ChunkGatedDeltaRule
