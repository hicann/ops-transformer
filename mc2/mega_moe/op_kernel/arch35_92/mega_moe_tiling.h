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
 * \file mega_moe_tiling.h
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING
#define ASCENDC_MEGA_MOE_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "common/mega_moe_gm_layout.h"

// MegaMoe adaptive UB buffer policy shared by host tiling and kernel address binding.
// 实际 UB 预算由 host tiling 从平台获取，并通过 availableUbBytes 参与以下配置计算。

/*
 * Unpermute buffer policy
 *
 * Unpermute 保留 1 个累加/输出 buffer，并支持在 EVENT_ID0~EVENT_ID5 上运行 2~6 个输入 buffer。
 * 实际 UB 仅按 tiling 选中的输入 buffer 数量分配；该上限不表示固定预留 6 个 buffer。
 *
 * 最低双 buffer 是基准 batch 的成立前提。按最坏合法规格 k=8192、BF16 topK weight、combine quant 计算：
 *   1 个累加槽 + 2 个输入槽 = 3 * (8192 * 2B + 8192 * 4B) = 144KB；
 *   8192 个 weight item       = 8192 * (4B + 2B)              = 48KB；
 *   BF16/FP32 scale scratch   = 1KB + 4KB                     = 5KB；
 *   当前 token 的 expertRanks 固定预留 32 * sizeof(int32_t) = 128B；
 *   合计约 197KB；当前支持平台动态获取的 UB 预算可容纳该布局。
 * 若修改 MAX_H、weight batch 基准、scale scratch 布局或最低 buffer 数，必须重新验证该前提。
 *
 * weight batch 延续原基准：tokensPerBatch * topK 不超过 1024 * 8 个 weight item。
 * CalcUnpermuteBufferConfig 先按该基准确定 batch，再用剩余 UB 反推输入 buffer 数量；上述最坏规格
 * 已保证基准 batch 至少容纳 2 个输入 buffer，因此计算结果可以安全保底到最低 buffer 数。
 *
 * 固定输入 buffer 数量后扩大 weight batch 时，为 FP32 weight 和可选输入转换 buffer 各预留一个
 * 32B 对齐块。统一预留 64B 可覆盖两个独立 buffer 各最多 31B 的 padding。
 *
 * DeQuantMxFp8 的 E8M0 scale 经 DIST_INTLV_B16 交织复制为 2 个 BF16，随后经 DIST_INTLV_B32
 * 交织复制为 4 个 FP32。因此 BF16 scratch 需要 scaleNum * 2 个元素，FP32 scratch 需要
 * scaleNum * 4 个元素；若修改 scale scratch 布局，需同步更新展开倍数。
 */
constexpr int32_t MIN_UNPERMUTE_INPUT_BUFFER_COUNT = 2;
constexpr int32_t MAX_UNPERMUTE_INPUT_BUFFER_COUNT = 6;
constexpr int32_t UNPERMUTE_BASE_TOPK = 8;
constexpr int32_t UNPERMUTE_BASE_TOKENS_PER_BATCH = 1024;
constexpr int32_t UNPERMUTE_WEIGHT_ITEMS_PER_BATCH = UNPERMUTE_BASE_TOPK * UNPERMUTE_BASE_TOKENS_PER_BATCH;
constexpr uint32_t UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES = 64U;
constexpr uint32_t UNPERMUTE_EXPERT_RANKS_UB_BYTES = 32U * sizeof(int32_t);
constexpr uint32_t DEQUANT_BF16_SCALE_EXPANSION = 2U;
constexpr uint32_t DEQUANT_FP32_SCALE_EXPANSION = 4U;

/*
 * Route batch shared policy
 *
 * Dispatch 和 SendMask 的 route batch 均按 ALIGN_256 个 item 对齐，使 bit mask 天然按 32B 对齐。
 * 两个基准值当前数值相同，只是各自 UB 预算计算后的结果，彼此独立，不要求保持相等。
 */

/*
 * Dispatch buffer policy
 *
 * Recv route batch 基准用于先选择 dispatch buffer 数量，再固定 buffer 数反推最终 batch。
 * Brecv 表示基准 batch 的 route item 数，D 表示 dispatch buffer 数：
 *   recvVariableBytes(Brecv, D) = Brecv / 8 + 2 * Brecv * sizeof(int32_t)
 *                                  + D * (tokenScaleBytes + 32)；
 *   recvFixedBytes = 32 + Align32(worldSize * moeExpertPerRank * 4) + worldSize * 32
 *                    + Align32(moeExpertPerRank * 4)；
 *   recvFixedBytes + recvVariableBytes(Brecv, D) <= availableUbBytes。
 *
 * 12288 的成立前提按 k=8192、moeExpertNum<=2048、worldSize<=1024、moeExpertPerRank<=1024 和 D=6
 * 保守验证。分别取各 tensor 的独立上限时：
 *   recvFixedBytes             <= 32 + 8192 + 32768 + 4096 = 45088B；
 *   route batch tensor         = 12288 / 8 + 2 * 12288 * 4 = 99840B；
 *   6 个 dispatch slot         <= 6 * (8192B token + 256B scale + 128B weight + 32B triple)
 *                               = 51648B；
 *   合计 196576B，约 192KB；当前支持平台动态获取的 UB 预算可容纳该布局。
 * 因此该基准在所有合法规格下至少支持双 buffer，且最坏规格也能支持当前上限 6 个 buffer。若修改
 * MAX_H、MAX_MOE_EXPERT_NUM、worldSize/moeExpertPerRank 上限、DispatchBuffInit 固定 tensor 布局或
 * dispatch buffer 数量上限，必须重新验证该前提。
 *
 * CalcDispatchBufferConfig 固定选中的 buffer 数后，用剩余 UB 扩大 batch，并向下对齐到 ALIGN_256。
 * 每个 route item 占 1 bit mask 和两个 int32 index，因此：
 *   maxRouteItems = routeItemBudgetBytes * 8 / 65。
 * CopyGMToGMPerToken 的 copyTmp、triple 和 event ID 三类槽位一一对应，MTE 启动轮沿用自适应 2~6 个 buffer，稳态使用前 2
 * 个槽，UB 地址保持不变。
 */
constexpr int32_t MIN_DISPATCH_BUFFER_COUNT = 2;
constexpr int32_t MAX_DISPATCH_BUFFER_COUNT = 6;
constexpr int32_t BASE_RECV_ROUTE_ITEMS_PER_BATCH = 12288;

// 即使最后一块未满，计数表的 VF 仍会加载一个完整向量。
constexpr uint32_t DISPATCH_CUMSUM_VECTOR_BYTES = 256U;
constexpr uint32_t MEGA_MOE_MAX_EXPERT_NUM = 2048U;
constexpr uint32_t MEGA_MOE_MAX_EXPERT_PER_RANK = 1024U;
// Activation 的两个 256 x 256 BF16 槽覆盖前 256 KiB，常驻表须避开这段固定工作区。
constexpr uint32_t ARCH92_FIXED_WORK_UB_BYTES = 2U * 256U * 256U * sizeof(uint16_t);
// 每个 AIV 固定预留 12 KiB：全专家 cumsum 8 KiB，本卡专家计数 4 KiB。
constexpr uint32_t ARCH92_DISPATCH_CUMSUM_BYTES = MEGA_MOE_MAX_EXPERT_NUM * sizeof(int32_t);
constexpr uint32_t ARCH92_DISPATCH_EXPERT_COUNT_BYTES = MEGA_MOE_MAX_EXPERT_PER_RANK * sizeof(int32_t);
constexpr uint32_t ARCH92_DISPATCH_TABLE_BYTES = ARCH92_DISPATCH_CUMSUM_BYTES + ARCH92_DISPATCH_EXPERT_COUNT_BYTES;

/*
 * SendMask buffer policy
 *
 * Send route batch 基准用于先选择 mask buffer 数量，再固定 buffer 数反推最终 batch。
 * Bsend 表示基准 batch 的 route item 数，N 表示 mask buffer 数：
 *   indexBytes 为路由下标宽度（2 或 4 字节），topkIds 中的专家号仍为 int32；
 *   sendVariableBytes(Bsend, N) = Bsend * (4 + indexBytes)
 *                               + N * CalcTopkValidIndexRingSlotBytes(Bsend, topK, indexBytes)；
 *   sendFixedBytes = resetBytes + quantScratchBytes + sendCntAccBytes；
 *   sendFixedBytes + sendVariableBytes(Bsend, N) <= availableUbBytes。
 *
 * MTE 的 MoE/shared 共用输入双缓冲和 2048B mxTemp，需要独立共享量化时增加共享输出双缓冲。
 * H=8192 时共享 FP8 输出双缓冲最多增加 16896B，FP4 最多增加 8704B。
 * count 阶段复用累计表之前已结束使用的量化/下标/ring 区；输入双缓冲至少 4096B，
 * 足以容纳一张卡最多 1024 个专家的 count 重排，无需额外预留重排区。
 * 基准 batch 保持 10240，host 将共享输出计入固定占用后选择 ring 深度并扩大 batch。
 * ring 槽含 mask 和筛选下标：Bsend / 8 + Align32(CeilDiv(Bsend + topK - 1, topK) * indexBytes)。
 * 此配置由 MTE 发送路径使用；URMA Layered 在 kernel 中单独计算发送布局。
 *
 * CalcTopkValidIndexBufferConfig 固定选中的 buffer 数后，用剩余 UB 扩大 batch，并向下对齐到 ALIGN_256。
 * SendMaskCal 使用 MTE3_V 的 EVENT_ID0~EVENT_ID5 管理 mask push ring。最低保留双 buffer，最多 6 个。
 */
constexpr int32_t MIN_SEND_MASK_BUFFER_COUNT = 2;
constexpr int32_t MAX_SEND_MASK_BUFFER_COUNT = 6;
constexpr int32_t BASE_SEND_ROUTE_ITEMS_PER_BATCH = 10240;

/*
 * Reset buffer policy
 *
 * ResetFlagList 分批清零的粒度，单位为 int32 元素。2048 * sizeof(int32_t) = 8KB；本核 flag 份额
 * 超过该值时复用同一片零 UB 分批推 GM，使 reset UB 与 BS 解耦。若调整 resetTensor 的 UB 预算或
 * 元素类型，需要按 targetResetUbBytes / sizeof(elementType) 重新计算。
 */
constexpr int32_t DISPATCH_RESET_BATCH = 2048;

// GMM2 M 方向的一个 tile 对应一个 combine token group；同步 slot 的粒度必须与该 tile 保持一致。
constexpr uint32_t COMBINE_TOKEN_GROUP_SIZE = 256U;

// Adaptive buffer configs written by host tiling and consumed by the kernel.
struct MegaMoeUnpermuteBufferConfig {
    int32_t tokensPerBatch;
    int32_t inputBufferCount;
    uint32_t bf16SlotElementCount;
    uint32_t fp32SlotElementCount;
    uint32_t topKWeightsBufferBytes;
    uint32_t topKWeightsConversionBufferBytes;
};

struct MegaMoeDispatchBufferConfig {
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
    int32_t bufferCount;
    uint32_t copyBufferBytes;
};

struct MegaMoeSendMaskBufferConfig {
    int32_t routeItemsPerBatch;
    int32_t routeBatchCount;
    int32_t bufferCount;
    uint32_t bufferBytes;
};

// Host-computed A8W4 layout, shared by Wave/Layered and routed/shared experts.
// Offsets are bytes; A and expanded W8 B tiles both use 256 x 256 bytes.
struct MegaMoeL1Layout {
    static constexpr uint64_t MAX_A_BUFFER_NUM = 6;
    static constexpr uint64_t MAX_B_BUFFER_NUM = 8;
    uint64_t aBufferNum;
    uint64_t bBufferNum;
    uint64_t tileK;
    uint64_t scaleK;
    uint64_t aSlotBytes;
    uint64_t bSlotBytes;
    uint64_t scaleASlotBytes;
    uint64_t scaleBSlotBytes;
    uint64_t aOffsets[MAX_A_BUFFER_NUM];
    uint64_t bOffsets[MAX_B_BUFFER_NUM];
    uint64_t scaleAOffsets[2];
    uint64_t scaleBOffsets[2];
};

// BF16 使用 256 行逻辑任务，普通 MMAD 在 L0 按 128 行分条，不申请 MX scale。
struct MegaMoeBf16GmmConfig {
    static constexpr uint32_t TILE_M = 256U;
    static constexpr uint32_t TILE_N = 256U;
    static constexpr uint32_t L1_TILE_K = 128U;
    static constexpr uint32_t L0_TILE_M = 128U;
    static constexpr uint32_t L0_TILE_K = 64U;
    static constexpr uint32_t BUFFER_COUNT = 2U;
    static constexpr uint32_t GMM1_UB_BUFFER_COUNT = 1U;
    // 固定输入槽、门控激活输出和逐行路由元数据的最大占用。
    static constexpr uint32_t ACTIVATION_UB_BYTES =
        TILE_M * (TILE_N * sizeof(uint16_t) * 3U / 2U + MegaMoeImpl::MemoryLayoutDetail::ALIGN_32);
};

struct MegaMoeTilingData {
    // 通信布局和本地计算布局直接供 host 与 kernel 共用；修改本结构后需配套更新算子包。
    MegaMoeImpl::CclCommLayoutParams comm;
    MegaMoeImpl::CclComputeLayoutParams compute;
    uint32_t blockNumPerEP;
    uint32_t blockAivNum;
    int64_t combineQuantMode;
    float clampLimit;
    uint8_t moeGmmMode;
    uint8_t sharedGmmMode;

    // Dispatch 的分核不改变 UB 布局，所有 AIV core 共用一套配置；对应 kernel DispatchBuffInit。
    MegaMoeDispatchBufferConfig dispatchBufferConfig;
    // SendMask 将全局 expertId 连续均分，前 remainder 个 core 会多处理一个 expert；
    // 对应 kernel SendAndQuantBuffInit 的选择条件和 SendMaskCal 的连续 expert 区间。
    MegaMoeSendMaskBufferConfig sendMaskConfigForCoreWithExtraExpert;
    MegaMoeSendMaskBufferConfig sendMaskConfigForCoreWithoutExtraExpert;
    uint32_t sendMaskCoreCountWithExtraExpert;
    // Unpermute 的 TilingByCore 生成若干完整 token chunk 和至多一个尾 chunk，分别预计算配置；
    // 对应 kernel UnpermuteBuffInit 的选择条件。
    MegaMoeUnpermuteBufferConfig unpermuteConfigForFullTokenChunk;
    MegaMoeUnpermuteBufferConfig unpermuteConfigForTailTokenChunk;
    uint32_t unpermuteFullTokenChunkCoreCount;
    bool isTopkWeightsPrefetch;
    uint8_t actMode;       // MegaMoeImpl::MegaMoeActMode 的底层协议值
    uint8_t actSubMode;    // MegaMoeImpl::MegaMoeActSubMode 的底层协议值
    float activationAlpha; // situ: linear_beta; swigluoai: alpha(默认1.702); swiglu/swiglustep: 占位
    float activationBeta;  // situ/swigluoai: beta(默认1.0); swiglu/swiglustep: 占位
    // MoE 和共享专家使用相同布局：true 表示每个专家一个二维 tensor，false 表示单个三维堆叠 tensor。
    bool isPerExpertWeightTensor;
    // URMA 分层拓扑；Host workspace 计算与 Device 通道映射必须使用同一个值。
    uint32_t rankNumPerServer;
    MegaMoeL1Layout a8w4L1Layout;
    // 预量化输入搬运 tile 配置。动态量化路径保持旧的 scratch 布局，这两个字段为 0。
    uint32_t preQuantTileTokens;
    uint32_t preQuantStageBytes;
    // A6 按实际可用 UB 容量计算常驻表基址；追加字段保持已有字段偏移不变。
    uint32_t dispatchTableUbBaseAddr;
    // HCCL 窗口总字节数（ccl_buffer_size），OOM 检测按整窗登记合法地址段。
    uint64_t cclBufferSize;
    uint32_t dispatchQuantMode;
    uint32_t topkIndexTypeBytes;
};
#endif
