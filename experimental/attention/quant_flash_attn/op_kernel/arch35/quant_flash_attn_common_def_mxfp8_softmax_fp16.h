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
 * \file quant_flash_attn_common_def_mxfp8_softmax_fp16.h
 * \brief QuantFlashAttn MxFP8 Softmax FP16 公共定义（tiling key 常量 + 尺寸常量 + 跨核同步 + ConstInfo/RunInfo）
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

#include <cstdint>

#include "quant_flash_attn_template_tiling_key.h"

// =====================================================================
// MxFP8 Softmax FP16 kernel 层公共定义（设计文档 design_mxfp8_pipeline.md）
// =====================================================================

// ===== 基础尺寸常量（设计文档 §2）=====
constexpr uint32_t S1_BASE_SIZE = 256;                   // gS1 分块步长（调度器 mBaseSize）
constexpr uint32_t S2_BASE_SIZE = 256;                   // S2 分块步长
constexpr uint32_t D_BASE_SIZE = 128;                    // QK reduce 轴
constexpr uint32_t DV_BASE_SIZE = 128;                   // V 输出维
constexpr uint32_t S2_SUB_LOOP_SIZE = S2_BASE_SIZE / 2;  // 128，C1 subLoop 的 M 轴（切 S2）
constexpr uint32_t S1_SUB_BLOCK_SIZE = S1_BASE_SIZE / 2; // 128，V0/V1 各持 S1 半（切 S1）
constexpr uint32_t ROWSUM_ROWS = 16;                     // C2 L0A 全 1 行数（捎带计算 rowsum 分母）

// ===== 流水深度（设计文档 §2）=====
constexpr uint32_t PRELOAD_N = 2;                           // 预取两轮任务，C1/V1/C2/V2 四段流水
constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1; // = 3

// ===== 跨核同步 flagId（设计文档 §4.2）=====
// 约束：C1/C2 各一个 Matmul 对象时，Matmul 高阶 API 内部占用 flagId [0, 3]，自定义 flagId 从 4 开始。
// subBlock 寻址规则：AIC 侧 flagId 0-15 对应 V0，16-31 对应 V1（即 +16 指向 V1）；V 核侧统一用 [0, 15]。
constexpr uint8_t SYNC_MODE_4 = 4;
constexpr uint16_t CROSS_CORE_SYNC_C1_V1 = 4;      // mm1Res UB，C1(AIC)/V1(AIV) 双向，每 subBlockId
constexpr uint16_t CROSS_CORE_SYNC_V1_P_READY = 5; // L1 P，V1(AIV)→C2(AIC)，每 C1 任务
constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 6;      // mm2Res UB，C2(AIC)/V2(AIV) 双向，每 V 核

// ===== metadata 头附加索引（与主线 AICPU metadata 布局对齐）=====
constexpr uint32_t QFA_HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

// ===== ConstInfo（精简自主线 CommonConstInfo：去 mask/PA/sink/tensorList/strides）=====
struct ConstInfo {
    // shape
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t realN2Size; // DN 语义（不合轴）：n2Size * gSize
    uint32_t realGSize;  // DN 语义（不合轴）：1
    uint64_t s1Size;
    uint64_t s2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    // seqlen 输入尺寸
    uint64_t cuSeqLensQSize;
    uint64_t cuSeqLensKVSize;
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;
    // softmax scale
    float scaleValue;
    // 核信息
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx; // AIV 子核编号：0→V0（S1 前半），1→V1（S1 后半）
    uint32_t coreNum;
    // 其他
    bool isSoftmaxLseEnable;
    bool needInitOutput;
};

// ===== RunInfo（设计文档 §4.1 全字段）=====
struct RunInfoMxfp8SoftmaxFp16 {
    // ==================== 基础字段 ====================

    uint32_t loop = 0; // 全局任务编号，用于 PRELOAD_TASK_CACHE_SIZE 取模索引
    uint32_t mloop = 0; // 当前处理第几行 [bN2, gS1]，UpdateAxisInfo 中递增，softmaxStateSlot = mloop % (PRELOAD_N+1)
    bool isValid = false; // 当前 RunInfo 槽位是否有效

    uint32_t bIdx = 0;      // batch 索引
    uint32_t n2Idx = 0;     // head 索引
    uint32_t realN2Idx = 0; // realN2 索引
    uint32_t gS1Idx = 0;    // GS1 起始索引
    uint32_t s1Idx = 0;     // S1 轴 token 索引
    uint32_t s2Idx = 0;     // S2 轴起始 token 索引
    uint64_t actS1Size = 1; // 当前 batch S1 实际长度
    uint64_t actS2Size = 1; // 当前 batch S2 实际长度
    uint32_t actMSize = 0;  // 当前切块 M 轴实际长度（≤ S1_BASE_SIZE）
    uint32_t actMSizeAlign32 = 0;

    uint32_t actSingleLoopS2Size = 0;      // 当前 S2 分块实际长度
    uint32_t actSingleLoopS2SizeAlign = 0; // 对齐到 32（BYTE_BLOCK / sizeof(fp8)）

    // ==================== V 核视角 ====================

    // 本 V 核在 [bN2, gS1] 行的 S1 轴起始偏移
    // V0: vecS1BaseIdx = 0，V1: vecS1BaseIdx = S1_BASE_SIZE / 2 = 128
    uint32_t vecS1BaseIdx = 0;

    // 本 V 核处理的 S1 轴实际长度（截断到 S1_BASE_SIZE/2 = 128）
    uint32_t actVecS1Size = 0;

    bool isFirstS2Loop =
        false; // 是否当前 [bN2, gS1] 行的首个 S2 分块（用于 Q/QScale 搬入、V1 accMax 初始化、V2 直接赋值分支）
    bool isLastS2Loop = false; // 是否当前 [bN2, gS1] 行的最后一个 S2 分块（V2 做最终除+输出）

    // ==================== softmax 在线状态（V1 维护） ====================

    // V1 维护 accMax 的 ring 槽位（跨 subLoop 与跨 S2 分块在线累积），索引 = mloop % (PRELOAD_N+1)
    // softmaxExpUB 独立按 loop % 3 索引，V1 每任务替换写入
    uint32_t softmaxStateSlot = 0;

    // ==================== P L1 ring 槽位（CubeBlock C2 读取 / VectorBlock V1 写入） ====================

    // P 按 S2 分块写入（同 [bN2,gS1] 行的不同 S2 块用不同槽，避免覆盖），索引 = loop % (PRELOAD_N+1)
    uint32_t pSlot = 0;
};

constexpr uint32_t L1_P_BUFCNT = PRELOAD_TASK_CACHE_SIZE * 2U; // = 6（3 任务 × 2 subLoop）

constexpr uint32_t L1_P_SINGLE_SLOT_SIZE = 128 * 128; // fp8 [128,128] = 16KB（subLoop 粒度）
// e8m0 scale 网格槽（subLoop 粒度）：[8 x 单元（S1 16 列/组 × 2B 对）][2 y 单元 + 1 pad] × 32B
// = 768B；y 单元 = 64 S2 行组，读侧 yStart 恒 0（subK → 槽选择）
constexpr uint32_t L1_PSCALE_SINGLE_SLOT_SIZE = 768;

// pscale 全前置（AIV 直写窗口内——pscale 影响所有任务，必须全部可写）
constexpr uint32_t L1_PSCALE_V0_BASE = 0;
constexpr uint32_t L1_PSCALE_V1_BASE = L1_PSCALE_V0_BASE + L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE;
constexpr uint32_t L1_P_SLOTS_BASE = L1_PSCALE_V1_BASE + L1_P_BUFCNT * L1_PSCALE_SINGLE_SLOT_SIZE;
// P 数据槽核间交错寻址（ring = loop×2+subLoopIdx，周期 6）：
// 槽序 v0r0,v1r0,v0r1,v1r1,…——窗口内前 7 槽，loop 0 双核全保
// common_def 同被 op_host（g++）引用，函数需 __aicore__ 标注而 host 无此属性——用宏
#define L1_PSLOT(core, ring) (L1_P_SLOTS_BASE + ((ring) * 2U + (core)) * L1_P_SINGLE_SLOT_SIZE)
// P ring 槽号（loop×2+subLoopIdx 自动 mod 6：loop 周期 3）
#define L1_PRING(loop, subLoop) (((loop) * 2U + (subLoop)) % 6U)
constexpr uint32_t L1_SHARED_REGION_SIZE = L1_P_SLOTS_BASE + 2U * L1_P_BUFCNT * L1_P_SINGLE_SLOT_SIZE;

// ===== V 核 UB 跨核契约（AIC Fixpipe 跨核写入 / VectorBlock 分配与消费，两侧同源引用）=====
// 单位约定（对齐 LocalTensor(TPosition, addr, size)：addr=字节地址、size=元素数）：
//   BASE 为字节地址；SLOT/SIZE 为元素数；mm1ResUB ring 槽位 = loop % UB_MM1RES_BUFCNT
// mm1ResUB：C1 FixpipeMm1 的目标（subBlockId 寻址核），V1 softmax 的输入
// mm2ResUB：C2 FixpipeMm2 的目标，V2 的输入（divOut/outputT 别名区见 block_vector 设计）
constexpr uint32_t UB_MM1RES_BASE = 0;               // 字节
constexpr uint32_t UB_MM1RES_SLOT = 128 * 128;       // half 元素/槽 [128,128] = 32KB
constexpr uint32_t UB_MM1RES_BUFCNT = 2;             // ring：loop % 2
constexpr uint32_t UB_MM2RES_BASE = 64 * 1024;       // 字节（= mm1ResUB 2 槽 × 32KB 之后）
constexpr uint32_t UB_MM2RES_SIZE = (128 + 1) * 128; // fp32 元素 [129,128] = 66,048B

#endif // QUANT_FLASH_ATTN_COMMON_DEF_H_
