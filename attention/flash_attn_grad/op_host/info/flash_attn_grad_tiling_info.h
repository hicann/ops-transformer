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
 * \file flash_attn_grad_tiling_info.h
 * \brief L1 契约层：输入/属性索引常量，以及 tiling 各阶段之间传递的 POD。
 *
 * 分层约定：
 *   L2 parser 是唯一读 gert::TilingContext 取轴与属性的地方，结果写入
 *   FagParsedInfo；L4 策略层只读 FagParsedInfo、只产出各自的 Plan，
 *   不得回写解析字段，也不得触碰 context。
 */

#ifndef FLASH_ATTN_GRAD_TILING_INFO_H_
#define FLASH_ATTN_GRAD_TILING_INFO_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace optiling {

// ---------------------------------------------------------------------------
// 输入 / 属性索引：全算子唯一定义处，禁止在别处再写字面量下标
// ---------------------------------------------------------------------------
constexpr size_t Q_INDEX = 0;
constexpr size_t K_INDEX = 1;
constexpr size_t V_INDEX = 2;
constexpr size_t DOUT_INDEX = 3;
constexpr size_t ATTN_OUT_INDEX = 4;
constexpr size_t SOFTMAX_LSE_INDEX = 5;
constexpr size_t CU_SEQLENS_Q_INDEX = 6;
constexpr size_t CU_SEQLENS_KV_INDEX = 7;
constexpr size_t SEQUSED_Q_INDEX = 8;
constexpr size_t SEQUSED_KV_INDEX = 9;
constexpr size_t SINKS_INDEX = 10;
constexpr size_t ATTN_MASK_INDEX = 11;
constexpr size_t METADATA_INDEX = 12;

constexpr size_t DQ_OUT_INDEX = 0;
constexpr size_t DK_OUT_INDEX = 1;
constexpr size_t DV_OUT_INDEX = 2;
constexpr size_t DSINK_OUT_INDEX = 3;

constexpr size_t ATTR_SOFTMAX_SCALE_INDEX = 0;
constexpr size_t ATTR_MASK_MODE_INDEX = 1;
constexpr size_t ATTR_WIN_LEFT_INDEX = 2;
constexpr size_t ATTR_WIN_RIGHT_INDEX = 3;
constexpr size_t ATTR_MAX_SEQLEN_Q_INDEX = 4;
constexpr size_t ATTR_MAX_SEQLEN_KV_INDEX = 5;
constexpr size_t ATTR_LAYOUT_Q_INDEX = 6;
constexpr size_t ATTR_LAYOUT_KV_INDEX = 7;
constexpr size_t ATTR_LAYOUT_OUT_INDEX = 8;

// ---------------------------------------------------------------------------
// 取值常量
// ---------------------------------------------------------------------------
constexpr int64_t MASK_MODE_NO_MASK = 0;
constexpr int64_t MASK_MODE_CAUSAL = 3;
constexpr int64_t MASK_MODE_WINDOW = 4;
// mask_mode 3/4 映射到 kernel 使用的 SparseType，写入 TilingData。
constexpr int64_t FAG_SPARSE_TYPE_DENSE = 0;
constexpr int64_t FAG_SPARSE_TYPE_CASUAL = 1;
constexpr int64_t FAG_SPARSE_TYPE_BAND = 2;
constexpr int64_t ATTN_MASK_DIM = 2048;

// tilingkey bit[1:0] 的取值。2 与 3 是保留位，host 永不产生（见 route 的兜底断言）。
constexpr int64_t TMPL_BN2GS1S2 = 0;
constexpr int64_t TMPL_BN2 = 1;
constexpr int64_t TMPL_RESERVED_UNUSED = 2;
constexpr int64_t TMPL_RESERVED_BN2S2 = 3;

// tilingkey bit2：BNSD 不占位，运行期靠视图参数区分，只有 TND 置 1。
constexpr int64_t LAYOUT_NOT_TND = 0;
constexpr int64_t LAYOUT_TND = 1;

constexpr int64_t MAX_HEAD_DIM = 768;

// ---------------------------------------------------------------------------
// L2 解析结果。策略层对它只读。
// ---------------------------------------------------------------------------
struct FagParsedInfo {
    // 平台
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    size_t libapiWorkspaceSize = 0;
    // swizzle 判据的分母。各 ascend950 变体从 16MB(950PR_950z) 到 128MB(9599)
    // 不等，必须来自平台而非硬编码；取不到时为 0，此时 swizzle 一律不开。
    uint64_t l2CacheSize = 0;

    // 布局
    std::string layoutQ = "BSND";
    int64_t layout = LAYOUT_NOT_TND;
    bool isTnd = false;
    // 等长 TND（所有 batch 的 seq 长度相同，且与 maxSeqlen 一致）的 GM 排布
    // 与 BSND 四维视图逐元素重合，host 仅对 tndAllSame 选择 BN2（layout=1
    // 二进制，lse 走 TND 2D 视图），变长 TND 由 route 挡回 GS1S2 兜底。
    // 由 ParseShapes 用 shape 算术判定（T1==b*s1 且 cu_seqlens 差分全等）。
    bool tndAllSame = false;
    bool isBnsd = false;

    // 轴
    int64_t b = 0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    // 物理 token 总数：TND 取 Q/K 的 T 轴，BSND/BNSD 为 B*S。
    // workspace 按物理存储分配；seqused 仅限制计算，不能缩小这两个值。
    // 仅 host 内部使用，不新增 TilingData ABI 字段。
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t n1 = 0;
    int64_t n2 = 0;
    int64_t g = 0;
    int64_t d = 0;
    int64_t dv = 0;

    // 属性
    float scaleValue = 0.0F;
    int64_t maskMode = MASK_MODE_NO_MASK;
    int64_t winLeft = -1;
    int64_t winRight = -1;
    int64_t sparseType = FAG_SPARSE_TYPE_DENSE;
    int64_t s1Token = 0;
    int64_t s2Token = 0;
    int64_t totalPerBatchNum = 0;
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;

    // 可选输入是否存在
    bool hasAttenMask = false;
    bool hasSeqused = false;
    bool hasSequsedQ = false;
    bool hasSequsedKv = false;
    // sinks 是否提供。占 tilingkey bit15（两种 template 都放开）：kernel 按该
    // 编译期常量决定是否算 dsink，非 sink 变体里相关代码被折叠掉。命中时按
    // head 在本 AIV 的 UB 累加器里精确累加，循环末尾写成槽位区本 AIV 行，
    // post 阶段由 AIV0 固定次序归约后整行写 dsink
    // （见 op_kernel/fag_block_vec.py 的 process_sink_partial/reduce）。
    bool hasSinks = false;
    // context->GetDeterministic()==1。BN2 单块仍走现网（tilingkey deter=0）。
    bool isDeterministic = false;

    // 布局视图参数：把 BSND/BNSD 折叠成同一种四维紧凑视图，使 kernel 只需
    // 一份 pl.load。推导见 parser 的 FillLayoutView。
    int64_t viewD0Q = 0;
    int64_t viewD2Q = 0;
    int64_t coefB0Q = 0;
    int64_t viewD0KV = 0;
    int64_t viewD2KV = 0;
    int64_t coefB0KV = 0;
    int64_t coefN0 = 0;
    int64_t coefN2 = 0;
};

// ---------------------------------------------------------------------------
// L4 各策略的产出
// ---------------------------------------------------------------------------
struct FagKernelPlan {
    int64_t tmpl = TMPL_BN2GS1S2;
    int64_t isBn2MultiBlk = 0;
    // BN2 MultiBlk + causal/band 且存在整行/整列无效 tile 时，kernel pre 清零 out。
    int64_t bn2NeedZero = 0;
    int64_t dAlign = 128;
    int64_t dvAlign = 128;
};

struct FagSchedulePlan {
    int64_t s1Outer = 0; // AICPU 口径的镜像值，仅用于推算核数
    int64_t s2Outer = 0;
    uint32_t usedCube = 0;
    uint32_t blockDim = 0;
    bool useActualCores = false;
    int64_t swizzle = 0;
    int64_t deterMaxRound = 0;
};

// workspace 具名分区。offset 是相对 kernel 拿到的 workspace 指针的字节偏移，
// 必须与 kernel 侧 make_ptr 的切分逐一对应。
struct FagWorkspaceSegment {
    size_t offset = 0;
    size_t size = 0;
};

struct FagWorkspacePlan {
    size_t libapiSize = 0;
    FagWorkspaceSegment dq; // T1*N1*D  fp32（N1 = N2*G，非 TND 的 T1=B*S1）
    FagWorkspaceSegment dk; // T2*N2*D  fp32（非 TND 的 T2=B*S2）
    FagWorkspaceSegment dv; // T2*N2*Dv fp32
    // dsink 槽位：[aivNum x N1] fp32。每 AIV 一行，post 由 AIV0 归约。
    FagWorkspaceSegment dsink;
    size_t totalSize = 0;
};

struct FagTilingPlan {
    FagKernelPlan kernel;
    FagSchedulePlan schedule;
    FagWorkspacePlan workspace;
    uint64_t tilingKey = 0;
};

// ---------------------------------------------------------------------------
// 决策追溯。目的是让"今天为什么选了 A"可以直接从日志读出来，而不必逆向条件树。
// ---------------------------------------------------------------------------
constexpr size_t FAG_TRACE_SLOT_NUM = 8;

struct FagRouteTrace {
    const char* chosen = "none";
    const char* rejectedName[FAG_TRACE_SLOT_NUM] = {};
    const char* rejectedReason[FAG_TRACE_SLOT_NUM] = {};
    size_t rejectedNum = 0;
    // swizzle 单独记：它不是候选之间的竞争，而是选中候选之后的一个开关。
    const char* swizzleReject = "accepted";

    void Reject(const char* name, const char* reason)
    {
        if (rejectedNum < FAG_TRACE_SLOT_NUM) {
            rejectedName[rejectedNum] = name;
            rejectedReason[rejectedNum] = reason;
            ++rejectedNum;
        }
    }
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_INFO_H_
