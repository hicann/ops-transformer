/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_swizzle.h"

#include <set>

namespace optiling {
namespace {

// 基本块尺寸。与 flash_attn_grad_block_outer.h 的 S1_INNER*S1CV_RATIO(64*2) 和
// S2_INNER*S2CV_RATIO(128*1) 必须相等 —— 这里估算的 s1Outer/s2Outer 要和分核
// 实际使用的口径一致，否则 liveHeads 估的就不是 metadata 真正的分核形态。
constexpr int64_t CUBE_BASE_M = 128;
constexpr int64_t CUBE_BASE_N = 128;
constexpr int64_t FP16_BYTES = 2;
constexpr int64_t FP32_BYTES = 4;

} // namespace

void DecideSwizzle(const FagParsedInfo &info, const FagKernelPlan &kernel, FagSchedulePlan &schedule,
                   FagRouteTrace &trace)
{
    // 决定块到核的映射方式。线性切分会让同时在跑的核散落在很多 head 上：
    // S=8192 D=128、36 核时，核 0 在 head 0 而核 35 已经在 head 31，于是 36 个核
    // 同时为 32 个 head 拉 K/V，工作集冲过 L2（标定机型 9599 上是 128MB，实际值
    // 取自平台，见 info.l2CacheSize）。msprof 显示 cube 与 vector
    // 两侧 mte2_ratio 都是 0.72，而 mac 反而从 0.938 掉到 0.733 —— 说明瓶颈已经
    // 变成 HBM 带宽而非算力。swizzle 改为按整列 s2 跨核发放，把同时在跑的核锁在
    // 同一个 head 内。
    //
    // 工作集要算一个 head
    // 碰到的所有张量（S=8192 时约 22MB/head）：q+dout+attn_out 读 fp16、dq 写 fp32、
    // k+v 读 fp16、dk+dv 写 fp32。只算 K+V 的话在 N=32 时正好落在 128MB 上并误判
    // 成"放得下"，而实测那里 swizzle 快 34%。
    //
    // swizzle 的分配粒度是一整列，列数太少就没法均衡（N=1 时 64 列分 36 核，
    // 核 0 拿 2 列而核 35 拿 1 列，2 倍倾斜，实测反慢 14%）。
    schedule.swizzle = 0;

    if (kernel.tmpl != TMPL_BN2GS1S2) {
        trace.swizzleReject = "not-bn2gs1s2";
        return;
    }
    if (info.layout == LAYOUT_TND) {
        // TND swizzle 是另一套实现（列前缀和在 AICPU），本轮不接。
        trace.swizzleReject = "tnd-layout";
        return;
    }
    if (schedule.usedCube != info.aicNum) {
        trace.swizzleReject = "partial-chip";
        return;
    }
    // 判据整个建立在"工作集是否超 L2"上，L2 未知就没有判据可言。部分型号
    // （如 KirinX90）平台上报 l2_size=0，此时保守不开：swizzle 只是性能选择，
    // 不开一定正确，而拿 0 当分母会让阈值退化成 0、变成恒开。
    if (info.l2CacheSize == 0) {
        trace.swizzleReject = "unknown-l2";
        return;
    }

    const int64_t s1Outer = (info.s1 + CUBE_BASE_M - 1) / CUBE_BASE_M;
    const int64_t s2Outer = (info.s2 + CUBE_BASE_N - 1) / CUBE_BASE_N;
    // GQA 下块网格是 b*n2*G*s1Outer*s2Outer，融合后的 batch 维是 b*n2*G，
    // 所以 G 在这里只是把 head 数乘上去（kernel 在 n2 内层走 g）。
    // mask 用 valid tiles（GetTotalPerBatchNum），让 L2 门看到真实工作集。
    const int64_t perHeadBlocks = info.totalPerBatchNum > 0 ? info.totalPerBatchNum : (s1Outer * s2Outer);
    const int64_t fusedHeads = info.b * info.n2 * info.g;
    const int64_t totalBlocks = fusedHeads * perHeadBlocks;
    if (perHeadBlocks <= 0 || info.aicNum == 0) {
        trace.swizzleReject = "degenerate-grid";
        return;
    }

    // 必须用 aicNum 而不是 blockDim：blockDim 是 CalcTschBlockDim 出来的混合调度
    // 值，而 kernel 里的 coreNum 来自 pl.get_block_num()（启动的 cube 数）。
    const int64_t cores = static_cast<int64_t>(info.aicNum);
    const int64_t perCore = (totalBlocks + cores - 1) / cores;
    // 线性切分下同时活跃的 head 有几个
    std::set<int64_t> liveHeads;
    for (int64_t c = 0; c < cores; ++c) {
        liveHeads.insert((c * perCore) / perHeadBlocks);
    }
    // 每个融合 (b,n2,g) 单元：Q 侧张量（q+dout+attn_out 读 fp16、dq 写 fp32）是本
    // g 私有的，KV 侧（k+v 读 fp16、dk+dv 写 fp32）被同一 KV head 的 G 个 Q head
    // 共享，故 KV 字节按 G 摊薄只记一份。
    const int64_t perHeadBytes = 3 * info.s1 * info.d * FP16_BYTES + info.s1 * info.d * FP32_BYTES +
                                 (2 * info.s2 * info.d * FP16_BYTES + 2 * info.s2 * info.d * FP32_BYTES) / info.g;
    const int64_t liveBytes = static_cast<int64_t>(liveHeads.size()) * perHeadBytes;

    // 判据取决于 S1 与 S2 的**方向**，不只是工作集大小。swizzle 买到的是"一个核
    // 走完一整列 s1Outer 块"，即 K/V 只载一次、dK/dV 在 L0C 连续累加；付出的是更
    // 粗的分配粒度（一列 = s1Outer 块）。这个权衡随列长而变。
    //
    // 1) S2 更长时（s2Outer > s1Outer）swizzle 从不获胜：列只有 s1Outer 块，复用
    //    窗口短，而列数 b*n*s2Outer 已经大到让线性切分自然待在一个 head 内。
    //    实测 0/5 胜（36 核，D=128，linear/swizzle us）：
    //      N8  1024/8192 106MB 237.2/250.3   N16 1024/8192 212MB 491.9/492.1
    //      N16 2048/8192 232MB 848.2/858.4   N24 1024/8192 318MB 723.1/728.0
    //      N32  512/8192 404MB 629.3/632.1
    //    后两个是 318/404MB 的工作集，远超任何阈值却仍是线性更快 —— 所以需要独立
    //    的方向判据，调阈值救不回来。
    //
    // 2) S1 占优时（s1Outer >= 2*s2Outer）拐点来得更早，阈值降到 1.25x L2。
    //    实测边界落在 138MB(线性) 与 172MB(swizzle) 之间：
    //      N12 8192/512  129MB 194.9/245.9   N12 8192/1024 138MB 330.0/353.0
    //      N24 4096/512  138MB 201.2/201.6   N16 8192/512  172MB 277.4/263.7
    //      N16 8192/1024 184MB 509.3/474.3   N20 8192/1024 230MB 781.8/583.1
    //      N24 8192/1024 276MB 994.6/698.5   N32 8192/512  344MB 725.9/523.4
    //    1.1x~1.4x 是同一个平台期（误判数相同），取中点 1.25x。
    //
    // 3) 对称情形保持原来的 2x L2，它是在 N=8/10/12/16/24 上标定的，不能回退：
    //      N=8 176MB 1556/1761, N=10 220MB 1904/2007, N=12 264MB 2833/2677,
    //      N=16 352MB 4121/3550, N=24 528MB 6483/4982
    //    N=10 在 220MB 仍偏好线性，而 S1 重的分支在 172MB 就该切换 —— 两个拐点
    //    确实不同，一个全局阈值服务不了两者。
    //
    // 全部 29 个实测点上：方向无关的旧 2x 规则误判 7 个（累计代价 67.7%），
    // 本判据误判 2 个（18.0%）。kernel 只消费 tilingkey 的 swizzle 位。
    if (s2Outer > s1Outer) {
        trace.swizzleReject = "s2-dominant";
        return;
    }
    // 1.25x 用整数表达，避免浮点：5*L2/4。
    // 阈值的两个倍数（1.25x / 2x）是在 128MB L2 的 Ascend950PR_9599 上标定的；
    // L2 更小的型号（950PR_950z 16MB、950DT_950x 32MB、957d 96MB、多个 112MB）
    // 按同样的倍数缩放，倍数本身尚未在那些型号上复标。
    const int64_t l2Bytes = static_cast<int64_t>(info.l2CacheSize);
    const bool s1Heavy = (s1Outer >= 2 * s2Outer);
    const int64_t threshold = s1Heavy ? (5 * l2Bytes / 4) : (2 * l2Bytes);
    if (liveBytes <= threshold) {
        trace.swizzleReject = "working-set-fits";
        return;
    }
    if (fusedHeads * s2Outer < 2 * cores) {
        trace.swizzleReject = "too-few-columns";
        return;
    }

    schedule.swizzle = 1;
    trace.swizzleReject = "accepted";
}

} // namespace optiling
