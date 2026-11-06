/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_dedup_gate.h
 * \brief mega_moe dispatch 去重收益门的门禁分数：按量化精度与 combine 去重状态分四套多项式。
 */
#ifndef MEGA_MOE_DEDUP_GATE_H
#define MEGA_MOE_DEDUP_GATE_H

#include <cstdint>

#include "../../../op_kernel/arch35_92/common/mega_moe_gm_layout.h"

namespace MegaMoeImpl {

/*
 * 门禁分数的形状向量。各量的口径是硬约束，错一项即与标定数据脱钩。
 * 字段名取对外资料的命名，注释给出对应的 tilingData 来源：
 *   epWorldSize        资料 ep_world_size             <- compute.epWorldSize
 *   maxTokensPerRank   资料 num_max_tokens_per_rank   <- compute.numMaxTokensPerRank
 *   hidden             资料 hidden（即 x 的 dim1）     <- compute.h，注意不是 compute.hiddenDim
 *   intermediateHidden 资料 intermediate_hidden       <- GetActivationOutputWidth(compute.hiddenDim, actMode)
 *                      即激活输出宽度；本门只在 SwiGLU 下生效，故恒等于 compute.hiddenDim / 2
 *   globalMoeExpertNum 全局 routed 专家数             <- compute.moeExpertPerRank * compute.epWorldSize
 *   topK               资料 num_topk                  <- compute.topK
 */
struct DedupGateShape {
    int64_t epWorldSize;
    int64_t maxTokensPerRank;
    int64_t hidden;
    int64_t intermediateHidden;
    int64_t globalMoeExpertNum;
    int64_t topK;
};

/*
 * 门禁分数的数学形式与常数含义
 * ---------------------------
 * dispatch 去重的净收益 = 省掉的重复 token 通信量 - prepass 建表与按 (专家, 来源卡) 槽支付的固定开销。
 * 两项都随形状变化且相互耦合：重复率随 topK 与卡数非线性饱和；每行字节数由 hidden 与
 * intermediateHidden 决定；固定开销随专家数增长。净收益因此不是任何单一维度的单调函数，
 * 用单个维度的阈值无法表达（这也是本门不写成"某量超过多少就开"的原因）。
 * 门禁分数 F 是对「净收益符号」的回归模型：各轴先按中心-标度归一化到 1 量级，再在归一化轴上做
 * 低阶多项式展开，交叉项承载上述耦合；判决只取 F 的符号。各公式里的常数即该回归的归一化参数
 * 与系数，只在下面的标定域内有效，域外不外推。
 * F 除符号外没有物理含义：它不是耗时或收益的预测值，禁止把 F 的大小当收益幅度使用。
 */
namespace DedupGateDetail {

constexpr int64_t DEDUP_GATE_CALIBRATED_WORLD = 4;
constexpr int64_t HIDDEN_ALIGN_A8W8 = 32;
constexpr int64_t HIDDEN_ALIGN_A8W4 = 64;

// 精度分档：两档各有独立标定的两套公式。A8W8 的 ND/NZ 共用一档；其余 gmm 模式不在标定范围内。
enum class CalibratedPrecision : uint8_t {
    A8W8,
    A8W4,
    NOT_CALIBRATED
};

inline CalibratedPrecision ResolveCalibratedPrecision(uint8_t gmmMode)
{
    if (gmmMode == GMM_MODE_A8W8_ND || gmmMode == GMM_MODE_A8W8_NZ) {
        return CalibratedPrecision::A8W8;
    }
    if (gmmMode == GMM_MODE_A8W4_NZ) {
        return CalibratedPrecision::A8W4;
    }
    return CalibratedPrecision::NOT_CALIBRATED;
}

// 标定域。hidden 的对齐按精度分列，与标定批的形状约束一致。
inline bool IsShapeInCalibratedDomain(const DedupGateShape& s, CalibratedPrecision precision)
{
    if (precision == CalibratedPrecision::NOT_CALIBRATED) {
        return false;
    }
    const int64_t hiddenAlign = precision == CalibratedPrecision::A8W8 ? HIDDEN_ALIGN_A8W8 : HIDDEN_ALIGN_A8W4;
    return s.epWorldSize == DEDUP_GATE_CALIBRATED_WORLD && s.maxTokensPerRank >= 4 && s.maxTokensPerRank <= 8192 &&
           s.hidden >= 2048 && s.hidden <= 3200 && s.hidden % hiddenAlign == 0 && s.intermediateHidden >= 768 &&
           s.intermediateHidden <= 1536 && s.intermediateHidden % 128 == 0 && s.globalMoeExpertNum >= 32 &&
           s.globalMoeExpertNum <= 128 && s.globalMoeExpertNum % 4 == 0 && s.topK >= 8 && s.topK <= 16;
}

/*
 * A8W8、combine 去重关（8 项）。记 B=maxTokensPerRank、K=topK 取原值，
 * hN=(hidden-2048)/1152、iN=(intermediateHidden-1152)/384、
 * eN=(globalMoeExpertNum-80)/48、kN=(K-12)/4：
 *   F = B*K/6144 - 1.58 - 0.23*kN + hN*(1.30 - 1.12*eN^2) + iN*(0.14*(kN-eN) + 0.27*eN^2)
 * 首项用 B*K 原值而非归一化量，表达待发送行数随总 token 数线性增长。
 */
inline double CalcScoreA8W8CombineOff(const DedupGateShape& s)
{
    const double hN = (static_cast<double>(s.hidden) - 2048.0) / 1152.0;
    const double iN = (static_cast<double>(s.intermediateHidden) - 1152.0) / 384.0;
    const double eN = (static_cast<double>(s.globalMoeExpertNum) - 80.0) / 48.0;
    const double kN = (static_cast<double>(s.topK) - 12.0) / 4.0;
    const double eN2 = eN * eN;
    return static_cast<double>(s.maxTokensPerRank) * static_cast<double>(s.topK) / 6144.0 - 1.58 - 0.23 * kN +
           hN * (1.30 - 1.12 * eN2) + iN * (0.14 * (kN - eN) + 0.27 * eN2);
}

/*
 * A8W4、combine 去重关（8 项）。bN=(B-1024)/(B+1024)、hN=(hidden-2624)/576，iN/eN/kN 同上：
 *   F = bN + 0.221*eN^2 + 0.027*hN*iN - 0.123*eN*kN^2 + 0.068*eN^2*kN + 0.099*hN*eN^2
 *       + 0.292*hN*iN*kN^2 - 0.374*hN*iN*eN^2*kN^2
 * B 走饱和型 bN 而非线性项：W4 下每行字节减半，通信收益随 B 增长更早触顶。
 */
inline double CalcScoreA8W4CombineOff(const DedupGateShape& s)
{
    const double bN =
        (static_cast<double>(s.maxTokensPerRank) - 1024.0) / (static_cast<double>(s.maxTokensPerRank) + 1024.0);
    const double hN = (static_cast<double>(s.hidden) - 2624.0) / 576.0;
    const double iN = (static_cast<double>(s.intermediateHidden) - 1152.0) / 384.0;
    const double eN = (static_cast<double>(s.globalMoeExpertNum) - 80.0) / 48.0;
    const double kN = (static_cast<double>(s.topK) - 12.0) / 4.0;
    return bN + 0.221 * eN * eN + 0.027 * hN * iN - 0.123 * eN * kN * kN + 0.068 * eN * eN * kN + 0.099 * hN * eN * eN +
           0.292 * hN * iN * kN * kN - 0.374 * hN * iN * eN * eN * kN * kN;
}

/*
 * A8W8、combine 去重开（8 项，按 B 的一次式展开、不是比值形式）。hN=(hidden-2624)/576，iN/eN/kN 同上：
 *   c = 0.343*hN - 0.213*kN - 0.070*iN + eN^2*(-0.217 + 0.239*kN) - iN*(0.040*eN + 0.057*hN)
 *   F = (B - 256) + (B + 256)*c
 * combine 去重已把回传量压掉一遍，形状项只调制 B 的斜率与截距，故写成一次式。
 */
inline double CalcScoreA8W8CombineOn(const DedupGateShape& s)
{
    const double b = static_cast<double>(s.maxTokensPerRank);
    const double hN = (static_cast<double>(s.hidden) - 2624.0) / 576.0;
    const double iN = (static_cast<double>(s.intermediateHidden) - 1152.0) / 384.0;
    const double eN = (static_cast<double>(s.globalMoeExpertNum) - 80.0) / 48.0;
    const double kN = (static_cast<double>(s.topK) - 12.0) / 4.0;
    const double c =
        0.343 * hN - 0.213 * kN - 0.070 * iN + eN * eN * (-0.217 + 0.239 * kN) - iN * (0.040 * eN + 0.057 * hN);
    return (b - 256.0) + (b + 256.0) * c;
}

/*
 * A8W4、combine 去重开（10 项）。bN=(B-256)/(B+256)、q=64/(B+64)、hN=(hidden-2624)/576，iN/eN/kN 同上：
 *   F = bN + 0.026*kN + 0.158*eN + 0.041*iN*eN*kN + 0.289*hN + 0.026*hN*eN*kN
 *       - 0.272*hN*iN - 0.126*hN*iN*eN - 0.884*q*hN + 0.671*q*hN*iN
 * q 项只在 B 很小时有权重，使 F 沿 B 轴呈 U 形：B 的下界附近可能回到微正（例如 B=4 处约 +0.006）。
 * 这是标定模型本身的形态，不是系数笔误，改动前先看 UT 的逐形状判决向量。
 */
inline double CalcScoreA8W4CombineOn(const DedupGateShape& s)
{
    const double bN =
        (static_cast<double>(s.maxTokensPerRank) - 256.0) / (static_cast<double>(s.maxTokensPerRank) + 256.0);
    const double q = 64.0 / (static_cast<double>(s.maxTokensPerRank) + 64.0);
    const double hN = (static_cast<double>(s.hidden) - 2624.0) / 576.0;
    const double iN = (static_cast<double>(s.intermediateHidden) - 1152.0) / 384.0;
    const double eN = (static_cast<double>(s.globalMoeExpertNum) - 80.0) / 48.0;
    const double kN = (static_cast<double>(s.topK) - 12.0) / 4.0;
    return bN + 0.026 * kN + 0.158 * eN + 0.041 * iN * eN * kN + 0.289 * hN + 0.026 * hN * eN * kN - 0.272 * hN * iN -
           0.126 * hN * iN * eN - 0.884 * q * hN + 0.671 * q * hN * iN;
}

} // namespace DedupGateDetail

/*
 * 形状与量化模式是否落在门禁分数的标定域内。
 * 返回 false 表示域外：调用方必须回落旧判据，不得对域外形状取分数判决。
 */
inline bool IsDedupGateShapeCalibrated(const DedupGateShape& shape, uint8_t gmmMode)
{
    return DedupGateDetail::IsShapeInCalibratedDomain(shape, DedupGateDetail::ResolveCalibratedPrecision(gmmMode));
}

/*
 * 取对应精度与 combine 去重状态的门禁分数。
 * 前置条件：IsDedupGateShapeCalibrated(shape, gmmMode) 为 true。
 */
inline double CalcDedupGateScore(const DedupGateShape& shape, uint8_t gmmMode, bool combineDedupOn)
{
    const DedupGateDetail::CalibratedPrecision precision = DedupGateDetail::ResolveCalibratedPrecision(gmmMode);
    if (precision == DedupGateDetail::CalibratedPrecision::A8W8) {
        return combineDedupOn ? DedupGateDetail::CalcScoreA8W8CombineOn(shape) :
                                DedupGateDetail::CalcScoreA8W8CombineOff(shape);
    }
    if (precision == DedupGateDetail::CalibratedPrecision::A8W4) {
        return combineDedupOn ? DedupGateDetail::CalcScoreA8W4CombineOn(shape) :
                                DedupGateDetail::CalcScoreA8W4CombineOff(shape);
    }
    // 标定精度之外只可能来自违反前置条件的调用；"精度未标定"这件事由 IsDedupGateShapeCalibrated 上报，
    // 此处取 0 让判决落到"不开"，不会把未标定形状放开。
    return 0.0;
}

/*
 * dispatch 去重的判决：门禁分数严格大于 0 才开。该阈值全仓只此一处，调用方不要自己拿分数比大小。
 * 分数恰为 0 的插值形状上，不同 FMA 收缩选项可能给出不同符号，故逐点判决不保证跨编译选项一致
 * （这类点全部落在标定格之外，且模型在该处预测收益为 0）。
 */
inline bool IsDispatchDedupBeneficial(double gateScore)
{
    return gateScore > 0.0;
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_DEDUP_GATE_H
