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
 * \file und_gen_qkv_rms_norm_rope_cache_mrope_vf.h
 * \brief UndGenQkvRmsNormRopeCache 的 Reg 矢量（VF）计算段：RMSNorm + MRoPE 融合
 *
 * 设计要点（三轴 cos_sin 不落 UB，用 Gather 在寄存器里合并）：
 *   1) axisLut 做成 gather 索引 gatherIndex[D/2]，每核 Init 期建一次：
 *        gatherIndex[lane] = axis(lane) * D + lane
 *      其中 axis(lane) 的规则与竞品 _mrope / golden.py:mrope_axis_map 完全一致
 *      （见 BuildMropeGatherIndex，注意 mrope_section[0] 不参与判断）。
 *   2) CopyIn 已经把每 token 的三轴 cos_sin 原样搬成 UB 上连续的 [3, D] 窗口
 *      （行序 T/H/W），VF 里只需两条 Gather：
 *        cos = Gather(rawToken,            gatherIndex)   // rawToken[axis*D + lane]
 *        sin = Gather(rawToken + D/2,      gatherIndex)   // rawToken[axis*D + D/2 + lane]
 *      同一份索引复用于 cos 和 sin，靠基址 +D/2 区分——因为 cos_sin_cache 每行的
 *      布局就是 [cos(0:D/2) | sin(0:D/2)]。
 *   3) 合并结果只存在 vreg 里，不占 UB。
 *
 * V 段（不参与 norm/rope）也并在同一个 VF 里：它与 Q/K 共用同一趟 token 循环，
 * 单个 V head 恰好是一个满的 bf16 VL（D=128 个 bf16 = 256B），一次 Load + 一次 Store 直通。
 *
 * 约束：本文件假定 D/2 恰好等于一个 float VL（950 上 VL=256B → 64 个 float，即 D=128），
 *       所有 RegTensor 操作都按「半宽 = 一个满 VL」写死，没有分段循环也没有尾块 mask。
 *       V 段的「一个 head = 一个满 bf16 VL」是同一个不变量的等价表述（D = 2 * (D/2)）。
 *       该不变量由 host 侧的 static_assert 绑定，见 op_host/..._tiling.h 的 VL_FP32_LANES；
 *       放宽 headDim 前必须先在这里补分段循环与尾块 mask。
 */

#ifndef UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_
#define UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace UndGenQkvRmsNormRopeCache {
namespace MicroAPI = AscendC::MicroAPI;

constexpr static MicroAPI::CastTrait CAST_B16_TO_B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                        MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

constexpr static MicroAPI::CastTrait CAST_FP32_TO_B16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                         MicroAPI::MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_RINT};

/**
 * @brief 本次调用的 UB 起址，逐 tile 变化
 *
 * 全部要求 32B 对齐。qkvIn 一行是 [N][D]，Q 头在前、K 头次之、V 头最后；
 * 输出的 q/k/v 三段各自独立，由调用方按 outLocal 的分段布局给出。
 */
struct QkvMropeTileAddr {
    __ubuf__ bfloat16_t *qkvIn;     // tile 内首 token 的输入行首，[token][N][D]
    __ubuf__ float *gammaAll;       // 4 份常驻 gamma，[undQ|undK|genQ|genK][D]，各 D 个 float
    __ubuf__ float *rawCosSin;      // 首 token 的三轴原始 cos_sin，[token][3][D]
    __ubuf__ uint32_t *gatherIndex; // axisLut 展开成的元素索引，[D/2]
    __ubuf__ bfloat16_t *qOut;      // 输出 q 段起址，[token][Hq][D]
    __ubuf__ bfloat16_t *kOut;      // 输出 k 段起址，[token][Hk][D]
    __ubuf__ bfloat16_t *vOut;      // 输出 v 段起址，[token][Hv][D]
};

/**
 * @brief head 数、各段步长与 RMSNorm 标量
 *
 * 全部由 shape 与 attr 决定，整个 kernel 生命周期不变，调用方在 Init 期填一次即可，
 * 不需要每个 tile 重算。步长单位一律是「元素」而非字节。
 */
struct QkvMropeTileShape {
    uint16_t qHeadNum;          // Hq
    uint16_t kHeadNum;          // Hk
    uint16_t vHeadNum;          // Hv
    uint32_t inTokenStride;     // 输入 token 步长 = N*D
    uint32_t cosSinTokenStride; // cos_sin token 步长 = 3*D
    uint32_t qOutTokenStride;   // q 段 token 步长 = Hq*D
    uint32_t kOutTokenStride;   // k 段 token 步长 = Hk*D
    uint32_t vOutTokenStride;   // v 段 token 步长 = Hv*D
    uint32_t headStride;        // head 步长 = D，输入与输出相同
    uint32_t halfDim;           // D/2，必须等于 VL_FP32
    float epsilon;              // RMSNorm eps
    float reciprocal;           // 1/D
};

/**
 * @brief 一个 token 内所有 head 共用的寄存器
 *
 * cos/sin 每 token 只 Gather 一次、gamma 每 token 只 Load 一次，Q/K 的 head 循环
 * 全程复用，所以打包成一束按引用传给 per-head 函数，避免参数列表铺开。
 */
struct MropeTokenRegs {
    MicroAPI::RegTensor<float> cosValue;
    MicroAPI::RegTensor<float> sinValue;
    MicroAPI::RegTensor<float> sinNeg; // -sin，用于把低半 RoPE 的减法折进乘加
    MicroAPI::RegTensor<float> qGammaLow;
    MicroAPI::RegTensor<float> qGammaHigh;
    MicroAPI::RegTensor<float> kGammaLow;
    MicroAPI::RegTensor<float> kGammaHigh;
};

/**
 * @brief 单个 head 的 "RMSNorm -> MRoPE 旋转 -> Cast bf16 写回"
 *
 * Q 和 K 的差别只在 gamma 与输出位置，计算本身完全一样，抽出来给两个 head
 * 循环共用；gamma 用哪一份由调用方从 regs 里挑好后传进来。
 *
 * 乘加一律走 MulAddDst（dst = src0 * src1 + dst）：一条顶 Mul + Add/Sub 两条，
 * 乘积保持全精度只舍入一次，中间量也少占一个 RegTensor。
 * 低半的 RoPE 要的是 dst -= a*b 而乘加原语只有加法版，所以用 regs.sinNeg
 * ——sin 取负是每 token 一次，摊到 Hq+Hk 个 head 上可忽略。
 *
 * NOTE: inHead / outHead 按引用传入，函数内用 post-update 访存把它们推到**下一个 head**。
 *       低半、高半两次访存各推 halfDim，两次之后正好走过 headStride（= D = 2*halfDim），
 *       所以调用方的 head 循环不需要再算 h * headStride，直接反复调用即可。
 */
__aicore__ inline void RmsNormRopeOneHead(__ubuf__ bfloat16_t *&inHead, __ubuf__ bfloat16_t *&outHead,
                                          MicroAPI::RegTensor<float> &gammaLow, MicroAPI::RegTensor<float> &gammaHigh,
                                          MropeTokenRegs &regs, const QkvMropeTileShape &shape, MicroAPI::MaskReg pFull)
{
    const int32_t halfStep = static_cast<int32_t>(shape.halfDim);
    MicroAPI::RegTensor<bfloat16_t> xLowB16;
    MicroAPI::RegTensor<bfloat16_t> xHighB16;
    MicroAPI::RegTensor<float> xLow;
    MicroAPI::RegTensor<float> xHigh;
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        xLowB16, inHead, halfStep);
    MicroAPI::LoadAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        xHighB16, inHead, halfStep); // 两次之后 inHead 已落到下一个 head
    MicroAPI::Cast<float, bfloat16_t, CAST_B16_TO_B32>(xLow, xLowB16, pFull);
    MicroAPI::Cast<float, bfloat16_t, CAST_B16_TO_B32>(xHigh, xHighB16, pFull);

    // ---- RMSNorm：x / sqrt(mean(x^2) + eps) * gamma，中间量全在 vreg ----
    MicroAPI::RegTensor<float> squareLow;
    MicroAPI::RegTensor<float> rms;
    MicroAPI::Mul(squareLow, xLow, xLow, pFull);
    MicroAPI::MulAddDst(squareLow, xHigh, xHigh, pFull); // squareLow += x_hi^2
    // 两个半宽相加后再归约，一次 Reduce 覆盖整个 D
    MicroAPI::ReduceSum(squareLow, squareLow, pFull);
    MicroAPI::Muls(squareLow, squareLow, shape.reciprocal, pFull);
    MicroAPI::Adds(squareLow, squareLow, shape.epsilon, pFull);
    MicroAPI::Sqrt(rms, squareLow, pFull);
    MicroAPI::Duplicate(rms, rms, pFull); // 广播 lane0
    MicroAPI::Div(xLow, xLow, rms, pFull);
    MicroAPI::Div(xHigh, xHigh, rms, pFull);
    MicroAPI::Mul(xLow, xLow, gammaLow, pFull);
    MicroAPI::Mul(xHigh, xHigh, gammaHigh, pFull);

    // ---- MRoPE 旋转（half-split，cos/sin 已由 Gather 合并）----
    //   out_lo = x_lo * cos - x_hi * sin  ->  用 sinNeg 把减法折成乘加
    //   out_hi = x_hi * cos + x_lo * sin
    MicroAPI::RegTensor<float> outLow;
    MicroAPI::RegTensor<float> outHigh;
    MicroAPI::Mul(outLow, xLow, regs.cosValue, pFull);
    MicroAPI::MulAddDst(outLow, xHigh, regs.sinNeg, pFull); // outLow += x_hi * (-sin)
    MicroAPI::Mul(outHigh, xHigh, regs.cosValue, pFull);
    MicroAPI::MulAddDst(outHigh, xLow, regs.sinValue, pFull); // outHigh += x_lo * sin

    MicroAPI::RegTensor<bfloat16_t> outLowB16;
    MicroAPI::RegTensor<bfloat16_t> outHighB16;
    MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_B16>(outLowB16, outLow, pFull);
    MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_B16>(outHighB16, outHigh, pFull);
    MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
        outHead, outLowB16, halfStep, pFull);
    MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK_B32>(
        outHead, outHighB16, halfStep, pFull); // 两次之后 outHead 已落到下一个 head
}

/**
 * @brief 对一个 tile 的 Q/K/V 做 "RMSNorm -> MRoPE -> Cast bf16"（V 直通）融合计算，整 tile 一次 VF
 *
 * Q 和 K 放在同一个 VF 里：它们在 UB 上本来就是同一行 token 内相邻的 head，
 * 拆成两次调用会让每 token 的 cos/sin Gather 做两遍。合到一起后 token 循环只走一趟，
 * cos/sin 只 Gather 一次，Q/K 两段 head 循环共用。
 *
 * V 段并在同一趟 token 循环里：它不参与 norm/rope，但源地址就在 Q/K 之后、目的地址
 * 就在输出 tile 的 v 段，放在这里每 token 的输入行只遍历一次就能处理完整行。
 * T_QKV 与 T_CACHE 同为 bf16，纯直通不做任何转换。
 *
 * gamma 逐 token 选择：cat_indices 可以把 und/gen 任意交错（竞品用例就是
 * [0,5,1,6,2,7,3,4] 这种），同一 tile 内相邻 token 的 gamma 可能不同。
 * gamma 只有 und/gen 两套，在 UB 上是 [undQ|undK|genQ|genK] 连续排布，于是本 token
 * 用哪一组只取决于 undMask 的第 t 位：
 *     gammaRow = gammaAll + (1 - undBit) * 2 * headStride
 * 「选 gamma」因此是一次标量地址算术，VF 里没有分支，也不占任何向量指令。
 *
 * @param addr      本 tile 的 UB 起址，见 QkvMropeTileAddr
 * @param shape     head 数、步长与 RMSNorm 标量，见 QkvMropeTileShape（Init 期填一次）
 * @param tokenSize tile 内 token 数，尾块可小于 ubFactor
 * @param undMask   tile 内各 token 的 und/gen 标志位图，bit t = 1 表示第 t 个 token 取
 *                  und 的 gamma。由 CopyIn 顺带产出（那里本来就要读 cat_indices 算源行），
 *                  省掉一次 GM 标量读。tokenSize <= 64 由 host tiling 保证
 */
__aicore__ inline void QkRmsNormMropeTileVF(const QkvMropeTileAddr &addr, const QkvMropeTileShape &shape,
                                            uint16_t tokenSize, uint64_t undMask)
{
    if (tokenSize == 0) {
        return;
    }
    const int32_t halfStep = static_cast<int32_t>(shape.halfDim);
    const int32_t headStep = static_cast<int32_t>(shape.headStride);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        // V 段按 bf16 原宽处理：一个 head = D 个 bf16 = 一个满 VL，掩码 lane 数是 pFull 的两倍
        MicroAPI::MaskReg pFullB16 = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<uint32_t> mropeIndex;

        // gather 索引与 token 无关，整个 tile 只加载一次
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(mropeIndex, addr.gatherIndex);

        // 地址一律用 post-update 自走，循环体内没有任何 t*stride / h*stride 计算：
        //   inPtr    每 head 推 headStride，一个 token 走完 Q/K/V 三段正好是 (Hq+Hk+Hv)*D
        //            = inTokenStride，直接落到下一个 token 的行首，整个 tile 不用重取基址
        //   qOut/kOut/vOutPtr 各推自己那段的 head 数，一个 token 正好走过各自的 TokenStride
        // 例外两处：
        //   1) cos_sin 用 Gather，而 Gather 只有 (dst, baseAddr, index, mask) 一个原型，
        //      既没有 post-update 也没有 AddrReg 重载，只能保留显式指针每 token 推一次；
        //   2) gamma 的基址由 undMask 决定，不是单调前进，同样只能显式算。
        __ubuf__ bfloat16_t *inPtr = addr.qkvIn;
        __ubuf__ bfloat16_t *qOutPtr = addr.qOut;
        __ubuf__ bfloat16_t *kOutPtr = addr.kOut;
        __ubuf__ bfloat16_t *vOutPtr = addr.vOut;
        __ubuf__ float *cosSinPtr = addr.rawCosSin;
        // gen 的 gamma 在 UB 上正好排在 und 之后 2*D 处（[undQ|undK|genQ|genK]）
        const uint32_t genGammaOffset = static_cast<uint32_t>(headStep) * 2U;

        for (uint16_t t = 0; t < tokenSize; ++t) {
            MropeTokenRegs regs;

            // 三轴合并：一条 Gather 从 [3, D] 窗口里按 axisLut 挑出 D/2 个 lane。
            // 本 token 的 Q、K 所有 head 共用这一份 cos/sin
            MicroAPI::Gather(regs.cosValue, cosSinPtr, mropeIndex, pFull);
            MicroAPI::Gather(regs.sinValue, cosSinPtr + shape.halfDim, mropeIndex, pFull);
            cosSinPtr += shape.cosSinTokenStride;
            // 低半 RoPE 是 out_lo = x_lo*cos - x_hi*sin，而乘加原语只有加法版；
            // 这里取一次负，让 head 内可以用 MulAddDst 把减法折进乘加。
            // 每 token 一条，摊到 Hq+Hk 个 head 上可忽略
            MicroAPI::Muls(regs.sinNeg, regs.sinValue, -1.0f, pFull);

            // 选 gamma：und 组在 gammaAll 开头，gen 组在 +2*D 处，用位图算偏移即可，无需分支。
            // 组内四份（q低|q高|k低|k高）在 UB 上连续，同一个指针连推 4 次 halfDim 取完；
            // 最后一次自增走出组尾，下个 token 会重新定基址，丢掉即可
            const uint32_t undBit = static_cast<uint32_t>((undMask >> t) & 1ULL);
            __ubuf__ float *gammaPtr = addr.gammaAll + (1U - undBit) * genGammaOffset;
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                regs.qGammaLow, gammaPtr, halfStep);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                regs.qGammaHigh, gammaPtr, halfStep);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                regs.kGammaLow, gammaPtr, halfStep);
            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                regs.kGammaHigh, gammaPtr, halfStep);

            for (uint16_t h = 0; h < shape.qHeadNum; ++h) {
                RmsNormRopeOneHead(inPtr, qOutPtr, regs.qGammaLow, regs.qGammaHigh, regs, shape, pFull);
            }

            // K 头在同一行 token 内紧跟 Q 头之后，inPtr 已经停在那里
            for (uint16_t h = 0; h < shape.kHeadNum; ++h) {
                RmsNormRopeOneHead(inPtr, kOutPtr, regs.kGammaLow, regs.kGammaHigh, regs, shape, pFull);
            }

            // V 头紧跟 K 头之后，不参与 norm/rope：一个 head 恰好一个满 bf16 VL，直接搬
            for (uint16_t h = 0; h < shape.vHeadNum; ++h) {
                MicroAPI::RegTensor<bfloat16_t> vValue;
                MicroAPI::LoadAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vValue, inPtr, headStep);
                MicroAPI::StoreAlign<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                     MicroAPI::StoreDist::DIST_NORM_B16>(vOutPtr, vValue, headStep, pFullB16);
            }
        }
    }
}

} // namespace UndGenQkvRmsNormRopeCache

#endif // UND_GEN_QKV_RMS_NORM_ROPE_CACHE_MROPE_VF_H_
