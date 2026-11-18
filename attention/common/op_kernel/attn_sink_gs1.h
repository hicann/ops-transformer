/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_sink_gs1.h
 * \brief gS1 合轴场景 sink 特性级 common: CopyIn(GM->UB 值区布局) + 注入链寄存器直写。
 * sinks 值类型为 float32(与 q dtype 解耦)。
 *
 * SinkCopyInGS1/SinkCopyInS1G/各直写 VF/SinkExpandMaxVf/SinkPrologueVF 均为各架构共用实现；
 * SinkPrologueGridVF 为 MXFP8 专用(网格化)变体。
 * VEC_M 为本 block 的每 vector 行数(softmax 槽 lane 数, 0 < VEC_M <= 128): arch35 为 mBaseSize/CV_RATIO,
 * arch92 为 mBaseSize。VEC_M > 64 时按 2 段展开, 尾段 lane 数 = VEC_M - 64。
 * Gs1/Rot/Gap 在 VEC_M < 64 时按 VEC_M 掩码(兼容槽容量等于 VEC_M 的调用方);
 * Brc/Contig 在 VEC_M <= 64 时按运行时 m 掩码。
 * 调用方须保证有效行数 actVecMSize <= VEC_M; 该约束不在运行期校验, 以免影响性能。
 *
 * 数学目标(Scheme B, prologue 注入): maxUb[r] = inUb[idx(r)], r < m(<= VEC_M)。
 *
 * idx(r) 依布局(值区布局由 SinkCopyInGS1/SinkCopyInS1G 决定, 直写函数只读不写):
 *   GS1(BNSD):       idx = (s0 + r) / actS1Size           —— SinkGs1DirectVF;
 *   S1G gSize==1:    恒同值, BRC 单值广播                  —— SinkBrcVF;
 *   S1G case1:       idx = r(值区连续)                     —— SinkContigDirectVF;
 *   S1G case2.1:     前 headLen 行 idx = headOff + r,
 *                    其余行 idx = r - headLen(gap 双段)    —— SinkGapDirectVF;
 *   S1G case2.2/3:   idx = (gStart + r) mod gSize(回绕)    —— SinkRotDirectVF。
 *
 * 数据流: 值区(float) 经 LoadAlign(BRC/连续)或 UB 基址 Gather(GS1/回绕/gap)取值 ->
 * (Compare/Select 拼合) -> 整段 StoreAlign 写 maxUb(唯一 REG->UB 落点), store mask 按 VEC_M 段展开。
 * 索引取模/整除用寄存器 Div 实现(无 Mod 指令): k = a - (a / d) * d。
 * 未用 lane 的 Gather 索引恒落值区槽内, 其 store 结果被 mask 掩掉, 不产生有效输出。
 */
#ifndef ATTN_SINK_GS1_H_
#define ATTN_SINK_GS1_H_

#include "kernel_operator.h"
#include "const_def.h"

using namespace AscendC;

namespace AttentionCommon {
using namespace Reg;

template <typename INPUT_T>
__aicore__ inline void SinkCopyInGS1(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start / actS1Size;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End / actS1Size;

    uint32_t gCount = gEnd - gStart + 1;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
    DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
}

template <typename INPUT_T>
__aicore__ inline void SinkCopyInS1G(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;

    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;

    if (s1Start == s1End) {
        // 情况1: 只有一个头块, 此时[gStart, gEnd]之间的行就是实际处理的M轴
        uint32_t gCount = gEnd - gStart + 1;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
    } else if (s1Start + 1 == s1End) {
        if (gStart > gEnd) {
            // 情况2.1: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart > gEnd, 此时它们不重叠.
            // 由于头尾块的长度都可能不对齐, 并且gSize有可能很大, 而UB要求32B对齐,
            // 约定先拷贝尾块的GM[0, gEnd]->UB[0, Align(gEnd + 1, 8) - 1],
            // 再拷贝头块的GM[gStart, gSize)->UB[Align(gEnd + 1, 8), Align(gEnd + 1, 8) + Align(gSize - gStart,
            // 8))
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = (gEnd + 1) * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);

            uint32_t ubOffset = AttentionCommon::Align(gEnd + 1, 8U);
            sinkCopyParams.blockLen = (gSize - gStart) * sizeof(INPUT_T);
            DataCopyPad(sinkTmpUb[ubOffset], sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);

        } else {
            // 情况2.2: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart <= gEnd, 此时它们重叠, 合并后的覆盖范围为[0, gSize)
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
        }
    } else {
        // 情况3: 头块 + 中间块 + 尾块, 合并后的覆盖范围为[0, gSize)
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
    }
}

/*!
 * \brief gSize==1 单值广播直写: maxUb[r] = inUb[0], r < VEC_M(<= 128)。
 * 前置: CopyIn 已将唯一 sink 值(float)放入 inUb[0]。BRC_B32 单值广播,
 * 段数由 VEC_M 编译期决定: VEC_M<=64 单段按 m 掩码; VEC_M>64 双段,
 * 尾段按 VEC_M-64 掩码(兼容非 64 对齐)。广播值对任意 lane 同值, VF 内无运行时分支。
 */
template <typename T, typename INPUT_T, uint32_t VEC_M>
__simd_vf__ inline void SinkBrcVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t m)
{
    static_assert(VEC_M > 0U && VEC_M <= 128U, "SinkBrcVF supports at most 2 stores (VEC_M <= 128)");
    RegTensor<T> valReg;
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(valReg, inUb);
    if constexpr (VEC_M > 64U) {
        // VEC_M 为模板常量, hiLanes 初值由编译器常量折叠; 保留可写左值供 UpdateMask POST_UPDATE
        uint32_t hiLanes = VEC_M - 64U;
        MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
        MaskReg pregHi = UpdateMask<T>(hiLanes);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, valReg, pregAll);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb + 64, valReg, pregHi);
    } else {
        uint32_t sreg = m;
        MaskReg pregF = UpdateMask<T>(sreg);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, valReg, pregF);
    }
}

/*!
 * \brief 连续值段直写: maxUb[r] = inUb[r], r < VEC_M(<= 128)。
 * 前置: inUb[0..m) 为本块行值连续段(float, 基址 32B 对齐)。LoadAlign 全宽读,
 * [m, VEC_M) 尾部读槽内残留, 尾部行不被消费; 段数由 VEC_M 编译期决定。
 */
template <typename T, typename INPUT_T, uint32_t VEC_M>
__simd_vf__ inline void SinkContigDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t m)
{
    static_assert(VEC_M > 0U && VEC_M <= 128U, "SinkContigDirectVF supports at most 2 segments (VEC_M <= 128)");
    RegTensor<T> fRegLo;
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegLo, inUb);
    if constexpr (VEC_M > 64U) {
        uint32_t hiLanes = VEC_M - 64U;
        MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
        MaskReg pregHi = UpdateMask<T>(hiLanes);
        RegTensor<T> fRegHi;
        LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegHi, inUb + 64);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, fRegLo, pregAll);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb + 64, fRegHi, pregHi);
    } else {
        uint32_t sreg = m;
        MaskReg pregF = UpdateMask<T>(sreg);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, fRegLo, pregF);
    }
}

/*!
 * \brief GS1(BNSD) 直写: maxUb[r] = inUb[(s0 + r) / actS1Size], r < VEC_M(<= 128)。
 * 前置: inUb[0..gCount) 为本块 g 值连续区(float, GS1 CopyIn 布局)。
 * 索引链 int32: Arange(s0 + segStart) -> Div(actS1Size) -> Gather(inUb)。
 * 未用 lane 索引最坏 (actS1Size-1+127)/actS1Size <= 128, 恒落值区槽内。
 * VEC_M<64 时 lo 段按 VEC_M 掩码(兼容槽容量等于 VEC_M 的调用方); VEC_M>=64 时整 64 lane 写。
 */
template <typename T, typename INPUT_T, uint32_t VEC_M>
__simd_vf__ inline void SinkGs1DirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t s0, uint32_t actS1Size,
                                        uint32_t)
{
    static_assert(VEC_M > 0U && VEC_M <= 128U, "SinkGs1DirectVF supports VEC_M in (0, 128]");
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> aszReg;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t loLanes = (VEC_M < 64U) ? VEC_M : 64U;
    MaskReg pregLo = UpdateMask<T>(loLanes);
    Duplicate<int32_t>(aszReg, static_cast<int32_t>(actS1Size));
    Arange<int32_t>(aReg, static_cast<int32_t>(s0));
    Div<int32_t>(qReg, aReg, aszReg, pregAll);
    RegTensor<uint32_t>& qLoU32 = (RegTensor<uint32_t>&)qReg;
    Gather(oReg, inUb, qLoU32, pregAll);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregLo);
    if constexpr (VEC_M > 64U) {
        uint32_t hiLanes = VEC_M - 64U;
        MaskReg pregHi = UpdateMask<T>(hiLanes);
        Arange<int32_t>(aReg, static_cast<int32_t>(s0 + 64U));
        Div<int32_t>(qReg, aReg, aszReg, pregAll);
        RegTensor<uint32_t>& qHiU32 = (RegTensor<uint32_t>&)qReg;
        Gather(oReg, inUb, qHiU32, pregAll);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregHi);
    }
}

/*!
 * \brief 旋转(S1G case2.2/3) 直写: maxUb[r] = inUb[(gStart + r) mod gSize], r < VEC_M(<= 128)。
 * 前置: inUb[0..gSize) 为本 n2 完整值区(float, case2.2/3 CopyIn 布局, gSize <= 128)。
 * case2.2(gStart <= gEnd)序列不回绕, 索引恒等重排, 同样正确。
 * 取模无 Mod 指令: k = a - (a / gSize) * gSize, k 恒 ∈ [0, gSize)(含未用 lane), 恒落值区槽内。
 * VEC_M<64 时 lo 段按 VEC_M 掩码; VEC_M>64 时 hi 段 lane 数 = VEC_M - 64。
 */
template <typename T, typename INPUT_T, uint32_t VEC_M>
__simd_vf__ inline void SinkRotDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t gStart, uint32_t gSize,
                                        uint32_t)
{
    static_assert(VEC_M > 0U && VEC_M <= 128U, "SinkRotDirectVF supports VEC_M in (0, 128]");
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> kReg;
    RegTensor<int32_t> gszReg;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t loLanes = (VEC_M < 64U) ? VEC_M : 64U;
    MaskReg pregLo = UpdateMask<T>(loLanes);
    Duplicate<int32_t>(gszReg, static_cast<int32_t>(gSize));
    Arange<int32_t>(aReg, static_cast<int32_t>(gStart));
    Div<int32_t>(qReg, aReg, gszReg, pregAll);
    Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
    Sub<int32_t>(kReg, aReg, kReg, pregAll);
    RegTensor<uint32_t>& kLoU32 = (RegTensor<uint32_t>&)kReg;
    Gather(oReg, inUb, kLoU32, pregAll);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregLo);
    if constexpr (VEC_M > 64U) {
        uint32_t hiLanes = VEC_M - 64U;
        MaskReg pregHi = UpdateMask<T>(hiLanes);
        Arange<int32_t>(aReg, static_cast<int32_t>(gStart + 64U));
        Div<int32_t>(qReg, aReg, gszReg, pregAll);
        Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
        Sub<int32_t>(kReg, aReg, kReg, pregAll);
        RegTensor<uint32_t>& kHiU32 = (RegTensor<uint32_t>&)kReg;
        Gather(oReg, inUb, kHiU32, pregAll);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregHi);
    }
}

/*!
 * \brief gap 双段(S1G case2.1) 直写。
 * 前置: 尾段值在 inUb[0..gEnd], 头段值在 inUb[headOff..headOff+headLen)
 * (headOff 8 元素对齐 = 32B, 由 CopyIn gap 布局约定; m = headLen + gEnd + 1,
 * 最坏 span = headOff + headLen <= 136)。
 * 行序列: r < headLen 取 inUb[headOff + r], 其余行取 inUb[r - headLen]。
 * 头段 Gather 基址 = inUb + headOff(32B 对齐), 索引 = r 本身;
 * 尾段索引 = r - headLen(Adds ZEROING: 头段 lane 置 0, 恒落槽内);
 * head/tail 分界为语义运行时量, Compares 产 mask 保留。
 * VEC_M<64 时 lo 段按 VEC_M 掩码; VEC_M>64 时 hi 段 lane 数 = VEC_M - 64。
 */
template <typename T, typename INPUT_T, uint32_t VEC_M>
__simd_vf__ inline void SinkGapDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t headOff, uint32_t headLen,
                                        uint32_t)
{
    static_assert(VEC_M > 0U && VEC_M <= 128U, "SinkGapDirectVF supports VEC_M in (0, 128]");
    RegTensor<T> gHead;
    RegTensor<T> gTail;
    RegTensor<T> oReg;
    RegTensor<int32_t> rReg;
    RegTensor<int32_t> idxTail;
    RegTensor<uint32_t>& rU32 = (RegTensor<uint32_t>&)rReg;
    RegTensor<uint32_t>& idxTailU32 = (RegTensor<uint32_t>&)idxTail;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    uint32_t loLanes = (VEC_M < 64U) ? VEC_M : 64U;
    MaskReg pregLo = UpdateMask<T>(loLanes);
    Arange<int32_t>(rReg, 0);
    MaskReg pregHeadLo;
    Compares<int32_t, CMPMODE::LT>(pregHeadLo, rReg, static_cast<int32_t>(headLen), pregAll);
    MaskReg pregTailLo;
    Compares<int32_t, CMPMODE::GE>(pregTailLo, rReg, static_cast<int32_t>(headLen), pregAll);
    Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregTailLo);
    Gather(gHead, inUb + headOff, rU32, pregHeadLo);
    Gather(gTail, inUb, idxTailU32, pregTailLo);
    Select<T>(oReg, gHead, gTail, pregHeadLo);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregLo);
    if constexpr (VEC_M > 64U) {
        uint32_t hiLanes = VEC_M - 64U;
        MaskReg pregHi = UpdateMask<T>(hiLanes);
        Arange<int32_t>(rReg, 64);
        MaskReg pregHeadHi;
        Compares<int32_t, CMPMODE::LT>(pregHeadHi, rReg, static_cast<int32_t>(headLen), pregAll);
        MaskReg pregTailHi;
        Compares<int32_t, CMPMODE::GE>(pregTailHi, rReg, static_cast<int32_t>(headLen), pregAll);
        Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregTailHi);
        Gather(gHead, inUb + headOff, rU32, pregHeadHi);
        Gather(gTail, inUb, idxTailU32, pregTailHi);
        Select<T>(oReg, gHead, gTail, pregHeadHi);
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregHi);
    }
}

/*!
 * \brief sink 注入 dispatcher(V 锁内调用), 全场景直写后即返回。
 * \param maxUb       [out] softmaxMax 槽位(本块 m 行, 唯一 REG->UB 落点)
 * \param sinkTmpUb   [in]  float 值区(CopyIn 布局: GS1 紧凑 gCount 值 / S1G 4 情况)
 * \param gs1Start    [in]  本 block 的 GS1 轴起始(相位换算基准)
 * \param actVecMSize [in]  有效行数 m, 调用方须保证 m <= VEC_M(运行期不做校验, 避免性能开销)
 * \param actS1Size   [in]  GS1 布局的 s1 维块大小
 * \param gSize       [in]  GQA 组大小
 * \tparam VEC_M      [in]  每 vector 行数(softmax 槽 lane 数, 编译期, 0 < VEC_M <= 128):
 *                          arch35 为 mBaseSize/CV_RATIO, arch92 为 mBaseSize
 */
template <typename T, typename INPUT_T, bool IS_GS1_LAYOUT, uint32_t VEC_M>
__aicore__ inline void SinkExpandMaxVf(const LocalTensor<T>& maxUb, const LocalTensor<INPUT_T>& sinkTmpUb,
                                       uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
{
    if (actVecMSize == 0) {
        return;
    }
    __ubuf__ T* maxUbPtr = (__ubuf__ T*)maxUb.GetPhyAddr();
    __ubuf__ INPUT_T* inUb = (__ubuf__ INPUT_T*)sinkTmpUb.GetPhyAddr();

    if (gSize == 1U) {
        SinkBrcVF<T, INPUT_T, VEC_M>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    if constexpr (IS_GS1_LAYOUT) {
        uint32_t s0 = gs1Start % actS1Size;
        SinkGs1DirectVF<T, INPUT_T, VEC_M>(maxUbPtr, inUb, s0, actS1Size, actVecMSize);
        return;
    }
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;
    if (s1Start == s1End) {
        // case1: 单 s1 行, 值区连续
        SinkContigDirectVF<T, INPUT_T, VEC_M>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    if (s1Start + 1 == s1End && gStart > gEnd) {
        // case2.1: 头段 [gStart, gSize) 与尾段 [0, gEnd] 不重叠, CopyIn 按 gap 布局分段存放
        uint32_t headOff = (gEnd + 1U + 7U) & ~7U;
        uint32_t headLen = gSize - gStart;
        SinkGapDirectVF<T, INPUT_T, VEC_M>(maxUbPtr, inUb, headOff, headLen, actVecMSize);
        return;
    }
    // case2.2/3: 值区回绕(m 跨多个 g 周期)
    SinkRotDirectVF<T, INPUT_T, VEC_M>(maxUbPtr, inUb, gStart, gSize, actVecMSize);
}

/*!
 * \brief pScale 的 ln 值预计算(BRC 单值读)。
 *        由调用方在 task 内只执行一次, 结果缓存在 32B 对齐槽位;
 *        后续 prologue 用 DIST_BRC_B32 直接广播, 避免每个 M 块重复做一次全 lane Ln。
 */
template <typename T>
__simd_vf__ inline void SinkLnScaleVF(__ubuf__ T* lnScaleUb, float pScale)
{
    RegTensor<T> vreg_ps;
    RegTensor<T> vreg_lnPs;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    // UpdateMask 需要可写左值(uint32_t&); 单 lane mask 只写首元素。
    uint32_t oneLane = 1U;
    MaskReg pregOne = UpdateMask<T>(oneLane);
    Duplicate(vreg_ps, static_cast<T>(pScale));
    Ln(vreg_lnPs, vreg_ps, pregAll);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(lnScaleUb, vreg_lnPs, pregOne);
}

/*!
 * \brief DN 前置(prologue)注入: maxUb[r] = sink - ln(pScale), sumUb[r] = pScale, r < m。
 * inUb 为单个 float32 sink(BRC 广播到全部 lane); lnScaleUb 为预计算的 ln(pScale)。
 * 与 FP8/HIF8 DN softmax VF 使用完全相同的 max 数值域: 不做 ln2 网格化, 也不截断 sink。
 * m 必须与紧随其后的 softmax VF 实际读取的 lane 数一致, 避免尾部读到未初始化数据。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkPrologueVF(__ubuf__ T* sumUb, __ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, float pScale,
                                       __ubuf__ INPUT_T* lnScaleUb, uint32_t m, bool sinkOwner = true)
{
    RegTensor<T> vreg_sink;
    RegTensor<T> vreg_ps;
    RegTensor<T> vreg_lnPs;
    RegTensor<T> vreg_max;
    // max must match softmax VF write layout: DIST_NORM_B16 + uint16-ALL mask
    MaskReg pregMax = CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    MaskReg pregSum = CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_sink, inUb);
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_lnPs, lnScaleUb);
    Duplicate(vreg_ps, static_cast<T>(pScale));
    Sub(vreg_max, vreg_sink, vreg_lnPs, pregMax);

    StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(maxUb, vreg_max, pregMax);
    if (m > 64U) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(maxUb + 64, vreg_max, pregMax);
    }

    // sum0 = pScale(owner 分片) / 0(非 owner 分片), 等价于 exp(sink - max0);
    // FD 合并按 max 归一后, sink 分母项只由 owner 分片计入一次。
    if (!sinkOwner) {
        Duplicate(vreg_ps, static_cast<T>(0.0));
    }
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(sumUb, vreg_ps, pregSum);
    if (m > 64U) {
        StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(sumUb + 64, vreg_ps, pregSum);
    }
}

/*!
 * \brief MXFP8 前置(prologue)注入(网格化版): maxUb[r] = ceil(s/ln2)*ln2 - ln_ps,
 *        sumUb[r] = exp(s - maxUb[r]), r < m。
 * MXFP8 的 P 量化要求 max 状态恒在 k*ln2 网格上(跨块重缩放系数 exp(m1-m2)=2^k 精确,
 * e8m0 per-group scale 无舍入, BMM2 分子与 VF 分母记账严格一致)。网格对 Max 封闭,
 * 因此初值网格化后全链路不变量保持; sum 为 fp32 记账可携带任意值(sink 的精确相对量)。
 * m0 - s ∈ [0, ln2) => sum0 ∈ (0.5, 1], 任意有限 sink 数值安全。
 * 仅当 |s| 大到 s*INV_LN2 溢出 fp32(>2e38) 时网格修正不可表示,
 * 退化为 m0 = s - ln(pScale)(sum0 = pScale), 数学上与网格化等价。
 * sinkOwner=false 的分片共享同一 m0 但 sum0=0, 供 FD 合并时避免分母重复累加。
 * 该实现依赖 MXFP8 量化域约束, 目前仅 QFA MXFP8 S1G prologue 使用。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkPrologueGridVF(__ubuf__ T* sumUb, __ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb,
                                           __ubuf__ INPUT_T* lnScaleUb, uint32_t m, bool sinkOwner = true)
{
    constexpr T LN2_C = static_cast<T>(0.693147180559945);      // ln(2)
    constexpr T INV_LN2_C = static_cast<T>(1.4426950408889634); // 1/ln(2)
    RegTensor<T> vreg_sink;
    RegTensor<T> vreg_ln_ps;
    RegTensor<T> vreg_t;
    RegTensor<T> vreg_fb;
    RegTensor<T> vreg_big;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    MaskReg pregF = UpdateMask<T>(m);
    MaskReg pregCmp;
    // ln(pScale) 由调用方在 task 内预计算一次, 此处只做 BRC 广播。
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(vreg_ln_ps, lnScaleUb);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(vreg_sink, inUb);
    // 网格化: m0 = ceil(s/ln2)*ln2 - ln(pScale)
    Muls(vreg_t, vreg_sink, INV_LN2_C, pregAll);
    Truncate<T, RoundMode::CAST_CEIL>(vreg_t, vreg_t, pregAll);
    Muls(vreg_t, vreg_t, LN2_C, pregAll);
    Sub(vreg_t, vreg_t, vreg_ln_ps, pregAll);
    // |s| > 2e38 时网格修正不可表示, 退化为 m0 = s - ln(pScale)
    Sub(vreg_fb, vreg_sink, vreg_ln_ps, pregAll);
    Duplicate(vreg_big, static_cast<T>(2.0e38), pregAll);
    Compare<T, CMPMODE::GT>(pregCmp, vreg_sink, vreg_big, pregF);
    Select(vreg_t, vreg_fb, vreg_t, pregCmp);
    Duplicate(vreg_big, static_cast<T>(-2.0e38), pregAll);
    Compare<T, CMPMODE::LT>(pregCmp, vreg_sink, vreg_big, pregF);
    Select(vreg_t, vreg_fb, vreg_t, pregCmp);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, vreg_t, pregF);
    // sum0 = exp(s - m0) ∈ (0.5, 1], 仅由 owner 分片计入
    if (sinkOwner) {
        FusedExpSub(vreg_sink, vreg_sink, vreg_t, pregF);
    } else {
        Duplicate(vreg_sink, static_cast<T>(0.0), pregF);
    }
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(sumUb, vreg_sink, pregF);
}

/*!
 * \brief int64 阈值钳位到 int32(向量 Compares 的标量操作数域)。
 *        阈值由 gs1Start 与有效 token 边界换算得到, 极端配置下可能超出 int32 范围;
 *        钳位保持判定语义: 小于 int32 下界的阈值等价于"全有效下界",
 *        大于 int32 上界的阈值等价于"全有效上界"。
 */
__aicore__ inline int32_t SinkClampToI32(int64_t v)
{
    return v < -2147483647LL ? -2147483647 : (v > 2147483647LL ? 2147483647 : static_cast<int32_t>(v));
}

/*!
 * \brief sink 无效行标记(S1G): 把 s1(r) ∉ [first, last] 的行 maxUb[r] 置为 minValue。
 *        S1G 下 s1(r) = (gs1Start + r) / realGSize 关于 r 单调, 无效行等价于行区间
 *        [loThr, hiThr), 阈值由标量一次算好(loThr = first*G - gs1Start,
 *        hiThr = (last+1)*G - gs1Start), 向量侧不再做 Div。
 *        r < m, m <= VEC_M(<= 64)。MXFP8 的 S1G 路径与 FP8/HIF8 的 DN 路径共用。
 */
template <typename T, uint32_t VEC_M>
__simd_vf__ inline void MarkInvalidRowMaxS1GVF(__ubuf__ T* maxUb, uint32_t m, int32_t loThr, int32_t hiThr, T minValue)
{
    static_assert(VEC_M > 0U && VEC_M <= 64U, "MarkInvalidRowMaxS1GVF supports at most 64 lanes");
    RegTensor<T> maxReg;
    RegTensor<T> minReg;
    RegTensor<int32_t> rowReg;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    MaskReg pregLt;
    MaskReg pregGe;
    Duplicate<T>(minReg, minValue, pregAll);
    Arange<int32_t>(rowReg, 0);
    Compares<int32_t, CMPMODE::LT>(pregLt, rowReg, loThr, pregAll);
    Compares<int32_t, CMPMODE::GE>(pregGe, rowReg, hiThr, pregAll);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(maxReg, maxUb);
    Select<T>(maxReg, minReg, maxReg, pregLt);
    Select<T>(maxReg, minReg, maxReg, pregGe);
    uint32_t sreg = m;
    MaskReg pregM = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, maxReg, pregM);
}

/*!
 * \brief sink 无效行标记(GS1): 把 s1(r) ∉ [first, last] 的行 maxUb[r] 置为 minValue。
 *        GS1 下 s1(r) = (gs1Start + r) % actS1Size, 行序为 g 外层、s1 内层,
 *        一个 vector 块可能跨越多个 G 组(即多次回绕), 不能用单次 wrap 的
 *        Adds + Select 近似。这里按 SinkRotDirectVF 的方式用 Div 取模:
 *        s1 = (base + r) - floor((base + r) / actS1Size) * actS1Size,
 *        s1 恒 ∈ [0, actS1Size), 对任意 m(<= VEC_M) 与 actS1Size 均成立。
 *        base 由标量一次算好; 取模只在存在无效行的标记路径执行, 不影响热路径。
 *        r < m, m <= VEC_M(<= 64)。
 */
template <typename T, uint32_t VEC_M>
__simd_vf__ inline void MarkInvalidRowMaxGs1VF(__ubuf__ T* maxUb, uint32_t m, int32_t base, int32_t actS1Size,
                                               int32_t firstValid, int32_t lastValid, T minValue)
{
    static_assert(VEC_M > 0U && VEC_M <= 64U, "MarkInvalidRowMaxGs1VF supports at most 64 lanes");
    RegTensor<T> maxReg;
    RegTensor<T> minReg;
    RegTensor<int32_t> rowReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> s1Reg;
    RegTensor<int32_t> aszReg;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    MaskReg pregLt;
    MaskReg pregGt;
    Duplicate<T>(minReg, minValue, pregAll);
    Duplicate<int32_t>(aszReg, actS1Size);
    Arange<int32_t>(rowReg, base);
    Div<int32_t>(qReg, rowReg, aszReg, pregAll);
    Muls<int32_t>(s1Reg, qReg, actS1Size, pregAll);
    Sub<int32_t>(s1Reg, rowReg, s1Reg, pregAll);
    Compares<int32_t, CMPMODE::LT>(pregLt, s1Reg, firstValid, pregAll);
    Compares<int32_t, CMPMODE::GT>(pregGt, s1Reg, lastValid, pregAll);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(maxReg, maxUb);
    Select<T>(maxReg, minReg, maxReg, pregLt);
    Select<T>(maxReg, minReg, maxReg, pregGt);
    uint32_t sreg = m;
    MaskReg pregM = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, maxReg, pregM);
}
} // namespace AttentionCommon

#endif // ATTN_SINK_GS1_H_
