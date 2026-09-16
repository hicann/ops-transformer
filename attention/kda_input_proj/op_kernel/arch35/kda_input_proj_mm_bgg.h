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
 * \file kda_input_proj_mm_bgg.h
 * \brief Stage1 AIC: beta/gate/g 三路 N 逻辑拼接成 tDim × nCntAll 任务池；核用
 *        basicIdx += coreNums 领取，按 path 聚簇后复用同一份 Blaze BlockMmad。
 */

#ifndef KDA_INPUT_PROJ_MM_BGG_H
#define KDA_INPUT_PROJ_MM_BGG_H

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

#include <type_traits>
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjMmBgg {
public:
    __aicore__ inline KdaInputProjMmBgg() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weightBeta, __gm__ uint8_t *weightGate,
                                __gm__ uint8_t *weightG, __gm__ uint8_t *beta, __gm__ uint8_t *gate, __gm__ uint8_t *g,
                                __gm__ uint8_t *workspace, const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    using DtypeX = typename TypePack::DtypeX;
    using DtypeBeta = typename TypePack::DtypeBeta;
    using DtypeGate = typename TypePack::DtypeGate;
    using DtypeG = typename TypePack::DtypeG;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using LayoutBBeta =
        std::conditional_t<TypePack::TRANS_WEIGHT_BETA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutBGate =
        std::conditional_t<TypePack::TRANS_WEIGHT_GATE, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutBG =
        std::conditional_t<TypePack::TRANS_WEIGHT_G, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
    using BiasType = float;

    __aicore__ inline int64_t MinI64(int64_t a, int64_t b) const
    {
        return (a < b) ? a : b;
    }

    template <class LayoutB, class DtypeC>
    __aicore__ inline void ProcessPath(uint32_t pathBegin, uint32_t pathCnt, uint32_t pathN, uint32_t nTile,
                                       __gm__ DtypeX *wAddr, __gm__ DtypeC *cAddr);

    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *workspace_{nullptr};
    __gm__ DtypeX *xAddr_{nullptr};
    __gm__ DtypeX *wBetaAddr_{nullptr};
    __gm__ DtypeX *wGateAddr_{nullptr};
    __gm__ DtypeX *wGAddr_{nullptr};
    __gm__ DtypeBeta *betaAddr_{nullptr};
    __gm__ DtypeGate *gateAddr_{nullptr};
    __gm__ DtypeG *gAddr_{nullptr};

    // 任务池视图：basicIdx = mIdx * nCntAll_ + nIdxAll，nIdxAll 按 beta|gate|g 顺序拼接。
    uint32_t nCntAll_{0};
    uint32_t tileNumAll_{0};
    uint32_t blockIdx_{0};
    uint32_t coreNums_{0};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjMmBgg<TypePack>::Init(__gm__ uint8_t *x, __gm__ uint8_t *weightBeta,
                                                         __gm__ uint8_t *weightGate, __gm__ uint8_t *weightG,
                                                         __gm__ uint8_t *beta, __gm__ uint8_t *gate, __gm__ uint8_t *g,
                                                         __gm__ uint8_t *workspace,
                                                         const optiling::KdaInputProjTilingData *__restrict tiling)
{
    workspace_ = workspace;
    tiling_ = tiling;
    xAddr_ = reinterpret_cast<__gm__ DtypeX *>(x);
    wBetaAddr_ = reinterpret_cast<__gm__ DtypeX *>(weightBeta);
    wGateAddr_ = reinterpret_cast<__gm__ DtypeX *>(weightGate);
    wGAddr_ = reinterpret_cast<__gm__ DtypeX *>(weightG);
    betaAddr_ = reinterpret_cast<__gm__ DtypeBeta *>(beta);
    gateAddr_ = reinterpret_cast<__gm__ DtypeGate *>(gate);
    gAddr_ = reinterpret_cast<__gm__ DtypeG *>(g);
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMmBgg<TypePack>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    if (tiling_ == nullptr) {
        return;
    }
    const optiling::KdaInputProjMmBggParams &mm = tiling_->mmBggParams;
    const optiling::KdaInputProjBaseParams &base = tiling_->baseParams;
    if (mm.tTile == 0U || mm.betaTile == 0U || mm.gateTile == 0U || mm.gTile == 0U || mm.hiddenstatesTile == 0U ||
        mm.hiddenstatesL0Tile == 0U || mm.tDim == 0U || base.tSize == 0U || base.hiddenSize == 0U) {
        return;
    }

    nCntAll_ = mm.betaDim + mm.gateDim + mm.gDim;
    tileNumAll_ = mm.tDim * nCntAll_;
    // 与 kernel_matmul_basic.h MatmulProcess 一致：直接用 GetBlockIdx/GetBlockNum，
    // 不除 GetTaskRation（官方仅在尾块再切路径用除后的 oriBlockIdx）。
    blockIdx_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
    coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
    if (nCntAll_ == 0U || tileNumAll_ == 0U || coreNums_ == 0U) {
        return;
    }

    // 三路复用同一批 L1 slot，且各自新建 BlockMmad（不共享 bufMgr 事件），
    // 路间必须加 PIPE_ALL 栅栏，否则上一路的 MTE2 还在写同一块 L1。
    ProcessPath<LayoutBBeta, DtypeBeta>(0U, mm.betaDim, base.betaSize, mm.betaTile, wBetaAddr_, betaAddr_);
    AscendC::PipeBarrier<PIPE_ALL>();
    ProcessPath<LayoutBGate, DtypeGate>(mm.betaDim, mm.gateDim, base.gateSize, mm.gateTile, wGateAddr_, gateAddr_);
    AscendC::PipeBarrier<PIPE_ALL>();
    ProcessPath<LayoutBG, DtypeG>(mm.betaDim + mm.gateDim, mm.gDim, base.gSize, mm.gTile, wGAddr_, gAddr_);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename TypePack>
template <class LayoutB, class DtypeC>
__aicore__ inline void KdaInputProjMmBgg<TypePack>::ProcessPath(uint32_t pathBegin, uint32_t pathCnt, uint32_t pathN,
                                                                uint32_t nTile, __gm__ DtypeX *wAddr,
                                                                __gm__ DtypeC *cAddr)
{
    if (pathCnt == 0U || pathN == 0U || nTile == 0U || wAddr == nullptr || cAddr == nullptr) {
        return;
    }
    const uint32_t pathEnd = pathBegin + pathCnt;

    // 本核在该路一个 tile 都没有时不构造 BlockMmad，省掉 L1 分片与 layout 变换。
    bool hasTile = false;
    for (uint32_t basicIdx = blockIdx_; basicIdx < tileNumAll_; basicIdx += coreNums_) {
        const uint32_t nIdxAll = basicIdx % nCntAll_;
        if (nIdxAll >= pathBegin && nIdxAll < pathEnd) {
            hasTile = true;
            break;
        }
    }
    if (!hasTile) {
        return;
    }

    const optiling::KdaInputProjMmBggParams &mm = tiling_->mmBggParams;
    const optiling::KdaInputProjBaseParams &base = tiling_->baseParams;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, DtypeX, LayoutA, DtypeX, LayoutB, DtypeC, LayoutC,
                                                    BiasType, LayoutBias>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<DtypeX>>>;
    using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<DtypeC>>>;
    using MakeLayoutBias =
        AscendC::Te::FrameLayoutFormat<LayoutBias, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<BiasType>>>;

    const int64_t m64 = static_cast<int64_t>(base.tSize);
    const int64_t k64 = static_cast<int64_t>(base.hiddenSize);
    const int64_t n64 = static_cast<int64_t>(pathN);

    // A 非转置：等价于 kernel_matmul_basic.h 的 MakeLayoutAGm，rowStride = k。
    auto layoutA = AscendC::Te::MakePatternLayout<LayoutA, AscendC::Te::LayoutTraitDefault<>>(
        AscendC::Te::MakeShape(int64_t{1}, AscendC::Te::MakeShape(AscendC::Te::MakeShape(AscendC::Te::_1{}, m64),
                                                                  AscendC::Te::MakeShape(AscendC::Te::_1{}, k64))),
        AscendC::Te::MakeStride(
            m64 * k64, AscendC::Te::MakeStride(AscendC::Te::MakeStride(AscendC::Te::_0{}, k64),
                                               AscendC::Te::MakeStride(AscendC::Te::_0{}, AscendC::Te::_1{}))));
    auto layoutB = MakeLayoutB{}(int64_t{1}, k64, n64);
    auto layoutC = MakeLayoutC{}(int64_t{1}, m64, n64);
    auto layoutBias = MakeLayoutBias{}(int64_t{1}, n64);

    auto gmA = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(xAddr_), layoutA);
    auto gmB = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(wAddr), layoutB);
    auto gmC = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cAddr), layoutC);
    // biasGmAddr 为空 => BlockMmad::isBias_ = false，bias 张量只作描述符、不会被读，
    // 且 initCmatrix 置真直接清零 L0C，比搬一遍零 bias 更省。
    __gm__ BiasType *biasAddr = nullptr;
    auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasAddr), layoutBias);

    BlockMmad blockMmad;
    typename BlockMmad::Params mmadParams;
    mmadParams.aGmAddr = reinterpret_cast<GM_ADDR>(xAddr_);
    mmadParams.bGmAddr = reinterpret_cast<GM_ADDR>(wAddr);
    mmadParams.cGmAddr = reinterpret_cast<GM_ADDR>(cAddr);
    mmadParams.biasGmAddr = nullptr;
    mmadParams.mL1 = mm.tTile;
    mmadParams.nL1 = nTile;
    mmadParams.kL1 = mm.hiddenstatesTile;
    mmadParams.mL0 = mm.tTile;
    mmadParams.nL0 = nTile;
    mmadParams.kL0 = mm.hiddenstatesL0Tile;
    mmadParams.l1Stages = 2U;
    mmadParams.l0cStages = 1U;
    blockMmad.Init(mmadParams);

    for (uint32_t basicIdx = blockIdx_; basicIdx < tileNumAll_; basicIdx += coreNums_) {
        const uint32_t nIdxAll = basicIdx % nCntAll_;
        if (nIdxAll < pathBegin || nIdxAll >= pathEnd) {
            continue;
        }
        const int64_t coordM = static_cast<int64_t>(basicIdx / nCntAll_) * static_cast<int64_t>(mm.tTile);
        const int64_t coordN = static_cast<int64_t>(nIdxAll - pathBegin) * static_cast<int64_t>(nTile);
        if (coordM >= m64 || coordN >= n64) {
            continue;
        }
        const int64_t shapeM = MinI64(static_cast<int64_t>(mm.tTile), m64 - coordM);
        const int64_t shapeN = MinI64(static_cast<int64_t>(nTile), n64 - coordN);

        auto subA = gmA.Slice(AscendC::Te::MakeCoord(int64_t{0}, AscendC::Te::MakeCoord(coordM, int64_t{0})),
                              AscendC::Te::MakeShape(int64_t{1}, AscendC::Te::MakeShape(shapeM, k64)));
        auto gmBlockA = AscendC::Te::Squeeze<0>(subA);
        auto subB = gmB.Slice(AscendC::Te::MakeCoord(int64_t{0}, AscendC::Te::MakeCoord(int64_t{0}, coordN)),
                              AscendC::Te::MakeShape(int64_t{1}, AscendC::Te::MakeShape(k64, shapeN)));
        auto gmBlockB = AscendC::Te::Squeeze<0>(subB);
        auto subC = gmC.Slice(AscendC::Te::MakeCoord(int64_t{0}, AscendC::Te::MakeCoord(coordM, coordN)),
                              AscendC::Te::MakeShape(int64_t{1}, AscendC::Te::MakeShape(shapeM, shapeN)));
        auto gmBlockC = AscendC::Te::Squeeze<0>(subC);
        auto gmBlockBias =
            gmBias.Slice(AscendC::Te::MakeCoord(int64_t{0}, coordN), AscendC::Te::MakeShape(int64_t{1}, shapeN));

        typename BlockMmad::TupleShape blockShape{shapeM, shapeN, k64, int64_t{1}};
        blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
    }
}

} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_MM_BGG_H
