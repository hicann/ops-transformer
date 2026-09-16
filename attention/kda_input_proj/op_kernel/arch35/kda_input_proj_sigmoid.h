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
 * \file kda_input_proj_sigmoid.h
 * \brief Stage2 AIV: inplace Sigmoid on beta (FP32). y = 1 / (1 + exp(-x))
 */

#ifndef KDA_INPUT_PROJ_SIGMOID_H
#define KDA_INPUT_PROJ_SIGMOID_H

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {

template <typename TypePack>
class KdaInputProjSigmoid {
public:
    __aicore__ inline KdaInputProjSigmoid() {}

    __aicore__ inline void Init(__gm__ uint8_t *beta, const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

protected:
    const optiling::KdaInputProjTilingData *tiling_{nullptr};
    __gm__ uint8_t *beta_{nullptr};

#ifndef __DAV_CUBE__
    using QueIn = AscendC::TQue<AscendC::QuePosition::VECIN, 2>;
    using QueOut = AscendC::TQue<AscendC::QuePosition::VECOUT, 1>;

    __aicore__ inline void ProcessAiv();
    __aicore__ inline uint32_t ExtentOf(uint32_t loopIdx, uint32_t loopNum, uint32_t ubTile, uint32_t tailExt) const;
    __aicore__ inline void CopyIn(QueIn &queIn, uint64_t gmOffset, uint32_t extent);
    __aicore__ inline void Compute(QueIn &queIn, QueOut &queOut, uint32_t extent);
    __aicore__ inline void CopyOut(QueOut &queOut, uint64_t gmOffset, uint32_t extent);

    AscendC::GlobalTensor<float> betaGm_;
#endif
};

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::Init(__gm__ uint8_t *beta,
                                                           const optiling::KdaInputProjTilingData *__restrict tiling)
{
    beta_ = beta;
    tiling_ = tiling;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::Process()
{
#ifndef __DAV_CUBE__
    ProcessAiv();
#endif
}

#ifndef __DAV_CUBE__
template <typename TypePack>
__aicore__ inline uint32_t KdaInputProjSigmoid<TypePack>::ExtentOf(uint32_t loopIdx, uint32_t loopNum, uint32_t ubTile,
                                                                   uint32_t tailExt) const
{
    return (loopIdx + 1U == loopNum) ? tailExt : ubTile;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::ProcessAiv()
{
    if (tiling_ == nullptr || beta_ == nullptr) {
        return;
    }
    const optiling::KdaInputProjSigmoidParams &sp = tiling_->sigmoidParams;
    if (sp.aivNum == 0 || sp.elemNum == 0 || sp.ubTile == 0) {
        return;
    }
    const uint32_t aivIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (aivIdx >= sp.aivNum) {
        return;
    }

    const bool isTail = (aivIdx + 1U == sp.aivNum);
    const uint32_t loopNum = isTail ? sp.tailUbLoop : sp.ubLoop;
    const uint32_t tailExt = isTail ? sp.tailUbTail : sp.ubTail;
    if (loopNum == 0) {
        return;
    }
    const uint64_t blockOff = static_cast<uint64_t>(aivIdx) * static_cast<uint64_t>(sp.blockTile);

    // loopNum==1 时预取分支永不触发，第二份 buffer 拿不到复用，只申请单份。
    const uint8_t bufNum = (loopNum > 1U) ? 2U : 1U;
    AscendC::TPipe pipe;
    QueIn queIn;
    QueOut queOut;
    pipe.InitBuffer(queIn, bufNum, sp.ubTile * sizeof(float));
    pipe.InitBuffer(queOut, bufNum, sp.ubTile * sizeof(float));
    betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta_));

    CopyIn(queIn, blockOff, ExtentOf(0, loopNum, sp.ubTile, tailExt));
    for (uint32_t i = 0; i < loopNum; ++i) {
        if (i + 1U < loopNum) {
            const uint64_t nextOff = blockOff + static_cast<uint64_t>(i + 1U) * sp.ubTile;
            CopyIn(queIn, nextOff, ExtentOf(i + 1U, loopNum, sp.ubTile, tailExt));
        }
        Compute(queIn, queOut, ExtentOf(i, loopNum, sp.ubTile, tailExt));
        CopyOut(queOut, blockOff + static_cast<uint64_t>(i) * sp.ubTile, ExtentOf(i, loopNum, sp.ubTile, tailExt));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::CopyIn(QueIn &queIn, uint64_t gmOffset, uint32_t extent)
{
    AscendC::LocalTensor<float> in = queIn.template AllocTensor<float>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = extent * static_cast<uint32_t>(sizeof(float));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPadExtParams<float> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0.0f;
    AscendC::DataCopyPad(in, betaGm_[gmOffset], copyParams, padParams);
    queIn.EnQue<float>(in);
}

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::Compute(QueIn &queIn, QueOut &queOut, uint32_t extent)
{
    AscendC::LocalTensor<float> in = queIn.template DeQue<float>();
    AscendC::LocalTensor<float> out = queOut.template AllocTensor<float>();
    __VEC_SCOPE__
    {
        // y = 1 / (1 + exp(-x))
        AscendC::Reg::RegTensor<float> vregOne;
        AscendC::Reg::RegTensor<float> vregX;
        AscendC::Reg::RegTensor<float> vregNegX;
        AscendC::Reg::RegTensor<float> vregExpNegX;
        AscendC::Reg::RegTensor<float> vregDenom;
        AscendC::Reg::RegTensor<float> vregY;
        AscendC::Reg::MaskReg pregMask;
        uint32_t remain = extent;
        pregMask = AscendC::Reg::CreateMask<float>();
        AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1),
                                                                                    pregMask);
        const uint16_t vfLoopNum = static_cast<uint16_t>((extent + (AscendC::VECTOR_REG_WIDTH / sizeof(float)) - 1) /
                                                         (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
        __local_mem__ float *inAddr = (__local_mem__ float *)in.GetPhyAddr();
        __local_mem__ float *outAddr = (__local_mem__ float *)out.GetPhyAddr();
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            pregMask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                vregX, inAddr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)));
            AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregNegX, vregX,
                                                                                   static_cast<float>(-1), pregMask);
            AscendC::Reg::Exp<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregExpNegX, vregNegX, pregMask);
            AscendC::Reg::Adds<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregDenom, vregExpNegX,
                                                                                   static_cast<float>(1), pregMask);
            AscendC::Reg::Div<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregY, vregOne, vregDenom, pregMask);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
                outAddr + i * (AscendC::VECTOR_REG_WIDTH / sizeof(float)), vregY, pregMask);
        }
    }
    queIn.FreeTensor(in);
    queOut.EnQue<float>(out);
}

template <typename TypePack>
__aicore__ inline void KdaInputProjSigmoid<TypePack>::CopyOut(QueOut &queOut, uint64_t gmOffset, uint32_t extent)
{
    AscendC::LocalTensor<float> out = queOut.template DeQue<float>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = extent * static_cast<uint32_t>(sizeof(float));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(betaGm_[gmOffset], out, copyParams);
    queOut.FreeTensor(out);
}
#endif

} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_SIGMOID_H
