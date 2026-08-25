/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_v2_expert_token_out.h
 * \brief
 */
#ifndef MOE_V2_EXPERT_TOKEN_OUT_REGBASE_H
#define MOE_V2_EXPERT_TOKEN_OUT_REGBASE_H

#include "moe_init_routing_v2_arch35_common.h"

namespace MoeInitRoutingV2 {
using namespace AscendC;

constexpr static uint32_t VL_INT32 = static_cast<uint32_t>(GetVRegSize()) / sizeof(int32_t);
constexpr static int64_t EXPERT_NUM = 256;
constexpr static uint32_t FOUR = 4;

constexpr static MicroAPI::CastTrait castTraitS322U8 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static MicroAPI::CastTrait castTraitU162U32Even = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static MicroAPI::CastTrait castTraitU162U32Odd = {
    MicroAPI::RegLayout::ONE,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

class MoeV2ExpertTokenOutRegBase {
public:
    __aicore__ inline MoeV2ExpertTokenOutRegBase(){};
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                GM_ADDR expandedRowIdx, GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress);
    __aicore__ inline void Compute(int64_t progress, LocalTensor<int32_t> &tokenIdxOut);
    __aicore__ inline void InitLocal();
    __aicore__ inline void CopyOutTokenGm();
    __aicore__ inline void CopyOutExpertTokens(GlobalTensor<int32_t> &output);
    template <bool cumsum>
    __aicore__ inline void CalcHistVF(__ubuf__ int32_t *expertSeqs, __ubuf__ int32_t *expertTokenIdxOutLocal,
                                      uint32_t count);

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> copyInQueue;
    TQue<QuePosition::VECIN, 1> expertTokenIdxCopyInQueue;
    TQue<QuePosition::VECOUT, 1> expertTokenIdxCopyOutQueue;

    GlobalTensor<int32_t> expertTokensCountOrCumsumGm;
    GlobalTensor<int32_t> expertTokensBeforeCapacityGm;
    GlobalTensor<int32_t> expandedExpertIdxGm;
    GlobalTensor<int32_t> expandedRowIdxGm;

    LocalTensor<int32_t> expertTokenIdxLocal;
    LocalTensor<int32_t> expertTokenIdxOutLocal;

    const MoeV2GatherOutComputeTilingData *srcToDstTilingData;

    int64_t coreNum;
    int64_t blockIdx;
    int64_t totalLength;
    int64_t currentLoopRows;
    int64_t coreRows;
    int64_t perLoopRows;
    int64_t lastLoopRows;
    int64_t expertNum;
    int64_t expertNumUbAlign;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    int64_t expertTokensBeforeCapacityFlag = 0;
};

__aicore__ inline void MoeV2ExpertTokenOutRegBase::InitLocal()
{
    expertTokenIdxLocal = expertTokenIdxCopyOutQueue.AllocTensor<int32_t>();
    Duplicate<int32_t>(expertTokenIdxLocal, 0, this->expertNumUbAlign);
}

__aicore__ inline void MoeV2ExpertTokenOutRegBase::CopyIn(int64_t progress)
{
    LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
    DataCopy(inLocal, expandedExpertIdxGm[progress * perLoopRows], Align(currentLoopRows, sizeof(int32_t)));
    copyInQueue.EnQue<int32_t>(inLocal);
}

__aicore__ inline void DHist(MicroAPI::RegTensor<uint16_t> &dst0, MicroAPI::RegTensor<uint16_t> &dst1,
                             MicroAPI::RegTensor<uint8_t> &src, MicroAPI::MaskReg &mask)
{
    dhistv2(dst0, src, mask, Bin_N0);
    dhistv2(dst1, src, mask, Bin_N1);
}

__aicore__ inline void CHist(MicroAPI::RegTensor<uint16_t> &dst0, MicroAPI::RegTensor<uint16_t> &dst1,
                             MicroAPI::RegTensor<uint8_t> &src, MicroAPI::MaskReg &mask)
{
    chistv2(dst0, src, mask, Bin_N0);
    chistv2(dst1, src, mask, Bin_N1);
}

template <bool cumsum>
__aicore__ inline void MoeV2ExpertTokenOutRegBase::CalcHistVF(__ubuf__ int32_t *expertSeqs,
                                                              __ubuf__ int32_t *tokenIdxOut, uint32_t count)
{
    uint16_t loopCount = CeilDiv(count, VL_INT32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<int32_t> expertIdxS32;
        MicroAPI::RegTensor<uint8_t> expertIdxU8;
        MicroAPI::RegTensor<uint16_t> histLowU16;
        MicroAPI::RegTensor<uint16_t> histHighU16;
        MicroAPI::RegTensor<uint32_t> histLowU32Even;
        MicroAPI::RegTensor<uint32_t> histLowU32Odd;
        MicroAPI::RegTensor<uint32_t> histHighU32Even;
        MicroAPI::RegTensor<uint32_t> histHighU32Odd;
        MicroAPI::RegTensor<uint32_t> hist[FOUR];
        MicroAPI::RegTensor<uint32_t> src[FOUR];
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg histMask = MicroAPI::CreateMask<int8_t, MicroAPI::MaskPattern::M4>();
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(histLowU16, 0, pregAll);
        MicroAPI::Duplicate(histHighU16, 0, pregAll);
        for (uint16_t i = 0; i < (uint16_t)(loopCount - 1); i++) {
            pregLoop = MicroAPI::UpdateMask<int32_t>(count);
            LoadAlign(expertIdxS32, expertSeqs + i * VL_INT32);
            Cast<uint8_t, int32_t, castTraitS322U8>(expertIdxU8, expertIdxS32, pregLoop);
            if constexpr (cumsum) {
                CHist(histLowU16, histHighU16, expertIdxU8, histMask);
            } else {
                DHist(histLowU16, histHighU16, expertIdxU8, histMask);
            }
        }

        uint32_t count4 = count * FOUR;
        MicroAPI::MaskReg tempMask = MicroAPI::UpdateMask<int8_t>(count4);
        for (uint16_t i = (uint16_t)(loopCount - 1); i < loopCount; i++) {
            pregLoop = MicroAPI::UpdateMask<int32_t>(count);
            LoadAlign(expertIdxS32, expertSeqs + i * VL_INT32);
            Cast<uint8_t, int32_t, castTraitS322U8>(expertIdxU8, expertIdxS32, pregLoop);
            MicroAPI::And(histMask, histMask, tempMask, pregAll);
            if constexpr (cumsum) {
                CHist(histLowU16, histHighU16, expertIdxU8, histMask);
            } else {
                DHist(histLowU16, histHighU16, expertIdxU8, histMask);
            }
        }
        MicroAPI::MaskReg pregAllU16 = MicroAPI::CreateMask<int16_t, MicroAPI::MaskPattern::ALL>();
        Cast<uint32_t, uint16_t, castTraitU162U32Even>(histLowU32Even, histLowU16, pregAllU16);
        Cast<uint32_t, uint16_t, castTraitU162U32Odd>(histLowU32Odd, histLowU16, pregAllU16);
        MicroAPI::Interleave<uint32_t>(hist[0], hist[1], histLowU32Even, histLowU32Odd);
        Cast<uint32_t, uint16_t, castTraitU162U32Even>(histHighU32Even, histHighU16, pregAllU16);
        Cast<uint32_t, uint16_t, castTraitU162U32Odd>(histHighU32Odd, histHighU16, pregAllU16);
        MicroAPI::Interleave<uint32_t>(hist[2], hist[3], histHighU32Even, histHighU32Odd);
        LoadAlign(src[0], (__ubuf__ uint32_t *)tokenIdxOut);
        Add<uint32_t>(hist[0], hist[0], src[0], pregAll);
        StoreAlign((__ubuf__ uint32_t *)tokenIdxOut, hist[0], pregAll);
        LoadAlign(src[1], (__ubuf__ uint32_t *)tokenIdxOut + VL_INT32);
        Add<uint32_t>(hist[1], hist[1], src[1], pregAll);
        StoreAlign((__ubuf__ uint32_t *)tokenIdxOut + VL_INT32, hist[1], pregAll);
        LoadAlign(src[2], (__ubuf__ uint32_t *)tokenIdxOut + 2 * VL_INT32);
        Add<uint32_t>(hist[2], hist[2], src[2], pregAll);
        StoreAlign((__ubuf__ uint32_t *)tokenIdxOut + 2 * VL_INT32, hist[2], pregAll);
        LoadAlign(src[3], (__ubuf__ uint32_t *)tokenIdxOut + 3 * VL_INT32);
        Add<uint32_t>(hist[3], hist[3], src[3], pregAll);
        StoreAlign((__ubuf__ uint32_t *)tokenIdxOut + 3 * VL_INT32, hist[3], pregAll);
    }
}

__aicore__ inline void MoeV2ExpertTokenOutRegBase::Compute(int64_t progress, LocalTensor<int32_t> &tokenIdxOut)
{
    LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
    if (this->dropPadMode == DROPLESS_MODE && expertTokensCountOrCumsumFlag == EXERPT_TOKENS_CUMSUM) {
        CalcHistVF<true>((__ubuf__ int32_t *)inLocal.GetPhyAddr(), (__ubuf__ int32_t *)tokenIdxOut.GetPhyAddr(),
                         (uint32_t)currentLoopRows);
    } else {
        CalcHistVF<false>((__ubuf__ int32_t *)inLocal.GetPhyAddr(), (__ubuf__ int32_t *)tokenIdxOut.GetPhyAddr(),
                          (uint32_t)currentLoopRows);
    }
    copyInQueue.FreeTensor(inLocal);
}

__aicore__ inline void MoeV2ExpertTokenOutRegBase::CopyOutExpertTokens(GlobalTensor<int32_t> &output)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(expertNum * sizeof(int32_t)), 0, 0, 0};
    SetAtomicAdd<int32_t>();
    DataCopyPad(output[0], this->expertTokenIdxOutLocal, copyParams);
    SetAtomicNone();
}

__aicore__ inline void MoeV2ExpertTokenOutRegBase::CopyOutTokenGm()
{
    if (this->dropPadMode == DROPLESS_MODE && expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
        CopyOutExpertTokens(expertTokensCountOrCumsumGm);
        return;
    }
    CopyOutExpertTokens(expertTokensBeforeCapacityGm);
}

template <typename TilingData>
__aicore__ inline void MoeV2ExpertTokenOutRegBase::Init(GM_ADDR expertTokensCountOrCumsum,
                                                        GM_ADDR expertTokensBeforeCapacity, GM_ADDR expandedRowIdx,
                                                        GM_ADDR workspace, const TilingData *tilingData, TPipe *tPipe)
{
    int64_t blockNum = GetBlockNum();
    this->pipe = tPipe;
    this->blockIdx = GetBlockIdx();

    this->coreNum = tilingData->coreNum;
    this->totalLength = tilingData->n * tilingData->k;
    this->srcToDstTilingData = &(tilingData->srcToDstComputeParamsOp);
    this->expertNum = tilingData->expertNum;
    this->dropPadMode = tilingData->dropPadMode;
    this->expertTokensCountOrCumsumFlag = tilingData->expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = tilingData->expertTokensBeforeCapacityFlag;

    if (this->blockIdx == this->srcToDstTilingData->needCoreNum - 1) {
        this->coreRows = this->srcToDstTilingData->lastCoreRows;
        this->perLoopRows = this->srcToDstTilingData->lastCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->lastCoreLastLoopRows;
    } else {
        this->coreRows = this->srcToDstTilingData->perCoreRows;
        this->perLoopRows = this->srcToDstTilingData->perCorePerLoopRows;
        this->lastLoopRows = this->srcToDstTilingData->perCoreLastLoopRows;
    }

    expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdx, Align(this->totalLength, sizeof(int32_t)));
    if (this->dropPadMode == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) {
        expertTokensCountOrCumsumGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensCountOrCumsum, this->expertNum);
    }
    if (this->dropPadMode == DROP_PAD_MODE && this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY) {
        expertTokensBeforeCapacityGm.SetGlobalBuffer((__gm__ int32_t *)expertTokensBeforeCapacity, this->expertNum);
    }

    expandedExpertIdxGm.SetGlobalBuffer(
        (__gm__ int32_t *)workspace + this->blockIdx * this->srcToDstTilingData->perCoreRows,
        Align(this->coreRows, sizeof(int32_t)));

    this->expertNumUbAlign = EXPERT_NUM;
    pipe->InitBuffer(copyInQueue, 1, this->perLoopRows * BLOCK_BYTES);
    pipe->InitBuffer(expertTokenIdxCopyInQueue, 1, this->expertNumUbAlign * sizeof(int32_t));
    pipe->InitBuffer(expertTokenIdxCopyOutQueue, 1, this->expertNumUbAlign * sizeof(int32_t));
}

__aicore__ inline void MoeV2ExpertTokenOutRegBase::Process()
{
    if (this->blockIdx < this->srcToDstTilingData->needCoreNum) {
        if ((this->dropPadMode == DROPLESS_MODE && this->expertTokensCountOrCumsumFlag > EXERPT_TOKENS_NONE) ||
            (this->dropPadMode == DROP_PAD_MODE &&
             this->expertTokensBeforeCapacityFlag == EXERPT_TOKENS_BEFORE_CAPACITY)) {
            int64_t loops = (coreRows + perLoopRows - 1) / perLoopRows;
            currentLoopRows = perLoopRows;
            InitLocal();
            for (int64_t loop = 0; loop < loops - 1; loop++) {
                CopyIn(loop);
                Compute(loop, expertTokenIdxLocal);
            }
            currentLoopRows = lastLoopRows;
            CopyIn(loops - 1);
            Compute(loops - 1, expertTokenIdxLocal);
            expertTokenIdxCopyOutQueue.EnQue<int32_t>(expertTokenIdxLocal);
            expertTokenIdxOutLocal = expertTokenIdxCopyOutQueue.DeQue<int32_t>();
            CopyOutTokenGm();
            expertTokenIdxCopyOutQueue.FreeTensor(this->expertTokenIdxOutLocal);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace MoeInitRoutingV2
#endif // MOE_V2_EXPERT_TOKEN_OUT_REGBASE_H
