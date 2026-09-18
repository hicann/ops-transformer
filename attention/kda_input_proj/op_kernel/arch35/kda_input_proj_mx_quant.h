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
 * \file kda_input_proj_mx_quant.h
 * \brief Stage1 AIV: DynamicMxQuant on x (tilingKey=33: tail-axis opti + scaleAlg=0 OCP, bf16 -> fp8e4m3)
 *
 * Static-tensor style: LocalTensor is constructed via (TPosition, byteOffset, elemCount) without TPipe/TQue.
 */

#ifndef KDA_INPUT_PROJ_MX_QUANT_H
#define KDA_INPUT_PROJ_MX_QUANT_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kda_input_proj_common.h"
#include "../kda_input_proj_tiling_data.h"

namespace KdaInputProj {
using namespace AscendC;

namespace {
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_EIGHT = 8;

constexpr uint32_t VF_LEN_16 = 128;
constexpr uint32_t VF_LEN_16_DOUBLE = VF_LEN_16 * 2;
constexpr uint32_t VF_LEN_32 = 64;
constexpr uint32_t ELEMENT_AFTER_REDUCE_ = 8;

constexpr uint16_t BF16_NAN_CUSTOM = 0x7f81;
constexpr uint16_t BF16_MAX_EXP = 0x7f80;
constexpr uint16_t FP8_DEFAULT_MAX_EXP = 0x00ff;
constexpr uint16_t BF16_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t BF16_SHR_NUM = 7;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;

__aicore__ inline event_t EvtMte2V(int64_t slot)
{
    return slot == 0 ? EVENT_ID0 : EVENT_ID1;
}
__aicore__ inline event_t EvtVMte3(int64_t slot)
{
    return slot == 0 ? EVENT_ID2 : EVENT_ID3;
}
__aicore__ inline event_t EvtMte3Mte2(int64_t slot)
{
    return slot == 0 ? EVENT_ID4 : EVENT_ID5;
}
__aicore__ inline event_t EvtMte3V(int64_t slot)
{
    return slot == 0 ? EVENT_ID6 : EVENT_ID7;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    return b == 0 ? 0 : a / b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    return CeilDiv(a, b) * b;
}

__aicore__ inline void InitUbEventFlags()
{
    for (int64_t slot = 0; slot < DB_BUFFER; ++slot) {
        SetFlag<HardEvent::MTE3_MTE2>(EvtMte3Mte2(slot));
        SetFlag<HardEvent::MTE3_V>(EvtMte3V(slot));
    }
}

__aicore__ inline void ClearUbEventFlags()
{
    for (int64_t slot = 0; slot < DB_BUFFER; ++slot) {
        WaitFlag<HardEvent::MTE3_MTE2>(EvtMte3Mte2(slot));
        WaitFlag<HardEvent::MTE3_V>(EvtMte3V(slot));
    }
}
} // namespace

template <typename TypePack>
class KdaInputProjMxQuant {
public:
    __aicore__ inline KdaInputProjMxQuant() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *quantX, __gm__ uint8_t *xScale,
                                const optiling::KdaInputProjTilingData *__restrict tiling);
    __aicore__ inline void Process();

private:
    using DtypeX = typename TypePack::DtypeX;

    __aicore__ inline void ParseTilingData(const optiling::KdaInputProjMxQuantParams *tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();
    __aicore__ inline void InitUbLayout();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                  int64_t ubFactorColNum, int64_t ubFactorColBlockNum, int64_t bufSlot);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum, int64_t bufSlot);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum,
                                   int64_t ubFactorColNum, int64_t ubFactorColBlockNum, int64_t bufSlot);
    __aicore__ inline void ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t *xLocalAddr, __ubuf__ uint16_t *maxExpAddr,
                                                uint16_t loopNum2VF);
    __aicore__ inline void ComputeScaleOcp(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
                                           __ubuf__ uint16_t *recipScaleLocalAddr, uint16_t loopNum1VF,
                                           uint32_t totalScaleInUB);
    __aicore__ inline void ComputeData(__ubuf__ bfloat16_t *xLocalAddr, __ubuf__ uint16_t *recipScaleLocalAddr,
                                       __ubuf__ int8_t *yLocalAddr, uint16_t loopNum2VF);

    GlobalTensor<DtypeX> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;

    LocalTensor<DtypeX> xLocal_[DB_BUFFER];
    LocalTensor<uint8_t> yLocal_[DB_BUFFER];
    LocalTensor<uint16_t> scaleLocal_[DB_BUFFER];
    LocalTensor<uint8_t> scaleLocalBytes_[DB_BUFFER];
    LocalTensor<uint16_t> maxExpLocal_[DB_BUFFER];
    LocalTensor<uint16_t> recipScaleLocal_[DB_BUFFER];

    const optiling::KdaInputProjTilingData *tiling_{nullptr};

    int64_t blockSize_{0};
    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};
    int64_t rowTileNum_{0};
    int64_t colTileNum_{0};
    int64_t rowNum_{1};
    int64_t colNum_{1};
    int64_t colNormalBlockNum_{0};
    int64_t colTailLen_{0};
    int64_t rowNormalBlockNum_{0};
    int64_t rowTailLen_{0};
    int64_t maxUbBlockNum_{0};

    int64_t coreIdx_{0};
    int64_t coreColIdx_{0};
    int64_t coreRowIdx_{0};
    int64_t xGmOffset_{0};
    int64_t scaleGmOffset_{0};
    int64_t scaleColNum_{0};

    int64_t ubFactorColNum_{0};
    int64_t ubFactorCol32BlockNum_{0};
    int64_t ubFactorRow1BlockNum_{0};
    int64_t ubFactorColLoopNum_{0};
    int64_t ubFactorRowLoopNum_{0};
    int64_t ubFactorColNormalBlockNum_{0};
    int64_t ubFactorColTailBlockNum_{0};
    int64_t ubFactorColNormalBlockLen_{0};
    int64_t ubFactorColTailBlockLen_{0};
    int64_t ubFactorRowNormalBlockNum_{0};
    int64_t ubFactorRowTailBlockNum_{0};

    uint16_t f8Emax_{0};
};

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::Init(__gm__ uint8_t *x, __gm__ uint8_t *quantX,
                                                           __gm__ uint8_t *xScale,
                                                           const optiling::KdaInputProjTilingData *__restrict tiling)
{
    tiling_ = tiling;
    if ASCEND_IS_AIC {
        return;
    }

#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(&tiling_->mxQuantParams);
    GetGmParams();
    GetUbParams();
    InitUbLayout();

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DtypeX *>(x) + xGmOffset_);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(quantX) + xGmOffset_);
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(xScale) + scaleGmOffset_);

    f8Emax_ = FP8_E4M3_MAX_EXP;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::InitUbLayout()
{
    const uint32_t xSlotElems = static_cast<uint32_t>(maxUbBlockNum_ * blockSize_);
    const uint32_t ySlotElems = static_cast<uint32_t>(maxUbBlockNum_ * blockSize_);
    const uint32_t scaleSlotElems = static_cast<uint32_t>(CeilAlign(maxUbBlockNum_, static_cast<int64_t>(VF_LEN_16)));
    const uint32_t maxExpSlotElems = static_cast<uint32_t>(maxUbBlockNum_ * 2);
    const uint32_t recipScaleSlotElems = static_cast<uint32_t>(maxUbBlockNum_ * 2);

    const uint32_t xSlotBytes = xSlotElems * static_cast<uint32_t>(sizeof(DtypeX));
    const uint32_t ySlotBytes = ySlotElems * static_cast<uint32_t>(sizeof(uint8_t));
    const uint32_t scaleSlotBytes = scaleSlotElems * static_cast<uint32_t>(sizeof(uint16_t));
    const uint32_t maxExpSlotBytes = maxExpSlotElems * static_cast<uint32_t>(sizeof(uint16_t));
    const uint32_t recipScaleSlotBytes = recipScaleSlotElems * static_cast<uint32_t>(sizeof(uint16_t));

    uint32_t xBase = 0;
    uint32_t yBase = xBase + static_cast<uint32_t>(DB_BUFFER) * xSlotBytes;
    uint32_t scaleBase = yBase + static_cast<uint32_t>(DB_BUFFER) * ySlotBytes;
    uint32_t maxExpBase = scaleBase + static_cast<uint32_t>(DB_BUFFER) * scaleSlotBytes;
    uint32_t recipScaleBase = maxExpBase + static_cast<uint32_t>(DB_BUFFER) * maxExpSlotBytes;

    for (int64_t i = 0; i < DB_BUFFER; ++i) {
        const uint32_t slot = static_cast<uint32_t>(i);
        xLocal_[i] = LocalTensor<DtypeX>(TPosition::VECIN, xBase + slot * xSlotBytes, xSlotElems);
        yLocal_[i] = LocalTensor<uint8_t>(TPosition::VECOUT, yBase + slot * ySlotBytes, ySlotElems);
        scaleLocal_[i] = LocalTensor<uint16_t>(TPosition::VECOUT, scaleBase + slot * scaleSlotBytes, scaleSlotElems);
        scaleLocalBytes_[i] =
            LocalTensor<uint8_t>(TPosition::VECOUT, scaleBase + slot * scaleSlotBytes, scaleSlotBytes);
        maxExpLocal_[i] =
            LocalTensor<uint16_t>(TPosition::VECCALC, maxExpBase + slot * maxExpSlotBytes, maxExpSlotElems);
        recipScaleLocal_[i] =
            LocalTensor<uint16_t>(TPosition::VECCALC, recipScaleBase + slot * recipScaleSlotBytes, recipScaleSlotElems);
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::ParseTilingData(
    const optiling::KdaInputProjMxQuantParams *tilingData)
{
    blockSize_ = tilingData->blockSize;
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    rowTileNum_ = tilingData->rowTileNum;
    colTileNum_ = tilingData->colTileNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colNormalBlockNum_ = tilingData->colNormalBlockNum;
    colTailLen_ = tilingData->colTailLen;
    rowNormalBlockNum_ = tilingData->rowNormalBlockNum;
    rowTailLen_ = tilingData->rowTailLen;
    maxUbBlockNum_ = tilingData->maxUbBlockNum;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::GetGmParams()
{
    coreIdx_ = static_cast<int64_t>(GetBlockIdx());
    if (colTileNum_ <= 0) {
        return;
    }
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ +
                 coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT * blockSize_;
    scaleColNum_ = CeilDiv(CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffset_ =
        coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::GetUbParams()
{
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorCol32BlockNum_ = CeilDiv(colTailLen_, blockSize_);
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorCol32BlockNum_ = colNormalBlockNum_ * DIGIT_EIGHT;
        ubFactorColNum_ = ubFactorCol32BlockNum_ * blockSize_;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRow1BlockNum_ = CeilDiv(rowTailLen_, DIGIT_ONE);
    } else {
        ubFactorRow1BlockNum_ = rowNormalBlockNum_ * DIGIT_ONE;
    }

    ubFactorColLoopNum_ = CeilDiv(ubFactorCol32BlockNum_, maxUbBlockNum_);
    ubFactorColNormalBlockNum_ = CeilDiv(ubFactorCol32BlockNum_, ubFactorColLoopNum_);
    // kernel CopyIn/CopyOut 按 scale pack-2 对齐，列方向头块 block 数需向上对齐到偶数
    ubFactorColNormalBlockNum_ = CeilAlign(ubFactorColNormalBlockNum_, DIGIT_TWO);
    ubFactorColTailBlockNum_ = ubFactorCol32BlockNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockNum_;
    ubFactorColNormalBlockLen_ = ubFactorColNormalBlockNum_ * blockSize_;
    ubFactorColTailBlockLen_ = ubFactorColNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockLen_;

    ubFactorRowNormalBlockNum_ = FloorDiv(maxUbBlockNum_, ubFactorColNormalBlockNum_);
    ubFactorRowLoopNum_ = CeilDiv(ubFactorRow1BlockNum_, ubFactorRowNormalBlockNum_);
    ubFactorRowNormalBlockNum_ = CeilDiv(ubFactorRow1BlockNum_, ubFactorRowLoopNum_);
    ubFactorRowTailBlockNum_ = ubFactorRow1BlockNum_ - (ubFactorRowLoopNum_ - DIGIT_ONE) * ubFactorRowNormalBlockNum_;
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }

    InitUbEventFlags();

    int64_t dim0LoopIdx = 0;
    int64_t dim1LoopIdx = 0;
    int64_t loopIdx = 0;
    for (dim0LoopIdx = 0; dim0LoopIdx < ubFactorRowLoopNum_ - 1; dim0LoopIdx++) {
        for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
            int64_t bufSlot = loopIdx % DB_BUFFER;
            CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                   ubFactorColNormalBlockNum_, bufSlot);
            Compute(ubFactorRowNormalBlockNum_, ubFactorColNormalBlockNum_, bufSlot);
            CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_,
                    ubFactorColNormalBlockNum_, bufSlot);
            loopIdx++;
        }
        int64_t bufSlot = loopIdx % DB_BUFFER;
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_,
               bufSlot);
        Compute(ubFactorRowNormalBlockNum_, ubFactorColTailBlockNum_, bufSlot);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_,
                ubFactorColTailBlockNum_, bufSlot);
        loopIdx++;
    }
    for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
        int64_t bufSlot = loopIdx % DB_BUFFER;
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
               ubFactorColNormalBlockNum_, bufSlot);
        Compute(ubFactorRowTailBlockNum_, ubFactorColNormalBlockNum_, bufSlot);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_,
                ubFactorColNormalBlockNum_, bufSlot);
        loopIdx++;
    }
    int64_t bufSlot = loopIdx % DB_BUFFER;
    CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_,
           bufSlot);
    Compute(ubFactorRowTailBlockNum_, ubFactorColTailBlockNum_, bufSlot);
    CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_,
            bufSlot);

    ClearUbEventFlags();
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                             int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                             int64_t ubFactorColBlockNum, int64_t bufSlot)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    WaitFlag<HardEvent::MTE3_MTE2>(EvtMte3Mte2(bufSlot));

    if (scaleIsNotOdd != 0) {
        // 补零必须先于本次 MTE2 搬入完成，Set/Wait 成对出现，不跨迭代残留
        Duplicate<DtypeX>(xLocal_[bufSlot], static_cast<DtypeX>(0), maxUbBlockNum_ * blockSize_);
        SetFlag<HardEvent::V_MTE2>(EvtMte2V(bufSlot));
        WaitFlag<HardEvent::V_MTE2>(EvtMte2V(bufSlot));
    }

    int64_t padRightNumTmp = (ubFactorColNum % 32 > 16) ? (32 - ubFactorColNum % 32) : 16;
    int64_t padRightNum = (ubFactorColNum % 32 == 0) ? 0 : padRightNumTmp;
    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;
    const bool contiguous = (ubFactorColNum == colNum_) && (padRightNum == 0) && (scaleIsNotOdd == 0);
    if (contiguous) {
        DataCopy(xLocal_[bufSlot], xGm_[offset], static_cast<uint32_t>(ubFactorRowNum * ubFactorColNum));
    } else {
        DataCopyExtParams copyInParam = {
            static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(DtypeX)),
            static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(DtypeX)),
            static_cast<uint32_t>(scaleIsNotOdd * sizeof(DtypeX)), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<DtypeX> padParams = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum),
                                                  static_cast<DtypeX>(0)};
        DataCopyPad(xLocal_[bufSlot], xGm_[offset], copyInParam, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EvtMte2V(bufSlot));
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx,
                                                              int64_t ubFactorRowNum, int64_t ubFactorColNum,
                                                              int64_t ubFactorColBlockNum, int64_t bufSlot)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    WaitFlag<HardEvent::V_MTE3>(EvtVMte3(bufSlot));

    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;
    if ((ubFactorColNum == colNum_) && (scaleIsNotOdd == 0)) {
        DataCopy(yGm_[offset], yLocal_[bufSlot], static_cast<uint32_t>(ubFactorRowNum * ubFactorColNum));
    } else {
        DataCopyExtParams copyOutParamY = {
            static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(uint8_t)),
            static_cast<uint32_t>(scaleIsNotOdd), static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(uint8_t)),
            static_cast<uint32_t>(0)};
        DataCopyPad(yGm_[offset], yLocal_[bufSlot], copyOutParamY);
    }

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    copyOutParamScale.blockCount = ubFactorRowNum;
    copyOutParamScale.blockLen = ubFactorColBlockNum + scaleIsNotOdd;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = scaleColNum_ - copyOutParamScale.blockLen;

    int64_t scaleOffset =
        dim0LoopIdx * ubFactorRowNormalBlockNum_ * scaleColNum_ + dim1LoopIdx * ubFactorColNormalBlockNum_;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[scaleOffset], scaleLocalBytes_[bufSlot], copyOutParamScale);

    SetFlag<HardEvent::MTE3_MTE2>(EvtMte3Mte2(bufSlot));
    SetFlag<HardEvent::MTE3_V>(EvtMte3V(bufSlot));
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::ComputeMaxExpOcpBf16(__ubuf__ bfloat16_t *xLocalAddr,
                                                                           __ubuf__ uint16_t *maxExpAddr,
                                                                           uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> x0;
        Reg::RegTensor<bfloat16_t> x1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, BF16_MAX_EXP);

        Reg::MaskReg mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignRegForStore ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t> &)x0, expMaskBF16, mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t> &)x1, expMaskBF16, mask);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, mask);
            Reg::ReduceDataBlock<ReduceType::MAX>(xMaxExp, xMaxExp, mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpAddr, xMaxExp, ureg,
                                                                            ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::ComputeScaleOcp(__ubuf__ uint16_t *maxExpAddr,
                                                                      __ubuf__ uint16_t *mxScaleLocalAddr,
                                                                      __ubuf__ uint16_t *recipScaleLocalAddr,
                                                                      uint16_t loopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f8Emax_);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(recipScaleLocalAddr, halfScale, VF_LEN_16,
                                                                          preMaskScale);
        }
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::ComputeData(__ubuf__ bfloat16_t *xLocalAddr,
                                                                  __ubuf__ uint16_t *recipScaleLocalAddr,
                                                                  __ubuf__ int8_t *yLocalAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1 = Reg::CreateMask<bfloat16_t>();
        Reg::MaskReg dataMask3 = Reg::CreateMask<float>();
        Reg::MaskReg dataMask4 = Reg::CreateMask<float>();
        Reg::MaskReg dataMask5 = Reg::CreateMask<fp8_e4m3fn_t>();
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<bfloat16_t> x0;
        Reg::RegTensor<bfloat16_t> x1;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<fp8_e4m3fn_t> x0ZeroFP8;
        Reg::RegTensor<fp8_e4m3fn_t> x0OneFP8;
        Reg::RegTensor<fp8_e4m3fn_t> x1ZeroFP8;
        Reg::RegTensor<fp8_e4m3fn_t> x1OneFP8;

        static constexpr Reg::CastTrait castTraitZero = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to80 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to81 = {Reg::RegLayout::ONE, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to82 = {Reg::RegLayout::TWO, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to83 = {Reg::RegLayout::THREE, Reg::SatMode::SAT,
                                                           Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);

            Reg::Mul(x0, x0, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);
            Reg::Mul(x1, x1, (Reg::RegTensor<bfloat16_t> &)halfScaleForMul, dataMask1);

            Reg::Cast<float, bfloat16_t, castTraitZero>(x0ZeroFP32, x0, dataMask1);
            Reg::Cast<float, bfloat16_t, castTraitOne>(x0OneFP32, x0, dataMask1);
            Reg::Cast<float, bfloat16_t, castTraitZero>(x1ZeroFP32, x1, dataMask1);
            Reg::Cast<float, bfloat16_t, castTraitOne>(x1OneFP32, x1, dataMask1);

            Reg::Cast<fp8_e4m3fn_t, float, castTrait32to80>(x0ZeroFP8, x0ZeroFP32, dataMask3);
            Reg::Cast<fp8_e4m3fn_t, float, castTrait32to81>(x1ZeroFP8, x1ZeroFP32, dataMask4);
            Reg::Cast<fp8_e4m3fn_t, float, castTrait32to82>(x0OneFP8, x0OneFP32, dataMask3);
            Reg::Cast<fp8_e4m3fn_t, float, castTrait32to83>(x1OneFP8, x1OneFP32, dataMask4);

            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x0OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t> &)x1ZeroFP8, (Reg::RegTensor<uint8_t> &)x1ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x1OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                     (Reg::RegTensor<uint8_t> &)x1ZeroFP8, dataMask5);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yLocalAddr, (Reg::RegTensor<int8_t> &)x0ZeroFP8, VF_LEN_16_DOUBLE, dataMask5);
        }
    }
}

template <typename TypePack>
__aicore__ inline void KdaInputProjMxQuant<TypePack>::Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum,
                                                              int64_t bufSlot)
{
    WaitFlag<HardEvent::MTE2_V>(EvtMte2V(bufSlot));
    WaitFlag<HardEvent::MTE3_V>(EvtMte3V(bufSlot));

    ubFactorColBlockNum = CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    uint32_t totalBlockNum = ubFactorRowBlockNum * ubFactorColBlockNum;

    uint16_t loopNum2VF = static_cast<uint16_t>(
        CeilDiv(static_cast<uint32_t>(totalBlockNum), static_cast<uint32_t>(ELEMENT_AFTER_REDUCE_)));
    uint16_t loopNum1VF =
        static_cast<uint16_t>(CeilDiv(static_cast<uint32_t>(totalBlockNum), static_cast<uint32_t>(VF_LEN_16)));

    auto xLocalAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xLocal_[bufSlot].GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(scaleLocal_[bufSlot].GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t *>(yLocal_[bufSlot].GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(maxExpLocal_[bufSlot].GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(recipScaleLocal_[bufSlot].GetPhyAddr());

    ComputeMaxExpOcpBf16(xLocalAddr, maxExpLocalAddr, loopNum2VF);
    ComputeScaleOcp(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, loopNum1VF, totalBlockNum);
    ComputeData(xLocalAddr, recipScaleLocalAddr, yLocalAddr, loopNum2VF);

    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EvtVMte3(bufSlot));
}

} // namespace KdaInputProj

#endif // KDA_INPUT_PROJ_MX_QUANT_H
