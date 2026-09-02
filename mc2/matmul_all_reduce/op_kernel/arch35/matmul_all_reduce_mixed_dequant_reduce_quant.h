/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_mixed_dequant_reduce_quant.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_MIXED_DEQUANT_REDUCE_QUANT_H
#define MATMUL_ALL_REDUCE_MIXED_DEQUANT_REDUCE_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "matmul_all_reduce_dynamic_quant_pertile_utils.h"
#include "../matmul_all_reduce_common.h"

namespace MatmulAllReduceMixedDequantReduceQuantImpl {
using namespace AscendC;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t OUT_LOOP_TWO = 2;

template <class T>
class MatmulAllReduceMixedDequantReduceQuant {
public:
    uint64_t outLoopNum_ = 0;
    uint64_t innerLoopNum_ = 0;
    uint64_t tailUsedCoreNum_ = 0;
    uint64_t procRowTileCnt_ = 0;
    uint64_t procRows_ = 0;
    uint64_t procRowsFirstTail_ = 0;
    uint64_t coreNum_ = 0;
    GM_ADDR inputAddr_;
    GM_ADDR outputAddr_;
    GlobalTensor<T> dequantInputGM_;
    GlobalTensor<T> quantOutputGM_;
    GlobalTensor<float> scaleInputGM_;
    GlobalTensor<float> scaleOutputGM_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> inTilesQ_;
    TQue<QuePosition::VECOUT, 1> outTilesQ_;
    TQue<QuePosition::VECIN, 1> inScalesQ_;
    TQue<QuePosition::VECOUT, 1> outScalesQ_;
    TBuf<TPosition::VECCALC> tempWorkTiles_;
    TBuf<TPosition::VECCALC> tempScaleBuf_;
    TBuf<TPosition::VECCALC> tempMskBuf_;
    TBuf<TPosition::VECCALC> tempReduceOut_;

    __aicore__ inline MatmulAllReduceMixedDequantReduceQuant() {}

    __aicore__ inline void Init(GM_ADDR inputAddr, GM_ADDR outputAddr, uint32_t tileMPerRank, uint32_t tileN,
                                uint32_t oneLineScaleCnt, uint32_t coreNum, uint32_t maxProcRows, TPipe *tPipe)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (GetBlockIdx() >= coreNum) {
            return;
        }
        pipe_ = tPipe;
        uint64_t procRows = tileMPerRank / coreNum;
        this->tailUsedCoreNum_ = tileMPerRank - procRows * coreNum;
        this->procRows_ = (procRows == 0) ? 1 : procRows;
        // m超长场景处理
        if (maxProcRows < this->procRows_) {
            this->procRows_ = maxProcRows;
        }
        this->outLoopNum_ = tileMPerRank / (this->procRows_ * coreNum);
        uint64_t firstTail = tileMPerRank - this->outLoopNum_ * (this->procRows_ * coreNum);
        if (firstTail != 0) {
            this->procRowsFirstTail_ = firstTail / coreNum;
            this->outLoopNum_ += (this->procRowsFirstTail_ == 0) ? 0 : 1;
            this->outLoopNum_ += (this->tailUsedCoreNum_ == 0) ? 0 : 1;
        }
        this->procRowTileCnt_ = maxProcRows / this->procRows_;
        this->innerLoopNum_ = Ceil(oneLineScaleCnt, this->procRowTileCnt_);
        this->coreNum_ = coreNum;
        this->inputAddr_ = inputAddr;
        this->outputAddr_ = outputAddr;
    }

    __aicore__ inline void Process(uint32_t tileN, uint32_t tileMPerRank, uint32_t rankNum, uint32_t totalNandSLen)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (GetBlockIdx() >= this->coreNum_) {
            return;
        }
        for (uint64_t i = 0; i < this->outLoopNum_; i++) {
            InnerProcess(i, rankNum, tileMPerRank, tileN, totalNandSLen);
        }
    }

    __aicore__ inline void InnerProcess(uint32_t outerIdx, uint32_t rankNum, uint32_t tileMPerRank, uint32_t tileN,
                                        uint32_t totalNandSLen)
    {
        bool isFirstTail = false;
        bool isSecondTail = false;
        uint64_t curRows = this->procRows_;
        if ((this->tailUsedCoreNum_ != 0) && (outerIdx == (this->outLoopNum_ - 1))) {
            curRows = 1;
            isSecondTail = true;
            if (GetBlockIdx() >= this->tailUsedCoreNum_) {
                return;
            }
        }
        if (this->procRowsFirstTail_ != 0) {
            isFirstTail = ((this->tailUsedCoreNum_ != 0) && (outerIdx == (this->outLoopNum_ - OUT_LOOP_TWO))) ||
                          ((this->tailUsedCoreNum_ == 0) && (outerIdx == (this->outLoopNum_ - 1)));
            curRows = isFirstTail ? this->procRowsFirstTail_ : curRows;
        }
        pipe_->Reset();
        uint64_t inputBlockLen = curRows * totalNandSLen;
        uint64_t inputBlockSize = inputBlockLen / sizeof(T);
        uint64_t scaleBlockSize = Ceil(inputBlockLen, sizeof(float));
        uint64_t inputOffset = this->procRows_ * totalNandSLen * outerIdx * this->coreNum_;
        if ((outerIdx > 0) && (outerIdx == (this->outLoopNum_ - 1)) && (this->procRowsFirstTail_ != 0) &&
            (this->tailUsedCoreNum_ != 0)) {
            inputOffset = this->procRows_ * totalNandSLen * (outerIdx - 1) * this->coreNum_;
            inputOffset += this->procRowsFirstTail_ * totalNandSLen * this->coreNum_;
        }
        uint64_t inputGmOffset = inputBlockLen * GetBlockIdx() + inputOffset;
        InitLocalBuffer(curRows);
        InitOutputGM(inputGmOffset, tileN, inputBlockSize, scaleBlockSize);
        for (uint64_t i = 0; i < this->innerLoopNum_; i++) {
            LocalTensor<float> reduceOut = tempReduceOut_.Get<float>();
            Duplicate(reduceOut, 0.0f, curRows * this->procRowTileCnt_ * TILELEN);
            for (uint64_t rank = 0; rank < rankNum; rank++) {
                uint64_t rankOffset = rank * tileMPerRank * totalNandSLen;
                InitInputGM(rankOffset, inputGmOffset, tileN, inputBlockSize, scaleBlockSize);
                ProcessDequantReduce(i, curRows, tileN, totalNandSLen, reduceOut);
            }
            ProcessQuant(i, curRows, tileN, totalNandSLen, reduceOut);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void InitLocalBuffer(uint64_t curRows)
    {
        uint64_t bufDataCnt = curRows * this->procRowTileCnt_ * TILELEN;
        pipe_->InitBuffer(inTilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(T));
        pipe_->InitBuffer(outTilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(T));
        pipe_->InitBuffer(inScalesQ_, DOUBLE_BUFFER,
                          Ceil(curRows * this->procRowTileCnt_ * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK);
        pipe_->InitBuffer(outScalesQ_, DOUBLE_BUFFER,
                          Ceil(curRows * this->procRowTileCnt_ * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK);
        pipe_->InitBuffer(tempWorkTiles_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempReduceOut_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempScaleBuf_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempMskBuf_, (Ceil(curRows * this->procRowTileCnt_ * sizeof(float), COMPARE_ALIGN_LEN) *
                                        COMPARE_ALIGN_LEN / sizeof(float)) *
                                           sizeof(int8_t));
    }

    __aicore__ inline void InitInputGM(uint64_t rankOffset, uint64_t gmOffset, uint32_t tileN, uint64_t inputBlockSize,
                                       uint64_t scaleBlockSize)
    {
        dequantInputGM_.SetGlobalBuffer((__gm__ T *)(this->inputAddr_ + gmOffset + rankOffset), inputBlockSize);
        scaleInputGM_.SetGlobalBuffer((__gm__ float *)(this->inputAddr_ + gmOffset + rankOffset + tileN * sizeof(T)),
                                      scaleBlockSize);
    }

    __aicore__ inline void InitOutputGM(uint64_t gmOffset, uint32_t tileN, uint64_t outputBlockSize,
                                        uint64_t scaleBlockSize)
    {
        quantOutputGM_.SetGlobalBuffer((__gm__ T *)(this->outputAddr_ + gmOffset), outputBlockSize);
        scaleOutputGM_.SetGlobalBuffer((__gm__ float *)(this->outputAddr_ + gmOffset + tileN * sizeof(T)),
                                       scaleBlockSize);
    }

    __aicore__ inline void ProcessDequantReduce(uint64_t innerIdx, uint64_t curRows, uint32_t tileN,
                                                uint32_t inputNandSLen, LocalTensor<float> reduceOut)
    {
        uint32_t procRowDataCnt = static_cast<uint32_t>(this->procRowTileCnt_ * TILELEN);
        bool isLast = innerIdx == (this->innerLoopNum_ - 1);
        bool nHasTail = isLast && ((procRowDataCnt * this->innerLoopNum_ - tileN) != 0);
        uint32_t curRowDataCnt = static_cast<uint32_t>(nHasTail ? (tileN - innerIdx * procRowDataCnt) : procRowDataCnt);
        uint32_t curRowDataLen = curRowDataCnt * sizeof(T);
        uint32_t curRowScaleCnt = static_cast<uint32_t>(Ceil(curRowDataCnt, TILELEN));
        uint32_t padCalCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt * TILELEN;
        uint32_t curScaleCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt;
        int64_t gToLNCopyStride = (curRows == 1) ? 0 : (static_cast<int64_t>(inputNandSLen) - curRowDataLen);
        int64_t gToLStrideBlkCnt = ((curRowScaleCnt * TILELEN - curRowDataCnt) * sizeof(T)) / UB_DATABLOCK;
        int64_t gToLSCopyStride =
            (curRows == 1) ? 0 : (static_cast<int64_t>(inputNandSLen) - curRowScaleCnt * sizeof(float));
        DataCopyExtParams gToLNCopyParams = {static_cast<uint16_t>(curRows), curRowDataLen, gToLNCopyStride,
                                             gToLStrideBlkCnt, 0};
        DataCopyExtParams gToLSCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowScaleCnt * sizeof(float)), gToLSCopyStride, 0,
                                             0};
        DataCopyPadExtParams<T> padParamsFp8 = {false, 0, 0, *(reinterpret_cast<T *>(uint8_t(0)))};
        LocalTensor<T> curTiles = inTilesQ_.AllocTensor<T>();
        LocalTensor<float> curScales = inScalesQ_.AllocTensor<float>();
        if (curRowDataCnt != (curRowScaleCnt * TILELEN)) {
            Duplicate(curTiles, *(reinterpret_cast<T *>(uint8_t(0))), padCalCnt);
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        DataCopyPad<T, PaddingMode::Normal>(curTiles, dequantInputGM_[innerIdx * this->procRowTileCnt_ * TILELEN],
                                            gToLNCopyParams, padParamsFp8);
        DataCopyPad<float, PaddingMode::Compact>(curScales, scaleInputGM_[innerIdx * this->procRowTileCnt_],
                                                 gToLSCopyParams, {false, 0, 0, 0});
        inTilesQ_.EnQue<T>(curTiles);
        inScalesQ_.EnQue<float>(curScales);
        LocalTensor<T> tilesLocal = inTilesQ_.DeQue<T>();
        LocalTensor<float> scalesLocal = inScalesQ_.DeQue<float>();
        LocalTensor<float> dequantOut = tempWorkTiles_.Get<float>();
        ComputeDequant(curScaleCnt, padCalCnt, dequantOut, tilesLocal, scalesLocal);
        Add(reduceOut, reduceOut, dequantOut, padCalCnt);
        inTilesQ_.FreeTensor(tilesLocal);
        inScalesQ_.FreeTensor(scalesLocal);
    }

    __aicore__ inline void ComputeDequant(uint32_t curScaleCnt, uint32_t padCalCnt, LocalTensor<float> dequantOut,
                                          LocalTensor<T> tilesLocal, LocalTensor<float> scalesLocal)
    {
        const uint32_t broadCastDst[BROADCAST_DIM] = {curScaleCnt, static_cast<uint32_t>(TILELEN)};
        const uint32_t broadCastSrc[BROADCAST_DIM] = {curScaleCnt, 1};
        LocalTensor<float> tempScale = tempScaleBuf_.Get<float>();
        Cast(dequantOut, tilesLocal, RoundMode::CAST_NONE, padCalCnt);
        Broadcast<float, BROADCAST_DIM, 1, false>(tempScale, scalesLocal, broadCastDst, broadCastSrc);
        Mul(dequantOut, dequantOut, tempScale, padCalCnt);
    }

    __aicore__ inline void ProcessQuant(uint64_t innerIdx, uint64_t curRows, uint32_t tileN, uint32_t outputNandSLen,
                                        LocalTensor<float> reduceOut)
    {
        uint32_t procRowDataCnt = static_cast<uint32_t>(this->procRowTileCnt_ * TILELEN);
        bool isLast = innerIdx == (this->innerLoopNum_ - 1);
        bool nHasTail = isLast && ((procRowDataCnt * this->innerLoopNum_ - tileN) != 0);
        uint32_t curRowDataCnt = static_cast<uint32_t>(nHasTail ? (tileN - innerIdx * procRowDataCnt) : procRowDataCnt);
        uint32_t curRowDataLen = curRowDataCnt * sizeof(float);
        uint32_t curRowScaleCnt = static_cast<uint32_t>(Ceil(curRowDataCnt, TILELEN));
        uint32_t padCalCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt * TILELEN;
        uint32_t curScaleCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt;
        int64_t lToGNCopyStride = (curRows == 1) ? 0 : static_cast<int64_t>(outputNandSLen) - curRowDataCnt * sizeof(T);
        int64_t lToGStrideBlkCnt = (curRowScaleCnt * TILELEN * sizeof(T) - curRowDataCnt * sizeof(T)) / UB_DATABLOCK;
        int64_t lToGSCopyStride =
            (curRows == 1) ? 0 : (static_cast<int64_t>(outputNandSLen) - curRowScaleCnt * sizeof(float));
        DataCopyExtParams lToGNCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowDataCnt * sizeof(T)), lToGStrideBlkCnt,
                                             lToGNCopyStride, 0};
        DataCopyExtParams lToGSCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowScaleCnt * sizeof(float)), 0, lToGSCopyStride,
                                             0};
        LocalTensor<T> curOutTiles = outTilesQ_.AllocTensor<T>();
        LocalTensor<float> curScale = outScalesQ_.AllocTensor<float>();
        LocalTensor<float> tempWorkTiles = tempWorkTiles_.Get<float>();
        LocalTensor<float> tempScale = tempScaleBuf_.Get<float>();
        LocalTensor<uint8_t> tempMskBuf = tempMskBuf_.Get<uint8_t>();
        AscendC::DynamicQuant<T>(curScaleCnt, padCalCnt, reduceOut, curOutTiles, curScale, tempWorkTiles, tempScale,
                                 tempMskBuf);
        outScalesQ_.EnQue<float>(curScale);
        outTilesQ_.EnQue<T>(curOutTiles);
        LocalTensor<float> outScaleLocal = outScalesQ_.DeQue<float>();
        LocalTensor<T> outTilesLocal = outTilesQ_.DeQue<T>();
        DataCopyPad<T, PaddingMode::Normal>(quantOutputGM_[innerIdx * this->procRowTileCnt_ * TILELEN], outTilesLocal,
                                            lToGNCopyParams);
        DataCopyPad<float, PaddingMode::Compact>(scaleOutputGM_[innerIdx * this->procRowTileCnt_], outScaleLocal,
                                                 lToGSCopyParams);
        outTilesQ_.FreeTensor(outTilesLocal);
        outScalesQ_.FreeTensor(outScaleLocal);
    }
};
} // namespace MatmulAllReduceMixedDequantReduceQuantImpl

#endif
