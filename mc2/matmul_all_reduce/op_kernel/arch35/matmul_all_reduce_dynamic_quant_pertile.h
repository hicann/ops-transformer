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
 * \file matmul_all_reduce_dynamic_quant_pertile.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_H
#define MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "matmul_all_reduce_dynamic_quant_pertile_utils.h"
#include "../matmul_all_reduce_common.h"

namespace MatmulAllReduceDynamicQuantPertileImpl {
using namespace AscendC;

constexpr uint32_t OUT_LOOP_TWO = 2;

template <class T, class U = float>
class MatmulAllReduceDynamicQuantPertile {
public:
    uint64_t outLoopNum_ = 0;
    uint64_t innerLoopNum_ = 0;
    uint64_t tailUsedCoreNum_ = 0;
    uint64_t procRowTileCnt_ = 0;
    uint64_t procRows_ = 0;
    uint64_t procRowsFirstTail_ = 0;
    GM_ADDR inputAddr_;
    GM_ADDR outputAddr_;
    GlobalTensor<float> quantInputGM_;
    GlobalTensor<T> quantOutputGM_;
    GlobalTensor<T> dequantInputGM_;
    GlobalTensor<U> dequantOutputGM_;
    GlobalTensor<float> scaleGM_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> tilesQ_;
    TQue<QuePosition::VECOUT, 1> outTilesQ_;
    TQue<QuePosition::VECIN, 1> inScalesQ_;
    TQue<QuePosition::VECOUT, 1> outScalesQ_;
    TBuf<TPosition::VECCALC> tempWorkTiles_;
    TBuf<TPosition::VECCALC> tempScaleBuf_;
    TBuf<TPosition::VECCALC> tempMskBuf_;

    __aicore__ inline MatmulAllReduceDynamicQuantPertile() {}

    __aicore__ inline void Init(GM_ADDR inputAddr, GM_ADDR outputAddr, uint32_t tileM, uint32_t tileN,
                                uint32_t oneLineScaleCnt, uint32_t coreNum, uint32_t maxProcRows, bool isQuant,
                                TPipe *tPipe)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (GetBlockIdx() >= coreNum) {
            return;
        }
        pipe_ = tPipe;
        uint64_t procRows = tileM / coreNum;
        this->procRows_ = (procRows == 0) ? 1 : procRows;
        this->tailUsedCoreNum_ = tileM - procRows * coreNum;
        // m超长场景处理
        if (maxProcRows < this->procRows_) {
            this->procRows_ = maxProcRows;
        }
        this->outLoopNum_ = tileM / (this->procRows_ * coreNum);
        uint64_t firstTail = tileM - this->outLoopNum_ * (this->procRows_ * coreNum);
        if (firstTail != 0) {
            this->procRowsFirstTail_ = firstTail / coreNum;
            this->outLoopNum_ += (this->procRowsFirstTail_ == 0) ? 0 : 1;
            this->outLoopNum_ += (this->tailUsedCoreNum_ == 0) ? 0 : 1;
        }
        this->procRowTileCnt_ = maxProcRows / this->procRows_;
        uint64_t procRowLen = isQuant ?
                                  ((this->procRowTileCnt_ * TILELEN + this->procRowTileCnt_) * sizeof(float)) :
                                  (this->procRowTileCnt_ * TILELEN * sizeof(T) + this->procRowTileCnt_ * sizeof(float));
        this->innerLoopNum_ = isQuant ? Ceil((tileN + oneLineScaleCnt) * sizeof(float), procRowLen) :
                                        Ceil(tileN * sizeof(T) + oneLineScaleCnt * sizeof(float), procRowLen);
        this->inputAddr_ = inputAddr;
        this->outputAddr_ = outputAddr;
    }

    __aicore__ inline void Process(uint32_t tileN, uint32_t coreNum, uint32_t totalNandSLen, bool isQuant)
    {
        if ASCEND_IS_AIC {
            return;
        }
        if (GetBlockIdx() >= coreNum) {
            return;
        }
        for (uint64_t i = 0; i < this->outLoopNum_; i++) {
            InnerProcess(i, tileN, coreNum, totalNandSLen, isQuant);
        }
    }

    __aicore__ inline void InnerProcess(uint32_t outerIdx, uint32_t tileN, uint32_t coreNum, uint32_t totalNandSLen,
                                        bool isQuant)
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
        if (isQuant) {
            InnerInitQuant(outerIdx, tileN, coreNum, totalNandSLen, curRows);
            for (uint64_t i = 0; i < this->innerLoopNum_; i++) {
                ProcessQuant(i, curRows, tileN, totalNandSLen);
            }
        } else {
            InnerInitDequant(outerIdx, tileN, coreNum, totalNandSLen, curRows);
            for (uint64_t i = 0; i < this->innerLoopNum_; i++) {
                ProcessDequant(i, curRows, tileN, totalNandSLen);
            }
        }
    }

    __aicore__ inline void InnerInitQuant(uint64_t outerIdx, uint32_t tileN, uint32_t coreNum, uint32_t totalNandSLen,
                                          uint64_t curRows)
    {
        uint64_t bufDataCnt = curRows * this->procRowTileCnt_ * TILELEN;
        uint64_t inputBlockSize = curRows * tileN;
        uint64_t outputBlockLen = curRows * totalNandSLen;
        uint64_t outputBlockSize = outputBlockLen / sizeof(T);
        uint64_t outputScaleSize = Ceil(outputBlockLen, sizeof(float));
        uint64_t inputOffset = this->procRows_ * tileN * outerIdx * coreNum;
        uint64_t outputOffset = this->procRows_ * totalNandSLen * outerIdx * coreNum;
        if ((outerIdx > 0) && (outerIdx == (this->outLoopNum_ - 1)) && (this->procRowsFirstTail_ != 0) &&
            (this->tailUsedCoreNum_ != 0)) {
            inputOffset = this->procRows_ * tileN * (outerIdx - 1) * coreNum;
            inputOffset += this->procRowsFirstTail_ * tileN * coreNum;
            outputOffset = this->procRows_ * totalNandSLen * (outerIdx - 1) * coreNum;
            outputOffset += this->procRowsFirstTail_ * totalNandSLen * coreNum;
        }
        quantInputGM_.SetGlobalBuffer((__gm__ float *)this->inputAddr_ + inputBlockSize * GetBlockIdx() + inputOffset,
                                      inputBlockSize);
        quantOutputGM_.SetGlobalBuffer((__gm__ T *)(this->outputAddr_ + outputBlockLen * GetBlockIdx() + outputOffset),
                                       outputBlockSize);
        scaleGM_.SetGlobalBuffer(
            (__gm__ float *)(this->outputAddr_ + outputBlockLen * GetBlockIdx() + outputOffset + tileN * sizeof(T)),
            outputScaleSize);
        pipe_->InitBuffer(tilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(outTilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(T));
        pipe_->InitBuffer(outScalesQ_, DOUBLE_BUFFER,
                          Ceil(curRows * this->procRowTileCnt_ * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK);
        pipe_->InitBuffer(tempWorkTiles_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempScaleBuf_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempMskBuf_, (Ceil(curRows * this->procRowTileCnt_ * sizeof(float), COMPARE_ALIGN_LEN) *
                                        COMPARE_ALIGN_LEN / sizeof(float)) *
                                           sizeof(int8_t));
    }

    __aicore__ inline void InnerInitDequant(uint64_t outerIdx, uint32_t tileN, uint32_t coreNum, uint32_t totalNandSLen,
                                            uint64_t curRows)
    {
        uint64_t bufDataCnt = curRows * this->procRowTileCnt_ * TILELEN;
        uint64_t inputBlockLen = curRows * totalNandSLen;
        uint64_t inputBlockSize = inputBlockLen / sizeof(T);
        uint64_t outputBlockSize = curRows * tileN;
        uint64_t scaleBlockSize = Ceil(inputBlockLen, sizeof(float));
        uint64_t inputOffset = this->procRows_ * totalNandSLen * outerIdx * coreNum;
        uint64_t outputOffset = this->procRows_ * tileN * outerIdx * coreNum;
        if ((outerIdx > 0) && (outerIdx == (this->outLoopNum_ - 1)) && (this->procRowsFirstTail_ != 0) &&
            (this->tailUsedCoreNum_ != 0)) {
            inputOffset = this->procRows_ * totalNandSLen * (outerIdx - 1) * coreNum;
            inputOffset += this->procRowsFirstTail_ * totalNandSLen * coreNum;
            outputOffset = this->procRows_ * tileN * (outerIdx - 1) * coreNum;
            outputOffset += this->procRowsFirstTail_ * tileN * coreNum;
        }
        dequantInputGM_.SetGlobalBuffer((__gm__ T *)(this->inputAddr_ + inputBlockLen * GetBlockIdx() + inputOffset),
                                        inputBlockSize);
        dequantOutputGM_.SetGlobalBuffer((__gm__ U *)this->outputAddr_ + outputBlockSize * GetBlockIdx() + outputOffset,
                                         outputBlockSize);
        scaleGM_.SetGlobalBuffer(
            (__gm__ float *)(this->inputAddr_ + inputBlockLen * GetBlockIdx() + inputOffset + tileN * sizeof(T)),
            scaleBlockSize);
        pipe_->InitBuffer(tilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(T));
        pipe_->InitBuffer(inScalesQ_, DOUBLE_BUFFER,
                          Ceil(curRows * this->procRowTileCnt_ * sizeof(float), UB_DATABLOCK) * UB_DATABLOCK);
        pipe_->InitBuffer(outTilesQ_, DOUBLE_BUFFER, bufDataCnt * sizeof(U));
        pipe_->InitBuffer(tempWorkTiles_, bufDataCnt * sizeof(float));
        pipe_->InitBuffer(tempScaleBuf_, bufDataCnt * sizeof(float));
    }

    __aicore__ inline void ProcessQuant(uint64_t innerIdx, uint64_t curRows, uint32_t tileN, uint32_t outputNandSLen)
    {
        uint32_t procRowDataCnt = static_cast<uint32_t>(this->procRowTileCnt_ * TILELEN);
        bool isLast = innerIdx == (this->innerLoopNum_ - 1);
        bool nHasTail = isLast && ((procRowDataCnt * this->innerLoopNum_ - tileN) != 0);
        uint32_t curRowDataCnt = static_cast<uint32_t>(nHasTail ? (tileN - innerIdx * procRowDataCnt) : procRowDataCnt);
        uint32_t curRowDataLen = curRowDataCnt * sizeof(float);
        uint32_t curRowScaleCnt = static_cast<uint32_t>(Ceil(curRowDataCnt, TILELEN));
        uint32_t padCalCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt * TILELEN;
        uint32_t curScaleCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt;
        int64_t gToLCopyStride = (curRows == 1) ? 0 : static_cast<int64_t>((tileN - curRowDataCnt) * sizeof(float));
        int64_t gToLStrideBlkCnt = ((curRowScaleCnt * TILELEN - curRowDataCnt) * sizeof(float)) / UB_DATABLOCK;
        int64_t lToGNCopyStride = (curRows == 1) ? 0 : static_cast<int64_t>(outputNandSLen) - curRowDataCnt * sizeof(T);
        int64_t lToGStrideBlkCnt = (curRowScaleCnt * TILELEN * sizeof(T) - curRowDataCnt * sizeof(T)) / UB_DATABLOCK;
        int64_t lToGSCopyStride =
            (curRows == 1) ? 0 : (static_cast<int64_t>(outputNandSLen) - curRowScaleCnt * sizeof(float));
        DataCopyExtParams gToLCopyParams = {static_cast<uint16_t>(curRows), curRowDataLen, gToLCopyStride,
                                            gToLStrideBlkCnt, 0};
        DataCopyExtParams lToGNCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowDataCnt * sizeof(T)), lToGStrideBlkCnt,
                                             lToGNCopyStride, 0};
        DataCopyExtParams lToGSCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowScaleCnt * sizeof(float)), 0, lToGSCopyStride,
                                             0};
        uint8_t rightPadCnt = ((curRowDataLen % UB_DATABLOCK) == 0) ?
                                  0 :
                                  static_cast<uint8_t>((UB_DATABLOCK - curRowDataLen % UB_DATABLOCK) / sizeof(float));
        DataCopyPadExtParams<float> padParams = {rightPadCnt != 0, 0, rightPadCnt, 0};
        LocalTensor<float> curTiles = tilesQ_.AllocTensor<float>();
        if (curRowDataCnt != (curRowScaleCnt * TILELEN)) {
            Duplicate(curTiles, 0.0f, padCalCnt);
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        DataCopyPad<float, PaddingMode::Normal>(curTiles, quantInputGM_[innerIdx * procRowDataCnt], gToLCopyParams,
                                                padParams);
        tilesQ_.EnQue<float>(curTiles);
        CalcQuant(innerIdx, curScaleCnt, padCalCnt, lToGNCopyParams, lToGSCopyParams);
    }

    __aicore__ inline void CalcQuant(uint64_t innerIdx, uint32_t curScaleCnt, uint32_t padCalCnt,
                                     DataCopyExtParams lToGNCopyParams, DataCopyExtParams lToGSCopyParams)
    {
        LocalTensor<float> tilesLocal = tilesQ_.DeQue<float>();
        LocalTensor<T> curOutTiles = outTilesQ_.AllocTensor<T>();
        LocalTensor<float> curScale = outScalesQ_.AllocTensor<float>();
        LocalTensor<float> tempWorkTiles = tempWorkTiles_.Get<float>();
        LocalTensor<float> tempScale = tempScaleBuf_.Get<float>();
        LocalTensor<uint8_t> tempMskBuf = tempMskBuf_.Get<uint8_t>();
        AscendC::DynamicQuant<T>(curScaleCnt, padCalCnt, tilesLocal, curOutTiles, curScale, tempWorkTiles, tempScale,
                                 tempMskBuf);
        outScalesQ_.EnQue<float>(curScale);
        outTilesQ_.EnQue<T>(curOutTiles);
        LocalTensor<float> outScaleLocal = outScalesQ_.DeQue<float>();
        LocalTensor<T> outTilesLocal = outTilesQ_.DeQue<T>();
        DataCopyPad<T, PaddingMode::Normal>(quantOutputGM_[innerIdx * this->procRowTileCnt_ * TILELEN], outTilesLocal,
                                            lToGNCopyParams);
        DataCopyPad<float, PaddingMode::Compact>(scaleGM_[innerIdx * this->procRowTileCnt_], outScaleLocal,
                                                 lToGSCopyParams);
        outTilesQ_.FreeTensor(outTilesLocal);
        outScalesQ_.FreeTensor(outScaleLocal);
        tilesQ_.FreeTensor(tilesLocal);
    }

    __aicore__ inline void ProcessDequant(uint64_t innerIdx, uint64_t curRows, uint32_t tileN, uint32_t inputNandSLen)
    {
        uint32_t procRowDataCnt = static_cast<uint32_t>(this->procRowTileCnt_ * TILELEN);
        bool isLast = innerIdx == (this->innerLoopNum_ - 1);
        bool nHasTail = isLast && ((procRowDataCnt * this->innerLoopNum_ - tileN) != 0);
        uint32_t curRowDataCnt = static_cast<uint32_t>(nHasTail ? (tileN - innerIdx * procRowDataCnt) : procRowDataCnt);
        uint32_t curRowDataLen = curRowDataCnt * sizeof(T);
        uint32_t curRowScaleCnt = static_cast<uint32_t>(Ceil(curRowDataCnt, TILELEN));
        uint32_t padCalCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt * TILELEN;
        uint32_t curScaleCnt = static_cast<uint32_t>(curRows) * curRowScaleCnt;
        int64_t lToGCopyStride = (curRows == 1) ? 0 : static_cast<int64_t>((tileN - curRowDataCnt) * sizeof(U));
        int64_t lToGStrideBlkCnt = ((curRowScaleCnt * TILELEN - curRowDataCnt) * sizeof(U)) / UB_DATABLOCK;
        int64_t gToLNCopyStride = (curRows == 1) ? 0 : (static_cast<int64_t>(inputNandSLen) - curRowDataLen);
        int64_t gToLStrideBlkCnt = ((curRowScaleCnt * TILELEN - curRowDataCnt) * sizeof(T)) / UB_DATABLOCK;
        int64_t gToLSCopyStride =
            (curRows == 1) ? 0 : (static_cast<int64_t>(inputNandSLen) - curRowScaleCnt * sizeof(float));
        DataCopyExtParams lToGCopyParams = {static_cast<uint16_t>(curRows),
                                            static_cast<uint32_t>(curRowDataCnt * sizeof(U)), lToGStrideBlkCnt,
                                            lToGCopyStride, 0};
        DataCopyExtParams gToLNCopyParams = {static_cast<uint16_t>(curRows), curRowDataLen, gToLNCopyStride,
                                             gToLStrideBlkCnt, 0};
        DataCopyExtParams gToLSCopyParams = {static_cast<uint16_t>(curRows),
                                             static_cast<uint32_t>(curRowScaleCnt * sizeof(float)), gToLSCopyStride, 0,
                                             0};
        DataCopyPadExtParams<T> padParamsFp8 = {false, 0, 0, *(reinterpret_cast<T *>(uint8_t(0)))};
        LocalTensor<T> curTiles = tilesQ_.AllocTensor<T>();
        LocalTensor<float> curScales = inScalesQ_.AllocTensor<float>();
        DataCopyPad<T, PaddingMode::Normal>(curTiles, dequantInputGM_[innerIdx * this->procRowTileCnt_ * TILELEN],
                                            gToLNCopyParams, padParamsFp8);
        DataCopyPad<float, PaddingMode::Compact>(curScales, scaleGM_[innerIdx * this->procRowTileCnt_], gToLSCopyParams,
                                                 {false, 0, 0, 0});
        tilesQ_.EnQue<T>(curTiles);
        inScalesQ_.EnQue<float>(curScales);
        CalcDequant(innerIdx, padCalCnt, lToGCopyParams, static_cast<uint32_t>(curRows * curRowScaleCnt));
    }

    __aicore__ inline void CalcDequant(uint64_t innerIdx, uint32_t padCalCnt, DataCopyExtParams lToGCopyParams,
                                       uint32_t curScaleCnt)
    {
        LocalTensor<U> outLocal = outTilesQ_.AllocTensor<U>();
        LocalTensor<T> tilesLocal = tilesQ_.DeQue<T>();
        LocalTensor<float> scalesLocal = inScalesQ_.DeQue<float>();
        LocalTensor<float> tempOut = tempWorkTiles_.Get<float>();
        LocalTensor<float> tempScale = tempScaleBuf_.Get<float>();
        AscendC::DynamicDequant<T, U>(curScaleCnt, padCalCnt, outLocal, tilesLocal, scalesLocal, tempOut, tempScale);
        outTilesQ_.EnQue<U>(outLocal);
        LocalTensor<U> outTiles = outTilesQ_.DeQue<U>();
        DataCopyPad<U, PaddingMode::Normal>(dequantOutputGM_[innerIdx * this->procRowTileCnt_ * TILELEN], outTiles,
                                            lToGCopyParams);
        outTilesQ_.FreeTensor(outTiles);
        tilesQ_.FreeTensor(tilesLocal);
        inScalesQ_.FreeTensor(scalesLocal);
    }
};
} // namespace MatmulAllReduceDynamicQuantPertileImpl
#endif // MATMUL_ALL_REDUCE_DYNAMIC_QUANT_PERTILE_H
