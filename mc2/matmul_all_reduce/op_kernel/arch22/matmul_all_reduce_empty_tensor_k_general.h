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
 * \file matmul_all_reduce_empty_tensor_k_general.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H
#define MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../matmul_all_reduce_common.h"
#include "matmul_all_reduce_add_x3.h"
#include "../weight_quant_matmul_all_reduce_tiling_data.h"
#include "../unquant_matmul_all_reduce_tiling_data.h"

constexpr uint32_t EMPTY_TENSOR_BIAS_UB_FACTOR = 1;

namespace MatmulAllReduceImpl {
using namespace AscendC;
template <typename yType>
class MatmulAllReduceEmptyTensorKGeneral {
public:
    __aicore__ inline MatmulAllReduceEmptyTensorKGeneral(MC2GmAddrs *addrs, MC2TilingHeader *tilingData, TPipe *tPipe)
        : addrs_(addrs),
          tPipe_(tPipe)
    {
        param_ = &(tilingData->param);
        cOffset_ = (uint64_t)param_->rankN * (uint64_t)param_->rankM;
#ifdef MC2_WEIGHT_QUANT
        biasFlag_ =
            (((Mc2Tiling::WeightQuantMatmulAllReduceTilingData *)tilingData)->tilematmulTiling.matmulTiling.isBias !=
             0U);
#else
        biasFlag_ =
            (((Mc2Tiling::MatmulAllReduce910TilingData *)tilingData)->tilematmulTiling.matmulTiling.isBias != 0U);
#endif
    }

    __aicore__ inline void Init()
    {
        hccl_.Init(GetHcclContext<0>());
        notifyFlag_ = (GetBlockIdx() == 0);
        if (notifyFlag_) {
            hcclHandleId_ =
                hccl_.AllReduce(addrs_->cGM, addrs_->outputGM, cOffset_, HCCL_DATA_TYPE, AscendC::HCCL_REDUCE_SUM, 1U);
        }
    }

    __aicore__ inline void Process()
    {
        // InitOutput value to 0. Supports Dtype: half/int32_t/float/uint32_t/
        uint64_t cSize = cOffset_ * sizeof(yType);
        cSize /= sizeof(half);
        GlobalTensor<half> cGlobal;
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(addrs_->cGM), cSize);
        InitOutput<half>(cGlobal, cSize, static_cast<half>(0.0));
        if (biasFlag_ || param_->isAdd) {
            SyncAll();
            if (biasFlag_) {
                ProcessBias();
            }
            if (param_->isAdd) {
                ProcessAdd();
            }
        }

        SyncAll();
        if (notifyFlag_) {
            hccl_.Commit(hcclHandleId_);
            hccl_.Wait(hcclHandleId_);
            hccl_.Finalize();
        }
    }

private:
    __aicore__ inline void ProcessBias()
    {
        // Init buffer
        tPipe_->Reset();
        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        // DataCopy from bias to cGM 32B aligned
        // total cGM size
        uint64_t cSize = cOffset_ * sizeof(yType);
        GlobalTensor<yType> cGlobalBias;
        cGlobalBias.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(addrs_->cGM), cSize);
        // init ub for datacopy
        LocalTensor<yType> bias = tmpBuf.Get<yType>();
        GlobalTensor<yType> biasGlobal;
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(addrs_->biasGM));
        TBuffAddr buffAddr;
        buffAddr.logicPos = (uint8_t)QuePosition::VECCALC;
        // max calc num
        // when n is large, split N to make sure that ub used by each datacopy intrinsic does not exceed the size of ub
        int64_t ubAlignCntSize = UB_ALIGN_SIZE / sizeof(yType);
        int64_t biasUbAlignSize = (param_->rankN / ubAlignCntSize) * ubAlignCntSize;
        int64_t maxTmpBufCount =
            TOTAL_UB_SIZE / EMPTY_TENSOR_BIAS_UB_FACTOR / sizeof(yType) / ubAlignCntSize * ubAlignCntSize;
        int64_t biasSize = param_->rankN;
        int64_t biasTailSize = biasSize - biasUbAlignSize;
        for (uint64_t offsetRankN = 0; offsetRankN < biasUbAlignSize; offsetRankN += maxTmpBufCount) {
            uint64_t calCount =
                (biasUbAlignSize - offsetRankN) > maxTmpBufCount ? maxTmpBufCount : (biasUbAlignSize - offsetRankN);
            DataCopy(bias, biasGlobal[offsetRankN], calCount);
            SyncAll();
            // offset M*N
            for (uint64_t i = 0; i < param_->rankM; ++i) {
                uint64_t offsetRankM = i * param_->rankN + offsetRankN;
                DataCopy(cGlobalBias[offsetRankM], bias, calCount);
                SyncAll();
            }
        }
        // DataCopy from bias to cGM 32B aligned tail
        if (biasTailSize > 0) {
            DataCopy(bias, biasGlobal[biasUbAlignSize], ubAlignCntSize);
            SyncAll();
            for (uint64_t i = 0; i < param_->rankM; ++i) {
                uint64_t offsetDst = i * param_->rankN + biasUbAlignSize;
                DataCopyPad(cGlobalBias[offsetDst], bias,
                            {1, static_cast<uint16_t>(biasTailSize * sizeof(yType)), 0, 0});
            }
            SyncAll();
        }
    }

    __aicore__ inline void ProcessAdd()
    {
        const size_t cOffset = param_->rankN * sizeof(yType);
        for (uint64_t i = 0; i < param_->rankM; ++i) {
            Matmul_All_Reduce_Add_X3<yType>(addrs_->cGM, addrs_->addGM, cOffset / sizeof(yType), param_->addX3UbCnt,
                                            tPipe_);
            addrs_->cGM += cOffset;
            addrs_->addGM += cOffset;
            SyncAll();
        }
    }

    MC2GmAddrs *addrs_;
    Mc2Tiling::RCSTiling *param_;
    bool biasFlag_{false};
    uint64_t cOffset_;
    TPipe *tPipe_;
    bool notifyFlag_{false};
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    AscendC::HcclHandle hcclHandleId_;
};

#ifdef MC2_WEIGHT_QUANT
#define GET_TILING_DATA_FOR_EMPTY_TENSOR() \
    GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::WeightQuantMatmulAllReduceTilingData, tilingData, tilingGM)
#else
#define GET_TILING_DATA_FOR_EMPTY_TENSOR() \
    GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::MatmulAllReduce910TilingData, tilingData, tilingGM)
#endif

#define INVOKE_MC2_EMPTY_TENSOR_OP_IMPL() \
    do { \
        GET_TILING_DATA_FOR_EMPTY_TENSOR(); \
        MC2GmAddrs addrs = {nullptr, nullptr, biasGM, addGM, cGM, nullptr, cGM}; \
        MatmulAllReduceEmptyTensorKGeneral<DTYPE_Y> op(&addrs, (MC2TilingHeader *)&tilingData, &tPipe); \
        op.Init(); \
        op.Process(); \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_EMPTY_TENSOR_K_GENERAL_H
