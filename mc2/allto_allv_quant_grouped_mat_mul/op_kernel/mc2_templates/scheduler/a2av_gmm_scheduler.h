/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file pipeline_template_comm_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "../a2av_gmm_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../communication/a2av_permute_engine.h"

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType, typename TilingDataType,
          typename GmmTilingDataType, typename GmmArrayAddrType>
class A2avGmmScheduler {
public:
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM,
                                GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
                                GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                GmmArrayAddrType *gmmArrayAddrIn, GmmArrayAddrType *mmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag)
    {
        // init member variables
        GET_TILING_DATA(tilingData, tilingGM);
        tilingData_ = &tilingData;
        e_ = tilingData_->taskTilingInfo.e;
        expertNum_ = tilingData_->taskTilingInfo.expertNum;
        gmmxScaleGm_ = gmmxScaleGM;
        // workspace偏移
        uint64_t workspaceOffset = 0;
        // x通信输出区域：isPermuteOut=true时通信结果写到workspace，AIV重排到permuteOut
        uint64_t commOutLen =
            Align(CeilDiv((tilingData_->taskTilingInfo.A) * (tilingData_->taskTilingInfo.H1),
                          PACK_FACTOR) * X_TYPE_SIZE,
                  TENSOR_LIST_SIZE);
        commOutGm_ = workspaceGM + workspaceOffset;
        workspaceOffset += commOutLen;
        permuteOutGm_ = permuteOutOptionalGM;
        computeScaleGm_ = gmmxScaleGm_;
        // gmmX comm init
        const void *hcclInitTiling = &(tilingData_->hcclA2avTilingInfo.hcclInitTiling);
        uint64_t hcclCcTilingOffset = offsetof(TilingDataType, hcclA2avTilingInfo) +
                                      offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
        commOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmxGM, commOutGm_, commOutGm_,
            static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));
        // scale commOut区域：MX场景scale通信输出
        if constexpr (MX_QUANT_MODE) {
            uint64_t scaleAxis = CeilDiv(tilingData_->taskTilingInfo.H1, SCALE_ALIGNMENT_BLOCK_SIZE) * 2;
            uint64_t permuteScaleOutSize = Align(tilingData_->taskTilingInfo.A * scaleAxis, TENSOR_LIST_SIZE);
            gmmxScaleCommOutGm_ = workspaceGM + workspaceOffset;
            workspaceOffset += permuteScaleOutSize;
            gmmxScalePermuteOutGm_ = gmmxScaleCommOutGm_;
            commOp.InitScaleBuffer(gmmxScaleGm_, gmmxScaleCommOutGm_, gmmxScalePermuteOutGm_);
            computeScaleGm_ = gmmxScalePermuteOutGm_;
        }
        computeWorkspaceGm_ = workspaceGM + workspaceOffset;
        isPermuteOut_ = tilingData_->isPermuteOut;
        if (tilingData_->isNeedMM != 0) {
            localComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                                computeWorkspaceGm_, tilingData_, &tilingData_->mmQuantTilingData, mmArrayAddrIn, tPipe,
                                isA2avGmmFlag);
        }
        computeOp.Init(commOutGm_, gmmweightGM, computeScaleGm_, gmmWeightScaleGM, gmmyGM, computeWorkspaceGm_,
                       tilingData_, &tilingData_->gmmQuantTilingData, gmmArrayAddrIn, tPipe, isA2avGmmFlag);
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->isNeedMM != 0) {
            localComputeOp.Process(0);
        }
        ProcessCommRoundMode();
        this->End();
    }

protected:
    __aicore__ inline void End()
    {
        commOp.End();
        computeOp.End();
        localComputeOp.End();
    }

private:
    __aicore__ inline void ProcessCommRoundMode()
    {
        for (uint32_t start = 0; start < e_; start += expertNum_) {
            uint32_t actualExpertNum = (start + expertNum_ > e_) ? (e_ - start) : expertNum_;
            if constexpr (MX_QUANT_MODE) {
                commOp.LaunchScaleBeforeCompute(start, actualExpertNum);
            }
            commOp.Launch(start, actualExpertNum);
        }
        for (uint32_t start = 0; start < e_; start += expertNum_) {
            uint32_t actualExpertNum = (start + expertNum_ > e_) ? (e_ - start) : expertNum_;
            if ASCEND_IS_AIV {
                if (GetBlockIdx() == 0) {
                    if constexpr (MX_QUANT_MODE) {
                        commOp.WaitScale(start);
                    }
                    commOp.Wait(start);
                }
                SyncAll<true>();
                AscendC::CrossCoreSetFlag<SYNC_MODE_AIV_TO_AIC, PIPE_V>(
                    SYNC_FLAG_ID_COMM_DONE + SYNC_FLAG_AIV1_OFFSET * GetSubBlockIdx());
                if (isPermuteOut_) {
                    commOp.PermuteData(start, actualExpertNum, permuteOutGm_);
                }
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreWaitFlag<SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(SYNC_FLAG_ID_COMM_DONE);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
                AscendC::CrossCoreWaitFlag<SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
                    SYNC_FLAG_ID_COMM_DONE + SYNC_FLAG_AIV1_OFFSET);
#endif
                computeOp.Process(start, actualExpertNum);
            }
        }
    }

    CommOpType commOp;
    ComputeOpType computeOp;
    LocalComputeOpType localComputeOp;
    GM_ADDR commOutGm_ = nullptr;
    GM_ADDR permuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleGm_ = nullptr;
    GM_ADDR gmmxScalePermuteOutGm_ = nullptr;
    GM_ADDR gmmxScaleCommOutGm_ = nullptr;
    GM_ADDR computeScaleGm_ = nullptr;
    GM_ADDR computeWorkspaceGm_ = nullptr;
    const TilingDataType *tilingData_ = nullptr;
    uint32_t e_ = 0U;
    uint32_t expertNum_ = 1U;
    bool isPermuteOut_ = false;
};
}; // namespace MC2KernelTemplate
#endif
