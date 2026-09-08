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
 * \file grouped_mat_mul_allto_allv_a3.cpp
 */
#include "arch22/grouped_mat_mul_allto_allv_mte_tiling_key.h"
#include "arch22/grouped_mat_mul_allto_allv_mte_tiling.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"
#include "arch22/grouped_mat_mul_allto_allv_mte.h"
#include "arch22/grouped_mat_mul_allto_allv_mte_catlass.h"

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

template <bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GMM_WEIGHT_TRANS, bool TILINGKEY_SHARED_MM_WEIGHT_TRANS>
__aicore__ inline void RunGmmA2avMtePath(GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR mmxOptionalGM,
                                         GM_ADDR mmweightOptionalGM, GM_ADDR yGM, GM_ADDR mmyOptionalGM,
                                         GM_ADDR userWorkspace, const GroupedMatMulAlltoAllvMteTilingData *tilingData,
                                         AscendC::TPipe *pipe)
{
    const auto *taskTilingInfo = &tilingData->taskTilingInfo;
    using CommOpType = GmmA2avMteOp<DTYPE_Y>;
    using ComputeOpType = CatlassGroupedMatmulOp<DTYPE_GMM_X, TILINGKEY_GMM_WEIGHT_TRANS, false>;
    using SharedGmmExpertOpType = CatlassGroupedMatmulOp<DTYPE_GMM_X, TILINGKEY_SHARED_MM_WEIGHT_TRANS, true>;
    using GmmA2avSchedulerType =
        GmmA2avMteScheduler<CommOpType, ComputeOpType, SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;

    // The documented memory semantics keep the grouped-matmul result in
    // workspace. AIV then publishes it into this rank's HCCL window through
    // MTE3 before peers are allowed to read it.
    GM_ADDR gmmOutputGM = userWorkspace;
    GM_ADDR cumsumGM = userWorkspace + tilingData->cumsumWorkspaceOffset;

    if ASCEND_IS_AIC {
        ComputeOpType computeOp;
        computeOp.Init(gmmxGM, gmmweightGM, gmmOutputGM, taskTilingInfo);

        SharedGmmExpertOpType shareComputeOp;
        shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmyOptionalGM, taskTilingInfo);

        GmmA2avSchedulerType::ProcessAic(computeOp, shareComputeOp, taskTilingInfo, tilingData->expertChunkRows);
    }

    if ASCEND_IS_AIV {
        CommOpType commOp;
        commOp.Init(taskTilingInfo, gmmOutputGM, yGM, cumsumGM, pipe, &tilingData->cocTiling,
                    tilingData->commBufferSize, tilingData->isA3);

        GmmA2avSchedulerType::ProcessAiv(commOp, taskTilingInfo, tilingData->expertChunkRows);
    }
}

template <bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GMM_WEIGHT_TRANS, bool TILINGKEY_SHARED_MM_WEIGHT_TRANS,
          int TILINGKEY_COMM_MODE>
__global__ __aicore__ void grouped_mat_mul_allto_allv(GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                      GM_ADDR sendCountsTensorOptionalGM,
                                                      GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
                                                      GM_ADDR mmweightOptionalGM, GM_ADDR yGM, GM_ADDR mmyOptionalGM,
                                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspaceGM == nullptr) {
        return;
    }
    SetSysWorkspace(workspaceGM);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }

    // The legacy layout remains the default for backward-compatible keys;
    // grouped_mat_mul_allto_allv_mte_tiling_key.h selects the independent MTE
    // path structure for AIV communication-mode keys.
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
    if constexpr (TILINGKEY_COMM_MODE == TILINGKEY_TPL_AIV) {
        GET_TILING_DATA_WITH_STRUCT(GroupedMatMulAlltoAllvMteTilingData, tilingData, tilingGM);
        if ASCEND_IS_AIV {
            TPipe pipe;
            RunGmmA2avMtePath<TILINGKEY_COMPUTE_MATMUL, TILINGKEY_GMM_WEIGHT_TRANS, TILINGKEY_SHARED_MM_WEIGHT_TRANS>(
                gmmxGM, gmmweightGM, mmxOptionalGM, mmweightOptionalGM, yGM, mmyOptionalGM, userWorkspace, &tilingData,
                &pipe);
        }
        if ASCEND_IS_AIC {
            RunGmmA2avMtePath<TILINGKEY_COMPUTE_MATMUL, TILINGKEY_GMM_WEIGHT_TRANS, TILINGKEY_SHARED_MM_WEIGHT_TRANS>(
                gmmxGM, gmmweightGM, mmxOptionalGM, mmweightOptionalGM, yGM, mmyOptionalGM, userWorkspace, &tilingData,
                nullptr);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(QuantGmmA2avTilingData, tilingData, tilingGM);
        QuantGmmA2avTilingData *tilingData_ = &tilingData;

        using HcclOpType = HcclA2avOp<DTYPE_Y, false, TILINGKEY_COMM_MODE>;
        using ComputeOpType =
            QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X,
                               DTYPE_Y, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANS, false, false>;
        using SharedGmmExpertOpType =
            QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X,
                               DTYPE_Y, CubeFormat::ND, false, TILINGKEY_SHARED_MM_WEIGHT_TRANS, true, false>;
        using GmmA2avSchedulerType =
            GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;

        GM_ADDR gmmOutputGM = userWorkspace;

        if ASCEND_IS_AIC {
            uint64_t wsOffset = tilingData_->workspaceInfo.wsGmmOutputSize;
            GM_ADDR gmmComputeWSGM = userWorkspace + wsOffset;
            wsOffset += tilingData_->workspaceInfo.wsGmmComputeWorkspaceSize;
            GM_ADDR sharedGmmComputeWSGM = userWorkspace + wsOffset;

            TPipe pipe;
            GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                               gmmArrayAddr_, tilingGM);
            ComputeOpType computeOp;
            computeOp.Init(gmmxGM, gmmweightGM, nullptr, nullptr, gmmOutputGM, gmmComputeWSGM, tilingData_,
                           &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);

            GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                               mmArrayAddr_, tilingGM);
            SharedGmmExpertOpType shareComputeOp;
            shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, mmyOptionalGM,
                                sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);

            GmmA2avSchedulerType::ProcessAic(computeOp, shareComputeOp, &tilingData_->taskTilingInfo);
        }

        if ASCEND_IS_AIV {
            const void *hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
            uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                                          offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
            HcclOpType hcclOp;
            hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData_->taskTilingInfo, gmmOutputGM, yGM,
                        static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));

            GmmA2avSchedulerType::ProcessAiv(hcclOp, &tilingData_->taskTilingInfo);
        }
    }
}
