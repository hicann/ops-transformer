/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_grouped_mat_mul_allto_allv.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "quant_grouped_mat_mul_allto_allv_tiling.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_key.h"

#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"

// apace hcomm 融合通路（结构对齐镜像算子 a2avgmm apace 实现的最新代码）：
// MX tiling 类按 CheckApaceImplCapable 动态编码 TILINGKEY_USING_APACE_IMPL，kernel
// 经 GET_TILING_DATA_WITH_STRUCT 直接解析算子 legacy tiling 数据（无独立紧凑结构），
// GMM 计算复用 apace GroupedMatmulKernel（每批栈上局部构造），AllToAllV 通信在
// AIC 核内经 Hccl 提交（Blaze 与 AIV 侧 KFC 通信引擎同 kernel 共存会致 AIV 通信
// 任务参数校验失败）。该组合经 tiling key 的 ASCENDC_TPL_KERNEL_TYPE_SEL 声明
// AIC_ONLY 任务类型；TT 量化实例恒编码 USING=0 走原通路（MIX_AIC_1_2）。
#include "../../common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_hcomm_impl.h"
#include "../../common/op_kernel/apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_scheduler.h"

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

#if (((ORIG_DTYPE_GMM_X == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_GMM_X == DT_FLOAT8_E5M2)) && \
     ((ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT8_E5M2))) || \
    ((ORIG_DTYPE_GMM_X == DT_FLOAT4_E2M1) && (ORIG_DTYPE_GMM_WEIGHT == DT_FLOAT4_E2M1))
#define GMM_A2AV_APACE_DTYPE_SUPPORTED 1
#else
#define GMM_A2AV_APACE_DTYPE_SUPPORTED 0
#endif

template <bool TILINGKEY_COMPUTE_MATMUL, bool TILINGKEY_GROUPED_MATMUL_TRANS, bool TILINGKEY_MATMUL_TRANS,
          int TILINGKEY_COMM_MODE, bool TILINGKEY_USING_APACE_IMPL>
__global__ __aicore__ void quant_grouped_mat_mul_allto_allv(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM,
    GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
    GM_ADDR mmweightOptionalGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR commQuantScaleGM, GM_ADDR yGM,
    GM_ADDR mmyOptionalGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    // A6(ascend960) AIC:AIV为1:1
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    if (workspaceGM == nullptr) {
        return;
    }
    SetSysWorkspace(workspaceGM);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }

    TPipe pipe;
    REGISTER_TILING_DEFAULT(QuantGmmA2avTilingData);
    if constexpr (TILINGKEY_USING_APACE_IMPL) {
#if MX_QUANT_MODE && GMM_A2AV_APACE_DTYPE_SUPPORTED
        GET_TILING_DATA(tilingData, tilingGM);
        constexpr HcclServerType serverType = TILINGKEY_COMM_MODE == TILINGKEY_TPL_CCU ?
                                                  HcclServerType::HCCL_SERVER_TYPE_CCU :
                                                  HcclServerType::HCCL_SERVER_TYPE_AICPU;
        using ApaceComputeOpType = Apace::QuantGmmComputeOp<QuantGmmA2avTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                                            DTYPE_Y, TILINGKEY_GROUPED_MATMUL_TRANS>;
        using ApaceSharedOpType = Apace::QuantMmSharedComputeOp<QuantGmmA2avTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT,
                                                                DTYPE_Y, TILINGKEY_MATMUL_TRANS>;
        using ApaceCommOpType = Apace::QuantA2avCommOp<QuantGmmA2avTilingData, DTYPE_Y, serverType>;
        using ApaceSchedulerType = Apace::GmmA2avHcommScheduler<ApaceCommOpType, ApaceComputeOpType, ApaceSharedOpType,
                                                                TILINGKEY_COMPUTE_MATMUL>;
        // workspace（apace 通路）：[0, wsGmmOutputSize) 路由专家 GMM 输出（sendBuf），
        // 其后表区复用 gmmComputeWS 区域（groupList + 三张偏移表 + mmGroupList，
        // 见 scheduler 头布局注释）
        GM_ADDR gmmOutputGM = userWorkspace;
        GM_ADDR gmmTableGM = userWorkspace + tilingData.workspaceInfo.wsGmmOutputSize;
        ApaceComputeOpType computeOp;
        computeOp.Init(&tilingData, gmmxGM, gmmxScaleGM, gmmweightGM, gmmWeightScaleGM, gmmOutputGM, gmmTableGM);
        ApaceSharedOpType sharedOp;
        sharedOp.Init(&tilingData, mmxOptionalGM, mmxScaleGM, mmweightOptionalGM, mmWeightScaleGM, mmyOptionalGM,
                      gmmTableGM);
        ApaceCommOpType commOp;
        commOp.Init(&tilingData, gmmOutputGM, yGM);
        ApaceSchedulerType scheduler(commOp, computeOp, sharedOp);
        scheduler.Process();
#endif
        return;
    }

    // 原通路（TT 量化实例走此路径；MIX_AIC_1_2 任务类型由 KERNEL_TYPE_SEL 声明，
    // AIC 计算 + AIV 通信，历史行为不变）
    GET_TILING_DATA(tilingData, tilingGM);
    const QuantGmmA2avTilingData* tilingData_ = &tilingData;
    const void* hcclInitTiling = &(tilingData_->hcclA2avTiling.hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(QuantGmmA2avTilingData, hcclA2avTiling) +
                                  offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling);
    constexpr CubeFormat W_FORMAT = CubeFormat::ND;
    constexpr bool USE_SEND_COUNTS = true;
    constexpr bool IS_SHARED_EXPERT = true;
    constexpr bool IS_NOT_SHARED_EXPERT = false;

    // workspace布局 (后通信模式):
    //   [0, wsGmmOutputSize)                 → 路由专家 GMM 输出 (每轮从偏移0开始写入)
    //   [+, wsGmmComputeWorkspaceSize)       → GmmComputeOp 内部空间
    //   [+, wsSharedGmmComputeWorkspaceSize) → SharedGmmComputeOp 内部空间
    uint64_t wsOffset = 0;
    GM_ADDR gmmOutputGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsGmmOutputSize;
    GM_ADDR gmmComputeWSGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsGmmComputeWorkspaceSize;
    GM_ADDR sharedGmmComputeWSGM = userWorkspace + wsOffset;
    wsOffset += tilingData_->workspaceInfo.wsSharedGmmComputeWorkspaceSize;

    using HcclOpType = HcclA2avOp<DTYPE_Y, false, TILINGKEY_COMM_MODE>;
    using GmmASWKernelType =
        Mc2GroupedMatmul::Mc2GmmASWKernel<DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float, DTYPE_GMM_X_SCALE, DTYPE_Y, W_FORMAT,
                                          TILINGKEY_GROUPED_MATMUL_TRANS, TILINGKEY_MATMUL_TRANS>;
    using ComputeOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_GMM_X_SCALE,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_GROUPED_MATMUL_TRANS, false, false>;
    using SharedGmmExpertOpType =
        QuantGroupedMatmul<QuantGmmA2avTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, DTYPE_MM_X_SCALE,
                           DTYPE_Y, CubeFormat::ND, false, TILINGKEY_MATMUL_TRANS, true, false>;
    using GmmA2avSchedulerType =
        GmmA2avScheduler<HcclOpType, ComputeOpType, SharedGmmExpertOpType, TILINGKEY_COMPUTE_MATMUL>;
    // hccl: sendBuffer=GMM输出缓冲 (rank-first布局, 无需permute重排)
    GM_ADDR sendBufferAddr = gmmOutputGM;
    HcclOpType hcclOp;
    hcclOp.Init(hcclInitTiling, hcclCcTilingOffset, &tilingData.taskTilingInfo, sendBufferAddr, yGM,
                static_cast<uint32_t>(tilingData_->taskTilingInfo.aivCoreNum));
    // gmm: yGM=GMM输出缓冲(通信发送源), workspaceGM=groupList+ptrTable区域
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, gmmBaseTiling, gmmArray,
                                       gmmArrayAddr_, tilingGM);
    ComputeOpType computeOp;
    computeOp.Init(gmmxGM, gmmweightGM, gmmxScaleGM, gmmWeightScaleGM, gmmOutputGM, gmmComputeWSGM, tilingData_,
                   &tilingData_->gmmBaseTiling, gmmArrayAddr_, &pipe);
    // sharedmm
    GET_NESTED_TILING_DATA_MEMBER_ADDR(QuantGmmA2avTilingData, GMMQuantTilingData, sharedGmmTiling, gmmArray,
                                       mmArrayAddr_, tilingGM);
    SharedGmmExpertOpType shareComputeOp;
    shareComputeOp.Init(mmxOptionalGM, mmweightOptionalGM, mmxScaleGM, mmWeightScaleGM, mmyOptionalGM,
                        sharedGmmComputeWSGM, tilingData_, &tilingData_->sharedGmmTiling, mmArrayAddr_, &pipe);
    GmmA2avSchedulerType gmmA2avScheduler(hcclOp, computeOp, shareComputeOp, &tilingData.taskTilingInfo);
    gmmA2avScheduler.Process();
}
