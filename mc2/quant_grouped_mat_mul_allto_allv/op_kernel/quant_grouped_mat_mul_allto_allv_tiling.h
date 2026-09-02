/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file quant_grouped_mat_mul_allto_allv_tiling.h
 * \brief Quant Grouped MatMul AlltoAllV TilingData 定义
 */
#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H__

#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"

#pragma once

// 使用公共命名空间中的类型
using MC2KernelTemplate::GMMQuantTilingData;
using MC2KernelTemplate::TaskTilingInfo;

/**
 * GMM A2AV Workspace 信息 (后通信模式: GMM计算 → AlltoAllv发送)
 *
 * 算子级 workspace 分为三部分:
 *   [0, wsGmmOutputSize) → 路由专家 GMM 输出缓冲 (rank-first布局), 传给 GmmComputeOp.Init 的 y 参数
 *   [wsGmmOutputSize, + wsGmmComputeWorkspaceSize) → 路由专家 GmmComputeOp 内部空间, 传给 GmmComputeOp.Init 的 tempAddr
 * 参数
 *   [+, + wsSharedGmmComputeWorkspaceSize) → 共享专家 SharedGmmComputeOp 内部空间, 传给 SharedGmmComputeOp.Init 的
 * tempAddr 参数
 *
 * 注: 后通信模式 recvBuffer=yGM (AlltoAllv直接写入输出tensor), 无需额外commRecv workspace
 *
 * GmmComputeOp workspace 内部布局 (由各自内部管理, tiling 侧仅需计算并分配总大小):
 *   TT模式 (MX_QUANT_MODE=false):
 *     [0, ep * epWorldSize * 8)                groupList: (expert,rank) 粒度的token计数
 *     [+, + ep * epWorldSize * 8)              cGroupOffsetTable: rank-first输出偏移表
 *     [+, + 512)                               ptrTable:  4 × 16B GetTensorAddr 双重间接指针表 (x, weight, scaleB, y)
 *     [+, + sizeof(float)*ep)                  ttWeightScaleRepeat: PerTensor weight scale 重复拷贝
 *     [+, + sizeof(float)*ep)                  ttXScaleRepeat: PerTensor x scale 重复拷贝
 *   MX模式 (MX_QUANT_MODE=true):
 *     [0, ep * epWorldSize * 8)                groupList
 *     [+, + ep * epWorldSize * 8)              cGroupOffsetTable
 *     [+, + 512)                               ptrTable
 *   总大小需按最大模式(TT)计算 = ep * epWorldSize * 8 * 2 + 512 + sizeof(float) * ep * 2
 *   SharedGmmComputeOp workspace (共享专家, 单group, 无cGroupOffsetTable):
 *     [0, ep * 8)  groupList + [+, 512) ptrTable + [+, sizeof(float)*ep*2) TT scale repeat
 */
struct GmmA2avWorkspaceInfo {
    uint64_t wsGmmOutputSize; // 路由专家 GMM 主输出缓冲大小 (x @ weight -> gmm_output -> hccl send)
    uint64_t wsGmmComputeWorkspaceSize;       // 路由专家 GmmComputeOp 内部空间大小
    uint64_t wsSharedGmmComputeWorkspaceSize; // 共享专家 SharedGmmComputeOp 内部空间大小
};

struct QuantGmmA2avTilingData {
    // ============ HCCL AlltoAllV Tiling ============
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTiling;

    // ============ Workspace 配置信息 ============
    GmmA2avWorkspaceInfo workspaceInfo;

    // ============ 核心配置信息 (Task/Loop/Comm) ============
    TaskTilingInfo taskTilingInfo;

    // ============ 共享专家 GMM Tiling（放在前面）============
    GMMQuantTilingData sharedGmmTiling; // 共享专家 GMM Tiling 数据

    // ============ 普通专家 GMM Tiling ============
    GMMQuantTilingData gmmBaseTiling; // 共享专家 GMM Tiling 数据，后续还会在kernel中根据任务刷新

    // ============ isPermuteOut ============
    bool isPermuteOut = false;
};
#endif
