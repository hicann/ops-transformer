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
 * \file a2av_common_tiling.h
 * \brief
 */

#ifndef A2AV_COMMON_H
#define A2AV_COMMON_H

#include "../../../../3rd/grouped_matmul/op_kernel/grouped_matmul_tiling_data_apt.h"

namespace MC2KernelTemplate {
static constexpr uint32_t MAX_EP_RANK_SIZE = 128U;
static constexpr uint32_t MAX_EXPERT_PER_EP = 1U;
static constexpr uint32_t MAX_EXPERT_SIZE = 256U;
static constexpr uint32_t MAX_EXPERT_PER_RANK = 32U;
static constexpr uint32_t TENSOR_LIST_SIZE = 512U;
static constexpr uint32_t SCALE_COMM_BATCH_THRESHOLD = 32U;
static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
static constexpr uint64_t SCALE_ALIGNMENT_BLOCK_SIZE = 64U;
static constexpr uint16_t MAX_BUFFER_SIZE = 65534U;
static constexpr uint16_t PERMUTE_BUF_SIZE = 32768U;

// 核间同步常量
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
static constexpr uint8_t SYNC_MODE_AIV_TO_AIC = 4U;
static constexpr uint16_t SYNC_FLAG_AIV1_OFFSET = 16U;
#else
static constexpr uint8_t SYNC_MODE_AIV_TO_AIC = 2U;
static constexpr uint16_t SYNC_FLAG_AIV1_OFFSET = 0U;
#endif
static constexpr uint8_t SYNC_MODE_AIC_BARRIER = 0U;
static constexpr uint8_t SYNC_MODE_AIC_TO_AIV = 2U;
static constexpr uint16_t SYNC_FLAG_ID_COMM_DONE = 8U;
static constexpr uint16_t SYNC_FLAG_ID_AIC_BARRIER = 9U;
static constexpr uint16_t SYNC_FLAG_ID_PERMUTE_DONE = 10U;

// 类型复用声明
using GMMQuantTilingData = Mc2GroupedMatmulTilingData::GMMQuantTilingData;
using GMMArray = Mc2GroupedMatmulTilingData::GMMArray;

struct HcclA2avTilingInfo {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling a2avCcTiling;
};

struct TaskTilingInfo {
    // Tensor维度参数（对应aclnn接口的输入输出shape）
    uint64_t BSK; // 参考各个算子aclnn接口
    uint64_t BS;  // mmXOptional的第一维: (BS, H)，batch*seq
    uint64_t H1;  // gmmX和gmmWeight的隐藏层维度H
    uint64_t H2;  // mmWeightOptional的隐藏层维度H
    uint64_t A;   // 参考各个算子aclnn接口
    uint64_t N1;  // gmmWeight/gmmY的输出维度N1
    uint64_t N2;  // mmWeightOptional/mmYOptional的输出维度N2

    // Expert并行参数
    uint64_t epWorldSize; // expert parallel world size (EP并行域大小)
    uint64_t e;           // 单卡上的专家数量

    // 平台信息
    uint64_t ubSize;     // UB大小
    uint64_t aivCoreNum; // AIV 核数量
    uint64_t aicCoreNum; // AIC 核数量

    // 循环调度参数
    uint32_t expertNum = 1U;    // 每次GMM计算合并的专家数量（计算批大小）
    uint32_t mainLoopExpertNum; // 主循环每次处理的expert数量（=expertNum）
    uint32_t tailLoopExpertNum; // 尾循环处理的expert数量（=e%expertNum的余数）
    uint32_t totalLoopCount;    // 总循环次数（=CeilDiv(e,expertNum))

    // 通信参数（对应sendCounts和recvCounts）
    int32_t sendCnt[MAX_EXPERT_SIZE]; // 每个expert的发送计数
    int32_t recvCnt[MAX_EXPERT_SIZE]; // 每个expert的接收计数
};

} // namespace MC2KernelTemplate
#endif // A2AV_COMMON_H
