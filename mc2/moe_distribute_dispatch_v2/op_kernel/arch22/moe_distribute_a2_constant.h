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
 * \file moe_distribute_a2_constant.h
 * \brief Define the constant for the A2 Dispatch and Combine operations.
 */
#ifndef MOE_DISTRIBUTE_A2_CONSTANT_H
#define MOE_DISTRIBUTE_A2_CONSTANT_H
namespace Mc2A2Kernel {
// ADump所需常量段
constexpr uint32_t BUFFERID_POS = 0U;
constexpr uint32_t OPOSITION_POS = 1U;
constexpr uint32_t TILING_EPRANKID_POS = 2U;
constexpr uint32_t MOE_NUM_POS = 3U;
constexpr uint32_t TILING_WORLDSIZE_POS = 4U;
constexpr uint32_t GLOBALBS_POS = 5U;
constexpr uint64_t OP_CNT_POSUL = 3UL;
constexpr uint32_t HCCL_DFX_POS = 8U;
constexpr uint32_t HCCL_DFX_NUM = 2U;
constexpr uint32_t HCCL_EPRANKId_POS = 0U;
constexpr uint32_t HCCL_WORLDSIZE_POS = 1U;
constexpr uint32_t BS_POS = 10U;
constexpr uint32_t ISLAYERED_POS = 11U;
constexpr uint32_t AIVNUM_POS = 12U;
constexpr uint32_t H_POS = 13U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t BITS32_PER_BLOCK = UB_ALIGN / sizeof(int32_t); // 8

constexpr uint32_t RUNPOS_INIT = 1U;
constexpr uint32_t RUNPOS_END = 2U;
// A2 Fullmesh
constexpr uint32_t FULLMESH_SEND_FIRST_EPRANKID_POS = 14U;
constexpr uint32_t FULLMESH_RECV_FIRST_EPRANKID_POS = 16U;
constexpr uint32_t FULLMESH_EPRANKID_POS = 18U;

// Dispatch
constexpr uint32_t RUNPOS_FULLMESH_DISPATCH_REORDERTOKEN = 3U;
constexpr uint32_t RUNPOS_FULLMESH_DISPATCH_SEND = 4U;
constexpr uint32_t RUNPOS_FULLMESH_DISPATCH_RECV = 5U;
constexpr uint32_t RUNPOS_FULLMESH_DISPATCH_GET_STATUS_CUM_SUM = 6U;
constexpr uint32_t RUNPOS_FULLMESH_DISPATCH_LOCAL_WINDOW_COPY = 7U;

// Combine
constexpr uint32_t RUNPOS_FULLMESH_COMBINE_SEND = 3U;
constexpr uint32_t RUNPOS_FULLMESH_COMBINE_RECV = 4U;

// A2 Hierarchy
// Dispatch
constexpr uint32_t RUNPOS_HIERARCHY_DISPATCH_REORDERTOKEN = 3U;
constexpr uint32_t RUNPOS_HIERARCHY_DISPATCH_INTER_SEND = 4U;
constexpr uint32_t RUNPOS_HIERARCHY_DISPATCH_INTER_FINISH = 5U;
constexpr uint32_t RUNPOS_HIERARCHY_DISPATCH_INTRA_FINISH = 6U;
constexpr uint32_t RUNPOS_HIERARCHY_DISPATCH_IPC2OUT = 7U;

// Combine
constexpr uint32_t RUNPOS_HIERARCHY_COMBINE_GM2IPC = 3U;
constexpr uint32_t RUNPOS_HIERARCHY_COMBINE_WAITIPC = 4U;
constexpr uint32_t RUNPOS_HIERARCHY_COMBINE_INTER_SEND = 5U;
constexpr uint32_t RUNPOS_HIERARCHY_COMBINE_INTER_RECV = 6U;

constexpr uint32_t HIERARCHY_INTER_NODE_SEND_POS = 14U;
constexpr uint32_t HIERARCHY_INTER_NODE_RECV_POS = 16U;
constexpr uint32_t HIERARCHY_INNER_SEND_POS = 18U;
constexpr uint32_t HIERARCHY_INNER_RECV_POS = 19U;
constexpr bool IS_SEND = true;
constexpr bool IS_RECV = false;
constexpr uint32_t IS_FULLMESH = 0;
constexpr uint32_t IS_HIERARCHY = 1;
constexpr static uint64_t ADUMP_DATA_SIZE = 50 * 1024UL;
constexpr static uint64_t ADUMP_DATA_SIZE_PER_AIV = 512UL;
constexpr static uint64_t ADUMP_DATA_ADDR_START = 768 * 1024UL;
constexpr static uint64_t ADUMP_STATUS_DISPATCH_ADDR = 0UL;
constexpr static uint64_t ADUMP_STATUS_COMBINE_ADDR = 64 * 1024UL;
constexpr static uint64_t ADUMP_DATA_DISPATCH_ADDR_START = ADUMP_DATA_ADDR_START;
constexpr static uint64_t ADUMP_DATA_COMBINE_ADDR_START = ADUMP_DATA_ADDR_START + ADUMP_DATA_SIZE;
constexpr static uint64_t BATCH_WRITE_ITEM_OFFSET = ADUMP_DATA_ADDR_START + ADUMP_DATA_SIZE * 2UL;
constexpr static uint64_t ADUMP_WIN_SIZE = 1024 * 1024UL;
constexpr static uint64_t FULLMESH_BUFFERID_ADDR = ADUMP_WIN_SIZE - WIN_ADDR_ALIGN;
constexpr static int32_t DUMP_FLAG_VALUE = 0xFFFFFFFF;
constexpr static int64_t TIMEOUT = 1000000 * 60 * 1; // 1 mins
} // namespace Mc2A2Kernel

#endif
