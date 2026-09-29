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
 * \file moe_distribute_dispatch_setup_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_SETUP_BASE_H
#define MOE_DISTRIBUTE_DISPATCH_SETUP_BASE_H

#include "../../common/op_kernel/mc2_kernel_utils.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

constexpr uint32_t MAX_RANK_NUM = 64U; // 最大卡数
constexpr uint32_t MAX_OP_NUM = 8U;    // MC2最大通信算子数
constexpr uint32_t WRITE_SQE_SIZE = 64U;
constexpr uint32_t WRITE_WITH_NOTIFY_SQE_SIZE = 96U;
// EP window 前 1MB(STATE_SIZE) meta 布局（arch35 D/C 隔离，支持单跑与联跑多轮）:
//   [0, 256KB)      D status bank0 (D_S=0)
//   [256KB, 512KB)  D status bank1 (D_S=1)
//   [512KB, 640KB)  C status bank0 (C_S=0)
//   [640KB, 768KB)  C status bank1 (C_S=1)
//   [768KB, 950KB)  D staging（本卡打包 status，再 WriteNbi）
//   [950KB, 1MB)    per-aiv 512B 控制：D_S@0, C_S@32, C flag@64
// token 数据在 1MB 之后（totalWinSize），D/C 固定半区 + 区内 totalWin/4 乒乓:
//   D: [0, totalWin/2)           offset = D_S * (totalWin/4)
//   C: [totalWin/2, totalWin)    offset = totalWin/2 + C_S * (totalWin/4)
constexpr uint64_t DISPATCH_STATE_OFFSET = 256U * 1024U;
constexpr uint64_t WIN_STATE_OFFSET = DISPATCH_STATE_OFFSET; // dispatch 状态 bank 间距
constexpr uint64_t COMBINE_STATUS_BASE = 512U * 1024U;
constexpr uint64_t COMBINE_STATE_BANK_STRIDE = 128U * 1024U;
constexpr uint64_t DISPATCH_STAGING_OFFSET = 768U * 1024U;
constexpr uint64_t STATE_WIN_OFFSET = 950U * 1024U;
constexpr uint64_t CTRL_DISPATCH_STATE_OFFSET = 0U;
constexpr uint64_t CTRL_COMBINE_STATE_OFFSET = 32U;
constexpr uint64_t CTRL_COMBINE_FLAG_OFFSET = 64U;
constexpr uint64_t WIN_PICI_OFFSET = 1024U * 1024U;
constexpr uint64_t PICI_WIN_SIZE = 512UL;
constexpr uint32_t NORMAL_CQE_SIZE = 64U;
constexpr uint32_t CQ_DEPTH_256 =
    256U; // 为cqeBuf申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
constexpr uint32_t UB_ALIGN = 32U; // UB按32字节对齐
constexpr uint32_t WIN_SQPI_OFFSET = 0U;
constexpr uint32_t WIN_SQCI_OFFSET = 1U;
constexpr uint32_t WIN_CQPI_OFFSET = 2U;
constexpr uint32_t WIN_CQCI_OFFSET = 3U;
constexpr uint32_t WIN_SQPILINEAR_OFFSET = 4U;
constexpr uint32_t WIN_CQCILINEAR_OFFSET = 5U;
constexpr uint32_t WIN_FIRST_TIME_CREATE_WIN_FLAG_OFFSET = 6U;
constexpr uint32_t UINT8_BITS_OFFSET = 8U;

// WQ 32bit offset
constexpr uint32_t WQ_JETTYID_OFFSET = 0U;
constexpr uint32_t WQ_WQESIZE_OFFSET = 4U;
constexpr uint32_t WQ_SQDEPTH_OFFSET = 5U;
constexpr uint32_t WQ_TP_ID_OFFSET = 12U;
constexpr uint32_t WQ_RMTEID_0_3_OFFSET = 13U;
constexpr uint32_t WQ_RMTEID_4_7_OFFSET = 14U;
constexpr uint32_t WQ_RMTEID_8_11_OFFSET = 15U;
constexpr uint32_t WQ_RMTEID_12_15_OFFSET = 16U;
constexpr uint32_t WQ_RMTOBJID_OFFSET = 17U;
constexpr uint32_t WQ_RMTTOKENVALUE_OFFSET = 18U;
constexpr uint32_t WQ_LOCALTOKENID_OFFSET = 19U;
// WQ 64bit offset
constexpr uint32_t WQ_SQVA_OFFSET = 1U;
constexpr uint32_t WQ_DBADDR_OFFSET = 5U;

// CQ 32bit offset
constexpr uint32_t CQ_JFCID_OFFSET = 0U;
constexpr uint32_t CQ_CQESIZE_OFFSET = 4U;
constexpr uint32_t CQ_CQDEPTH_OFFSET = 5U;
// CQ 64bit offset
constexpr uint32_t CQ_CQVA_OFFSET = 1U;
constexpr uint32_t CQ_DBADDR_OFFSET = 5U;

// URMA protocol 8bit offset
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_2 = 2U; // udf_flg:1 inline_en:1 cqe:1 se:1 fence:1 odr:3
constexpr uint32_t SQE_COMMON_UINT8_OFFSET_3 = 3U; // owner:1 rmt_jetty_type:2 token_en:1 nf:1 rsv:3
constexpr uint32_t SQE_COMMON_TARGET_HINT_OFFSET = 4U;
constexpr uint32_t SQE_COMMON_OPCODE_OFFSET = 5U;
constexpr uint32_t SQE_COMMON_SGE_NUM_OFFSET = 11U;
constexpr uint32_t CQE_STATUS_OFFSET = 3U;
constexpr uint32_t CQE_SUBSTATUS_OFFSET = 2U;
constexpr uint32_t CQE_ENTRY_IDX_HIGH_OFFSET = 5U;
constexpr uint32_t CQE_ENTRY_IDX_LOW_OFFSET = 4U;

// URMA protocol 32bit offset
constexpr uint32_t SQE_COMMON_UINT32_OFFSET_2 = 2U; // sge_num:8 tp_id:24
constexpr uint32_t SQE_COMMON_RMT_JETTY_OR_SEG_ID_OFFSET = 3U;
constexpr uint32_t SQE_COMMON_RMT_EID_31_0_OFFSET = 4U;
constexpr uint32_t SQE_COMMON_RMT_EID_63_32_OFFSET = 5U;
constexpr uint32_t SQE_COMMON_RMT_EID_95_64_OFFSET = 6U;
constexpr uint32_t SQE_COMMON_RMT_EID_127_96_OFFSET = 7U;
constexpr uint32_t SQE_COMMON_RMT_TOKEN_VALUE_OFFSET = 8U;
constexpr uint32_t SQE_UDF_OFFSET = 9U;
constexpr uint32_t SQE_WITH_NOTIFY_UDF_OFFSET = SQE_UDF_OFFSET;
constexpr uint32_t SQE_LENGTH_OFFSET = 12U;
constexpr uint32_t SQE_WITH_NOTIFY_LENGTH_OFFSET = 20U;
constexpr uint32_t SQE_TOKEN_ID_OFFSET = 13U;                    // rsv:12 token_id:20
constexpr uint32_t SQE_WITH_NOTIFY_TOKEN_ID_OFFSET = 21U;        // rsv:12 token_id:20
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_OFFSET = 12U; // rsv:12 notify_token_id:20
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_TOKEN_VALUE_OFFSET = 13U;

// URMA protocol 64bit offset
constexpr uint32_t SQE_RMT_ADDR_OFFSET = 5U;
constexpr uint32_t SQE_WITH_NOTIFY_RMT_ADDR_OFFSET = SQE_RMT_ADDR_OFFSET;
constexpr uint32_t SQE_DATA_ADDR_OFFSET = 7U;
constexpr uint32_t SQE_WITH_NOTIFY_DATA_ADDR_OFFSET = 11U;
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET = 7U;
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET = 8U;

struct HcclAiRMAWQ {
    uint32_t jettyId;
    uint64_t sqVA;     // SQE在HBM上起始地址
    uint32_t wqeSize;  // 一个WQEBB占用内存大小（64B）
    uint32_t sqDepth;  // 可用的WQEBB个数
    uint64_t headAddr; // AIV无依赖
    uint64_t tailAddr; // AIV无依赖
    uint64_t dbAddr;   // JFSDoorBell地址
    uint32_t tp_id;
    uint8_t rmtEid[16];
    uint32_t rmtObjId; // rmtTokenID
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    uint32_t jfcId;
    uint64_t cqVA;    // CQE在HBM上起始地址
    uint32_t cqeSize; // 一个CQE占用内存大小（64B）
    uint32_t cqDepth; // 可用的CQE个数
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr; // JFCDoorBell地址
};

struct HcclCombinOpParam {
    uint64_t workSpace;                // client和server之间通信的地址
    uint64_t workSpaceSize;            // client和server之间通信的空间大小
    uint32_t rankId;                   // 当前卡rankId
    uint32_t rankDim;                  // 总卡数
    uint64_t winSize;                  // ccu不使用
    uint64_t windowsIn[MAX_RANK_NUM];  // ccu不使用
    uint64_t windowsOut[MAX_RANK_NUM]; // ccu不使用

    // for ccu
    uint64_t xnAddr;  // Xn寄存器其实地址
    uint64_t ckeAddr; // CKE寄存器其实地址
    uint64_t msAddr;  // MS地址，预留
    uint64_t msSize;  // 可写的MS个数，预留

    uint32_t opType[MAX_OP_NUM];
    uint8_t algorithmType[MAX_OP_NUM];

    HcclAiRMAWQ sqs[MAX_RANK_NUM];
    HcclAiRMACQ cqs[MAX_RANK_NUM];
};

#endif // MOE_DISTRIBUTE_DISPATCH_SETUP_BASE_H
