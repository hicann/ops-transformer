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
 * \file moe_distribute_combine_setup_base.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_BASE_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_BASE_H

#if __has_include("../common/mc2_kernel_utils.h")
#include "../common/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

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
constexpr uint64_t WIN_STATE_OFFSET = 350U * 1024U;
constexpr uint64_t STATE_WIN_OFFSET = 950U * 1024U;
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
constexpr uint32_t UINT32_BITS_OFFSET = 32U;
constexpr uint32_t NUM_TWO = 2U;

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
constexpr uint32_t SQE_COMMON_UINT64_OFFSET_1 = 1U; // rsv:12 rmt_jetty_or_seg_id:20 sge_num:8 tp_id:24
constexpr uint32_t SQE_COMMON_RMT_EID_63_0_OFFSET = 2U;
constexpr uint32_t SQE_COMMON_RMT_EID_127_64_OFFSET = 3U;
constexpr uint32_t SQE_RMT_ADDR_OFFSET = 5U;
constexpr uint32_t SQE_WITH_NOTIFY_RMT_ADDR_OFFSET = SQE_RMT_ADDR_OFFSET;
constexpr uint32_t SQE_DATA_ADDR_OFFSET = 7U;
constexpr uint32_t SQE_WITH_NOTIFY_DATA_ADDR_OFFSET = 11U;
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET = 7U;
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET = 8U;
constexpr uint32_t SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_VALUE_OFFSET = 6U;

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

// URMA通信接口
__aicore__ inline GM_ADDR GetURMASqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");

    return (GM_ADDR)((__gm__ HcclAiRMAWQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->sqs) + dstRankId);
}

__aicore__ inline GM_ADDR GetURMACqInfoGMAddr(GM_ADDR hcclContext, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or dstRankId >= MAX_RANK_NUM");

    return (GM_ADDR)((__gm__ HcclAiRMACQ *)(((__gm__ HcclCombinOpParam *)hcclContext)->cqs) + dstRankId);
}

__aicore__ inline void GetURMASqInfoTensor(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR sqInfoGMAddr = GetURMASqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((sqInfoGMAddr != nullptr), "GetURMASqInfoGMAddr failed");

    AscendC::GlobalTensor<uint8_t> sqInfoGlobalTensor;
    sqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoGMAddr));
    AscendC::DataCopyExtParams sqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMAWQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(sqInfoTensor, sqInfoGlobalTensor, sqInfoParams, sqInfoPadParams);
}

__aicore__ inline void GetURMACqInfoTensor(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, GM_ADDR hcclContext,
                                           uint32_t dstRankId)
{
    GM_ADDR cqInfoGMAddr = GetURMACqInfoGMAddr(hcclContext, dstRankId);
    ascendc_assert((cqInfoGMAddr != nullptr), "GetURMACqInfoGMAddr failed");

    AscendC::GlobalTensor<uint8_t> cqInfoGlobalTensor;
    cqInfoGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoGMAddr));
    AscendC::DataCopyExtParams cqInfoParams = {1U, static_cast<uint32_t>(sizeof(HcclAiRMACQ)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqInfoPadParams{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(cqInfoTensor, cqInfoGlobalTensor, cqInfoParams, cqInfoPadParams);
}

__aicore__ inline void GenerateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_SQE_SIZE); // 初始化为全0
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    AscendC::LocalTensor<uint64_t> sqeTensorU64 = sqeTensor.ReinterpretCast<uint64_t>();

    // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    // opcode=0x3
    sqeTensorU64(0) = (static_cast<uint64_t>(0x3U) << (SQE_COMMON_OPCODE_OFFSET * UINT8_BITS_OFFSET)) +
                      (static_cast<uint64_t>(0b10110000U) << (SQE_COMMON_UINT8_OFFSET_3 * UINT8_BITS_OFFSET));
}

__aicore__ inline void GenerateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                      uint32_t sqeCount = 1)
{
    AscendC::Duplicate<uint8_t>(sqeTensor, 0, WRITE_WITH_NOTIFY_SQE_SIZE); // 初始化为全0
    AscendC::SyncFunc<AscendC::HardEvent::V_S>();
    AscendC::LocalTensor<uint64_t> sqeTensorU64 = sqeTensor.ReinterpretCast<uint64_t>();

    // owner=1, rmt_jetty_type=0b01, token_en=1, nf=0
    // opcode=0x5，使用write with notify SQE
    sqeTensorU64(0) = (static_cast<uint64_t>(0x5U) << (SQE_COMMON_OPCODE_OFFSET * UINT8_BITS_OFFSET)) +
                      (static_cast<uint64_t>(0b10110000U) << (SQE_COMMON_UINT8_OFFSET_3 * UINT8_BITS_OFFSET));
}

__aicore__ inline void UpdateCommonSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                       const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    // 更新tp_id(24b), jetty_id(20b), rmt_eid(128b), rmt_token_value(32b)

    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> templateSqeU64 = sqeTensor.ReinterpretCast<uint64_t>();

    templateSqeU64(SQE_COMMON_UINT64_OFFSET_1) =
        (static_cast<uint64_t>(sqInfoU32(WQ_RMTOBJID_OFFSET)) << UINT32_BITS_OFFSET) + // rmt_jetty_or_seg_id
        static_cast<uint64_t>((sqInfoU32(WQ_TP_ID_OFFSET) & 0x00ffffff) | 0x01000000); // sge_num=1 tp_id

    // rmt_eid
    templateSqeU64(SQE_COMMON_RMT_EID_63_0_OFFSET) =
        (static_cast<uint64_t>(sqInfoU32(WQ_RMTEID_4_7_OFFSET)) << UINT32_BITS_OFFSET) +
        static_cast<uint64_t>(sqInfoU32(WQ_RMTEID_0_3_OFFSET));
    templateSqeU64(SQE_COMMON_RMT_EID_127_64_OFFSET) =
        (static_cast<uint64_t>(sqInfoU32(WQ_RMTEID_12_15_OFFSET)) << UINT32_BITS_OFFSET) +
        static_cast<uint64_t>(sqInfoU32(WQ_RMTEID_8_11_OFFSET));

    templateSqeU32(SQE_COMMON_RMT_TOKEN_VALUE_OFFSET) = sqInfoU32(WQ_RMTTOKENVALUE_OFFSET); // rmt_token_value
}

__aicore__ inline void UpdateCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                          const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    UpdateCommonSQE(sqeTensor, sqInfoTensor);

    // 更新rmt_token_id(20b)
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    templateSqeU32(SQE_TOKEN_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET); // rmt_token_id
}

__aicore__ inline void UpdateCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor,
                                                    const AscendC::LocalTensor<uint8_t> &sqInfoTensor)
{
    UpdateCommonSQE(sqeTensor, sqInfoTensor);

    // 更新rmt_token_id(20b), notify_token_value(32b), notify_token_id(20b)
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> templateSqeU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> templateSqeU64 = sqeTensor.ReinterpretCast<uint64_t>();
    templateSqeU32(SQE_WITH_NOTIFY_TOKEN_ID_OFFSET) = sqInfoU32(WQ_RMTOBJID_OFFSET); // rmt_token_id
    templateSqeU64(SQE_WITH_NOTIFY_NOTIFY_TOKEN_ID_VALUE_OFFSET) =
        (static_cast<uint64_t>(sqInfoU32(WQ_RMTTOKENVALUE_OFFSET)) << UINT32_BITS_OFFSET) + // notify_token_value
        static_cast<uint64_t>(sqInfoU32(WQ_RMTOBJID_OFFSET));                               // notify_token_id
}

__aicore__ inline void SetCommWriteSQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                       uint64_t rmtAddr, uint32_t length, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(SQE_RMT_ADDR_OFFSET) = rmtAddr;
    sqeLocalU64(SQE_DATA_ADDR_OFFSET) = dataAddr;
    sqeLocalU32(SQE_LENGTH_OFFSET) = length;
    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00000001;
    }
}

__aicore__ inline void SetCommWriteWithNotifySQE(const AscendC::LocalTensor<uint8_t> &sqeTensor, uint64_t dataAddr,
                                                 uint64_t rmtAddr, uint32_t length, uint64_t notifyAddr,
                                                 uint64_t notifyData, uint8_t cqe)
{
    // 设置cqe(1 bit), odr(3 bits), data_addr(64 bits), rmt_addr(64 bits), length(32 bits), notify_addr(64 bits),
    // notify_data(64 bits)

    AscendC::LocalTensor<uint32_t> sqeLocalU32 = sqeTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqeLocalU64 = sqeTensor.ReinterpretCast<uint64_t>();

    sqeLocalU64(SQE_WITH_NOTIFY_RMT_ADDR_OFFSET) = rmtAddr;
    sqeLocalU64(SQE_WITH_NOTIFY_DATA_ADDR_OFFSET) = dataAddr;
    sqeLocalU32(SQE_WITH_NOTIFY_LENGTH_OFFSET) = length;
    sqeLocalU64(SQE_WITH_NOTIFY_NOTIFY_ADDR_OFFSET) = notifyAddr; // notify_addr
    sqeLocalU64(SQE_WITH_NOTIFY_NOTIFY_DATA_OFFSET) = notifyData; // notify_data

    if (cqe > 0U) {
        // cqe=1, odr=0b010
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00100010;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11111010;
    } else {
        // cqe=0, odr=0b001
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) &= 0b11011001;
        sqeTensor(SQE_COMMON_UINT8_OFFSET_2) |= 0b00000001;
    }
}

__aicore__ inline void SendJFSDoorBell(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t sqPi)
{
    // 敲JFS DoorBell，更新硬件的sqPi
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    st_dev(sqPi, (__gm__ uint32_t *)(sqInfoU64(WQ_DBADDR_OFFSET)), 0); // Scalar操作
}

__aicore__ inline void SendJFCDoorBell(const AscendC::LocalTensor<uint8_t> &cqInfoTensor, uint32_t cqCi)
{
    // 敲JFC DoorBell，更新硬件的cqCi
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();

    uint64_t jfcDbValue =
        (static_cast<uint64_t>(cqInfoU32(CQ_JFCID_OFFSET)) << UINT32_BITS_OFFSET) + static_cast<uint64_t>(cqCi);
    st_dev(jfcDbValue, (__gm__ uint64_t *)(cqInfoU64(CQ_DBADDR_OFFSET)), 0); // Scalar操作
}

__aicore__ inline bool CheckCqeStatus(const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                      const AscendC::LocalTensor<uint8_t> &cqeTensor, uint32_t &outCqCi,
                                      uint32_t &outCqCiLinear, uint32_t &newestCompletedPi)
{
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqInfoTensor.ReinterpretCast<uint64_t>();

    uint32_t cqeSize = cqInfoU32(CQ_CQESIZE_OFFSET);
    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(CQ_CQVA_OFFSET)));
    AscendC::DataCopyExtParams cqeParams = {1U, 8U, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> cqePadParams{false, 0U, 0U, 0U};

    AscendC::DataCopyPad(cqeTensor, cqGlobalTensor[cqeSize * outCqCi], cqeParams, cqePadParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (cqeTensor(CQE_STATUS_OFFSET) == 0xff) {
        return true;
    } else if (cqeTensor(CQE_STATUS_OFFSET) != 0) {
        // 退出kernel并报错
        ascendc_assert(false, "CQE status is abnormal! status is %d, substatus is %d.\n", cqeTensor(CQE_STATUS_OFFSET),
                       cqeTensor(CQE_SUBSTATUS_OFFSET));
    }

    // status==0，处理当前CQE，从entry_idx获取对应WQE的sqPi
    newestCompletedPi = (static_cast<uint32_t>(cqeTensor(CQE_ENTRY_IDX_HIGH_OFFSET)) << UINT8_BITS_OFFSET) +
                        static_cast<uint32_t>(cqeTensor(CQE_ENTRY_IDX_LOW_OFFSET));
    // 把CQE的status设置为无效值，并写回CQ
    cqeTensor(CQE_STATUS_OFFSET) = 0xff;
    AscendC::SyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(cqGlobalTensor[cqeSize * outCqCi], cqeTensor, cqeParams);
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_MTE2>(); // 防止cqeTensor被下一轮加载覆盖

    return false;
}

__aicore__ inline void PollCommCQUpdateSQCI(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqInfoTensor,
                                            const AscendC::LocalTensor<uint8_t> &cqeTensor, uint32_t &outSqCi,
                                            uint32_t &outCqCi, uint32_t &outCqCiLinear)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqInfoTensor.ReinterpretCast<uint32_t>();

    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);
    uint32_t cqDepth = cqInfoU32(CQ_CQDEPTH_OFFSET);

    uint32_t pollTimes = 0;
    uint32_t newestCompletedPi = 0;
    while (pollTimes < cqDepth - 1) {
        if (CheckCqeStatus(cqInfoTensor, cqeTensor, outCqCi, outCqCiLinear, newestCompletedPi)) {
            break;
        }

        // 递增本地的outCqCi
        outCqCi = (outCqCi + 1) % cqDepth;
        ++outCqCiLinear;
        ++pollTimes;
    }

    // 更新本地的outSqCi
    if (pollTimes > 0) {
        outSqCi = (newestCompletedPi + 1) % sqDepth;
    }
    AscendC::SyncFunc<AscendC::HardEvent::MTE3_S>(); // 等cqe status写回无效值
    // 通过敲JFC DoorBell，更新硬件的cqCi
    SendJFCDoorBell(cqInfoTensor, outCqCiLinear);
}

__aicore__ inline void InvalidateCqeStatus(const AscendC::LocalTensor<uint8_t> &cqeInfoTensor,
                                           const AscendC::LocalTensor<uint8_t> &cqeTensor)
{
    // 为cqeBuf_申请256*32B空间，初始化HGM上的CQ空间时，如果cqDepth>256，则循环多次DataCopy
    AscendC::LocalTensor<uint64_t> cqInfoU64 = cqeInfoTensor.ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint32_t> cqInfoU32 = cqeInfoTensor.ReinterpretCast<uint32_t>();

    AscendC::GlobalTensor<uint8_t> cqGlobalTensor;
    cqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(cqInfoU64(CQ_CQVA_OFFSET)));
    uint32_t cqDepth = cqInfoU32(CQ_CQDEPTH_OFFSET);
    uint32_t cqeNum = (cqDepth > CQ_DEPTH_256) ? CQ_DEPTH_256 : cqDepth;

    AscendC::Duplicate<uint8_t>(cqeTensor, 0xff, UB_ALIGN * cqeNum); // 初始化为全1
    AscendC::SyncFunc<AscendC::HardEvent::V_MTE3>();

    AscendC::DataCopyExtParams dataCopyParams = {static_cast<uint16_t>(cqeNum), UB_ALIGN, 0U,
                                                 NORMAL_CQE_SIZE - UB_ALIGN, 0U};
    uint32_t loopTimes = AscendC::Ceil(cqDepth, CQ_DEPTH_256);
    if (loopTimes > 1) {
        for (uint32_t i = 0; i < loopTimes - 1; ++i) {
            AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * i], cqeTensor, dataCopyParams);
        }
        uint32_t tailBlockCqeNum = cqDepth - CQ_DEPTH_256 * (loopTimes - 1);
        if (tailBlockCqeNum != CQ_DEPTH_256) {
            dataCopyParams = {static_cast<uint16_t>(tailBlockCqeNum), UB_ALIGN, 0U, NORMAL_CQE_SIZE - UB_ALIGN, 0U};
        }
    }
    AscendC::DataCopyPad(cqGlobalTensor[NORMAL_CQE_SIZE * CQ_DEPTH_256 * (loopTimes - 1)], cqeTensor, dataCopyParams);
}

__aicore__ inline uint32_t GetAvailableSpace(const AscendC::LocalTensor<uint8_t> &sqInfoTensor, uint32_t &outSqPi,
                                             uint32_t &outSqCi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();

    // 根据本地的outSqPi outSqCi 计算SQ可用空间大小（SQ不会用满，会预留一个WQE）
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);

    if (outSqPi == outSqCi) {
        return (sqDepth - 1);
    } else if ((outSqPi + 1) % sqDepth == outSqCi) {
        return 0;
    } else if (outSqPi > outSqCi) {
        return (sqDepth - (outSqPi - outSqCi) - 1);
    }
    return (outSqCi - outSqPi - 1);
}

__aicore__ inline void PutCommSQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                  const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sendSqeIdx, uint32_t sqPi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(WQ_WQESIZE_OFFSET); // 单个WQEBB占用大小（64B），由HCCL传递
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);

    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(WQ_SQVA_OFFSET)));

    uint32_t targetSqPi = (sqPi + sendSqeIdx) % sqDepth;

    AscendC::DataCopy(sqGlobalTensor[sqeSize * targetSqPi], sqeTensor, WRITE_SQE_SIZE);
}

__aicore__ inline void PutCommNotifySQE(const AscendC::LocalTensor<uint8_t> &sqInfoTensor,
                                        const AscendC::LocalTensor<uint8_t> &sqeTensor, uint32_t sendSqeIdx,
                                        uint32_t sqPi)
{
    AscendC::LocalTensor<uint32_t> sqInfoU32 = sqInfoTensor.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint64_t> sqInfoU64 = sqInfoTensor.ReinterpretCast<uint64_t>();
    uint32_t sqeSize = sqInfoU32(WQ_WQESIZE_OFFSET); // 单个WQEBB占用大小（64B），由HCCL传递
    uint32_t sqDepth = sqInfoU32(WQ_SQDEPTH_OFFSET);

    AscendC::GlobalTensor<uint8_t> sqGlobalTensor;
    sqGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(sqInfoU64(WQ_SQVA_OFFSET)));

    // 计算当前SQE的目标位置（每个notify SQE占用2个WQEBB）
    uint32_t targetSqPi = (sqPi + (sendSqeIdx << 1)) % sqDepth;

    // 检查是否需要切块处理环形缓冲区边界
    if (likely(sqDepth - targetSqPi >= NUM_TWO)) {
        // 有足够空间，直接拷贝单个SQE
        AscendC::DataCopy(sqGlobalTensor[sqeSize * targetSqPi], sqeTensor, WRITE_WITH_NOTIFY_SQE_SIZE);
    } else {
        // 跨越尾部边界，需要切块拷贝
        // 先拷贝第一部分到尾部最后一个WQEBB位置
        AscendC::DataCopy(sqGlobalTensor[sqeSize * targetSqPi], sqeTensor, sqeSize);
        // 拷贝剩余部分到头部
        AscendC::DataCopy(sqGlobalTensor, sqeTensor[sqeSize], WRITE_WITH_NOTIFY_SQE_SIZE - sqeSize);
    }
}

__aicore__ inline void GetPICI(const AscendC::LocalTensor<uint32_t> &piCiTensor, GM_ADDR hcclContext,
                               uint32_t curRankId, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    // MTE批量搬运：32字节 = 8个uint32_t
    AscendC::DataCopyParams copyParams = {1U, 1U, 0U, 0U};
    AscendC::DataCopy(piCiTensor, piCiGlobalTensor, copyParams);
}

__aicore__ inline void UpdatePICI(const AscendC::LocalTensor<uint32_t> &piCiTensor, GM_ADDR hcclContext,
                                  uint32_t curRankId, uint32_t dstRankId)
{
    ascendc_assert((hcclContext != nullptr && curRankId < MAX_RANK_NUM && dstRankId < MAX_RANK_NUM),
                   "hcclContext is nullptr or curRankId >= MAX_RANK_NUM or dstRankId >= MAX_RANK_NUM");

    GM_ADDR piCiSpaceGM = (GM_ADDR)(((__gm__ HcclCombinOpParam *)hcclContext)->windowsOut[curRankId] + WIN_PICI_OFFSET +
                                    PICI_WIN_SIZE * dstRankId);
    AscendC::GlobalTensor<uint32_t> piCiGlobalTensor;
    piCiGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)piCiSpaceGM);

    // MTE批量搬运：32字节 = 8个uint32_t
    AscendC::DataCopyParams copyParams = {1U, 1U, 0U, 0U};
    AscendC::DataCopy(piCiGlobalTensor, piCiTensor, copyParams);
}

#endif // MOE_DISTRIBUTE_BASE_H
