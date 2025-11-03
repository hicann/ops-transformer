/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rac_server_stub.h
 * \brief
 */

#ifndef _RAC_SERVET_STUB_H_
#define _RAC_SERVET_STUB_H_

enum RANK_MSG_TYPE { RANK_ADDR = 1, RANK_WORK = 2, RANK_ADD_WORK = 3, RANK_END };

constexpr uint32_t AC_MAX_RANK_NUM = 8;  // 最大有8个卡
constexpr uint32_t AC_MAX_AIV = 16;      // 最多有64个AIC
constexpr uint32_t AC_MSG_CNT = 16;      // 可以创建64个消息
constexpr uint32_t AC_MSG_VALID_MASK = 0x5CDF123A;
constexpr uint32_t HCCL_COMM_DOMAIN_KEY_MAX_LEN = 128;

enum AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_GATHER,
    HCCL_CMD_MAX
};

enum HcclReduceOp {
    HCCL_REDUCE_SUM = 0,  /* *< sum */
    HCCL_REDUCE_PROD = 1, /* *< prod */
    HCCL_REDUCE_MAX = 2,  /* *< max */
    HCCL_REDUCE_MIN = 3,  /* *< min */
    HCCL_REDUCE_RESERVED  /* *< reserved */
};

enum MC2_BUFFER_TYPE {
    MC2_BUFFER_TYPE_DEFAULT = 0,
    MC2_BUFFER_TYPE_OUTPUT,
    MC2_BUFFER_TYPE_WINDOW_IN,
    MC2_BUFFER_TYPE_WINDOW_OUT,
    MC2_BUFFER_TYPE_WORKSPACE,
    MC2_BUFFER_TYPE_INPUT,
    MC2_BUFFER_TYPE_COMMOUT,
    MC2_BUFFER_TYPE_END
};

/**
 * @brief HCCL data type
 */
enum HcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,   /* *< int8 */
    HCCL_DATA_TYPE_INT16 = 1,  /* *< int16 */
    HCCL_DATA_TYPE_INT32 = 2,  /* *< int32 */
    HCCL_DATA_TYPE_FP16 = 3,   /* *< fp16 */
    HCCL_DATA_TYPE_FP32 = 4,   /* *< fp32 */
    HCCL_DATA_TYPE_INT64 = 5,  /* *< int64 */
    HCCL_DATA_TYPE_UINT64 = 6, /* *< uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,  /* *< uint8 */
    HCCL_DATA_TYPE_UINT16 = 8, /* *< uint16 */
    HCCL_DATA_TYPE_UINT32 = 9, /* *< uint32 */
    HCCL_DATA_TYPE_FP64 = 10,  /* *< fp64 */
    HCCL_DATA_TYPE_BFP16 = 11, /* *< bfp16 */
    HCCL_DATA_TYPE_RESERVED    /* *< reserved */
};

struct HcclSignalInfo {
    uint64_t resId;  // 在代表event时为eventid，notify时为notifyid
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
};
// TP8卡
struct HcclCombinOpSignalParam {
    HcclSignalInfo noIpcNotifys[AC_MAX_RANK_NUM * 2];
    HcclSignalInfo ipcNotifys[AC_MAX_RANK_NUM * 4];
    HcclSignalInfo noIpcEvents[AC_MAX_RANK_NUM];
    HcclSignalInfo aicpuNotify;
    HcclSignalInfo aicpuOpNotify[2];  // 集合通信AICPU展开资源
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
};

struct HcclConfig {
    uint8_t determinism;  // 确定性计算开关
};

// HCCL 代码直调时直接传此结构：
struct HcclAicpuOpParam2 {
    uint32_t valid;
    HcclReduceOp op_type;  // 32b
    uint64_t send_buffer;  //
    uint64_t recv_buffer;
    uint64_t count;

    // offset 32B
    HcclDataType hccl_data_type;
    uint8_t rank_dim;
    uint8_t rank_id;
    uint8_t isLast;    // 是否最后一个下
    uint8_t funID;     // 功能ID，1地址消息；  2开始工作；
    uint8_t blockDim;  // 等价于启动的msgQueCnt
    uint8_t blockIdx;
    uint8_t sendCnt;  // 发送计数
    uint8_t rcvCnt;
    AicpuComType comm_type;  // 32b

    uint32_t groupNumber;  // 每组成员的个数
    uint8_t everyTurnRsp;  // 每轮都需要等待执行结束发送响应，再执行下一轮
    int8_t validCnt;
    int8_t sendValidCnt;  // 在发送时, 由AIC填写; Aicpu 回应时复制到validCnt 此处
    uint8_t res3;
    uint64_t stride;  // 表示列宽度
};

template <typename T, typename U>
struct same_type {
    enum { value = 0 };
};
template <typename T>
struct same_type<T, T> {
    enum { value = 1 };
};
template <class T> __aicore__ inline HcclDataType GetDataType()
{
    if constexpr (same_type<T, int8_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_INT8;
    } else if constexpr (same_type<T, int16_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_INT16;
    } else if constexpr (same_type<T, int32_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_INT32;
    } else if constexpr (same_type<T, half>::value) {
        return HcclDataType::HCCL_DATA_TYPE_FP16;
    } else if constexpr (same_type<T, bfloat16_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_BFP16;
    } else if constexpr (same_type<T, float>::value) {
        return HcclDataType::HCCL_DATA_TYPE_FP32;
    } else if constexpr (same_type<T, int64_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_INT64;
    } else if constexpr (same_type<T, uint64_t>::value) {
        return HcclDataType::HCCL_DATA_TYPE_UINT64;
    }
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

class HcclServer {
   public:
    struct KFCMsg {
        HcclAicpuOpParam2 notifyMsg[AC_MSG_CNT];
        HcclAicpuOpParam2 notifyCnt[AC_MSG_CNT];
    };
    __gm__ KFCMsg *msgNotify;

    int sendPos;
    int lastCnt;
    int lastReadCnt;

    __aicore__ inline void Init(GM_ADDR win, uint8_t debugMode) {}

    __aicore__ inline void TurnNotifyRun(uint32_t blockIdx) {}

    __aicore__ inline void TurnNotifyRun(uint32_t blockIdx, uint32_t usedCoreNum) {}

    __aicore__ inline void TurnNotifyRun(uint32_t blockIdx, uint32_t usedCoreNum, uint32_t curTurn) {}

    __aicore__ inline void TurnNotifyRunA8W8Bf16(uint32_t blockIdx, uint32_t usedCoreNum, uint32_t curTurn) {}

    // 等待aicpu执行完成1次任务
    __aicore__ inline void TurnWait(uint32_t turn) {}

    __aicore__ inline void SetQueue() {}
};

#endif  // _RAC_SERVET_STUB_H_
