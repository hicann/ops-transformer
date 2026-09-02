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
 * \file grouped_mat_mul_all_reduce_tiling_def.h
 * \brief
 */
#ifndef GROUPED_MATMUL_ALL_REDUCE_TILING_DEF_H
#define GROUPED_MATMUL_ALL_REDUCE_TILING_DEF_H

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

constexpr uint16_t MAX_TENSOR_CNT = 64;
constexpr uint16_t MAX_CORE_CONT = 20;

#pragma pack(1)
struct GMMAllReduceBaseParams {
    uint32_t groupNum;
    uint32_t coreNum;
    uint32_t activeType;
    uint32_t ubBaseM;
    uint32_t ubBaseN;
    uint32_t ubCalSize;
    uint32_t ubRestSize;
    uint32_t workspaceSize;
    int32_t mList[MAX_TENSOR_CNT] = {0};
    int32_t kList[MAX_TENSOR_CNT] = {0};
    int32_t nList[MAX_TENSOR_CNT] = {0};
};
#pragma pack()

#pragma pack(1)
struct GMMAllReduceCoreTiling {
    GMMAllReduceBaseParams baseParams;
    TCubeTiling mmTilingData;
    uint32_t notifyOff;
    uint32_t debugMode;
};
#pragma pack()

#pragma pack(1)
struct AllReduceMsg {
    uint64_t sendOff;      // 发送数据地址偏移，count * dataTypeSize
    uint64_t recvOff;      // 接收数据地址偏移, count * dataTypeSize
    uint64_t tailSendOff;  // 尾块发送数据地址偏移，count * dataTypeSize
    uint64_t tailRecvOff;  // 尾块接收数据地址偏移, count * dataTypeSize
    uint64_t sendCnt;      // 整块发送数据个数
    uint64_t recvCnt;      // 尾块接收数据个数
    uint64_t tailSendCnt;  // 尾块发送数据个数
    uint64_t tailRecvCnt;  // 尾块接收数据个数
    uint64_t totalCnt;     // 总数据个数
    uint32_t turnNum;      // 总轮次
    uint32_t tailNum;      // 尾块的轮次
    uint32_t stride;       // 跳写间隔
    uint32_t workspaceOff; // 使用workspace作为recvbuf时的workspace偏移
    uint32_t notifyOff;    // device notify write/read value偏移

    uint16_t notifyBeginCnt; // notift write value的使用个数
    uint16_t notifyEndCnt;   // notift write value的使用个数
    uint8_t useBufferType;   // recvBuf类型
    uint8_t funID;           // funtion ID
    uint8_t dataType;        // hccl 数据类型
    uint8_t groupNum;        // groupNum

    uint8_t reuseMode;  // 不复用填turnNum，内存优化选择复用的内存块个数
    uint8_t commType;   // 通信类型
    uint8_t reduceOp;   // reduce op type
    uint8_t commOrder;  // 通信顺序，0表示通信在前，1表示通信在后
    uint8_t waitPolicy; // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
                        // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    uint8_t rspPolicy; // 任务执行结束时的响应策略， 2、最后通知一次，
                       // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    uint8_t exitPolicy; // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
                        // 等待AIC通知退出(可以多次执行任务)。
    uint8_t commAlg;   // 用于指定具体通信算法。
                       // 0：defualt, 1：fullmesh, 2：doublering, 3：switchwing
    uint8_t taskType;  // 从参数获取通信任务，直接下发。AIC自己发Record激活
    uint8_t debugMode; // 调测模式
                       // 1:单独执行CUBE
                       // 2:单独执行Vector
                       // 4:单独执行AICPU KFC算子
                       // 8:KFC等待通信结束
                       // 16:KFC统计各阶段耗时
    uint8_t stepSize;  // 用于指定通算频率步长
    uint8_t reserve2;
};
#pragma pack()

#pragma pack(1)
struct KFCGroupedTiling {
    uint32_t groupNum;
    uint32_t reserve;
    uint8_t msg[MAX_TENSOR_CNT * 80];
};
#pragma pack()

#pragma pack(1)
struct GMMAllReduceTilingData {
    KFCGroupedTiling aicpuTiling;
    GMMAllReduceCoreTiling aicoreTiling;
};
#pragma pack()

inline void InitGroupedMatMulAllReduceTilingData(uint8_t *tiling, GMMAllReduceTilingData *constData)
{
    memcpy(constData, tiling, sizeof(GMMAllReduceTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    GMMAllReduceTilingData tilingData; \
    InitGroupedMatMulAllReduceTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA_MEMBER(tilingType, member, var, tiling) \
    auto var = ((tilingType *)((uint8_t *)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tilingType *)0)->member); \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#endif // GROUPED_MATMUL_ALL_REDUCE_TILING_DEF_H
