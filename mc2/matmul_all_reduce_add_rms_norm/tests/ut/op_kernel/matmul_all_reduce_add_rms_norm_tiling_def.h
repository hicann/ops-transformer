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
 * \file matmul_all_reduce_add_rms_norm_tiling_def.h
 * \brief
 */

#ifndef MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H
#define MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_DEF_H

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "test_add_rms_norm.h"
#include "../../../../matmul_all_reduce/tests/ut/op_kernel/weight_quant_batch_matmul_v2_tiling.h"
#if __has_include("../../../../common/inc/hccl_stub.h")
#include "../../../../common/inc/hccl_stub.h"
#endif

#ifndef _ADD_RMS_NORM_TILING_H_
#define _ADD_RMS_NORM_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct AddRMSNormTilingData {
    uint32_t num_row;
    uint32_t num_col;
    uint32_t block_factor;
    uint32_t row_factor;
    uint32_t ub_factor;
    float epsilon;
    float avg_factor;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
    __ubuf__ tilingStruct* tilingDataPointer =                                 \
            reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                            \
    AddRMSNormTilingData tilingData;                                               \
    INIT_TILING_DATA(AddRMSNormTilingData, tilingDataPointer, tilingPointer);  \
    (tilingData).num_row = tilingDataPointer->num_row;                              \
    (tilingData).num_col = tilingDataPointer->num_col;                          \
    (tilingData).block_factor = tilingDataPointer->block_factor;                  \
    (tilingData).row_factor = tilingDataPointer->row_factor;            \
    (tilingData).ub_factor = tilingDataPointer->ub_factor;              \
    (tilingData).epsilon = tilingDataPointer->epsilon;            \
    (tilingData).avg_factor = tilingDataPointer->avg_factor;
#endif

#define __aicore__

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#endif

constexpr uint16_t MAX_TENSOR_CONT = 256;
constexpr uint16_t MAX_CORE_CONT = 64;

struct RCSTiling
{
    uint32_t rankDim = 0;
    uint32_t rankID = 0;
    uint32_t commtype = 0;
    uint32_t subtype = 0;

    uint32_t tileCnt = 0;  // 整块的个数
    uint32_t tailM = 0;
    uint32_t tailCnt = 0;
    uint32_t biasLen = 0;
    uint32_t isAdd = 0;

    uint32_t rankM = 0;    // 存放用户原始输入的mValue
    uint32_t rankN = 0;    // 存放用户原始输入的mValue
    uint32_t rankK = 0;
    uint32_t gatherIndex = 0;
    uint32_t isTransposeA = 0;
    uint32_t isTransposeB = 0;

    uint32_t storageGather = 0;
    uint32_t nd2NzWorkLen = 0;
    uint32_t cToFloatLen = 0;
    uint32_t gatherLen = 0;
    uint32_t workspaceAddr4 = 0;
    uint32_t aicCoreNum = 0;
    uint32_t ubCalcM = 0;
    uint32_t ubCalcN = 0;
    uint32_t needUbBuffer = 0;
    uint32_t addX3UbCnt = 0;
};

struct Mc2Msg
{
    uint64_t sendOff = 0;       // 发送数据地址偏移，count * dataTypeSize
    uint64_t recvOff = 0;       // 接收数据地址偏移 count * dataTypeSize
    uint64_t tailSendOff = 0;   // 尾块发送数据地址偏移，count * dataTypeSize
    uint64_t tailRecvOff = 0;   // 尾块接收数据地址偏移 count * dataTypeSize
    uint64_t sendCnt = 0;       // 整块发送数据个数
    uint64_t recvCnt = 0;       // 尾块接收数据个数
    uint64_t tailSendCnt = 0;   // 尾块发送数据个数
    uint64_t tailRecvCnt = 0;   // 尾块接收数据个数
    uint64_t totalCnt = 0;      // 总数据个数
    uint32_t turnNum = 0;       // 总轮次
    uint32_t tailNum = 0;       // 尾块的轮次
    uint32_t stride = 0;        // 跳写间隔
    uint32_t workspaceOff = 0;  // 使用workspace作为recvbuf时的workspace偏移
    uint32_t notifyOff = 0;     // device notify write/read value偏移

    uint16_t notifyBeginCnt = 0;// notift write value的使用个数
    uint16_t notifyEndCnt = 0;  // notift write value的使用个数
    uint8_t useBufferType = 0;   // recvBuf类型
    uint8_t funID = 0;          // funtion ID
    uint8_t dataType = 0;       // hccl 数据类型
    uint8_t groupNum = 0;       // groupNum

    uint8_t reuseMode = 0;      // 不复用填turnNum，内存优化选择复用的内存块个数
    uint8_t commType = 0;       // 通信类型
    uint8_t reduceOp = 0;       // reduce op type
    uint8_t commOrder = 0;      // 通信顺序，0表示通信在前，1表示通信在后
    uint8_t waitPolicy = 0;     // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
    // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    uint8_t rspPolicy = 0;      // 任务执行结束时的响应策略， 2、最后通知一次，
    // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    uint8_t exitPolicy = 0;     // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
    // 等待AIC通知退出(可以多次执行任务)。
    uint8_t commAlg = 0;        // 用于指定具体通信算法。
    // 0：defualt 1：fullmesh 2：doublering 3：switchwing
    uint8_t taskType = 0;       // 从参数获取通信任务，直接下发。AIC自己发Record激活
    uint8_t debugMode = 0;      // 调测模式
    // 1:单独执行CUBE
    // 2:单独执行Vector
    // 4:单独执行AICPU KFC算子
    // 8:KFC等待通信结束
    // 16:KFC统计各阶段耗时
    uint8_t stepSize = 0;       // 用于指定通算频率步长
    uint8_t reserve2 = 0;
};

struct Mc2L2cacheUseInfo{
    uint32_t l2CacheFlag;
};

struct Mc2L2cacheTilePara{
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};

struct Mc2MatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isHf32;
};

struct Mc2MatmulV3TilingData {
    TCubeTiling matmulTiling;
    Mc2L2cacheTilePara tileL2cacheTiling;
    Mc2MatMulRunInfo matmulRunInfo;
    Mc2L2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};

struct MatmulAllReduceTilingData
{
    Mc2Msg msg;
    RCSTiling param;
    Mc2MatmulV3TilingData tilematmulTiling;
    Mc2MatmulV3TilingData tailmatmulTiling;
};

struct MatmulAllReduce910TilingData {
    Mc2Msg msg;
    RCSTiling param;
    Mc2MatmulV3TilingData tilematmulTiling;
    Mc2MatmulV3TilingData tailmatmulTiling;
};

struct AddRMSNormTilingeKeyData
{
    uint32_t ARNKeyTile = 0;
    uint32_t ARNKeyTail = 0;
    uint32_t ARNBlockDimTile = 0;
    uint32_t ARNBlockDimTail = 0;
};

struct MatmulAllReduceAddRmsNormTilingData
{
    MatmulAllReduceTilingData matmulAllReduceTilingData;
    AddRMSNormTilingData addRMSNormTileTilingData;
    AddRMSNormTilingData addRMSNormTailTilingData;
    AddRMSNormTilingeKeyData addRmsNormTilingeKeyData;
};

inline void InitMatmulAllReduceAddRmsNormTilingData(uint8_t* tiling,
                                                    MatmulAllReduceAddRmsNormTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MatmulAllReduceAddRmsNormTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    MatmulAllReduceAddRmsNormTilingData tiling_data;                                                    \
    InitMatmulAllReduceAddRmsNormTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)            \
    auto var = ((tiling_type *)((uint8_t*)AscendC::GmAlloc(1024)))->member; \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);              \
    InitTilingData<decltype(var)>(tiling + offset##var, &var);

#endif  // FOREACH_MINIMUM_SCALAR_TILING_DEF_H
