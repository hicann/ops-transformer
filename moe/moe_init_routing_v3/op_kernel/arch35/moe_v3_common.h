/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_common.h
 * \brief
 */
#ifndef MOE_V3_COMMON_H_REGBASE
#define MOE_V3_COMMON_H_REGBASE

#include "kernel_operator.h"
#include "moe_init_routing_v3_arch35_tiling_def.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

// 一个scale块元素数量为64个，即1个scale块对应64个x的元素
constexpr int64_t SCALE_BLOCK_SIZE = 64LL;
// scale的第三维度大小
constexpr int64_t SCALE_THIRD_DIM_SIZE = 2LL;
// scale与x的对应系数为32
constexpr int64_t SCALE_FACTOR_WITH_X = 32LL;

constexpr int64_t GATHER_OUT_BUFFER_NUM = 2;

constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int8_t OVERFLOW_MODE_CTRL = 60;

constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t ONE_REPEAT_COMPARE_NUM = 64;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t GATHER = 0;
constexpr int64_t SCATTER = 1;

// 量化模式（与 host quantMode 对齐）：-1 非量化 / 0 静态 / 1 动态(int8)
constexpr int64_t CS_QUANT_MODE_UNQUANT = -1LL;
constexpr int64_t CS_QUANT_MODE_STATIC = 0LL;
constexpr int64_t CS_QUANT_MODE_DYNAMIC = 1LL;
constexpr float CS_DYNAMIC_QUANT_INT8_SCALE = 127.0f;

// INT4对称量化scale分母，使用2^(n-1)-1=7而非8；
// 对称量化范围[-7,7]保证每对正负值有对称编码，-8(1000)编码不会被使用。
constexpr float DYNAMIC_QUANT_INT4_SYM_SCALE = 7.0f;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr int64_t DROPLESS_MODE = 0LL;
constexpr int64_t DROP_PAD_MODE = 1LL;
constexpr int64_t EXPERT_ID_VALUE_NUM = 2LL;

constexpr uint16_t FLOAT_REG_TENSOR_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;

// fp32指数位偏移
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;

// MX量化一个块的元素数量为32个，即1个scale对应32个x的元素
constexpr int64_t MX_BLOCK_SIZE = 32LL;
// 一个VL放多少fp32->fp8的元素个数，即按fp32算的一个VL能放几个元素
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64LL;
// fp16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t FP16_EMASK_AND_INF_VAL = 0x7c00;
// bf16中指数部分Mask，同时也表示bf16的INF值
constexpr uint16_t BF16_EMASK_AND_INF_VAL = 0x7f80;
// fp32中指数部分Mask，同时也表示fp32的INF值
constexpr uint32_t FP32_EMASK_AND_INF_VAL = 0x7f800000;
// bfloat16的nan值（与inf值不同）
constexpr uint16_t BF16_NAN_VAL = 0x7f81;
// 对于x的目标类型为e2m1的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E2M1 = 0x0100;
// 对于x的目标类型为e5m2的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E5M2 = 0x0780;
// 对于x的目标类型为e4m3的maxExp来说，最小的maxExp应该是多少
constexpr uint16_t LOWER_BOUND_OF_MAX_EXP_FOR_E4M3 = 0x0400;
// e8m0的inf/nan值（按定义应该是nan值）
constexpr uint16_t FP8_E8M0_NAN_VAL = 0x00ff;
// e8m0的极小值，用于写0
constexpr uint16_t FP8_E8M0_SPECIAL_MIN = 0x0040;
// 异常值：负0
constexpr int32_t NEG_ZERO = 0x80000000;
// bf16指数位的偏移位数
constexpr int16_t BF16_EXP_SHR_BITS = 7;
// fp32指数位的偏移位数
constexpr int16_t FP32_EXP_SHR_BITS = 23;
// 用于计算halfScale=BF16_EXP_INVSUB-sharedExp，得到的halfScale就是1/realScale，可以用于量化xQuant=x*halfScale
constexpr uint16_t BF16_EXP_INVSUB = 0x7f00;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr int64_t EXERPT_TOKENS_COUNT = 1;
constexpr int64_t EXERPT_TOKENS_KEY_VALUE = 2;
constexpr int64_t EXERPT_TOKENS_NONE = 0;

// CountingSort 性能模板相关常量
constexpr int64_t COUNTING_SORT_MIN_N = 256LL;
constexpr int64_t COUNTING_SORT_MAX_N = 1024LL * 1024LL;
constexpr int64_t COUNTING_SORT_FILTER_CHUNK_SIZE = 4096LL;
constexpr int64_t COUNTING_SORT_MAX_PER_LOOP_COLS_BYTES = 64LL * 1024LL;
constexpr int64_t COUNTING_SORT_MAX_ACTUAL_EXPERT_NUM = 256LL;
constexpr int64_t COUNTING_SORT_ONE_BLOCK_ELEMENT = 8LL;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

// 元素级对齐（例如将 actualExpertNum 对齐到 8）：返回对齐后的元素数
__aicore__ inline int64_t AlignElem(int64_t elementNum, int64_t elementAlign)
{
    if (elementAlign <= 0) {
        return elementNum;
    }
    return (elementNum + elementAlign - 1) / elementAlign * elementAlign;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COMMON_H_REGBASE
