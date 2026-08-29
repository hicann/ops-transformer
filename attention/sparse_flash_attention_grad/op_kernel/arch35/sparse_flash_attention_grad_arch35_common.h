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
 * \file sparse_flash_attention_grad_arch35_common.h
 * \brief
 */

#pragma once

#include "kernel_operator.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include <cstdint>

using namespace regbaseutil;
using AscendC::Cast;
using AscendC::RoundMode;

namespace commondef {
// L0C自主管理相关
constexpr uint32_t MIN_L0C_BUF_NUM = 4;
constexpr int32_t DK_DV_L0C_BUF_NUM = 2;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
// BSND 的 N（kernel 里为 gSize）<= 该阈值时：gather 64、K/Q/Dy 双槽常驻，无 common scratch
constexpr int64_t SFAG_HEAD_N_L1_RESIDENT = 64;
constexpr uint32_t SFAG_GATHER_S2_HEAD_N = 64;
constexpr uint32_t SFAG_L1_DS_P_EACH = 64 * 1024;
constexpr uint32_t SFAG_L1_COMMON_SIZE = 104 * 1024;
// 当前判断仅在FP32场景生效，后续需考虑FP16/BF16并结合L0DB开关
#define IS_L0_EXCEED(M, N, K, T1) (M * K * sizeof(T1) > L0_MAX_SIZE || K * N * sizeof(T1) > L0_MAX_SIZE);
constexpr uint32_t RESERVED_WORKSPACE_SIZE = 64 * 1024;
constexpr bool INPUT_DISABLE = 0;
constexpr bool INPUT_ENABLE = 1;

// SPLIT_AXIS Enum
constexpr uint8_t BN2GS1S2 = 0;
constexpr uint8_t BN2 = 1;
constexpr uint8_t BN2S2 = 5;

// D_TYPE
constexpr uint8_t FLOAT16_PRECISION = 3;
constexpr uint8_t BFLOAT16 = 2;

constexpr uint32_t BSNGD = 1;
constexpr uint32_t SBNGD = 2;
constexpr uint32_t BNGSD = 3;
constexpr uint32_t TND = 4;

constexpr uint32_t PREFIX_LENGTH = 64;
constexpr uint32_t SEQ_ARR_LENGTH = 256;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t INPUT_NUMS = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FLOAT_BLOCK_SIZE = 32 / sizeof(float);
constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t PSE_PERFORMANCE_MODE = 0x12;
constexpr uint32_t MAX_CUBE_CORE_NUM = 36;
constexpr uint32_t MAX_SUM_REDUCE_AXIS_SIZE = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t CV_CORE_RATIO = 2;
constexpr uint32_t GROUP_TSCM_MASK = 0x4;
constexpr uint32_t TSCM_BUF_NUM = 9;
// rope
constexpr uint32_t ROPE_D_RATIO = 9;
constexpr uint32_t ROPE_D_64 = 64;

// sparse mode
constexpr uint32_t NO_MASK = 0; // 未传入attenmask，不做mask操作
constexpr uint32_t ALL_MASK = 1;
constexpr uint32_t LEFT_UP_CAUSAL = 2;    // 左上角点划分的三角部分
constexpr uint32_t RIGHT_DOWN_CAUSAL = 3; // 右下角点划分的下三角部分
constexpr uint32_t BAND = 4;
constexpr uint32_t PREFIX = 5;
constexpr uint32_t PREFIX_COMPRESS = 6;
constexpr uint32_t RIGHT_DOWN_CASUAL_BAND = 7;
constexpr uint32_t BAND_LEFT_UP_CASUAL = 8;

// pse shape type same as tiling
constexpr uint32_t PSE_SHAPE_TYPE_BNSS = 0;
constexpr uint32_t PSE_SHAPE_TYPE_BN1S = 1;
constexpr uint32_t PSE_SHAPE_TYPE_1NSS = 2;
constexpr uint32_t PSE_SHAPE_TYPE_BNHS = 3;
constexpr uint32_t PSE_SHAPE_TYPE_1NHS = 4;
constexpr uint32_t PSE_B_N2_G_SLOPE = 5;
constexpr uint32_t PSE_1_N2_G_SLOPE = 6;

constexpr uint32_t PSE_COMPRESS_H = 1024;
constexpr uint32_t VREG_SIZE = 256;
constexpr uint32_t MAX_CONTINUOUS_BLOCK_NUM = 6;

// task step
constexpr int8_t TASK_INIT = 0;
constexpr int8_t TASK_C1C2 = 1;
constexpr int8_t TASK_C3C4C5 = 2;
constexpr int8_t TASK_SCATTERADD = 3;

struct DeterConstInfo {
    uint8_t usedCubeCoreNum;
    uint8_t usedVectorCoreNum;
    uint8_t eachVecCoreS1Offset;
    uint8_t eachVecCoreS2Offset;
    uint32_t dqEachVectorSize;
    uint32_t dkvEachVectorSize;
    // 确定性计算MTE3的stride
    int64_t deterBStride;
    int64_t deterN2Stride;
    int64_t deterGStride;
    int64_t deterS1oStride;
    uint32_t deterDqkSrcStride;
    uint32_t deterDvSrcStride;
    uint32_t deterDqDstStride;
    uint32_t deterDkDstStride;
    uint32_t deterDvDstStride;
    uint32_t deterVecCoreS1Offset;
    uint32_t deterDkVecCoreS2Offset;
    uint32_t deterDvVecCoreS2Offset;
    bool noNeedDeter;
    // event_id
    event_t eventIDScalarToMte2;
    event_t eventIDMte2ToScalar;
    event_t eventIDScalarToMte3;
    event_t eventIDMte3ToScalar;
    event_t eventIDMte3ToMte2;
    event_t eventIDMte2ToMte3;
};

struct FagConstInfo {
    ConstInfo<false, false> commonConstInfo{0};
    DeterConstInfo deterConstInfo{0};
    int64_t bSize;
    int64_t n2Size;
    float scaleValue;
    float attenMaskMinValue;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t n2GS1oS2o;
    int64_t gS1oS2o;
    int64_t s1oS2o;

    __gm__ uint8_t *seqS1_addr;
    __gm__ uint8_t *seqS2_addr;

    int64_t s1Token;
    int64_t s2Token;
    int64_t sparseMode;
    int64_t s1Outer;
    int64_t s2Outer;
    int64_t s1CvTail;
    int64_t s1Tail;
    int64_t s2Tail;
    uint32_t sfmgMaxLoopSize;
    uint32_t dAlignToBlock;
    uint32_t dAlignToBlockForFp8;
    int64_t mm2Ka;
    int64_t mm2Kb;
    int64_t mm3Ka;
    int64_t mm4Kb;
    int64_t dRopeSize = 64; // rope旋转的维度
    int64_t dTotalSize;

    int64_t selectedBlockCount = 2048;
    int64_t selectedBlockSize = 1;
    int64_t selectedCountOffset = 0;
    // true: host tiling 判定 gSize <= SFAG_HEAD_N_L1_RESIDENT，gather 64 + K/Q/Dy 双槽常驻
    bool isHeadNLe64 = false;
};

// fp8反量化因子
struct QuantScaleInfo {
    float deqScaleQValue = 1.0f;
    float deqScaleKValue = 1.0f;
    float deqScaleVValue = 1.0f;
    float deqScaleDyValue = 1.0f;
    float deqScaleOValue = 1.0f;
};

struct FagRunInfo {
    RunInfo<false> commonRunInfo{0};
    QuantScaleInfo quantScaleInfo;
    int64_t s2oIdx;
    int64_t s2CvBegin;
    int64_t s2CvEnd;
    int64_t kGmS2SplitOffset = 0;
    int64_t vGmS2SplitOffset = 0;
    uint8_t kvPingPong;
    int32_t s1RealSizeAlign2;
    int32_t s2RealSizeAlign2;
    int64_t dAlign16;
    int32_t halfS2RealSize; // vector侧实际的s2基本块大小，如果Cube基本块=128，那么halfS2RealSize=64
    int32_t
        firstHalfS2RealSize; // 当s2RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s2 size
    uint8_t qDxPingPongIdx;      // N<=64：Q/Dy L1 槽位，切 S1 时翻转
    uint8_t isS2IdxNoChange;     // s2Idx是否变化
    uint8_t isNextS2IdxNoChange; // 下一个基本块的s2Idx是否变化（是否切换了列）
    // BN2模板使用
    bool isLastS1Outer = false;  // 标记BN2扩展模板中是否是S1轴要处理的最后一个s1outer
    bool isFirstS1Outer = false; // 标记BN2扩展模板中是否是S1轴要处理的第一个s1outer

    // TND需要记录上一次的基本块的信息，用于优化scalar
    int64_t lastBatchIdx = 0;
    int64_t lastBatchTotalBaseIdx = 0;
    int64_t lastBatchTotalS1BOffset = 0;
    int64_t lastBatchTotalS1BRopeOffset = 0;
    int64_t lastBatchTotalS1BOffsetForDv = 0;
    int64_t lastBatchTotalS2BOffset = 0;
    int64_t lastBatchTotalS2BRopeOffset = 0;
    int64_t lastBatchTotalS2BOffsetForDv = 0;
    int64_t lastBatchTotalS1S2SizeAlign = 0;
    int64_t lastBatchTotalS1S2Size = 0;
    int64_t lastBatchTotalS2Size = 0;

    int64_t dyOffset;
    int64_t queryOffsetWithRope;
    int64_t keyOffsetWithRope;
    int64_t queryOffsetWithRopeForMm12;
    int64_t keyOffsetWithRopeForMm12;

    // BSND: BS TND: T
    int64_t t1Index = 0;
    int64_t t2Index = 0;
    int64_t n2Index = 0;
    int64_t actualSelectedBlockCount = 0;
    int64_t blkCntOffset = 0;
    int64_t actualSelCntOffset = 0;

    int64_t kSelectedWsAddr = 0;
    int64_t mm4ResWsAddr = 0;
    int64_t mm5ResWsAddr = 0;
    int64_t halfGRealSize = 0;
    int64_t firstHalfGRealSize = 0;
    // Deter 使用：sTaskId = Q-wave ordinal；deterTaskId = scatter protocol epoch
    int64_t sTaskId = 0;
    int64_t sTaskIdMod2 = 0;
    int64_t deterTaskId = 0;
    int64_t deterTaskIdMod2 = 0;

    bool isS1IdxNoChange;     // s1Idx是否变化
    bool isNextS1IdxNoChange; // 下一个基本块的s1Idx是否变化（是否切换了行）

    int8_t taskStep = TASK_INIT;
};

__aicore__ inline uint32_t AlignTo(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

__aicore__ inline int64_t AlignTo16(int64_t num)
{
    return (num + 15) >> 4 << 4;
}

__aicore__ inline int64_t AlignTo32(int64_t num)
{
    return (num + 31) >> 5 << 5;
}
} // namespace commondef
