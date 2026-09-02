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
 * \file flash_attention_score_grad_common.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_COMMON_H
#define FLASH_ATTENTION_SCORE_GRAD_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include <cstdint>
#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;
using namespace regbaseutil;
using AscendC::Cast;
using AscendC::RoundMode;

namespace commondef {
constexpr uint8_t SYNC_C1_TO_V2_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C2_TO_V2_FLAG[2] = {2, 3};
constexpr uint8_t SYNC_V3_TO_C3_FLAG = 4;
constexpr uint8_t SYNC_V4_TO_C5_FLAG = 5;
constexpr uint8_t SYNC_C3_TO_V5_FLAG = 6;
constexpr uint8_t SYNC_C4_TO_V6_FLAG = 7;
constexpr uint8_t SYNC_C4_TO_V3_FLAG = 8;
constexpr uint8_t SYNC_C5_TO_V4_FLAG = 9;
constexpr uint8_t SYNC_DETER_FIX_FLAG = 10;
constexpr uint8_t SYNC_DK_DETER_FIX_FLAG = 11;
constexpr uint8_t SYNC_DV_DETER_FIX_FLAG = 12;
// for small D preload 2 times
constexpr uint8_t SYNC_V2_TO_C1_FLAG[2] = {10, 11};
constexpr uint8_t SYNC_V2_TO_C2_FLAG[2] = {12, 13};

// MM_IDX
constexpr uint8_t DQ_IDX = 0;
constexpr uint8_t DK_IDX = 1;
constexpr uint8_t DV_IDX = 2;

// quant flag
constexpr uint8_t SYNC_COMPUTE_DKV_FLAG = 2;
constexpr uint8_t SYNC_TRANSFER_DKV_FLAG = 3;
constexpr uint8_t SYNC_TRANSFER_DQ_FLAG = 4;
constexpr uint8_t SYNC_UB2L1_P_FLAG = 9;
constexpr uint8_t SYNC_UB2L1_DS_FLAG = 10;
constexpr uint8_t SYNC_PDS_TO_DKV_FLAG = 8;
constexpr uint8_t SYNC_PDS_TO_DQ_FLAG = 7;
constexpr uint8_t SYNC_DETER_FLAG = 11;

// 最小Swizzle块数量
constexpr uint32_t MIN_SWIZZLE_S1 = 16384;
// Swizzle块数量，16K对应8块，随S增大倍数增大
constexpr uint32_t BASE_SWIZZLE_BLOCK_NUM = 8;
constexpr uint32_t M_SWIZZLE_SIZE = 32768;
constexpr uint32_t N_SWIZZLE_SIZE = 32768;
constexpr uint32_t SWIZZLE_CONTINUOUS_BLOCK_NUM = 16;
constexpr uint8_t MULTIPLY_COEF = 8;

// shift left by three bits
constexpr uint8_t kShiftToMultiplyByEight = 3;

// quant
constexpr uint16_t QUANT_S2_BASE_COUNT = 8;
constexpr uint16_t QUANT_S1_BASE_COUNT = 64;
constexpr uint16_t QUANT_S1_BASE_COUNT_DB = 128;

constexpr uint16_t QUANT_UB2L1_SRC_STRIDE = 15;
constexpr uint16_t QUANT_UB2L1_SRC_OFFSET = 512;

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
constexpr uint32_t PRE_INIT_UB_SIZE = 6 * 1024;

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

constexpr uint32_t MAX_SUM_BNS8 = 0;
constexpr uint32_t MAX_SUM_TND = 1;

constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FLOAT_BLOCK_SIZE = 32 / sizeof(float);
constexpr uint32_t MAX_CUBE_CORE_NUM = 36;
constexpr uint32_t MAX_SUM_REDUCE_AXIS_SIZE = 32;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t CV_CORE_RATIO = 2;
// rope
constexpr uint32_t ROPE_D_RATIO = 3;
constexpr uint32_t ROPE_D_128 = 128;
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

// deter sparse type
// 非确定性
constexpr uint8_t NO_DETER = 0;
// 确定性老实现方案
constexpr uint8_t DETER_OLD = 1;
constexpr uint8_t DETER_DENSE = 2;
constexpr uint8_t DETER_CAUSAL = 3;
constexpr uint8_t DETER_BAND = 4;

constexpr uint16_t ALIGN_OFFSET_16 = 15;
constexpr uint16_t ALIGN_OFFSET_32 = 31;
constexpr uint16_t ALIGN_OFFSET_64 = 63;
constexpr uint16_t ALIGN_OFFSET_128 = 127;
constexpr uint16_t ALIGN_OFFSET_512 = 511;

constexpr uint16_t OFFSET_BITS_4 = 4;
constexpr uint16_t OFFSET_BITS_5 = 5;
constexpr uint16_t OFFSET_BITS_6 = 6;
constexpr uint16_t OFFSET_BITS_7 = 7;
constexpr uint16_t OFFSET_BITS_9 = 9;

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
constexpr uint16_t UNROLL_FACTOR = 2;

constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_FIVE = 5;
constexpr uint32_t NUM_EIGHT = 8;

constexpr static bool end = true;

constexpr SyncAllConfig syncAllConfigMte2ToMte2 = {PIPE_MTE2, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte2 = {PIPE_MTE3, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte3 = {PIPE_MTE3, PIPE_MTE3};
constexpr SyncAllConfig syncAllConfigMte2ToMte3 = {PIPE_MTE2, PIPE_MTE3};

__aicore__ inline uint32_t AlignTo(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

__aicore__ inline int64_t AlignTo16(int64_t num)
{
    return (num + ALIGN_OFFSET_16) >> OFFSET_BITS_4 << OFFSET_BITS_4;
}

__aicore__ inline int64_t AlignTo32(int64_t num)
{
    return (num + ALIGN_OFFSET_32) >> OFFSET_BITS_5 << OFFSET_BITS_5;
}

__aicore__ inline int64_t AlignTo64(int64_t num)
{
    return (num + ALIGN_OFFSET_64) >> OFFSET_BITS_6 << OFFSET_BITS_6;
}

__aicore__ inline int64_t AlignTo128(int64_t num)
{
    return (num + ALIGN_OFFSET_128) >> OFFSET_BITS_7 << OFFSET_BITS_7;
}

__aicore__ inline int64_t AlignTo512(int64_t num)
{
    return (num + ALIGN_OFFSET_512) >> OFFSET_BITS_9 << OFFSET_BITS_9;
}

__aicore__ constexpr bool IS_DETER_OLD(const uint8_t deterSparseType)
{
    return deterSparseType == DETER_OLD;
}

__aicore__ constexpr bool IS_DETER_NEW(const uint8_t deterSparseType)
{
    return deterSparseType != NO_DETER && deterSparseType != DETER_OLD;
}

__aicore__ constexpr bool NEED_DETER_PREFIX(const uint8_t deterSparseType, bool isTnd)
{
    return isTnd && IS_DETER_NEW(deterSparseType);
}

__aicore__ constexpr bool NEED_DETER(const uint8_t deterSparseType)
{
    return deterSparseType != NO_DETER;
}

template <typename T, bool IS_WRITE_UB>
struct DqkvResPos {
    using PosType = typename std::conditional<IS_WRITE_UB, LocalTensor<T> &, GlobalTensor<T> &>::type;
};

template <typename T1>
__aicore__ constexpr bool GET_IS_L1_PRELOAD(const uint32_t HEAD_DIM_ALIGN, const uint32_t SPLIT_AXIS,
                                            const bool IS_DETER_OLD, const bool IS_TND, const bool FP8_OPEN_TSCM,
                                            const bool IS_ROPE)
{
    return (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned192) &&
            ((IS_DETER_OLD == 0) || (IS_DETER_OLD == 1 && IS_ROPE)) && (!IsSameType<T1, float>::value) &&
            (!IsSameType<T1, fp8_e5m2_t>::value) && (!IsSameType<T1, fp8_e4m3fn_t>::value) &&
            (!IsSameType<T1, hifloat8_t>::value)) ||
           (FP8_OPEN_TSCM && (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) &&
            (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value ||
             IsSameType<T1, hifloat8_t>::value));
}

template <typename T1>
__aicore__ constexpr bool GET_IS_L1_REUSE(const uint32_t HEAD_DIM_ALIGN, const bool IS_DETER_OLD,
                                          const bool FP8_OPEN_TSCM)
{
    return (((!IS_DETER_OLD && HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) ||
             (IS_DETER_OLD && HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned192))) &&
            (!IsSameType<T1, float>::value && !IsSameType<T1, fp8_e5m2_t>::value &&
             !IsSameType<T1, fp8_e4m3fn_t>::value && !IsSameType<T1, hifloat8_t>::value)) ||
           (FP8_OPEN_TSCM && (HEAD_DIM_ALIGN <= static_cast<uint32_t>(DTemplateType::Aligned256)) &&
            (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value ||
             IsSameType<T1, hifloat8_t>::value));
}

template <typename T1>
__aicore__ constexpr bool GET_IS_NZ_OUT(const uint8_t SPLIT_AXIS, const uint32_t HEAD_DIM_ALIGN, const bool IS_TND,
                                        const bool IS_ROPE, const bool IS_DETER_OLD)
{
    return (SPLIT_AXIS == BN2GS1S2 &&
            (!IsSameType<T1, float>::value && !IsSameType<T1, fp8_e5m2_t>::value &&
             !IsSameType<T1, fp8_e4m3fn_t>::value && !IsSameType<T1, hifloat8_t>::value) &&
            HEAD_DIM_ALIGN == static_cast<uint32_t>(DTemplateType::Aligned128) && !IS_DETER_OLD);
}

// 判断DK/DV能否驻留在L0C的宏。
// 计算公式：max(mm1_size, mm2_size, mm3_size) + mm4_size + mm5_size <= L0C_MAX_SIZE
// 其中 mm*_size 对应不同矩阵乘的中间结果大小。
#define IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    (((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) + ((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) + \
     ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) : \
                                        (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))) <= L0C_MAX_SIZE

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_TYPE) \
    X(CALC_TYPE) \
    X(OUTDTYPE)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(IS_ATTEN_MASK, bool, false) \
    X(IS_PSE, bool, false) \
    X(IS_DROP, bool, false) \
    X(IS_TND, bool, false) \
    X(IS_BN2_MULTIBLK, bool, false) \
    X(DETER_SPARSE_TYPE, uint8_t, 0) \
    X(IS_N_EQUAL, bool, false) \
    X(IS_D_NO_EQUAL, bool, false) \
    X(IS_ROPE, bool, false) \
    X(IS_NZ_OUT, bool, false) \
    X(IS_TND_SWIZZLE, bool, false) \
    X(SPLIT_AXIS, uint8_t, 0) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type(name) = (default_val),
#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
#define TEMPLATES_DEF_BASE \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */
#define TEMPLATES_DEF_BASE_NO_DEFAULT \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS \
    ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#define FagOldTilingType \
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase< \
        NEED_DETER(DETER_SPARSE_TYPE), NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), IS_TND, false> *__restrict

#define FagTilingType \
    const __gm__ FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2Regbase<NEED_DETER(DETER_SPARSE_TYPE), \
                                                                        NEED_DETER_PREFIX(DETER_SPARSE_TYPE, IS_TND), \
                                                                        IS_TND, IS_TND_SWIZZLE> *__restrict

struct LoopInfo {
    int64_t bIdx{0};
    int64_t n2Idx{0};
    int64_t gIdx{0};
    int64_t s1oIdx{0};
    int64_t s2oIdx{0};
};

struct Bn2MultiBlkInfo {
    int64_t s2oDimIdx{0};
    int64_t s2OuterTmp{0}; // TND场景此值不可直接使用constinfo中的S2Outer
    int64_t s2SparseLeft{0};
    int64_t s2SparseRight{0};
};

struct DeterConstInfo {
    uint8_t usedCubeCoreNum;
    uint8_t usedVectorCoreNum;
    uint8_t eachVecCoreS1Offset;
    uint8_t eachVecCoreS2Offset;
    uint32_t dqEachVectorSize;
    uint32_t dkvEachVectorSize;
    // 确定性计算MTE3的stride
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

    // quant
    float pScale;    // 量化参数
    float dsScale;   // 量化参数
    float pScaleD;   // 反量化参数
    float dsScaleD;  // 反量化参数
    float pScaleLog; // log(pScale)
    int64_t copyOutDStride;

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
    int64_t s2Tail;
    uint32_t sfmgMaxLoopSize;
    uint32_t dAlignToBlock;
    uint32_t dAlign16;
    uint32_t dvAlign16;
    int64_t mm2Ka;
    int64_t mm2Kb;
    int64_t mm3Ka;
    int64_t mm4Kb;
    int64_t dRopeSize = 64;          // rope旋转的维度
    uint32_t continuousBlockNum = 0; // 核内连续块数量
    // swizzle相关
    int64_t mSwizzleBlockNum = 0;
    int64_t mSwizzleBlockNumTail = 0;
    int64_t nSwizzleBlockNum = 0;
    int64_t nSwizzleBlockNumTail = 0;
    int64_t leftUpTotalRound = 0;
    int64_t leftDownTotalRound = 0;
    int64_t rightUpTotalRound = 0;
    int64_t rightDownTotalRound = 0;
    int64_t leftSingleColTotalRound = 0;
    int64_t leftTotalRound = 0;
    int64_t batchTotalRound = 0;
    int64_t actualS1Outer;         // m
    int64_t actualS2Outer;         // n
    int64_t firstValidColBlockNum; // p
    int64_t firstValidRowBlockNum; // q
    int64_t mGap;                  // 无效行数
    int64_t nGap;                  // 无效列数
    int64_t s2FirstWidth;          // L1
    int64_t s2SecondWidth;         // L2
    int64_t s2FirstBlockNum;       // R1
    int64_t s2SecondBlockNum;      // R2
    // sink相关
    uint32_t isSink = 0;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;
    // 核数
    uint32_t aicCoreNum = 0;
    // tnd max sum layout
    uint32_t tndMaxSumLayout = 0;
    bool enablePreSfmg = false;
};

// fp8反量化因子
struct QuantScaleInfo {
    float deqScaleQValue = 1.0f;
    float deqScaleKValue = 1.0f;
    float deqScaleVValue = 1.0f;
    float deqScaleDyValue = 1.0f;
    float deqScaleOValue = 1.0f;
};
// FP8场景字段
struct QuantRunInfo {
    uint32_t s1ProcessSize;
    uint32_t s2ProcessSize;
    uint32_t s1Idx;
    uint32_t s2Idx;
    bool isDqFixOut;
    bool isDkFixOut;
    bool isDvFixOut;

    float qkDScale;             // qDScale * kDScale
    float vdyDScale;            // dyDScale * vDScale
    float dsScaleDMulDeqScaleK; // dsScaleD * deqScaleKValue
    float dsScaleDMulDeqScaleQ; // pScaleD * deqScaleDyValue
    float pScaleDMulDeqScaleDy; // dsScaleD * deqScaleQValue

    bool isDkvCompleted = true;
    bool isDqCompleted = true;

    // 512基本块内部，再切128
    int64_t innerS1RealSize[4];
    int64_t innerS2RealSize[4];
    int64_t innerS1LoopNum;
    int64_t innerS2LoopNum;

    int64_t qInnerOffset[4];
    int64_t kvInnerOffset[4];

    bool kvNeedAtomic;
};

struct FagRunInfo {
    RunInfo<false> commonRunInfo{0};
    QuantScaleInfo quantScaleInfo;
    QuantRunInfo quantRunInfo;
    int64_t s2oIdx;
    int64_t s2CvBegin;
    int64_t s2CvEnd;
    int64_t kGmS2SplitOffset = 0;
    int64_t vGmS2SplitOffset = 0;
    int32_t halfS2RealSize; // vector侧实际的s2基本块大小，如果Cube基本块=128，那么halfS2RealSize=64
    int32_t
        firstHalfS2RealSize; // 当s2RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s2 size
    int32_t s1AlignedSize;
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
    // 只有确定性计算使用
    bool completed = true;
    int64_t dyOffset;
    int64_t queryOffsetWithRope;
    int64_t keyOffsetWithRope;
    int64_t queryOffsetWithRopeForMm12;
    int64_t keyOffsetWithRopeForMm12;
    int8_t specialS2Index = -1;

    bool isFirstBlock = true;
    bool isKeyReuse = false;
    bool isValueReuse = false;
    bool isNextKeyReuse = false;
    bool isLastProcessBlock = false;
    bool isFirstProcessBlock = false;

    int64_t maxsumOffset;

    // sink场景使用
    int64_t dsinkWorkSpaceOffset = 0;
    int64_t sinkN1Idx = 0;
};

} // namespace commondef
#endif // FLASH_ATTENTION_SCORE_GRAD_COMMON_H
