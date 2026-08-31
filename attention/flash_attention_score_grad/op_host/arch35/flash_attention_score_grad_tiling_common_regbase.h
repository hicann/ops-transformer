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
 * \file flash_attention_score_grad_tiling_common_regbase.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <vector>
#include <register/tilingdata_base.h>
#include <register/op_impl_registry.h>
#include <tiling/tiling_api.h>
#include "op_host/tiling_base.h"
#include "op_host/tiling_type.h"
#include "../../op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h"

namespace optiling {
namespace fag {

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t LAST_AXIS_IDX = 1;
constexpr size_t SEC_LAST_AXIS_IDX = 2;
constexpr uint32_t MULT_BASE = 2;
constexpr uint32_t BATCH_MAX_SIZE = 2048;
constexpr uint32_t PREFIX_COMPRESS_S1_SIZE = 3072;
constexpr uint32_t ATTEN_MASK_COMPRESS_LIMIT = 2048;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;

constexpr size_t RESERVED_WORKSPACE_SIZE = static_cast<size_t>(64 * 1024);
constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;
constexpr uint32_t DTYPE_ENUM_INDEX_4 = 4;
constexpr uint32_t DTYPE_ENUM_INDEX_5 = 5;
constexpr uint32_t DTYPE_ENUM_INDEX_6 = 6;
constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;

// pse
constexpr uint32_t PSE_DIM_NUM_1 = 1;
constexpr uint32_t PSE_DIM_NUM_2 = 2;
constexpr uint32_t PSE_NORMAL_SHAPE_DIM = 4;

constexpr uint32_t INPUT_IDX_ATTEN_MASK = 7; // 7 是 AttenMask 输入索引
constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_0 = 0;
constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_1 = 1;

constexpr uint32_t ATTEN_MASK_DIM_LENGTH_2 = 2;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_4 = 4;
constexpr int64_t COMPRESS_ATTEN_MASK_SIZE = 2048 * 2048;

constexpr uint32_t INPUT_FORMAT_BN2GS2D = 3; // BNSD
constexpr uint32_t INPUT_FORMAT_S2BN2GD = 2; // SBH
constexpr uint32_t INPUT_FORMAT_BS2N2GD = 1; // BSH  BSND
constexpr uint32_t INPUT_FORMAT_TND = 4;     // TND
constexpr uint32_t INPUT_DIM_0 = 0;          // BSH  BSND
constexpr uint32_t INPUT_DIM_1 = 1;
constexpr uint32_t INPUT_DIM_2 = 2;
constexpr uint32_t INPUT_DIM_3 = 3;

constexpr uint32_t CORE_INIT_NUM = 40;

constexpr uint32_t QUERY_IDX = 0;
constexpr uint32_t KEY_IDX = 1;
constexpr uint32_t VALUE_IDX = 2;
constexpr uint32_t HEAD_DIM_IDX = 3;
constexpr uint32_t DROP_MASK_IDX = 5;
constexpr uint32_t PRE_TOKEN_ATTR_IDX = 2;
constexpr uint32_t NEXT_TOKEN_ATTR_IDX = 3;
constexpr uint32_t HEAD_ATTR_IDX = 4;
constexpr uint32_t LAYOUT_ATTR_IDX = 5;
constexpr uint32_t SEED_ATTR_IDX = 9;
constexpr uint32_t OFFSET_ATTR_IDX = 10;
constexpr uint32_t OUTDTYPE_ATTR_IDX = 11;
constexpr uint32_t TND_SOFTMAX_IN_ATTR_IDX = 12;
constexpr uint32_t DSINKOUT_IDX = 6;

constexpr uint32_t GM_ALIGN = 512;

constexpr uint32_t TOTAL_BLOCK_DIMENSION = 2;
constexpr uint32_t CALCULATED_BLOCK_DIMENSION = 4;
constexpr uint32_t BEGIN_IDX = 0;
constexpr uint32_t END_IDX = 1;
constexpr uint32_t SUM_S1S2 = 2;
constexpr uint32_t SUM_ALL = 3;
constexpr uint32_t LENGTH_IDX = 2;

constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t INT64_BYTES = 8;
constexpr uint32_t INT64_BLOCK_NUM = 32 / sizeof(int64_t);

constexpr uint32_t AICV_RATIO_DEFAULT = 2;
constexpr uint32_t S1CV_RATIO_DEFAULT = 2;
constexpr uint32_t S2CV_RATIO_DEFAULT = 1;
constexpr size_t WORKSPACE_BUFFER = static_cast<size_t>(20 * 1024 * 1024);
constexpr uint32_t BIT_NUMS = 8;
constexpr int64_t ALIGN128 = 128;
constexpr int64_t BN2_MAX_S = 128;
constexpr int64_t BN2S2_MAX_S = 1024;
constexpr int64_t BN2_MULTIBLK_SEQ = 640;
constexpr int64_t BN2_MULTIBLK_BN_128 = 128;
constexpr int64_t BN2_MULTIBLK_BN_256 = 256;
constexpr int64_t BN2_MAX_D = 512;
constexpr int64_t BN2S2_WRITE_UB_D = 128;
constexpr int64_t ROPE_D_192 = 192;
constexpr int64_t ROPE_D_64 = 64;
constexpr int64_t NEGATIVE_128 = -128;

constexpr uint32_t PRE_BUFFER_SIZE = static_cast<uint32_t>(112 * 1024);
constexpr uint32_t REGBASE_POST_BASE = static_cast<uint32_t>(128 * 128);
constexpr uint32_t ROPE_POST_BASE = static_cast<uint32_t>(96 * 192);
constexpr uint32_t CAST_BUFFER_LEN = static_cast<uint32_t>(60 * 1024);
constexpr uint32_t OUTPUT_BUFFER_LEN = static_cast<uint32_t>(30 * 1024);
constexpr int64_t OUTINDEX = static_cast<int64_t>(-1);
constexpr int64_t ALIGN64 = 64;
constexpr int64_t INT64_NUM = 32;
constexpr uint32_t DKDV_OUT = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t UB_RESERVE_SPACE = 26 * 1024;

constexpr int64_t LARGE_INVALID_NUM = 3072;

constexpr uint32_t CORE_LIST_NUM = 36;
constexpr uint32_t ARRAY_LENGTH = 3;
constexpr uint32_t DETER_LENGTH = 4;
// deter swizzle threshold settings
constexpr int64_t GQA_DENSE_MAX_ROUND_GROWTH_PERCENT = 3;
constexpr int64_t GQA_DENSE_MAX_INVALID_PERCENT = 4;
constexpr int64_t GQA_DENSE_PERCENT_BASE = 100;

constexpr int64_t TND_DETER_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT = 90;
constexpr int64_t TND_DETER_SWIZZLE_MAX_ROUND_GROWTH_PERCENT = 3;
constexpr int64_t TND_DETER_SWIZZLE_PERCENT_BASE = 100;

constexpr int64_t DETER_BAND_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT = 90;
constexpr int64_t DETER_BAND_SWIZZLE_MAX_ROUND_GROWTH_PERCENT = 3;
constexpr int64_t DETER_BAND_SWIZZLE_PERCENT_BASE = 100;

constexpr uint32_t NZ_OUT_MIN_S_SIZE = 2048;
constexpr uint32_t FP16_C0_SIZE = 16;
constexpr uint32_t TND_SWIZZLE_MIN_S1_SIZE = 2048;
constexpr uint32_t TND_SWIZZLE_MIN_S1_SIZE_1 = 1024;

constexpr uint32_t SFMG_DEFAULT_BURST_NUM = 128;
constexpr uint32_t SFMG_DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t SFMG_FP32_ACCUMULATOR_SIZE = 8;

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    DY,
    PSE_SHIFT,
    DROP_MASK,
    PADDING_MASK,
    ATTEN_MASK,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SOFTMAX_IN,
    ATTENTION_IN,
    PREFIX_N,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_START_IDX,
    KV_START_IDX,
    D_SCALE_Q,
    D_SCALE_K,
    D_SCALE_V,
    D_SCALE_DY,
    D_SCALE_O,
    QUERY_ROPE_IDX,
    KEY_ROPE_IDX,
    SINK_IDX,
    D_SCALE_DS_IDX,
    D_SCALE_P_IDX
};

enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_EQUAL_S_MODE,
    PREFIX_COMPRESS_MODE,
    RIGHT_DOWN_CASUAL_BAND_MODE,
    BAND_LEFT_UP_CASUAL_MODE
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    KEEP_PROB,
    PRE_TOKENS,
    NEXT_TOKENS,
    HEAD_NUM,
    INPUT_LAYOUT,
    INNER_PRECISE,
    SPARSE_MODE,
    PSETYPE,
    SEED,
    OFFSET,
    OUT_TYPE,
    TND_SOFTMAX_IN
};

enum class PseShapeType : uint32_t {
    PSE_SHAPE_TYPE_BNSS,
    PSE_SHAPE_TYPE_BN1S,
    PSE_SHAPE_TYPE_1NSS,
    PSE_SHAPE_TYPE_BNHS,
    PSE_SHAPE_TYPE_1NHS,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

enum class PseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0,
    PSE_OUTER_ADD_MUL_TYPE, // default
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE
};

enum class AttenDataType : uint32_t {
    ATTEN_MASK_TYPE_SAME = 0,   // 0 表示 AttenMask 数据类型与 qkv 一致
    ATTEN_MASK_TYPE_U8_BOOL = 1 // 1 表示 AttenMask 数据类型为 u8 bool
};

enum class SparseMode : uint32_t {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8
};

enum class ConstAxisTemplateType : uint32_t {
    AlignTo16 = 1,
    AlignTo32 = 2,
    AlignTo64 = 3,
    AlignTo128 = 4,
    AlignTo256 = 5,
    AlignTo512 = 6,

    AlignTo48 = 7,
    AlignTo80 = 8,
    AlignTo96 = 9,
    AlignTo112 = 10,
    AlignTo160 = 11,
    AlignTo192 = 12,
    AlignTo224 = 13,
    Other = 0
};

enum class ConstAxisTemplateNum : uint32_t {
    NUM1 = 1,
    NUM16 = 16,
    NUM32 = 32,
    NUM64 = 64,
    NUM80 = 80,
    NUM96 = 96,
    NUM112 = 112,
    NUM128 = 128,
    NUM114 = 144,
    NUM160 = 160,
    NUM176 = 176,
    NUM192 = 192,
    NUM208 = 208,
    NUM224 = 224,
    NUM240 = 240,
    NUM256 = 256,
    NUM512 = 512,
    NUM768 = 768,
};

enum class SplitAxisEnum : uint32_t {
    BN2GS1S2 = 0,
    BN2 = 1,
    B = 2,
    N2 = 3,
    BN2GS1 = 4,
    BN2S2 = 5,
    NONE = 9
};

enum class AttenMaskShapeType : uint32_t {
    ATTENMASKBN2GS1S2,
    ATTENMASKBS1S2,
    ATTENMASKS1S2,
    ATTENMASKTT = 99
};

enum class PseLayoutType : uint32_t {
    pseS1S2 = 0,
    pse1S2 = 1,
    pseSlopeBn = 2,
    pseSlopeN = 3
};

enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3 // 超L2优化暂不支持sparse的场景
};

enum class DeterSparseType : uint32_t {
    NO_DETER = 0,  // 非确定性
    DETER_OLD = 1, // 确定性老实现方案
    DETER_DENSE = 2,
    DETER_CAUSAL = 3,
    DETER_BAND = 4
};

struct DeterPrefixData {
    std::vector<int64_t> prefix0 = {0};
    std::vector<int64_t> prefix1 = {0};
    std::vector<int64_t> prefix2 = {0};
    std::vector<int64_t> deterPrefix = {0};
    std::vector<int64_t> deterPrefixAlign = {0};
    std::vector<int64_t> mNewList = {};
    std::vector<int64_t> nNewList = {};
    std::vector<int64_t> pNewList = {};
    std::vector<int64_t> qNewList = {};
};

struct TndBandDeterRoundInfo {
    int64_t lastBatchId = 0;
    uint64_t lastValidRound = 0;
    uint64_t coreFirstBatchLastRound = 0;
    uint64_t coreLastBatchStartRound = 0;
};

struct TndBaseInfo {
    // for swizzle
    uint64_t tndSwizzleS1S2PrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    uint64_t tndSwizzleS1S2AlignPrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    uint64_t tndS2BlockPrefixSum[TND_SWIZZLE_PREFIX_NUM] = {0};
    bool isTndSwizzle = false;

    // no swizzle
    uint64_t tndStartBIdx[CORE_LIST_NUM] = {0};
    uint64_t tndS1S2PrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndS1S2AlignPrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndPrefixSum[CORE_LIST_NUM] = {0};
    uint64_t tndPrefixValidSum[CORE_LIST_NUM] = {0};

    bool isS1GreaterThanS2 = true;
    bool isS1LessThanS2 = true;
    bool isSeqExistZero = false;
};

struct FuzzyBaseInfoParamsRegbase { // 频繁使用的基础参数
    uint64_t coreNum;
    uint64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;

    int64_t b;
    int64_t s1;
    int64_t n1;
    int64_t n2;
    int64_t s2;
    int64_t g;
    int64_t d;
    int64_t d1; // head dim of value

    int64_t s1Outer;
    int64_t s1Inner;
    int64_t s1CvInner;
    int64_t s1Tail;
    int64_t s1CvTail;

    int64_t s2Outer;
    uint32_t s1CvRatio = 1;
    uint32_t s2CvRatio = 1;
    int64_t cvS2Inner;
    int64_t s2Inner;
    int64_t s2Tail;
    int64_t s2CvTail;

    uint32_t sfmgdInner;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t sumS1S2Product = 0;

    uint32_t queryType;
    uint32_t pseOptional;
    uint32_t pseType = static_cast<uint32_t>(PseType::PSE_OUTER_ADD_MUL_TYPE);
    uint32_t pseShapeType = 0;
    uint32_t pseLayoutType = 0;
    uint32_t pseDtype = 0;
    uint32_t attenMaskOptional;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t dropoutIsDivisibleBy8 = 0;
    uint32_t layoutType;
    float scaleValue;
    float keepProb;
    float dsScale = 1;
    float pScale = 1;
    float pScaleLog = 1;
    uint32_t bandIdx;
    int64_t seed;
    int64_t offset;
    DtypeEnum outDtype;
    int64_t keepProbUint8;

    uint32_t calTypeSize;
    int64_t s1Token;
    int64_t s2Token;
    uint32_t blockOuter;
    int64_t blockFactor;
    int64_t maxValidBBLen = 0;
    bool noNeedDeter = true;
    uint64_t dqIsNeedDeter[CORE_LIST_NUM];
    uint64_t dkDvIsNeedDeter[CORE_LIST_NUM];

    uint64_t qSize;
    uint64_t kSize;
    uint64_t vSize;
    uint64_t dropMaskSize;
    bool dropMaskOuter;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    uint32_t sparseMode;
    uint32_t prefixN[BATCH_MAX_SIZE] = {0};

    bool isAllSame = false;
    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    uint32_t tmpBufferSize = 0;

    DtypeEnum inputDtype;
    bool isDeterministic = false;
    uint32_t deterSparseType;
    uint8_t sparseType;
    bool isS1S2Same = false;
    bool coreDivide = false;
    uint8_t tndLineDeter = 0;
    int64_t deterMaxRound = 0;
    // 每个 batch 的前缀面积总和 prefix,
    // 小于128b传完整的前缀和，大于128b的，按步长传部分前缀和，在kernel内组装完整的前缀和
    int64_t deterPrefixThreshold = 128;
    int64_t deterPrefixStep = 1;
    int64_t deterPrefix[132] = {0};
    int64_t deterPrefixAlign[132] = {0};
    int64_t deterPrefix0[132] = {0};
    int64_t deterPrefix1[132] = {0};
    int64_t deterPrefix2[132] = {0};
    // 确定性计算需要全核同步的轮次
    uint64_t startNeedSyncRound[CORE_LIST_NUM];
    uint64_t endNeedSyncRound[CORE_LIST_NUM];
    int64_t separateDkOffset[CORE_LIST_NUM];

    bool isBn2 = false;
    bool isBn2MultiBlk = false;
    bool hasRope = false;
    SplitAxisEnum splitAxis = SplitAxisEnum::BN2GS1S2;
    bool sValueZeroUnderTND = false;
    bool isInvalidCol = false;
    bool isInvalidRow = false;
    uint64_t tailZeroCount = 0;

    ConstAxisTemplateNum s1TemplateType = ConstAxisTemplateNum::NUM128;
    ConstAxisTemplateNum s2TemplateType = ConstAxisTemplateNum::NUM128;
    ConstAxisTemplateNum dTemplateType = ConstAxisTemplateNum::NUM64;

    int64_t qStartIdx;
    int64_t kvStartIdx;
    bool enableSwizzle = false;
    bool isSplitByBlockIdx = false;
    DeterBandScheduleMode deterBandScheduleMode = DeterBandScheduleMode::DISABLED;
    uint32_t sinkOptional = 0;
    uint64_t sinkSize = 0;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;
    uint32_t tndMaxSumLayout = 0; // 0: BNS8, 1: TND
    bool enablePreSfmg = 0;
    bool isNzOut = false;
};

template <class T>
inline auto AlignTo(const T n, const T alignSize) -> T
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T>
auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

inline int64_t AbsCeil(int64_t num1, int64_t num2)
{
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (std::abs(num1) + std::abs(num2) - 1) / std::abs(num2);
    return isNegative ? -result : result;
}

inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

template <class T>
inline auto CeilDivideBy(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <class T>
inline std::vector<T> SliceVector(const std::vector<T> &arr, const int64_t step)
{
    if (step == 1) {
        return arr;
    }

    std::vector<T> result;
    for (auto it = arr.begin(); it < arr.end(); it += step) {
        result.push_back(*it);
    }
    return result;
}

ge::graphStatus CheckSoftmaxMaxShape(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1,
                                     bool isQuant);
ge::graphStatus CheckSoftmaxSumShape(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1,
                                     bool isQuant);
ge::graphStatus CheckAttentionInShape(const gert::TilingContext *context);
ge::graphStatus CheckSoftmaxDtype(const gert::TilingContext *context);
ge::graphStatus CheckAttentionInDtype(const gert::TilingContext *context);
ge::graphStatus CheckShapeValid(const gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d);
ge::graphStatus CheckTndShapeValid(const gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d);
ge::graphStatus CheckDtypeValid(const gert::TilingContext *context);

ge::graphStatus CheckAttenMaskShape(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool CheckIsLargeInvalidBlk(const FuzzyBaseInfoParamsRegbase &fBaseParams);
void JudgeIsNeedDeter(FuzzyBaseInfoParamsRegbase &fBaseParams, std::array<int64_t, CORE_LIST_NUM> &dqOffset,
                      std::array<int64_t, CORE_LIST_NUM> &dkDvOffset, std::array<int64_t, CORE_LIST_NUM> &dqOffsetpre,
                      std::array<int64_t, CORE_LIST_NUM> &dkDvOffsetpre, int64_t calcNum, bool &noNeedDeter,
                      bool &dqNeedDeterpre, bool &dkDvNeedDeterpre);
void GetOffset(FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t &currentDqOffset, int64_t &currentDkDvOffset,
               int64_t blockIdx);
int64_t GetTotalPerBatchNum(FuzzyBaseInfoParamsRegbase &fBaseParams, uint8_t sparseType);
void PrintShapeInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
bool CheckSparseModeValue(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool CheckVarLenSparseModeValue(FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus CheckUnpadTokensInfo(FuzzyBaseInfoParamsRegbase &fBaseParams);
int64_t FindBandIdx(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool IsNewDeter(const FuzzyBaseInfoParamsRegbase &fBaseParams);
bool CheckPrefixNExist(FuzzyBaseInfoParamsRegbase &fBaseParams, const int64_t bIdx, const int64_t prefixN,
                       std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx);
void CalcleBandDeterParam(FuzzyBaseInfoParamsRegbase &fBaseParams);
void CalcleCausalDeterParam(FuzzyBaseInfoParamsRegbase &fBaseParams);
void SetSparsePrefixBlockInterval(const FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t bIdx, int64_t nIdx,
                                  std::vector<std::vector<std::pair<int64_t, int64_t>>> &s1ValidIdx,
                                  int64_t (&blockStarts)[CORE_LIST_NUM], int64_t (&blockEnds)[CORE_LIST_NUM],
                                  uint32_t &coreNum, int64_t &tmepBlock);
std::pair<uint32_t, uint32_t> GetS1S2TemplateType(FuzzyBaseInfoParamsRegbase &fBaseParams);
uint32_t GetDTemplateType(FuzzyBaseInfoParamsRegbase &fBaseParams);
void GetCommS1S2OuterInfo(FuzzyBaseInfoParamsRegbase &fBaseParams, const int64_t prefixN,
                          std::vector<std::pair<int64_t, int64_t>> &s1ValidIdx);
void GetCommonS1S2OuterIndex(const FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t (*parseInfo)[ARRAY_LENGTH],
                             int64_t gTail, int64_t &s1oIdx, int64_t &s2oIdx);
void CalcleActualToken(FuzzyBaseInfoParamsRegbase &fBaseParams, int64_t batchIdx, int64_t &actualCalcS1Token,
                       int64_t &actualCalcS2Token);
ge::graphStatus ProcessOptionalInput(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
void ProcessDropoutIsDivisibleBy8(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessDropoutInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessQuantInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessSinkInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessSparseModeInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessTokensInfo(FuzzyBaseInfoParamsRegbase &fBaseParams);
void SetQKVStartIdx(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus ProcessPseNormal(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                                 const char *inputLayout);
ge::graphStatus ProcessPseInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                               const char *inputLayout);
void SetPseLayout(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool SetSparseParams(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
void SetSplitAxis(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                  const TndBaseInfo &tndBaseInfo);
void DetermineMode(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool SupportTrans2BS2N2GD(const FuzzyBaseInfoParamsRegbase &fBaseParams);
ge::graphStatus SetAttenMaskShapeType(FuzzyBaseInfoParamsRegbase &fBaseParams, const gert::StorageShape *attenMaskShape,
                                      size_t dimNum);
ge::graphStatus ProcessInnerPseInfo(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams,
                                    size_t pseShapeDim);
ge::graphStatus ProcessPseSparseMode8(FuzzyBaseInfoParamsRegbase &fBaseParams);
bool SetPrefixSparseParams(const gert::TilingContext *context_, FuzzyBaseInfoParamsRegbase &fBaseParams);
} // namespace fag
} // namespace optiling
