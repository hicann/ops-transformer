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
 * \file stem_indexer_common.h
 * \brief
 */
#ifndef stem_indexer_COMMON_H
#define stem_indexer_COMMON_H
using namespace AscendC;
namespace SICommon {

// 与tiling的layout保持一致
enum class SI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BNSD = 2,
    BNSD = 3
};
// Sync mode between Cube and Vector.
constexpr uint32_t SI_SYNC_MODE4 = 4U;
constexpr uint32_t AIV0_AIV1_OFFSET = 16U;
constexpr uint32_t MM1_RES_BUFFER_NUM = 3U;
constexpr uint32_t VBIAS_BUFFER_NUM = 3U;
// M=64时每个AIV的QK结果为32KB；TopK histogram使用独立UB，不占用结果槽。
constexpr uint32_t MM1_RES_SLOT_BYTES = 32U * 1024U;
constexpr uint32_t CROSS_VC_EVENT = 0U;
constexpr uint32_t CROSS_CV_EVENT = CROSS_VC_EVENT + MM1_RES_BUFFER_NUM;
// Buffer size in bytes.
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32U;
constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64U;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256U;
constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512U;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024U;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048U;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096U;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192U;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384U;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768U;
constexpr uint32_t TRUNK_LEN_256 = 256U;
// Invalid index.
constexpr int INVALID_IDX = -1;
struct RowIdx4 {
    uint32_t v0 = 0U;
    uint32_t v1 = 0U;
    uint32_t v2 = 0U;
    uint32_t v3 = 0U;
};

struct TopkNum4 {
    uint32_t v0 = 0U;
    uint32_t v1 = 0U;
    uint32_t v2 = 0U;
    uint32_t v3 = 0U;
};

struct S2ValidLen4 {
    uint32_t v0 = 0U;
    uint32_t v1 = 0U;
    uint32_t v2 = 0U;
    uint32_t v3 = 0U;
};

template <typename T>
__aicore__ inline void SetLane(T &value4, uint32_t lane, uint32_t value)
{
    if (lane == 0U) {
        value4.v0 = value;
    } else if (lane == 1U) {
        value4.v1 = value;
    } else if (lane == 2U) {
        value4.v2 = value;
    } else {
        value4.v3 = value;
    }
}

template <typename T>
__aicore__ inline uint32_t GetLane(const T &value4, uint32_t lane)
{
    return (lane == 0U) ? value4.v0 : ((lane == 1U) ? value4.v1 : ((lane == 2U) ? value4.v2 : value4.v3));
}

template <typename Q_T, typename K_T, typename OUT_T, const bool CAUSAL = false, const int TOPK_SCORE_PRECISION = 1,
          typename... Args>
struct SIType {
    static_assert(TOPK_SCORE_PRECISION == 1 || TOPK_SCORE_PRECISION == 2,
                  "TOPK_SCORE_PRECISION must be 1(uint32) or 2(uint16)");

    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;

    using scoreType = std::conditional_t<TOPK_SCORE_PRECISION == 1, uint32_t, uint16_t>;

    static constexpr bool causalFlag = CAUSAL;
    static constexpr bool pageAttention = false;
    static constexpr SI_LAYOUT layout = SI_LAYOUT::BNSD;
    static constexpr SI_LAYOUT keyLayout = SI_LAYOUT::BNSD;
};
// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U; // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;  // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t promptLen = 0U;
    uint32_t s2ValidSize = 0U;
    bool curActSeqLenIsZero = false;
    uint32_t actMBaseSize = 0U;    // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    bool isNeedLD = false;         // 该基本块是否需要LD
};
struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;
    uint32_t s2LoopEnd;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;

    int64_t tensorQueryOffset;
    int64_t tensorKeyOffset;
    int64_t tensorKeyScaleOffset;
    int64_t tensorWeightsOffset;
    int64_t tensorVBiasOffset;
    int64_t indiceOutOffset;
    int64_t indiceLenOffset;
    uint32_t promptLen;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isNeedLD = false;
    int64_t saveWorkSpaceIdx = 0LL;
};

struct ConstInfo {
    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t headDim = 0ULL;
    uint64_t maxQb = 0ULL;
    uint64_t maxKb = 0ULL;
    uint64_t sparseCount = 0ULL; // topK选取大小
    uint64_t kSeqSize = 0ULL;    // kv最大S长度
    uint64_t qSeqSize = 1ULL;    // q最大S长度
    uint32_t usedCoreNum = 0U;
    uint32_t causal = 1U;
    uint32_t stemBlockSize = 128U;
    uint32_t stemStride = 16U;
    uint32_t initialBlocks = 4U;
    uint32_t windowSize = 4U;
    float rSquare = 1.0f / 64.0f;
    float alpha = 1.0f;
    float kBlockNumRateMedium = 0.2f;
    uint32_t kBlockNumBiasMedium = 30U;
    float kBlockNumRateLarge = 0.1f;
    uint32_t kBlockNumBiasLarge = 30U;
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    SI_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    bool isLDOpen = false;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;    // 当前核是否参与规约任务
    int64_t saveWorkSpaceIdx = 0LL; // 存放LD参数的地址
    uint32_t bn2Idx = 0U;           // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U; // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U; // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    int64_t indiceOutCoreOffset = 0LL; // 最终输出索引搬出Topk的初始偏移地址
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false; // 当前核是否需要进行Decode归约任务
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace SICommon

#endif // stem_indexer_COMMON_H
