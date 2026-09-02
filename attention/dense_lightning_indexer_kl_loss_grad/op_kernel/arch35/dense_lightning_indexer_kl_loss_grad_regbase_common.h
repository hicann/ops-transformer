/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file dense_lightning_indexer_kl_loss_grad_regbase_common.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_COMMON_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_REGBASE_COMMON_H

#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;

static constexpr uint32_t VEC_SY_BASESIZE = 128;
static constexpr uint32_t PROCESS_KV_SIZE = 8192;
static constexpr uint32_t VEC_P_BASESIZE = 128;
static constexpr uint32_t ATTN_SOFTMAX_REDUCE_BASESIZE = 8192;
// task step
constexpr int8_t TASK_INIT = 0;
constexpr int8_t TASK_C1 = 1;
constexpr int8_t TASK_V1 = 2;
constexpr int8_t TASK_C2 = 3;

enum class DLILayout {
    BSND = 0,
    TND = 1
};

enum class DLISparseMode {
    DefaultMask = 0,
    RightDown = 3
};

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
constexpr uint32_t UB_MAX_SIZE = 96 * 1024;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
constexpr uint32_t MODE_NUM_2 = 2;
constexpr uint32_t MODE_NUM_3 = 3;
constexpr uint32_t ALIGN_NUM_2 = 2;
constexpr uint32_t BLOCK_SINGLE_LEN = 16;

struct GatherParams {
    int32_t s2ProcessSize;
    int32_t dValue;
    int32_t dRopeValue;
    // topkIndex two line idx
    int64_t realS2Idx1;
    int64_t realS2Idx2;
    bool needGatherRope;
};

/* * @name 核内数据结构定义
 * @{
 */
struct DLIKGradConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_32 = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128 = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_9K = 9 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_11K = 11 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_33K = 33 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;
    /* * \brief 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;
    /* * \brief metadata分核信息 */
    uint32_t totalNum;
    uint32_t formerCoreProcessNNum;
    uint32_t remainCoreProcessNNum;
    uint32_t remainCoreNum;
    uint32_t usedCoreNum;
    uint32_t maxSeqKV;

    /* * \brief TilingData中的信息 */
    uint32_t bSize;
    uint32_t totalSize;
    uint32_t n2Size;              // 现阶段n2Size默认等于1
    uint32_t gSizeQuery;          // 128或者64
    uint32_t gSizeQueryIndex;     // 64或者32
    uint32_t s1Size;              // 支持泛化
    uint32_t s2Size;              // 支持泛化
    uint32_t dSizeQuery;          // 默认不带Rope，固定等于512
    uint32_t dSizeQueryIndex;     // 默认不带Rope，固定等于128
    uint32_t dSizeQueryRope = 64; // Rope，固定等于64
    uint32_t kSize;               // 现阶段只支持2048
    uint32_t kSizeAlign128;
    uint32_t cmpRatio;
    uint32_t maxSeqlenK;
    int64_t totalCost;
    DLISparseMode sparseMode; // 0或者3
    float scaleValue;
    float pScaler;

    // 轴的乘积提取一些公共信息，减少scalar
    int32_t s2BaseSize;
    int32_t dSizeRope;
    int32_t gatherKeyDbNum;
    int32_t gatherKeyIndexDbNum;
    int32_t gatherKeySize;
    int32_t gatherKeyIndexSize;
    int32_t sparseBlockSize = 1;

    int32_t syKBaseSize;
    int32_t pKBaseSize;
    int64_t bmm3BaseOffset = 0;
};

struct DLIKGradRunInfo {
    uint32_t taskId;
    uint32_t taskIdMod2;
    uint32_t taskIdMod3;
    uint32_t sTaskId;
    uint32_t sTaskIdMod3;
    uint32_t deterTaskId;
    uint32_t deterTaskIdMod3;
    uint32_t bIdx;
    uint32_t s1Idx;
    uint32_t t1Index;
    uint32_t t2Index;
    uint32_t kIdx;
    int64_t accumS1Idx;    // 当前循环累加的T1
    int64_t accumS2Idx;    // 当前循环累加的T2
    uint32_t actS1Size;    // 当前batch的S1Size
    uint32_t actS2Size;    // 当前batch的S2Size
    uint32_t cmpResidualK; // 当前s2方向压缩的残差
    uint32_t kBaseSize;    // k方向切分的基本块大小，在P和阶段
    uint32_t kRealSize;    // k切分之后的实际大小，在P和阶段
    uint32_t kTailSize;    // k切分之后的尾块大小，在P和阶段
    uint32_t kRealSizeAlign8;
    uint32_t kLoopTimes; // k方向的循环次数，在P和阶段
    uint64_t s2Offset;
    int64_t kProcessSize;
    bool isLastBlock;
    bool isAlign64;

    uint32_t nBaseSizeP;
    uint32_t nRealSizeP;
    uint32_t nBaseSizeSY;
    uint32_t nRealSizeSY;
    uint32_t nIdxP;
    uint32_t nIdxSY;

    // 存放一些offset，减少重复计算
    int32_t s2SparseLen;
    int32_t s2RealSize;
    int32_t s2BaseSize;
    int32_t s2LoopTimes;
    int32_t s2TailSize;
    int32_t s2CurSize;
    int32_t nIndexSize;
    int32_t firstNIndexSize;
    int32_t weightOffset;
    int32_t reduceSumOffset;

    // 根据bIdx、s1Idx算出的偏移
    int64_t queryTensorOffset = 0;
    int64_t queryRopeTensorOffset = 0;
    int64_t queryIndexTensorOffset = 0;
    int64_t keyIndexTensorOffset = 0;
    int64_t topkGmBaseOffset = 0;
    int64_t deterResOffset = 0;
    int64_t s2BlockOffset;
    bool isLastK = false;

    int8_t taskStep = TASK_INIT;
};

struct DLIKGradKRunInfo {
    uint32_t kTaskId;
    uint32_t kTaskIdMod2;
    uint32_t kTaskIdMod3;
    uint32_t s2RealBaseSize;
    int32_t kProcessSize;
    int32_t s2Idx;
    int32_t s2SingleLoopTimes;
    int32_t s2SingleIdx;
    int32_t s2SingleCurSize;
    bool isS2end;
    bool isAlign64;
    int32_t dValue;
    int32_t dRopeValue;
};

constexpr uint8_t SYNC_MM2_TO_V1_FLAG[3] = {1, 2, 3};
constexpr uint8_t SYNC_V2_TO_C3_FLAG[3] = {10, 11, 12};
constexpr uint8_t DETER_TO_FIX_SYNC_FLAG = 9;
constexpr uint8_t SCATTER_CUBE_SYNC_FLAG = 4;
constexpr uint8_t SYNC_C3_TO_V3_FLAG = 15;

constexpr uint8_t SYNC_C3_TO_V3_DETER_MTE2_FLAG = 6;
constexpr uint8_t SYNC_C3_TO_V3_DETER_SA_FLAG = 0;

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(INPUT_T) \
    X(OUT_T) \
    X(T)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(Layout_Q, DLILayout, DLILayout::TND) \
    X(Layout_KT, DLILayout, DLILayout::TND) \
    X(SparseMode, DLISparseMode, DLISparseMode::RightDown) \
    X(HasCuSeqlensQ, bool, false) \
    X(HasCuSeqlensK, bool, false) \
    X(HasSequsedQ, bool, false) \
    X(HasSequsedK, bool, false) \
    X(HasCmpResidualK, bool, false) \
    X(Deterministic, bool, false)

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

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass */
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

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T>
__aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

#endif
