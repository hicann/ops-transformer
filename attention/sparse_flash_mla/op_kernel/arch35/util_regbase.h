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
 * \file util_regbase.h
 * \brief
 */

#ifndef SMLA_UTIL_REGBASE_H
#define SMLA_UTIL_REGBASE_H

#include "util.h"
#include "common/util_regbase_const_info.h"

using AscendC::QuePosition;
using AscendC::TQue;

namespace regbaseutil {
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
enum class VselrIndexEnum {
    GT_64_AND_LTE_128_INDEX = 0,
    GT_0_AND_LTE_64_INDEX = 1
};

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int64_t gSplitSize;            /* split-G模式下当前AIC处理的G轴行数 */ \
    int64_t s2LoopEndIdx;          /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2LineStartIdx = 0;    /* S2方向按行的起始位置 */ \
    int64_t s2LineEndIdx;          /* S2方向按行的结束位置 */ \
    int64_t s2OriLoopEndIdx;       /* S2方向的循环控制信息 souter层确定 */ \
    int64_t s2OriLineStartIdx = 0; /* S2方向按行的起始位置 */ \
    int64_t s2OriLineEndIdx;       /* S2方向按行的结束位置 */ \
    int64_t s2CmpLoopEndIdx; \
    int64_t s2CmpLineStartIdx = 0; \
    int64_t s2CmpLineEndIdx; \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    uint32_t mRealSize; \
    uint32_t halfMRealSize; \
    uint32_t firstHalfMRealSize; \
    int64_t attentionOutOffset; /* attentionOut的offset souter层确定 */ \
    int32_t actualS1Size;       /* Q的actualSeqLength */ \
    int32_t actualS2OriSize;    /* ori_kv的真实使用长度 */ \
    int32_t actualS2CmpSize;    /* cmp_kv的真实使用长度 */ \
    int32_t cmpResidual         /* cmp的余数，用于mask计算 */

struct RunParamStr { // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int64_t gs1LoopStartIdx;
    int64_t gs1LoopEndIdx;
    // BN循环生产的数据

    int64_t preTokensPerBatchOri = MAX_PRE_NEXT_TOKENS;  // 左上顶点的pretoken
    int64_t nextTokensPerBatchOri = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    int64_t preTokensPerBatchCmp = MAX_PRE_NEXT_TOKENS;  // 左上顶点的pretoken
    int64_t nextTokensPerBatchCmp = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;     // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset; // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t mOuterOffset;
    int64_t cubeMOuterOffset;
    uint32_t oriSparseBlockCount;
    uint32_t cmpSparseBlockCount;

    // lse 输出offset
    int64_t softmaxLseOffset; // souter层确定

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
    // FD S2-split
    int64_t firstFdDataWorkspaceIdx = 0;
    bool isCrossCoreSplit = false;
    int64_t s2SplitIdx = 0;
    bool isFirstS2SplitCore = true;
    int64_t baseBlockNumPerReductionBlock = 1;
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* s2的起始位置，sparse场景下可能不是0 */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* s2循环当前的循环index */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* s1轴的index */ \
    int64_t loop = 0;   /* for v0 perload loop */ \
    int64_t boIdx = 0;  /* b轴的index */ \
    int64_t n2oIdx = 0; /* n2轴的index */ \
    int64_t goIdx = 0;  /* g轴的index */ \
    int32_t s1RealSize; \
    int32_t halfS1RealSize; /* vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64 */ \
    int32_t \
        firstHalfS1RealSize; /* 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 \
                                size */ \
    int32_t mRealSize; \
    int32_t halfMRealSize; \
    int32_t firstHalfMRealSize; \
    int32_t s2RealSize; /* s2方向基本块的真实长度 */ \
    int32_t s2RealSizeUpdate; \
    int64_t s2AlignedSize; /* s2方向基本块对齐到16之后的长度 */ \
    int32_t vec2MBaseSize; \
    int32_t vec2MRealSize; \
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
    int64_t attentionOutOffset; \
    int32_t actualS1Size;          /* 非TND场景=总s1Size, Tnd场景下当前batch对应的s1 */ \
    int32_t actualS2CmpSize;       /* cmp_kv的真实使用长度 */ \
    int32_t cmpResidual;           /* cmp的余数，用于mask计算 */ \
    int64_t preTokensPerBatchOri;  /* vector2 左上顶点的pretoken */ \
    int64_t nextTokensPerBatchOri; /* vector2 ori 左上顶点的nexttoken */ \
    int64_t nextTokensPerBatchCmp; /* vector2 cmp 左上顶点的nexttoken */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    bool isCmp; \
    uint8_t resv[3]; \
    int64_t sOuterOffset; \
    int64_t mOuterOffset; \
    bool isCrossCoreSplit = false; \
    int64_t s2SplitIdx = 0; \
    bool isFirstS2SplitCore = true; \
    int64_t baseBlockNumPerReductionBlock = 1

struct RunInfo {
    COMMON_RUN_INFO;
    // 推理新增
    // lse 输出offset
    int64_t softmaxLseOffset;

    int64_t qSNumInOneBlock;
    int64_t oriKvLoopEndIdx;
    int64_t cmpKvLoopEndIdx;
    uint32_t oriSparseBlockCount;
    uint32_t cmpSparseBlockCount;
    int64_t firstFdDataWorkspaceIdx = 0;
    bool isFirstBase = true;
    bool isLastBase = true;
    bool needReduce = false;
    int64_t reduceBlockId = 0;
};

struct ConstInfo {
    SMLA_CONST_INFO_COMMON_FIELDS;
    SMLA_CONST_INFO_KV_STRIDE_FIELDS;
    SMLA_CONST_INFO_SPARSE_ONLY_FIELDS;
    SMLA_CONST_INFO_TOPK_FIELDS;
    SMLA_CONST_INFO_LSE_FIELDS;
};
} // namespace regbaseutil

#endif // SMLA_UTIL_REGBASE_H
