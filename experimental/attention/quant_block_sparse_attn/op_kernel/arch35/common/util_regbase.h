/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#ifndef FLASH_ATTENTION_UTIL_REGBASE_H
#define FLASH_ATTENTION_UTIL_REGBASE_H

#include "util.h"
#include "../../quant_block_sparse_attn_common.h"

using AscendC::QuePosition;
using AscendC::TQue;

namespace regbaseutil {
constexpr uint16_t regBytes = 256;
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
enum class VselrIndexEnum {
    GT_64_AND_LTE_128_INDEX = 0,
    GT_0_AND_LTE_64_INDEX = 1,
    DN_INDEX = 2,
    NZ_INDEX = 3,
    QSCALE_GATHER_INDEX = 4
};
enum class DTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int32_t s2LoopEndIdx; /* S2方向的循环控制信息 souter层确定 */ \
    /* cube视角的sOuter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定 */ \
    uint32_t s1RealSize; \
    uint32_t s1RealSizeAlign32; /* dn场景使用 */ \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    int64_t tensorQOffset;      /* query的offset souter层确定 */ \
    int64_t attentionOutOffset; /* attentionOut的offset souter层确定 */ \
    int64_t actualS1Size;       /* Q的实际长度，由cuSeqQlen相邻前缀差计算 */ \
    int64_t actualS2Size;       /* KV的实际长度，由seqUsedKvlen读取 */ \
    uint64_t b1SSOffsetAlign16; \
    int64_t qRopeNBGOffset; /* QueryRope 的 offset */ \
    int64_t kRopeNBGOffset; /* G方向上,不同g的KeyRope的offset */

struct RunParamStr { // 分核与切块需要使用到参数
    COMMON_RUN_PARAM;
    /* 推理新增 */
    int32_t actSparseLen;
    int64_t s1LoopStart;
    int64_t s1LoopEnd;
    // BN循环生产的数据
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS;  // 左上顶点的pretoken
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // 左上顶点的nexttoken

    // NBS1循环生产的数据
    int64_t sOuterOffset;     // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t cubeSOuterOffset; // 单个S内 souter的 souterIdx * halfS1RealSize souter层确定
    int64_t keyOffset;        // mm1 Key 的offset,后续更名为KFinalOffset

    // q k v attenMask不同轴的offset
    // B轴offset
    int64_t qBOffset;
    int64_t qBScalarOffset;

    // lse 输出offset
    int64_t softmaxLseOffset; // souter层确定
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx;  /* s2起始, sparse场景可能非0 */ \
    int64_t s2LoopCount; /* s2循环当前index */ \
    int64_t s2LoopLimit; \
    int64_t qBOffset; \
    int64_t qBScalarOffset; \
    int64_t s1oIdx = 0; /* s1轴index */ \
    int64_t boIdx = 0;  /* b轴index */ \
    int64_t n2oIdx = 0; /* n2轴index */ \
    int64_t goIdx = 0;  /* g轴index */ \
    int32_t s1RealSize; \
    int32_t s1RealSizeAlign32; \
    int32_t halfS1RealSize;      /* vec侧s1基本块大小, Cube=128时为64 */ \
    int32_t firstHalfS1RealSize; /* s1非2倍数时v0比v1少算一行 */ \
    int32_t s2RealSize;          /* s2基本块真实长度 */ \
    int64_t s2AlignedSize;       /* s2对齐到16后的长度 */ \
    int32_t s2SparseBlk1RealSize; \
    int32_t s2SparseBlk1RealAlignedSize; \
    int32_t s2SparseBlk2RealSize; \
    int32_t s2SparseBlk2RealAlignedSize; \
    int64_t sparseBase = 0;          /* sparseIndices当前(b,n2,g,qb)基址 */ \
    int64_t sparseS2TokenOffset = 0; /* sparseIndices映射后KV token offset */ \
    int32_t actSparseLen = 0;        /* sparseSeqLen当前实际选块数 */ \
    int32_t vec2S1BaseSize;          /* vec2切分后S1大小, 如64切成两份32 */ \
    int32_t vec2S1RealSize;          /* vec2切分后S1尾块, 如63切成32和31 */ \
    int64_t vecCoreOffset;           /* vec核相对cube起始s1偏移 */ \
    int64_t queryOffset;             /* mm1 Query offset */ \
    int64_t keyOffset;               /* mm1 Key offset */ \
    int64_t valueOffset;             /* mm2 Value offset */ \
\
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
\
    int64_t attentionOutOffset; \
    int64_t actualS1Size;       /* 当前batch的s1 */ \
    int64_t actualS2Size;       /* 当前batch的s2 */ \
    int64_t preTokensPerBatch;  /* vec2左上顶点pretoken */ \
    int64_t nextTokensPerBatch; /* vec2左上顶点nexttoken */ \
    int64_t b1SSOffsetAlign; /* TND s2对齐16后前面batch的s1*s2之和 */ \
    int64_t deScaleKvOffset; /* KV反量化scale在Gm偏移, shape[B,N2,1,Ceil(S2,128),1] */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset;

struct RunInfo {
    COMMON_RUN_INFO;
    /* sparseIndices 偏移量计算 */
    int64_t sparseBlkIdx1; /* 256/128 第一个sparse块 */
    int64_t sparseBlkIdx2; /* 256/128 第二个sparse块 */
    int64_t phyBlkNumIdx1; // blockTable映射后的blockNum的Idx
    int64_t phyBlkNumIdx2;
    bool sparseBlk1PartialMask; // true表示部分计算  false表示全量计算
    bool sparseBlk2PartialMask; // true表示部分计算  false表示全量计算
    // lse 输出offset
    int64_t softmaxLseOffset;
};

#define COMMON_CONST_INFO \
    /* 全局的基本块信息 */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t bSize; \
    int64_t t1Size; \
    int64_t dSize; \
    int64_t dSizeV; \
    int64_t dBasicBlock; \
    int64_t dSizeRope; \
    int64_t n1Size; \
    int64_t gSize; /* g轴的大小 */ \
    int64_t n2Size; \
    /* 轴的乘积 */ \
    int64_t gD; \
    int64_t n2D; \
    int64_t bN2D; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t bN2Dv; \
    int64_t n2G; \
    int64_t n2GD; \
    int64_t bN2GD; \
    int64_t n2GDv; \
    int64_t bN2GDv; \
    int64_t gS2; \
    int64_t s1Dr; \
    int64_t gS1Dr; \
    int64_t n2GS1Dr; \
    int64_t s2Dr; \
    int64_t n2S2Dr; \
    int64_t gDr; \
    int64_t n2Dr; \
    int64_t bN2Dr; \
    int64_t n2GDr; \
    int64_t bN2GDr; \
    int32_t s2BaseN2D; \
    int32_t s1BaseN2GD; \
    int64_t s2BaseBN2D; \
    int64_t s1BaseBN2GD; \
    int32_t s1BaseD; \
    int32_t s2BaseD; \
    int64_t s2BaseN2Dv; \
    int64_t s2BaseBN2Dv; \
    int64_t s1BaseN2GDv; \
    int64_t s1BaseBN2GDv; \
    int32_t s1BaseDv; \
    int32_t s2BaseDv; \
    int64_t s1OuterSize; \
    /* matmul跳读参数 */ \
    int64_t mm1Ka; \
    int64_t mm1Kb; \
    int64_t mm2Kb; \
    /* dq 或者attentionOut的Stride */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t layoutType; \
    uint8_t subBlockIdx; \
    bool softMaxCheckRes; \
    float scaleValue; \
    float pScale;

#define INFER_CONST_INFO \
    /* sparseIndices 相关 */ \
    uint32_t maxQb; \
    uint32_t maxKb; \
    /* 推理新增 */ \
    bool isRowInvalid;       /* 是否使能行无效 */ \
    bool isGqa; \
\
    uint32_t seqUsedQlenSize;  /* 去掉前导0后的cuSeqQlen长度 */ \
    uint32_t seqUsedKvlenSize; /* seqUsedKvlen 的长度 */ \
    uint32_t isKvContinuous;   /* 是否为tensorlist */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t blockTableDim2; \
    uint32_t blockSize; \
    uint32_t paLayoutType; \
    uint32_t paBlockNumSum; \
    uint32_t paBlockStride; \
    uint32_t combineDim; \
    bool rsvd1; \
    bool isSoftmaxLseEnable; \
    int64_t queryRightPaddingSize; \
    int64_t kvRightPaddingSize; \
    /* 后量化 */ \
    bool isPostQuantPerChnl; \
    bool isPostQuantBF16; \
    bool isPostQuantOffsetExist; \
    float postQuantScaleValue; \
    float postQuantOffsetValue; \
    /* sparseBlockSize */ \
    uint32_t qSparseBlockSize;  /* sparse_q_block_size (= 128)  */ \
    uint32_t kvSparseBlockSize; /* sparse_kv_block_size (= 128) */

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t bSize; \
    int64_t t1Size; \
    uint32_t n2Size; \
    uint32_t gSize; \
    uint32_t dSize : 16; \
    uint32_t dSizeV : 16; \
    /* special params */ \
    int64_t preTokens; \
    int64_t nextTokens; \
    uint32_t attenMaskS2Size; \
    /* core params */ \
    uint32_t s1OuterSize; \
    uint32_t bandIndex; \
    uint32_t compressMode : 4; \
    uint32_t layoutType : 4; \
    uint32_t dSizeRope : 11; \
    uint32_t splitCoreMode : 1; \
    uint32_t coreNum

struct ConstInfo {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
    int64_t n2GS1o;
    int64_t gS1o;
    uint32_t maxBlockNumPerBatch;
};

/* only support b32 or b64 */
template <bool isPa = false>
struct CVSharedParams;

/* CVSharedParams需要小于等于CacheLine的大小：128Bytes */
template <>
struct CVSharedParams<false> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;

    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;
};
template <>
struct CVSharedParams<true> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;

    uint32_t seqUsedQlenSize;
    uint32_t seqUsedKvlenSize;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;

    int32_t qSparseBlockSize;
    int32_t kvSparseBlockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    uint32_t paLayoutType;
    uint32_t paBlockStride;
};
} // namespace regbaseutil

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
