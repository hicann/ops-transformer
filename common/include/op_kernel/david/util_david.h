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
 * \file util_david.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_DAVID_H
#define FLASH_ATTENTION_UTIL_DAVID_H

#include "../util.h"

using AscendC::TQue;
using AscendC::QuePosition;

namespace davidutil {
constexpr static uint16_t regBytes = 256;
enum class VselrIndexEnum {
    GT_64_AND_LTE_128_INDEX = 0,
    GT_0_AND_LTE_64_INDEX = 1
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
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3 // 超L2优化暂不支持sparse的场景
};

template <bool isInfer = false>
struct RunParam;

template <>
struct RunParam<false> { // 分核与切块需要使用到参数
    int64_t boIdx;
    int64_t s1oIdx;
    int64_t n2oIdx;
    int64_t goIdx;

    int32_t s2LoopStartIdx; // S2方向的循环控制信息  souter层确定
    int32_t s2LoopEndIdx;   // S2方向的循环控制信息  souter层确定
    int64_t s2LineStartIdx; // S2方向按行的起始位置
    int64_t s2LineEndIdx;   // S2方向按行的结束位置

    uint32_t s1RealSize; // cube视角的souter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定
    uint32_t s1RealSizeAlign32; // dn场景使用
    uint32_t halfS1RealSize;
    uint32_t firstHalfS1RealSize;

    int64_t tensorQOffset;      // query的offset  souter层确定
    int64_t attentionOutOffset; // attentionOut的offset  souter层确定
};

template <>
struct RunParam<true> { // 分核与切块需要使用到参数
    int64_t boIdx;
    int64_t s1oIdx;
    int64_t n2oIdx;
    int64_t goIdx;

    int32_t s2LoopStartIdx; // S2方向的循环控制信息  souter层确定
    int32_t s2LoopEndIdx;   // S2方向的循环控制信息  souter层确定
    int64_t s2LineStartIdx; // S2方向按行的起始位置
    int64_t s2LineEndIdx;   // S2方向按行的结束位置

    uint32_t s1RealSize; // cube视角的souter，在SAMEAB场景中cubeSOuterSize为两倍的 halfS1RealSize souter层确定
    uint32_t s1RealSizeAlign32; // dn场景使用
    uint32_t halfS1RealSize;
    uint32_t firstHalfS1RealSize;

    int64_t tensorQOffset;      // query的offset  souter层确定
    int64_t attentionOutOffset; // attentionOut的offset  souter层确定

    /* 推理新增 */

    // N循环生产的数据
    int32_t batchLoopEnd; // batch方向的循环控制信息 结束idx

    // NB循环生产的数据
    int64_t s2InCurrentBatch;                 // Tensorlist场景，不同batch的S2长度，后续用计算KvStride
    int64_t actualSeqLengthPerBatch = 0;      // Q的actualSeqLength
    int64_t actualSeqLengthOfMlaPerBatch = 0; // 在mla场景下Q的actualSeqLength
    int64_t actualSeqLengthKVPerBatch = 0;    // KV的actualSeqLength
    uint32_t singleProcessCubeSOuterSizeTail; // cube分核时，cube的souter的尾块，B循环时更新
    uint32_t unalignSInner;                   // s2方向尾块长度，非对齐  B循环时更新
    uint32_t singleProcessSInnerSizeTail;     // s2方向尾块长度，按64对齐
    uint32_t maskInnerTailAlign;              // s2方向尾块长度，按mask数据类型对齐(maskTypeByteNum)
    uint32_t padSize;                         // maskInnerTailAlign - unalignSInner
    uint32_t pseShiftInnerTailAlign;          // s2方向尾块长度，按pse数据类型对齐(pseShiftTypeByteNum)
    uint32_t pseShiftPadSize;                 // pseShiftInnerTailAlign - unalignSInner
    int64_t preTokensPerBatch = 0;            // 左上顶点的pretoken
    int64_t nextTokensPerBatch = 0;           // 左上顶点的nexttoken
    int64_t nextTokensOfMlaPerBatch = 0;      // 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效
    int32_t s1LoopEnd;                        // S1方向的循环控制信息 结束idx
    int32_t taskBatch;                        // sIdx PageAttention时mm需要使用

    // NBS1循环生产的数据
    int64_t sOuterOffset;         // 单个S内 souter的 souterIdx * singleProcessSOuterSize
    int64_t cubeSOuterOffset;     // 单个S内 souter的 souterIdx * singleProcessSOuterSize
    int64_t keyCoreOffset;        // Souter方向上，不同souter的Key的offset
    int64_t kRopeNBGOffset;       // G方向上，不同g的KeyRope的offset
    int64_t valueCoreOffset;      // Souter方向上，不同souter的value的offset
    uint64_t attenMaskCoreOffset; // Souter方向上，不同souter的attenMask的offset
    uint64_t pseShiftCoreOffset;  // Souter方向上，不同souter的pseShift的offset
    int64_t qRopeNBGOffset;       // QueryRope 的 offset

    // q k v attenMask不同轴的offset
    // B轴offset
    int64_t qBOffset; // bIdx * seqSize * multiHeadQ，后续更名为qBOffset
    int64_t qRopeBOffset;

    // Batch轴确定后，可以确定的信息
    int32_t sOuterBlockNum;     // S1方向的切块个数，后续更名为s1SplitNum
    uint32_t maxInnerLoopTimes; // S2方向的切块个数，用于判断是否为尾块，后续更名为s2SplitNum

    // 左padding
    int64_t queryLeftPaddingSize;
    int64_t kvLeftPaddingSize;

    // lse 输出offset
    int64_t softmaxLseOffset;
};

template <bool isInfer = false>
struct RunInfo;

template <>
struct RunInfo<true> {
    int64_t s2StartIdx; // s2的起始位置，sparse场景下可能不是0
    int64_t s2EndIdx;
    int64_t s2LoopCount; // s2循环当前的循环index
    int64_t s2LoopLimit;
    int64_t s1oIdx = 0; // s1轴的index
    int64_t boIdx = 0;  // b轴的index
    int64_t n2oIdx = 0; // n2轴的index
    int64_t goIdx = 0;  // g轴的index
    int32_t s1RealSize;
    int32_t s1RealSizeAlign32;
    int32_t halfS1RealSize; // vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64
    int32_t
        firstHalfS1RealSize; // 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size
    int32_t s2RealSize;     // s2方向基本块的真实长度
    int32_t s2VecSize;      // s2方向本次参与vec计算的长度
    int64_t s2AlignedSize;  // s2方向基本块对齐到16之后的长度
    int32_t vec2S1BaseSize; // vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32。
    int32_t
        vec2S1RealSize; // vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31。
    int64_t vecS1Offset; // vec核基于cube核起始处s1方向偏移
    int64_t vecS2Offset; // vec核基于cube核起始处s1方向偏移

    int64_t taskId;
    int64_t multiCoreInnerIdx = 0;

    int64_t qOffset;
    int64_t attentionOutOffset;
    int64_t
        s1SizeAcc; // 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）
    int64_t
        s2SizeAcc; // 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）
    int64_t s1SizeThisBatch; // 非TND场景=总s1Size, Tnd场景下当前batch对应的s1
    int64_t s2SizeThisBatch; // 非TND场景=总s2Size, Tnd场景下当前batch对应的s2
    int64_t b1SSOffset;      // 非TND = boIdx * s1 * s2; TND = 前面boIdx个batch的s1*S2之和。
    int64_t b1SSOffsetAlign; // TND场景s2 16对齐之后，前面batch的s1*s2之和
    int64_t deScaleQOffset;  // Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1]
    int64_t deScaleKvOffset; // KV的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, 1, Ceil(S2, 128), 1]
    float descaleQK = 1.0f;  // deScaleQValue * deScaleKValue
    float deSCaleVValue;
    uint8_t taskIdMod2;
    uint8_t taskIdMod3;
    uint8_t multiCoreIdxMod2 = 0;
    uint8_t multiCoreIdxMod3 = 0;

    // 这4个变量，后续可以改成枚举，变成1个变量
    bool isInnerTail;       // 是否为s2方向的尾块
    bool isFirstInnerIter;  // s2方向的控制信息，是否为首轮
    bool isSecondInnerIter; // s2方向的控制信息，是否为第二轮
    bool isLastInnerIter;   // s2方向的控制信息，是否为最后一轮，并非尾块 sparse

    bool splitPingPong; // 用于两轮sinner之间的ping pong
    bool taskPingPong;  // 用于sinner内每个task之间的ping pong

    int64_t sInnerOffsetDataSize; // mm1 PageAttention 记录s2轴的偏移
    int32_t taskBatch;            // mm1 PageAttention sIdx
    int64_t tensorAOffset;        // mm1 拷贝自RunParam中的tensorAOffset，后续更名为qFinalOffset
    int64_t tensorBOffset;        // mm1 Key 的 offset，后续更名为kFinalOffset
    int64_t qRopeOffset;
    int64_t kRopeOffset;

    uint32_t mm1SingleCoreN; // mm2 s2方向非尾块时等于singleProcessSInnerSize 区分尾块  // 重复 考虑删除
                             // singleProcessSInnerSizeNow
    uint32_t
        singleProcessSInnerBmmTail; // mm2 vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于unalignSInner
    int64_t valueOffset;            // mm2 Value 的 offset  复用tensorBOffset，后续更名为vFinalOffset

    uint32_t singleProcessSInnerSizeNow; // vector1 s2方向切块大小 区分尾块

    uint32_t maskCopyInCol; // vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于maskInnerTailAlign
    uint32_t padSize;       // vector1 maskInnerTailAlign - unalignSInner
    uint64_t attenMaskOffset;    // vector1 mask的 offset，后续更名为???
    uint64_t attenMaskOffsetPre; // vector1 band场景 第二个mask的 offset，后续更名为???

    uint32_t pseShiftCopyInCol; // vector1 s2方向非尾块时等于singleProcessSInnerSize，尾块时等于pseShiftInnerTailAlign
    uint32_t pseShiftPadSize; // vector1 pseShiftInnerTailAlign - unalignSInner
    uint64_t pseShiftOffset;  // vector1 pse 的 offset

    int64_t preTokensPerBatch;       // vector2 左上顶点的pretoken
    int64_t nextTokensPerBatch;      // vector2 左上顶点的nexttoken
    int64_t nextTokensOfMlaPerBatch; // vector2 在mla场景下左上顶点的nexttoken，用于计算BNSD的行无效
    int64_t actualSeqLengthPerBatch; // vector2 Q的actualSeqLength
    int64_t actualSeqLengthOfMlaPerBatch; // vector2 在mla场景下Q的actualSeqLength
    int64_t actualSeqLengthKVPerBatch;    // vector2 KV的actualSeqLength
    int64_t sOuterOffset;                 // vector2 单个S内 souter的 souterIdx * singleProcessSOuterSize

    // 伪量化相关
    int32_t antiKVRowLoopTimes; // 新增：antiquantKV循环搬入次数
    int32_t antiKVRowTailSize;  // 新增：antiquantKV尾块搬入行数

    // lse 输出offset
    int64_t softmaxLseOffset;
};

template <>
struct RunInfo<false> {
    int64_t s2StartIdx; // s2的起始位置，sparse场景下可能不是0
    int64_t s2EndIdx;
    int64_t s2LoopCount; // s2循环当前的循环index
    int64_t s2LoopLimit;
    int64_t s1oIdx = 0; // s1轴的index
    int64_t boIdx = 0;  // b轴的index
    int64_t n2oIdx = 0; // n2轴的index
    int64_t goIdx = 0;  // g轴的index
    int32_t s1RealSize;
    int32_t s1RealSizeAlign32;
    int32_t halfS1RealSize; // vector侧实际的s1基本块大小，如果Cube基本块=128，那么halfS1RealSize=64
    int32_t
        firstHalfS1RealSize; // 当s1RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s1 size
    int32_t s2RealSize;     // s2方向基本块的真实长度
    int32_t s2VecSize;      // s2方向本次参与vec计算的长度
    int64_t s2AlignedSize;  // s2方向基本块对齐到16之后的长度
    int32_t vec2S1BaseSize; // vector2侧开循环之后，经过切分的S1大小，例如把64切分成两份32。
    int32_t
        vec2S1RealSize; // vector2侧开循环之后，经过切分的S1的尾块大小，例如把63切分成两份32和31，第二份的实际大小是31。
    int64_t vecCoreOffset; // vec核基于cube核起始处s1方向偏移
    int64_t vecS1Offset;   // vec核基于cube核起始处s1方向偏移
    int64_t vecS2Offset;   // vec核基于cube核起始处s1方向偏移
    int64_t actualS1Size;  // Q的actualSeqLength
    int64_t actualS2Size;  // KV的actualSeqLength

    int64_t taskId;
    int64_t multiCoreInnerIdx = 0;

    int64_t qOffset;
    int64_t attentionOutOffset;
    int64_t
        s1SizeAcc; // 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）
    int64_t
        s2SizeAcc; // 对于非TND场景 = boIdx * pseInfo.s2Size; TND场景等于前面boIdx个batch的s2之和（每个batch的s2不同）
    int64_t s1SizeThisBatch; // 非TND场景=总s1Size, Tnd场景下当前batch对应的s1
    int64_t s2SizeThisBatch; // 非TND场景=总s2Size, Tnd场景下当前batch对应的s2
    int64_t b1SSOffset;      // 非TND = boIdx * s1 * s2; TND = 前面boIdx个batch的s1*S2之和。
    int64_t b1SSOffsetAlign; // TND场景s2 16对齐之后，前面batch的s1*s2之和
    int64_t deScaleQOffset;  // Q的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, G, Ceil(S1, 128), 1]
    int64_t deScaleKvOffset; // KV的反量化scale内容在Gm中的偏移 原始shape为 [B, N2, 1, Ceil(S2, 128), 1]
    float descaleQK = 1.0f;  // deScaleQValue * deScaleKValue
    float deSCaleVValue;
    uint8_t taskIdMod2;
    uint8_t taskIdMod3;
    uint8_t multiCoreIdxMod2 = 0;
    uint8_t multiCoreIdxMod3 = 0;
};

template <bool isInfer = false>
struct ConstInfo;

template <>
struct ConstInfo<true> {
    /* 全局的基本块信息 */
    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    int64_t dSize;
    int64_t dSizeV;
    int64_t gSize; // g轴的大小
    int64_t n2Size;
    int64_t s1Size; // s1总大小
    int64_t s2Size; // s2总大小
    int64_t s1BlockCnt;
    int64_t s2BlockCnt;
    int64_t
        attenMaskS2Size; // attention mask的s2的总大小，TND场景下，且format=attenMaskTT的时候等于所有batch中的最大的s2
    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1Dv;
    int64_t gS1Dv;
    int64_t n2GS1Dv;
    int64_t s2Dv;
    int64_t n2S2Dv;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t bN2D;
    int64_t gDv;
    int64_t n2Dv;
    int64_t bN2Dv;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
    int64_t n2GDv;
    int64_t bN2GDv;
    int64_t gS2;

    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseD;
    int64_t s2BaseN2Dv;
    int64_t s2BaseBN2Dv;
    int64_t s1BaseN2GDv;
    int64_t s1BaseBN2GDv;
    int64_t s1BaseDv;
    int64_t s2BaseDv;
    int64_t s1OuterSize;

    // matmul跳读参数
    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    // dq 或者attentionOut的Stride
    int64_t attentionOutStride;

    uint32_t subBlockIdx;
    uint32_t layoutType;

    float scaleValue;

    /* 推理新增 */
    int64_t aicIdx;
    int32_t outerLoopStart; // 分核后，单个核S1方向的开始idx
    int32_t outerLoopEnd;   // 分核后，单个核S1方向的结束idx

    int64_t preTokens;  // 用户输入的preToken
    int64_t nextTokens; // 用户输入的nextTokens

    bool isRowInvalid; // 是否使能行无效

    uint32_t actualSeqLenSize;   // 用户输入的actualseq的长度
    uint32_t actualSeqLenKVSize; // 用户输入的actualseq_kv的长度
    bool isActualLenDimsNull;    // 判断是否有actualseq
    bool isActualLenDimsKVNull;  // 判断是否有actualseq_kv
    uint32_t isKvContinuous;     // 是否为tensorlist

    uint32_t n1RopeD; // n * ropeD
    uint32_t n2RopeD; // n * ropeD / headNumRatio

    int32_t attentionMaskType; // sparse 类型
    uint32_t isMaskMultiBatch; // attenMask 是否为多 batch
    uint32_t maskQsSize;       // mask S1

    // service mm1 mm2 pageAttention
    uint32_t blockTableDim2;
    uint32_t blockSize;
    uint32_t paLayoutType;
    uint32_t paBlockNumSum;

    uint32_t pseShiftTypeByteNum;

    // service vector1
    uint32_t typeByteNum; // BYTE_BLOCK(32) / datatypesize
    // service vector2
    uint32_t outputTypeByteNum; // BYTE_BLOCK(32) / outputDataTypeSize;
    uint32_t isBSNDOut;         // LAYOUT是否为BNSD_BSND

    // B N1 N2 G S1 S2长度
    uint32_t headNumRatio; // G，GQA场景，每组下面有多少个S*D，MHA场景下为1，后续更名为gSize
    uint32_t gOfMla;       // 在mla场景下headNumRatio为1，gOfMla为真实g
    uint32_t ropeDSize;    // S1，Q sequence长度，后续更名为s1Size

    // q k v attenMask不同轴的stride
    uint32_t maskKVsSize;
    uint64_t rStride;

    // 左padding
    bool isQHasLeftPadding;
    bool isKVHasLeftPadding;
    int64_t queryRightPaddingSize;
    int64_t kvRightPaddingSize;

    // lse输出
    bool isSoftmaxLseEnable;
    int64_t totalSoftmaxLseOutputSize;

    // IFA GS合轴
    bool isGqa;
};

template <>
struct ConstInfo<false> {
    /* 全局的基本块信息 */
    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    int64_t dSize;
    int64_t dSizeV;
    int64_t gSize; // g轴的大小
    int64_t n2Size;
    int64_t s1Size; // s1总大小
    int64_t s2Size; // s2总大小
    int64_t s1BlockCnt;
    int64_t s2BlockCnt;
    int64_t
        attenMaskS2Size; // attention mask的s2的总大小，TND场景下，且format=attenMaskTT的时候等于所有batch中的最大的s2
    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t s1D;
    int64_t gS1D;
    int64_t n2GS1D;
    int64_t s2D;
    int64_t n2S2D;
    int64_t s1Dv;
    int64_t gS1Dv;
    int64_t n2GS1Dv;
    int64_t s2Dv;
    int64_t n2S2Dv;
    int64_t s1S2;
    int64_t gS1S2;
    int64_t gS1;
    int64_t n2GS1;
    int64_t gD;
    int64_t n2D;
    int64_t bN2D;
    int64_t gDv;
    int64_t n2Dv;
    int64_t bN2Dv;
    int64_t n2G;
    int64_t n2GD;
    int64_t bN2GD;
    int64_t n2GDv;
    int64_t bN2GDv;
    int64_t gS2;

    int64_t s2BaseN2D;
    int64_t s2BaseBN2D;
    int64_t s1BaseN2GD;
    int64_t s1BaseBN2GD;
    int64_t s1BaseD;
    int64_t s2BaseD;
    int64_t s2BaseN2Dv;
    int64_t s2BaseBN2Dv;
    int64_t s1BaseN2GDv;
    int64_t s1BaseBN2GDv;
    int64_t s1BaseDv;
    int64_t s2BaseDv;
    int64_t s1OuterSize;

    // matmul跳读参数
    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;
    // dq 或者attentionOut的Stride
    int64_t attentionOutStride;

    uint32_t subBlockIdx;
    uint32_t layoutType;
    float scaleValue;
    uint32_t s1VecBaseSize; // david121 fag vecbase
};
} // namespace davidutil

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
