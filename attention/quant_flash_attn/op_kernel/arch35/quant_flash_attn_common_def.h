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
 * \file quant_flash_attn_common_def.h
 * \brief QuantFlashAttn算子tiling和kernel共用的相关定义
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_H_

static constexpr uint32_t FIA_SYNC_MODE2 = 2;
static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
static constexpr float FLOAT_ZERO = 0;
static constexpr float FLOAT_MAX = 3.402823466e+38F;
static constexpr float FLOAT_INF = 3e+99;

static constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);
static constexpr uint32_t QFA_HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

enum class FA_LAYOUT : uint32_t {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

enum class inferFaLayOutTypeEnum {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
    LAYOUT_NTD_TND = 5,
    LAYOUT_NTD = 6
};

enum class inferDTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned72 = 72,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    NotAligned,
};

enum class inferS1TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    NotAligned,
};

enum class inferS2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

// q_out layout → (inputLayout, outputLayout)
//   0=BSND → (BSH, BSH)
//   1=BNSD → (BNSD, BNSD)
//   2=TND  → (TND, TND)
//   3=NTD  → (NTD, TND)
static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[4][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
    {inferFaLayOutTypeEnum::LAYOUT_NTD, inferFaLayOutTypeEnum::LAYOUT_TND},
};

// q_out layout
//   0: BSND (BSND排布与BSH一样)
//   1: BNSD
//   2: TND
//   3: NTD
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
#define InOutLayoutType_NTD 3
// backward compat
#define InOutLayoutType_BSND_BSND InOutLayoutType_BSND
#define InOutLayoutType_BNSD_BNSD InOutLayoutType_BNSD
#define InOutLayoutType_TND_TND InOutLayoutType_TND
#define InOutLayoutType_NTD_TND InOutLayoutType_NTD

#define KvLayoutType_NO_PA 0
#define KvLayoutType_PA_BBND 1
#define KvLayoutType_PA_BNBD 2
#define KvLayoutType_PA_NZ 3

struct ConfigParams {
    inferS1TemplateType s1;
    inferS2TemplateType s2;
    inferDTemplateType d;
    inferDTemplateType dv;
};

static constexpr ConfigParams ConfigValue[] = {
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned64,
     inferDTemplateType::Aligned64},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned128,
     inferDTemplateType::Aligned128},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256, inferDTemplateType::Aligned256,
     inferDTemplateType::Aligned256},
    {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned512, inferDTemplateType::Aligned72,
     inferDTemplateType::Aligned72},
};

#define Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64 0
#define Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128 1
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 2
#define Config_S1Aligned128_S2Aligned256_DAligned256_DVAligned256 3
#define Config_S1Aligned128_S2Aligned512_DAligned72_DVAligned72 4

#define QFA_MXFP8_FP32_PREFILL 1
#define QFA_MXFP8_FP32_DECODE 2
#define QFA_GQA_FP8_FULLQUANT 6
#define QFA_HIF8_FP32 0

#define false 0
#define true 1

struct FDparamsX {
    uint32_t fdCoreEnable;
    uint32_t fdBN2Idx;
    uint32_t fdMIdx;
    uint32_t fdS2SplitNum;
    uint32_t mStart;
    uint32_t mLen;
    uint32_t fdWorkspaceIdx;
};

struct RunInfoX {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t realN2Idx = 0;
    uint64_t actS1Size = 1;
    uint64_t actS2Size = 1;
    uint32_t actMSize = 0;
    uint32_t actMSizeAlign32;
    uint32_t actVecMSize;
    uint32_t vecMbaseIdx;

    uint32_t actSingleLoopS2Size = 0;
    uint32_t actSingleLoopS2SizeAlign;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0;

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

struct StridesConstInfo {
    uint64_t bnStride = 0;
    uint64_t n2Stride = 0;
};

struct CommonConstInfo {
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t gSize;
    uint32_t n2Size;
    uint32_t realGSize;
    uint32_t realN2Size;
    uint64_t s1Size;
    uint64_t s2Size;
    uint64_t cuSeqLensQSize;
    uint64_t cuSeqLensKVSize;
    uint64_t seqUsedQSize;
    uint64_t seqUsedKvSize;

    uint32_t sparseMode;
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    float scaleValue;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    FA_LAYOUT outputLayout;
    bool needInitOutput;

    StridesConstInfo keyStrides;
    StridesConstInfo valueStrides;
    StridesConstInfo kDescaleStrides;
    StridesConstInfo vDescaleStrides;
};

struct PAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
};

struct LseConstInfo {
    bool isSoftmaxLseEnable;
};

struct SinkConstInfo {
    bool learnableSinkFlag = false;
};

struct TensorListConstInfo {
    bool isKvContinuous;
};

struct ConstInfo_t : CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo, TensorListConstInfo {};

#endif // QUANT_FLASH_ATTN_COMMON_DEF_H_
