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
 * \file quant_flash_attn_common_def_arch92.h
 * \brief QuantFlashAttn算子tiling和kernel共用的相关定义
 */

#ifndef QUANT_FLASH_ATTN_COMMON_DEF_ARCH92_H_
#define QUANT_FLASH_ATTN_COMMON_DEF_ARCH92_H_

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

static constexpr uint32_t QFA_HEAD_NEED_INIT_OUTPUT_INDEX = 15U;
// metadata 头第 1 项与 AICPU 的 HEAD_IS_FD_INDEX 保持一致，表示本次调度实际采用 FD。
static constexpr uint32_t QFA_HEAD_IS_FD_INDEX = 1U;

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
static constexpr inferFaLayOutTypeEnum InOutLayoutTypeValue[3][2] = {
    {inferFaLayOutTypeEnum::LAYOUT_BSH, inferFaLayOutTypeEnum::LAYOUT_BSH},
    {inferFaLayOutTypeEnum::LAYOUT_BNSD, inferFaLayOutTypeEnum::LAYOUT_BNSD},
    {inferFaLayOutTypeEnum::LAYOUT_TND, inferFaLayOutTypeEnum::LAYOUT_TND},
};

// q_out layout
//   0: BSND (BSND排布与BSH一样)
//   1: BNSD
//   2: TND
#define InOutLayoutType_BSND 0
#define InOutLayoutType_BNSD 1
#define InOutLayoutType_TND 2
// backward compat
#define InOutLayoutType_BSND_BSND InOutLayoutType_BSND
#define InOutLayoutType_BNSD_BNSD InOutLayoutType_BNSD
#define InOutLayoutType_TND_TND InOutLayoutType_TND

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

static constexpr ConfigParams ConfigValue[] = {{inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256,
                                                inferDTemplateType::Aligned64, inferDTemplateType::Aligned64},
                                               {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256,
                                                inferDTemplateType::Aligned72, inferDTemplateType::Aligned72},
                                               {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned256,
                                                inferDTemplateType::Aligned128, inferDTemplateType::Aligned128},
                                               {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned128,
                                                inferDTemplateType::Aligned256, inferDTemplateType::Aligned256},
                                               {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned512,
                                                inferDTemplateType::Aligned64, inferDTemplateType::Aligned64},
                                               {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned512,
                                                inferDTemplateType::Aligned72, inferDTemplateType::Aligned72},
                                               {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned512,
                                                inferDTemplateType::Aligned128, inferDTemplateType::Aligned128},
                                               {inferS1TemplateType::Aligned64, inferS2TemplateType::Aligned512,
                                                inferDTemplateType::Aligned256, inferDTemplateType::Aligned256},
                                               {inferS1TemplateType::Aligned128, inferS2TemplateType::Aligned1024,
                                                inferDTemplateType::Aligned128, inferDTemplateType::Aligned128}};

#define Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64 0
#define Config_S1Aligned128_S2Aligned256_DAligned72_DVAligned72 1
#define Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128 2
#define Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256 3
#define Config_S1Aligned64_S2Aligned512_DAligned64_DVAligned64 4
#define Config_S1Aligned64_S2Aligned512_DAligned72_DVAligned72 5
#define Config_S1Aligned64_S2Aligned512_DAligned128_DVAligned128 6
#define Config_S1Aligned64_S2Aligned512_DAligned256_DVAligned256 7
#define Config_S1Aligned128_S2Aligned1024_DAligned128_DVAligned128 8

#define QFA_MXFP8_FP32_PREFILL 1
#define QFA_MXFP8_FP32_DECODE 2
#define QFA_HIFP4_FP16_PREFILL 10

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
    uint32_t actMSizeAlign16;
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

// 普通 MXFP8 特化不包含尾任务状态。
static constexpr uint32_t QFA_VTAIL_SIZE = 64U;
static constexpr uint32_t QFA_MXFP_SCALE_GROUP_ELEMENTS = 32U;
static constexpr uint32_t QFA_L1_ALIGNMENT_BYTES = BUFFER_SIZE_BYTE_32B;
// BF16 使用 16 位存储；此头文件与主机侧共用，不能依赖内核的 bfloat16_t 类型。
static constexpr uint32_t QFA_BF16_ELEMENT_BYTES = sizeof(uint16_t);

// FD 每次最多处理 8 行；每行 sum/max 用 8 个 float 广播为一个 32 字节块。
static constexpr uint32_t QFA_FD_MAX_ROWS = 8U;
static constexpr uint32_t QFA_FD_LSE_BYTES = 6U * BUFFER_SIZE_BYTE_1K;
static constexpr uint32_t QFA_FD_BROADCAST_ELEMENTS = BUFFER_SIZE_BYTE_32B / sizeof(float);

struct RunInfoVTailX : RunInfoX {
    uint32_t kvTailLen = 0;
    bool isVtailTask = false;
    bool isLastDnTask = false;
};

template <uint32_t M, uint32_t N, bool HasVTail>
struct QfaPBufferLayout {
    // 保留既有 P 环形槽深度；这些槽数不适用于 softmax 状态或输出累计缓冲。
    static constexpr uint32_t NO_VTAIL_SLOT_COUNT = 3U;
    static constexpr uint32_t VTAIL_SLOT_COUNT = 4U;
    // 主段 P 及其 E8M0 量化系数均为每元素一字节。
    static constexpr uint32_t MAIN_BYTES = M * N;
    static constexpr uint32_t SCALE_BYTES = MAIN_BYTES / QFA_MXFP_SCALE_GROUP_ELEMENTS;
    static constexpr uint32_t TAIL_OFFSET = MAIN_BYTES + SCALE_BYTES;
    static constexpr uint32_t TAIL_BYTES = HasVTail ? M * QFA_VTAIL_SIZE * QFA_BF16_ELEMENT_BYTES : 0U;
    static constexpr uint32_t SLOT_BYTES = TAIL_OFFSET + TAIL_BYTES;
    static constexpr uint32_t SLOT_COUNT = HasVTail ? VTAIL_SLOT_COUNT : NO_VTAIL_SLOT_COUNT;
    static_assert(TAIL_OFFSET % QFA_L1_ALIGNMENT_BYTES == 0U, "P tail must be block aligned");
};

// ND 的 L1 静态地址规划；无尾布局与原实现逐字节一致。
template <uint32_t M, uint32_t N, uint32_t D, uint32_t DV, bool HasVTail>
struct QfaNdL1Layout {
    using P = QfaPBufferLayout<M, N, HasVTail>;
    static constexpr uint32_t Q_SLOT_COUNT = 2U;       // Q 数据及量化系数使用同一双缓冲索引。
    static constexpr uint32_t S2_SUB_BLOCK_COUNT = 2U; // ND 在内部将每个 S2 基本块拆成两半处理。
    static constexpr uint32_t S2_SUB_BLOCK_SIZE = N / S2_SUB_BLOCK_COUNT;
    static constexpr uint32_t D256_KV_SLOT_COUNT = 4U;
    static constexpr uint32_t OTHER_KV_SLOT_COUNT = 6U;
    static constexpr uint32_t P_BYTES = P::SLOT_COUNT * P::SLOT_BYTES;
    static constexpr uint32_t P_DATA_STRIDE = HasVTail ? P::SLOT_BYTES : P::MAIN_BYTES;
    static constexpr uint32_t P_SCALE_BASE = HasVTail ? P::MAIN_BYTES : P::SLOT_COUNT * P::MAIN_BYTES;
    static constexpr uint32_t P_SCALE_STRIDE = HasVTail ? P::SLOT_BYTES : P::SCALE_BYTES;
    static constexpr uint32_t Q_BASE = P_BYTES;
    static constexpr uint32_t Q_SLOT_BYTES = M * D;
    static constexpr uint32_t Q_SCALE_BASE = Q_BASE + Q_SLOT_COUNT * Q_SLOT_BYTES;
    static constexpr uint32_t Q_SCALE_SLOT_BYTES = Q_SLOT_BYTES / QFA_MXFP_SCALE_GROUP_ELEMENTS;
    static constexpr uint32_t KV_BASE = Q_SCALE_BASE + Q_SLOT_COUNT * Q_SCALE_SLOT_BYTES;
    static constexpr uint32_t MX_KV_BYTES = S2_SUB_BLOCK_SIZE * D;
    static constexpr uint32_t TAIL_V_BYTES = QFA_VTAIL_SIZE * DV * QFA_BF16_ELEMENT_BYTES;
    static constexpr uint32_t KV_SLOT_BYTES = HasVTail && TAIL_V_BYTES > MX_KV_BYTES ? TAIL_V_BYTES : MX_KV_BYTES;
    static constexpr uint32_t KV_SCALE_SLOT_BYTES = MX_KV_BYTES / QFA_MXFP_SCALE_GROUP_ELEMENTS;
    // 保留既有 KV 轮转及事件配对：D=256 使用四槽，其他 D 档使用六槽。
    static constexpr uint32_t KV_SLOTS =
        D == static_cast<uint32_t>(inferDTemplateType::Aligned256) ? D256_KV_SLOT_COUNT : OTHER_KV_SLOT_COUNT;
    static constexpr uint32_t KV_SCALE_BASE = KV_BASE + KV_SLOTS * KV_SLOT_BYTES;
    static constexpr uint32_t TOTAL_BYTES = KV_SCALE_BASE + KV_SLOTS * KV_SCALE_SLOT_BYTES;
    static_assert(Q_BASE % QFA_L1_ALIGNMENT_BYTES == 0U && Q_SCALE_BASE % QFA_L1_ALIGNMENT_BYTES == 0U &&
                      KV_BASE % QFA_L1_ALIGNMENT_BYTES == 0U && KV_SCALE_BASE % QFA_L1_ALIGNMENT_BYTES == 0U,
                  "ND L1 regions must be block aligned");
    static_assert(!HasVTail || TAIL_V_BYTES <= KV_SLOT_BYTES, "BF16 V tail must fit a KV slot");
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
    uint64_t maxSeqlenQ;
    uint64_t maxSeqlenKv;

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
    // 模板具备 FD 能力不等于本次任务启用 FD；该值来自 metadata 头。
    bool enableFlashDecode = false;
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

struct ConstInfo_t : CommonConstInfo, PAConstInfo, LseConstInfo, SinkConstInfo {};

struct ConstInfoVTail_t : ConstInfo_t {
    uint32_t seqUsedKvTailSize = 0;
    uint32_t tailMaxBlockNum = 0;
    StridesConstInfo vTailStrides;
};

#endif // QUANT_FLASH_ATTN_COMMON_DEF_ARCH92_H_
