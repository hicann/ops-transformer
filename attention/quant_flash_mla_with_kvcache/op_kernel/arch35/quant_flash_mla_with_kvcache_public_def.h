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
 * \file quant_flash_mla_with_kvcache_public_def.h
 * \brief QuantFlashMlaWithKvcache kernel公共定义（RunInfoX/ConstInfo等，自FIA MLA裁剪适配）
 */

#ifndef QUANT_FLASH_MLA_WITH_KVCACHE_PUBLIC_DEF_H_
#define QUANT_FLASH_MLA_WITH_KVCACHE_PUBLIC_DEF_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "util.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"

namespace AttentionCommon {

// MLA输出layout（值域与FIA_LAYOUT一致，便于复用公共CopyOut逻辑）
enum class QMLA_LAYOUT : uint32_t {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

// 输出layout（tiling下发值）→ kernel内部QMLA_LAYOUT映射
// tiling QmlaOutLayout: BSND=0, BNSD=1, TND=2, NTD=3
__aicore__ inline QMLA_LAYOUT ConvertToQmlaKernelLayout(uint32_t layoutOut)
{
    switch (layoutOut) {
        case 1U: // BNSD
            return QMLA_LAYOUT::BNSD;
        case 2U: // TND
            return QMLA_LAYOUT::TND;
        case 3U: // NTD
            return QMLA_LAYOUT::NTD;
        case 0U: // BSND
        default:
            return QMLA_LAYOUT::BSND;
    }
}

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
    uint32_t s2LocalIdx = 0; // 每核本地 S2 索引，从0开始累加，用于判断本核内 S2 的第几个 base 块
    uint32_t realN2Idx = 0;   // GS1合轴时为n2Idx，不合轴时为n1Idx
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;     // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;     // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;
    uint32_t kvL1BufId = 0; // bmm1加载K时使用的L1 buffer id, bmm2复用V时直接使用, 避免轮转推算错位
    bool keyPrefetched = false; // K是否已由IterateBmm1Load提前加载到L1, 未预取时bmm1现场加载
    bool qPreloaded = false;    // Q是否已由IterateQPreload提前加载到L1, 未预取时bmm1现场加载
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块

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
    /* 轴长度 */
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;     // head_dim_qk = 576 (nope 512 + rope 64)
    uint32_t dSizeV;    // head_dim_v = 512
    uint32_t dSizeRope; // rope维度 = dSize - dSizeV = 64
    uint32_t dBasicBlock;
    uint32_t gSize;  /* MLA下g轴承载Q头数: n1Size / n2Size */
    uint32_t n2Size; /* KV_N, MLA固定为1 */
    uint32_t realGSize;
    uint32_t realN2Size;
    uint64_t s1Size;
    uint64_t s2Size;
    uint32_t cuSeqLensQSize; // cu_seqlens_q元素个数(B+1, 含前导0), 0表示未传入
    uint32_t seqUsedQSize;   // seqused_q元素个数(B), 0表示未传入
    uint32_t kvSeqUsedSize;  // cache_seqlens元素个数(B), MLA场景必传

    /* strides */
    StridesConstInfo keyStrides;
    StridesConstInfo valueStrides;
    StridesConstInfo kRopeStrides;

    /* mask */
    uint32_t sparseMode; // 0: NO_MASK 3: CAUSAL(RIGHT_DOWN_CAUSAL)
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    float scaleValue;

    /* 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    /* 输出shape */
    QMLA_LAYOUT outputLayout;
    bool needInitOutput;

    /* FlashDecode: metadata header统一标志, 所有核读取同一值, 决定是否走FD流程 */
    bool enableFlashDecode;
};

/* Paged Attention */
struct PAConstInfo {
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
};

/* SoftmaxLse */
struct LseConstInfo {
    bool isSoftmaxLseEnable;
};

struct QmlaConstInfo : CommonConstInfo, PAConstInfo, LseConstInfo {};

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

} // namespace AttentionCommon

#endif // QUANT_FLASH_MLA_WITH_KVCACHE_PUBLIC_DEF_H_
