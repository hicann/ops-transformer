/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEMORY_COPY_ARCH92_QUANT_FLASH_ATTN_H
#define MEMORY_COPY_ARCH92_QUANT_FLASH_ATTN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memcopy/gm_layout.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/memcopy/offset_calculator_v2.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_l1_tensor.h"
#include "../../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../../common/op_kernel/memcopy/gm_coord.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_ub.h"
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"
#include "../arch35/memory_copy_arch35_quant_flash_attn.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;

// ----------------------------------------------FIA public define --------------------------------
namespace AttentionCommon {
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

// 将isCheckTiling设置为false
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG = {false};
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class FIA_LAYOUT : uint32_t {
    BSH = 0,
    BSND = 0,
    BNSD = 1,
    NZ = 2,
    TND = 3,
    NBSD = 4,
    NTD = 5
};

template <typename Q_T, typename KV_T, typename OUT_T, typename ORIGIN_T, const bool PAGE_ATTENTION = false,
          const bool FLASH_DECODE = false, FIA_LAYOUT LAYOUT_T = FIA_LAYOUT::BSH, const uint8_t ANTIQUANT_MODE = 0,
          const bool SHARED_PREFIX = false, FIA_LAYOUT KV_LAYOUT_T = FIA_LAYOUT::BSH,
          const bool SOFTMAX_WITH_BRC = false, const bool ENABLE_TREE = false, typename... Args>
struct FIAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    using orginalType = ORIGIN_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr FIA_LAYOUT layout = LAYOUT_T;
    static constexpr uint8_t antiquantMode = ANTIQUANT_MODE;
    static constexpr bool sharedPrefix = SHARED_PREFIX;
    static constexpr FIA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool softmaxWithBrc = SOFTMAX_WITH_BRC;
    static constexpr bool enableTree = ENABLE_TREE;
};

struct FDparams {
    uint32_t* bN2IdxOfFdHead;
    uint32_t* gS1IdxOfFdHead;
    uint32_t* s2SplitNumOfFdHead;
    uint32_t* gS1SplitNumOfFdHead;
    uint32_t* gS1LastPartSizeOfFdHead;
    uint32_t* gS1IdxEndOfFdHead;
    uint32_t* gS1IdxEndOfFdHeadSplit;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
};

struct RunInfo {
    bool isValid = false;
    bool isChangeBatch = false;
    bool isFirstSInnerLoop = false;
    bool isLastS2Loop = false;
    uint32_t loop = 0;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1; // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1; // 当前处理head的S2轴实际大小

    uint32_t actMBaseSize = 0;
    uint32_t actualSingleProcessSInnerSize = 0;
    uint32_t actualSingleProcessSInnerSizeAlign = 0;

    uint32_t curSInnerLoopTimes = 0;
    uint32_t bn2IdxInCurCore = 0;
    uint32_t s2BatchOffset = 0;
    uint32_t tndIsS2SplitCore = 0;
    uint32_t tndCoreStartKVSplitPos = 0;

    uint64_t tensorAOffset = 0;
    uint64_t tensorBOffset = 0;
    uint64_t tensorARopeOffset = 0;
    uint64_t tensorBRopeOffset = 0;
    uint64_t attenOutOffset = 0;
    uint64_t attenMaskOffset = 0;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;
    uint64_t accumTmpOutNum = 0;

    uint64_t qPaddingBeginOffset = 0;
    uint64_t kvPaddingBeginOffset = 0;
};

#define HAS_ATTENTION_COMMON_CONST_INFO 1
struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
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
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;
    static constexpr float FLOAT_INF = 3e+99;
    // 整个AICORE的任务信息, 左闭右开区间[ (bN2Start, gS1Start, s2Start), (bN2End, gS1End, s2End) )
    uint32_t bN2Start = 0U;
    uint32_t gS1Start = 0U;
    uint32_t s2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1End = 0U;
    uint32_t s2End = 0U;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    float scaleValue = 0;
    uint32_t mmResUbSize = 0U;   // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U; // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U; // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum = 0ULL;
    uint64_t headDim = 0;
    uint64_t headDimRope = 0;
    uint64_t headDimAlign = 0;
    uint64_t kvSeqSize = 0ULL; // kv最大S长度
    uint64_t qSeqSize = 1ULL;  // q最大S长度
    int64_t preToken = 0;
    int64_t nextToken = 0;
    uint64_t systemPrefixMaxLen = 0;
    uint64_t attenMaskBatchStride = 0ULL;
    uint64_t qLeftPaddingSize = 0;
    uint64_t kvLeftPaddingSize = 0;
    uint32_t kvCacheBlockSize = 0;    // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    FIA_LAYOUT outputLayout;          // 输出的Transpose格式
    uint32_t systemPrefixLen = 0;
    uint32_t subBlockNum = 2; // AI Core上 AIC与AIV的数量默认为1:2

    uint32_t pseShiftS1 = 0U;
    uint32_t pseShiftS2 = 0U;
    uint32_t attenMaskStride = 0ULL;
    uint32_t sparseMode = 0;
    uint32_t tndFDCoreArrLen = 0U;     // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U; // TNDFlashDecoding kv起始位置

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;
    uint32_t l2CacheOffFlag = 0;

    bool attenMaskFlag = false;
    bool accumQSeqFlag = false;
    bool accumKVSeqFlag = false;
    bool needInit = false;
    bool isRowInvalid = false;      // 是否使能行无效
    bool isExistRowInvalid = false; // 实际是否存在行无效

    bool batchContinuous = true;
    bool ropeSplitMode = false;

    bool pseShiftFlag = false;
    bool pseShiftByBatch = false;

    bool softmaxLseFlag = false;
    bool isLegacyIfa = false;

    bool isQHasLeftPadding = false;
    bool isKVHasLeftPadding = false;

    bool headS2Split = false;
    bool tailS2Split = false;
    bool systemPrefixFlag = false;

    bool isPostQuantPerChn = false;
    bool isPostQuantTypeBf16 = false;
};

struct FusedTransposeInfo {
    // 以下是FlashDecode分支区分的信息
    uint32_t n2Idx = 0;
    uint32_t bIdx = 0;

    // 以下是需要用公式计算的信息
    uint32_t s1StartIdx = 0;
    uint32_t s1EndIdx = 0;
    uint32_t s1Count = 0;
    uint32_t gStartIdx = 0;
    uint32_t gEndIdx = 0;
    uint32_t gCount = 0;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline void GetGS1Idx(uint32_t gS1Idx, uint32_t& gIdx, uint32_t& s1Idx,
                                 AttentionCommon::ConstInfo& constInfo)
{
    // GS1
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NBSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        gIdx = gS1Idx / constInfo.qSeqSize;
        s1Idx = gS1Idx % constInfo.qSeqSize;
    } else {
        // S1G
        s1Idx = gS1Idx / constInfo.gSize;
        gIdx = gS1Idx % constInfo.gSize;
    }
}

__aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}
} // namespace AttentionCommon

// ----------------------------------------------FIA_LAYOUT format conversion --------------------------------
template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}
template <FIA_LAYOUT LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    }
}

template <FIA_LAYOUT KV_LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr GmFormat GetKVFormat()
{
    if constexpr (PAGE_ATTENTION) {
        static_assert(
            (KV_LAYOUT_T == FIA_LAYOUT::BSH) || (KV_LAYOUT_T == FIA_LAYOUT::BNSD) || (KV_LAYOUT_T == FIA_LAYOUT::NZ),
            "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when PageAttention");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NZ) {
            return GmFormat::PA_NZ;
        }
    } else {
        static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) || (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                          (KV_LAYOUT_T == FIA_LAYOUT::TND) || (KV_LAYOUT_T == FIA_LAYOUT::NTD),
                      "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when KV Continuous or TensorList");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::BSND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::TND) {
            return GmFormat::TND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NTD) {
            return GmFormat::NTD;
        }
    }
}

template <FIA_LAYOUT OUT_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetOutGmFormat()
{
    static_assert((OUT_LAYOUT_T == FIA_LAYOUT::BSH) || (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                      (OUT_LAYOUT_T == FIA_LAYOUT::TND) || (OUT_LAYOUT_T == FIA_LAYOUT::NTD) ||
                      (OUT_LAYOUT_T == FIA_LAYOUT::NBSD),
                  "Get OutAttention GmFormat fail, OUT_LAYOUT_T is incorrect");
    if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NBSD) {
        return GmFormat::NGBSD;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr bool IsSupportPse()
{
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::BSH) {
        return true;
    } else {
        return false;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetPseUbFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get PSE UbFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    } else {
        return UbFormat::S1G;
    }
}

// ---------------------------------------Set attention Gm To Zero--------------------------------
template <GmFormat FORMAT, typename OUT_T, typename OffsetCalcType>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, OffsetCalcType& offsetCalculator,
                                           GlobalTensor<OUT_T>& attentionOutGm)
{
    if constexpr (FORMAT == GmFormat::TNGD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetFullActualSeqLength(bIdx);
        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::NGTD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetFullActualSeqLength(bIdx);
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * offsetCalculator.GetDimD(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::BNGSD) {
        uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, 0, 0);
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
    } else if constexpr (FORMAT == GmFormat::BSNGD) {
        uint32_t s1Size = offsetCalculator.GetDimS1();
        for (int s1Idx = 0; s1Idx < s1Size; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::NGBSD) {
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideB(), 0);
        }
    }
}

// ----------------------------------------------LayOutTypeEnum format conversion --------------------------------
template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetOutGmFormat()
{
    static_assert((LAYOUT == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT == LayOutTypeEnum::LAYOUT_NTD) ||
                      (LAYOUT == LayOutTypeEnum::LAYOUT_NBSD),
                  "Get OutAttention GmFormat fail, OUT_LAYOUT_T is incorrect");
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_NTD) {
        return GmFormat::NGTD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_NBSD) {
        return GmFormat::NGBSD;
    }
}

template <LayOutTypeEnum LAYOUT_T>
__aicore__ inline constexpr UbFormat GetPseUbFormat()
{
    static_assert((LAYOUT_T == LayOutTypeEnum::LAYOUT_BSH) || (LAYOUT_T == LayOutTypeEnum::LAYOUT_BNSD) ||
                      (LAYOUT_T == LayOutTypeEnum::LAYOUT_TND) || (LAYOUT_T == LayOutTypeEnum::LAYOUT_NTD),
                  "Get PSE UbFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT_T == LayOutTypeEnum::LAYOUT_NTD) {
        return UbFormat::GS1;
    } else {
        return UbFormat::S1G;
    }
}

template <LayOutTypeEnum LAYOUT_T>
__aicore__ inline constexpr bool IsSupportPse()
{
    if constexpr (LAYOUT_T == LayOutTypeEnum::LAYOUT_BNSD || LAYOUT_T == LayOutTypeEnum::LAYOUT_BSH) {
        return true;
    } else {
        return false;
    }
}

// ----------------------------------------------Copy LSE UB To Gm --------------------------------
template <typename T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyLseUbToGm {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::S1G && GM_FORMAT == GmFormat::LSE_BN2GS1) {
            CopyBSND(softmaxLseGm, lseSrc, gmCoord);
        } else if constexpr (UB_FORMAT == UbFormat::GS1 && GM_FORMAT == GmFormat::LSE_BN2GS1) {
            CopyBNSD(softmaxLseGm, lseSrc, gmCoord);
        } else if constexpr (UB_FORMAT == UbFormat::S1G && GM_FORMAT == GmFormat::LSE_TN2G) {
            CopyTND(softmaxLseGm, lseSrc, gmCoord);
        } else if constexpr (UB_FORMAT == UbFormat::S1G && GM_FORMAT == GmFormat::LSE_N2GT) {
            CopyTNDtoNT(softmaxLseGm, lseSrc, gmCoord);
        } else if constexpr (UB_FORMAT == UbFormat::GS1 && GM_FORMAT == GmFormat::LSE_N2GT) {
            CopyNTD(softmaxLseGm, lseSrc, gmCoord);
        }
    }

private:
    template <typename FaGmTensorType>
    __aicore__ inline void CopyBSND(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = softmaxLseGm.offsetCalculator;
        uint32_t s1Size = offsetCalculator.GetDimS1();
        uint64_t gSize = offsetCalculator.GetDimG();

        uint32_t startS1Idx = gmCoord.gS1Idx / gSize;
        uint32_t startGIdx = gmCoord.gS1Idx % gSize;
        uint32_t endS1Idx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) / gSize;
        uint32_t endGIdx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) % gSize;
        uint64_t ubOffset = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            uint64_t outOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, startGIdx, s1Idx);
            uint32_t curDealRowCount = (s1Idx != endS1Idx) ? (gSize - startGIdx) : (endGIdx + 1 - startGIdx);

            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (s1Size - 1) * sizeof(T);

            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }

    template <typename FaGmTensorType>
    __aicore__ inline void CopyBNSD(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = softmaxLseGm.offsetCalculator;

        uint64_t qActSeqLens = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        ;
        uint64_t s1Size = offsetCalculator.GetDimS1();
        uint64_t startGIdx = gmCoord.gS1Idx / qActSeqLens;
        uint64_t startS1Idx = gmCoord.gS1Idx % qActSeqLens;
        uint64_t outOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, startGIdx, startS1Idx);
        uint64_t ubOffset = 0;

        if ((qActSeqLens - startS1Idx) >= gmCoord.gS1DealSize) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = gmCoord.gS1DealSize;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            return;
        }
        // gmCoord.gS1DealSize > 当前actQs剩余部分，分块搬运: 首块 + 中间块 + 尾块
        uint64_t headActSeq = qActSeqLens - startS1Idx;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = headActSeq;
        dataCopyParams.blockLen = sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += s1Size - qActSeqLens + headActSeq;
        ubOffset += headActSeq * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        uint64_t pendingCount = gmCoord.gS1DealSize - headActSeq;
        while (pendingCount > qActSeqLens) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = qActSeqLens;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            outOffset += s1Size;
            ubOffset += qActSeqLens * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
            pendingCount -= qActSeqLens;
        }
        if (pendingCount > 0) {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = pendingCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
        }
    }

    template <typename FaGmTensorType>
    __aicore__ inline void CopyTND(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = softmaxLseGm.offsetCalculator;
        uint64_t gSize = offsetCalculator.GetDimG();
        uint32_t startS1Idx = gmCoord.gS1Idx / gSize;
        uint32_t startGIdx = gmCoord.gS1Idx % gSize;
        uint32_t endS1Idx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) / gSize;
        uint32_t endGIdx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) % gSize;
        uint64_t ubOffset = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            uint64_t outOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, startGIdx, s1Idx);
            uint32_t curDealRowCount = (s1Idx != endS1Idx) ? (gSize - startGIdx) : (endGIdx + 1 - startGIdx);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }

    // TND layout LSE 输出转成 N-major (N 在外, T 在内) 排布: [N2*G, T]
    // bN2Offset = n2Idx * realGSize * t1Size + prefixBS1
    // 单次写入: outOffset = bN2Offset + gIdx * t1Size + s1Idx
    //         = n2Idx * realGSize * t1Size + prefixBS1 + gIdx * t1Size + s1Idx
    // 即最终 GM 排布为 [N2, G, T], 与 host infershape 声明的 (N, T) 对齐 (N=N2*G)
    template <typename FaGmTensorType>
    __aicore__ inline void CopyTNDtoNT(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = softmaxLseGm.offsetCalculator;
        uint64_t gSize = offsetCalculator.GetDimG();
        uint64_t t1Size = offsetCalculator.GetDimT();
        uint32_t startS1Idx = gmCoord.gS1Idx / gSize;
        uint32_t startGIdx = gmCoord.gS1Idx % gSize;
        uint32_t endS1Idx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) / gSize;
        uint32_t endGIdx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) % gSize;
        uint64_t ubOffset = 0;

        for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
            uint64_t outOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, startGIdx, s1Idx);
            uint32_t curDealRowCount = (s1Idx != endS1Idx) ? (gSize - startGIdx) : (endGIdx + 1 - startGIdx);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (t1Size - 1) * sizeof(T);
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            startGIdx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }

    template <typename FaGmTensorType>
    __aicore__ inline void CopyNTD(FaGmTensorType& softmaxLseGm, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = softmaxLseGm.offsetCalculator;
        uint32_t s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        uint64_t gSize = offsetCalculator.GetDimG();
        uint64_t n2Size = offsetCalculator.GetDimN2();

        uint32_t startS1Idx = gmCoord.gS1Idx % s1Size;
        uint32_t startGIdx = gmCoord.gS1Idx / s1Size;
        uint32_t endS1Idx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) % s1Size;
        uint32_t endGIdx = (gmCoord.gS1Idx + gmCoord.gS1DealSize - 1) / s1Size;
        uint64_t ubOffset = 0;

        for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
            uint64_t outOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdx, startS1Idx);
            uint32_t curDealRowCount = (gIdx != endGIdx) ? (s1Size - startS1Idx) : (endS1Idx + 1 - startS1Idx);
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = curDealRowCount;
            dataCopyParams.blockLen = sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (gSize * n2Size - 1) * sizeof(T);
            DataCopyPad(softmaxLseGm.gmTensor[outOffset], lseSrc[ubOffset], dataCopyParams);
            startS1Idx = 0;
            ubOffset += curDealRowCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }
};
#endif // MEMORY_COPY_ARCH92_QUANT_FLASH_ATTN_H
