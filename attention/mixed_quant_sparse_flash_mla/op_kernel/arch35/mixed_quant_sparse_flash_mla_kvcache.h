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
 * \file mixed_quant_sparse_flash_mla_kvcache.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_KVCACHE_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(
    RunParamStr<HIGH_PERF> &mqCacheRunParam, const ConstInfo<HIGH_PERF> &mqCacheConstInfo,
    GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
    GlobalTensor<int32_t> &cuSeqlensCmpKvGm, GlobalTensor<int32_t> &actualSeqQlenGm,
    GlobalTensor<int32_t> &actualSeqOriKvlenGm, GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
    GlobalTensor<int32_t> &cmpResidualKvGm, bool hasCuSeqlensOriKv, bool hasCuSeqlensCmpKv, bool hasActualSeqQlen,
    bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen)
{
    int32_t actualS1Size = 0;
    int32_t actualS2OriSize = 0;
    int32_t actualS2CmpSize = 0;
    int32_t bIdx = mqCacheRunParam.boIdx;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        actualS1Size = (!hasActualSeqQlen) ? (cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx)) :
                                             actualSeqQlenGm.GetValue(bIdx);
    } else {
        actualS1Size = (!hasActualSeqQlen) ? mqCacheConstInfo.s1Size : actualSeqQlenGm.GetValue(bIdx);
    }

    if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        if (hasActualSeqOriKvlen) {
            actualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
        } else {
            actualS2OriSize = cuSeqlensOriKvGm.GetValue(bIdx + 1) - cuSeqlensOriKvGm.GetValue(bIdx);
        }
    } else {
        actualS2OriSize = (!hasActualSeqOriKvlen) ? mqCacheConstInfo.s2Size : actualSeqOriKvlenGm.GetValue(bIdx);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
            if (hasActualSeqCmpKvlen) {
                actualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            } else if (hasCuSeqlensCmpKv) {
                actualS2CmpSize = cuSeqlensCmpKvGm.GetValue(bIdx + 1) - cuSeqlensCmpKvGm.GetValue(bIdx);
            }
        } else {
            actualS2CmpSize = (!hasActualSeqCmpKvlen) ? mqCacheConstInfo.cmpS2Size : actualSeqCmpKvlenGm.GetValue(bIdx);
        }
    }

    mqCacheRunParam.actualS1Size = actualS1Size;
    mqCacheRunParam.actualS2OriSize = actualS2OriSize;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqCacheRunParam.actualS2CmpSize = actualS2CmpSize;
        if (mqCacheConstInfo.cmpMaskMode == 0) {
            mqCacheRunParam.nextTokensPerBatchCmp = mqCacheRunParam.actualS2CmpSize * mqCacheConstInfo.cmpRatio;
        } else {
            mqCacheRunParam.cmpResidual = (mqCacheConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
            mqCacheRunParam.nextTokensPerBatchCmp =
                (int64_t)mqCacheRunParam.actualS2CmpSize * mqCacheConstInfo.cmpRatio + mqCacheRunParam.cmpResidual -
                mqCacheRunParam.actualS1Size;
        }
    }
    const int64_t casualOffset = mqCacheRunParam.actualS2OriSize - mqCacheRunParam.actualS1Size;
    if (mqCacheConstInfo.oriMaskMode == 3U) {
        mqCacheRunParam.nextTokensPerBatchOri = casualOffset;
        mqCacheRunParam.preTokensPerBatch = mqCacheRunParam.actualS1Size;
    } else if (mqCacheConstInfo.oriMaskMode == 4U) {
        mqCacheRunParam.preTokensPerBatch = (mqCacheConstInfo.oriWinLeft == -1) ?
                                                mqCacheRunParam.actualS1Size :
                                                mqCacheConstInfo.oriWinLeft - casualOffset;
        mqCacheRunParam.nextTokensPerBatchOri = (mqCacheConstInfo.oriWinRight == -1) ?
                                                    mqCacheRunParam.actualS2OriSize :
                                                    casualOffset + mqCacheConstInfo.oriWinRight;
    } else if (mqCacheConstInfo.oriMaskMode == 0) {
        mqCacheRunParam.nextTokensPerBatchOri = mqCacheRunParam.actualS2OriSize;
        mqCacheRunParam.preTokensPerBatch = mqCacheRunParam.actualS1Size;
    }
    mqCacheRunParam.preTokensPerBatch =
        Min(mqCacheRunParam.preTokensPerBatch, static_cast<int64_t>(mqCacheRunParam.actualS1Size));
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(
    RunParamStr<HIGH_PERF> &mqCacheRunParam, const ConstInfo<HIGH_PERF> &mqCacheConstInfo,
    GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
    GlobalTensor<int32_t> &cuSeqlensCmpKvGm, GlobalTensor<int32_t> &actualSeqQlenGm,
    GlobalTensor<int32_t> &actualSeqOriKvlenGm, GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
    GlobalTensor<int32_t> &cmpResidualKvGm, bool hasCuSeqlensOriKv, bool hasCuSeqlensCmpKv, bool hasActualSeqQlen,
    bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(mqCacheRunParam, mqCacheConstInfo, cuSeqlensQGm, cuSeqlensOriKvGm,
                                           cuSeqlensCmpKvGm, actualSeqQlenGm, actualSeqOriKvlenGm, actualSeqCmpKvlenGm,
                                           cmpResidualKvGm, hasCuSeqlensOriKv, hasCuSeqlensCmpKv, hasActualSeqQlen,
                                           hasActualSeqOriKvlen, hasActualSeqCmpKvlen);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                         const ConstInfo<HIGH_PERF> &mqCacheConstInfo, bool lastBN, int64_t nextGs1Idx,
                                         int64_t gS1StartIdx, int64_t s2EndIdx)
{
    // 计算每个基本块可以拷贝多少行s
    mqCacheRunParam.qSNumInOneBlock = 1;
    mqCacheRunParam.gs1LoopStartIdx = gS1StartIdx;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t mqsmlaSkipThreshold = 0;
            if (mqCacheRunParam.nextTokensPerBatchOri < 0 && mqCacheRunParam.nextTokensPerBatchCmp < 0) {
                mqsmlaSkipThreshold =
                    Min(-mqCacheRunParam.nextTokensPerBatchOri, -mqCacheRunParam.nextTokensPerBatchCmp);
            }
            if (mqsmlaSkipThreshold > 0) {
                int64_t mqsmlaGs1LoopStartIdx =
                    mqsmlaSkipThreshold / mqCacheRunParam.qSNumInOneBlock * mqCacheRunParam.qSNumInOneBlock;
                if (mqsmlaGs1LoopStartIdx > gS1StartIdx) {
                    mqCacheRunParam.gs1LoopStartIdx = mqsmlaGs1LoopStartIdx;
                }
            }
        } else {
            if (mqCacheRunParam.nextTokensPerBatchOri < 0) {
                int64_t mqsmlaGs1LoopStartIdx = mqCacheRunParam.nextTokensPerBatchOri * (-1) /
                                                mqCacheRunParam.qSNumInOneBlock * mqCacheRunParam.qSNumInOneBlock;
                if (mqsmlaGs1LoopStartIdx > gS1StartIdx) {
                    mqCacheRunParam.gs1LoopStartIdx = mqsmlaGs1LoopStartIdx;
                }
            }
        }
    }

    int32_t mqsmlaGs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        mqsmlaGs1LoopEndIdx = mqCacheRunParam.actualS1Size;
    } else { // SWA/HCA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        mqsmlaGs1LoopEndIdx =
            (mqCacheRunParam.actualS1Size + mqCacheRunParam.qSNumInOneBlock - 1) / mqCacheRunParam.qSNumInOneBlock;
    }
    if (!lastBN) {
        mqCacheRunParam.gs1LoopEndIdx = mqsmlaGs1LoopEndIdx;
    } else {
        uint32_t mqsmlaActualNextGs1Idx = s2EndIdx == 0 ? nextGs1Idx : nextGs1Idx + 1;
        mqCacheRunParam.gs1LoopEndIdx =
            (nextGs1Idx == 0 && s2EndIdx == 0) ? mqsmlaGs1LoopEndIdx : mqsmlaActualNextGs1Idx;
    }

    if (mqCacheRunParam.gs1LoopStartIdx > mqCacheRunParam.gs1LoopEndIdx) {
        mqCacheRunParam.gs1LoopStartIdx = mqCacheRunParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                          const ConstInfo<HIGH_PERF> &mqCacheConstInfo, uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * mqCacheRunParam.qSNumInOneBlock;
    if (mqCacheRunParam.actualS1Size == 0) {
        mqCacheRunParam.s1RealSize = 0;
        mqCacheRunParam.mRealSize = 0;
    } else {
        mqCacheRunParam.s1RealSize =
            Min(mqCacheRunParam.qSNumInOneBlock, mqCacheRunParam.actualS1Size - cubeSOuterOffset);
        mqCacheRunParam.mRealSize = mqCacheRunParam.s1RealSize * mqCacheConstInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            mqCacheRunParam.mRealSize = mqCacheRunParam.s1RealSize * mqCacheRunParam.gSplitSize;
        }
    }

    mqCacheRunParam.cubeMOuterOffset = cubeSOuterOffset * mqCacheConstInfo.gSize;
    mqCacheRunParam.halfMRealSize = (mqCacheRunParam.mRealSize + 1) >> 1;
    mqCacheRunParam.firstHalfMRealSize = mqCacheRunParam.halfMRealSize;
    if (mqCacheConstInfo.subBlockIdx == 1) {
        mqCacheRunParam.halfMRealSize = mqCacheRunParam.mRealSize - mqCacheRunParam.halfMRealSize;
        mqCacheRunParam.mOuterOffset = mqCacheRunParam.cubeMOuterOffset + mqCacheRunParam.firstHalfMRealSize;
    } else {
        mqCacheRunParam.mOuterOffset = mqCacheRunParam.cubeMOuterOffset;
    }

    mqCacheRunParam.halfS1RealSize = (mqCacheRunParam.s1RealSize + 1) >> 1;
    mqCacheRunParam.firstHalfS1RealSize = mqCacheRunParam.halfS1RealSize;
    if (mqCacheConstInfo.subBlockIdx == 1) {
        mqCacheRunParam.halfS1RealSize = mqCacheRunParam.s1RealSize - mqCacheRunParam.halfS1RealSize;
        mqCacheRunParam.sOuterOffset = cubeSOuterOffset + mqCacheRunParam.firstHalfMRealSize / mqCacheConstInfo.gSize;
    } else {
        mqCacheRunParam.sOuterOffset = cubeSOuterOffset;
    }
    mqCacheRunParam.cubeSOuterOffset = cubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                            const ConstInfo<HIGH_PERF> &mqCacheConstInfo, int32_t sIdx,
                                            GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            seqOffset = cuSeqlensQGm.GetValue(sIdx);
        } else {
            seqOffset = sIdx * mqCacheConstInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * mqCacheConstInfo.n2GDv;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND || LAYOUT_T == QSMLA_LAYOUT::TND) {
            mqCacheRunParam.attentionOutOffset =
                attentionOutSeqOffset + mqCacheRunParam.sOuterOffset * mqCacheConstInfo.n2GDv +
                mqCacheRunParam.n2oIdx * mqCacheConstInfo.gDv + mqCacheRunParam.goIdx * mqCacheConstInfo.dSizeV;
        }
        if (mqCacheConstInfo.subBlockIdx == 1) {
            mqCacheRunParam.attentionOutOffset += mqCacheRunParam.firstHalfMRealSize * mqCacheConstInfo.dSizeV;
        }
        if constexpr (!HIGH_PERF) {
            if (mqCacheConstInfo.isSoftmaxLseEnable) {
                if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                    // [N2, T, G] (TND)
                    mqCacheRunParam.softmaxLseOffset =
                        mqCacheRunParam.n2oIdx * mqCacheConstInfo.s1Size * mqCacheConstInfo.gSize +
                        (seqOffset + mqCacheRunParam.sOuterOffset) * mqCacheConstInfo.gSize;
                } else {
                    // [B, N2, S1, G] (BSND)
                    mqCacheRunParam.softmaxLseOffset =
                        sIdx * mqCacheConstInfo.n2Size * mqCacheConstInfo.s1Size * mqCacheConstInfo.gSize +
                        mqCacheRunParam.n2oIdx * mqCacheConstInfo.s1Size * mqCacheConstInfo.gSize +
                        mqCacheRunParam.sOuterOffset * mqCacheConstInfo.gSize;
                }
                if constexpr (IS_SPLIT_G) {
                    uint32_t aicIdxLocal = mqCacheConstInfo.aivIdx >> 1U;
                    if (aicIdxLocal % 2U != 0) {
                        mqCacheRunParam.softmaxLseOffset += mqCacheRunParam.goIdx;
                    }
                }
                if (mqCacheConstInfo.subBlockIdx == 1) {
                    mqCacheRunParam.softmaxLseOffset += mqCacheRunParam.firstHalfMRealSize;
                }
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                      const ConstInfo<HIGH_PERF> &mqCacheConstInfo, uint32_t sOuterLoopIdx,
                                      GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t mqsmlaSkipThreshold = 0;
            if (mqCacheRunParam.nextTokensPerBatchOri < 0 && mqCacheRunParam.nextTokensPerBatchCmp < 0) {
                mqsmlaSkipThreshold =
                    Min(-mqCacheRunParam.nextTokensPerBatchOri, -mqCacheRunParam.nextTokensPerBatchCmp);
            }
            if (mqsmlaSkipThreshold > 0) {
                if (mqCacheRunParam.s1oIdx <
                    mqsmlaSkipThreshold / mqCacheRunParam.qSNumInOneBlock * mqCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        } else {
            if (mqCacheRunParam.nextTokensPerBatchOri < 0) {
                if (mqCacheRunParam.s1oIdx < (mqCacheRunParam.nextTokensPerBatchOri * (-1)) /
                                                 mqCacheRunParam.qSNumInOneBlock * mqCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(mqCacheRunParam, mqCacheConstInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(mqCacheRunParam, mqCacheConstInfo, mqCacheRunParam.boIdx, cuSeqlensQGm);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr<HIGH_PERF> &mqCacheRunParam, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        if (mqCacheRunParam.boIdx > 0 &&
            cuSeqlensQGm.GetValue(mqCacheRunParam.boIdx + 1) - cuSeqlensQGm.GetValue(mqCacheRunParam.boIdx) == 0) {
            return true;
        }
    }
    return false;
}

TEMPLATE_INTF
__aicore__ inline int64_t ClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeS2LoopInfo(int64_t bnIndex, int64_t gS1Index, GlobalTensor<int32_t> &cuSeqlensQGm,
                                         GlobalTensor<int32_t> &oriTopkLengthGm, GlobalTensor<int32_t> &cmpTopkLengthGm,
                                         RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                         const ConstInfo<HIGH_PERF> &mqCacheConstInfo)
{
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (mqCacheRunParam.actualS2OriSize == 0) {
            mqCacheRunParam.oriKvLoopEndIdx = 0;
            mqCacheRunParam.cmpKvLoopEndIdx = 0;
            mqCacheRunParam.s2LoopEndIdx = 0;
            mqCacheRunParam.s2CmpLineStartIdx = 0;
            return true;
        }
    }
    uint32_t mqsmlaS2BaseSize = mqCacheConstInfo.s2BaseSize;

    uint32_t oriSparseBlockCount = mqCacheConstInfo.oriSparseBlockCount;
    uint32_t cmpSparseBlockCount = mqCacheConstInfo.cmpSparseBlockCount;
    if constexpr (!HIGH_PERF) {
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(mqCacheRunParam.boIdx);
            oriSparseBlockCount = mqCacheConstInfo.hasOriTopkLength ?
                                      Min(oriTopkLengthGm.GetValue(actualSeqQPrefixSum + mqCacheRunParam.s1oIdx),
                                          mqCacheConstInfo.oriSparseBlockCount) :
                                      mqCacheConstInfo.oriSparseBlockCount;
            cmpSparseBlockCount = mqCacheConstInfo.hasCmpTopkLength ?
                                      Min(cmpTopkLengthGm.GetValue(actualSeqQPrefixSum + mqCacheRunParam.s1oIdx),
                                          mqCacheConstInfo.cmpSparseBlockCount) :
                                      mqCacheConstInfo.cmpSparseBlockCount;
        } else {
            uint64_t bsndTopkIdx = mqCacheRunParam.boIdx * mqCacheConstInfo.s1Size + mqCacheRunParam.s1oIdx;
            oriSparseBlockCount = mqCacheConstInfo.hasOriTopkLength ?
                                      Min(oriTopkLengthGm.GetValue(bsndTopkIdx), mqCacheConstInfo.oriSparseBlockCount) :
                                      mqCacheConstInfo.oriSparseBlockCount;
            cmpSparseBlockCount = mqCacheConstInfo.hasCmpTopkLength ?
                                      Min(cmpTopkLengthGm.GetValue(bsndTopkIdx), mqCacheConstInfo.cmpSparseBlockCount) :
                                      mqCacheConstInfo.cmpSparseBlockCount;
        }
    }

    // orikv
    mqCacheRunParam.s2LineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        mqCacheRunParam.cubeSOuterOffset - mqCacheRunParam.preTokensPerBatch, 0, mqCacheRunParam.actualS2OriSize);
    mqCacheRunParam.s2LineOriEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        mqCacheRunParam.cubeSOuterOffset + mqCacheRunParam.nextTokensPerBatchOri + mqCacheRunParam.s1RealSize, 0,
        mqCacheRunParam.actualS2OriSize);
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t oriSparseRangeLen = mqCacheRunParam.s2LineOriEndIdx - mqCacheRunParam.s2LineStartIdx;
        mqCacheRunParam.s2LineStartIdx = 0;
        mqCacheRunParam.s2LineOriEndIdx = Min(oriSparseRangeLen, oriSparseBlockCount);
        mqCacheRunParam.s2LineOriEndIdx = Min(mqCacheRunParam.s2LineOriEndIdx, mqCacheRunParam.actualS2OriSize);
    }
    mqCacheRunParam.oriKvLoopEndIdx =
        (mqCacheRunParam.s2LineOriEndIdx - mqCacheRunParam.s2LineStartIdx + mqsmlaS2BaseSize - 1) / mqsmlaS2BaseSize;

    // cmpkv
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::SWA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqCacheRunParam.s2CmpLineStartIdx = 0;
        mqCacheRunParam.s2CmpLineEndIdx = 0;
        mqCacheRunParam.cmpKvLoopEndIdx = 0;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE) {
        mqCacheRunParam.s2CmpLineStartIdx = 0;
        mqCacheRunParam.s2LineCmpEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (mqCacheRunParam.cubeSOuterOffset + mqCacheRunParam.s1RealSize + mqCacheRunParam.nextTokensPerBatchCmp) /
                mqCacheConstInfo.cmpRatio,
            0, mqCacheRunParam.actualS2CmpSize);
        mqCacheRunParam.s2CmpLineEndIdx = Min(mqCacheRunParam.s2LineCmpEndIdx, mqCacheRunParam.actualS2CmpSize);
        mqCacheRunParam.cmpKvLoopEndIdx = (mqCacheRunParam.s2CmpLineEndIdx + mqsmlaS2BaseSize - 1) / mqsmlaS2BaseSize;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) { // CSA / ORI_CMP_SPARSE
        mqCacheRunParam.s2CmpLineStartIdx = 0;
        mqCacheRunParam.s2LineCmpEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (mqCacheRunParam.cubeSOuterOffset + mqCacheRunParam.s1RealSize + mqCacheRunParam.nextTokensPerBatchCmp) /
                mqCacheConstInfo.cmpRatio,
            0, mqCacheRunParam.actualS2CmpSize);
        mqCacheRunParam.s2CmpLineEndIdx = Min(mqCacheRunParam.s2LineCmpEndIdx, cmpSparseBlockCount);
        mqCacheRunParam.s2CmpLineEndIdx = Min(mqCacheRunParam.s2CmpLineEndIdx, mqCacheRunParam.actualS2CmpSize);
        mqCacheRunParam.cmpKvLoopEndIdx = (mqCacheRunParam.s2CmpLineEndIdx + mqsmlaS2BaseSize - 1) / mqsmlaS2BaseSize;
    }

    mqCacheRunParam.s2LoopEndIdx = mqCacheRunParam.oriKvLoopEndIdx + mqCacheRunParam.cmpKvLoopEndIdx;
    return (mqCacheRunParam.s2LoopEndIdx == 0);
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr<HIGH_PERF> &mqCacheRunParam,
                                          RunInfo<HIGH_PERF> &mqCacheRunInfo)
{
    mqCacheRunInfo.boIdx = mqCacheRunParam.boIdx;
    mqCacheRunInfo.preTokensPerBatch = mqCacheRunParam.preTokensPerBatch;
    mqCacheRunInfo.nextTokensPerBatchOri = mqCacheRunParam.nextTokensPerBatchOri;
    mqCacheRunInfo.actualS1Size = mqCacheRunParam.actualS1Size;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        mqCacheRunInfo.actualS2CmpSize = mqCacheRunParam.actualS2CmpSize;
        mqCacheRunInfo.cmpResidual = mqCacheRunParam.cmpResidual;
        mqCacheRunInfo.cmpKvLoopEndIdx = mqCacheRunParam.cmpKvLoopEndIdx;
    }
    if constexpr (!HIGH_PERF) {
        mqCacheRunInfo.softmaxLseOffset = mqCacheRunParam.softmaxLseOffset;
    }
    mqCacheRunInfo.qSNumInOneBlock = mqCacheRunParam.qSNumInOneBlock;
    mqCacheRunInfo.oriKvLoopEndIdx = mqCacheRunParam.oriKvLoopEndIdx;
    mqCacheRunInfo.isCmp = mqCacheRunInfo.s2LoopCount >= mqCacheRunInfo.oriKvLoopEndIdx;
}

#endif // MIXED_QUANT_SPARSE_FLASH_MLA_KVCACHE_H
