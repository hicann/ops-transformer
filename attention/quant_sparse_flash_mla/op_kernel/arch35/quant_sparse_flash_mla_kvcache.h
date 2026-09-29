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
 * \file quant_sparse_flash_mla_kvcache.h
 * \brief
 */
#ifndef QUANT_SPARSE_FLASH_MLA_KVCACHE_H
#define QUANT_SPARSE_FLASH_MLA_KVCACHE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "quant_sparse_flash_mla_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(
    RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo, GlobalTensor<int32_t> &cuSeqlensQGm,
    GlobalTensor<int32_t> &cuSeqlensOriKvGm, GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> &actualSeqQlenGm, GlobalTensor<int32_t> &actualSeqOriKvlenGm,
    GlobalTensor<int32_t> &actualSeqCmpKvlenGm, GlobalTensor<int32_t> &cmpResidualKvGm, bool hasCuSeqlensOriKv,
    bool hasCuSeqlensCmpKv, bool hasActualSeqQlen, bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen)
{
    int32_t localActualS1Size = 0;
    int32_t localActualS2OriSize = 0;
    int32_t localActualS2CmpSize = 0;
    int32_t bIdx = qsCacheRunParam.boIdx;
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        localActualS1Size = (!hasActualSeqQlen) ? (cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx)) :
                                                  actualSeqQlenGm.GetValue(bIdx);
    } else {
        localActualS1Size = (!hasActualSeqQlen) ? qsCacheConstInfo.s1Size : actualSeqQlenGm.GetValue(bIdx);
    }

    if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
        if (hasActualSeqOriKvlen) {
            localActualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
        } else {
            localActualS2OriSize = cuSeqlensOriKvGm.GetValue(bIdx + 1) - cuSeqlensOriKvGm.GetValue(bIdx);
        }
    } else {
        localActualS2OriSize = (!hasActualSeqOriKvlen) ? qsCacheConstInfo.s2Size : actualSeqOriKvlenGm.GetValue(bIdx);
    }

    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if constexpr (KV_LAYOUT_T == QSMLA_LAYOUT::TND) {
            if (hasActualSeqCmpKvlen) {
                localActualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            } else if (hasCuSeqlensCmpKv) {
                localActualS2CmpSize = cuSeqlensCmpKvGm.GetValue(bIdx + 1) - cuSeqlensCmpKvGm.GetValue(bIdx);
            }
        } else {
            localActualS2CmpSize =
                (!hasActualSeqCmpKvlen) ? qsCacheConstInfo.cmpS2Size : actualSeqCmpKvlenGm.GetValue(bIdx);
        }
    }

    qsCacheRunParam.actualS1Size = localActualS1Size;
    qsCacheRunParam.actualS2OriSize = localActualS2OriSize;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        qsCacheRunParam.actualS2CmpSize = localActualS2CmpSize;
        if (qsCacheConstInfo.cmpMaskMode == 0) {
            qsCacheRunParam.nextTokensPerBatchCmp = qsCacheRunParam.actualS2CmpSize * qsCacheConstInfo.cmpRatio;
        } else {
            qsCacheRunParam.cmpResidual = (qsCacheConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
            qsCacheRunParam.nextTokensPerBatchCmp =
                (int64_t)qsCacheRunParam.actualS2CmpSize * qsCacheConstInfo.cmpRatio + qsCacheRunParam.cmpResidual -
                qsCacheRunParam.actualS1Size;
        }
    }

    if (qsCacheConstInfo.oriMaskMode == 0) {
        qsCacheRunParam.nextTokensPerBatchOri = qsCacheRunParam.actualS2OriSize;
        qsCacheRunParam.preTokensPerBatch = qsCacheRunParam.actualS1Size;
    } else if (qsCacheConstInfo.oriMaskMode == 3) { // 3: RightDownCausal模式
        qsCacheRunParam.nextTokensPerBatchOri = qsCacheRunParam.actualS2OriSize - qsCacheRunParam.actualS1Size;
        qsCacheRunParam.preTokensPerBatch = qsCacheRunParam.actualS1Size;
    } else if (qsCacheConstInfo.oriMaskMode == 4) { // 4: Band模式
        const int64_t casualOffset = qsCacheRunParam.actualS2OriSize - qsCacheRunParam.actualS1Size;
        qsCacheRunParam.preTokensPerBatch = (qsCacheConstInfo.oriWinLeft == -1) ?
                                                qsCacheRunParam.actualS1Size :
                                                qsCacheConstInfo.oriWinLeft - casualOffset;
        qsCacheRunParam.nextTokensPerBatchOri = (qsCacheConstInfo.oriWinRight == -1) ?
                                                    qsCacheRunParam.actualS2OriSize :
                                                    casualOffset + qsCacheConstInfo.oriWinRight;
        qsCacheRunParam.preTokensPerBatch =
            Min(qsCacheRunParam.preTokensPerBatch, static_cast<int64_t>(qsCacheRunParam.actualS1Size));
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(
    RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo, GlobalTensor<int32_t> &cuSeqlensQGm,
    GlobalTensor<int32_t> &cuSeqlensOriKvGm, GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
    GlobalTensor<int32_t> &actualSeqQlenGm, GlobalTensor<int32_t> &actualSeqOriKvlenGm,
    GlobalTensor<int32_t> &actualSeqCmpKvlenGm, GlobalTensor<int32_t> &cmpResidualKvGm, bool hasCuSeqlensOriKv,
    bool hasCuSeqlensCmpKv, bool hasActualSeqQlen, bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(qsCacheRunParam, qsCacheConstInfo, cuSeqlensQGm, cuSeqlensOriKvGm,
                                           cuSeqlensCmpKvGm, actualSeqQlenGm, actualSeqOriKvlenGm, actualSeqCmpKvlenGm,
                                           cmpResidualKvGm, hasCuSeqlensOriKv, hasCuSeqlensCmpKv, hasActualSeqQlen,
                                           hasActualSeqOriKvlen, hasActualSeqCmpKvlen);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo, bool lastBN,
                                         int64_t nextGs1Idx, int64_t gS1StartIdx, int64_t s2EndIdx)
{
    // 计算每个基本块可以拷贝多少行s
    qsCacheRunParam.qSNumInOneBlock = 1;
    qsCacheRunParam.gs1LoopStartIdx = gS1StartIdx;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t qsmlaSkipThreshold = 0;
            if (qsCacheRunParam.nextTokensPerBatchOri < 0 && qsCacheRunParam.nextTokensPerBatchCmp < 0) {
                qsmlaSkipThreshold =
                    Min(-qsCacheRunParam.nextTokensPerBatchOri, -qsCacheRunParam.nextTokensPerBatchCmp);
            }
            if (qsmlaSkipThreshold > 0) {
                int64_t qsmlaGs1LoopStartIdx =
                    qsmlaSkipThreshold / qsCacheRunParam.qSNumInOneBlock * qsCacheRunParam.qSNumInOneBlock;
                if (qsmlaGs1LoopStartIdx > gS1StartIdx) {
                    qsCacheRunParam.gs1LoopStartIdx = qsmlaGs1LoopStartIdx;
                }
            }
        } else {
            if (qsCacheRunParam.nextTokensPerBatchOri < 0) {
                int64_t qsmlaGs1LoopStartIdx = qsCacheRunParam.nextTokensPerBatchOri * (-1) /
                                               qsCacheRunParam.qSNumInOneBlock * qsCacheRunParam.qSNumInOneBlock;
                if (qsmlaGs1LoopStartIdx > gS1StartIdx) {
                    qsCacheRunParam.gs1LoopStartIdx = qsmlaGs1LoopStartIdx;
                }
            }
        }
    }

    int32_t qsmlaGs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        qsmlaGs1LoopEndIdx = qsCacheRunParam.actualS1Size;
    } else { // SWA/HCA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        qsmlaGs1LoopEndIdx =
            (qsCacheRunParam.actualS1Size + qsCacheRunParam.qSNumInOneBlock - 1) / qsCacheRunParam.qSNumInOneBlock;
    }
    if (!lastBN) {
        qsCacheRunParam.gs1LoopEndIdx = qsmlaGs1LoopEndIdx;
    } else {
        uint32_t actualNextGs1Idx = s2EndIdx == 0 ? nextGs1Idx : nextGs1Idx + 1;
        qsCacheRunParam.gs1LoopEndIdx = (nextGs1Idx == 0 && s2EndIdx == 0) ? qsmlaGs1LoopEndIdx : actualNextGs1Idx;
    }

    if (qsCacheRunParam.gs1LoopStartIdx > qsCacheRunParam.gs1LoopEndIdx) {
        qsCacheRunParam.gs1LoopStartIdx = qsCacheRunParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo,
                                          uint32_t sOuterLoopIdx)
{
    int64_t qsmlaCubeSOuterOffset = sOuterLoopIdx * qsCacheRunParam.qSNumInOneBlock;
    if (qsCacheRunParam.actualS1Size == 0) {
        qsCacheRunParam.s1RealSize = 0;
        qsCacheRunParam.mRealSize = 0;
    } else {
        qsCacheRunParam.s1RealSize =
            Min(qsCacheRunParam.qSNumInOneBlock, qsCacheRunParam.actualS1Size - qsmlaCubeSOuterOffset);
        qsCacheRunParam.mRealSize = qsCacheRunParam.s1RealSize * qsCacheConstInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            qsCacheRunParam.mRealSize = qsCacheRunParam.s1RealSize * qsCacheRunParam.gSplitSize;
        }
    }

    qsCacheRunParam.cubeMOuterOffset = qsmlaCubeSOuterOffset * qsCacheConstInfo.gSize;
    qsCacheRunParam.halfMRealSize = (qsCacheRunParam.mRealSize + 1) >> 1;
    qsCacheRunParam.firstHalfMRealSize = qsCacheRunParam.halfMRealSize;
    if (qsCacheConstInfo.subBlockIdx == 1) {
        qsCacheRunParam.halfMRealSize = qsCacheRunParam.mRealSize - qsCacheRunParam.halfMRealSize;
        qsCacheRunParam.mOuterOffset = qsCacheRunParam.cubeMOuterOffset + qsCacheRunParam.firstHalfMRealSize;
    } else {
        qsCacheRunParam.mOuterOffset = qsCacheRunParam.cubeMOuterOffset;
    }

    qsCacheRunParam.halfS1RealSize = (qsCacheRunParam.s1RealSize + 1) >> 1;
    qsCacheRunParam.firstHalfS1RealSize = qsCacheRunParam.halfS1RealSize;
    if (qsCacheConstInfo.subBlockIdx == 1) {
        qsCacheRunParam.halfS1RealSize = qsCacheRunParam.s1RealSize - qsCacheRunParam.halfS1RealSize;
        qsCacheRunParam.sOuterOffset =
            qsmlaCubeSOuterOffset + qsCacheRunParam.firstHalfMRealSize / qsCacheConstInfo.gSize;
    } else {
        qsCacheRunParam.sOuterOffset = qsmlaCubeSOuterOffset;
    }
    qsCacheRunParam.cubeSOuterOffset = qsmlaCubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo,
                                            int32_t sIdx, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if ASCEND_IS_AIV {
        int64_t qsmlaSeqOffset = 0;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
            qsmlaSeqOffset = cuSeqlensQGm.GetValue(sIdx);
        } else {
            qsmlaSeqOffset = sIdx * qsCacheConstInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = qsmlaSeqOffset * qsCacheConstInfo.n2GDv;
        if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND || LAYOUT_T == QSMLA_LAYOUT::TND) {
            qsCacheRunParam.attentionOutOffset =
                attentionOutSeqOffset + qsCacheRunParam.sOuterOffset * qsCacheConstInfo.n2GDv +
                qsCacheRunParam.n2oIdx * qsCacheConstInfo.gDv + qsCacheRunParam.goIdx * qsCacheConstInfo.dSizeV;
        }
        if (qsCacheConstInfo.subBlockIdx == 1) {
            qsCacheRunParam.attentionOutOffset += qsCacheRunParam.firstHalfMRealSize * qsCacheConstInfo.dSizeV;
        }
        if (qsCacheConstInfo.isSoftmaxLseEnable) {
            if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
                // [N2, T, G] (TND)
                qsCacheRunParam.softmaxLseOffset =
                    qsCacheRunParam.n2oIdx * qsCacheConstInfo.s1Size * qsCacheConstInfo.gSize +
                    (qsmlaSeqOffset + qsCacheRunParam.sOuterOffset) * qsCacheConstInfo.gSize;
            } else {
                // [B, N2, S1, G] (BSND)
                qsCacheRunParam.softmaxLseOffset =
                    sIdx * qsCacheConstInfo.n2Size * qsCacheConstInfo.s1Size * qsCacheConstInfo.gSize +
                    qsCacheRunParam.n2oIdx * qsCacheConstInfo.s1Size * qsCacheConstInfo.gSize +
                    qsCacheRunParam.sOuterOffset * qsCacheConstInfo.gSize;
            }
            if constexpr (IS_SPLIT_G) {
                uint32_t qsmlaAicIdx = qsCacheConstInfo.aivIdx >> 1U;
                if (qsmlaAicIdx % 2U != 0) {
                    qsCacheRunParam.softmaxLseOffset += qsCacheRunParam.goIdx;
                }
            }
            if (qsCacheConstInfo.subBlockIdx == 1) {
                qsCacheRunParam.softmaxLseOffset += qsCacheRunParam.firstHalfMRealSize;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo,
                                      uint32_t sOuterLoopIdx, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t qsmlaSkipThreshold = 0;
            if (qsCacheRunParam.nextTokensPerBatchOri < 0 && qsCacheRunParam.nextTokensPerBatchCmp < 0) {
                qsmlaSkipThreshold =
                    Min(-qsCacheRunParam.nextTokensPerBatchOri, -qsCacheRunParam.nextTokensPerBatchCmp);
            }
            if (qsmlaSkipThreshold > 0) {
                if (qsCacheRunParam.s1oIdx <
                    qsmlaSkipThreshold / qsCacheRunParam.qSNumInOneBlock * qsCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        } else {
            if (qsCacheRunParam.nextTokensPerBatchOri < 0) {
                if (qsCacheRunParam.s1oIdx < (qsCacheRunParam.nextTokensPerBatchOri * (-1)) /
                                                 qsCacheRunParam.qSNumInOneBlock * qsCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(qsCacheRunParam, qsCacheConstInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(qsCacheRunParam, qsCacheConstInfo, qsCacheRunParam.boIdx, cuSeqlensQGm);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr &qsCacheRunParam, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        if (qsCacheRunParam.boIdx > 0 &&
            cuSeqlensQGm.GetValue(qsCacheRunParam.boIdx + 1) - cuSeqlensQGm.GetValue(qsCacheRunParam.boIdx) == 0) {
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
                                         RunParamStr &qsCacheRunParam, const ConstInfo &qsCacheConstInfo)
{
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (qsCacheRunParam.actualS2OriSize == 0) {
            qsCacheRunParam.oriKvLoopEndIdx = 0;
            qsCacheRunParam.cmpKvLoopEndIdx = 0;
            qsCacheRunParam.s2LoopEndIdx = 0;
            qsCacheRunParam.s2CmpLineStartIdx = 0;
            return true;
        }
    }
    uint32_t s2BaseSize = qsCacheConstInfo.s2BaseSize;

    if constexpr (LAYOUT_T == QSMLA_LAYOUT::TND) {
        uint64_t qsmlaActualSeqQPrefixSum = cuSeqlensQGm.GetValue(qsCacheRunParam.boIdx);
        qsCacheRunParam.oriSparseBlockCount =
            qsCacheConstInfo.hasOriTopkLength ?
                Min(oriTopkLengthGm.GetValue(qsmlaActualSeqQPrefixSum + qsCacheRunParam.s1oIdx),
                    qsCacheConstInfo.oriSparseBlockCount) :
                qsCacheConstInfo.oriSparseBlockCount;
        qsCacheRunParam.cmpSparseBlockCount =
            qsCacheConstInfo.hasCmpTopkLength ?
                Min(cmpTopkLengthGm.GetValue(qsmlaActualSeqQPrefixSum + qsCacheRunParam.s1oIdx),
                    qsCacheConstInfo.cmpSparseBlockCount) :
                qsCacheConstInfo.cmpSparseBlockCount;
    } else {
        uint64_t qsmlaBsndTopkIdx = qsCacheRunParam.boIdx * qsCacheConstInfo.s1Size + qsCacheRunParam.s1oIdx;
        qsCacheRunParam.oriSparseBlockCount =
            qsCacheConstInfo.hasOriTopkLength ?
                Min(oriTopkLengthGm.GetValue(qsmlaBsndTopkIdx), qsCacheConstInfo.oriSparseBlockCount) :
                qsCacheConstInfo.oriSparseBlockCount;
        qsCacheRunParam.cmpSparseBlockCount =
            qsCacheConstInfo.hasCmpTopkLength ?
                Min(cmpTopkLengthGm.GetValue(qsmlaBsndTopkIdx), qsCacheConstInfo.cmpSparseBlockCount) :
                qsCacheConstInfo.cmpSparseBlockCount;
    }

    qsCacheRunParam.s2LineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        qsCacheRunParam.cubeSOuterOffset - qsCacheRunParam.preTokensPerBatch, 0, qsCacheRunParam.actualS2OriSize);
    qsCacheRunParam.s2LineOriEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        qsCacheRunParam.cubeSOuterOffset + qsCacheRunParam.nextTokensPerBatchOri + qsCacheRunParam.s1RealSize, 0,
        qsCacheRunParam.actualS2OriSize);
    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t oriSparseRangeLen = qsCacheRunParam.s2LineOriEndIdx - qsCacheRunParam.s2LineStartIdx;
        qsCacheRunParam.s2LineStartIdx = 0;
        qsCacheRunParam.s2LineOriEndIdx = Min(oriSparseRangeLen, qsCacheRunParam.oriSparseBlockCount);
        qsCacheRunParam.s2LineOriEndIdx = Min(qsCacheRunParam.s2LineOriEndIdx, qsCacheRunParam.actualS2OriSize);
    }
    qsCacheRunParam.oriKvLoopEndIdx =
        (qsCacheRunParam.s2LineOriEndIdx - qsCacheRunParam.s2LineStartIdx + s2BaseSize - 1) / s2BaseSize;

    if constexpr (TEMPLATE_MODE == QSMLATemplateMode::SWA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        qsCacheRunParam.s2CmpLineStartIdx = 0;
        qsCacheRunParam.s2CmpLineEndIdx = 0;
        qsCacheRunParam.cmpKvLoopEndIdx = 0;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::HCA_TEMPLATE_MODE) {
        qsCacheRunParam.s2CmpLineStartIdx = 0;
        qsCacheRunParam.s2LineCmpEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (qsCacheRunParam.cubeSOuterOffset + qsCacheRunParam.s1RealSize + qsCacheRunParam.nextTokensPerBatchCmp) /
                qsCacheConstInfo.cmpRatio,
            0, qsCacheRunParam.actualS2CmpSize);
        qsCacheRunParam.s2CmpLineEndIdx = Min(qsCacheRunParam.s2LineCmpEndIdx, qsCacheRunParam.actualS2CmpSize);
        qsCacheRunParam.cmpKvLoopEndIdx = (qsCacheRunParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    } else if constexpr (TEMPLATE_MODE == QSMLATemplateMode::CSA_TEMPLATE_MODE ||
                         TEMPLATE_MODE == QSMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        qsCacheRunParam.s2CmpLineStartIdx = 0;
        qsCacheRunParam.s2LineCmpEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
            (qsCacheRunParam.cubeSOuterOffset + qsCacheRunParam.s1RealSize + qsCacheRunParam.nextTokensPerBatchCmp) /
                qsCacheConstInfo.cmpRatio,
            0, qsCacheRunParam.actualS2CmpSize);
        qsCacheRunParam.s2CmpLineEndIdx = Min(qsCacheRunParam.s2LineCmpEndIdx, qsCacheRunParam.cmpSparseBlockCount);
        qsCacheRunParam.s2CmpLineEndIdx = Min(qsCacheRunParam.s2CmpLineEndIdx, qsCacheRunParam.actualS2CmpSize);
        qsCacheRunParam.cmpKvLoopEndIdx = (qsCacheRunParam.s2CmpLineEndIdx + s2BaseSize - 1) / s2BaseSize;
    }

    qsCacheRunParam.s2LoopEndIdx = qsCacheRunParam.oriKvLoopEndIdx + qsCacheRunParam.cmpKvLoopEndIdx;
    return (qsCacheRunParam.s2LoopEndIdx == 0);
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr &qsCacheRunParam, RunInfo &qsCacheRunInfo)
{
    qsCacheRunInfo.boIdx = qsCacheRunParam.boIdx;
    qsCacheRunInfo.preTokensPerBatch = qsCacheRunParam.preTokensPerBatch;
    qsCacheRunInfo.nextTokensPerBatchOri = qsCacheRunParam.nextTokensPerBatchOri;
    qsCacheRunInfo.actualS1Size = qsCacheRunParam.actualS1Size;
    if constexpr (TEMPLATE_MODE != QSMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != QSMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        qsCacheRunInfo.actualS2CmpSize = qsCacheRunParam.actualS2CmpSize;
        qsCacheRunInfo.cmpResidual = qsCacheRunParam.cmpResidual;
        qsCacheRunInfo.cmpKvLoopEndIdx = qsCacheRunParam.cmpKvLoopEndIdx;
    }
    qsCacheRunInfo.softmaxLseOffset = qsCacheRunParam.softmaxLseOffset;
    qsCacheRunInfo.qSNumInOneBlock = qsCacheRunParam.qSNumInOneBlock;
    qsCacheRunInfo.oriKvLoopEndIdx = qsCacheRunParam.oriKvLoopEndIdx;
    qsCacheRunInfo.isCmp = qsCacheRunInfo.s2LoopCount >= qsCacheRunInfo.oriKvLoopEndIdx;
    qsCacheRunInfo.oriSparseBlockCount = qsCacheRunParam.oriSparseBlockCount;
    qsCacheRunInfo.cmpSparseBlockCount = qsCacheRunParam.cmpSparseBlockCount;
}

#endif // QUANT_SPARSE_FLASH_MLA_KVCACHE_H
