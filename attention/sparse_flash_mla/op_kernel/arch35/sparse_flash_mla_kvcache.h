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
 * \file sparse_flash_mla_kvcache.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_KVCACHE_H
#define SPARSE_FLASH_MLA_KVCACHE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "sparse_flash_mla_common_arch35.h"
#include "util_regbase.h"

using namespace matmul;
using namespace regbaseutil;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace SMLAKernel;

TEMPLATE_INTF
__aicore__ inline void GetSingleCoreParam(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                          GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
                                          GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
                                          GlobalTensor<int32_t> &actualSeqQlenGm,
                                          GlobalTensor<int32_t> &actualSeqOriKvlenGm,
                                          GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
                                          GlobalTensor<int32_t> &cmpResidualKvGm, bool hasActualSeqQlen,
                                          bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv)
{
    int32_t actualS1Size = 0;
    int32_t actualS2OriSize = 0;
    int32_t actualS2CmpSize = 0;
    int32_t bIdx = smlaCacheRunParam.boIdx;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        actualS1Size = (!hasActualSeqQlen) ? (cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx)) :
                                             actualSeqQlenGm.GetValue(bIdx);
    } else {
        actualS1Size = (!hasActualSeqQlen) ? smlaCacheConstInfo.s1Size : actualSeqQlenGm.GetValue(bIdx);
    }

    if (smlaCacheConstInfo.isActualLenDimsOriKVNull) {
        actualS2OriSize = smlaCacheConstInfo.s2Size;
    } else {
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
            if (hasActualSeqOriKvlen) {
                actualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
            } else {
                actualS2OriSize = cuSeqlensOriKvGm.GetValue(bIdx + 1) - cuSeqlensOriKvGm.GetValue(bIdx);
            }
        } else {
            actualS2OriSize = actualSeqOriKvlenGm.GetValue(bIdx);
        }
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
            if (hasActualSeqCmpKvlen) {
                actualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            } else if (hasCuSeqlensCmpKv) {
                actualS2CmpSize = cuSeqlensCmpKvGm.GetValue(bIdx + 1) - cuSeqlensCmpKvGm.GetValue(bIdx);
            }
        } else {
            if (!hasActualSeqCmpKvlen) {
                actualS2CmpSize = smlaCacheConstInfo.cmpS2Size;
            } else {
                actualS2CmpSize = actualSeqCmpKvlenGm.GetValue(bIdx);
            }
        }
    }

    smlaCacheRunParam.actualS1Size = actualS1Size;
    smlaCacheRunParam.actualS2OriSize = actualS2OriSize;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        smlaCacheRunParam.actualS2CmpSize = actualS2CmpSize;
        if (smlaCacheConstInfo.cmpMaskMode == 0) {
            smlaCacheRunParam.nextTokensPerBatchCmp = smlaCacheRunParam.actualS2CmpSize * smlaCacheConstInfo.cmpRatio;
        } else {
            smlaCacheRunParam.cmpResidual = (smlaCacheConstInfo.cmpRatio != 1) ? cmpResidualKvGm.GetValue(bIdx) : 0;
            smlaCacheRunParam.nextTokensPerBatchCmp =
                (int64_t)smlaCacheRunParam.actualS2CmpSize * smlaCacheConstInfo.cmpRatio +
                smlaCacheRunParam.cmpResidual - smlaCacheRunParam.actualS1Size;
        }
    }

    if (smlaCacheConstInfo.oriMaskMode == 3) { // 3: RightDownCausal模式
        smlaCacheRunParam.nextTokensPerBatchOri = smlaCacheRunParam.actualS2OriSize - smlaCacheRunParam.actualS1Size;
        smlaCacheRunParam.preTokensPerBatchOri = smlaCacheRunParam.actualS1Size;
    } else if (smlaCacheConstInfo.oriMaskMode == 4) { // 4: Band模式
        const int64_t casualOffset = smlaCacheRunParam.actualS2OriSize - smlaCacheRunParam.actualS1Size;

        smlaCacheRunParam.preTokensPerBatchOri = (smlaCacheConstInfo.oriWinLeft == -1) ?
                                                     smlaCacheRunParam.actualS1Size :
                                                     smlaCacheConstInfo.oriWinLeft - casualOffset;

        smlaCacheRunParam.nextTokensPerBatchOri = (smlaCacheConstInfo.oriWinRight == -1) ?
                                                      smlaCacheRunParam.actualS2OriSize :
                                                      casualOffset + smlaCacheConstInfo.oriWinRight;

        smlaCacheRunParam.preTokensPerBatchOri =
            Min(smlaCacheRunParam.preTokensPerBatchOri, static_cast<int64_t>(smlaCacheRunParam.actualS1Size));
    } else if (smlaCacheConstInfo.oriMaskMode == 0) {
        smlaCacheRunParam.nextTokensPerBatchOri = smlaCacheRunParam.actualS2OriSize;
        smlaCacheRunParam.preTokensPerBatchOri = smlaCacheRunParam.actualS1Size;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeParamBatch(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                         GlobalTensor<int32_t> &cuSeqlensQGm, GlobalTensor<int32_t> &cuSeqlensOriKvGm,
                                         GlobalTensor<int32_t> &cuSeqlensCmpKvGm,
                                         GlobalTensor<int32_t> &actualSeqQlenGm,
                                         GlobalTensor<int32_t> &actualSeqOriKvlenGm,
                                         GlobalTensor<int32_t> &actualSeqCmpKvlenGm,
                                         GlobalTensor<int32_t> &cmpResidualKvGm, bool hasActualSeqQlen,
                                         bool hasActualSeqOriKvlen, bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv)
{
    GetSingleCoreParam<TEMPLATE_INTF_ARGS>(smlaCacheRunParam, smlaCacheConstInfo, cuSeqlensQGm, cuSeqlensOriKvGm,
                                           cuSeqlensCmpKvGm, actualSeqQlenGm, actualSeqOriKvlenGm, actualSeqCmpKvlenGm,
                                           cmpResidualKvGm, hasActualSeqQlen, hasActualSeqOriKvlen,
                                           hasActualSeqCmpKvlen, hasCuSeqlensCmpKv);
}

TEMPLATE_INTF
__aicore__ inline void ComputeS1LoopInfo(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                         bool lastBN, int64_t nextGs1Idx, int64_t gS1StartIdx, int64_t s2EndIdx = 0)
{
    smlaCacheRunParam.qSNumInOneBlock = 1;
    smlaCacheRunParam.gs1LoopStartIdx = gS1StartIdx;
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t smlaSkipThreshold = 0;
            if (smlaCacheRunParam.nextTokensPerBatchOri < 0 && smlaCacheRunParam.nextTokensPerBatchCmp < 0) {
                smlaSkipThreshold =
                    Min(-smlaCacheRunParam.nextTokensPerBatchOri, -smlaCacheRunParam.nextTokensPerBatchCmp);
            }
            if (smlaSkipThreshold > 0) {
                int64_t smlaGs1LoopStartIdx =
                    smlaSkipThreshold / smlaCacheRunParam.qSNumInOneBlock * smlaCacheRunParam.qSNumInOneBlock;
                if (smlaGs1LoopStartIdx > gS1StartIdx) {
                    smlaCacheRunParam.gs1LoopStartIdx = smlaGs1LoopStartIdx;
                }
            }
        } else {
            if (smlaCacheRunParam.nextTokensPerBatchOri < 0) {
                int64_t smlaGs1LoopStartIdx = smlaCacheRunParam.nextTokensPerBatchOri * (-1) /
                                              smlaCacheRunParam.qSNumInOneBlock * smlaCacheRunParam.qSNumInOneBlock;
                if (smlaGs1LoopStartIdx > gS1StartIdx) {
                    smlaCacheRunParam.gs1LoopStartIdx = smlaGs1LoopStartIdx;
                }
            }
        }
    }

    int32_t smlaGs1LoopEndIdx = 0;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        smlaGs1LoopEndIdx = smlaCacheRunParam.actualS1Size;
    } else { // SWA/HCA
        // 不需要取topk, 每次计算gSize行, 循环qs次
        smlaGs1LoopEndIdx = (smlaCacheRunParam.actualS1Size + smlaCacheRunParam.qSNumInOneBlock - 1) /
                            smlaCacheRunParam.qSNumInOneBlock;
    }
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        smlaCacheRunParam.gs1LoopEndIdx = smlaGs1LoopEndIdx;
    } else { // 最后一个bn, 从数组下一个元素取值
        uint32_t smlaActualNextGs1Idx = s2EndIdx == 0 ? nextGs1Idx : nextGs1Idx + 1;
        smlaCacheRunParam.gs1LoopEndIdx = (nextGs1Idx == 0 && s2EndIdx == 0) ? smlaGs1LoopEndIdx : smlaActualNextGs1Idx;
    }

    if (smlaCacheRunParam.gs1LoopStartIdx > smlaCacheRunParam.gs1LoopEndIdx) {
        smlaCacheRunParam.gs1LoopStartIdx = smlaCacheRunParam.gs1LoopEndIdx;
    }
}

TEMPLATE_INTF
__aicore__ inline void ComputeSouterParam(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                          uint32_t sOuterLoopIdx)
{
    int64_t smlaCubeSOuterOffset = sOuterLoopIdx * smlaCacheRunParam.qSNumInOneBlock;
    if (smlaCacheRunParam.actualS1Size == 0) {
        smlaCacheRunParam.s1RealSize = 0;
        smlaCacheRunParam.mRealSize = 0;
    } else {
        smlaCacheRunParam.s1RealSize =
            Min(smlaCacheRunParam.qSNumInOneBlock, smlaCacheRunParam.actualS1Size - smlaCubeSOuterOffset);
        smlaCacheRunParam.mRealSize = smlaCacheRunParam.s1RealSize * smlaCacheConstInfo.gSize;
        if constexpr (IS_SPLIT_G) {
            smlaCacheRunParam.mRealSize = smlaCacheRunParam.s1RealSize * smlaCacheRunParam.gSplitSize;
        }
    }

    smlaCacheRunParam.cubeMOuterOffset = smlaCubeSOuterOffset * smlaCacheConstInfo.gSize;
    smlaCacheRunParam.halfMRealSize = (smlaCacheRunParam.mRealSize + 1) >> 1;
    smlaCacheRunParam.firstHalfMRealSize = smlaCacheRunParam.halfMRealSize;
    if (smlaCacheConstInfo.subBlockIdx == 1) {
        smlaCacheRunParam.halfMRealSize = smlaCacheRunParam.mRealSize - smlaCacheRunParam.halfMRealSize;
        smlaCacheRunParam.mOuterOffset = smlaCacheRunParam.cubeMOuterOffset + smlaCacheRunParam.firstHalfMRealSize;
    } else {
        smlaCacheRunParam.mOuterOffset = smlaCacheRunParam.cubeMOuterOffset;
    }

    smlaCacheRunParam.halfS1RealSize = (smlaCacheRunParam.s1RealSize + 1) >> 1;
    smlaCacheRunParam.firstHalfS1RealSize = smlaCacheRunParam.halfS1RealSize;
    if (smlaCacheConstInfo.subBlockIdx == 1) {
        smlaCacheRunParam.halfS1RealSize = smlaCacheRunParam.s1RealSize - smlaCacheRunParam.halfS1RealSize;
        smlaCacheRunParam.sOuterOffset =
            smlaCubeSOuterOffset + smlaCacheRunParam.firstHalfMRealSize / smlaCacheConstInfo.gSize;
    } else {
        smlaCacheRunParam.sOuterOffset = smlaCubeSOuterOffset;
    }
    smlaCacheRunParam.cubeSOuterOffset = smlaCubeSOuterOffset;
}

TEMPLATE_INTF
__aicore__ inline void LoopSOuterOffsetInit(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                            int32_t sIdx, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if ASCEND_IS_AIV {
        int64_t seqOffset = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            seqOffset = cuSeqlensQGm.GetValue(sIdx);
        } else {
            seqOffset = sIdx * smlaCacheConstInfo.s1Size;
        }

        int64_t attentionOutSeqOffset = seqOffset * smlaCacheConstInfo.n2GDv;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND || LAYOUT_T == SMLA_LAYOUT::TND) {
            smlaCacheRunParam.attentionOutOffset =
                attentionOutSeqOffset + smlaCacheRunParam.sOuterOffset * smlaCacheConstInfo.n2GDv +
                smlaCacheRunParam.n2oIdx * smlaCacheConstInfo.gDv + smlaCacheRunParam.goIdx * smlaCacheConstInfo.dSizeV;
        }
        if (smlaCacheConstInfo.subBlockIdx == 1) {
            smlaCacheRunParam.attentionOutOffset += smlaCacheRunParam.firstHalfMRealSize * smlaCacheConstInfo.dSizeV;
        }
        if (smlaCacheConstInfo.isSoftmaxLseEnable) {
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
                // [N2, T, G] (TND)
                smlaCacheRunParam.softmaxLseOffset =
                    smlaCacheRunParam.n2oIdx * smlaCacheConstInfo.s1Size * smlaCacheConstInfo.gSize +
                    (seqOffset + smlaCacheRunParam.sOuterOffset) * smlaCacheConstInfo.gSize;
            } else {
                // [B, N2, S1, G] (BSND)
                smlaCacheRunParam.softmaxLseOffset =
                    sIdx * smlaCacheConstInfo.n2Size * smlaCacheConstInfo.s1Size * smlaCacheConstInfo.gSize +
                    smlaCacheRunParam.n2oIdx * smlaCacheConstInfo.s1Size * smlaCacheConstInfo.gSize +
                    smlaCacheRunParam.sOuterOffset * smlaCacheConstInfo.gSize;
            }
            if (IS_SPLIT_G) {
                smlaCacheRunParam.softmaxLseOffset += smlaCacheRunParam.goIdx;
            }
            if (smlaCacheConstInfo.subBlockIdx == 1) {
                smlaCacheRunParam.softmaxLseOffset += smlaCacheRunParam.firstHalfMRealSize;
            }
        }
    }
}

TEMPLATE_INTF
__aicore__ inline bool ComputeParamS1(RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo,
                                      uint32_t sOuterLoopIdx, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::HCA_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
            int64_t smlaSkipThreshold = 0;
            if (smlaCacheRunParam.nextTokensPerBatchOri < 0 && smlaCacheRunParam.nextTokensPerBatchCmp < 0) {
                smlaSkipThreshold =
                    Min(-smlaCacheRunParam.nextTokensPerBatchOri, -smlaCacheRunParam.nextTokensPerBatchCmp);
            }
            if (smlaSkipThreshold > 0) {
                if (smlaCacheRunParam.s1oIdx <
                    smlaSkipThreshold / smlaCacheRunParam.qSNumInOneBlock * smlaCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        } else {
            if (smlaCacheRunParam.nextTokensPerBatchOri < 0) {
                if (smlaCacheRunParam.s1oIdx < (smlaCacheRunParam.nextTokensPerBatchOri * (-1)) /
                                                   smlaCacheRunParam.qSNumInOneBlock *
                                                   smlaCacheRunParam.qSNumInOneBlock) {
                    return true;
                }
            }
        }
    }

    ComputeSouterParam<TEMPLATE_INTF_ARGS>(smlaCacheRunParam, smlaCacheConstInfo, sOuterLoopIdx);

    LoopSOuterOffsetInit<TEMPLATE_INTF_ARGS>(smlaCacheRunParam, smlaCacheConstInfo, smlaCacheRunParam.boIdx,
                                             cuSeqlensQGm);
    return false;
}

TEMPLATE_INTF
__aicore__ inline bool ComputeLastBN(RunParamStr &smlaCacheRunParam, GlobalTensor<int32_t> &cuSeqlensQGm)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // TND格式下 相邻Batch中当actualSeqQlen相等时则返回true
        if (smlaCacheRunParam.boIdx > 0 &&
            cuSeqlensQGm.GetValue(smlaCacheRunParam.boIdx + 1) - cuSeqlensQGm.GetValue(smlaCacheRunParam.boIdx) == 0) {
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
                                         RunParamStr &smlaCacheRunParam, const ConstInfo &smlaCacheConstInfo)
{
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (smlaCacheRunParam.actualS2OriSize == 0) {
            smlaCacheRunParam.oriKvLoopEndIdx = 0;
            smlaCacheRunParam.cmpKvLoopEndIdx = 0;
            smlaCacheRunParam.s2LoopEndIdx = 0;
            smlaCacheRunParam.s2CmpLineStartIdx = 0;
            return true;
        }
    }
    uint32_t smlaS2BaseSize = smlaCacheConstInfo.s2BaseSize;

    // 计算topk length
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t smlaActualSeqQPrefixSum = cuSeqlensQGm.GetValue(smlaCacheRunParam.boIdx);
        smlaCacheRunParam.oriSparseBlockCount =
            smlaCacheConstInfo.hasOriTopkLength ?
                Min(oriTopkLengthGm.GetValue(smlaActualSeqQPrefixSum + smlaCacheRunParam.s1oIdx),
                    smlaCacheConstInfo.oriSparseBlockCount) :
                smlaCacheConstInfo.oriSparseBlockCount;
        smlaCacheRunParam.cmpSparseBlockCount =
            smlaCacheConstInfo.hasCmpTopkLength ?
                Min(cmpTopkLengthGm.GetValue(smlaActualSeqQPrefixSum + smlaCacheRunParam.s1oIdx),
                    smlaCacheConstInfo.cmpSparseBlockCount) :
                smlaCacheConstInfo.cmpSparseBlockCount;
    } else {
        uint64_t smlaBsndTopkIdx = smlaCacheRunParam.boIdx * smlaCacheConstInfo.s1Size + smlaCacheRunParam.s1oIdx;
        smlaCacheRunParam.oriSparseBlockCount =
            smlaCacheConstInfo.hasOriTopkLength ?
                Min(oriTopkLengthGm.GetValue(smlaBsndTopkIdx), smlaCacheConstInfo.oriSparseBlockCount) :
                smlaCacheConstInfo.oriSparseBlockCount;
        smlaCacheRunParam.cmpSparseBlockCount =
            smlaCacheConstInfo.hasCmpTopkLength ?
                Min(cmpTopkLengthGm.GetValue(smlaBsndTopkIdx), smlaCacheConstInfo.cmpSparseBlockCount) :
                smlaCacheConstInfo.cmpSparseBlockCount;
    }

    // orikv
    smlaCacheRunParam.s2OriLineStartIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        smlaCacheRunParam.cubeSOuterOffset - smlaCacheRunParam.preTokensPerBatchOri, 0,
        smlaCacheRunParam.actualS2OriSize);
    smlaCacheRunParam.s2OriLineEndIdx = ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>(
        smlaCacheRunParam.cubeSOuterOffset + smlaCacheRunParam.nextTokensPerBatchOri + smlaCacheRunParam.s1RealSize, 0,
        smlaCacheRunParam.actualS2OriSize);
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        int64_t oriSparseRangeLen = smlaCacheRunParam.s2OriLineEndIdx - smlaCacheRunParam.s2OriLineStartIdx;
        smlaCacheRunParam.s2OriLineStartIdx = 0;
        smlaCacheRunParam.s2OriLineEndIdx = Min(oriSparseRangeLen, smlaCacheRunParam.oriSparseBlockCount);
        smlaCacheRunParam.s2OriLineEndIdx = Min(smlaCacheRunParam.s2OriLineEndIdx, smlaCacheRunParam.actualS2OriSize);
    }
    smlaCacheRunParam.oriKvLoopEndIdx =
        (smlaCacheRunParam.s2OriLineEndIdx - smlaCacheRunParam.s2OriLineStartIdx + smlaS2BaseSize - 1) / smlaS2BaseSize;

    // cmpkv
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::SWA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        smlaCacheRunParam.s2CmpLineStartIdx = 0;
        smlaCacheRunParam.s2CmpLineEndIdx = 0;
        smlaCacheRunParam.cmpKvLoopEndIdx = 0;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::HCA_TEMPLATE_MODE) {
        smlaCacheRunParam.s2CmpLineStartIdx = 0;
        smlaCacheRunParam.s2CmpLineEndIdx =
            ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>((smlaCacheRunParam.cubeSOuterOffset + smlaCacheRunParam.s1RealSize +
                                                     smlaCacheRunParam.nextTokensPerBatchCmp) /
                                                        smlaCacheConstInfo.cmpRatio,
                                                    0, smlaCacheRunParam.actualS2CmpSize);
        smlaCacheRunParam.s2CmpLineEndIdx = Min(smlaCacheRunParam.s2CmpLineEndIdx, smlaCacheRunParam.actualS2CmpSize);
        smlaCacheRunParam.cmpKvLoopEndIdx = (smlaCacheRunParam.s2CmpLineEndIdx + smlaS2BaseSize - 1) / smlaS2BaseSize;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                         TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        smlaCacheRunParam.s2CmpLineStartIdx = 0;
        smlaCacheRunParam.s2CmpLineEndIdx =
            ClipSInnerTokenCube<TEMPLATE_INTF_ARGS>((smlaCacheRunParam.cubeSOuterOffset + smlaCacheRunParam.s1RealSize +
                                                     smlaCacheRunParam.nextTokensPerBatchCmp) /
                                                        smlaCacheConstInfo.cmpRatio,
                                                    0, smlaCacheRunParam.actualS2CmpSize);
        smlaCacheRunParam.s2CmpLineEndIdx =
            Min(smlaCacheRunParam.s2CmpLineEndIdx, smlaCacheRunParam.cmpSparseBlockCount);
        smlaCacheRunParam.s2CmpLineEndIdx = Min(smlaCacheRunParam.s2CmpLineEndIdx, smlaCacheRunParam.actualS2CmpSize);
        smlaCacheRunParam.cmpKvLoopEndIdx = (smlaCacheRunParam.s2CmpLineEndIdx + smlaS2BaseSize - 1) / smlaS2BaseSize;
    }

    smlaCacheRunParam.s2LoopEndIdx = smlaCacheRunParam.oriKvLoopEndIdx + smlaCacheRunParam.cmpKvLoopEndIdx;
    return (smlaCacheRunParam.s2LoopEndIdx == 0);
}

TEMPLATE_INTF
__aicore__ inline void InitTaskParamByRun(const RunParamStr &smlaCacheRunParam, RunInfo &smlaCacheRunInfo,
                                          const ConstInfo &smlaCacheConstInfo)
{
    smlaCacheRunInfo.boIdx = smlaCacheRunParam.boIdx;
    smlaCacheRunInfo.preTokensPerBatchOri = smlaCacheRunParam.preTokensPerBatchOri;
    smlaCacheRunInfo.nextTokensPerBatchOri = smlaCacheRunParam.nextTokensPerBatchOri;
    smlaCacheRunInfo.actualS1Size = smlaCacheRunParam.actualS1Size;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        smlaCacheRunInfo.actualS2CmpSize = smlaCacheRunParam.actualS2CmpSize;
        smlaCacheRunInfo.cmpResidual = smlaCacheRunParam.cmpResidual;
    }
    smlaCacheRunInfo.softmaxLseOffset = smlaCacheRunParam.softmaxLseOffset;
    smlaCacheRunInfo.qSNumInOneBlock = smlaCacheRunParam.qSNumInOneBlock;
    smlaCacheRunInfo.oriKvLoopEndIdx = smlaCacheRunParam.oriKvLoopEndIdx;
    smlaCacheRunInfo.cmpKvLoopEndIdx = smlaCacheRunParam.cmpKvLoopEndIdx;
    smlaCacheRunInfo.isCmp = smlaCacheRunInfo.s2LoopCount >= smlaCacheRunInfo.oriKvLoopEndIdx;
    smlaCacheRunInfo.oriSparseBlockCount = smlaCacheRunParam.oriSparseBlockCount;
    smlaCacheRunInfo.cmpSparseBlockCount = smlaCacheRunParam.cmpSparseBlockCount;
}

#endif // SPARSE_FLASH_MLA_KVCACHE_H
