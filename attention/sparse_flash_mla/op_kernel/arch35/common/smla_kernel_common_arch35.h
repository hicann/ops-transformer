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
 * \file smla_kernel_common_arch35.h
 * \brief sparse_flash_mla CSA/SWA 两个 kernel 共用的主循环辅助函数（header-only）。
 *        原为 SparseFlashMlaCsaKernel / SparseFlashMlaSwaKernel 的私有成员函数，
 *        现将完全一致的实现提取为自由函数，成员变量依赖通过函数入参传入。
 */

#ifndef SMLA_KERNEL_COMMON_ARCH35_H
#define SMLA_KERNEL_COMMON_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../sparse_flash_mla_common_arch35.h"
#include "../sparse_flash_mla_kvcache.h"
#include "../util_regbase.h"
#include "smla_metadata_common.h"
#include "static_buffer.h"
#include "flash_decode.h"
#include "../../../../common/op_kernel/attn_buffer.h"

using namespace regbaseutil;
using namespace optiling;
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace SMLAKernel;
using namespace AttentionCommon;

// ===================== 无模板参数的公共函数 =====================
__aicore__ inline int64_t GetSmlaSeqLen(int32_t batchIndex, bool useExplicitLength, bool useCumulativeLength,
                                        GlobalTensor<int32_t> &explicitLengthGm,
                                        GlobalTensor<int32_t> &cumulativeLengthGm, int64_t fallbackLength)
{
    if (useExplicitLength) {
        return explicitLengthGm.GetValue(batchIndex);
    } else if (useCumulativeLength) {
        return cumulativeLengthGm.GetValue(batchIndex + 1) - cumulativeLengthGm.GetValue(batchIndex);
    } else {
        return fallbackLength;
    }
}

__aicore__ inline int64_t ConvertS2MetadataBlockToToken(const RunParamStr &smlaRunParam, const ConstInfo &constInfo,
                                                        uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t smlaOriLength = smlaRunParam.s2OriLineEndIdx - smlaRunParam.s2OriLineStartIdx;
    int64_t cmpLen = smlaRunParam.s2CmpLineEndIdx - smlaRunParam.s2CmpLineStartIdx;
    int64_t reductionBlockSize = smlaRunParam.baseBlockNumPerReductionBlock * s2BaseSize;
    int64_t oriReductionBlockNum = (smlaOriLength + reductionBlockSize - 1) / reductionBlockSize;
    int64_t reductionBlockIdx = static_cast<int64_t>(s2BlockIdx);
    if (reductionBlockIdx < oriReductionBlockNum) {
        int64_t oriToken = reductionBlockIdx * reductionBlockSize;
        return oriToken < smlaOriLength ? oriToken : smlaOriLength;
    }
    int64_t cmpToken = (reductionBlockIdx - oriReductionBlockNum) * reductionBlockSize;
    return smlaOriLength + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

__aicore__ inline bool ApplyS2MetadataRange(RunParamStr &smlaRunParam, ConstInfo &constInfo, int64_t s2StartPoint,
                                            int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask)
{
    int64_t oriStart = smlaRunParam.s2OriLineStartIdx;
    int64_t oriEnd = smlaRunParam.s2OriLineEndIdx;
    int64_t smlaOriLength = oriEnd - oriStart;
    int64_t cmpStart = smlaRunParam.s2CmpLineStartIdx;
    int64_t cmpEnd = smlaRunParam.s2CmpLineEndIdx;
    int64_t cmpLen = cmpEnd - cmpStart;
    int64_t totalLen = smlaOriLength + cmpLen;

    int64_t effectiveS2EndPoint = (isLastS2RangeTask && s2EndPoint == 0) ? totalLen : s2EndPoint;
    int64_t rangeStart = isFirstS2RangeTask ? s2StartPoint : 0;
    rangeStart = rangeStart < 0 ? 0 : rangeStart;
    rangeStart = rangeStart < totalLen ? rangeStart : totalLen;
    int64_t rangeEnd = isLastS2RangeTask ? effectiveS2EndPoint : totalLen;
    rangeEnd = rangeEnd < 0 ? 0 : rangeEnd;
    rangeEnd = rangeEnd < totalLen ? rangeEnd : totalLen;
    if (rangeEnd <= rangeStart) {
        smlaRunParam.oriKvLoopEndIdx = 0;
        smlaRunParam.cmpKvLoopEndIdx = 0;
        smlaRunParam.s2LoopEndIdx = 0;
        smlaRunParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    smlaRunParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    smlaRunParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < smlaOriLength ? rangeStart : smlaOriLength;
    int64_t oriRangeEnd = rangeEnd < smlaOriLength ? rangeEnd : smlaOriLength;
    smlaRunParam.s2OriLineStartIdx = oriStart + oriRangeStart;
    smlaRunParam.s2OriLineEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > smlaOriLength ? rangeStart - smlaOriLength : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > smlaOriLength ? rangeEnd - smlaOriLength : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    smlaRunParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    smlaRunParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriRangeLen = smlaRunParam.s2OriLineEndIdx - smlaRunParam.s2OriLineStartIdx;
    int64_t cmpRangeLen = smlaRunParam.s2CmpLineEndIdx - smlaRunParam.s2CmpLineStartIdx;
    smlaRunParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    smlaRunParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    smlaRunParam.s2LoopEndIdx = smlaRunParam.oriKvLoopEndIdx + smlaRunParam.cmpKvLoopEndIdx;
    return smlaRunParam.s2LoopEndIdx == 0;
}

__aicore__ inline void ComputeBmm1Tail(RunInfo &smlaRunInfo, RunParamStr &smlaRunParam, const ConstInfo &constInfo)
{
    // ------------------------S1 Base Related---------------------------
    smlaRunInfo.s1RealSize = smlaRunParam.s1RealSize;
    smlaRunInfo.halfS1RealSize = smlaRunParam.halfS1RealSize;
    smlaRunInfo.firstHalfS1RealSize = smlaRunParam.firstHalfS1RealSize;
    smlaRunInfo.mRealSize = smlaRunParam.mRealSize;
    smlaRunInfo.halfMRealSize = smlaRunParam.halfMRealSize;
    smlaRunInfo.firstHalfMRealSize = smlaRunParam.firstHalfMRealSize;

    smlaRunInfo.vec2MBaseSize = smlaRunInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    smlaRunInfo.s2RealSize = constInfo.s2BaseSize;
    smlaRunInfo.s2AlignedSize = smlaRunInfo.s2RealSize;
    int64_t curS2LoopCnt = (smlaRunInfo.s2LoopCount >= smlaRunParam.oriKvLoopEndIdx) ?
                               (smlaRunInfo.s2LoopCount - smlaRunParam.oriKvLoopEndIdx) :
                               smlaRunInfo.s2LoopCount;
    if (smlaRunInfo.s2StartIdx + (curS2LoopCnt + 1) * smlaRunInfo.s2RealSize > smlaRunInfo.s2EndIdx) {
        smlaRunInfo.s2RealSize = smlaRunInfo.s2EndIdx - curS2LoopCnt * smlaRunInfo.s2RealSize - smlaRunInfo.s2StartIdx;
        smlaRunInfo.s2AlignedSize = Align(smlaRunInfo.s2RealSize);
    }
}

__aicore__ inline void ParseFdRunInfo(FdRunInfo &fdRunInfo, const ConstInfo &constInfo,
                                      GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t aivIdx = static_cast<uint32_t>(constInfo.aivIdx);
    fdRunInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_CORE_ENABLE_INDEX, true)) != 0;
    if (!fdRunInfo.coreEnable) {
        return;
    }
    fdRunInfo.bn2Idx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_BN2_IDX_INDEX, true));
    fdRunInfo.mIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_IDX_INDEX, true));
    fdRunInfo.workspaceIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_IDX_INDEX, true));
    fdRunInfo.workspaceNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_NUM_INDEX, true));
    fdRunInfo.mStartIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_START_INDEX, true));
    fdRunInfo.mNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_NUM_INDEX, true));
}

// ===================== 需要 TEMPLATE_INTF 的公共函数 =====================
TEMPLATE_INTF
__aicore__ inline void ComputeConstexpr(ConstInfo &constInfo)
{
    // 计算轴的乘积
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

TEMPLATE_INTF
__aicore__ inline void InitUniqueRunInfo(const RunParamStr &smlaRunParam, RunInfo &smlaRunInfo,
                                         const ConstInfo &constInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(smlaRunParam, smlaRunInfo, constInfo);
}

TEMPLATE_INTF
__aicore__ inline void SetRunInfo(RunInfo &smlaRunInfo, RunParamStr &smlaRunParam, int64_t smlaTaskId,
                                  int64_t smlaS2LoopCount, int64_t smlaS2LoopLimit, int64_t smlaCoreInnerIndex,
                                  const ConstInfo &constInfo)
{
    if (smlaS2LoopCount < smlaRunParam.oriKvLoopEndIdx) {
        smlaRunInfo.s2StartIdx = smlaRunParam.s2OriLineStartIdx;
        smlaRunInfo.s2EndIdx = smlaRunParam.s2OriLineEndIdx;
    } else {
        smlaRunInfo.s2StartIdx = smlaRunParam.s2CmpLineStartIdx;
        smlaRunInfo.s2EndIdx = smlaRunParam.s2CmpLineEndIdx;
    }
    smlaRunInfo.s2LoopCount = smlaS2LoopCount;
    if (smlaRunInfo.multiCoreInnerIdx != smlaCoreInnerIndex) {
        smlaRunInfo.s1oIdx = smlaRunParam.s1oIdx;
        smlaRunInfo.boIdx = smlaRunParam.boIdx;
        smlaRunInfo.n2oIdx = smlaRunParam.n2oIdx;
        smlaRunInfo.goIdx = smlaRunParam.goIdx;
        smlaRunInfo.multiCoreInnerIdx = smlaCoreInnerIndex;
        smlaRunInfo.multiCoreIdxMod2 = smlaCoreInnerIndex & 1;
        smlaRunInfo.multiCoreIdxMod3 = smlaCoreInnerIndex % 3; // 3：获取大小为3的组内的索引
    }

    smlaRunInfo.taskId = smlaTaskId;
    smlaRunInfo.taskIdMod2 = smlaTaskId & 1;
    smlaRunInfo.taskIdMod3 = smlaTaskId % 3; // 3：同上
    smlaRunInfo.s2LoopLimit = smlaS2LoopLimit;

    smlaRunInfo.actualS1Size = smlaRunParam.actualS1Size;
    smlaRunInfo.attentionOutOffset = smlaRunParam.attentionOutOffset;
    smlaRunInfo.sOuterOffset = smlaRunParam.sOuterOffset;
    smlaRunInfo.firstFdDataWorkspaceIdx = smlaRunParam.firstFdDataWorkspaceIdx;
    smlaRunInfo.isCrossCoreSplit = smlaRunParam.isCrossCoreSplit;
    smlaRunInfo.s2SplitIdx = smlaRunParam.s2SplitIdx;
    smlaRunInfo.isFirstS2SplitCore = smlaRunParam.isFirstS2SplitCore;
    int64_t safeBaseBlockNum =
        smlaRunParam.baseBlockNumPerReductionBlock > 0 ? smlaRunParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t reductionLoopCount = smlaS2LoopCount;
    if constexpr (IS_BATCH_CONSISTENCY) {
        // 进入 CMP 时补齐规约计数，不增加实际计算。
        if (smlaS2LoopCount >= smlaRunParam.oriKvLoopEndIdx) {
            reductionLoopCount +=
                (safeBaseBlockNum - smlaRunParam.oriKvLoopEndIdx % safeBaseBlockNum) % safeBaseBlockNum;
        }
    }
    int64_t baseBlockIdInReduceBlock = reductionLoopCount % safeBaseBlockNum;
    smlaRunInfo.reduceBlockId = reductionLoopCount / safeBaseBlockNum;
    smlaRunInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    smlaRunInfo.isLastBase = baseBlockIdInReduceBlock == safeBaseBlockNum - 1LL || smlaS2LoopCount == smlaS2LoopLimit;
    if constexpr (IS_BATCH_CONSISTENCY) {
        smlaRunInfo.isLastBase = smlaRunInfo.isLastBase || smlaS2LoopCount + 1 == smlaRunParam.oriKvLoopEndIdx;
    }
    smlaRunInfo.needReduce = smlaRunInfo.reduceBlockId > 0;
    ComputeBmm1Tail(smlaRunInfo, smlaRunParam, constInfo);
    InitUniqueRunInfo<TEMPLATE_INTF_ARGS>(smlaRunParam, smlaRunInfo, constInfo);
}

TEMPLATE_INTF
__aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &smlaRunParam,
                                                const ConstInfo &constInfo, int32_t aicIdx)
{
    // GS1合轴, 不切G, 只切S1
    smlaRunParam.s1oIdx = gS1Index * smlaRunParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t halfG = (constInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        smlaRunParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
        smlaRunParam.gSplitSize = (aicIdx % 2 == 0) ? halfG : (constInfo.gSize - halfG); // 2：AIC切分数量
    } else {
        smlaRunParam.goIdx = 0;
        smlaRunParam.gSplitSize = constInfo.gSize;
    }
}

// ===================== 依赖 Cube/Vec Block 的公共函数 =====================
template <typename T, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FreeEvent(fa_base_matmul::StaticBuffer<T> (&bmm1Buffers)[2], CubeBlockType &cubeBlock,
                                 VecBlockType &vecBlock, ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[0].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[0].idx) + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[1].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[1].idx) + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
        cubeBlock.FreeEvent();
    } else {
        vecBlock.FreeEvent(constInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void InitLocalBuffer(VecBlockType &vecBlock, CubeBlockType &cubeBlock, ConstInfo &constInfo,
                                       uint32_t vUbBase, uint32_t l1CubeBase)
{
    vecBlock.InitLocalBuffer(constInfo, vUbBase);
    cubeBlock.InitLocalBuffer(l1CubeBase);
}

#endif // SMLA_KERNEL_COMMON_ARCH35_H
