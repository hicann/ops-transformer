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
__aicore__ inline int64_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                    GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                    int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    } else {
        return defaultSize;
    }
}

__aicore__ inline int64_t ConvertS2MetadataBlockToToken(const RunParamStr &runParam, const ConstInfo &constInfo,
                                                        uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    int64_t reductionBlockSize = runParam.baseBlockNumPerReductionBlock * s2BaseSize;
    int64_t oriReductionBlockNum = (oriLen + reductionBlockSize - 1) / reductionBlockSize;
    int64_t reductionBlockIdx = static_cast<int64_t>(s2BlockIdx);
    if (reductionBlockIdx < oriReductionBlockNum) {
        int64_t oriToken = reductionBlockIdx * reductionBlockSize;
        return oriToken < oriLen ? oriToken : oriLen;
    }
    int64_t cmpToken = (reductionBlockIdx - oriReductionBlockNum) * reductionBlockSize;
    return oriLen + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

__aicore__ inline bool ApplyS2MetadataRange(RunParamStr &runParam, ConstInfo &constInfo, int64_t s2StartPoint,
                                            int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask)
{
    int64_t oriStart = runParam.s2OriLineStartIdx;
    int64_t oriEnd = runParam.s2OriLineEndIdx;
    int64_t oriLen = oriEnd - oriStart;
    int64_t cmpStart = runParam.s2CmpLineStartIdx;
    int64_t cmpEnd = runParam.s2CmpLineEndIdx;
    int64_t cmpLen = cmpEnd - cmpStart;
    int64_t totalLen = oriLen + cmpLen;

    int64_t effectiveS2EndPoint = (isLastS2RangeTask && s2EndPoint == 0) ? totalLen : s2EndPoint;
    int64_t rangeStart = isFirstS2RangeTask ? s2StartPoint : 0;
    rangeStart = rangeStart < 0 ? 0 : rangeStart;
    rangeStart = rangeStart < totalLen ? rangeStart : totalLen;
    int64_t rangeEnd = isLastS2RangeTask ? effectiveS2EndPoint : totalLen;
    rangeEnd = rangeEnd < 0 ? 0 : rangeEnd;
    rangeEnd = rangeEnd < totalLen ? rangeEnd : totalLen;
    if (rangeEnd <= rangeStart) {
        runParam.oriKvLoopEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        runParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    runParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    runParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < oriLen ? rangeStart : oriLen;
    int64_t oriRangeEnd = rangeEnd < oriLen ? rangeEnd : oriLen;
    runParam.s2OriLineStartIdx = oriStart + oriRangeStart;
    runParam.s2OriLineEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > oriLen ? rangeStart - oriLen : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > oriLen ? rangeEnd - oriLen : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    runParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    runParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriRangeLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpRangeLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    runParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return runParam.s2LoopEndIdx == 0;
}

__aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam, const ConstInfo &constInfo)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = (runInfo.s2LoopCount >= runParam.oriKvLoopEndIdx) ?
                               (runInfo.s2LoopCount - runParam.oriKvLoopEndIdx) :
                               runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
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
__aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo, const ConstInfo &constInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo, constInfo);
}

TEMPLATE_INTF
__aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                  int64_t s2LoopLimit, int64_t multiCoreInnerIdx, const ConstInfo &constInfo)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2OriLineStartIdx;
        runInfo.s2EndIdx = runParam.s2OriLineEndIdx;
    } else {
        runInfo.s2StartIdx = runParam.s2CmpLineStartIdx;
        runInfo.s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.firstFdDataWorkspaceIdx = runParam.firstFdDataWorkspaceIdx;
    runInfo.isCrossCoreSplit = runParam.isCrossCoreSplit;
    runInfo.s2SplitIdx = runParam.s2SplitIdx;
    runInfo.isFirstS2SplitCore = runParam.isFirstS2SplitCore;
    int64_t safeBaseBlockNum =
        runParam.baseBlockNumPerReductionBlock > 0 ? runParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t baseBlockIdInReduceBlock = s2LoopCount % safeBaseBlockNum;
    runInfo.reduceBlockId = s2LoopCount / safeBaseBlockNum;
    runInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    runInfo.isLastBase = baseBlockIdInReduceBlock == safeBaseBlockNum - 1LL || s2LoopCount == s2LoopLimit;
    runInfo.needReduce = runInfo.reduceBlockId > 0;
    ComputeBmm1Tail(runInfo, runParam, constInfo);
    InitUniqueRunInfo<TEMPLATE_INTF_ARGS>(runParam, runInfo, constInfo);
}

TEMPLATE_INTF
__aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam,
                                                const ConstInfo &constInfo, int32_t aicIdx)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t halfG = (constInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
        runParam.gSplitSize = (aicIdx % 2 == 0) ? halfG : (constInfo.gSize - halfG);
    } else {
        runParam.goIdx = 0;
        runParam.gSplitSize = constInfo.gSize;
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
