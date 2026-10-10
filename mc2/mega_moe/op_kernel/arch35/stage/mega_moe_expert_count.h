/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_EXPERT_COUNT_H
#define MEGA_MOE_EXPERT_COUNT_H

#include "kernel_operator.h"
#include "../common/mega_moe_constants.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

// count 整表检查失败后的重试间隔。
constexpr int64_t COUNT_TABLE_POLL_BACKOFF_CYCLES = 500;

// 只检查 UB 快照的高 8 位轮次标签，不修改 count；成功后由前缀和计算去除标签。
// expectedSyncRoundTag 使用 GetSyncRoundTag 返回的高 8 位标签；pending[0] 为零时才可消费快照。
// pending 的各 lane 累积所有向量的失败结果，尾部无效 lane 不参与比较或写回。
__simd_vf__ inline void CheckCountRoundTagsVF(__ubuf__ uint32_t* counts, __ubuf__ uint32_t* pending,
                                              uint32_t elementCount, uint32_t expectedSyncRoundTag)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(uint32_t);
    AscendC::Reg::RegTensor<uint32_t> countReg;
    AscendC::Reg::RegTensor<uint32_t> epochReg;
    AscendC::Reg::RegTensor<uint32_t> pendingReg;
    AscendC::Reg::RegTensor<uint32_t> oneReg;
    AscendC::Reg::RegTensor<uint32_t> resultReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(pendingReg, 0U, fullMask);
    AscendC::Reg::Duplicate(oneReg, 1U, fullMask);
    for (uint32_t offset = 0U; offset < elementCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - offset;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<uint32_t>(validCount);
        AscendC::Reg::MaskReg mismatchMask;
        AscendC::Reg::LoadAlign(countReg, counts + offset);
        AscendC::Reg::ShiftRights(epochReg, countReg, static_cast<int16_t>(24), activeMask);
        AscendC::Reg::Compares<uint32_t, AscendC::CMPMODE::NE>(mismatchMask, epochReg, expectedSyncRoundTag >> 24U,
                                                               activeMask);
        AscendC::Reg::Select(pendingReg, oneReg, pendingReg, mismatchMask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(resultReg, pendingReg, fullMask);
    AscendC::Reg::StoreAlign(pending, resultReg, firstLaneMask);
}

// MTE GM count 表按 512B 槽存放；搬入后的 counts 为连续 UB 表。
// pending 为独立、32B 对齐的 UB 暂存区（至少一个 int32）。
// 首次立即读取；失败后等待 500 cycle，再重读整表。成功快照直接交给后续 VF cumsum。
// 调用方保证在 AIV 上执行，且 elementCount > 0。
__aicore__ inline void WaitForCountTable(const AscendC::GlobalTensor<int32_t>& countGm,
                                         const AscendC::LocalTensor<int32_t>& counts,
                                         const AscendC::LocalTensor<int32_t>& pending, uint32_t elementCount,
                                         uint32_t expectedSyncRoundTag)
{
    // 首次搬运前保护此前的 Vector 访问；重试时上一轮 V_S 和 Scalar 判断已保证 VF 完成。
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID2>();
    while (true) {
        // Compact 将每个 GM 槽首部的 4B 紧密排入 UB，保持后续 epoch 校验和前缀和布局不变。
        AscendC::DataCopyPad<int32_t, AscendC::PaddingMode::Compact>(
            counts, countGm,
            {static_cast<uint16_t>(elementCount), sizeof(int32_t),
             static_cast<int64_t>(MTE_COUNT_SLOT_BYTES) - static_cast<int64_t>(sizeof(int32_t)), 0U, 0U},
            {true, 0U, 0U, 0U});
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        asc_vf_call<CheckCountRoundTagsVF>(reinterpret_cast<__ubuf__ uint32_t*>(counts.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ uint32_t*>(pending.GetPhyAddr()), elementCount,
                                           expectedSyncRoundTag);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        if (pending.GetValue(0U) == 0) {
            return;
        }
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < COUNT_TABLE_POLL_BACKOFF_CYCLES) {
        }
    }
}

/*
 * MegaMoe count 表的融合版本：先复用 int32 prefix-scan，再在同一次 VF 中提取每个专家的累计尾值
 * 并作差。相比先调用通用前缀和、再启动第二个 VF，融合实现省去一次 asc_vf_call 和外部流水同步。
 */
__simd_vf__ inline void ComputeExpertCountTablesVF(__ubuf__ int32_t* count, __ubuf__ int32_t* expertCounts,
                                                   uint32_t elementCount, uint32_t expertCount, uint32_t worldSize,
                                                   uint32_t maxOutputSize)
{
    // 去除快照中的高 8 位轮次标签，只累加低 24 位 count。
    InclusivePrefixSumInt32<0x00FFFFFF>(count, elementCount);

    // prefix 已写回 UB；后续从同一区域 Gather 前建立 VEC_STORE -> VEC_LOAD 可见性。
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> expertEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> previousEndPrefixReg;
    AscendC::Reg::RegTensor<int32_t> expertCountReg;
    AscendC::Reg::RegTensor<int32_t> previousVectorEndReg;
    AscendC::Reg::RegTensor<int32_t> maxOutputSizeReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> gatherIndexReg;
    AscendC::Reg::RegTensor<int32_t> previousLaneIndexReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    int32_t maxOutputSizeInt32 = static_cast<int32_t>(maxOutputSize);
    AscendC::Reg::Arange(laneIndexReg, 0);
    AscendC::Reg::Duplicate(previousVectorEndReg, 0, fullMask);
    AscendC::Reg::Duplicate(maxOutputSizeReg, maxOutputSizeInt32, fullMask);
    for (uint32_t expertOffset = 0U; expertOffset < expertCount; expertOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = expertCount - expertOffset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);

        AscendC::Reg::Adds(gatherIndexReg, laneIndexReg, static_cast<int32_t>(expertOffset), fullMask);
        AscendC::Reg::Muls(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize), fullMask);
        AscendC::Reg::Adds(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize - 1U), fullMask);
        AscendC::Reg::Gather(expertEndPrefixReg, count,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(gatherIndexReg), activeMask);

        AscendC::Reg::Adds(previousLaneIndexReg, laneIndexReg, -1, fullMask);
        AscendC::Reg::Maxs(previousLaneIndexReg, previousLaneIndexReg, 0, fullMask);
        AscendC::Reg::Gather(previousEndPrefixReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(previousLaneIndexReg));
        AscendC::Reg::Select(previousEndPrefixReg, previousVectorEndReg, previousEndPrefixReg, firstLaneMask);
        AscendC::Reg::MaskReg overflowMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, expertEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(expertCountReg, maxOutputSizeReg, expertEndPrefixReg, overflowMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(overflowMask, previousEndPrefixReg, maxOutputSizeInt32,
                                                              activeMask);
        AscendC::Reg::Select(previousEndPrefixReg, maxOutputSizeReg, previousEndPrefixReg, overflowMask);
        AscendC::Reg::Sub(expertCountReg, expertCountReg, previousEndPrefixReg, activeMask);
        AscendC::Reg::StoreAlign(expertCounts + expertOffset, expertCountReg, activeMask);

        AscendC::Reg::Duplicate(previousLaneIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        AscendC::Reg::Gather(previousVectorEndReg, expertEndPrefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(previousLaneIndexReg));
    }
}

__aicore__ inline void ComputeExpertCountTables(LocalTensor<int32_t> countTensor,
                                                LocalTensor<int32_t> expertCountTensor, uint32_t expertCount,
                                                uint32_t worldSize, uint32_t maxOutputSize)
{
    __ubuf__ int32_t* count = reinterpret_cast<__ubuf__ int32_t*>(countTensor.GetPhyAddr());
    __ubuf__ int32_t* expertCounts = reinterpret_cast<__ubuf__ int32_t*>(expertCountTensor.GetPhyAddr());
    asc_vf_call<ComputeExpertCountTablesVF>(count, expertCounts, expertCount * worldSize, expertCount, worldSize,
                                            maxOutputSize);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_EXPERT_COUNT_H
