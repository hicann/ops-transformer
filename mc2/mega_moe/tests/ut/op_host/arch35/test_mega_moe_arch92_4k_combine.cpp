/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include "../../../../op_kernel/arch92/common/mega_moe_layout_arch92.h"

namespace MegaMoeUT {
using namespace MegaMoeImpl;

static constexpr CclCommLayoutParams MakeRowComm(int64_t hidden, uint32_t elementsPerByte)
{
    CclCommLayoutParams p{};
    p.numMaxTokensPerRank = 257;
    p.topK = 6;
    p.h = hidden;
    p.moeExpertPerRank = 24;
    p.epWorldSize = 4;
    p.yDtypeSize = SIZE_BF_16;
    p.elemsPerByte = elementsPerByte;
    p.dispatchStorage = {static_cast<uint32_t>(MemoryLayoutDetail::BITS_PER_BYTE) / elementsPerByte};
    p.topoType = TOPO_TYPE_MTE;
    p.serverNum = 1;
    p.combineRowAlignmentBytes = CommPacketArch92::COMBINE_ROW_ALIGNMENT_BYTES;
    return p;
}

TEST(MegaMoeArch92RowCombine, RowAddressesAlignWithoutIncreasingPayload)
{
    for (int64_t h : {1088, 4096, 5120, 7168, 8192}) {
        for (uint32_t elemsPerByte : {1U, 2U}) {
            const auto p = MakeRowComm(h, elemsPerByte);
            const auto layout = CalcMteCclCommLayout(p);
            const int64_t payload = h * SIZE_BF_16;
            const int64_t alignment = p.combineRowAlignmentBytes;
            EXPECT_EQ(CalcCombineTokenBytes(h, SIZE_BF_16, false), payload);
            EXPECT_EQ(layout.combineSend.offsetBytes % alignment, 0);
            EXPECT_EQ(layout.combineRowStrideBytes, MemoryLayoutDetail::CeilAlign(payload, alignment));
            EXPECT_EQ(layout.combineSend.sizeBytes, p.numMaxTokensPerRank * p.topK * layout.combineRowStrideBytes);
            for (int64_t row : {int64_t{0}, p.numMaxTokensPerRank * p.topK - 1}) {
                EXPECT_EQ((layout.combineSend.offsetBytes + row * layout.combineRowStrideBytes) % alignment, 0);
                EXPECT_LE(row * layout.combineRowStrideBytes + payload, layout.combineSend.sizeBytes);
            }
        }
    }
    constexpr auto p = MakeRowComm(7168, 1);
    constexpr auto layout = CalcMteCclCommLayout(p);
    static_assert(layout.combineRowStrideBytes == 16384);
    static_assert(CalcCombineTokenBytes(p.h, SIZE_BF_16, false) == 14336);
}

TEST(MegaMoeArch92RowCombine, CapacityIncludesAlignmentOfTheActualPeerBase)
{
    constexpr auto p = MakeRowComm(7168, 2);
    constexpr auto layout = CalcMteCclCommLayout(p);
    constexpr int64_t alignment = p.combineRowAlignmentBytes;
    for (int64_t residue : {0, 1, 32, 512, 2048, 4095}) {
        const int64_t window = 0x100000000LL + residue;
        const int64_t dataBase = MemoryLayoutDetail::CeilAlign(window + EXCEPTION_DUMP_REGION_SIZE, alignment);
        EXPECT_EQ((dataBase + layout.combineSend.offsetBytes) % alignment, 0);
        EXPECT_LE(dataBase - window + layout.totalSize, CalcCclComputeOffset(p));
    }
}

TEST(MegaMoeArch92RowCombine, DisabledPolicyKeepsTheOriginalMteCapacity)
{
    for (int64_t bs : {1, 72, 257, 2048, 8192}) {
        for (int64_t h : {1088, 4096, 7168, 8192}) {
            auto p = MakeRowComm(h, 1);
            p.numMaxTokensPerRank = bs;
            p.combineRowAlignmentBytes = 0U;
            const int64_t indexBytes = bs * p.topK <= 32768 ? 2 : 4;
            const int64_t routeBytes = MemoryLayoutDetail::CeilAlign(
                p.moeExpertPerRank * p.epWorldSize * MemoryLayoutDetail::CeilAlign(bs * indexBytes, 32), 512);
            const int64_t tokenBytes = MemoryLayoutDetail::CeilAlign(
                MemoryLayoutDetail::CeilAlign(h, 256) + MemoryLayoutDetail::CeilDiv(h, 32), 32);
            const int64_t legacy = EXCEPTION_DUMP_REGION_SIZE + PEERMEM_DATA_OFFSET + PEERMEM_MTE_COUNT_REGION_SIZE +
                                   routeBytes + MemoryLayoutDetail::CeilAlign(bs * tokenBytes, 512) +
                                   MemoryLayoutDetail::CeilAlign(bs * p.topK * h * SIZE_BF_16, 512);
            EXPECT_EQ(CalcCclComputeOffset(p), legacy);
        }
    }
}

TEST(MegaMoeArch92RowCombine, QuantizedAndUrmaLayoutsKeepTheirOriginalGeometry)
{
    static_assert(CommPacketArch92::GetCombineRowAlignmentBytes(false, false) == 0U);
    static_assert(CommPacketArch92::GetCombineRowAlignmentBytes(true, true) == 0U);
    static_assert(CommPacketArch92::GetCombineRowAlignmentBytes(true, false) ==
                  (CommPacketArch92::ROW_MODE == 0U ? 0U : CommPacketArch92::COMBINE_ROW_ALIGNMENT_BYTES));
    for (int64_t topology : {TOPO_TYPE_MTE, TOPO_TYPE_URMA, TOPO_TYPE_URMA_ROCE}) {
        auto p = MakeRowComm(7168, 1);
        p.topoType = topology;
        p.serverNum = 2;
        p.isQuantCombine = topology == TOPO_TYPE_MTE;
        p.combineRowAlignmentBytes =
            CommPacketArch92::GetCombineRowAlignmentBytes(topology == TOPO_TYPE_MTE, p.isQuantCombine);
        EXPECT_EQ(p.combineRowAlignmentBytes, 0U);
        const auto layout = CalcCclCommLayout(p);
        EXPECT_EQ(layout.combineRowStrideBytes, CalcCombineTokenBytes(p.h, p.yDtypeSize, p.isQuantCombine));
        const int64_t exceptionBytes = topology == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
        EXPECT_EQ(CalcCclComputeOffset(p),
                  MemoryLayoutDetail::CeilAlign(exceptionBytes + layout.totalSize, MemoryLayoutDetail::ALIGN_512));
    }
}

TEST(MegaMoeArch92RowCombine, NativeW4ReadyAndActivationWidthsStayConsistent)
{
    for (uint8_t mode : {GMM_MODE_A8W4_NZ, GMM_MODE_A4W4_ND, GMM_MODE_A4W4_NZ}) {
        for (int64_t cores : {32, 64}) {
            CclComputeLayoutParams p{};
            p.h = 7168;
            p.hiddenDim = 4096;
            p.bs = 256;
            p.maxOutputSize = 1024;
            p.moeExpertPerRank = 24;
            p.epWorldSize = 4;
            p.aicNum = cores;
            p.numMaxTokensPerRank = 256;
            p.topK = 6;
            p.sharedExpertNum = 2;
            p.sharedInputIndependent = true;
            const auto layout = LayoutArch92::MakeWorkspaceLayout(p, mode, mode, true);
            EXPECT_EQ(layout.mte.sync.gmm2Ready.sizeBytes, p.moeExpertPerRank * cores * 16 * SIZE_INT_32);
            EXPECT_GE(layout.mte.sync.gmm2Ready.offsetBytes, layout.reset.flags.offsetBytes);
            EXPECT_LE(layout.mte.sync.gmm2Ready.offsetBytes + layout.mte.sync.gmm2Ready.sizeBytes,
                      layout.reset.flags.offsetBytes + layout.reset.flags.sizeBytes);
            EXPECT_EQ(layout.routed.gmm2Input.data.sizeBytes,
                      MemoryLayoutDetail::CeilAlign(p.maxOutputSize * p.hiddenDim / p.activationOutputDivisor,
                                                    MemoryLayoutDetail::ALIGN_512));
            EXPECT_EQ(layout.routed.gmm1Input.data.sizeBytes,
                      MemoryLayoutDetail::CeilAlign(p.maxOutputSize * p.h / (IsA4W4GmmMode(mode) ? 2 : 1),
                                                    MemoryLayoutDetail::ALIGN_512));
            EXPECT_EQ(layout.shared.gmm2Input.data.sizeBytes,
                      MemoryLayoutDetail::CeilAlign(p.bs * p.sharedExpertNum * p.hiddenDim / p.activationOutputDivisor,
                                                    MemoryLayoutDetail::ALIGN_512));
            const TensorStorageSpec inputStorage{IsA4W4GmmMode(mode) ? MemoryLayoutDetail::FP4_BITS :
                                                                       MemoryLayoutDetail::FP8_BITS};
            const int64_t sharedInputBytes = p.bs * CalcTokenScaleLayout(p.h, inputStorage, 0, false).tokenScaleBytes;
            EXPECT_EQ(layout.shared.gmm1Input.data.sizeBytes,
                      MemoryLayoutDetail::CeilAlign(sharedInputBytes, MemoryLayoutDetail::ALIGN_512));
            const auto original = LayoutArch92::MakeWorkspaceLayout(p, mode, mode);
            EXPECT_EQ(original.mte.sync.gmm2Ready.offsetBytes, -1);
            EXPECT_EQ(original.routed.gmm2Input.data.sizeBytes, layout.routed.gmm2Input.data.sizeBytes);
        }
    }
}
} // namespace MegaMoeUT
