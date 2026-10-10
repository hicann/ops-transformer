/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP8_HPP
#define GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#ifdef __DAV_CUBE__
#include "c_api/cube_datamove/cube_datamove.h"
#endif

namespace NpuArch::Gemm::Block {

template <class ElementLocalGlobalMax_>
struct BlockMmadTla<CopyGlobalMaxL1ToUBBsaMxfp8, void, void, ElementLocalGlobalMax_, void, void, void, void, void> {
public:
    using DispatchPolicy = CopyGlobalMaxL1ToUBBsaMxfp8;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    static constexpr uint8_t VEC0 = 0;
    static constexpr uint8_t VEC1 = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource)
    {
        // L1 区域顺序：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        // localGlobalMaxL1 = 256*4
        for (uint32_t i = 0; i < MXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            localGlobalMaxL1[i] = resource.l1Buf.template GetBufferByByte<ElementLocalGlobalMax>(
                MXFP8::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP8::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        // Cube 用 TPosition::VECCALC 绝对偏移建 peer UB，copy_cbuf_to_ubuf 才能写到两个 AIV。
        peerGlobalMaxUB = AscendC::LocalTensor<ElementLocalGlobalMax>(
            AscendC::TPosition::VECCALC, NpuArch::Epilogue::Block::MXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET,
            L1_SINGLE_GLOBAL_MAX_SIZE * NpuArch::Epilogue::Block::MXFP8::UB_PEER_GLOBAL_MAX_CNT);
    }
    __aicore__ inline ~BlockMmadTla() {}

    template <class TileInfoT>
    __aicore__ inline void operator()(TileInfoT& delay3TileInfo)
    {
#ifdef __DAV_CUBE__
        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax0 = localGlobalMaxL1[delay3TileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax1 =
            localGlobalMaxL1[delay3TileInfo.tileMaxIdx][L1_SINGLE_GLOBAL_MAX_SIZE];
        AscendC::LocalTensor<ElementLocalGlobalMax> peerGlobalMax =
            peerGlobalMaxUB[delay3TileInfo.tileMaxIdx * L1_SINGLE_GLOBAL_MAX_SIZE];

        // AIC 上 GetSubBlockIdx 恒 0，普通 DataCopy 只写一面且写不到两个 AIV 的 UB。
        // L1 row0 → AIV0，row1 → AIV1（UB→L1 写 coord(1-sub)，即交换 peer）。
        // 用异步 asc_copy_l12ub，不要 _sync。_sync = copy + pipe_barrier(PIPE_ALL)，
        // Cube 和 Vec 一起停。完成序靠调用方 CrossCoreSetFlag<PIPE_MTE1>。
        constexpr uint16_t burstLen = static_cast<uint16_t>(L1_SINGLE_GLOBAL_MAX_SIZE * sizeof(ElementLocalGlobalMax) /
                                                            BLOCK_SIZE); // 128 half = 8 * 32B
        __ubuf__ void* dst = (__ubuf__ void*)peerGlobalMax.GetPhyAddr();
        __cbuf__ void* src0 = (__cbuf__ void*)localGlobalMax0.GetPhyAddr();
        __cbuf__ void* src1 = (__cbuf__ void*)localGlobalMax1.GetPhyAddr();
        asc_copy_l12ub(dst, src0, false, 1, burstLen, 0, 0); // VEC0
        asc_copy_l12ub(dst, src1, true, 1, burstLen, 0, 0);  // VEC1
#endif
    }

protected:
    // MXFP8 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<half> localGlobalMaxL1[MXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
    AscendC::LocalTensor<half> peerGlobalMaxUB;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP8_HPP
