/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp8_vf/vf_computeScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_onlyUpdateScale_dn_mxfp8.h"
#include "mxfp8_vf/vf_computeScale_dn_mxfp8_qs64.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP8Kernel;

template <MXQuantMode MX_QUANT_MODE_, class ElementGroupMax_, class ElementDm_, class PScaleType_>
class BlockEpilogue<EpilogueComputePScaleBsaMxfp8<true, MX_QUANT_MODE_>, ElementGroupMax_, ElementDm_, PScaleType_> {
public:
    using DispatchPolicy = EpilogueComputePScaleBsaMxfp8<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementGroupMax = ElementGroupMax_;
    using PScaleType = PScaleType_;
    using ElementPScale = typename PScaleType::Element;
    using LayoutPScale = typename PScaleType::Layout;
    using ElementDm = ElementDm_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        for (uint32_t i = 0; i < MXFP8::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(MXFP8::UB_P_SCALE_BUF_OFFSET +
                                                                                       MXFP8::UB_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(MXFP8::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(MXFP8::UB_UPDATE_SCALE_BUF_OFFSET +
                                                                               MXFP8::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1Pscale>
    __aicore__ inline void operator()(TensorL1Pscale& l1PScaleTensor, TileInfo const& tileInfo, TaskInfo& taskInfo)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<ElementGroupMax> localGlobalMax = localGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> peerGlobalMax = peerGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> softmaxMaxOld = softmaxMaxUBTensor;
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        // 如果当前s2只有一个softmax，就另外一个核不用做pscale操作
        if (tileInfo.kvsFirstTileStartVecCore != subBlockIdx && tileInfo.isTileGoupFirstTile) {
            if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
                Mxfp8VF::computeOnlyScale<true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm, static_cast<uint16_t>(subBlockIdx));
            } else {
                Mxfp8VF::computeOnlyScale<false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    localGlobalMax, peerGlobalMax, softmaxMaxOld, dm, static_cast<uint16_t>(subBlockIdx));
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        uint16_t firstLoop = 0;
        uint16_t secondLoop = 0;
        uint16_t firstLoopStart = 0;
        uint16_t secondLoopStart = 0;
        GetPScaleParams(tileInfo, subBlockIdx, firstLoopStart, firstLoop, secondLoopStart, secondLoop);
        if (firstLoop == 0 && secondLoop == 0) {
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        AscendC::LocalTensor<ElementPScale> pscale1 = pscaleUBTensor[0];
        AscendC::LocalTensor<ElementPScale> pscale2 = pscaleUBTensor[firstLoop];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax1 = localGroupMaxUBTensor[firstLoopStart];
        AscendC::LocalTensor<ElementGroupMax> localGroupMax2 = localGroupMaxUBTensor[secondLoopStart];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);

        // 与 softmax 一致：group-max 仍是 128 宽，pscale 走 qs128。
        const bool sRowIsQs128 = (taskInfo.qsActBaseTileAlign128 == MXFP8::QS_BASE_SIZE);
        if (tileInfo.curKvsTileLoopIdx / MXFP8Kernel::TILE_GROUP_N == 0) {
            if (sRowIsQs128) {
                Mxfp8VF::computePscale<MX_QUANT_MODE, true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2, localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx));
            } else {
                Mxfp8VF::computePscaleQS64CallVF<MX_QUANT_MODE, true, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2, localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx));
            }
        } else {
            if (sRowIsQs128) {
                Mxfp8VF::computePscale<MX_QUANT_MODE, false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2, localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx));
            } else {
                Mxfp8VF::computePscaleQS64CallVF<MX_QUANT_MODE, false, ElementGroupMax, MXFP8::QS_BASE_SIZE>(
                    pscale1, pscale2, localGroupMax1, localGroupMax2, localGlobalMax, peerGlobalMax, softmaxMaxOld, dm,
                    firstLoop, secondLoop, static_cast<uint16_t>(subBlockIdx));
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);

        // 与 softmax 一致：按 128 宽写 L1。
        bool isQsAlign128 = taskInfo.qsActBaseTileAlign128 == MXFP8::QS_BASE_SIZE;
        CopyPScaleUbToL1(l1PScaleTensor, pscale1, pscale2, subBlockIdx, firstLoopStart, firstLoop, secondLoopStart,
                         secondLoop, isQsAlign128);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF1_FLAG);
    }

    template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1(TensorDst& l1PScaleTensor, AscendC::LocalTensor<ElementPScale>& pscale1Src,
                                            AscendC::LocalTensor<ElementPScale>& pscale2Src, uint32_t subBlockIdx,
                                            uint16_t firstLoopStart, uint16_t firstLoop, uint16_t secondLoopStart,
                                            uint16_t secondLoop, bool isQsAlign128)
    {
        // GetPScaleParams 的 firstLoopStart 是 AIV 本地 groupMax 环（loop/2%20）。
        // L1 P-scale 与 P 一样按 loop%20 分槽；dstStride 再隔行写入（AIV0 偶数槽 / AIV1 奇数槽）。
        // 组 0：groupMax 0 → L1 0，skip-2 得到 0,2,…,14。
        // 组 1：groupMax 8 必须映射成 L1 16，否则 PV 读 loop%20=16 用到组 0 残留 scale。
        CopyPScaleUbToL1FromGroupMax(l1PScaleTensor, pscale1Src, subBlockIdx, firstLoopStart, firstLoop, isQsAlign128);
        if (secondLoop != 0) {
            CopyPScaleUbToL1FromGroupMax(l1PScaleTensor, pscale2Src, subBlockIdx, secondLoopStart, secondLoop,
                                         isQsAlign128);
        }
    }

private:
    template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1Seg(TensorDst& l1PScaleTensor, AscendC::LocalTensor<ElementPScale>& src,
                                               uint16_t l1MStart, uint16_t count, bool isQsAlign128)
    {
        if (count == 0) {
            return;
        }
        uint32_t l1PScaleTensorNSize = tla::get<1>(l1PScaleTensor.shape());
        auto l1TlaTile =
            tla::GetTile(l1PScaleTensor, tla::MakeCoord(l1MStart, 0), tla::MakeShape(1, l1PScaleTensorNSize));
        uint32_t dstOffset = l1TlaTile.layout()(l1TlaTile.coord());
        uint16_t pscaleGroupLen = static_cast<uint16_t>(l1PScaleTensorNSize / MXFP8::DATA_BLOCK_BYTE);
        uint16_t pscaleGroupHalfLen = static_cast<uint16_t>(l1PScaleTensorNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
        AscendC::DataCopyParams rp;
        rp.blockCount = count;
        rp.blockLen = isQsAlign128 ? pscaleGroupLen : pscaleGroupHalfLen;
        rp.srcStride = isQsAlign128 ? 0 : pscaleGroupHalfLen;
        rp.dstStride = isQsAlign128 ? pscaleGroupLen : (pscaleGroupHalfLen + pscaleGroupLen);
        AscendC::DataCopy(l1TlaTile.data()[dstOffset], src, rp);
    }

    template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1FromGroupMax(TensorDst& l1PScaleTensor,
                                                        AscendC::LocalTensor<ElementPScale>& src, uint32_t subBlockIdx,
                                                        uint16_t groupMaxStart, uint16_t count, bool isQsAlign128)
    {
        if (count == 0) {
            return;
        }
        uint32_t l1Cnt = tla::get<0>(l1PScaleTensor.shape());
        uint16_t l1Start = static_cast<uint16_t>((2 * groupMaxStart + subBlockIdx) % l1Cnt);
        uint16_t fit = 0;
        for (uint16_t s = l1Start; s < l1Cnt; s += 2) {
            ++fit;
        }
        uint16_t n1 = count < fit ? count : fit;
        CopyPScaleUbToL1Seg(l1PScaleTensor, src, l1Start, n1, isQsAlign128);
        if (n1 < count) {
            uint32_t srcOff = static_cast<uint32_t>(n1) * tla::get<1>(l1PScaleTensor.shape());
            AscendC::LocalTensor<ElementPScale> src2 = src[srcOff];
            CopyPScaleUbToL1Seg(l1PScaleTensor, src2, static_cast<uint16_t>(subBlockIdx),
                                static_cast<uint16_t>(count - n1), isQsAlign128);
        }
    }

    __aicore__ inline void GetPScaleParams(const TileInfo& tileInfo, uint32_t subBlockIdx, uint16_t& firstLoopStart,
                                           uint16_t& firstLoop, uint16_t& secondLoopStart, uint16_t& secondLoop)
    {
        uint16_t isLoopFirstTaskVecCore = (subBlockIdx == tileInfo.kvsFirstTileStartVecCore);
        uint32_t loop = tileInfo.loop;
        uint32_t curS2LoopIdx = tileInfo.curKvsTileLoopIdx;
        // 单核目前为止第几个 softmax（跨 batch）
        uint32_t groupMaxEndLoop = (loop >> 1) - (subBlockIdx != 0 && loop % 2 == 0);
        // kvs128：1 基块 = 1 槽，末槽就是 groupMaxEndLoop。
        uint32_t groupMaxEndIdx = groupMaxEndLoop % GROUP_MAX_SPACE_LEN;

        // 两个 AIV 交替 softmax，每核只处理自己做过的 softmax。
        // 一槽一 softmax，直接 ceil/floor(组内 tile 数 / 2)。
        uint16_t tilesInGroup =
            static_cast<uint16_t>((curS2LoopIdx + 1) - (curS2LoopIdx / WHOLE_PROCESS_LOOP) * WHOLE_PROCESS_LOOP);
        if (tilesInGroup == 0) {
            tilesInGroup = WHOLE_PROCESS_LOOP;
        }
        uint16_t toProcessGroupMaxLoopLen = isLoopFirstTaskVecCore ? static_cast<uint16_t>((tilesInGroup + 1) / 2) :
                                                                     static_cast<uint16_t>(tilesInGroup / 2);
        if (toProcessGroupMaxLoopLen > WHOLE_PROCESS_LOOP) {
            toProcessGroupMaxLoopLen = WHOLE_PROCESS_LOOP;
        }
        firstLoop = toProcessGroupMaxLoopLen;
        secondLoop = 0;
        int32_t firstLoopStartNotU =
            static_cast<int32_t>(groupMaxEndIdx) - static_cast<int32_t>(toProcessGroupMaxLoopLen) + 1;
        if (firstLoopStartNotU < 0) {
            firstLoopStart = static_cast<uint16_t>(firstLoopStartNotU + GROUP_MAX_SPACE_LEN);
            firstLoop = static_cast<uint16_t>(GROUP_MAX_SPACE_LEN - firstLoopStart);
            secondLoop = static_cast<uint16_t>(toProcessGroupMaxLoopLen - firstLoop);
            secondLoopStart = 0;
        } else {
            firstLoopStart = static_cast<uint16_t>(firstLoopStartNotU);
            secondLoopStart = static_cast<uint16_t>(firstLoopStart + firstLoop);
        }
    }

private:
    static constexpr uint32_t GROUP_MAX_SPACE_LEN = 20; // UB_LOCAL_GROUP_MAX_CNT
    static constexpr uint32_t WHOLE_PROCESS_LOOP = 16;  // TILE_GROUP_N

    AscendC::LocalTensor<ElementPScale> pscaleUBTensor[MXFP8::UB_P_SCALE_CNT];                    // pScale(复用P)
    AscendC::LocalTensor<ElementGroupMax> peerGlobalMaxUBTensor[MXFP8::UB_PEER_GLOBAL_MAX_CNT];   // peerGlobalMax
    AscendC::LocalTensor<ElementGroupMax> softmaxMaxUBTensor;                                     // softmaxMax
    AscendC::LocalTensor<ElementGroupMax> localGroupMaxUBTensor[MXFP8::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementGroupMax> localGlobalMaxUBTensor[MXFP8::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP8::UB_UPDATE_SCALE_CNT];                       // updateScale
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH35_MXFP8
