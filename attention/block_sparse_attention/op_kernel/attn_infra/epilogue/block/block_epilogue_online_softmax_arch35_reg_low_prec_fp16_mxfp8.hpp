/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP8_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "../../../arch35/kernel_utils.hpp"
#include "mxfp8_vf/vf_nd2nz_indexes_dn_mxfp8.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_qs128_kvs128.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_qs64_kvs128.h"
#include "mxfp8_vf/vf_mm1_res_pre_padding_align_kvs32_multi_mxfp8.h"
#include "mxfp8_vf/vf_mm1_res_pre_padding_align_kvs32_multi_qs64_mxfp8.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_align_qs128_kvs32_multi.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_align_qs128_kvs32.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_align_qs64_kvs32_multi.h"
#include "mxfp8_vf/vf_softmax_dn_cast_nz_mxfp8_align_qs64_kvs32.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP8Kernel;

template <MXQuantMode MX_QUANT_MODE_, class OutputType_, class LayoutS_, class ScaleType_>
class BlockEpilogue<EpilogueOnlineSoftmaxBsaMxfp8<true, MX_QUANT_MODE_>, OutputType_, Gemm::GemmType<half, LayoutS_>,
                    ScaleType_> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsaMxfp8<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element; // P
    using ElementInput = half;                           // S
    using ElementMax = ElementInput;
    using ElementDisguiseP = uint8_t;
    using ElementPScale = typename ScaleType_::Element;
    using ElementIndex = uint8_t;
    using LayoutPL1 = typename OutputType_::Layout;
    using LayoutPUB = layout::RowMajor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP8::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(MXFP8::UB_S_BUF_OFFSET +
                                                                                 MXFP8::UB_S_INNER_BUF_OFFSET * i);
        }
        oTmpUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP8::UB_OTMP_BUF_OFFSET);
        localRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP8::UB_LOCAL_ROW_SUM_BUF_OFFSET);
        globalRowSumUBTensor = resource.ubBuf.template GetBufferByByte<float>(MXFP8::UB_GLOBAL_ROW_SUM_BUF_OFFSET);

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP8::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(MXFP8::UB_P_BUF_OFFSET +
                                                                                     MXFP8::UB_P_INNER_BUF_OFFSET * i);
        }
        oTransUBTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MXFP8::UB_O_TRANS_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_P_SCALE_CNT; i++) {
            pscaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementPScale>(MXFP8::UB_P_SCALE_BUF_OFFSET +
                                                                                       MXFP8::UB_P_SCALE_BUF_SIZE * i);
        }
        oUBTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(MXFP8::UB_O_BUF_OFFSET);

        // L1 <-> UB
        for (uint32_t i = 0; i < MXFP8::UB_PEER_GLOBAL_MAX_CNT; i++) {
            peerGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }

        // 常驻 buffer
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementMax>(MXFP8::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP8::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP8::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP8::UB_UPDATE_SCALE_CNT; i++) {
            updateScaleUBTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                MXFP8::UB_UPDATE_SCALE_BUF_OFFSET + MXFP8::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP8::UB_INDEX_BUF_OFFSET);

        //  Init Tensor
        Mxfp8VF::InitIndexesAndDuplicateCallVF<ElementMax>(indexUBTensor, localGlobalMaxUBTensor[0]);
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1P>
    __aicore__ inline void operator()(TensorL1P& l1PTensorTla, GemmCoord& actualBlockShape, TileInfo const& tileInfo,
                                      TaskInfo const& taskInfo)
    {
        uint32_t spBuffIdx = GetSPBufferIdx(tileInfo.loop);
        // 跨 tile 同步: 每个 vec core 都需要一个跨 tile 的同步
        if (tileInfo.isKvsFirstTilePerCore) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);

        const uint32_t actKvsTile = actualBlockShape.m();
        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        // QK 在 qs=64 时 N=64，S 仍按 128 宽行的前 64 列存放。softmax 继续走 qs128 VF，
        // 有效列与后 64 列互不影响。qs64 VF 的 P 落点与当前 PV 的 K-outer 不一致。
        const bool sRowIsQs128 = (taskInfo.qsActBaseTileAlign128 == MXFP8::QS_BASE_SIZE);

        if (actKvsTile == MXFP8::KVS_BASE_SIZE) {
            if (sRowIsQs128) {
                // qs=128 或短尾 pad 到 128：整基块 softmax
                Mxfp8VF::SoftmaxWithGroupMaxQs128Kvs128CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                                                              MXFP8::KVS_BASE_SIZE, MXFP8::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor);
            } else {
                // qs=64：只算 UB 行前 64 列
                Mxfp8VF::SoftmaxWithGroupMaxQs64Kvs128CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                                                             MXFP8::KVS_BASE_SIZE, MXFP8::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor);
            }

        } else {
            // kvs 不足 128：仍走 kvs32 VF，不改有效 token 的 softmax
            if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                if (sRowIsQs128) {
                    Mxfp8VF::Mm1ResPrePaddingAlignKvs32MultiCallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));

                } else {
                    // 未 pad 的 qs=64
                    Mxfp8VF::Mm1ResPrePaddingAlignKvs32MultiQs64CallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            // softmax
            if (tileInfo.kvsActBaseTileAlign32 == MXFP8::DATA_BLOCK_BYTE) {
                if (sRowIsQs128) {
                    // softmax padding 32, qs=128 或 qs64 pad 到 128
                    Mxfp8VF::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                      ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor);
                } else {
                    // softmax padding 32, 未 pad 的 qs=64
                    Mxfp8VF::SoftmaxWithGroupMaxAlignQs64Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                     ElementDisguiseP>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor);
                }
            } else {
                if (sRowIsQs128) {
                    // softmax padding 32 multi >= 64, qs=128 或 qs64 pad 到 128
                    Mxfp8VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                           ElementDisguiseP, MXFP8::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                } else {
                    // softmax padding 32 multi >= 64, 未 pad 的 qs=64
                    Mxfp8VF::SoftmaxWithGroupMaxAlignQs64Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                          ElementDisguiseP, MXFP8::QS_BASE_SIZE>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor,
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
                }
            }
            // kvs32 VF 只写前 align32 组；满拷前把未写尾组填 0（stride=512 避开 ping-pong 交错）
            AscendC::PipeBarrier<PIPE_V>();
            ZeroPUBUnwrittenKvsGroups(spBuffIdx, tileInfo.kvsActBaseTileAlign32, sRowIsQs128);
        }
        // }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1：NZ 物理视图按整基块 16KB 固定（M=64×N=512，half-N 拷 16KB）
        // kvs128 时不能再用 KVS/4(=32)，否则只搬 8KB。
        constexpr uint32_t UB_P_NZ_M = MXFP8::UB_P_BUF_SIZE / (MXFP8::QS_BASE_SIZE * 2);
        auto ubPLayoutTla = tla::MakeLayout<ElementDisguiseP, LayoutPUB>(UB_P_NZ_M, MXFP8::QS_BASE_SIZE * 4);
        auto ubPTensorTla = tla::MakeTensor(pUBTensor[0], ubPLayoutTla, Arch::PositionUB{});

        CopyPUBToPL1(l1PTensorTla, ubPTensorTla, tileInfo, taskInfo, spBuffIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP8::SYNC_VEC1_RES_BUF0_FLAG + spBuffIdx);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUBToPL1(TensorDst const& l1PTensorTla, TensorSrc const& ubPTensorTla,
                                        TileInfo const& tileInfo, TaskInfo const& taskInfo, uint32_t spBuffIdx)
    {
        uint32_t pUBTlaMSize = tla::get<0>(ubPTensorTla.shape());
        uint32_t pUBTlaNSize = tla::get<1>(ubPTensorTla.shape());

        // 短 kvs 的未写 P 尾组已在 UB 填 0，可以继续满拷 16KB。
        // qs pad 到 128 必须留在条件里，否则 odd 高半面不搬（~50% fulfill）。
        if (taskInfo.qsActBaseTileAlign128 == MXFP8::QS_BASE_SIZE ||
            tileInfo.kvsActBaseTileAlign32 == MXFP8::KVS_BASE_SIZE) {
            // K-outer：C0 高平面在 +8064，搬满 64 个 TLA 行（16KB）。
            AscendC::DataCopyParams repeatParams;
            repeatParams.blockCount = static_cast<uint16_t>(pUBTlaMSize);
            repeatParams.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
            repeatParams.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
            repeatParams.dstStride = 0;

            auto pUBTensorTlaTile =
                GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP8::UB_P_INNER_BUF_ELEMENT_OFFSET),
                        tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());
            AscendC::DataCopy(l1PTensorTla.data(), pUBTensorTlaTile.data()[srcOffset], repeatParams);
        } else if (tileInfo.kvsActBaseTileAlign32 != 0) {
            // 低位、高位分开搬，中间间隙不搬。
            AscendC::DataCopyParams repeatParams1;
            repeatParams1.blockCount = static_cast<uint16_t>(
                tileInfo.kvsActBaseTileAlign32 /
                (pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图低位(高位)地址8行
            repeatParams1.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
            repeatParams1.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
            repeatParams1.dstStride = 0;

            auto pUBTensorTlaTile1 =
                tla::GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP8::UB_P_INNER_BUF_ELEMENT_OFFSET),
                             tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2));
            auto srcOffset1 = pUBTensorTlaTile1.layout()(pUBTensorTlaTile1.coord());
            AscendC::DataCopy(l1PTensorTla.data(), ubPTensorTla.data()[srcOffset1], repeatParams1);

            if (taskInfo.qsActBaseTileAlign128 == MXFP8::QS_BASE_SIZE) {
                // 高位搬运（qs pad 到 128 后 even/odd 高半面有有效 token）
                AscendC::DataCopyParams repeatParams2;
                repeatParams2.blockCount = static_cast<uint16_t>(
                    tileInfo.kvsActBaseTileAlign32 /
                    (pUBTlaNSize / 2 /
                     MXFP8::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图高位(低位)地址8行
                repeatParams2.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
                repeatParams2.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP8::DATA_BLOCK_BYTE);
                repeatParams2.dstStride = 0;

                uint32_t pL1NHalfSize = tla::get<1, 0>(l1PTensorTla.shape());
                uint32_t xL1P = tileInfo.kvsActBaseTileAlign64;
                uint32_t yL1P = 0;
                if (tileInfo.kvsActBaseTileAlign64 == MXFP8::KVS_BASE_SIZE) {
                    xL1P = 0;
                    yL1P = pL1NHalfSize;
                }
                auto pUBTensorTlaTile2 = tla::GetTile(pUBTensorTlaTile1, tla::MakeCoord(pUBTlaMSize / 2, 0u),
                                                      tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2)); // 遗留
                auto pl1TensorTlaTile = tla::GetTile(l1PTensorTla, tla::MakeCoord(xL1P, yL1P),
                                                     tla::MakeShape(MXFP8::KVS_BASE_SIZE, pL1NHalfSize));

                auto srcOffset2 = pUBTensorTlaTile2.layout()(pUBTensorTlaTile2.coord());
                auto dstOffset2 = pl1TensorTlaTile.layout()(pl1TensorTlaTile.coord());

                AscendC::DataCopy(pl1TensorTlaTile.data()[dstOffset2], pUBTensorTlaTile2.data()[srcOffset2],
                                  repeatParams2);
            }
        }
    }

    // kvs32 VF：4 行×64 e4m3 打 256B，StoreAlign 按 VL128 拆到两个地址。
    // preg_vl128 写 dest；preg_vl128_not 按 lane 落到 dest+128。因此：
    //   lo 槽 [pOff, pOff+256) = even 的 pOff + odd 的 pOff+128
    //   hi 槽 [pOff+8192, pOff+8192+256)，不是 +8064（+8064 是 Store 的 dest，
    //     vl128_not 才落到 +8192；Duplicate(+8064, 256B) 会写穿上一 TLA 行的 ping-pong 另一槽）。
    // 64×512 行主序：+8192=row16 col0（本槽 256B）；+8064=row15 col384（另一槽）。
    // qs[64:128) 同理用 +16384 / +24576，不用 +24448。
    // j=0,2,4,6 的 pOff 已是 512B 行起点，Duplicate 256B 只填 col[0:256)，不碰另一槽。
    __aicore__ inline void ZeroPUBUnwrittenKvsGroups(uint32_t spBuffIdx, uint32_t kvsAlign32, bool qs128Planes)
    {
        if (kvsAlign32 == 0 || kvsAlign32 >= MXFP8::KVS_BASE_SIZE) {
            return;
        }
        constexpr uint32_t GROUP_STRIDE = 2048;
        constexpr uint32_t SLOT = 256;
        constexpr uint32_t ITER_PER_GROUP = 8;
        constexpr uint32_t PSCALE_GROUP_CNT = 4;
        constexpr uint32_t HI_PLANE = 8192;  // TLA 16-31，vl128_not 落点
        constexpr uint32_t QS_HI_LO = 16384; // TLA 32-47
        constexpr uint32_t QS_HI_HI = 24576; // TLA 48-63，vl128_not 落点
        uint32_t firstGroup = kvsAlign32 / 32;
        auto p = pUBTensor[spBuffIdx];
        for (uint32_t i = firstGroup; i < PSCALE_GROUP_CNT; ++i) {
            for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
                uint32_t pOff = i * GROUP_STRIDE + static_cast<uint32_t>(j) * SLOT;
                AscendC::Duplicate(p[pOff], static_cast<ElementDisguiseP>(0), SLOT);
                AscendC::Duplicate(p[pOff + HI_PLANE], static_cast<ElementDisguiseP>(0), SLOT);
                if (qs128Planes) {
                    AscendC::Duplicate(p[pOff + QS_HI_LO], static_cast<ElementDisguiseP>(0), SLOT);
                    AscendC::Duplicate(p[pOff + QS_HI_HI], static_cast<ElementDisguiseP>(0), SLOT);
                }
            }
        }
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetSPBufferIdx(const uint32_t loop)
    {
        return loop / 2 % 2;
    }

    __aicore__ inline uint32_t GetLocalGroupMaxBufIdx(uint32_t loop)
    {
        // kvs128：每基块 4 个 group-max = 1KB，连续占槽（不再 2KB 双槽）
        return loop / 2 % MXFP8::UB_LOCAL_GROUP_MAX_CNT;
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput> sUBTensor[MXFP8::UB_S_BUF_CNT];                       // mm1Res(S)
    AscendC::LocalTensor<float> oTmpUBTensor;                                                // mm2Res(PV)
    AscendC::LocalTensor<float> localRowSumUBTensor;                                         // LocalRowSum
    AscendC::LocalTensor<float> globalRowSumUBTensor;                                        // GlobalRowSum
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP8::UB_P_BUF_CNT];                   // vec1Res(P)
    AscendC::LocalTensor<bfloat16_t> oTransUBTensor;                                         // attnTrans(空间复用P)
    AscendC::LocalTensor<ElementPScale> pscaleUBTensor[MXFP8::UB_P_SCALE_CNT];               // pScale(复用P)
    AscendC::LocalTensor<bfloat16_t> oUBTensor;                                              // attentionOut
    AscendC::LocalTensor<ElementMax> peerGlobalMaxUBTensor[MXFP8::UB_PEER_GLOBAL_MAX_CNT];   // peerGlobalMax
    AscendC::LocalTensor<ElementMax> softmaxMaxUBTensor;                                     // softmaxMax
    AscendC::LocalTensor<ElementMax> localGroupMaxUBTensor[MXFP8::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementMax> localGlobalMaxUBTensor[MXFP8::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<float> updateScaleUBTensor[MXFP8::UB_UPDATE_SCALE_CNT];             // updateScale
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                                        // Index
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP8_HPP
