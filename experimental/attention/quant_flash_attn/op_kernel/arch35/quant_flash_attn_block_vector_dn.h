/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
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
 * \file quant_flash_attn_block_vector_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_
#define QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_

#include "vf/vf_softmax_dn_cast_nz_mxfp4_qs128_kvs256.h"
#include "vf/vf_softmax_nd_cast_nz_mxfp4.h"
#include "vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32.h"
#include "vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32_multi.h"
#include "vf/vf_nd2nz_indexes_dn_mxfp4.h"
#include "vf/vf_computeScale_dn_mxfp4.h"
#include "vf/vf_updateScale_dn_mxfp4.h"
#include "vf/vf_attenOut_dn_mxfp4.h"
#include "vf/vf_onlyUpdateScale_dn_mxfp4.h"
#include "vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"

#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"
#include "quant_flash_attn_block_vec_fd_dn.h"
#include "../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../common/op_kernel/memcopy/fa_ub_tensor.h"
#include "../../common/op_kernel/memcopy/copy_ub_to_gm.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;

namespace QFA_KERNEL {

template <QFA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr GmFormat GetAttentionOutGmFormat()
{
    static_assert(
        (LAYOUT_OUT == QFA_LAYOUT::BSND) || (LAYOUT_OUT == QFA_LAYOUT::BNSD) || (LAYOUT_OUT == QFA_LAYOUT::TND),
        "Get OUT GmFormat fail, LAYOUT_OUT is incorrect");
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_OUT == QFA_LAYOUT::TND) {
        return GmFormat::TNGD;
    }
}

template <QFA_LAYOUT LAYOUT_OUT>
__aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert(
        (LAYOUT_OUT == QFA_LAYOUT::BNSD) || (LAYOUT_OUT == QFA_LAYOUT::BSND) || (LAYOUT_OUT == QFA_LAYOUT::TND),
        "Get OutAttention UB GmFormat fail, LAYOUT is incorrect");
    if constexpr (LAYOUT_OUT == QFA_LAYOUT::TND) {
        return UbFormat::S1G; // TND: M 行 t-major (r=t*G+g), 写出走 S1G 分支
    } else {
        return UbFormat::GS1; // BSND/BNSD 摊平: 行序 GS1
    }
}

template <typename QFAT, const bool ND = false>
class QuantFlashAttnBlockVectorDn {
public:
    /* =================编译期常量的基本块信息================= */
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool SOFTMAX_DN = true;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;

    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;
    static constexpr QFA_LAYOUT LAYOUT_OUT = QFAT::outLayout;

private:
    // 初始化基础常量
    const ConstInfo& constInfo;
    const SeqLensTool<LAYOUT_Q, SEQLEN_T>& qSeqLensTool;
    const SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool;

    static constexpr uint16_t s1BaseSize = 128;
    static constexpr uint16_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;

    static constexpr half MINNEG_VALUE = -65504;
    static constexpr uint32_t UPDATE_LEN_SIZE = 64;

    static constexpr uint32_t LOCAL_GROUP_MAX_SIZE = 128 * 128 / 32 * sizeof(half);
    static constexpr uint32_t LOCAL_GLOBAL_MAX_SIZE = 128 * sizeof(half);
    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;
    static constexpr uint32_t SINGLE_PSCALE_SPACE_SIZE = 32 * 5 * 8;
    static constexpr uint32_t SINGLE_GROUP_MAX_SPACE_SIZE = 4 * 128;

    static constexpr uint32_t mm1ResOffset = 128;
    static constexpr uint32_t ve1ResOffset = 256;

    // =================================UB Buffer=================================
    // ND: M 维实际恒 <= ND_M_ROWS(与 host 端 ND_MAX_M_ROWS 一致), 全部按 M 行开, 不再按 128
    static constexpr uint32_t ND_M_ROWS = 64U;
    static constexpr uint32_t UB_S_SIZE = ((ND ? ND_M_ROWS : 128U) * 256U);
    static constexpr uint32_t UB_PV_SIZE = 128U * 64U;
    static constexpr uint32_t UB_PSCALE_SIZE = 32 * 5 * 8;
    static constexpr uint32_t UB_ROWSUM_SIZE = 64U;
    static constexpr uint32_t UB_P_SIZE = ND ? ((ND_M_ROWS * 256U) > 8192U ? (ND_M_ROWS * 256U) : 8192U) : (64U * 256U);
    static constexpr uint32_t UB_ATTENTIONOUT_SIZE = 128U * 64U;
    static constexpr uint32_t UB_ATTENTIONTRANS_SIZE = 128 * 64 * 9 / 8;
    static constexpr uint32_t UB_MAX_SIZE = 128U;
    static constexpr uint32_t UB_LOCALGROUPMAX_SIZE = 128U * 4U;
    static constexpr uint32_t UB_UPDATE_SIZE = 64U;
    static constexpr uint32_t UB_INDEX_SIZE = (ND ? 3328U : 256U);
    static constexpr uint32_t ND_MAX_STAGE_SIZE = 128U * 4U;      // [M][4] 组标量暂存
    static constexpr uint32_t ND_P_CHUNK_BYTES = ND_M_ROWS * 64U; // 须与 cube 侧 P 读取(actMAlign=128)口径一致
    static constexpr uint32_t ND_P_PLANE_BYTES = ND_M_ROWS * 32U; // 单平面 = 128 M 行 x 32B
    static constexpr uint32_t ND_S_CHUNK_SIZE = ND_M_ROWS * 128U; // 每 128 个 s2 的 S, 单位 half
    static constexpr uint32_t QK_S2_SPLIT_ND = 128;               // ND 下 cube 每个 s2 分片

    static constexpr uint32_t UB_S_BUFCNT = 2;
    static constexpr uint32_t UB_PV_BUFCNT = 1;
    static constexpr uint32_t UB_PSCALE_BUFCNT = 8;

public:
    // UB 地址(本 block 的布局; cube 需要的地址在 cube block 里另有一份)
    static constexpr uint32_t UB_S_BYTES = UB_S_SIZE * UB_S_BUFCNT * 2U;
    static constexpr uint32_t UB_P_BYTES = UB_P_SIZE * 2U;
    static constexpr uint32_t UB_PV_BYTES = UB_PV_SIZE * UB_PV_BUFCNT * 4U;
    static constexpr uint32_t UB_ROWSUM_BYTES = UB_ROWSUM_SIZE * 4U;
    static constexpr uint32_t UB_ATTENTIONOUT_BYTES = UB_ATTENTIONOUT_SIZE * 4U;
    static constexpr uint32_t UB_ATTENTIONOUT_OFFSET = UB_S_BYTES + UB_PV_BYTES + 2U * UB_ROWSUM_BYTES + UB_P_BYTES;
    static constexpr uint32_t UB_PEER_GMAX_BYTES = 128U * 4U * sizeof(half);
    static constexpr uint32_t UB_PEER_GMAX_OFFSET = UB_ATTENTIONOUT_OFFSET + UB_ATTENTIONOUT_BYTES;
    static constexpr uint32_t UB_ND2NZ_IDX_OFFSET =
        UB_PEER_GMAX_OFFSET + 128U * 4U * 2U + 128U * 2U + 128U * 4U * 20U * 2U + 128U * 4U * 2U + 64U * 4U * 4U;
    static constexpr uint32_t UB_FD_MAX_STAGE_OFFSET =
        ND ? (UB_ND2NZ_IDX_OFFSET + UB_INDEX_SIZE + ND_MAX_STAGE_SIZE) : (UB_ATTENTIONOUT_OFFSET - 4096U);
    // FD 的 UB 预留(AIV 侧): ND 的 FA 只用到 ~111KB, 182KB 起留给 FD
    using VecFdBlockType = QuantFlashAttnBlockVecFdDn<QFAT>;
    static constexpr uint32_t FD_UB_BASE = 229888U;
    static constexpr uint32_t FD_UB_A_OFFSET = FD_UB_BASE - VecFdBlockType::GetFdAUbBytes();
    static constexpr uint32_t FD_UB_B_OFFSET = FD_UB_BASE + UB_PEER_GMAX_BYTES;

private:
    static constexpr uint32_t UB_ROWSUM_BUFCNT = 1;
    static constexpr uint32_t UB_P_BUFCNT = 2;
    static constexpr uint32_t UB_ATTENTIONOUT_BUFCNT = 1;
    static constexpr uint32_t UB_MAX_BUFCNT = 1;
    static constexpr uint32_t UB_PEER_MAX_BUFCNT = 4;
    static constexpr uint32_t UB_LOCALGLOBALMAX_BUFCNT = 4;
    static constexpr uint32_t UB_LOCALGROUPMAX_BUFCNT = 20;
    static constexpr uint32_t UB_UPDATE_BUFCNT = 4;
    static constexpr uint32_t UB_INDEX_BUFCNT = 1;
    static constexpr uint32_t UB_FD_BUFCNT = 8;
    static constexpr uint32_t INIT_OUTPUT_UB_BYTES =
        UB_S_SIZE * UB_S_BUFCNT * sizeof(half) + UB_PV_SIZE * UB_PV_BUFCNT * sizeof(float) +
        UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float) + UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float) +
        UB_P_SIZE * UB_P_BUFCNT * sizeof(uint8_t) + UB_ATTENTIONOUT_SIZE * UB_ATTENTIONOUT_BUFCNT * sizeof(float) +
        UB_MAX_SIZE * UB_MAX_BUFCNT * sizeof(half) + UB_MAX_SIZE * UB_MAX_BUFCNT * sizeof(half) +
        UB_LOCALGROUPMAX_SIZE * UB_LOCALGROUPMAX_BUFCNT * sizeof(half);

    LocalTensor<half> mm1ResUB;
    LocalTensor<float> mm2ResUB;
    LocalTensor<uint8_t> pscaleUB;
    LocalTensor<half> peerGlobalMaxUB;
    LocalTensor<uint8_t> vec1ResUB;
    LocalTensor<float> localRowsumUB;
    LocalTensor<float> attentionOutUB;
    LocalTensor<bfloat16_t> attentionTransUB;
    LocalTensor<float> globalRowsumUB;
    LocalTensor<half> localGroupMaxUB;
    LocalTensor<half> localGlobalMaxUB;
    LocalTensor<half> softmaxMaxUB;
    LocalTensor<float> updateScaleUB;
    LocalTensor<uint8_t> nd2nzIndexUB;
    LocalTensor<half> ndMaxStageUB;

    LocalTensor<float> sumFDUB;
    LocalTensor<half> maxFDUB;

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_P_SIZE = (ND ? 64U : 128U) * 256U / 2; // ND: 64 行 = 8K, DN: 128 行 = 16K
    static constexpr uint32_t L1_P_DESCALE_SIZE = 32 * 5 * 8;           // 1.25K
    static constexpr uint32_t L1_P_BUFCNT = 20;
    // 与 cube 侧共用同一套 L1 段偏移(禁止硬编码): ND 下该段比 DN 前移 286720B
    static constexpr uint32_t L1_GLOBAL_MAX_OFFSET =
        ((ND ? ND_M_ROWS : 128U) * 256U / 2U * 20U + 128U * 2U * (256U / 64U + 1U) * 20U + 128U * 128U / 2U +
         128U * (128U / 32U) + 512U * 128U / 2U * 4U + 512U * (128U / 32U) * 4U + 256U * (128U + 16U) / 2U +
         (256U / 64U) * ((128U + 16U) / 16U) * 32U);

    LocalTensor<uint8_t> pL1Tensor;
    LocalTensor<uint8_t> pScaleL1;
    LocalTensor<half> localGlobalMaxL1;

    /* =====================GM变量==================== */

    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();

    // V2 attentionOut 变量
    static constexpr bool OUT_IS_TND = IS_TND<LAYOUT_OUT>();
    FaGmTensor<OUT_T, OUT_FORMAT, SEQLEN_T, OUT_IS_TND> outGmTensor;
    CopyAttenOutUbToGm<OUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_OUT>()> AttenOutUbToGm;
    // FD partial workspace
    GlobalTensor<float> fdAccumOutFdGm; // [slot][d=128][m=128] fp32 未归一化局部分子
    GlobalTensor<float> fdSumFdGm;      // [slot][m] fp32 局部分母
    GlobalTensor<float> fdMaxFdGm;      // [slot][m] fp32 局部行 max (ln 域)
    LocalTensor<float> fdMaxStage_;     // FA 侧 max partial 暂存 [m=64] fp32

    static constexpr uint32_t SYNC_VEC1_RES_BUF0_MUTEX_ID = 17;
    static constexpr uint32_t SYNC_VEC1_RES_BUF1_MUTEX_ID = 18;
    static constexpr uint32_t SYNC_GMAX_UB_TO_L1_BUF0_MUTEX_ID = 19;
    static constexpr uint32_t SYNC_GMAX_UB_TO_L1_BUF1_MUTEX_ID = 20;
    static constexpr uint32_t SYNC_GMAX_UB_TO_L1_BUF2_MUTEX_ID = 21;
    static constexpr uint32_t SYNC_GMAX_UB_TO_L1_BUF3_MUTEX_ID = 22;
    static constexpr uint32_t SYNC_ATTN_BUF_MUTEX_ID = 23;
    static constexpr uint32_t SYNC_INIT_OUTPUT_MUTEX_ID = 24;
    static constexpr uint32_t FD_MAX_STAGE_FLOATS = 64U;

public:
    // 初始化 Vec Block 层
    __aicore__ inline QuantFlashAttnBlockVectorDn(ConstInfo& constInfo, SeqLensTool<LAYOUT_Q, SEQLEN_T>& qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool)
        : constInfo(constInfo),
          qSeqLensTool(qSeqLensTool),
          kvSeqLensTool(kvSeqLensTool){};

    __aicore__ inline void InitInput(__gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace)
    {
        // 初始化 attentionOut GM Buffer 及 GmTensor
        InitAttentionOutBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize,
                               qSeqLensTool, outGmTensor, attentionOut);
        // FD partial workspace 三段: accumOut [slot][d=128][m=128] fp32 / sum、max 各 [slot][m=128] 1 个 fp32
        fdAccumOutFdGm.SetGlobalBuffer((__gm__ float*)workspace);
        fdSumFdGm.SetGlobalBuffer((__gm__ float*)(workspace + constInfo.accumOutSize));
        fdMaxFdGm.SetGlobalBuffer((__gm__ float*)(workspace + constInfo.accumOutSize + constInfo.logSumExpSize));
    }

    // FD 归约复用的转置暂存 (FA 段结束后空闲)
    __aicore__ inline LocalTensor<float> GetFdTransUb()
    {
        return attentionTransUB.template ReinterpretCast<float>();
    }

    // FA 侧 max partial 暂存绑到 FD 独占区, 与 FA 段 UB 不再共用
    __aicore__ inline void InitFdMaxStage(uint32_t ubOffset)
    {
        fdMaxStage_ = LocalTensor<float>(TPosition::VECIN, ubOffset, FD_MAX_STAGE_FLOATS);
    }

    // 初始化 UB
    __aicore__ inline void InitTensors()
    {
        // =================================L1 Tensor Init=================================
        uint32_t addrL1Start = 0; // 16K * 20 = 320K
        pL1Tensor = LocalTensor<uint8_t>(TPosition::A1, addrL1Start, L1_P_SIZE * L1_P_BUFCNT);
        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;
        pScaleL1 = LocalTensor<uint8_t>(TPosition::A1, addrL1Start, L1_P_DESCALE_SIZE * L1_P_BUFCNT); // 1K * 20 = 20K

        // 长度按 4 槽 x 2 subblock x 128 half 声明: 旧代码只声明了 128 half,
        // 而写入索引最大到 tileMaxIdx*256 + 128 = 896, 越界声明
        localGlobalMaxL1 = LocalTensor<half>(TPosition::A1, L1_GLOBAL_MAX_OFFSET, 256U * 4U);

        // =================================UB Tensor Init=================================
        uint32_t addrUBStart = 0;
        uint32_t addrUBReuseStart = 0;
        mm1ResUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_S_SIZE * UB_S_BUFCNT);

        addrUBStart += UB_S_SIZE * UB_S_BUFCNT * sizeof(half);
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_PV_SIZE * UB_PV_BUFCNT);

        addrUBStart += UB_PV_SIZE * UB_PV_BUFCNT * sizeof(float);
        localRowsumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT);

        addrUBStart += UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float);
        globalRowsumUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT);

        addrUBStart += UB_ROWSUM_SIZE * UB_ROWSUM_BUFCNT * sizeof(float);
        vec1ResUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, UB_P_SIZE * UB_P_BUFCNT);

        addrUBReuseStart = addrUBStart;
        attentionTransUB = LocalTensor<bfloat16_t>(TPosition::VECCALC, addrUBReuseStart,
                                                   UB_ATTENTIONTRANS_SIZE * UB_ATTENTIONOUT_BUFCNT);

        addrUBReuseStart += UB_ATTENTIONTRANS_SIZE * UB_ATTENTIONOUT_BUFCNT * sizeof(bfloat16_t);
        pscaleUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBReuseStart, UB_PSCALE_SIZE * UB_PSCALE_BUFCNT);

        addrUBReuseStart += UB_PSCALE_SIZE * UB_PSCALE_BUFCNT * sizeof(uint8_t);
        sumFDUB = LocalTensor<float>(TPosition::VECCALC, addrUBReuseStart, UB_ROWSUM_SIZE * UB_FD_BUFCNT);

        addrUBReuseStart += UB_ROWSUM_SIZE * UB_FD_BUFCNT * sizeof(float);
        maxFDUB = LocalTensor<half>(TPosition::VECCALC, addrUBReuseStart, UB_MAX_SIZE * UB_FD_BUFCNT);

        addrUBStart += UB_P_SIZE * UB_P_BUFCNT * sizeof(uint8_t);
        attentionOutUB =
            LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_ATTENTIONOUT_SIZE * UB_ATTENTIONOUT_BUFCNT);

        addrUBStart += UB_ATTENTIONOUT_SIZE * UB_ATTENTIONOUT_BUFCNT * sizeof(float);
        peerGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_PEER_MAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_PEER_MAX_BUFCNT * sizeof(half);
        softmaxMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_MAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_MAX_BUFCNT * sizeof(half);
        localGroupMaxUB =
            LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_LOCALGROUPMAX_SIZE * UB_LOCALGROUPMAX_BUFCNT);

        addrUBStart += UB_LOCALGROUPMAX_SIZE * UB_LOCALGROUPMAX_BUFCNT * sizeof(half);
        localGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, UB_MAX_SIZE * UB_LOCALGLOBALMAX_BUFCNT);

        addrUBStart += UB_MAX_SIZE * UB_LOCALGLOBALMAX_BUFCNT * sizeof(half);
        updateScaleUB = LocalTensor<float>(TPosition::VECCALC, addrUBStart, UB_UPDATE_SIZE * UB_UPDATE_BUFCNT);

        addrUBStart += UB_UPDATE_SIZE * UB_UPDATE_BUFCNT * sizeof(float);
        nd2nzIndexUB = LocalTensor<uint8_t>(TPosition::VECCALC, addrUBStart, UB_INDEX_SIZE * UB_INDEX_BUFCNT);

        addrUBStart += UB_INDEX_SIZE * UB_INDEX_BUFCNT * sizeof(uint8_t);
        if constexpr (ND) {
            ndMaxStageUB = LocalTensor<half>(TPosition::VECCALC, addrUBStart, ND_MAX_STAGE_SIZE);
        }

        // ================================= Init Value =================================
        Mxfp4Api::InitIndexesAndDuplicateCallVF<half>(nd2nzIndexUB, localGlobalMaxUB);
        if constexpr (ND) {
            Mxfp4Api::InitNdIndexesCallVF<half>(nd2nzIndexUB);
        }
    }

    __aicore__ inline void ComputeVec1(const VecRunInfo& runInfo)
    {
        uint32_t buffIdx = GetBufferIdx(runInfo.loop);
        if (unlikely(runInfo.isS2FirstTilePerCore)) { // 1/8结构频率, 每个vector core都需要一个跨tile的同步
            Mutex::Lock<PIPE_V>(SYNC_GMAX_UB_TO_L1_BUF0_MUTEX_ID + runInfo.tileMaxIdx);
        }
        Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID + buffIdx);

        LocalTensor<uint8_t> vec1Res = vec1ResUB[buffIdx * (ND ? UB_P_SIZE : ve1ResOffset)];
        LocalTensor<half> mm1Res = mm1ResUB[buffIdx * (ND ? UB_S_SIZE : mm1ResOffset)];
        LocalTensor<half> localGrpMax = GetLocalGrpMaxUbByLoopIdx(runInfo.loop);
        LocalTensor<half> localGlobalMax = GetLocalGlobalMaxUbByCurIdx(runInfo.tileMaxIdx);
        if constexpr (ND) {
            // 每个 128 个 s2 一片: P 分形直接按 [平面][行][32B] 写, 组标量写 4 组 x 128 M
            uint32_t actM4 = (runInfo.actMSize + 3) / 4 * 4;
            for (uint32_t c = 0; c < 2; ++c) {
                uint32_t s2Done = c * QK_S2_SPLIT_ND;
                if (s2Done >= runInfo.actSingleLoopS2Size) {
                    break;
                }
                uint32_t s2Rest = runInfo.actSingleLoopS2Size - s2Done;
                uint32_t actS2c = s2Rest < QK_S2_SPLIT_ND ? s2Rest : QK_S2_SPLIT_ND;
                LocalTensor<uint8_t> pChunk = vec1Res[c * ND_P_CHUNK_BYTES];
                LocalTensor<half> sChunk = mm1Res[c * ND_S_CHUNK_SIZE];
                LocalTensor<half> grpMaxC = localGrpMax[c * SINGLE_GROUP_MAX_SPACE_SIZE];
                // CAUSAL: 本 tile 全可见时用无 mask 实例, 否则用带掩码实例
                const bool ndAllVisible = (!HAS_MASK) || ((runInfo.maskBase + static_cast<int32_t>(s2BaseSize) - 1) *
                                                              static_cast<int32_t>(constInfo.gSize) <=
                                                          0);
                if (runInfo.isFirstS2Loop && c == 0) {
                    if (ndAllVisible) {
                        Mxfp4Api::softmaxNdCastNzMxfp4CallVF<true, false, half, uint8_t>(
                            pChunk, sChunk, ndMaxStageUB, grpMaxC, localGlobalMax, nd2nzIndexUB, actM4, actS2c,
                            runInfo.maskBase, static_cast<int32_t>(constInfo.gSize), s2Done);
                    } else {
                        Mxfp4Api::softmaxNdCastNzMxfp4CallVF<true, true, half, uint8_t>(
                            pChunk, sChunk, ndMaxStageUB, grpMaxC, localGlobalMax, nd2nzIndexUB, actM4, actS2c,
                            runInfo.maskBase, static_cast<int32_t>(constInfo.gSize), s2Done);
                    }
                } else {
                    if (ndAllVisible) {
                        Mxfp4Api::softmaxNdCastNzMxfp4CallVF<false, false, half, uint8_t>(
                            pChunk, sChunk, ndMaxStageUB, grpMaxC, localGlobalMax, nd2nzIndexUB, actM4, actS2c,
                            runInfo.maskBase, static_cast<int32_t>(constInfo.gSize), s2Done);
                    } else {
                        Mxfp4Api::softmaxNdCastNzMxfp4CallVF<false, true, half, uint8_t>(
                            pChunk, sChunk, ndMaxStageUB, grpMaxC, localGlobalMax, nd2nzIndexUB, actM4, actS2c,
                            runInfo.maskBase, static_cast<int32_t>(constInfo.gSize), s2Done);
                    }
                }
            }
        } else {
            // CAUSAL: 全可见 tile (末行阈值 <= 0) 走无 mask 编译实例, 逐行 mask 开销整块不生成
            // 逐行阈值 = (maskBase + kvRow)*gSize - maskRem, 取 tile 内最大 kvRow 判全可见
            bool tileAllVisible =
                (runInfo.maskBase + static_cast<int32_t>(s2BaseSize) - 1) * static_cast<int32_t>(constInfo.gSize) -
                    runInfo.maskRem <=
                0;
            // CAUSAL: maskBase = k_tile - q_tile - (sk-sq), 逐行阈值 t = maskBase + kvRow,
            // 可见 ⇔ lane >= t*gSize - maskRem (lane 为 tile 内 M 行号; S1G 合轴 r=gS1*mBaseSize+lane)
            if (unlikely(runInfo.isFirstS2Loop)) {
                if (likely(runInfo.actMSize == s1BaseSize &&
                           runInfo.actSingleLoopS2Size == s2BaseSize)) { // s1=128, s2=256 softmax
                    if (HAS_MASK && tileAllVisible) {
                        Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<true, half, uint8_t, false, s2BaseSize,
                                                                       s1BaseSize>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    } else {
                        Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<true, half, uint8_t, HAS_MASK, s2BaseSize,
                                                                       s1BaseSize>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    }
                } else {
                    // s2 padding 32 multi
                    if (runInfo.actSingleLoopS2Size != runInfo.actSingleLoopS2SizeAlign) {
                        Mxfp4Api::Mm1ResPrePaddingAlignKvs32MultiCallVF<half>(
                            mm1Res, static_cast<uint16_t>(runInfo.actSingleLoopS2Size),
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign));
                        PipeBarrier<PIPE_V>();
                    }
                    // softmax
                    if (runInfo.actSingleLoopS2SizeAlign == AttentionCommon::BYTE_BLOCK) { // softmax_padding_32
                        Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<true, half, uint8_t, HAS_MASK>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    } else { // softmax_padding_32_multi >= 64
                        Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<true, half, uint8_t, HAS_MASK>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign64),
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    }
                }
            } else {
                if (likely(runInfo.actMSize == s1BaseSize &&
                           runInfo.actSingleLoopS2Size == s2BaseSize)) { // s1=128, s2=256 softmax
                    if (HAS_MASK && tileAllVisible) {
                        Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<false, half, uint8_t, false, s2BaseSize,
                                                                       s1BaseSize>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    } else {
                        Mxfp4Api::softmaxWithGroupMaxQs128Kvs256CallVF<false, half, uint8_t, HAS_MASK, s2BaseSize,
                                                                       s1BaseSize>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    }
                } else {
                    // s2 padding 32 multi
                    if (runInfo.actSingleLoopS2Size != runInfo.actSingleLoopS2SizeAlign) {
                        Mxfp4Api::Mm1ResPrePaddingAlignKvs32MultiCallVF<half>(
                            mm1Res, static_cast<uint16_t>(runInfo.actSingleLoopS2Size),
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign));
                        PipeBarrier<PIPE_V>();
                    }
                    // softmax
                    if (runInfo.actSingleLoopS2SizeAlign == AttentionCommon::BYTE_BLOCK) { // softmax_padding_32
                        Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<false, half, uint8_t, HAS_MASK>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    } else { // softmax_padding_32_multi >= 64
                        Mxfp4Api::SoftmaxWithGroupMaxAlignQs128Kvs32MultiCallVF<false, half, uint8_t, HAS_MASK>(
                            vec1Res, mm1Res, localGrpMax, localGlobalMax, nd2nzIndexUB,
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                            static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign64),
                            static_cast<int32_t>(constInfo.gSize), runInfo.maskBase, runInfo.maskRem);
                    }
                }
            }
        }

        Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID + buffIdx);
        Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID + buffIdx);

        if constexpr (ND) {
            // P 在 L1 为 [s2/64 平面][M 行][32B], UB 侧为 [M 行][s2 的 fp4] 密排,
            // 按 32B 跨步各搬一个平面 (源行距 64B, 目的行距 32B)
            uint32_t rows = (runInfo.actMSize + 3) / 4 * 4;
            for (uint32_t c = 0; c < 2; ++c) {
                if (c * QK_S2_SPLIT_ND >= runInfo.actSingleLoopS2Size) {
                    break;
                }
                // L1 的 P 区: 每个 chunk 的 plane 按实际 actM 行数(512B 粒度)分配
                // 每个 chunk 的 plane 跟随实际 actM 行数(×512B), 与 cube 端 LoadPToL0 的 srcStride 对齐
                const uint32_t l1Plane = (rows + 15U) / 16U * 512U; // 按实际 actM 行数(512B 粒度)
                const uint32_t l1Chunk = l1Plane * 2U;
                LocalTensor<uint8_t> pL1 = pL1Tensor[(runInfo.loop % 20) * L1_P_SIZE + c * l1Chunk];
                LocalTensor<uint8_t> pUb = vec1Res[c * ND_P_CHUNK_BYTES];
                DataCopy(pL1, pUb, {static_cast<uint16_t>(rows), 1, 1, 0});
                DataCopy(pL1[l1Plane], pUb[32], {static_cast<uint16_t>(rows), 1, 1, 0});
            }
        } else if (likely(runInfo.actMSize == s1BaseSize &&
                          runInfo.actSingleLoopS2Size == s2BaseSize)) { // s1=128, s2=256 softmax
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2)], vec1Res,
                     {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / 4), 8, 8, 0});
        } else {
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2)], vec1Res,
                     {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / AttentionCommon::BYTE_BLOCK * 8 / 2), 8,
                      8, 0});
            DataCopy(pL1Tensor[(runInfo.loop % 20) * (128 * 256 / 2) +
                               runInfo.actSingleLoopS2SizeAlign64 / AttentionCommon::BYTE_BLOCK * 8 / 2 * 256],
                     vec1Res[BUFFER_SIZE_BYTE_16K],
                     {static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign / AttentionCommon::BYTE_BLOCK * 8 / 2), 8,
                      8, 0});
        }
        Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID + buffIdx);
    }

    __aicore__ inline void CopyGMaxUbToL1(const VecRunInfo& runInfo)
    {
        LocalTensor<half> localGlobalMax = GetLocalGlobalMaxUbByCurIdx(runInfo.tileMaxIdx);

        if (unlikely(runInfo.s2FirstStartVecCore != constInfo.subBlockIdx && runInfo.isC2Sync)) { // ~1/32结构频率
            DataCopy(
                localGlobalMaxL1[runInfo.tileMaxIdx * 256 + (1 - constInfo.subBlockIdx) * L1_SINGLE_GLOBAL_MAX_SIZE],
                localGlobalMax, {1, 8, 0, 0});
            return;
        }

        Mutex::Unlock<PIPE_V>(SYNC_GMAX_UB_TO_L1_BUF0_MUTEX_ID + runInfo.tileMaxIdx);
        Mutex::Lock<PIPE_MTE3>(SYNC_GMAX_UB_TO_L1_BUF0_MUTEX_ID + runInfo.tileMaxIdx);
        DataCopy(localGlobalMaxL1[runInfo.tileMaxIdx * 256 + (1 - constInfo.subBlockIdx) * L1_SINGLE_GLOBAL_MAX_SIZE],
                 localGlobalMax, {1, 8, 0, 0});
    }

    __aicore__ inline void UpdatePScale(const VecRunInfo& runInfo)
    {
        LocalTensor<half> localGlobalMax1 = GetLocalGlobalMaxUbByCurIdx(runInfo.tileMaxIdx);
        LocalTensor<half> localGlobalMax2 = GetPeerGlobalMaxUbByCurIdx(runInfo.tileMaxIdx);
        LocalTensor<half> softmaxMaxOld = softmaxMaxUB;
        LocalTensor<float> urs = GetUpdateScaleByCurIdx(runInfo.updateScaleNum);

        if (unlikely(runInfo.s2FirstStartVecCore != constInfo.subBlockIdx && runInfo.isC2Sync)) { // ~1/32结构频率
            if (unlikely(runInfo.curS2LoopIdx / 16 == 0)) {
                Mxfp4Api::computeOnlyScale<true, half, s1BaseSize>(localGlobalMax1, localGlobalMax2, softmaxMaxOld, urs,
                                                                   constInfo.subBlockIdx);
            } else {
                Mxfp4Api::computeOnlyScale<false, half, s1BaseSize>(localGlobalMax1, localGlobalMax2, softmaxMaxOld,
                                                                    urs, constInfo.subBlockIdx);
            }
            PipeBarrier<PIPE_V>();
            // FD: softmaxMaxUB 会被后续任务覆盖, 最终 max 须当场写 workspace 槽
            if (unlikely(runInfo.isLastS2Loop && runInfo.isS2SplitCore)) {
                CopyFdMaxOut(runInfo);
            }
            return;
        }

        uint16_t firstLoop = 0;
        uint16_t secondLoop = 0;
        uint16_t firstLoopStart = 0;
        uint16_t secondLoopStart = 0;

        GetPScaleParams(runInfo, firstLoopStart, firstLoop, secondLoopStart, secondLoop);

        LocalTensor<uint8_t> pscale1 = this->pscaleUB;
        LocalTensor<uint8_t> pscale2 = this->pscaleUB[firstLoop * SINGLE_PSCALE_SPACE_SIZE];
        LocalTensor<half> localGroupMax1 = this->localGroupMaxUB[firstLoopStart * SINGLE_GROUP_MAX_SPACE_SIZE];
        LocalTensor<half> localGroupMax2 = this->localGroupMaxUB[secondLoopStart * SINGLE_GROUP_MAX_SPACE_SIZE];

        Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        if (unlikely(runInfo.curS2LoopIdx / 16 == 0)) {
            Mxfp4Api::computePscale<true, half, s1BaseSize>(pscale1, pscale2, localGroupMax1, localGroupMax2,
                                                            localGlobalMax1, localGlobalMax2, softmaxMaxOld, urs,
                                                            firstLoop, secondLoop, constInfo.subBlockIdx);
        } else {
            Mxfp4Api::computePscale<false, half, s1BaseSize>(pscale1, pscale2, localGroupMax1, localGroupMax2,
                                                             localGlobalMax1, localGlobalMax2, softmaxMaxOld, urs,
                                                             firstLoop, secondLoop, constInfo.subBlockIdx);
        }
        Mutex::Unlock<PIPE_MTE3>(SYNC_GMAX_UB_TO_L1_BUF0_MUTEX_ID + runInfo.tileMaxIdx);
        Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);

        DataCopy(pScaleL1[constInfo.subBlockIdx * (32 * 5 * 8) + firstLoopStart * (32 * 5 * 8)], pscale1,
                 {static_cast<uint16_t>(firstLoop), 40, 0, 40});
        if (secondLoop != 0) {
            DataCopy(pScaleL1[constInfo.subBlockIdx * (32 * 5 * 8) + secondLoopStart * (32 * 5 * 8)],
                     pscale1[firstLoop * (32 * 5 * 8)], {static_cast<uint16_t>(secondLoop), 40, 0, 40});
        }
        Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        // FD: softmaxMaxUB 会被后续任务覆盖, 最终 max 须当场写 workspace 槽
        if (unlikely(runInfo.isLastS2Loop && runInfo.isS2SplitCore)) {
            CopyFdMaxOut(runInfo);
        }
    }

    // FD max partial: softmaxMaxUB 本 subblock 行段 [sub*64, +actM) 转 fp32 后写 workspace 槽
    // (softmaxMaxUB 会被后续任务覆盖, 须在 UpdatePScale 出终值处当场写)
    // 暂存借用 FA reuse 区 (与 pscaleUB/attentionTransUB 同区), 必须持 vec1Res 所有权锁: V 写一段、MTE3 读一段
    __aicore__ inline void CopyFdMaxOut(const VecRunInfo& runInfo)
    {
        uint32_t vec2MBaseSize = s1BaseSize / 2;
        uint32_t rowBase = constInfo.subBlockIdx * vec2MBaseSize;
        uint32_t actDealMSize = (constInfo.subBlockIdx == 0) ?
                                    (runInfo.actMSize < vec2MBaseSize ? runInfo.actMSize : vec2MBaseSize) :
                                    (runInfo.actMSize > vec2MBaseSize ? runInfo.actMSize - vec2MBaseSize : 0);
        if (actDealMSize == 0) {
            return;
        }
        uint64_t slotRow = (uint64_t)runInfo.faTmpOutWsPos * s1BaseSize + rowBase;
        Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        Cast(fdMaxStage_, softmaxMaxUB[rowBase], RoundMode::CAST_NONE, actDealMSize);
        // 管线 max 为 log2 域, 归约侧按 exp(差值) 重标定, 此处换算到 ln 域存储
        Muls(fdMaxStage_, fdMaxStage_, Mxfp4Api::LN2, actDealMSize);
        Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
        DataCopyExtParams params{1, static_cast<uint32_t>(actDealMSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(fdMaxFdGm[slotRow], fdMaxStage_, params);
        Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
        Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
    }

    // FD partial 写出: acc 槽存未归一化局部分子 (attentionOutUB [d=128][m=64] fp32 → [slot][d=128][m=128]),
    // sum 紧凑 1 fp32/行; max 已在 UpdatePScale 写出, 最终 out 由归并 pass 合成
    __aicore__ inline void CopyFdPartialOut(const VecRunInfo& runInfo, uint32_t actDealMSize)
    {
        uint32_t mSize = s1BaseSize / 2;
        uint32_t rowBase = constInfo.subBlockIdx * mSize;
        uint64_t slotRow = (uint64_t)runInfo.faTmpOutWsPos * s1BaseSize + rowBase;
        // acc/sum 均由 V 段产出, 经 attn buffer mutex 把 V 完成态传给 MTE3
        Mutex::Unlock<PIPE_V>(SYNC_ATTN_BUF_MUTEX_ID);
        Mutex::Lock<PIPE_MTE3>(SYNC_ATTN_BUF_MUTEX_ID);
        // acc: UB [d][m] 行距 256B → GM [d][m=128] 行距 512B, 只落本 subblock 的 m 段
        DataCopyExtParams accParams{static_cast<uint16_t>(constInfo.dSize),
                                    static_cast<uint32_t>(mSize * sizeof(float)), 0,
                                    static_cast<uint32_t>(mSize * sizeof(float)), 0};
        DataCopyPad(fdAccumOutFdGm[(uint64_t)runInfo.faTmpOutWsPos * dBaseSize * s1BaseSize + rowBase], attentionOutUB,
                    accParams);
        // sum 紧凑 fp32 槽直出 (槽内每行 1 个 fp32; globalRowsumUB 为本 subblock 行段)
        DataCopyExtParams compactParams{1, static_cast<uint32_t>(actDealMSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(fdSumFdGm[slotRow], globalRowsumUB, compactParams);
        Mutex::Unlock<PIPE_MTE3>(SYNC_ATTN_BUF_MUTEX_ID);
    }

    __aicore__ inline void ComputeVec2(const VecRunInfo& runInfo)
    {
        LocalTensor<float> urs = GetUpdateScaleByCurIdx(runInfo.updateScaleNum);

        // 锁与数值初始化必须分开判定：
        //   锁：沿用 master 的"前 16 个 tile 都取锁"语义。本 subblock 在 actDealMSize==0 的收尾路径
        //       有一次未配对的 Unlock，依赖这段超配的锁计数；若收紧成"只首 tile 取锁"，两个 AIV
        //       子块变成严格互斥，持有方等 peer 数据、peer 等锁 → 死锁
        //       （实测 B=33/4K、B=64/64K 必挂，与 FD、与 batch>核数均无关）。
        //   数值：只有本核 s2 循环的第一个 tile 做初始化，其后一律累加。
        if (unlikely(runInfo.curS2LoopIdx / 16 == 0)) {
            Mutex::Lock<PIPE_V>(SYNC_ATTN_BUF_MUTEX_ID);
            Mxfp4Api::processUpdate<false>(attentionOutUB, mm2ResUB, urs, globalRowsumUB, localRowsumUB);
        } else {
            Mxfp4Api::processUpdate<true>(attentionOutUB, mm2ResUB, urs, globalRowsumUB, localRowsumUB);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            uint32_t vec2MBaseSize = (s1BaseSize + 1) / 2;
            uint32_t actDealMSize = runInfo.actMSize < vec2MBaseSize ? runInfo.actMSize : vec2MBaseSize;
            if (constInfo.subBlockIdx != 0) {
                actDealMSize = runInfo.actMSize < vec2MBaseSize ? 0 : (runInfo.actMSize - vec2MBaseSize);
            }
            PipeBarrier<PIPE_V>();
            if (unlikely(runInfo.isS2SplitCore)) { // FD: 本行 S2 被切, partial 写 workspace, 最终 out 由归并 pass 写出
                if (actDealMSize != 0) {
                    CopyFdPartialOut(runInfo, actDealMSize);
                } else {
                    Mutex::Unlock<PIPE_V>(SYNC_ATTN_BUF_MUTEX_ID);
                }
                return;
            }
            if (actDealMSize != 0) {
                Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
                Mutex::Lock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
                LocalTensor<float> outTensorFp32Ub = attentionTransUB.template ReinterpretCast<float>();
                Mxfp4Api::processOut(outTensorFp32Ub, attentionOutUB, globalRowsumUB);
                Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
                Mutex::Unlock<PIPE_V>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
                Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
                Mutex::Lock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);
                LocalTensor<OUT_T> outTensorBf16Ub = attentionTransUB.template ReinterpretCast<OUT_T>();
                LocalTensor<OUT_T> attentionOutUBBf16 = attentionOutUB.template ReinterpretCast<OUT_T>();
                uint32_t columnCnt = constInfo.dSize + AttentionCommon::BYTE_BLOCK / sizeof(OUT_T);
                DataCopy(attentionOutUBBf16, outTensorBf16Ub, actDealMSize * columnCnt);
                Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF0_MUTEX_ID);
                Mutex::Unlock<PIPE_MTE3>(SYNC_VEC1_RES_BUF1_MUTEX_ID);

                Mutex::Unlock<PIPE_V>(SYNC_ATTN_BUF_MUTEX_ID);
                Mutex::Lock<PIPE_MTE3>(SYNC_ATTN_BUF_MUTEX_ID);

                GmCoord gmCoord;
                gmCoord.bIdx = runInfo.bIdx;
                gmCoord.n2Idx = runInfo.n2Idx;
                gmCoord.gS1Idx = runInfo.gS1Idx + constInfo.subBlockIdx * vec2MBaseSize;
                gmCoord.dIdx = 0;
                gmCoord.gS1DealSize = actDealMSize;
                gmCoord.dDealSize = constInfo.dSize;

                FaUbTensor<OUT_T> ubTensor{
                    .tensor = attentionOutUBBf16, .rowCount = actDealMSize, .colCount = columnCnt};

                AttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
            } else {
                Mutex::Unlock<PIPE_V>(SYNC_ATTN_BUF_MUTEX_ID);
                return;
            }
            Mutex::Unlock<PIPE_MTE3>(SYNC_ATTN_BUF_MUTEX_ID);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        InitOutputSingleCore();
        SyncAll();
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize =
            (LAYOUT_Q == QFA_LAYOUT::TND) ? static_cast<int64_t>(constInfo.t1Size) : constInfo.bSize * constInfo.s1Size;

        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (likely(singleInitOutputSize > 0)) {
            Mutex::Lock<PIPE_V>(SYNC_INIT_OUTPUT_MUTEX_ID);
            constexpr int64_t initChunkSize = INIT_OUTPUT_UB_BYTES / sizeof(OUT_T);
            LocalTensor<OUT_T> initUb = mm1ResUB.template ReinterpretCast<OUT_T>();
            Duplicate<OUT_T>(initUb, static_cast<OUT_T>(0), initChunkSize);
            Mutex::Unlock<PIPE_V>(SYNC_INIT_OUTPUT_MUTEX_ID);
            Mutex::Lock<PIPE_MTE3>(SYNC_INIT_OUTPUT_MUTEX_ID);
            int64_t yOffset = constInfo.aivIdx * singleCoreSize;
            int64_t movRound = singleInitOutputSize / initChunkSize;
            int64_t movTail = singleInitOutputSize - movRound * initChunkSize;
            DataCopyExtParams ub2GmParams{1, static_cast<uint32_t>(initChunkSize * sizeof(OUT_T)), 0, 0, 0};
            for (int64_t i = 0; i < movRound; ++i) {
                DataCopyPad(outGmTensor.gmTensor[yOffset], initUb, ub2GmParams);
                yOffset += initChunkSize;
            }
            if (movTail != 0) {
                ub2GmParams.blockLen = static_cast<uint32_t>(movTail * sizeof(OUT_T));
                DataCopyPad(outGmTensor.gmTensor[yOffset], initUb, ub2GmParams);
            }
            Mutex::Unlock<PIPE_MTE3>(SYNC_INIT_OUTPUT_MUTEX_ID);
        }
    }

    // V2 attentionOut GM初始化
    __aicore__ inline void InitAttentionOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize,
                                                  uint32_t qSeqSize, uint32_t headDim,
                                                  const SeqLensTool<LAYOUT_Q, SEQLEN_T>& qSeqLensTool,
                                                  FaGmTensor<OUT_T, OUT_FORMAT, SEQLEN_T, OUT_IS_TND>& outGmTensor,
                                                  __gm__ uint8_t* attentionOut)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetBufferIdx(const uint32_t loop)
    {
        uint32_t vecNum = 2;
        uint32_t doubleBuffer = 2;
        return loop / vecNum % doubleBuffer;
    }

    __aicore__ inline LocalTensor<half> GetPeerGlobalMaxUbByCurIdx(uint32_t spaceIdx)
    {
        return peerGlobalMaxUB[spaceIdx * L1_SINGLE_GLOBAL_MAX_SIZE];
    }

    __aicore__ inline LocalTensor<half> GetLocalGlobalMaxUbByCurIdx(uint32_t spaceIdx)
    {
        return localGlobalMaxUB[spaceIdx * L1_SINGLE_GLOBAL_MAX_SIZE];
    }

    __aicore__ inline LocalTensor<half> GetLocalGrpMaxUbByLoopIdx(uint32_t loop)
    {
        uint32_t spaceIdx = loop / 2 * 2 % 20;
        return localGroupMaxUB[spaceIdx * SINGLE_GROUP_MAX_SPACE_SIZE];
    }

    __aicore__ inline LocalTensor<float> GetUpdateScaleByCurIdx(uint32_t spaceIdx)
    {
        return updateScaleUB[spaceIdx * UB_UPDATE_SIZE];
    }

    __aicore__ inline void GetPScaleParams(const VecRunInfo& runInfo, uint16_t& firstLoopStart, uint16_t& firstLoop,
                                           uint16_t& secondLoopStart, uint16_t& secondLoop)
    {
        uint32_t subBlockIdx = constInfo.subBlockIdx;
        uint16_t isLoopFirstTaskVecCore = (subBlockIdx == runInfo.s2FirstStartVecCore);
        uint32_t loop = runInfo.loop;
        uint32_t GROUP_MAX_SPACE_LEN = 20;
        uint32_t WHOLE_PROCESS_LOOP = 16;

        uint32_t curS2LoopIdx = runInfo.curS2LoopIdx;
        // 求单核目前为止，计算对应的第几个softmax(即第几个groupmax 8*128)（跨batch）
        // 等价groupMaxEndLoop = (loop % 2 == 0) ? ((subBlockIdx == 0) ? (loop / 2) : ((loop / 2) - 1)) : (loop / 2);
        uint32_t groupMaxEndLoop = (loop >> 1) - (subBlockIdx != 0 && loop % 2 == 0);
        // 这一轮pscale的任务loop在groupMax空间下发的groupMaxEndIdx，*2的原因是因为groupmax一片空间承载4*128
        uint32_t groupMaxEndIdx = groupMaxEndLoop * 2 % GROUP_MAX_SPACE_LEN + 1;

        // toProcessGroupMaxLoopLen 求当前FD切核对应的Tile(16个softmax是一个tile)在每个核上有占用几片groupmax(4*128)空间
        uint16_t toProcessGroupMaxLoopLen =
            (curS2LoopIdx + 1) - (curS2LoopIdx / WHOLE_PROCESS_LOOP) * WHOLE_PROCESS_LOOP + isLoopFirstTaskVecCore;
        toProcessGroupMaxLoopLen = toProcessGroupMaxLoopLen / 2 * 2;
        toProcessGroupMaxLoopLen =
            toProcessGroupMaxLoopLen < WHOLE_PROCESS_LOOP ? toProcessGroupMaxLoopLen : WHOLE_PROCESS_LOOP;
        firstLoop = (toProcessGroupMaxLoopLen / 2 + 1) / 2 * 2;
        secondLoop = toProcessGroupMaxLoopLen - firstLoop;
        int32_t firstLoopStartNotU = groupMaxEndIdx - toProcessGroupMaxLoopLen + 1;
        if (firstLoopStartNotU < 0) {
            firstLoopStart = firstLoopStartNotU + GROUP_MAX_SPACE_LEN;
            firstLoop = GROUP_MAX_SPACE_LEN - firstLoopStart;
            secondLoop = toProcessGroupMaxLoopLen - firstLoop;
            secondLoopStart = 0;
        } else {
            firstLoopStart = firstLoopStartNotU;
            secondLoopStart = firstLoopStart + firstLoop; // 向上对齐到2
        }
        // 两片groupmax空间对应其中的一个softmax任务
        firstLoop /= 2;
        secondLoop /= 2;
    }
};
} // namespace QFA_KERNEL
#endif // QUANT_FLASH_ATTN_BLOCK_VECTOR_DN_H_
