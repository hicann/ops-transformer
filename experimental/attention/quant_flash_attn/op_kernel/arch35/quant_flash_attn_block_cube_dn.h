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
 * \file quant_flash_atten_block_cube_dn.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_DN_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_DN_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attn_template_tiling_key.h"
#include "quant_flash_attn_common_def.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;

namespace QFA_KERNEL {

template <QFA_LAYOUT LAYOUT>
__aicore__ inline constexpr bool IS_TND()
{
    return (LAYOUT == QFA_LAYOUT::TND);
}

template <typename T>
__aicore__ inline constexpr bool IS_4_BIT_WIDTH()
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    return (IsSameType<T, int4b_t>::value || IsSameType<T, hifloat4x2_t>::value || IsSameType<T, fp4x2_e2m1_t>::value);
#else
    return IsSameType<T, int4b_t>::value;
#endif
}

template <typename T>
__aicore__ inline constexpr uint32_t GetBlockElemCnt()
{
    if constexpr (IS_4_BIT_WIDTH<T>()) {
        return AttentionCommon::BYTE_BLOCK * 2;
    } else {
        return AttentionCommon::BYTE_BLOCK / sizeof(T);
    }
}

template <typename QFAT, const bool ND = false>
class QuantFlashAttnBlockCubeDn {
public:
    static constexpr bool ND_FLAG = ND; // 供 kernel 侧按 ND 调整 M 维对齐
    using QUANT_T = typename QFAT::quantType;
    using SCALE_T = typename QFAT::scaleType;
    using OUT_T = typename QFAT::outputType;
    using SEQLEN_T = uint32_t;
    static constexpr bool SOFTMAX_DN = true;
    static constexpr bool PAGE_ATTENTION = QFAT::pageAttention;
    static constexpr bool HAS_MASK = QFAT::hasMask;
    static constexpr QFA_LAYOUT LAYOUT_Q = QFAT::qLayout;
    static constexpr QFA_LAYOUT LAYOUT_KV = QFAT::kvLayout;

    static constexpr uint32_t mBaseSize = 128;
    static constexpr uint32_t s2BaseSize = 256;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;

private:
    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t QK_L0_S2_SPLIT_SIZE = 128;
    using COMPUTE_T = float;
    using DATA_T = uint8_t;
    static constexpr uint8_t VEC0 = 0;
    static constexpr uint8_t VEC1 = 1;

    const ConstInfo& constInfo;
    const SeqLensTool<LAYOUT_Q, SEQLEN_T>& qSeqLensTool;
    const SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool;

    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<DATA_T> queryGm;
    GlobalTensor<DATA_T> keyGm;
    GlobalTensor<DATA_T> valueGm;
    GlobalTensor<SCALE_T> queryScaleGm;
    GlobalTensor<SCALE_T> keyScaleGm;
    GlobalTensor<SCALE_T> valueScaleGm;

    static constexpr bool Q_TND = (LAYOUT_Q == QFA_LAYOUT::TND);
    static constexpr bool KV_TND = (LAYOUT_KV == QFA_LAYOUT::TND);
    // PA blockSize ∈ {64..1024} 均为 2 的幂: 页号/块内偏移用移位代替除模
    uint32_t paBlockShift = 0;
    uint32_t paBlockMask = 0;
    uint32_t kvHeadNumCached = 0;
    DataCopyParams kvNzFullTileParams; // PA_NZ: 整 tile 在同一 page 内的常量搬运参数
    Nd2NzParams qNd2NzParams;
    uint64_t qGmStrideB;
    uint64_t qGmStrideN2;
    uint64_t qGmStrideS1;
    Nd2NzParams kvNd2NzParams;
    uint64_t kvGmStrideB;
    uint64_t kvGmStrideN2;
    uint64_t kvGmStrideS2;
    Dn2NzParams qScaleDn2NzParams;
    uint64_t qScaleGmStrideB;
    uint64_t qScaleGmStrideN2;
    uint64_t qScaleGmStrideS1;
    Dn2NzParams kScaleDn2NzParams;
    uint64_t kScaleGmStrideB;
    uint64_t kScaleGmStrideN2;
    uint64_t kScaleGmStrideS2;
    Nd2NzParams vScaleNd2NzParams;
    uint64_t vScaleGmStrideB;
    uint64_t vScaleGmStrideN2;
    uint64_t vScaleGmStrideS2;

    // =================================L1 Buffer=================================
    // ND: M 维实际恒 <= ND_M_ROWS(与 host 端 ND_MAX_M_ROWS 一致), 按 M 行开缓冲
    static constexpr uint32_t ND_M_ROWS = 64;
    // UB 里 P 区的字节数(2 个 chunk, 与 vector block 的 UB_P_BYTES 必须一致)
    static constexpr uint32_t UB_N_P_BYTES =
        (ND ? ((ND_M_ROWS * 256U) > 8192U ? (ND_M_ROWS * 256U) : 8192U) : (64U * 256U)) * 2U;
    static constexpr uint32_t L1_Q_SIZE = 128 * 128 / 2;                              // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_Q_DESCALE_SIZE = 128 * (128 / 32) * sizeof(SCALE_T); // 0.5K
    static constexpr uint32_t L1_KV_SIZE = 512 * 128 / 2; // 32K, 2个fp4_e2m1元素为1B, s2单次最大256行
    static constexpr uint32_t L1_KV_DESCALE_SIZE = 512 * (128 / 32) * sizeof(SCALE_T); // 2K
    static constexpr uint32_t L1_P_SIZE = (ND ? 64U : 128U) * 256U / 2; // ND: 64 行 = 8K, DN: 128 行 = 16K
    static constexpr uint32_t L1_P_DESCALE_SIZE = 128 * 2 * (256 / 64 + 1) * sizeof(SCALE_T); // 1.25K
    static constexpr uint32_t L1_Q_BUFCNT = 1;                                                // 单槽, Q_MUTEX 串行
    static constexpr uint32_t L1_KV_BUFCNT = 4;
    static constexpr uint32_t L1_P_BUFCNT = 20;

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    // 静态
    LocalTensor<DATA_T> pL1Tensor;
    LocalTensor<SCALE_T> pDescaleL1Tensor;
    LocalTensor<DATA_T> qL1Tensor;
    LocalTensor<SCALE_T> qDescaleL1Tensor;
    LocalTensor<DATA_T> kvL1Tensor;
    LocalTensor<SCALE_T> kvDescaleL1Tensor;
    LocalTensor<half> localGlobalMaxL1;
    LocalTensor<DATA_T> vSumL1Tensor;
    LocalTensor<SCALE_T> vScaleSumL1Tensor;

    // =================================L0 Buffer=================================
    static constexpr uint32_t QK_L0A_SIZE = 128 * 128 / 2;                 // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t QK_L0B_SIZE = 128 * 128 / 2;                 // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t QK_L0C_SIZE = 128 * 128 * sizeof(COMPUTE_T); // 64K
    static constexpr uint32_t QK_L0AB_BUFCNT = 2;
    static constexpr uint32_t QK_L0C_BUFCNT = 2;
    static constexpr uint32_t PV_L0A_SIZE = 256 * (128 + 16) / 2;                 // 18K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t PV_L0B_SIZE = 256 * (128 + 16) / 2;                 // 18K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t PV_L0C_SIZE = (128 + 16) * 128 * sizeof(COMPUTE_T); // 72K
    static constexpr uint32_t V_SCALE_L0A_SIZE = (256 / 64) * ((128 + 16) / 16) * 32;
    static constexpr uint32_t PV_L0AB_BUFCNT = 2;
    static constexpr uint32_t PV_L0C_BUFCNT = 1;

    // 静态
    // L0A
    LocalTensor<DATA_T> qkL0ATensor;
    LocalTensor<DATA_T> pvL0ATensor;

    // L0B
    LocalTensor<DATA_T> qkL0BTensor;
    LocalTensor<DATA_T> pvL0BTensor;

    // L0C
    LocalTensor<float> qkL0CTensor;
    LocalTensor<float> pvL0CTensor;

    // UB
    LocalTensor<half> mm1ResUB;
    LocalTensor<float> mm2ResUB;
    LocalTensor<half> peerGlobalMaxUB;

    static constexpr uint32_t UB_MM1_RES_OFFSET = 0U;
    static constexpr uint32_t UB_MM2_RES_OFFSET = ((ND ? ND_M_ROWS : 128U) * 256U * 2U * 2U);
    static constexpr uint32_t UB_PEER_GLOBAL_MAX_OFFSET =
        ((((ND ? ND_M_ROWS : 128U) * 256U * 2U * 2U) + 128U * 64U * 4U + 2U * 64U * 4U + UB_N_P_BYTES) +
         (128U * 64U * 4U));

    static constexpr uint32_t Q_MUTEX_ID_BASE = 0;

    static constexpr uint32_t KV_MUTEX_ID_BASE = 1;
    uint32_t kvBufId = 0;
    uint32_t kDataBufId = 0;
    uint32_t vDataBufId = 0;
    static constexpr uint32_t VSUM_MUTEX_ID_BASE = 9;
    static constexpr uint32_t P_FILL_MUTEX_ID = 25;
    static constexpr uint32_t QK_L0AB_MUTEX_ID_BASE = 10;
    uint32_t qkL0abBufId = 0;
    static constexpr uint32_t PV_L0AB_MUTEX_ID_BASE = 12;
    uint32_t pvL0abBufId = 0;
    static constexpr uint32_t QK_L0C_MUTEX_ID_BASE = 14;
    uint32_t qkL0cBufId = 0;
    static constexpr uint32_t PV_L0C_MUTEX_ID_BASE = 16;
    uint32_t pvL0cBufId = 0;

    uint32_t headDimInt8 = 0;

    LoadData2DParamsV2 loadKParamsA;
    LoadData2DMxParams loadKParamsMx;
    MmadParams mmadQKParams;
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeMm1Params;

public:
    __aicore__ inline QuantFlashAttnBlockCubeDn(ConstInfo& constInfo, SeqLensTool<LAYOUT_Q, SEQLEN_T>& qSeqLensTool,
                                                SeqLensTool<LAYOUT_KV, SEQLEN_T>& kvSeqLensTool)
        : constInfo(constInfo),
          qSeqLensTool(qSeqLensTool),
          kvSeqLensTool(kvSeqLensTool){};

    __aicore__ inline void InitInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* dequantScaleKey,
                                     __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
            while ((1U << paBlockShift) < constInfo.blockSize) {
                ++paBlockShift;
            }
            paBlockMask = constInfo.blockSize - 1U;
        }
        kvHeadNumCached = QfaKvHeadNum(constInfo.n2Size, constInfo.gSize, constInfo.gRealSize);

        headDimInt8 = (constInfo.dSize / GetBlockElemCnt<QUANT_T>()) * GetBlockElemCnt<DATA_T>();
        queryGm.SetGlobalBuffer((__gm__ DATA_T*)query);
        keyGm.SetGlobalBuffer((__gm__ DATA_T*)key);
        valueGm.SetGlobalBuffer((__gm__ DATA_T*)value);
        queryScaleGm.SetGlobalBuffer((__gm__ SCALE_T*)dequantScaleQuery);
        keyScaleGm.SetGlobalBuffer((__gm__ SCALE_T*)dequantScaleKey);
        valueScaleGm.SetGlobalBuffer((__gm__ SCALE_T*)dequantScaleValue);

        InitGmCopyConsts();
    }

    __aicore__ inline void InitTensors()
    {
        // L1
        uint32_t addrL1Start = 0;
        pL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_P_SIZE * L1_P_BUFCNT); // 16K * 20 = 320K

        addrL1Start += L1_P_SIZE * L1_P_BUFCNT;
        pDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_P_DESCALE_SIZE * L1_P_BUFCNT); // 1.25K * 20 = 25K

        addrL1Start += L1_P_DESCALE_SIZE * L1_P_BUFCNT;
        qL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_Q_SIZE * L1_Q_BUFCNT); // 8K * 1 = 8K

        addrL1Start += L1_Q_SIZE * L1_Q_BUFCNT;
        qDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_Q_DESCALE_SIZE * L1_Q_BUFCNT); // 0.5K * 1 = 0.5K

        addrL1Start += L1_Q_DESCALE_SIZE * L1_Q_BUFCNT;
        kvL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, L1_KV_SIZE * L1_KV_BUFCNT); // 32K * 4 = 128K

        addrL1Start += L1_KV_SIZE * L1_KV_BUFCNT;
        kvDescaleL1Tensor =
            LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, L1_KV_DESCALE_SIZE * L1_KV_BUFCNT); // 2K * 4 = 8K

        addrL1Start += L1_KV_DESCALE_SIZE * L1_KV_BUFCNT;
        vSumL1Tensor = LocalTensor<DATA_T>(TPosition::A1, addrL1Start, PV_L0A_SIZE); // 18K * 1 = 18K

        addrL1Start += PV_L0A_SIZE;
        vScaleSumL1Tensor = LocalTensor<SCALE_T>(TPosition::A1, addrL1Start, V_SCALE_L0A_SIZE); // 1.125K * 1

        addrL1Start += V_SCALE_L0A_SIZE;
        // 不再依赖顺序累加的结果: 与 vector 侧共用 DnL1Map 的常量, 并用 static_assert 锁死两者一致.
        // ND 下 L1_P_SIZE 变小会让这一段整体前移, 若 vector 侧还硬编码 DN 的偏移,
        // peer max 就会写到一个没人读的地址(交换静默失效).
        static_assert(
            (L1_P_SIZE * L1_P_BUFCNT + 128U * 2U * (256U / 64U + 1U) * 20U + 128U * 128U / 2U + 128U * (128U / 32U) +
             512U * 128U / 2U * 4U + 512U * (128U / 32U) * 4U + 256U * (128U + 16U) / 2U +
             (256U / 64U) * ((128U + 16U) / 16U) * 32U) ==
                L1_P_SIZE * L1_P_BUFCNT + L1_P_DESCALE_SIZE * L1_P_BUFCNT + L1_Q_SIZE * L1_Q_BUFCNT +
                    L1_Q_DESCALE_SIZE * L1_Q_BUFCNT + L1_KV_SIZE * L1_KV_BUFCNT + L1_KV_DESCALE_SIZE * L1_KV_BUFCNT +
                    PV_L0A_SIZE + V_SCALE_L0A_SIZE,
            "Dn(L1_P_SIZE * L1_P_BUFCNT + 128U * 2U * (256U / 64U + 1U) * 20U + 128U * 128U / 2U + 128U * "
            "(128U / 32U) + 512U * 128U / 2U * 4U + 512U * (128U / 32U) * 4U + 256U * (128U + 16U) / 2U + (256U / 64U) "
            "* ((128U + 16U) / 16U) * 32U) 与 cube 侧 L1 分段不一致: peer max 会交换到错误地址");
        localGlobalMaxL1 =
            LocalTensor<half>(TPosition::A1,
                              (L1_P_SIZE * L1_P_BUFCNT + 128U * 2U * (256U / 64U + 1U) * 20U + 128U * 128U / 2U +
                               128U * (128U / 32U) + 512U * 128U / 2U * 4U + 512U * (128U / 32U) * 4U +
                               256U * (128U + 16U) / 2U + (256U / 64U) * ((128U + 16U) / 16U) * 32U),
                              256U * 4U);

        // L0A
        uint32_t addrL0AStart = 0;
        qkL0ATensor = LocalTensor<DATA_T>(TPosition::A2, addrL0AStart, QK_L0A_SIZE * QK_L0AB_BUFCNT); // 8K * 2 = 16K

        addrL0AStart += QK_L0A_SIZE * QK_L0AB_BUFCNT;
        pvL0ATensor = LocalTensor<DATA_T>(TPosition::A2, addrL0AStart, PV_L0A_SIZE * PV_L0AB_BUFCNT); // 18K * 2 = 36K

        // L0B
        uint32_t addrL0BStart = 0;
        qkL0BTensor = LocalTensor<DATA_T>(TPosition::B2, addrL0BStart, QK_L0B_SIZE * QK_L0AB_BUFCNT); // 8K * 2 = 16K

        addrL0BStart += QK_L0B_SIZE * QK_L0AB_BUFCNT;
        pvL0BTensor = LocalTensor<DATA_T>(TPosition::B2, addrL0BStart, PV_L0B_SIZE * PV_L0AB_BUFCNT); // 8K * 2 = 16K

        // L0C
        uint32_t addrL0CStart = 0;
        qkL0CTensor = LocalTensor<float>(TPosition::CO1, addrL0CStart,
                                         QK_L0C_SIZE * QK_L0C_BUFCNT / sizeof(float)); // 64K * 2 = 128K

        addrL0CStart += QK_L0C_SIZE * QK_L0C_BUFCNT;
        pvL0CTensor = LocalTensor<float>(TPosition::CO1, addrL0CStart,
                                         PV_L0C_SIZE * PV_L0C_BUFCNT / sizeof(float)); // 72K * 1 = 72K

        InitL0BufferForReduceSum();
        InitL0MXBufferForReduceSum();

        // Vector UB
        mm1ResUB = LocalTensor<half>(TPosition::VECCALC, UB_MM1_RES_OFFSET, 256 * 128 * 2);
        mm2ResUB = LocalTensor<float>(TPosition::VECCALC, UB_MM2_RES_OFFSET, 128 * 64);
        peerGlobalMaxUB = LocalTensor<half>(TPosition::VECCALC, UB_PEER_GLOBAL_MAX_OFFSET, 128 * 4);

        InitComputeParams();
    }

    __aicore__ inline void InitComputeParams()
    {
        loadKParamsA.kStartPosition = 0;
        loadKParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();
        loadKParamsA.srcStride = s2BaseSize * 2 / 16;
        loadKParamsA.ifTranspose = false;
        loadKParamsMx.yStartPosition = 0;
        loadKParamsMx.yStep = loadKParamsA.kStep;
        loadKParamsMx.srcStride = loadKParamsMx.yStep;
        loadKParamsMx.dstStride = loadKParamsMx.yStep;

        mmadQKParams.k = constInfo.dSize;
        mmadQKParams.cmatrixInitVal = true;
        mmadQKParams.cmatrixSource = false;
        mmadQKParams.disableGemv = true;

        fixpipeMm1Params.dstStride = 512 / sizeof(half); // S UB 行距 256B, ND/DN 一致
        fixpipeMm1Params.quantPre = QuantMode_t::QF322F16_PRE;
        fixpipeMm1Params.deqScalar = static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(&constInfo.scaleValue));
        fixpipeMm1Params.dualDstCtl = 0;
        fixpipeMm1Params.params.ndNum = 1;
        fixpipeMm1Params.params.dstNdStride = 512 / sizeof(half);

        if constexpr (PAGE_ATTENTION) {
            // 满 tile(rows == s2BaseSize)时所有字段与块无关, 预构一次
            kvNzFullTileParams.blockCount = 2;
            kvNzFullTileParams.blockLen = s2BaseSize;
            kvNzFullTileParams.dstStride = s2BaseSize;
            kvNzFullTileParams.srcStride = constInfo.blockSize - s2BaseSize;
        }
    }

    __aicore__ inline void ComputeMm1(const CubeRunInfo& info)
    {
        if (unlikely(info.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(Q_MUTEX_ID_BASE);
            CopyQGmToL1(info);
            CopyQScaleGmToL1(info);
            Mutex::Unlock<PIPE_MTE2>(Q_MUTEX_ID_BASE);
            Mutex::Lock<PIPE_MTE1>(Q_MUTEX_ID_BASE);
            for (uint32_t i = 0; i < QK_L0AB_BUFCNT; ++i) {
                Mutex::Lock<PIPE_MTE1>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                LoadQToL0(info); // 128 * 128
                Mutex::Unlock<PIPE_MTE1>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                qkL0abBufId = (qkL0abBufId + 1) % 2;
            }
            Mutex::Unlock<PIPE_MTE1>(Q_MUTEX_ID_BASE);
        }

        if (!info.prefetched) {
            Mutex::Lock<PIPE_MTE2>(KV_MUTEX_ID_BASE + kvBufId * 2);
            Mutex::Lock<PIPE_MTE2>(KV_MUTEX_ID_BASE + kvBufId * 2 + 1);
            CopyKvGmToL1(info, keyGm);
            CopyKScaleGmToL1(info);
            Mutex::Unlock<PIPE_MTE2>(KV_MUTEX_ID_BASE + kvBufId * 2);
            Mutex::Unlock<PIPE_MTE2>(KV_MUTEX_ID_BASE + kvBufId * 2 + 1);
            kDataBufId = kvBufId;
            kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;
        }

        {
            uint32_t kHalf = KV_MUTEX_ID_BASE + kDataBufId * 2 + (info.prefetched ? 1 : 0);
            Mutex::Lock<PIPE_MTE1>(kHalf);
            uint32_t loopCnt = CeilDiv((uint32_t)info.actSingleLoopS2Size, (uint32_t)QK_L0_S2_SPLIT_SIZE);
            uint32_t actS2Size = QK_L0_S2_SPLIT_SIZE;
            uint32_t actS2SizeAlign = QK_L0_S2_SPLIT_SIZE;
            for (uint32_t loop = 0; loop < loopCnt; ++loop) {
                if (loop + 1 == loopCnt) {
                    actS2Size = info.actSingleLoopS2Size - loop * QK_L0_S2_SPLIT_SIZE;
                    actS2SizeAlign = info.actSingleLoopS2SizeAlign16 - loop * QK_L0_S2_SPLIT_SIZE;
                }

                Mutex::Lock<PIPE_M>(QK_L0C_MUTEX_ID_BASE + qkL0cBufId);
                {
                    Mutex::Lock<PIPE_MTE1>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                    LoadKToL0(info, loop, actS2SizeAlign);
                    Mutex::Unlock<PIPE_MTE1>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                    Mutex::Lock<PIPE_M>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                    MatmulQK(info, actS2Size);
                    Mutex::Unlock<PIPE_M>(QK_L0AB_MUTEX_ID_BASE + qkL0abBufId);
                    qkL0abBufId = (qkL0abBufId + 1) % 2;
                }
                Mutex::Unlock<PIPE_M>(QK_L0C_MUTEX_ID_BASE + qkL0cBufId);
                Mutex::Lock<PIPE_FIX>(QK_L0C_MUTEX_ID_BASE + qkL0cBufId);
                FixpipeMm1(info, loop, actS2Size, actS2SizeAlign); // 128 * 256
                Mutex::Unlock<PIPE_FIX>(QK_L0C_MUTEX_ID_BASE + qkL0cBufId);
                qkL0cBufId = (qkL0cBufId + 1) % 2;
            }
            Mutex::Unlock<PIPE_MTE1>(kHalf);
        }
    }

    __aicore__ inline void ComputeMm2(const CubeRunInfo& info)
    {
        if constexpr (!ND) {
            if (info.actSingleLoopS2SizeAlign != info.actSingleLoopS2SizeAlign64) {
                Mutex::Lock<PIPE_MTE2>(P_FILL_MUTEX_ID);
                InitConstValueParams<uint16_t> PL1InitParams(
                    1, static_cast<uint16_t>(info.actSingleLoopS2SizeAlign64 - info.actSingleLoopS2SizeAlign), 0,
                    0x0000);
                uint32_t l1P_Base_Offset = (info.loop % 20) * L1_P_SIZE;
                Fill(pL1Tensor[l1P_Base_Offset + info.actSingleLoopS2SizeAlign * 32]
                         .template ReinterpretCast<uint16_t>(),
                     PL1InitParams);
                if (likely(info.actMSizeAlign128 == mBaseSize)) {
                    Fill(pL1Tensor[l1P_Base_Offset + 2 * info.actSingleLoopS2SizeAlign * 32 + 32 * 32]
                             .template ReinterpretCast<uint16_t>(),
                         PL1InitParams);
                }
                Mutex::Unlock<PIPE_MTE2>(P_FILL_MUTEX_ID);
            }
        }
        uint32_t vBuf = info.prefetched ? vDataBufId : kvBufId;
        uint32_t vHalfA = KV_MUTEX_ID_BASE + vBuf * 2;
        bool needVFill = (info.actSingleLoopS2Size != info.actSingleLoopS2SizeAlign64);
        if (info.prefetched) {
            if (unlikely(needVFill)) {
                Mutex::Lock<PIPE_MTE2>(vHalfA + 1); // 预取分支内只有 pad Fill, 无 Fill 时无需加锁
            }
        } else {
            Mutex::Lock<PIPE_MTE2>(vHalfA);
            Mutex::Lock<PIPE_MTE2>(vHalfA + 1);
        }
        if (needVFill) {
            InitConstValueParams<uint16_t> kvL1InitParams(
                1, static_cast<uint16_t>((info.actSingleLoopS2SizeAlign64 - info.actSingleLoopS2Size)), 0, 0x0000);
            uint32_t l1V_Base_Offset = vBuf * L1_KV_SIZE;
            uint32_t vOffset = info.prefetched ? s2BaseSize * (constInfo.dSize / 4) : 0;
            Fill(kvL1Tensor[l1V_Base_Offset + vOffset + constInfo.dSize / 4 * info.actSingleLoopS2Size]
                     .template ReinterpretCast<uint16_t>(),
                 kvL1InitParams);
            Fill(kvL1Tensor[l1V_Base_Offset + vOffset + constInfo.dSize / 4 * (2 * s2BaseSize) +
                            constInfo.dSize / 4 * info.actSingleLoopS2Size]
                     .template ReinterpretCast<uint16_t>(),
                 kvL1InitParams);
        }

        if (!info.prefetched) {
            CopyKvGmToL1(info, valueGm);
            CopyVScaleGmToL1(info);
        }

        if (info.prefetched) {
            if (unlikely(needVFill)) {
                Mutex::Unlock<PIPE_MTE2>(vHalfA + 1);
            }
        } else {
            Mutex::Unlock<PIPE_MTE2>(vHalfA);
            Mutex::Unlock<PIPE_MTE2>(vHalfA + 1);
        }
        if (!info.prefetched) {
            vDataBufId = kvBufId;
            kvBufId = (kvBufId + 1) % L1_KV_BUFCNT;
        }
        if (unlikely(info.isLastS2Loop)) {
            InitL0MXBufferForReduceSum(false);
        }
        uint32_t vHalf = vHalfA + (info.prefetched ? 1 : 0);
        if (unlikely(info.isC2Sync)) {
            Mutex::Lock<PIPE_M>(PV_L0C_MUTEX_ID_BASE);
        }
        Mutex::Lock<PIPE_MTE1>(PV_L0AB_MUTEX_ID_BASE + pvL0abBufId);
        {
            bool isPadFill = unlikely(info.actSingleLoopS2SizeAlign != info.actSingleLoopS2SizeAlign64);
            if (isPadFill) {
                Mutex::Lock<PIPE_MTE1>(P_FILL_MUTEX_ID);
            }
            LoadPToL0(info);
            if (isPadFill) {
                Mutex::Unlock<PIPE_MTE1>(P_FILL_MUTEX_ID);
            }
            Mutex::Lock<PIPE_MTE1>(vHalf);
            LoadVToL0(info);
            Mutex::Unlock<PIPE_MTE1>(vHalf);
            Mutex::Unlock<PIPE_MTE1>(PV_L0AB_MUTEX_ID_BASE + pvL0abBufId);
            Mutex::Lock<PIPE_M>(PV_L0AB_MUTEX_ID_BASE + pvL0abBufId);
            MatmulPV(info);
            Mutex::Unlock<PIPE_M>(PV_L0AB_MUTEX_ID_BASE + pvL0abBufId);
            pvL0abBufId = (pvL0abBufId + 1) % PV_L0AB_BUFCNT;
        }
        if (unlikely(info.isUpdatePScale)) {
            Mutex::Unlock<PIPE_M>(PV_L0C_MUTEX_ID_BASE);
            Mutex::Lock<PIPE_FIX>(PV_L0C_MUTEX_ID_BASE);
            FixpipeMm2(info);
            Mutex::Unlock<PIPE_FIX>(PV_L0C_MUTEX_ID_BASE);
        }
    }

    __aicore__ inline void CopyGMaxL1ToUb(const CubeRunInfo& runInfo)
    {
        LocalTensor<half> localGlobalMax0 = localGlobalMaxL1[runInfo.tileMaxIdx * 256];
        LocalTensor<half> localGlobalMax1 = localGlobalMaxL1[runInfo.tileMaxIdx * 256 + L1_SINGLE_GLOBAL_MAX_SIZE];
        LocalTensor<half> peerGlobalMax = peerGlobalMaxUB[runInfo.tileMaxIdx * 128];

        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = 8;
        intriParams.srcGap = 0;
        intriParams.dstGap = 0;

        DataCopyL1ToUB<half, VEC0>(peerGlobalMax, localGlobalMax0, intriParams);
        DataCopyL1ToUB<half, VEC1>(peerGlobalMax, localGlobalMax1, intriParams);
    }

private:
    __aicore__ inline void InitGmCopyConsts()
    {
        const uint32_t kvHeadNum = kvHeadNumCached; // N_kv
        {
            uint32_t dByte = headDimInt8; // Q数据D维, DATA_T(uint8_t)元素数
            qNd2NzParams.ndNum = 1;
            qNd2NzParams.dValue = dByte;
            qNd2NzParams.dstNzNStride = 1;
            qNd2NzParams.srcNdMatrixStride = 0;
            qNd2NzParams.dstNzMatrixStride = 0;
            if constexpr (Q_TND) {
                // (T, N_q, D), token 行距 = N_q * dByte (S1G 时 N_q = n2Size*gSize)
                qNd2NzParams.srcDValue = static_cast<uint64_t>(constInfo.n2Size) * constInfo.gSize * dByte;
                qGmStrideS1 = static_cast<uint64_t>(constInfo.n2Size) * constInfo.gSize * dByte;
                qGmStrideN2 = dByte;
            } else {
                qNd2NzParams.srcDValue = dByte;
                qGmStrideS1 = dByte;
                qGmStrideN2 = static_cast<uint64_t>(constInfo.gSize) * constInfo.s1Size * dByte;
                qGmStrideB = static_cast<uint64_t>(constInfo.n2Size) * qGmStrideN2;
            }

            uint32_t dScale = constInfo.dSize / MXFP_GROUP_SIZE; // Q/K scale D维, SCALE_T元素数
            qScaleDn2NzParams.dnNum = 1;
            qScaleDn2NzParams.nValue = dScale / 2;
            qScaleDn2NzParams.srcDnMatrixStride = 0;
            qScaleDn2NzParams.dstNzC0Stride = dScale / 2;
            qScaleDn2NzParams.dstNzNStride = 1;
            qScaleDn2NzParams.dstNzMatrixStride = dScale / 2;
            if constexpr (Q_TND) {
                // q_descale (T, N_q, D/64, 2), token 行距 = N_q * dScale (S1G 时 N_q = n2Size*gSize)
                qScaleDn2NzParams.srcDValue = static_cast<uint64_t>(constInfo.n2Size) * constInfo.gSize * dScale / 2;
                qScaleGmStrideS1 = static_cast<uint64_t>(constInfo.n2Size) * constInfo.gSize * dScale;
                qScaleGmStrideN2 = dScale;
            } else {
                qScaleDn2NzParams.srcDValue = dScale / 2;
                qScaleGmStrideS1 = dScale;
                qScaleGmStrideN2 = static_cast<uint64_t>(constInfo.gSize) * constInfo.s1Size * dScale;
                qScaleGmStrideB = static_cast<uint64_t>(constInfo.n2Size) * qScaleGmStrideN2;
            }

            kScaleDn2NzParams.dnNum = 1;
            kScaleDn2NzParams.nValue = dScale / 2;
            kScaleDn2NzParams.srcDnMatrixStride = 0;
            kScaleDn2NzParams.dstNzC0Stride = dScale / 2;
            kScaleDn2NzParams.dstNzNStride = 1;
            kScaleDn2NzParams.dstNzMatrixStride = dScale / 2;
            if constexpr (KV_TND) {
                // k_descale (T, N_kv, D/64, 2), token 行距 = N_kv * dScale
                kScaleDn2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dScale / 2;
                kScaleGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dScale;
                kScaleGmStrideN2 = dScale;
            } else if constexpr (PAGE_ATTENTION) {
                if (constInfo.paLayoutType == PA_LAYOUT_BNBD) {
                    // k_descale PA_BNBD: (Bn, N, Bs, D/64, 2), token 行距 = dScale, head 段距 = Bs*dScale
                    kScaleDn2NzParams.srcDValue = dScale / 2;
                    kScaleGmStrideS2 = dScale;
                    kScaleGmStrideN2 = static_cast<uint64_t>(constInfo.blockSize) * dScale;
                } else if (constInfo.paLayoutType == PA_LAYOUT_BBND) {
                    // k_descale PA_BBND: (Bn, Bs, N, D/64, 2), token 行距 = N_kv*dScale, head 段距 = dScale
                    kScaleDn2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dScale / 2;
                    kScaleGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dScale;
                    kScaleGmStrideN2 = dScale;
                }
                // PA_NZ 由 CopyKScaleGmToL1 内直拷, 不初始化 Dn2Nz/stride
                kScaleGmStrideB =
                    static_cast<uint64_t>(kvHeadNum) * static_cast<uint64_t>(constInfo.blockSize) * dScale;
            } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
                kScaleDn2NzParams.srcDValue = dScale / 2;
                kScaleGmStrideS2 = dScale;
                kScaleGmStrideN2 = static_cast<uint64_t>(constInfo.s2Size) * dScale;
                kScaleGmStrideB = static_cast<uint64_t>(kvHeadNum) * kScaleGmStrideN2;
            }

            uint32_t dScaleV = 2 * constInfo.dSize; // V scale D维(2*MXFP_GROUP_SIZE分形)
            vScaleNd2NzParams.ndNum = 1;
            vScaleNd2NzParams.dValue = dScaleV / 2;
            // MTE1 块主序交织读
            vScaleNd2NzParams.dstNzC0Stride = dScaleV / 32;
            vScaleNd2NzParams.dstNzNStride = 1;
            vScaleNd2NzParams.srcNdMatrixStride = 0;
            if constexpr (KV_TND) {
                // v_descale (Tg, N_kv, D, 2), group 行距 = N_kv * dScaleV, Tg = Σceil(len_i/64)
                vScaleNd2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dScaleV / 2;
                vScaleGmStrideN2 = dScaleV;
                vScaleGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dScaleV;
            } else if constexpr (PAGE_ATTENTION) {
                uint32_t blockGroups = (constInfo.blockSize + 63) / 64;
                if (constInfo.paLayoutType == PA_LAYOUT_BNBD) {
                    // v_descale PA_BNBD: (Bn, N, ceil(Bs/64), D, 2), 组行距 = dScaleV, head 段距 = 组数*dScaleV
                    vScaleNd2NzParams.srcDValue = dScaleV / 2;
                    vScaleGmStrideS2 = dScaleV;
                    vScaleGmStrideN2 = static_cast<uint64_t>(blockGroups) * dScaleV;
                } else if (constInfo.paLayoutType == PA_LAYOUT_BBND) {
                    // v_descale PA_BBND: (Bn, ceil(Bs/64), N, D, 2), 组行距 = N_kv*dScaleV, head 段距 = dScaleV
                    vScaleNd2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dScaleV / 2;
                    vScaleGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dScaleV;
                    vScaleGmStrideN2 = dScaleV;
                }
                // PA_NZ 由 CopyVScaleGmToL1 内直拷, 不初始化 Nd2Nz/stride
                vScaleGmStrideB = static_cast<uint64_t>(kvHeadNum) * static_cast<uint64_t>(blockGroups) * dScaleV;
            } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
                vScaleNd2NzParams.srcDValue = dScaleV / 2;
                vScaleGmStrideN2 = static_cast<uint64_t>((constInfo.maxSeqlenKv + 63) / 64) * dScaleV;
                vScaleGmStrideB = static_cast<uint64_t>(kvHeadNum) * vScaleGmStrideN2;
                vScaleGmStrideS2 = dScaleV;
            }
        }
        {
            uint32_t dByte = headDimInt8;
            kvNd2NzParams.ndNum = 1;
            kvNd2NzParams.dValue = dByte;
            kvNd2NzParams.dstNzC0Stride = s2BaseSize * 2;
            kvNd2NzParams.dstNzNStride = 1;
            kvNd2NzParams.srcNdMatrixStride = 0;
            kvNd2NzParams.dstNzMatrixStride = 0;
            if constexpr (KV_TND) {
                // (T, N_kv, D), token 行距 = N_kv * dByte
                kvNd2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dByte;
                kvGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dByte;
                kvGmStrideN2 = dByte;
            } else if constexpr (PAGE_ATTENTION) {
                if (constInfo.paLayoutType == PA_LAYOUT_BNBD) {
                    // PA_BNBD: (Bn, N, Bs, D), 块内 token 行距 = dByte, head 段距 = Bs*dByte
                    kvNd2NzParams.srcDValue = dByte;
                    kvGmStrideS2 = dByte;
                    kvGmStrideN2 = static_cast<uint64_t>(constInfo.blockSize) * dByte;
                } else if (constInfo.paLayoutType == PA_LAYOUT_BBND) {
                    // PA_BBND: (Bn, Bs, N, D), 块内 token 行距 = N_kv*dByte, head 段距 = dByte
                    kvNd2NzParams.srcDValue = static_cast<uint64_t>(kvHeadNum) * dByte;
                    kvGmStrideS2 = static_cast<uint64_t>(kvHeadNum) * dByte;
                    kvGmStrideN2 = dByte;
                } else if (constInfo.paLayoutType == PA_LAYOUT_NZ) {
                    // PA_NZ: (Bn, N, D/32, Bs, 16B), 块内 token 行距 = 16B, d1 平面距 = Bs*16B, head 段距 = Bs*64B
                    kvNd2NzParams.srcDValue = 16;
                    kvGmStrideS2 = 16;
                    kvGmStrideN2 = static_cast<uint64_t>(constInfo.blockSize) * 64;
                }
                kvGmStrideB = static_cast<uint64_t>(kvHeadNum) * static_cast<uint64_t>(constInfo.blockSize) * dByte;
            } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
                kvNd2NzParams.srcDValue = dByte;
                kvGmStrideS2 = dByte;
                kvGmStrideN2 = static_cast<uint64_t>(constInfo.s2Size) * dByte;
                kvGmStrideB = static_cast<uint64_t>(kvHeadNum) * kvGmStrideN2;
            }
        }
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        Mutex::Lock<PIPE_MTE2>(VSUM_MUTEX_ID_BASE);
        InitConstValueParams<uint16_t> vL1InitParams(1, static_cast<uint16_t>(PV_L0A_SIZE / 32), 0, 0x6666);
        Fill(vSumL1Tensor.template ReinterpretCast<uint16_t>(), vL1InitParams);

        Mutex::Unlock<PIPE_MTE2>(VSUM_MUTEX_ID_BASE);
        Mutex::Lock<PIPE_MTE2>(VSUM_MUTEX_ID_BASE);

        InitConstValueParams<uint16_t> vScaleL1InitParams(1, static_cast<uint16_t>(V_SCALE_L0A_SIZE / 32), 0, 0x7d7d);
        Fill(vScaleSumL1Tensor.template ReinterpretCast<uint16_t>(), vScaleL1InitParams);
        Mutex::Unlock<PIPE_MTE2>(VSUM_MUTEX_ID_BASE);
    }

    __aicore__ inline void InitL0MXBufferForReduceSum(bool flushAllSlots = true)
    {
        Mutex::Lock<PIPE_MTE1>(VSUM_MUTEX_ID_BASE);

        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = (128 + 16) / 16;
        loadData2DParamsA.kStep = 256 / 64;
        loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
        loadData2DParamsA.ifTranspose = false;

        LoadData2DMxParams loadData2DMXParamsA;
        loadData2DMXParamsA.xStartPosition = 0;
        loadData2DMXParamsA.yStartPosition = 0;
        loadData2DMXParamsA.xStep = (128 + 16) / 16;
        loadData2DMXParamsA.yStep = 256 / 64;
        loadData2DMXParamsA.srcStride = loadData2DMXParamsA.yStep;
        loadData2DMXParamsA.dstStride = loadData2DMXParamsA.yStep;

        uint32_t slotBegin = flushAllSlots ? 0 : pvL0abBufId;
        uint32_t slotEnd = flushAllSlots ? PV_L0AB_BUFCNT : pvL0abBufId + 1;
        for (uint32_t slot = slotBegin; slot < slotEnd; slot++) {
            Mutex::Lock<PIPE_MTE1>(PV_L0AB_MUTEX_ID_BASE + slot);
            uint32_t pvL0AOffset = slot * (PV_L0A_SIZE / sizeof(DATA_T));
            LoadData(pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(), vSumL1Tensor.ReinterpretCast<QUANT_T>(),
                     vScaleSumL1Tensor, loadData2DParamsA, loadData2DMXParamsA);
            Mutex::Unlock<PIPE_MTE1>(PV_L0AB_MUTEX_ID_BASE + slot);
        }
        Mutex::Unlock<PIPE_MTE1>(VSUM_MUTEX_ID_BASE);
    }

    __aicore__ inline void CopyQGmToL1(const CubeRunInfo& info)
    {
        if constexpr (Q_TND) {
            if (unlikely(constInfo.gSize > 1)) {
                CopyQGmToL1S1G(info);
                return;
            }
        }
        qNd2NzParams.nValue = info.actMSize;
        qNd2NzParams.dstNzC0Stride = info.actMSizeAlign128;
        uint64_t gmOffset;
        if constexpr (Q_TND) {
            gmOffset = (qSeqLensTool.cuSeqLensParser.GetTBase(info.bIdx) + info.gS1Idx) * qGmStrideS1 +
                       info.n2Idx * qGmStrideN2;
        } else {
            gmOffset = info.bIdx * qGmStrideB + info.n2Idx * qGmStrideN2 + info.gS1Idx * qGmStrideS1;
        }
        DataCopy(qL1Tensor, queryGm[gmOffset], qNd2NzParams);
    }

    // S1G 合轴: 行 r=(t,g) t 主序, 目标 L1 布局与摊平 Nd2Nz 一致 (C0 半平面距 actMSizeAlign128*32B)。
    // N_kv=1 (MQA): (t,g) 行在 GM 连续, 单次 Nd2Nz 直拷; GQA: 对齐 tile 逐 (半平面, g) NDDMA,
    // 非对齐 tile (r0%G!=0 或尾组) 拆 首残组/整组/尾残组 三段 NDDMA (段内 GM 行连续)。
    __aicore__ inline void CopyQGmToL1S1G(const CubeRunInfo& info)
    {
        uint32_t gSize = constInfo.gSize;
        uint32_t qHeadNum = constInfo.n2Size * gSize;
        uint64_t tBase = qSeqLensTool.cuSeqLensParser.GetTBase(info.bIdx);
        qNd2NzParams.nValue = info.actMSize;
        qNd2NzParams.dstNzC0Stride = info.actMSizeAlign128;
        if (constInfo.n2Size == 1) {
            // (t,g) 行 GM 连续: 行距/基址均按 dByte (非 flat 的 qHeadNum 步长)
            qNd2NzParams.srcDValue = headDimInt8;
            uint64_t gmOffset = (tBase * gSize + info.gS1Idx) * headDimInt8;
            DataCopy(qL1Tensor, queryGm[gmOffset], qNd2NzParams);
            return;
        }
        uint32_t r0 = info.gS1Idx;
        uint32_t m = info.actMSize;
        uint32_t t0 = r0 / gSize;
        uint32_t g0 = r0 % gSize;
        uint64_t colBase = info.n2Idx * gSize * 64ULL;
        if (g0 == 0 && m % gSize == 0) { // 对齐 tile: 行 r = t*G + g (t-major, 与 qscale/out/mask 一致)
            uint32_t tCnt = m / gSize;
            DataCopyParams nddma;
            nddma.blockLen = 1; // 32B 半行
            if (gSize == 1) {   // G=1: 行 GM 等距, 单次 NDDMA
                nddma.blockCount = tCnt;
                nddma.srcStride = qHeadNum * 2 - 1; // t 间跳 qHeadNum*64B (32B 块)
                nddma.dstStride = 0;
                uint64_t headBase = (tBase + t0) * qHeadNum * 64ULL + colBase;
                for (uint32_t p = 0; p < 2; p++) { // 0=行低 32B (C0), 1=行高 32B
                    DataCopy(qL1Tensor[p * info.actMSizeAlign128 * 32], queryGm[headBase + p * 32], nddma);
                }
            } else {
                // G>1: NDDMA dstStride>0 不支持, 按 token 逐段 (段内 g 行 GM 连续)
                nddma.blockCount = gSize;
                nddma.srcStride = 1; // token 内相邻 g 行距 64B
                nddma.dstStride = 0;
                for (uint32_t t = 0; t < tCnt; t++) {
                    uint64_t src = (tBase + t0 + t) * qHeadNum * 64ULL + colBase;
                    for (uint32_t p = 0; p < 2; p++) {
                        DataCopy(qL1Tensor[p * info.actMSizeAlign128 * 32 + t * gSize * 32], queryGm[src + p * 32],
                                 nddma);
                    }
                }
            }
            return;
        }
        // 非对齐 tile: 段内 (同 token) GM 行连续, 逐段 NDDMA; dst 行号 = 全局行 - r0
        uint32_t r1 = r0 + m - 1;
        uint32_t t1 = r1 / gSize;
        uint32_t g1 = r1 % gSize;
        DataCopyParams seg;
        seg.blockLen = 1;
        seg.srcStride = 1; // token 内相邻 g 行距 64B
        seg.dstStride = 0;
        uint32_t rowsA = (t0 == t1) ? m : (gSize - g0);
        seg.blockCount = rowsA;
        uint64_t srcA = (tBase + t0) * qHeadNum * 64ULL + colBase + g0 * 64ULL;
        for (uint32_t p = 0; p < 2; p++) {
            DataCopy(qL1Tensor[p * info.actMSizeAlign128 * 32], queryGm[srcA + p * 32], seg);
        }
        if (t0 < t1) {
            uint32_t done = rowsA;
            uint32_t tCntB = t1 - t0 - 1;
            if (tCntB > 0) {
                seg.blockCount = tCntB;
                seg.srcStride = qHeadNum * 2 - 1;
                seg.dstStride = gSize - 1; // 同 g 的相邻 token 在 L1 间隔 gSize 行 (t-major)
                uint64_t srcB = (tBase + t0 + 1) * qHeadNum * 64ULL + colBase;
                for (uint32_t g = 0; g < gSize; g++) {
                    for (uint32_t p = 0; p < 2; p++) {
                        DataCopy(qL1Tensor[p * info.actMSizeAlign128 * 32 + (done + g) * 32],
                                 queryGm[srcB + g * 64ULL + p * 32], seg);
                    }
                }
                done += tCntB * gSize;
            }
            seg.blockCount = m - done; // 尾残组: token t1 的 0..g1
            seg.srcStride = 1;
            seg.dstStride = 0;
            uint64_t srcC = (tBase + t1) * qHeadNum * 64ULL + colBase;
            for (uint32_t p = 0; p < 2; p++) {
                DataCopy(qL1Tensor[p * info.actMSizeAlign128 * 32 + done * 32], queryGm[srcC + p * 32], seg);
            }
        }
    }

    // PA: 查 blockTable 得物理块号, 拼块内偏移 (head 段 + token 段)
    __aicore__ inline uint64_t GetKvBlockBase(const CubeRunInfo& info, uint32_t s2Idx, uint64_t blockBytes,
                                              uint64_t strideN2, uint64_t strideS2)
    {
        uint32_t blockIdxInBatch = s2Idx >> paBlockShift;
        uint32_t bsIdx = s2Idx & paBlockMask;
        uint64_t blockIdx =
            static_cast<uint64_t>(blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + blockIdxInBatch));
        return blockIdx * blockBytes + info.kvHeadIdx * strideN2 + bsIdx * strideS2;
    }

    // PA_NZ: 物理块基址 (块字节容量 = kvHeadNum * Bs * 64B)
    __aicore__ inline uint64_t GetNzBlockBase(const CubeRunInfo& info, uint32_t s2Idx, uint64_t headBytes,
                                              uint64_t kvHeadNum)
    {
        uint32_t blockIdxInBatch = s2Idx >> paBlockShift;
        uint64_t blockIdx =
            static_cast<uint64_t>(blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + blockIdxInBatch));
        return blockIdx * headBytes * kvHeadNum + info.kvHeadIdx * headBytes + (s2Idx & paBlockMask) * 32;
    }

    // PA_NZ K/V 数据: 块内 (D/64=2, Bs, 64 逻辑元素=32B), d0=32B 整块。
    // 分块 NDDMA: blockCount=2 (d1 段), blockLen=copyRowCnt 块(32B), srcStride=Bs-copyRowCnt (段间跳 d1 平面),
    // dstStride=rowCount-copyRowCnt → L1 = token 行主序 64B(2C0), 行内 C0 按 d1 排列。
    __aicore__ inline void CopyKvNzGmToL1(const CubeRunInfo& info, const GlobalTensor<DATA_T>& gmTensor,
                                          uint64_t l1BaseOffset)
    {
        uint32_t kvHeadNum = kvHeadNumCached; // N_kv
        uint64_t headBytes = static_cast<uint64_t>(constInfo.blockSize) * 64;
        uint32_t copied = 0;
        uint32_t curS2 = info.s2Idx;
        while (copied < info.pairKVCopyS2Size) {
            uint32_t rows = constInfo.blockSize - (curS2 & paBlockMask);
            if (copied + rows > info.pairKVCopyS2Size) {
                rows = info.pairKVCopyS2Size - copied;
            }
            uint64_t base = GetNzBlockBase(info, curS2, headBytes, kvHeadNum);
            if (likely(rows == static_cast<uint32_t>(s2BaseSize))) {
                DataCopy(kvL1Tensor[l1BaseOffset + copied * 32], gmTensor[base], kvNzFullTileParams);
            } else {
                DataCopyParams intriParams;
                intriParams.blockCount = 2;                         // d1 段数 (D/64)
                intriParams.blockLen = rows;                        // 每段 rows 个 32B 块 (每 token 一块)
                intriParams.dstStride = s2BaseSize * 2 - rows;      // L1 半平面距 (块)
                intriParams.srcStride = constInfo.blockSize - rows; // GM 段间跳 d1 平面
                DataCopy(kvL1Tensor[l1BaseOffset + copied * 32], gmTensor[base], intriParams);
            }
            copied += rows;
            curS2 += rows;
        }
    }

    // K/V 共用: PA 时按 block 边界分块查表, 非 PA 时按 TND/BNSD 整块 ND2NZ
    __aicore__ inline void CopyKvGmToL1(const CubeRunInfo& info, const GlobalTensor<DATA_T>& kvGm)
    {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        kvNd2NzParams.nValue = info.pairKVCopyS2Size;
        if constexpr (PAGE_ATTENTION) {
            if (constInfo.paLayoutType == PA_LAYOUT_NZ) {
                CopyKvNzGmToL1(info, kvGm, l1BaseOffset);
                return;
            }
            // 按 block 边界分块 ND2NZ, 每块查一次表; L1 行偏移 = copied*32 (每 token 行 64B=2*C0)
            uint32_t copied = 0;
            uint32_t curS2 = info.s2Idx;
            while (copied < info.pairKVCopyS2Size) {
                uint32_t rows = constInfo.blockSize - (curS2 & paBlockMask);
                if (copied + rows > info.pairKVCopyS2Size) {
                    rows = info.pairKVCopyS2Size - copied;
                }
                Nd2NzParams chunkParams = kvNd2NzParams; // 局部参数, 防跨异步指令重排
                chunkParams.nValue = rows;
                uint64_t gmOffset = GetKvBlockBase(info, curS2, kvGmStrideB, kvGmStrideN2, kvGmStrideS2);
                DataCopy(kvL1Tensor[l1BaseOffset + copied * 32], kvGm[gmOffset], chunkParams);
                copied += rows;
                curS2 += rows;
            }
            return;
        }
        uint64_t gmOffset;
        if constexpr (KV_TND) {
            gmOffset = (kvSeqLensTool.cuSeqLensParser.GetTBase(info.bIdx) + info.s2Idx) * kvGmStrideS2 +
                       info.kvHeadIdx * kvGmStrideN2;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            gmOffset = info.bIdx * kvGmStrideB + info.kvHeadIdx * kvGmStrideN2 + info.s2Idx * kvGmStrideS2;
        }
        DataCopy(kvL1Tensor[l1BaseOffset], kvGm[gmOffset], kvNd2NzParams);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQScaleGmToL1(const CubeRunInfo& info)
    {
        if constexpr (Q_TND) { // TND 统一 N2TGD (G=1 含内): fixed n2 的 (t,g) 行 GM 连续, 单次 Dn2Nz
            CopyQScaleGmToL1S1G(info);
            return;
        }
        qScaleDn2NzParams.dValue = info.actMSize;
        uint64_t gmOffset =
            info.bIdx * qScaleGmStrideB + info.n2Idx * qScaleGmStrideN2 + info.gS1Idx * qScaleGmStrideS1;
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(queryScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(qDescaleL1Tensor.template ReinterpretCast<bfloat16_t>(), gmTensorCast, qScaleDn2NzParams);
    }

    // S1G qscale: layout_q_descale=N2TGD (N2,T,G,4B) — fixed n2 的 (t,g) 行 GM 连续 (4B/行)。
    // L1 布局: nValue=2, dstNzC0Stride=dScaleHalf, dstNzNStride=1
    __aicore__ inline void CopyQScaleGmToL1S1G(const CubeRunInfo& info)
    {
        uint32_t dScaleHalf = constInfo.dSize / MXFP_GROUP_SIZE / 2; // bf16 视图列单位 (2B)
        qScaleDn2NzParams.dnNum = 1;
        qScaleDn2NzParams.nValue = dScaleHalf;
        qScaleDn2NzParams.dValue = info.actMSize;
        qScaleDn2NzParams.srcDValue = dScaleHalf;
        qScaleDn2NzParams.srcDnMatrixStride = 0;
        qScaleDn2NzParams.dstNzC0Stride = dScaleHalf;
        qScaleDn2NzParams.dstNzNStride = 1;
        qScaleDn2NzParams.dstNzMatrixStride = 0;
        uint64_t gmOffset = (static_cast<uint64_t>(constInfo.t1Size) * constInfo.gSize * info.n2Idx +
                             qSeqLensTool.cuSeqLensParser.GetTBase(info.bIdx) * constInfo.gSize + info.gS1Idx) *
                            qScaleGmStrideN2;
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(queryScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(qDescaleL1Tensor.template ReinterpretCast<bfloat16_t>(), gmTensorCast, qScaleDn2NzParams);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKScaleGmToL1(const CubeRunInfo& info)
    {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        kScaleDn2NzParams.dValue = info.pairKVCopyS2Size;
        if constexpr (PAGE_ATTENTION) {
            if (constInfo.paLayoutType == PA_LAYOUT_NZ) {
                // PA_NZ k_descale: (Bn, N, Bs/16, 2, 16, 2); 源 cell (t16, d3) 32B 连续, t16 主序与
                // 消费端 MTE1 MX (xStep=tokens/16, srcStride=2 块) 一致 → 每 chunk 连续 slab 直拷
                uint32_t copied = 0;
                uint32_t curS2 = info.s2Idx;
                uint32_t kvHeadNum = kvHeadNumCached; // N_kv
                uint64_t headBytes = static_cast<uint64_t>(constInfo.blockSize) * 4;
                while (copied < info.pairKVCopyS2Size) {
                    uint32_t rows = constInfo.blockSize - (curS2 & paBlockMask);
                    if (copied + rows > info.pairKVCopyS2Size) {
                        rows = info.pairKVCopyS2Size - copied;
                    }
                    uint32_t blockIdxInBatch = curS2 >> paBlockShift;
                    uint64_t blockIdx = static_cast<uint64_t>(
                        blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + blockIdxInBatch));
                    uint64_t src = blockIdx * headBytes * kvHeadNum + info.kvHeadIdx * headBytes +
                                   ((curS2 & paBlockMask) >> 4) * 64;
                    // 消费端以 16-token cell 为单位读 k_descale: 末块不足 16 行也要整 cell 拷入,
                    // 否则尾部 token 读到上一轮残留 scale.
                    uint32_t cellRows = ((rows + 15U) / 16U) * 16U;
                    uint32_t curInBlock = curS2 & paBlockMask;
                    uint32_t availInBlock = constInfo.blockSize - (curInBlock & ~15U);
                    if (cellRows > availInBlock) {
                        cellRows = availInBlock;
                    }
                    DataCopy(kvDescaleL1Tensor[offset + copied / 16 * 64], keyScaleGm[src], cellRows / 16 * 64);
                    copied += rows;
                    curS2 += rows;
                }
                return;
            }
            uint32_t copied = 0;
            uint32_t curS2 = info.s2Idx;
            while (copied < info.pairKVCopyS2Size) {
                uint32_t rows = constInfo.blockSize - (curS2 & paBlockMask);
                if (copied + rows > info.pairKVCopyS2Size) {
                    rows = info.pairKVCopyS2Size - copied;
                }
                Dn2NzParams chunkParams = kScaleDn2NzParams; // 局部参数, 防跨异步指令重排
                chunkParams.dValue = rows;
                uint64_t gmOffset = GetKvBlockBase(info, curS2, kScaleGmStrideB, kScaleGmStrideN2, kScaleGmStrideS2);
                GlobalTensor<bfloat16_t> gmTensorCast;
                gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(keyScaleGm[gmOffset].GetPhyAddr()));
                DataCopy(kvDescaleL1Tensor[offset + copied * 4].template ReinterpretCast<bfloat16_t>(), gmTensorCast,
                         chunkParams);
                copied += rows;
                curS2 += rows;
            }
            return;
        }
        uint64_t gmOffset = 0; // 未支持组合兜底
        if constexpr (KV_TND) {
            gmOffset = (kvSeqLensTool.cuSeqLensParser.GetTBase(info.bIdx) + info.s2Idx) * kScaleGmStrideS2 +
                       info.kvHeadIdx * kScaleGmStrideN2;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            gmOffset = info.bIdx * kScaleGmStrideB + info.kvHeadIdx * kScaleGmStrideN2 + info.s2Idx * kScaleGmStrideS2;
        }
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(keyScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(kvDescaleL1Tensor[offset].template ReinterpretCast<bfloat16_t>(), gmTensorCast, kScaleDn2NzParams);
    }

    __aicore__ inline static uint32_t NdAlignM(uint32_t actM)
    {
        return AttentionCommon::Align(actM, 16U);
    }

    __aicore__ inline void LoadQToL0(const CubeRunInfo& info)
    {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = info.actMSizeAlign128 / 16;
        loadData2DParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();
        loadData2DParamsA.srcStride = loadData2DParamsA.mStep;
        loadData2DParamsA.dstStride = loadData2DParamsA.mStep;
        loadData2DParamsA.ifTranspose = false;

        LoadData2DMxParams loadData2DMxParamsA;
        loadData2DMxParamsA.xStartPosition = 0;
        loadData2DMxParamsA.yStartPosition = 0;
        loadData2DMxParamsA.xStep = info.actMSizeAlign128 / 16;
        loadData2DMxParamsA.yStep = loadData2DParamsA.kStep;
        loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
        loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;

        uint32_t qkL0Offset = qkL0abBufId * (QK_L0B_SIZE / sizeof(DATA_T));
        if constexpr (ND) {
            // ND: Q 作 A 操作数 (m=M, k=D); L1 平面行距固定 actMSizeAlign128
            uint32_t ndM = NdAlignM(info.actMSize);
            loadData2DParamsA.mStep = ndM / 16;
            loadData2DParamsA.srcStride = info.actMSizeAlign128 / 16;
            loadData2DParamsA.dstStride = ndM / 16;
            loadData2DMxParamsA.xStep = ndM / 16;
            LoadData(qkL0ATensor[qkL0Offset].ReinterpretCast<QUANT_T>(), qL1Tensor.ReinterpretCast<QUANT_T>(),
                     qDescaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);
            return;
        }
        LoadData(qkL0BTensor[qkL0Offset].ReinterpretCast<QUANT_T>(), qL1Tensor.ReinterpretCast<QUANT_T>(),
                 qDescaleL1Tensor, loadData2DParamsA, loadData2DMxParamsA);
    }

    __aicore__ inline void LoadKToL0(const CubeRunInfo& info, uint32_t subLoop, uint32_t actS2SizeAlign)
    {
        uint32_t mStart = (info.prefetched ? (s2BaseSize / 16) : 0) + subLoop * (QK_L0_S2_SPLIT_SIZE / 16);
        uint32_t mStep = actS2SizeAlign / 16;
        loadKParamsA.mStartPosition = mStart;
        loadKParamsA.mStep = mStep;
        loadKParamsA.dstStride = mStep;
        loadKParamsMx.xStartPosition = mStart;
        loadKParamsMx.xStep = mStep;

        uint32_t qkL0Offset = qkL0abBufId * (QK_L0A_SIZE / sizeof(DATA_T));
        uint32_t kL1Offset = kDataBufId * (L1_KV_SIZE / sizeof(DATA_T));

        uint32_t kScaleL1Offset = kDataBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        if constexpr (ND) {
            // ND: K 作 B 操作数 (k=D, n=s2), L1 分形与 DN 的 A 操作数一致
            LoadData(qkL0BTensor[qkL0Offset].ReinterpretCast<QUANT_T>(),
                     kvL1Tensor[kL1Offset].ReinterpretCast<QUANT_T>(), kvDescaleL1Tensor[kScaleL1Offset], loadKParamsA,
                     loadKParamsMx);
            return;
        }
        LoadData(qkL0ATensor[qkL0Offset].ReinterpretCast<QUANT_T>(), kvL1Tensor[kL1Offset].ReinterpretCast<QUANT_T>(),
                 kvDescaleL1Tensor[kScaleL1Offset], loadKParamsA, loadKParamsMx);
    }

    __aicore__ inline void MatmulQK(const CubeRunInfo& info, uint32_t actS2Size)
    {
        if constexpr (ND) {
            mmadQKParams.m = NdAlignM(info.actMSize);
            mmadQKParams.n = actS2Size;
        } else {
            mmadQKParams.m = actS2Size;
            mmadQKParams.n = info.actMSizeAlign128;
        }
        uint32_t qkL0AOffset = qkL0abBufId * (QK_L0A_SIZE / sizeof(DATA_T));
        uint32_t qkL0BOffset = qkL0abBufId * (QK_L0B_SIZE / sizeof(DATA_T));
        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Mmad(qkL0CTensor[qkL0COffset], qkL0ATensor[qkL0AOffset].ReinterpretCast<QUANT_T>(),
             qkL0BTensor[qkL0BOffset].ReinterpretCast<QUANT_T>(), mmadQKParams);
    }

    __aicore__ inline void FixpipeMm1(const CubeRunInfo& info, uint32_t subLoop, uint32_t actS2Size,
                                      uint32_t actS2SizeAlign)
    {
        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        if constexpr (ND) {
            // ND: L0C 为 [m=M][n=s2], 行主序写 [actM 行][s2] 到 UB, 块 = 32KB (行距 128 half)
            fixpipeMm1Params.nSize = static_cast<uint16_t>(actS2SizeAlign);
            fixpipeMm1Params.mSize = static_cast<uint16_t>(info.actMSize);
            fixpipeMm1Params.srcStride = static_cast<uint16_t>(NdAlignM(info.actMSize));
            fixpipeMm1Params.dstStride = QK_L0_S2_SPLIT_SIZE;
            fixpipeMm1Params.subBlockId = info.loop % 2;
            fixpipeMm1Params.params.srcNdStride = static_cast<uint16_t>((actS2Size + 1) >> 1 << 1);
            fixpipeMm1Params.params.dstNdStride = QK_L0_S2_SPLIT_SIZE;
            uint32_t ubOffset = ((info.loop / 2 % 2) * 2 + subLoop) * QK_L0_S2_SPLIT_SIZE * ND_M_ROWS;
            Fixpipe<half, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm1ResUB[ubOffset], qkL0CTensor[qkL0COffset], fixpipeMm1Params);
            return;
        }
        fixpipeMm1Params.nSize = (info.actMSizeAlign128 + 7) >> 3 << 3;
        fixpipeMm1Params.mSize = (actS2Size + 1) >> 1 << 1;
        fixpipeMm1Params.srcStride = ((actS2Size + 15) / 16) * 16;
        fixpipeMm1Params.subBlockId = info.loop % 2;
        fixpipeMm1Params.params.srcNdStride = (actS2Size + 1) >> 1 << 1;

        uint32_t ubOffset = subLoop * QK_L0_S2_SPLIT_SIZE * 256;
        Fixpipe<half, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm1ResUB[info.loop / 2 % 2 * 128 + ubOffset],
                                                   qkL0CTensor[qkL0COffset], fixpipeMm1Params);
    }

    __aicore__ inline void CopyVScaleGmToL1(const CubeRunInfo& info)
    {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        vScaleNd2NzParams.nValue = (info.pairKVCopyS2Size + 63) / 64;
        if constexpr (PAGE_ATTENTION) {
            if (constInfo.paLayoutType == PA_LAYOUT_NZ) {
                // PA_NZ v_descale: (Bn, N, D/16, Bs/64, 16, 2); 源 (d3, tg) 均为 32B 连续块,
                // L1 目标为 (tg + 8*d3) 个 C0 → 一次 blockCount=8 的搬运覆盖全部 d3 平面
                uint32_t groupCnt = (info.pairKVCopyS2Size + 63) / 64;
                uint32_t copied = 0;
                uint32_t curS2 = info.s2Idx;
                uint64_t headBytes = static_cast<uint64_t>(constInfo.blockSize) * 4;
                uint32_t tgPerBlock = 1U << (paBlockShift - 6); // blockSize / 64
                while (copied < groupCnt) {
                    uint32_t gRows = tgPerBlock - ((curS2 >> 6) & (tgPerBlock - 1));
                    if (copied + gRows > groupCnt) {
                        gRows = groupCnt - copied;
                    }
                    uint32_t blockIdxInBatch = curS2 >> paBlockShift;
                    uint64_t blockIdx = static_cast<uint64_t>(
                        blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + blockIdxInBatch));
                    uint64_t blockBase = blockIdx * headBytes * kvHeadNumCached + info.kvHeadIdx * headBytes;
                    uint32_t bsIdx = (curS2 & paBlockMask) >> 6;
                    DataCopyParams vScaleParams;
                    vScaleParams.blockCount = 8;                 // d3 = D/16 平面数, 每平面 tg 组 32B
                    vScaleParams.blockLen = gRows;               // 每平面本 chunk 的组数
                    vScaleParams.srcStride = tgPerBlock - gRows; // 源平面间距 (组)
                    vScaleParams.dstStride = 8 - gRows;          // L1 相邻 d3 平面间隔 8 个 C0
                    DataCopy(kvDescaleL1Tensor[offset + copied * 32], valueScaleGm[blockBase + bsIdx * 32],
                             vScaleParams);
                    copied += gRows;
                    curS2 += gRows * 64;
                }
                return;
            }
            // v_descale 行 = 64-token 组(组不跨块), 每组占 L1 2D 字节
            uint32_t groupCnt = (info.pairKVCopyS2Size + 63) / 64;
            uint32_t copied = 0;
            uint32_t curS2 = info.s2Idx;
            while (copied < groupCnt) {
                uint32_t groupsPerBlock = 1U << (paBlockShift - 6); // blockSize / 64
                uint32_t gRows = groupsPerBlock - ((curS2 >> 6) & (groupsPerBlock - 1));
                if (copied + gRows > groupCnt) {
                    gRows = groupCnt - copied;
                }
                Nd2NzParams chunkParams = vScaleNd2NzParams; // 局部参数, 防跨异步指令重排
                chunkParams.nValue = gRows;
                uint32_t blockIdxInBatch = curS2 >> paBlockShift;
                uint32_t bsIdx = (curS2 & paBlockMask) >> 6;
                uint64_t blockIdx = static_cast<uint64_t>(
                    blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + blockIdxInBatch));
                uint64_t gmOffset =
                    blockIdx * vScaleGmStrideB + info.kvHeadIdx * vScaleGmStrideN2 + bsIdx * vScaleGmStrideS2;
                GlobalTensor<bfloat16_t> gmTensorCast;
                gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(valueScaleGm[gmOffset].GetPhyAddr()));
                DataCopy(kvDescaleL1Tensor[offset + copied * 32].template ReinterpretCast<bfloat16_t>(), gmTensorCast,
                         chunkParams);
                copied += gRows;
                curS2 += gRows * 64;
            }
            return;
        }
        uint64_t gmOffset = 0; // 未支持组合兜底
        if constexpr (KV_TND) {
            // v_descale 按各自长度拼接: 起点 = Σ_{i<b} ceil(len_i/64),
            // len_i 取 cu 差分(物理打包长度), 与 descale 分配口径一致.
            uint64_t vScaleTBase = 0;
            for (uint32_t i = 0; i < info.bIdx; i++) {
                // 物理打包长度(cu 差分), 与 v_descale 分配口径一致
                uint64_t lenI =
                    kvSeqLensTool.cuSeqLensParser.GetTBase(i + 1) - kvSeqLensTool.cuSeqLensParser.GetTBase(i);
                vScaleTBase += (lenI + 63) >> 6;
            }
            gmOffset = (vScaleTBase + info.s2Idx / 64) * vScaleGmStrideS2 + info.kvHeadIdx * vScaleGmStrideN2;
        } else if constexpr (LAYOUT_KV == QFA_LAYOUT::BNSD) {
            gmOffset =
                info.bIdx * vScaleGmStrideB + info.kvHeadIdx * vScaleGmStrideN2 + (info.s2Idx / 64) * vScaleGmStrideS2;
        }
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(valueScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(kvDescaleL1Tensor[offset].template ReinterpretCast<bfloat16_t>(), gmTensorCast, vScaleNd2NzParams);
    }

    __aicore__ inline void LoadPToL0(const CubeRunInfo& info)
    {
        LoadData2DParamsV2 loadData2DParamsB;
        loadData2DParamsB.mStartPosition = 0;
        loadData2DParamsB.kStartPosition = 0;
        loadData2DParamsB.mStep = (info.actSingleLoopS2SizeAlign64 + 15) / 16;
        loadData2DParamsB.kStep = info.actMSizeAlign128 / GetBlockElemCnt<QUANT_T>();

        loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
        loadData2DParamsB.dstStride =
            AttentionCommon::Align((uint32_t)info.actMSizeAlign128, (uint32_t)GetBlockElemCnt<QUANT_T>()) / 16;
        loadData2DParamsB.ifTranspose = true;

        LoadData2DMxParams load2DMxParamsB;
        load2DMxParamsB.xStartPosition = 0;
        load2DMxParamsB.yStartPosition = 0;
        load2DMxParamsB.xStep = info.actMSizeAlign128 / 16;
        load2DMxParamsB.yStep = (info.actSingleLoopS2SizeAlign64 + 63) / 64;
        load2DMxParamsB.srcStride = 5;
        load2DMxParamsB.dstStride = load2DMxParamsB.yStep;

        if constexpr (ND) {
            // ND: P 在 L1 为 [s2/64 平面][M 行][32B], 与 Q 同构直接作 B 操作数 (k=s2, n=M)
            // 源: P 槽只装真实 M 行(1 个 16 行步), 平面间隔 512B;
            // 目的: L0B 仍按 128 列分形排布, dstStride 必须保持 128/16=8, 不能跟着源侧一起缩
            loadData2DParamsB.mStep = (info.actMSize + 15U) / 16U;
            loadData2DParamsB.kStep = (info.actSingleLoopS2SizeAlign64 + 63) / 64;
            loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
            loadData2DParamsB.dstStride = info.actMSizeAlign128 / 16;
            loadData2DParamsB.ifTranspose = false;
        }

        uint32_t pvL0BOffset = pvL0abBufId * (PV_L0B_SIZE / sizeof(DATA_T));
        uint32_t pL1Offset = (info.loop % 20) * (L1_P_SIZE / sizeof(DATA_T));
        uint32_t pScaleL1Offset = (info.loop % 20) * (L1_P_DESCALE_SIZE / sizeof(SCALE_T));

        LoadData(pvL0BTensor[pvL0BOffset].ReinterpretCast<QUANT_T>(), pL1Tensor[pL1Offset].ReinterpretCast<QUANT_T>(),
                 pDescaleL1Tensor[pScaleL1Offset], loadData2DParamsB, load2DMxParamsB);
    }

    __aicore__ inline void LoadVToL0(const CubeRunInfo& info)
    {
        LoadData2DParamsV2 loadData2DParamsA;
        loadData2DParamsA.mStartPosition = info.prefetched ? (s2BaseSize / 16) : 0;
        loadData2DParamsA.kStartPosition = 0;
        loadData2DParamsA.mStep = info.actSingleLoopS2SizeAlign64 / 16;
        loadData2DParamsA.kStep = constInfo.dSize / GetBlockElemCnt<QUANT_T>();

        loadData2DParamsA.srcStride = s2BaseSize * 2 / 16;
        loadData2DParamsA.dstStride = (constInfo.dSize + 15) / 16 + 1;
        loadData2DParamsA.ifTranspose = true;

        LoadData2DMxParams load2DMxParamsA;
        load2DMxParamsA.xStartPosition = 0;
        load2DMxParamsA.yStartPosition = info.prefetched ? (s2BaseSize / 64) : 0;
        load2DMxParamsA.xStep = (constInfo.dSize + 15) / 16;
        load2DMxParamsA.yStep =
            (info.actSingleLoopS2SizeAlign64 + GetBlockElemCnt<QUANT_T>() - 1) / GetBlockElemCnt<QUANT_T>();
        load2DMxParamsA.srcStride = s2BaseSize * 2 / 64;
        load2DMxParamsA.dstStride = load2DMxParamsA.yStep;

        uint32_t pvL0AOffset = pvL0abBufId * (PV_L0A_SIZE / sizeof(DATA_T));
        uint32_t vL1Offset = vDataBufId * (L1_KV_SIZE / sizeof(DATA_T));

        uint32_t vScaleL1Offset = vDataBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        LoadData(pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(), kvL1Tensor[vL1Offset].ReinterpretCast<QUANT_T>(),
                 kvDescaleL1Tensor[vScaleL1Offset], loadData2DParamsA, load2DMxParamsA);
    }

    __aicore__ inline void MatmulPV(const CubeRunInfo& info)
    {
        MmadParams mmadParams;
        mmadParams.m = constInfo.dSize + 16;
        mmadParams.n = info.actMSizeAlign128;
        mmadParams.k = info.actSingleLoopS2SizeAlign64;
        mmadParams.cmatrixInitVal = info.isC2Sync;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;
        uint32_t pvL0AOffset = pvL0abBufId * (PV_L0A_SIZE / sizeof(DATA_T));
        uint32_t pvL0BOffset = pvL0abBufId * (PV_L0B_SIZE / sizeof(DATA_T));
        uint32_t pvL0COffset = pvL0cBufId * (PV_L0C_SIZE / sizeof(COMPUTE_T));
        Mmad(pvL0CTensor[pvL0COffset], pvL0ATensor[pvL0AOffset].ReinterpretCast<QUANT_T>(),
             pvL0BTensor[pvL0BOffset].ReinterpretCast<QUANT_T>(), mmadParams);
    }

    __aicore__ inline void FixpipeMm2(const CubeRunInfo& info)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (info.actMSizeAlign128 + 7) >> 3 << 3;
        fixpipeParams.mSize = constInfo.dSize + 1;
        fixpipeParams.srcStride = constInfo.dSize + 16;
        fixpipeParams.dstStride = 64; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        // n=128 时按 N 劈 64+64 给两个 AIV; ND 的 n 收到 16 后无列可劈, 用单目标并显式指定目标 subblock
        fixpipeParams.dualDstCtl = 2; // N 维按 64+64 拆给两个 AIV
        fixpipeParams.subBlockId = 0;

        uint32_t pvL0COffset = pvL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Fixpipe<COMPUTE_T, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm2ResUB, pvL0CTensor[pvL0COffset], fixpipeParams);
    }
};

} // namespace QFA_KERNEL
#endif
