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

template <typename QFAT>
class QuantFlashAttnBlockCubeDn {
public:
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

    static constexpr bool DIRECT_KV_COPY = (LAYOUT_KV == QFA_LAYOUT::BNSD) && !PAGE_ATTENTION;
    static constexpr bool DIRECT_Q_COPY = (LAYOUT_Q == QFA_LAYOUT::BNSD) && !PAGE_ATTENTION;
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
    static constexpr uint32_t L1_Q_SIZE = 128 * 128 / 2;                              // 8K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_Q_DESCALE_SIZE = 128 * (128 / 32) * sizeof(SCALE_T); // 0.5K
    static constexpr uint32_t L1_KV_SIZE = 512 * 128 / 2; // 32K, 2个fp4_e2m1元素为1B, s2单次最大256行
    static constexpr uint32_t L1_KV_DESCALE_SIZE = 512 * (128 / 32) * sizeof(SCALE_T); // 2K
    static constexpr uint32_t L1_P_SIZE = 128 * 256 / 2;                               // 16K, 2个fp4_e2m1元素为1B
    static constexpr uint32_t L1_P_DESCALE_SIZE = 128 * 2 * (256 / 64 + 1) * sizeof(SCALE_T); // 1.25K
    static constexpr uint32_t L1_Q_BUFCNT = 1;
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

    static constexpr uint32_t UB_MM1_RES_OFFSET = 0;
    static constexpr uint32_t UB_MM2_RES_OFFSET = 128 * 1024;
    static constexpr uint32_t UB_PEER_GLOBAL_MAX_OFFSET = 229888;

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
        }

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
        localGlobalMaxL1 = LocalTensor<half>(TPosition::A1, addrL1Start, 256 * 4); // 512b

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

        fixpipeMm1Params.dstStride = 512 / sizeof(half);
        fixpipeMm1Params.quantPre = QuantMode_t::QF322F16_PRE;
        fixpipeMm1Params.deqScalar = static_cast<uint64_t>(*reinterpret_cast<const int32_t*>(&constInfo.scaleValue));
        fixpipeMm1Params.dualDstCtl = 0;
        fixpipeMm1Params.params.ndNum = 1;
        fixpipeMm1Params.params.dstNdStride = 512 / sizeof(half);
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
            CopyKGmToL1(info);
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
                FixpipeMm1(info, loop, actS2Size); // 128 * 256
                Mutex::Unlock<PIPE_FIX>(QK_L0C_MUTEX_ID_BASE + qkL0cBufId);
                qkL0cBufId = (qkL0cBufId + 1) % 2;
            }
            Mutex::Unlock<PIPE_MTE1>(kHalf);
        }
    }

    __aicore__ inline void ComputeMm2(const CubeRunInfo& info)
    {
        if (info.actSingleLoopS2SizeAlign != info.actSingleLoopS2SizeAlign64) {
            Mutex::Lock<PIPE_MTE2>(P_FILL_MUTEX_ID);
            InitConstValueParams<uint16_t> PL1InitParams(
                1, static_cast<uint16_t>(info.actSingleLoopS2SizeAlign64 - info.actSingleLoopS2SizeAlign), 0, 0x0000);
            uint32_t l1P_Base_Offset = (info.loop % 20) * L1_P_SIZE;
            Fill(pL1Tensor[l1P_Base_Offset + info.actSingleLoopS2SizeAlign * 32].template ReinterpretCast<uint16_t>(),
                 PL1InitParams);
            if (likely(info.actMSizeAlign128 == mBaseSize)) {
                Fill(pL1Tensor[l1P_Base_Offset + 2 * info.actSingleLoopS2SizeAlign * 32 + 32 * 32]
                         .template ReinterpretCast<uint16_t>(),
                     PL1InitParams);
            }
            Mutex::Unlock<PIPE_MTE2>(P_FILL_MUTEX_ID);
        }
        uint32_t vBuf = info.prefetched ? vDataBufId : kvBufId;
        uint32_t vHalfA = KV_MUTEX_ID_BASE + vBuf * 2;
        bool needVFill = (info.actSingleLoopS2Size != info.actSingleLoopS2SizeAlign64);
        Mutex::Lock<PIPE_MTE2>(vHalfA);
        Mutex::Lock<PIPE_MTE2>(vHalfA + 1);
        if (needVFill) {
            InitConstValueParams<uint16_t> kvL1InitParams(
                1, static_cast<uint16_t>((info.actSingleLoopS2SizeAlign64 - info.actSingleLoopS2Size)), 0, 0x0000);
            uint32_t l1V_Base_Offset = vBuf * L1_KV_SIZE;
            uint32_t vOffset = info.prefetched ? s2BaseSize * (constInfo.dSize / 4) : 0;
            Fill(kvL1Tensor[l1V_Base_Offset + vOffset + constInfo.dSize / 4 * info.actSingleLoopS2Size]
                     .template ReinterpretCast<uint16_t>(),
                 kvL1InitParams);
            Fill(kvL1Tensor[l1V_Base_Offset + vOffset + constInfo.dSize / 4 * info.actSingleLoopS2Size * 2 +
                            (info.actSingleLoopS2SizeAlign64 - info.actSingleLoopS2Size) * constInfo.dSize / 4]
                     .template ReinterpretCast<uint16_t>(),
                 kvL1InitParams);
        }

        if (!info.prefetched) {
            CopyVGmToL1(info);
            CopyVScaleGmToL1(info);
        }

        Mutex::Unlock<PIPE_MTE2>(vHalfA);
        Mutex::Unlock<PIPE_MTE2>(vHalfA + 1);
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
        if constexpr (DIRECT_Q_COPY) {
            uint32_t dByte = headDimInt8; // Q数据D维, DATA_T(uint8_t)元素数
            qNd2NzParams.ndNum = 1;
            qNd2NzParams.dValue = dByte;
            qNd2NzParams.srcDValue = dByte;
            qNd2NzParams.dstNzNStride = 1;
            qNd2NzParams.srcNdMatrixStride = 0;
            qNd2NzParams.dstNzMatrixStride = 0;
            qGmStrideS1 = dByte;
            qGmStrideN2 = static_cast<uint64_t>(constInfo.gSize) * constInfo.s1Size * dByte;
            qGmStrideB = static_cast<uint64_t>(constInfo.n2Size) * qGmStrideN2;

            uint32_t kvHeadNum = constInfo.n2Size / constInfo.gRealSize;

            uint32_t dScale = constInfo.dSize / MXFP_GROUP_SIZE; // Q/K scale D维, SCALE_T元素数
            qScaleDn2NzParams.dnNum = 1;
            qScaleDn2NzParams.nValue = dScale / 2;
            qScaleDn2NzParams.srcDnMatrixStride = 0;
            qScaleDn2NzParams.srcDValue = dScale / 2;
            qScaleDn2NzParams.dstNzC0Stride = dScale / 2;
            qScaleDn2NzParams.dstNzNStride = 1;
            qScaleDn2NzParams.dstNzMatrixStride = dScale / 2;
            qScaleGmStrideS1 = dScale;
            qScaleGmStrideN2 = static_cast<uint64_t>(constInfo.gSize) * constInfo.s1Size * dScale;
            qScaleGmStrideB = static_cast<uint64_t>(constInfo.n2Size) * qScaleGmStrideN2;

            kScaleDn2NzParams.dnNum = 1;
            kScaleDn2NzParams.nValue = dScale / 2;
            kScaleDn2NzParams.srcDnMatrixStride = 0;
            kScaleDn2NzParams.srcDValue = dScale / 2;
            kScaleDn2NzParams.dstNzC0Stride = dScale / 2;
            kScaleDn2NzParams.dstNzNStride = 1;
            kScaleDn2NzParams.dstNzMatrixStride = dScale / 2;
            kScaleGmStrideS2 = dScale;
            kScaleGmStrideN2 = static_cast<uint64_t>(constInfo.s2Size) * dScale;
            kScaleGmStrideB = static_cast<uint64_t>(kvHeadNum) * kScaleGmStrideN2;

            uint32_t dScaleV = 2 * constInfo.dSize; // V scale D维(2*MXFP_GROUP_SIZE分形)
            vScaleNd2NzParams.ndNum = 1;
            vScaleNd2NzParams.dValue = dScaleV / 2;
            vScaleNd2NzParams.srcDValue = dScaleV / 2;
            vScaleNd2NzParams.dstNzC0Stride = s2BaseSize * 2 / 64;
            vScaleNd2NzParams.dstNzNStride = 1;
            vScaleNd2NzParams.srcNdMatrixStride = 0;
            vScaleGmStrideS2 = dScaleV;
            vScaleGmStrideN2 = static_cast<uint64_t>((constInfo.maxSeqlenKv + 63) / 64) * dScaleV;
            vScaleGmStrideB = static_cast<uint64_t>(kvHeadNum) * vScaleGmStrideN2;
        }
        if constexpr (DIRECT_KV_COPY) {
            uint32_t dByte = headDimInt8;
            kvNd2NzParams.ndNum = 1;
            kvNd2NzParams.dValue = dByte;
            kvNd2NzParams.srcDValue = dByte;
            kvNd2NzParams.dstNzC0Stride = s2BaseSize * 2;
            kvNd2NzParams.dstNzNStride = 1;
            kvNd2NzParams.srcNdMatrixStride = 0;
            kvNd2NzParams.dstNzMatrixStride = 0;
            kvGmStrideS2 = dByte;
            kvGmStrideN2 = static_cast<uint64_t>(constInfo.s2Size) * dByte;
            kvGmStrideB = static_cast<uint64_t>(constInfo.n2Size / constInfo.gRealSize) * kvGmStrideN2;
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
        qNd2NzParams.nValue = info.actMSize;
        qNd2NzParams.dstNzC0Stride = info.actMSizeAlign128;
        uint64_t gmOffset = info.bIdx * qGmStrideB + info.n2Idx * qGmStrideN2 + info.gS1Idx * qGmStrideS1;
        DataCopy(qL1Tensor, queryGm[gmOffset], qNd2NzParams);
    }

    __aicore__ inline void CopyKGmToL1(const CubeRunInfo& info)
    {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint64_t gmOffset = info.bIdx * kvGmStrideB + info.kvHeadIdx * kvGmStrideN2 + info.s2Idx * kvGmStrideS2;
        kvNd2NzParams.nValue = info.pairKVCopyS2Size;
        DataCopy(kvL1Tensor[l1BaseOffset], keyGm[gmOffset], kvNd2NzParams);
    }

    __aicore__ inline void CopyVGmToL1(const CubeRunInfo& info)
    {
        uint64_t l1BaseOffset = kvBufId * (L1_KV_SIZE / sizeof(DATA_T));
        uint64_t gmOffset = info.bIdx * kvGmStrideB + info.kvHeadIdx * kvGmStrideN2 + info.s2Idx * kvGmStrideS2;
        kvNd2NzParams.nValue = info.pairKVCopyS2Size;
        DataCopy(kvL1Tensor[l1BaseOffset], valueGm[gmOffset], kvNd2NzParams);
    }

    // copy query scale with full s1g
    __aicore__ inline void CopyQScaleGmToL1(const CubeRunInfo& info)
    {
        qScaleDn2NzParams.dValue = info.actMSize;
        uint64_t gmOffset =
            info.bIdx * qScaleGmStrideB + info.n2Idx * qScaleGmStrideN2 + info.gS1Idx * qScaleGmStrideS1;
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(queryScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(qDescaleL1Tensor.template ReinterpretCast<bfloat16_t>(), gmTensorCast, qScaleDn2NzParams);
    }

    // copy key scale with full s2
    __aicore__ inline void CopyKScaleGmToL1(const CubeRunInfo& info)
    {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        kScaleDn2NzParams.dValue = info.pairKVCopyS2Size;
        uint64_t gmOffset =
            info.bIdx * kScaleGmStrideB + info.kvHeadIdx * kScaleGmStrideN2 + info.s2Idx * kScaleGmStrideS2;
        GlobalTensor<bfloat16_t> gmTensorCast;
        gmTensorCast.SetGlobalBuffer((__gm__ bfloat16_t*)(keyScaleGm[gmOffset].GetPhyAddr()));
        DataCopy(kvDescaleL1Tensor[offset].template ReinterpretCast<bfloat16_t>(), gmTensorCast, kScaleDn2NzParams);
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
        uint32_t qL1Offset = 0;
        uint32_t qScaleL1Offset = 0;
        LoadData(qkL0BTensor[qkL0Offset].ReinterpretCast<QUANT_T>(), qL1Tensor[qL1Offset].ReinterpretCast<QUANT_T>(),
                 qDescaleL1Tensor[qScaleL1Offset], loadData2DParamsA, loadData2DMxParamsA);
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
        LoadData(qkL0ATensor[qkL0Offset].ReinterpretCast<QUANT_T>(), kvL1Tensor[kL1Offset].ReinterpretCast<QUANT_T>(),
                 kvDescaleL1Tensor[kScaleL1Offset], loadKParamsA, loadKParamsMx);
    }

    __aicore__ inline void MatmulQK(const CubeRunInfo& info, uint32_t actS2Size)
    {
        mmadQKParams.m = actS2Size;
        mmadQKParams.n = info.actMSizeAlign128;
        uint32_t qkL0AOffset = qkL0abBufId * (QK_L0A_SIZE / sizeof(DATA_T));
        uint32_t qkL0BOffset = qkL0abBufId * (QK_L0B_SIZE / sizeof(DATA_T));
        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Mmad(qkL0CTensor[qkL0COffset], qkL0ATensor[qkL0AOffset].ReinterpretCast<QUANT_T>(),
             qkL0BTensor[qkL0BOffset].ReinterpretCast<QUANT_T>(), mmadQKParams);
    }

    __aicore__ inline void FixpipeMm1(const CubeRunInfo& info, uint32_t subLoop, uint32_t actS2Size)
    {
        fixpipeMm1Params.nSize = (info.actMSizeAlign128 + 7) >> 3 << 3;
        fixpipeMm1Params.mSize = (actS2Size + 1) >> 1 << 1;
        fixpipeMm1Params.srcStride = ((actS2Size + 15) / 16) * 16;
        fixpipeMm1Params.subBlockId = info.loop % 2;
        fixpipeMm1Params.params.srcNdStride = (actS2Size + 1) >> 1 << 1;

        uint32_t qkL0COffset = qkL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        uint32_t ubOffset = subLoop * QK_L0_S2_SPLIT_SIZE * 256;
        Fixpipe<half, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm1ResUB[info.loop / 2 % 2 * 128 + ubOffset],
                                                   qkL0CTensor[qkL0COffset], fixpipeMm1Params);
    }

    __aicore__ inline void CopyVScaleGmToL1(const CubeRunInfo& info)
    {
        uint32_t offset = kvBufId * (L1_KV_DESCALE_SIZE / sizeof(SCALE_T));
        vScaleNd2NzParams.nValue = (info.pairKVCopyS2Size + 63) / 64;
        vScaleNd2NzParams.dstNzMatrixStride = vScaleNd2NzParams.nValue;
        uint64_t gmOffset =
            info.bIdx * vScaleGmStrideB + info.kvHeadIdx * vScaleGmStrideN2 + (info.s2Idx / 64) * vScaleGmStrideS2;
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
        fixpipeParams.dualDstCtl = 2; // 双目标模式，按N维度拆分

        uint32_t pvL0COffset = pvL0cBufId * (QK_L0C_SIZE / sizeof(COMPUTE_T));
        Fixpipe<COMPUTE_T, COMPUTE_T, CFG_ROW_MAJOR_UB>(mm2ResUB, pvL0CTensor[pvL0COffset], fixpipeParams);
    }
};

} // namespace QFA_KERNEL
#endif
