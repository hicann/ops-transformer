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
 * \file mixed_quant_flash_attn_block_cube_antiquant_gqa.h
 * \brief
 */
#ifndef FLASH_ATTENTION_ANTIQUANT_GQA_BLOCK_CUBE_H_
#define FLASH_ATTENTION_ANTIQUANT_GQA_BLOCK_CUBE_H_
#include "kernel_operator_list_tensor_intf.h"
#include "util.h"
#include "../../../common/op_kernel/matmul.h"
#include "memory_copy_arch35_mixed_quant_flash_attn.h"
#include "../utils/mixed_quant_flash_attn_utils.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {
template <typename T1>
static __aicore__ inline constexpr T1 Align(T1 s, T1 alignment)
{
    if constexpr (IsSameType<T1, uint64_t>::value || IsSameType<T1, uint32_t>::value ||
                  IsSameType<T1, uint16_t>::value || IsSameType<T1, uint8_t>::value) {
        return (s + alignment - 1) & (~(alignment - 1));
    } else {
        return uint64_t(s + alignment - 1) & (~uint64_t(alignment - 1));
    }
}

template <typename MQFA_T>
class CubeBlockBase {
public:
    using Q_T = typename MQFA_T::qType;
    using KV_T = typename MQFA_T::kvType;
    using KV_SCALE_T = typename MQFA_T::kvScaleType;
    using MM_T = float;
    static constexpr uint32_t mBaseSize = (uint32_t)MQFA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)MQFA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)MQFA_T::dBaseSize;
    static constexpr bool PAGE_ATTENTION = MQFA_T::pageAttention;
    static constexpr uint8_t QUANT_COMPUTE_MODE = MQFA_T::quantComputeMode;
    static constexpr LayOutTypeEnum LAYOUT_Q = MQFA_T::layoutQ;
    static constexpr uint8_t LAYOUT_KV = MQFA_T::layoutKV;
    static constexpr bool USE_DN = MQFA_T::useDN;

    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t HIFP_GROUP_SIZE = 64U;
    static constexpr uint32_t GROUP_SIZE =
        (std::is_same_v<KV_T, fp4x2_e2m1_t> ? MXFP_GROUP_SIZE :
                                              (std::is_same_v<KV_T, hifloat4x2_t> ? HIFP_GROUP_SIZE : 1U));
    static constexpr uint32_t L1_Q_BUF_NUM = 2;
    static constexpr uint32_t L1_P_BUF_NUM = 2;
    static constexpr uint32_t L1_KV_BUF_NUM = 4;
    static constexpr uint32_t L1_KVSCALE_BUF_NUM = 4;
    static constexpr uint32_t L0C_BUF_NUM = 2;

    static constexpr uint32_t L1_Q_SLOT_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_P_SLOT_BYTES = mBaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_SLOT_BYTES =
        (std::is_same_v<KV_T, fp4x2_e2m1_t> || std::is_same_v<KV_T, hifloat4x2_t>) ?
            dBaseSize * s2BaseSize * sizeof(KV_T) / 2 :
            dBaseSize * s2BaseSize * sizeof(KV_T);
    static constexpr uint32_t L1_KVSCALE_SLOT_BYTES = 512 * dBaseSize * sizeof(KV_SCALE_T) / GROUP_SIZE;
    static constexpr uint32_t L0AB_BUF_SIZE = 32 * 1024;
    static constexpr uint32_t L0C_SLOT_BYTES = 128 * 1024;
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;

    // 与真实 CubeBlock 的 L1_BUF_VIEW 完全一致(成员顺序/大小/偏移相同)
    struct L1_BUF_VIEW {
        // bank 0
        uint8_t l1PBuffer0[L1_P_SLOT_BYTES];
        uint8_t l1QBuffer0[L1_Q_SLOT_BYTES];
        uint8_t l1KVBuffer0[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer2[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer4[L1_KV_SLOT_BYTES];
        uint8_t l1KVScaleBuf0[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf2[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf4[L1_KVSCALE_SLOT_BYTES];
        uint8_t
            bank0Pad[256 * 1024 - L1_P_SLOT_BYTES - L1_Q_SLOT_BYTES - L1_KV_SLOT_BYTES * 3 - L1_KVSCALE_SLOT_BYTES * 3];
        // bank1
        uint8_t l1PBuffer1[L1_P_SLOT_BYTES];
        uint8_t l1QBuffer1[L1_Q_SLOT_BYTES];
        uint8_t l1KVBuffer1[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer3[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer5[L1_KV_SLOT_BYTES];
        uint8_t l1KVScaleBuf1[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf3[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf5[L1_KVSCALE_SLOT_BYTES];
        uint8_t
            bank1Pad[256 * 1024 - L1_P_SLOT_BYTES - L1_Q_SLOT_BYTES - L1_KV_SLOT_BYTES * 3 - L1_KVSCALE_SLOT_BYTES * 3];
    };

    __aicore__ inline CubeBlockBase(ConstInfoX& constInfo){};
};

// 偏移量计算函数 (使用宏简化)
#define GET_OFFSET(type, member) reinterpret_cast<size_t>(&(reinterpret_cast<type*>(0)->member))

template <typename MQFA_T>
class FAAntiQuantGqaBlockCube {
public:
    using Q_T = typename MQFA_T::qType;
    using KV_T = typename MQFA_T::kvType;
    using KV_SCALE_T = typename MQFA_T::kvScaleType;
    using MM_T = float;
    static constexpr uint32_t mBaseSize = (uint32_t)MQFA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)MQFA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)MQFA_T::dBaseSize;
    static constexpr bool PAGE_ATTENTION = MQFA_T::pageAttention;
    static constexpr uint8_t QUANT_COMPUTE_MODE = MQFA_T::quantComputeMode;
    static constexpr LayOutTypeEnum LAYOUT_Q = MQFA_T::layoutQ;
    static constexpr uint8_t LAYOUT_KV = MQFA_T::layoutKV;
    static constexpr bool USE_DN = MQFA_T::useDN;
    using ConstInfoX = ConstInfo_t<FiaKernelType::ANTI_QUANT>;

    using MM1_RES_BUF_T = LocalTensor<MM_T>;
    using MM2_RES_BUF_T = LocalTensor<MM_T>;
    using MM1_ABUF_T = LocalTensor<Q_T>;
    using MM2_ABUF_T = LocalTensor<Q_T>;

    using L0AType =
        BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    using L0BType =
        BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    /* ==================== 静态变量 ==================== */
    // L1同步事件 (c2p: MTE1等MTE2)
    static constexpr uint32_t L1Q_EVENT_0 = 0;
    static constexpr uint32_t L1Q_EVENT_1 = 1;
    // KV/KVScale共用4buf事件 (slot 0-3)
    static constexpr uint32_t L1KV_EVENT_0 = 2;
    static constexpr uint32_t L1KV_EVENT_1 = 3;
    static constexpr uint32_t L1KV_EVENT_2 = 4;
    static constexpr uint32_t L1KV_EVENT_3 = 5;
    // L0同步事件 （c2p: M<->MTE1)
    static constexpr uint32_t L0A_EVENT_0 = 14; // L0A ping
    static constexpr uint32_t L0A_EVENT_1 = 15; // L0A pong
    // L0B同步事件 (c2p: M<->MTE1)
    static constexpr uint32_t L0B_EVENT_0 = 16; // L0B ping
    static constexpr uint32_t L0B_EVENT_1 = 17; // L0B pong
    // L0C同步事件 (c2p: M<->FIXP)
    static constexpr uint32_t L0C_EVENT_0 = 18; // L0C ping
    static constexpr uint32_t L0C_EVENT_1 = 19; // L0C pong
                                                /* ==================== ==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_Q>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<LAYOUT_Q, LAYOUT_KV>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKDescaleGmFormat<LAYOUT_Q, LAYOUT_KV>();
    static constexpr GmFormat V_SCALE_FORMAT = GetVDescaleGmFormat<LAYOUT_Q, LAYOUT_KV>();

    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr uint32_t HIFP_GROUP_SIZE = 64U;
    static constexpr uint32_t GROUP_SIZE =
        (std::is_same_v<KV_T, fp4x2_e2m1_t> ? MXFP_GROUP_SIZE :
                                              (std::is_same_v<KV_T, hifloat4x2_t> ? HIFP_GROUP_SIZE : 1U));
    static constexpr bool IS_PTG_PCG =
        (QUANT_COMPUTE_MODE == A16C4_KV_MXFP4_SOFTMAX_FP32 || QUANT_COMPUTE_MODE == A16C4_KV_HIF4_SOFTMAX_FP32);
    static constexpr bool IS_PC = (QUANT_COMPUTE_MODE == AntiquantMode_FP8_PC);
    static constexpr FixpipeConfig FIXPIPE_ROW_MAJOR_UB_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr FixpipeConfig FIXPIPE_COLUMN_MAJOR_UB_CONFIG = {CO2Layout::COLUMN_MAJOR, true};
    /* ===================== Buffer 分配 ====================*/
    static constexpr uint32_t L1_Q_BUF_NUM = 2;
    static constexpr uint32_t L1_P_BUF_NUM = 2;
    static constexpr uint32_t L1_KV_BUF_NUM = 4;
    static constexpr uint32_t L1_KVSCALE_BUF_NUM = 4;
    static constexpr uint32_t L0C_BUF_NUM = 2;

    static constexpr uint32_t L0AB_BUF_SIZE = 32 * 1024;
    static constexpr uint32_t L0C_SLOT_BYTES = 128 * 1024;
    static constexpr uint32_t L1_Q_SLOT_BYTES = mBaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_P_SLOT_BYTES = mBaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t L1_KV_SLOT_BYTES =
        (std::is_same_v<KV_T, fp4x2_e2m1_t> || std::is_same_v<KV_T, hifloat4x2_t>) ?
            dBaseSize * s2BaseSize * sizeof(KV_T) / 2 :
            dBaseSize * s2BaseSize * sizeof(KV_T);
    static constexpr uint32_t L1_KVSCALE_SLOT_BYTES = 512 * dBaseSize * sizeof(KV_SCALE_T) / GROUP_SIZE;

    /* ==================== GM变量 ==================== */
    FaGmTensor<Q_T, Q_FORMAT, int32_t> queryGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t> keyGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t> valueGm;
    FaGmTensor<KV_SCALE_T, K_SCALE_FORMAT, int32_t> kScaleGm;
    FaGmTensor<KV_SCALE_T, V_SCALE_FORMAT, int32_t> vScaleGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;
    CopyKeyScaleGmToL1<KV_SCALE_T, K_SCALE_FORMAT> copyKeyScaleGmToL1;
    CopyValueScaleGmToL1<KV_SCALE_T, V_SCALE_FORMAT, L1Format::NZ, ScaleTrans::DN2NZ> copyValueScaleGmToL1;
    /* ===================== LocalBuffer ====================*/
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager;
    LocalTensor<uint8_t> l1QBuffers[L1_Q_BUF_NUM];
    LocalTensor<uint8_t> l1KVBuffers[L1_KV_BUF_NUM];           // k/v共用
    LocalTensor<uint8_t> l1KVScaleBuffers[L1_KVSCALE_BUF_NUM]; // k/v scale共用
    L0AType l0ABuffers;
    L0BType l0BBuffers;
    LocalTensor<uint8_t> l0CBuffers;

    uint32_t qBufId = 0;
    uint32_t kvBufId = 0;      // k/v共用轮转计数器
    uint32_t kvScaleBufId = 0; // k/v scale共用轮转计数器
    uint32_t l0CBufId = 0;
    const ConstInfoX& constInfo;
    // L1 buf布局
    struct L1_BUF_VIEW {
        // bank 0
        uint8_t l1PBuffer0[L1_P_SLOT_BYTES];
        uint8_t l1QBuffer0[L1_Q_SLOT_BYTES];
        uint8_t l1KVBuffer0[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer2[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer4[L1_KV_SLOT_BYTES];
        uint8_t l1KVScaleBuf0[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf2[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf4[L1_KVSCALE_SLOT_BYTES];
        uint8_t
            bank0Pad[256 * 1024 - L1_P_SLOT_BYTES - L1_Q_SLOT_BYTES - L1_KV_SLOT_BYTES * 3 - L1_KVSCALE_SLOT_BYTES * 3];
        // bank1
        uint8_t l1PBuffer1[L1_P_SLOT_BYTES];
        uint8_t l1QBuffer1[L1_Q_SLOT_BYTES];
        uint8_t l1KVBuffer1[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer3[L1_KV_SLOT_BYTES];
        uint8_t l1KVBuffer5[L1_KV_SLOT_BYTES];
        uint8_t l1KVScaleBuf1[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf3[L1_KVSCALE_SLOT_BYTES];
        uint8_t l1KVScaleBuf5[L1_KVSCALE_SLOT_BYTES];
        uint8_t
            bank1Pad[256 * 1024 - L1_P_SLOT_BYTES - L1_Q_SLOT_BYTES - L1_KV_SLOT_BYTES * 3 - L1_KVSCALE_SLOT_BYTES * 3];
    };

    /*============================================================================== */
    __aicore__ inline FAAntiQuantGqaBlockCube(ConstInfoX& constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock()
    {
        InitLocalBuffer();
        AllocEventID();
    }

    __aicore__ inline void FreeCube()
    {
        FreeEventID();
    }

    __aicore__ inline void InitLocalBuffer()
    {
        // perchannel Q为共享buf
        if constexpr (!IS_PC) {
            l1QBuffers[0] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1QBuffer0), L1_Q_SLOT_BYTES);
            l1QBuffers[1] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1QBuffer1), L1_Q_SLOT_BYTES);
        }
        // K/V共用6buf, 按bank布局分配(偶数slot进bank0, 奇数slot进bank1)
        l1KVBuffers[0] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVBuffer0), L1_KV_SLOT_BYTES);
        l1KVBuffers[1] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVBuffer1), L1_KV_SLOT_BYTES);
        l1KVBuffers[2] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVBuffer2), L1_KV_SLOT_BYTES);
        l1KVBuffers[3] = LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVBuffer3), L1_KV_SLOT_BYTES);
        // K/V Scale共用4buf, 按bank布局分配
        if constexpr (IS_PTG_PCG) {
            l1KVScaleBuffers[0] =
                LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVScaleBuf0), L1_KVSCALE_SLOT_BYTES);
            l1KVScaleBuffers[1] =
                LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVScaleBuf1), L1_KVSCALE_SLOT_BYTES);
            l1KVScaleBuffers[2] =
                LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVScaleBuf2), L1_KVSCALE_SLOT_BYTES);
            l1KVScaleBuffers[3] =
                LocalTensor<uint8_t>(TPosition::A1, GET_OFFSET(L1_BUF_VIEW, l1KVScaleBuf3), L1_KVSCALE_SLOT_BYTES);
        }
        // TODO
        l0aBufferManager.Init(64 * 1024);
        l0bBufferManager.Init(64 * 1024);
        l0ABuffers.Init(l0aBufferManager, L0AB_BUF_SIZE, L0A_EVENT_0, L0A_EVENT_1);
        l0BBuffers.Init(l0bBufferManager, L0AB_BUF_SIZE, L0B_EVENT_0, L0B_EVENT_1);
        l0CBuffers = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUF_NUM * L0C_SLOT_BYTES);
    }

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID()
    {
        l0ABuffers.Uninit(l0aBufferManager);
        l0BBuffers.Uninit(l0bBufferManager);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                         __gm__ uint8_t* attenMask, __gm__ uint8_t* cuSeqLensQ,
                                         __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* seqUsedQ,
                                         __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* blockTable,
                                         __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale)
    {
        if (constInfo.seqUsedQSize != 0U) {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t*)seqUsedQ, constInfo.seqUsedQSize);
        }
        if (constInfo.seqUsedKvSize != 0U) {
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t*)seqUsedKv, constInfo.seqUsedKvSize);
        }
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }

        InitQGmTensor(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize, seqUsedGmQ,
                      constInfo.seqUsedQSize, queryGm, query);
        InitKVGmTensor(constInfo.bSize, constInfo.s2Size, seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.n2Size,
                       constInfo.blockSize, constInfo.dSize, keyGm, key);
        InitKVGmTensor(constInfo.bSize, constInfo.s2Size, seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.n2Size,
                       constInfo.blockSize, constInfo.dSize, valueGm, value);
        if constexpr (IS_PTG_PCG) {
            InitKScaleGmTensor(constInfo.bSize, constInfo.s2Size, seqUsedGmKv, constInfo.seqUsedKvSize,
                               constInfo.n2Size, constInfo.blockSize, constInfo.dSize / GROUP_SIZE, kScaleGm, kDescale);
            uint32_t s2GroupSize = (constInfo.s2Size + GROUP_SIZE - 1) / GROUP_SIZE;
            if (std::is_same_v<KV_T, fp4x2_e2m1_t>) {
                s2GroupSize = s2GroupSize + (s2GroupSize & 1); // 对齐到偶数
            }
            InitVScaleGmTensor(constInfo.bSize, s2GroupSize, seqUsedGmKv, constInfo.seqUsedKvSize, constInfo.n2Size,
                               constInfo.blockSize / GROUP_SIZE, constInfo.dSize, vScaleGm, vDescale);
        }
    }

    __aicore__ inline void InitQGmTensor(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                         uint32_t headDim, GlobalTensor<int32_t> seqUsedGmQ, uint32_t actualLenQDims,
                                         FaGmTensor<Q_T, Q_FORMAT, int32_t>& qGmTensor, __gm__ uint8_t* gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T*)gm);
        if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, seqUsedGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitKVGmTensor(uint32_t batchSize, uint32_t kvSeqSize, GlobalTensor<int32_t> seqUsedGmKv,
                                          uint32_t actualLenDims, uint32_t n2Size, uint32_t kvCacheBlockSize,
                                          uint32_t headDim, FaGmTensor<KV_T, KV_FORMAT, int32_t>& kvGmTensor,
                                          __gm__ uint8_t* gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T*)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = (std::is_same_v<KV_T, fp4x2_e2m1_t> || std::is_same_v<KV_T, hifloat4x2_t> ?
                               32 / sizeof(KV_T) * 2 : // KV 4bit sizeof(KV_T) = 1
                               32 / sizeof(KV_T));
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, seqUsedGmKv, actualLenDims);
        }
    }

    __aicore__ inline void InitKScaleGmTensor(uint32_t batchSize, uint32_t kvSeqSize, GlobalTensor<int32_t> seqUsedGmQ,
                                              uint32_t actualLenDims, uint32_t n2Size, uint32_t kvScaleBlockSize,
                                              uint32_t headDim,
                                              FaGmTensor<KV_SCALE_T, K_SCALE_FORMAT, int32_t>& kScaleGmTensor,
                                              __gm__ uint8_t* gm)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_SCALE_T*)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_K_SCALE_PA_NZ) {
            if constexpr (std::is_same_v<KV_T, hifloat4x2_t>) {
                uint32_t bs0d0 = 16U;
                uint32_t bs1 = kvScaleBlockSize / 16U;
                kScaleGmTensor.offsetCalculator.Init(n2Size, bs1, headDim, bs0d0, blockTableGm,
                                                     constInfo.maxBlockNumPerBatch);
            } else {
                // for mxfp4
                uint32_t bs0d0 = 32U;
                uint32_t bs1 = kvScaleBlockSize / 16U;
                kScaleGmTensor.offsetCalculator.Init(n2Size, bs1, headDim / 2, bs0d0, blockTableGm,
                                                     constInfo.maxBlockNumPerBatch);
            }
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvScaleBlockSize, headDim, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim, seqUsedGmKv, actualLenDims);
        }
    }

    __aicore__ inline void InitVScaleGmTensor(uint32_t batchSize, uint32_t kvSeqSize, GlobalTensor<int32_t> seqUsedGmQ,
                                              uint32_t actualLenDims, uint32_t n2Size, uint32_t kvScaleBlockSize,
                                              uint32_t headDim,
                                              FaGmTensor<KV_SCALE_T, V_SCALE_FORMAT, int32_t>& vScaleGmTensor,
                                              __gm__ uint8_t* gm)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_SCALE_T*)gm);
        if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_V_SCALE_PA_NZ) {
            if constexpr (std::is_same_v<KV_T, hifloat4x2_t>) {
                uint32_t d0s0 = 16U;
                uint32_t d1 = headDim / d0s0;
                vScaleGmTensor.offsetCalculator.Init(n2Size, kvScaleBlockSize, d1, d0s0, blockTableGm,
                                                     constInfo.maxBlockNumPerBatch);
            } else {
                // for mxfp4
                uint32_t d0s0 = 32U;
                uint32_t d1 = headDim / 16U;
                vScaleGmTensor.offsetCalculator.Init(n2Size, kvScaleBlockSize / 2, d1, d0s0, blockTableGm,
                                                     constInfo.maxBlockNumPerBatch);
            }
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_BnNDBs) {
            vScaleGmTensor.offsetCalculator.Init(n2Size, headDim, kvScaleBlockSize, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<V_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_BNDS) {
            vScaleGmTensor.offsetCalculator.Init(batchSize, n2Size, headDim, kvSeqSize);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = Align(runInfo.actMSize, 16U);

        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};
        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0U,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = static_cast<uint32_t>(constInfo.dSize)};
        copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = Align(runInfo.actSingleLoopS2Size, 16U);

        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0U,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = static_cast<uint32_t>(constInfo.dSize)};
        copyKvGmToL1(l1Tensor, keyGm, gmCoord);
    }

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<KV_SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        FaL1Tensor<KV_SCALE_T, L1Format::NZ> l1Tensor{
            .tensor = dstTensor,
            .rowCount = static_cast<uint32_t>(constInfo.dSize / GROUP_SIZE) // 无效参数
        };

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0U,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = static_cast<uint32_t>(constInfo.dSize / GROUP_SIZE)};
        copyKeyScaleGmToL1(l1Tensor, kScaleGm, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t dstStride = Align(runInfo.actSingleLoopS2Size, 16U);

        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0U,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = static_cast<uint32_t>(constInfo.dSize)};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    __aicore__ inline void CopyValueScaleTile(const LocalTensor<KV_SCALE_T>& dstTensor, RunInfoX& runInfo)
    {
        uint32_t s2DealSize = (runInfo.actSingleLoopS2Size + GROUP_SIZE - 1) / GROUP_SIZE;
        if (std::is_same_v<KV_T, fp4x2_e2m1_t>) {
            s2DealSize = s2DealSize + (s2DealSize & 1); // 对齐到偶数
        }

        FaL1Tensor<KV_SCALE_T, L1Format::NZ> l1Tensor{
            .tensor = dstTensor,
            .rowCount = s2DealSize // PA_NZ有效
        };

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx / GROUP_SIZE,
                          .dIdx = 0U,
                          .s2DealSize = s2DealSize,
                          .dDealSize = static_cast<uint32_t>(constInfo.dSize)};
        copyValueScaleGmToL1(l1Tensor, vScaleGm, gmCoord);
    }

    // 全载
    // 外部L1切入K时，需要传入cmatrixInitVal的标记
    template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL,
              ABLayout BL, typename L0AType, typename L0BType, typename AScaleType = float, typename BScaleType = float,
              typename L0ADType = A, typename L0BDType = B>
    __aicore__ inline void MatmulFull(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                      L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                      struct MMParam& param,
                                      const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<BScaleType>(),
                                      const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>())
    {
        auto l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>();
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK,
                                                param.singleM); // d,s2
        } else {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
        }

        auto l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value || IsSameType<L0BDType, fp4x2_e2m1_t>::value ||
                      IsSameType<L0BDType, hifloat4x2_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
        } else {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
        }
        l0aBuffer.template Set<HardEvent::MTE1_M>();

        l0aBuffer.template Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = param.singleK;
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        if (mmadParams.m == 1) {
            mmadParams.m = 16;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.template Set<HardEvent::M_MTE1>();
    }

    // 切K
    template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL,
              ABLayout BL, typename L0AType, typename L0BType, typename AScaleType = float, typename BScaleType = float,
              typename L0ADType = A, typename L0BDType = B>
    __aicore__ inline void MatmulK(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                   L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                   const MMParam& param,
                                   const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<BScaleType>(),
                                   const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>())
    {
        uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
        uint32_t tailSize = param.singleK % baseK;
        uint32_t tailK = tailSize ? tailSize : baseK;
        uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
        uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
        if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                      IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
            L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
            L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
        }
        if constexpr (IsSameType<B, hifloat4x2_t>::value || IsSameType<B, fp4x2_e2m1_t>::value) {
            L1Boffset = param.isRightTranspose ? ((param.singleN + 31) >> 5 << 5) * baseK : baseK << 6;
        }

        for (uint32_t k = 0; k < kLoops; k++) {
            uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
            auto l0aBuffer = aL0BuffsDb.Get();
            l0aBuffer.template Wait<HardEvent::M_MTE1>(); // mte1等Matmul：上一轮matmul完成后才能搬运新数据到L0A
            LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
            if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
                LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, k * L1Aoffset, tileK,
                                                    param.singleM); // s2,
            } else {
                LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM); // s2*d,d,s2
            }

            auto l0bBuffer = bL0BuffsDb.Get();
            LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
            uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
            if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value || IsSameType<L0BDType, fp4x2_e2m1_t>::value ||
                          IsSameType<L0BDType, hifloat4x2_t>::value) {
                LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, k * L1Boffset, tileK,
                                             param.singleN, loopNum); // tileK.D
            } else {
                LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
            }
            l0aBuffer.template Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
            l0aBuffer.template Wait<HardEvent::MTE1_M>();
            MmadParams mmadParams;
            mmadParams.m = param.singleM;
            if (param.realM != 0) {
                mmadParams.m = param.realM;
            }
            mmadParams.n = param.singleN;
            mmadParams.k = tileK;
            if (mmadParams.m == 1) { // m等于1或默认开GEMV模式，文档上没有写怎么关闭GEMV，所以规避当做矩阵运算
                mmadParams.m = 16;
            }
            mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
            mmadParams.cmatrixSource = false;
            if (param.unitFlag != 0) {
                mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                          UNITFLAG_EN_OUTER_LAST :
                                          UNITFLAG_ENABLE;
            }
            Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

            l0aBuffer.template Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
        }
    }

    __aicore__ inline void IterateBmm1(MM1_ABUF_T& inputBuf, MM1_RES_BUF_T& outputBuf, uint32_t outBufId,
                                       RunInfoX& runInfo)
    {
        if constexpr (IS_PTG_PCG) {
            IterateBmm1PTG(outputBuf, outBufId, runInfo);
        } else if constexpr (IS_PC) {
            IterateBmm1PC(inputBuf, outputBuf, outBufId, runInfo);
        }
    }

    __aicore__ inline void IterateBmm1PTG(MM1_RES_BUF_T& outputBuf, uint32_t outBufId, RunInfoX& runInfo)
    {
        // 搬运 Query
        LocalTensor<Q_T> mm1ATensor = l1QBuffers[qBufId].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            Mutex::Lock<PIPE_MTE2>(L1Q_EVENT_0 + qBufId);
            CopyQueryTile(mm1ATensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(L1Q_EVENT_0 + qBufId);
            Mutex::Lock<PIPE_MTE1>(L1Q_EVENT_0 + qBufId);
        }
        // 搬运 key
        LocalTensor<KV_T> mm1BTensor = l1KVBuffers[kvBufId].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(L1KV_EVENT_0 + kvBufId);
        CopyKeyTile(mm1BTensor, runInfo);
        // 搬运key scale
        LocalTensor<KV_SCALE_T> mm1BScaleTensor = l1KVScaleBuffers[kvScaleBufId].template ReinterpretCast<KV_SCALE_T>();
        CopyKeyScaleTile(mm1BScaleTensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(L1KV_EVENT_0 + kvBufId);
        Mutex::Lock<PIPE_MTE1>(L1KV_EVENT_0 + kvBufId);
        {
            MMParam param = {
                (uint32_t)runInfo.actMSize,            // singleM
                (uint32_t)runInfo.actSingleLoopS2Size, // singleN
                (uint32_t)(constInfo.dSize),           // singleK
                false,                                 // isLeftTranspose
                true                                   // isRightTranspose
            };

            uint32_t l0CBufOffset = l0CBufId * L0C_SLOT_BYTES;
            LocalTensor<MM_T> mm1ResL0C = l0CBuffers[l0CBufOffset].template ReinterpretCast<MM_T>();
            Mutex::Lock<PIPE_M>(L0C_EVENT_0 + l0CBufId);
            if constexpr (dBaseSize <= 128) {
                MatmulFull<Q_T, KV_T, MM_T, mBaseSize, s2BaseSize, dBaseSize, ABLayout::MK, ABLayout::KN>(
                    mm1ATensor, mm1BTensor, l0ABuffers, l0BBuffers, mm1ResL0C, param, mm1BScaleTensor);
            }
            Mutex::Unlock<PIPE_M>(L0C_EVENT_0 + l0CBufId);
            Mutex::Lock<PIPE_FIX>(L0C_EVENT_0 + l0CBufId);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + outBufId);

            FixpipeMm1(outputBuf, mm1ResL0C, runInfo);

            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + outBufId);
            Mutex::Unlock<PIPE_FIX>(L0C_EVENT_0 + l0CBufId);

            l0CBufId = (l0CBufId + 1) % L0C_BUF_NUM;
        }
        Mutex::Unlock<PIPE_MTE1>(L1KV_EVENT_0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUF_NUM;
        kvScaleBufId = (kvScaleBufId + 1) % L1_KVSCALE_BUF_NUM;
        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(L1Q_EVENT_0 + qBufId);
            qBufId = (qBufId + 1) % L1_Q_BUF_NUM;
        }
    }

    __aicore__ inline void IterateBmm1PC(MM1_ABUF_T& inputBuf, MM1_RES_BUF_T& outputBuf, uint32_t outBufId,
                                         RunInfoX& runInfo)
    {
        LocalTensor<Q_T> mm1ATensor = l1QBuffers[qBufId].template ReinterpretCast<Q_T>();
        if (runInfo.isFirstS2Loop) {
            // Query V0加载
            uint32_t qBufId = MOD2(runInfo.mloop);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1Q_0 + qBufId);
        }
        // 搬运 key
        LocalTensor<KV_T> mm1BTensor = l1KVBuffers[kvBufId].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(L1KV_EVENT_0 + kvBufId);
        CopyKeyTile(mm1BTensor, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(L1KV_EVENT_0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(L1KV_EVENT_0 + kvBufId);

        MMParam param = {
            (uint32_t)runInfo.actMSize,            // singleM
            (uint32_t)runInfo.actSingleLoopS2Size, // singleN
            (uint32_t)(constInfo.dSize),           // singleK
            false,                                 // isLeftTranspose
            true                                   // isRightTranspose
        };

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT_0 + l0CBufId);

        uint32_t l0CBufOffset = l0CBufId * L0C_SLOT_BYTES;
        LocalTensor<MM_T> mm1ResL0C = l0CBuffers[l0CBufOffset].template ReinterpretCast<MM_T>();
        if constexpr (dBaseSize <= 128) {
            MatmulN<Q_T, KV_T, MM_T, mBaseSize, 256, dBaseSize, ABLayout::MK, ABLayout::KN>(
                mm1ATensor, mm1BTensor, l0ABuffers, l0BBuffers, mm1ResL0C, param);
        }

        SetFlag<HardEvent::MTE1_MTE2>(L1KV_EVENT_0 + kvBufId);
        SetFlag<HardEvent::M_FIX>(L0C_EVENT_0 + l0CBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT_0 + l0CBufId);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + outBufId);
        FixpipeMm1(outputBuf, mm1ResL0C, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + outBufId);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT_0 + l0CBufId);

        // update bufId
        if (unlikely(runInfo.isLastS2Loop)) {
            qBufId = (qBufId + 1) % L1_Q_BUF_NUM;
        }
        kvBufId = (kvBufId + 1) % L1_KV_BUF_NUM;
        l0CBufId = (l0CBufId + 1) % L0C_BUF_NUM;
    }

    __aicore__ inline void IterateBmm2(MM2_ABUF_T& inputBuf, MM2_RES_BUF_T& outputBuf, uint32_t inBufId,
                                       uint32_t outBufId, RunInfoX& runInfo)
    {
        if constexpr (IS_PTG_PCG) {
            IterateBmm2PCG(inputBuf, outputBuf, inBufId, outBufId, runInfo);
        } else if constexpr (IS_PC) {
            IterateBmm2PC(inputBuf, outputBuf, inBufId, outBufId, runInfo);
        }
    }

    __aicore__ inline void IterateBmm2PCG(MM2_ABUF_T& inputBuf, MM2_RES_BUF_T& outputBuf, uint32_t inBufId,
                                          uint32_t outBufId, RunInfoX& runInfo)
    {
        // P V1已加载
        LocalTensor<Q_T> mm2ATensor = inputBuf;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1P_0 + inBufId);
        // 搬运value
        LocalTensor<KV_T> mm2BTensor = l1KVBuffers[kvBufId].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(L1KV_EVENT_0 + kvBufId);
        CopyValueTile(mm2BTensor, runInfo);
        // 搬运vscale
        LocalTensor<KV_SCALE_T> mm2BScaleTensor = l1KVScaleBuffers[kvScaleBufId].template ReinterpretCast<KV_SCALE_T>();
        CopyValueScaleTile(mm2BScaleTensor, runInfo);
        Mutex::Unlock<PIPE_MTE2>(L1KV_EVENT_0 + kvBufId);
        Mutex::Lock<PIPE_MTE1>(L1KV_EVENT_0 + kvBufId);
        {
            MMParam param = {
                (uint32_t)runInfo.actMSize,            // singleM
                (uint32_t)constInfo.dSize,             // singleN
                (uint32_t)runInfo.actSingleLoopS2Size, // singleK
                USE_DN,                                // isLeftTranspose
                false                                  // isRightTranspose
            };
            uint32_t l0CBufOffset = l0CBufId * L0C_SLOT_BYTES;
            LocalTensor<MM_T> mm2ResL0C = l0CBuffers[l0CBufOffset].template ReinterpretCast<MM_T>();
            Mutex::Lock<PIPE_M>(L0C_EVENT_0 + l0CBufId);
            if constexpr (dBaseSize <= 128 && mBaseSize <= 32) {
                MatmulFull<Q_T, KV_T, MM_T, mBaseSize, dBaseSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                    mm2ATensor, mm2BTensor, l0ABuffers, l0BBuffers, mm2ResL0C, param, mm2BScaleTensor);
            } else {
                MatmulK<Q_T, KV_T, MM_T, mBaseSize, dBaseSize, 256, ABLayout::MK, ABLayout::KN>(
                    mm2ATensor, mm2BTensor, l0ABuffers, l0BBuffers, mm2ResL0C, param, mm2BScaleTensor);
            }
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1P_0 + inBufId);
            Mutex::Unlock<PIPE_M>(L0C_EVENT_0 + l0CBufId);
            Mutex::Lock<PIPE_FIX>(L0C_EVENT_0 + l0CBufId);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + outBufId);

            FixpipeMm2(outputBuf, mm2ResL0C, runInfo);

            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + outBufId);
            Mutex::Unlock<PIPE_FIX>(L0C_EVENT_0 + l0CBufId);
            l0CBufId = (l0CBufId + 1) % L0C_BUF_NUM;
        }
        Mutex::Unlock<PIPE_MTE1>(L1KV_EVENT_0 + kvBufId);
        kvBufId = (kvBufId + 1) % L1_KV_BUF_NUM;
        kvScaleBufId = (kvScaleBufId + 1) % L1_KVSCALE_BUF_NUM;
    }

    __aicore__ inline void IterateBmm2PC(MM2_ABUF_T& inputBuf, MM1_RES_BUF_T& outputBuf, uint32_t inBufId,
                                         uint32_t outBufId, RunInfoX& runInfo)
    {
        // P V1已加载
        LocalTensor<Q_T> mm2ATensor = inputBuf;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CC_L1P_0 + inBufId);
        // 搬运value
        LocalTensor<KV_T> mm2BTensor = l1KVBuffers[kvBufId].template ReinterpretCast<KV_T>();
        WaitFlag<HardEvent::MTE1_MTE2>(L1KV_EVENT_0 + kvBufId);
        CopyValueTile(mm2BTensor, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(L1KV_EVENT_0 + kvBufId);
        WaitFlag<HardEvent::MTE2_MTE1>(L1KV_EVENT_0 + kvBufId);

        MMParam param = {
            (uint32_t)runInfo.actMSize,            // singleM
            (uint32_t)constInfo.dSize,             // singleN
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            USE_DN,                                // isLeftTranspose
            false                                  // isRightTranspose
        };
        if constexpr (!USE_DN) {
            param.realM = (uint32_t)runInfo.actMSize;
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT_0 + l0CBufId);
        uint32_t l0CBufOffset = l0CBufId * L0C_SLOT_BYTES;
        LocalTensor<MM_T> mm2ResL0C = l0CBuffers[l0CBufOffset].template ReinterpretCast<MM_T>();
        if constexpr (dBaseSize <= 128) {
            MatmulK<Q_T, KV_T, MM_T, mBaseSize, dBaseSize, 256, ABLayout::MK, ABLayout::KN>(
                mm2ATensor, mm2BTensor, l0ABuffers, l0BBuffers, mm2ResL0C, param);
        }

        SetFlag<HardEvent::MTE1_MTE2>(L1KV_EVENT_0 + kvBufId);
        SetFlag<HardEvent::M_FIX>(L0C_EVENT_0 + l0CBufId);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT_0 + l0CBufId);

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + outBufId);
        FixpipeMm2(outputBuf, mm2ResL0C, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + outBufId);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT_0 + l0CBufId);
        // update bufId
        kvBufId = (kvBufId + 1) % L1_KV_BUF_NUM;
        l0CBufId = (l0CBufId + 1) % L0C_BUF_NUM;
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                      RunInfoX& runInfo)
    {
        if constexpr (USE_DN) { // DN:S2*S1
            FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixpipeParams;
            fixpipeParams.nSize = runInfo.actSingleLoopS2Size;
            fixpipeParams.mSize = Align(runInfo.actMSize, 8U);
            fixpipeParams.srcStride = Align(static_cast<uint32_t>(fixpipeParams.mSize), 16U);
            fixpipeParams.dstStride = Align(runInfo.actMSize, 8U);
            fixpipeParams.dualDstCtl = 1U; // v121无效
            fixpipeParams.params.dnNum = 1U;
            fixpipeParams.params.srcNzMatrixStride = 0U;
            fixpipeParams.params.dstDnMatrixStride = 0U;
            fixpipeParams.params.srcNzC0Stride = 1U;

            Fixpipe<MM_T, MM_T, FIXPIPE_COLUMN_MAJOR_UB_CONFIG>(dstTensor, l0C, fixpipeParams);
        } else {
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = Align(runInfo.actSingleLoopS2Size, 8U);
            fixpipeParams.mSize = runInfo.actMSize;
            fixpipeParams.srcStride = Align(static_cast<uint32_t>(fixpipeParams.mSize), 16U);
            fixpipeParams.dstStride = Align(s2BaseSize, 8U);
            fixpipeParams.dualDstCtl = 1U; // v121无效
            fixpipeParams.params.ndNum = 1U;
            fixpipeParams.params.srcNdStride = 0U;
            fixpipeParams.params.dstNdStride = 0U;
            fixpipeParams.quantPre = QF322F32_PRE;
            float scale = constInfo.scaleValue;
            fixpipeParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&scale));

            Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB_CONFIG>(dstTensor, l0C, fixpipeParams);
        }
    }

    __aicore__ inline void FixpipeMm2(const LocalTensor<MM_T>& dstTensor, const LocalTensor<MM_T>& l0C,
                                      RunInfoX& runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = Align(dBaseSize, 16U);
        fixpipeParams.mSize = runInfo.actMSize;
        fixpipeParams.srcStride = Align(static_cast<uint32_t>(runInfo.actMSize), 16U);
        fixpipeParams.dstStride = Align(dBaseSize, 16U);
        fixpipeParams.dualDstCtl = 1; // V121无效参数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<MM_T, MM_T, FIXPIPE_ROW_MAJOR_UB_CONFIG>(dstTensor, l0C, fixpipeParams);
    }
}; // FAAntiQuantGqaBlockCube
} // namespace BaseApi
#endif // FLASH_ATTENTION_ANTIQUANT_BLOCK_CUBE_H_
