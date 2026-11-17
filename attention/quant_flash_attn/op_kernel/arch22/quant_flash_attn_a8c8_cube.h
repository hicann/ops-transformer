/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_A8_C8_CUBE_H_
#define QUANT_FLASH_ATTN_A8_C8_CUBE_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "quant_flash_attn_a8_c8_kernel_tiling.h"
using namespace AscendC;
#include "attn_infra/arch/fused_arch.hpp"
#include "attn_infra/arch/fused_cross_core_sync.hpp"
#include "attn_infra/arch/fused_resource.hpp"
#include "attn_infra/gemm/block/fused_block_mmad.hpp"
#include "attn_infra/gemm/fused_gemm_dispatch_policy.hpp"
#include "attn_infra/gemm/fused_gemm_type.hpp"
#include "attn_infra/fused_gemm_coord.hpp"
using namespace NpuArch;
#include "kernel_tiling/kernel_tiling.h"
#include "attention/fused_infer_attention_score/op_kernel/arch22/fia_public_define.h"
namespace QuantFlashAttnA8C8 {
struct QfaInt8Params {
    uint32_t hpRows = 8U;
    uint32_t hpVec2Rows = 8U;
    uint32_t hpScratchBytes = 32768U;
    uint32_t hpStateRows = 32U;
    uint64_t keyHeadStride = 0ULL;
    uint64_t valueHeadStride = 0ULL;
    uint64_t qfaKDescaleStrideBlock = 0ULL;
    uint64_t qfaKDescaleStrideHead = 0ULL;
    uint64_t qfaKDescaleStrideSlot = 0ULL;
    uint64_t qfaQDescaleStrideToken = 0ULL;
    uint64_t qfaVDescaleStrideHead = 0ULL;
};

// State holder only; the sole executable QFA policy below is FP32.
struct QfaInt8Policy {
    using Params = QfaInt8Params;
    struct CubeState : QfaInt8Params {
        __gm__ int32_t* cuSeqLengthsQ = nullptr;
    };
    struct VecState : QfaInt8Params {
        using KV_T = int8_t;
        using COMPUTE_T = float;
        bool qfaShortQRoute_ = false;
        GlobalTensor<float> qfaQDescaleGm;
        GlobalTensor<float> qfaKDescaleGm;
        GlobalTensor<float> qfaVDescaleGm;
        GlobalTensor<float> qfaPScaleGm;
        GlobalTensor<int32_t> qfaBlockTableGm;
        uint32_t qfaFdCoreIdx_ = 0U;
        uint32_t qfaFdStrideElems_ = 0U;
        TBuf<TPosition::VECCALC> qfaPreloadBuf;
        TBuf<TPosition::VECCALC> qfaAccBuf;
        TBuf<TPosition::VECCALC> qfaFdLseBuf;
        TBuf<TPosition::VECCALC> qfaPQuantScaleBuf;
        TBuf<TPosition::VECCALC> qfaJointQScaleBuf;
        TBuf<TPosition::VECCALC> qfaJointKScaleBuf;
        LocalTensor<half> qfaJointSOwner_;
        LocalTensor<half> qfaJointSSlots_[2];
        LocalTensor<KV_T> qfaJointPSlots_[2];
        LocalTensor<half> qfaJointOOwner_;
        LocalTensor<COMPUTE_T> qfaJointVec2Owner_;
        LocalTensor<COMPUTE_T> qfaJointAccOwner_;
        LocalTensor<COMPUTE_T> qfaJointTmpOwner_;
        LocalTensor<float> qfaJointKFloat_;
        LocalTensor<half> qfaJointKHalf_[2];
        LocalTensor<KV_T> qfaJointPOwner_;
        bool qfaJointActive_ = false;
        bool qfaJointSFreePending_[2] = {false, false};
        bool qfaJointPFreePending_[2] = {false, false};
        uint32_t qfaJointGroupSeq_ = 0U;
        uint32_t qfaJointPrefetchedPanel_ = ~0U;
        uint32_t qfaJointKPending_ = ~0U;
        uint32_t qfaJointHalfTag_[2] = {~0U, ~0U};
        event_t qfaJointInv_;
        event_t qfaJointKReady_;
        event_t qfaJointKFree_;
        event_t qfaJointOReady_[2];
        event_t qfaJointOFree_[2];
        event_t qfaJointOutDone_;
        TBuf<TPosition::VECCALC> qfaExpHalfBuff;
        uint32_t qfaTaskQdStride_ = 0U;
        uint32_t qfaTaskBtCount_ = 0U;
        float qfaPScaleCache_ = 0.0F;
        float qfaVScaleCache_ = 0.0F;
        bool qfaPScaleLoaded_ = false;
        bool qfaQdCached_ = false;
        event_t qfaScoreFree_[2];
        event_t qfaScoreReady_[2];
        event_t qfaPReady_[2];
        event_t qfaPFree_[2];
        GlobalTensor<half> qfaKScaleDenseGm;
        uint32_t qfaKScaleDenseStride_ = 0U;
        uint32_t qfaKScaleDenseKvHeads_ = 0U;
        SoftMaxTiling qfaSfTilingCache_{};
        uint32_t qfaSfTilingKey0_ = ~0U;
        uint32_t qfaSfTilingKey1_ = ~0U;
        uint32_t qfaSfTilingKey2_ = ~0U;
        static constexpr bool qfaJointRoute_ = true;
        static constexpr bool qfaVec1DoubleBuffer_ = true;
    };
};
struct QfaHostPlanPolicy : QfaInt8Policy {
    static constexpr bool ENABLED = true;
    static constexpr bool HIGH_PRECISION = true;
    static constexpr bool HOST_PLAN = true;
    struct VecState : QfaInt8Policy::VecState {
        uint32_t hpRows = 8U;
        uint32_t hpVec2Rows = 8U;
        uint32_t hpScratchBytes = 32768U;
        uint32_t hpStateRows = 32U;
        LocalTensor<int32_t> hpScoreOwner;
        LocalTensor<int32_t> hpPvOwner;
        bool hpPvActive = false;
        TBuf<TPosition::VECCALC> hpPvBuf;
        event_t hpKReady[2];
        event_t hpKFree[2];
        bool hpKPending[2] = {false, false};
        bool hpKUsed[2] = {false, false};
        uint32_t hpKTag[2] = {~0U, ~0U};
        uint32_t planFdPrefix = 0U;
        uint32_t planFdRow = 0U;
        uint32_t planFdRows = 0U;
        uint32_t planCachedBatch = ~0U;
    };
};

} // namespace QuantFlashAttnA8C8

#include "attention/common/op_kernel/array.h"
#include "attention/common/op_kernel/axis.h"
#include "attention/fused_infer_attention_score/op_kernel/arch22/kernel_common.h"
#include "attention/common/op_kernel/vector_common.h"
// The unchanged FIA header expects these names from its including kernel.
// Supply that context in QFA without altering the FIA implementation.
namespace QuantFlashAttnA8C8 {
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K;
} // namespace QuantFlashAttnA8C8
using QuantFlashAttnA8C8::FLOAT_INF;
using QuantFlashAttnA8C8::BUFFER_SIZE_BYTE_8K;
using QuantFlashAttnA8C8::BUFFER_SIZE_BYTE_16K;
using fa_base_vector::RowMuls;
#include "attention/fused_infer_attention_score/op_kernel/arch22/fia_block_vec_nonquant.h"
using namespace AttentionCommon;
using namespace fa_base_vector;

namespace QuantFlashAttnA8C8 {

using optiling::QuantFlashAttnA8C8TilingData;

// QFA-only INT8 Cube service: one selected format family, raw INT32 output.
// Array/Axis/PA loaders remain the existing common implementations.
constexpr uint32_t QFA_HALF_P_READY = 10U;

__aicore__ inline bool QfaUseHalfP(const RunInfo& info)
{
    return info.actMBaseSize == 128U && info.actualSingleProcessSInnerSize == 512U;
}

// Pack the first half of both AIV row ranges together in the internal P/PV workspace.
__aicore__ inline uint32_t QfaHalfPRow(uint32_t start, uint32_t row)
{
    return start / 2U + (row / 32U) * 64U + row % 32U;
}

template <typename FIAT, bool D128_PANEL_LAYOUT = false, bool HEAD_PAIR = false, uint32_t PAIR_MAX_M = 32U>
class QfaInt8Cube : private QfaHostPlanPolicy::CubeState {
    using QfaPolicy = QfaHostPlanPolicy;
    using T = float;
    using Q_T = int8_t;
    using KV_T = int8_t;
    using MM_OUT_T = int32_t;
    using MM1_GM_T = int32_t;
    using MM2_GM_T = int32_t;
    static constexpr bool QFA_INT8 = true;
    static constexpr bool TENSOR_LIST = false;
    static constexpr FIA_LAYOUT LAYOUT_T = FIA_LAYOUT::TND;
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVFormat<FIAT::kvLayout, true>();
    struct CFG {
        static constexpr bool ENABLE_UNIFLAG = false;
        static constexpr uint32_t PRELOAD_NUM = 2U;
        static constexpr bool S2_BASICSIZE_IS_1024 = false;
        static constexpr bool ENABLE_V1C2_TILE_SYNC = false;
    };
    struct BufSnapshot {
        uint64_t signature = static_cast<uint64_t>(-1);
        uint32_t firstBufId;
        uint32_t bufCnt = 0U;
    };
    __aicore__ inline static constexpr bool QfaUseK128()
    {
        return true;
    }
    /**
     * @pre alignment必须是2的幂，且s>=0
     */
    template <typename T1>
    static __aicore__ inline constexpr T1 AlignToPowerOfTwo(T1 s, T1 alignment)
    {
        if constexpr (IsSameType<T1, uint64_t>::value || IsSameType<T1, uint32_t>::value ||
                      IsSameType<T1, uint16_t>::value || IsSameType<T1, uint8_t>::value) {
            return (s + alignment - 1) & (~(alignment - 1));
        } else {
            return uint64_t(s + alignment - 1) & (~uint64_t(alignment - 1));
        }
    }

    template <typename T1>
    static __aicore__ inline constexpr size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T1, int4b_t>::value) {
            return AlignToPowerOfTwo(s, 64);
        } else {
            return AlignToPowerOfTwo(s, ONE_BLK_SIZE / sizeof(T1));
        }
    }

    template <typename T1>
    __aicore__ inline constexpr uint32_t GetC0Num()
    {
        if constexpr (IsSameType<T1, int4b_t>::value) {
            return 64;
        } else {
            return ONE_BLK_SIZE / sizeof(T1);
        }
    }

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_Q_SIZE =
        HEAD_PAIR ? PAIR_MAX_M * 128U : (D128_PANEL_LAYOUT ? 128U * 128U : 256U * 192U);

    static constexpr uint32_t L1_V_SIZE = 256U * 256U;
    static constexpr uint32_t L1_KP_SIZE = HEAD_PAIR ? 512U * 256U : (D128_PANEL_LAYOUT ? 512U * 128U : 128U * 256U);
    static constexpr uint32_t L1_P_SIZE = HEAD_PAIR ? PAIR_MAX_M * 256U : 128U * 256U;
    static constexpr uint32_t L1_P_BUFCNT = HEAD_PAIR || D128_PANEL_LAYOUT ? 2U : 0U;

    static constexpr uint32_t L1_Q_BUFCNT = D128_PANEL_LAYOUT ? 1U : 2U;
    static constexpr uint32_t L1_V_BUFCNT = 2;
    static constexpr uint32_t L1_KP_BUFCNT = HEAD_PAIR || D128_PANEL_LAYOUT ? 2U : 3U;

    static_assert((L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T) + L1_V_SIZE * L1_V_BUFCNT * sizeof(KV_T) +
                   (L1_KP_SIZE * L1_KP_BUFCNT + L1_P_SIZE * L1_P_BUFCNT) * sizeof(KV_T)) <= 512U * 1024U,
                  "QFA L1 budget exceeded (Q + V + KP pools)");

    Array<LocalTensor<Q_T>, L1_Q_BUFCNT, L1_Q_SIZE> qL1Tensor;
    Array<LocalTensor<KV_T>, L1_V_BUFCNT, L1_V_SIZE> vL1Tensor;
    Array<LocalTensor<KV_T>, L1_KP_BUFCNT, L1_KP_SIZE> kpL1Tensor;

    BufSnapshot qL1Snapshot;
    BufSnapshot vL1Snapshot;

    uint32_t qL1BufId = 0;
    uint32_t vL1BufId = 0;
    uint32_t kpL1BufId = 0;

    TBuf<TPosition::A1> L1_Q;
    TBuf<TPosition::A1> L1_V;
    TBuf<TPosition::A1> L1_KP;

    // =================================L0 Buffer=================================
    static constexpr uint32_t L0A_PP_SIZE = 128 * 128; // 32KB
    static constexpr uint32_t L0B_PP_SIZE = 128 * 256;
    static constexpr uint32_t L0C_PP_SIZE = 128 * 128; // 64KB
    static_assert(!QFA_INT8 || (L0A_PP_SIZE >= 128U * 128U && L0B_PP_SIZE >= 128U * 256U),
                  "QFA K128 must fit each existing L0A/L0B slot");

    static constexpr uint32_t L0AB_BUFCNT = 2;
    static constexpr uint32_t L0C_BUFCNT = 2;
    static_assert(L0A_PP_SIZE * (HEAD_PAIR ? 4U : (D128_PANEL_LAYOUT ? 3U : 2U)) * sizeof(Q_T) <= 64U * 1024U,
                  "QFA L0A budget exceeded");

    Array<LocalTensor<Q_T>, L0AB_BUFCNT, L0A_PP_SIZE> aL0Tensor;
    Array<LocalTensor<KV_T>, L0AB_BUFCNT, L0B_PP_SIZE> bL0Tensor;
    Array<LocalTensor<MM_OUT_T>, L0C_BUFCNT, L0C_PP_SIZE> cL0Tensor;

    uint32_t pairHead_ = 0U;
    uint64_t pairQSignature_[2] = {~0ULL, ~0ULL};
    uint32_t abL0BufId = 0;
    uint32_t cL0BufId = 0;

    TBuf<TPosition::A2> L0_A;
    TBuf<TPosition::B2> L0_B;
    TBuf<TPosition::CO1> L0_C;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1
    static constexpr uint32_t Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t V_EVENT0 = Q_EVENT0 + L1_Q_BUFCNT;
    static constexpr uint32_t KP_EVENT0 = V_EVENT0 + L1_V_BUFCNT;
    static constexpr uint32_t P_EVENT0 = KP_EVENT0 + L1_KP_BUFCNT;
    static_assert(P_EVENT0 + L1_P_BUFCNT <= 8U, "QFA L1 events exceeded");

    // m <> mte1
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0_READY_EVENT = EVENT_ID7;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    // =================================params=================================
    FaGmTensor<Q_T, Q_FORMAT> queryGmTensor;
    FaGmTensor<Q_T, Q_FORMAT> queryRopeGmTensor;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;

    FaGmTensor<KV_T, KV_FORMAT> keyGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> keyRopeGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> valueGmTensor;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    ConstInfo constInfo{};

    // key和value的TensorList原始地址
    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> kRopeGm;
    Array<GlobalTensor<MM1_GM_T>, CFG::PRELOAD_NUM> mm1ResGm;

    // mm2
    Array<GlobalTensor<KV_T>, CFG::PRELOAD_NUM> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    Array<GlobalTensor<MM2_GM_T>, CFG::PRELOAD_NUM> mm2ResGm;

    __gm__ uint8_t* actualSequenceLengthsQ = nullptr;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    __gm__ uint8_t* actualSeqLengths = nullptr;

    // page attention
    GlobalTensor<int32_t> blockTableGm;

    // Load L1->L0 params
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {false, false};
    uint32_t load3DL1SizeCfg = 0;
    LoadData3DParamsV2<Q_T> loadData3DParams;
    LoadData2DParams mm1LoadDataBTransposeToL0Params;
    LoadData2DParams mm2LoadDataBToL0Params;

public:
    static constexpr bool PAIRED_HEAD_SERVICE = HEAD_PAIR;
    static constexpr bool HOST_PLAN = true;
    static constexpr bool OPTIMIZED_SERVICE = false;
    __aicore__ inline QfaHostPlanPolicy::CubeState& QfaState()
    {
        return static_cast<QfaHostPlanPolicy::CubeState&>(*this);
    }
    __aicore__ inline void InitQfaParams(const QfaInt8Params& params)
    {
        QfaState().keyHeadStride = params.keyHeadStride;
        QfaState().valueHeadStride = params.valueHeadStride;
        QfaState().qfaKDescaleStrideBlock = params.qfaKDescaleStrideBlock;
        QfaState().qfaKDescaleStrideHead = params.qfaKDescaleStrideHead;
        QfaState().qfaKDescaleStrideSlot = params.qfaKDescaleStrideSlot;
        QfaState().qfaQDescaleStrideToken = params.qfaQDescaleStrideToken;
        QfaState().qfaVDescaleStrideHead = params.qfaVDescaleStrideHead;
    }
    __aicore__ inline void Init(
        __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
        __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ, __gm__ uint8_t* actualSeqLengths,
        __gm__ uint8_t* deqScale1, __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2, __gm__ uint8_t* quantScale2,
        __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
        __gm__ uint8_t* blockTable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
        __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset, __gm__ uint8_t* valueAntiquantScale,
        __gm__ uint8_t* valueAntiquantOffset, __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix,
        __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
        __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse)
    {
        QfaState().cuSeqLengthsQ = reinterpret_cast<__gm__ int32_t*>(actualSeqLengthsQ);
        this->actualSequenceLengthsQ = actualSeqLengthsQ;
        blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        const uint32_t qkTensorD =
            constInfo.ropeSplitMode ? constInfo.headDim : constInfo.headDim + constInfo.headDimRope;
        queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(query));
        queryGmTensor.gmTensor = queryGm;
        keyPtr = key;
        valuePtr = value;
        keyGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(value));
        if (constInfo.l2CacheOffFlag) {
#ifndef ASCENDC_OOM
            keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        }
        keyGmTensor.gmTensor = keyGm;
        valueGmTensor.gmTensor = valueGm;
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, qkTensorD, blockTableGm,
                                              constInfo.maxBlockNumPerBatch, constInfo.keyBnStride,
                                              QfaState().keyHeadStride);
            valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, constInfo.headDim,
                                                blockTableGm, constInfo.maxBlockNumPerBatch, constInfo.valueBnStride,
                                                QfaState().valueHeadStride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            constexpr uint32_t d0 = 32U;
            const uint32_t d1 = qkTensorD / d0;
            keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1, d0, blockTableGm,
                                              constInfo.maxBlockNumPerBatch, constInfo.keyBnStride,
                                              QfaState().keyHeadStride);
            valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, constInfo.headDim / d0,
                                                d0, blockTableGm, constInfo.maxBlockNumPerBatch,
                                                constInfo.valueBnStride, QfaState().valueHeadStride);
        }
    }

    __aicore__ inline void InitParams(const ConstInfo& constInfo)
    {
        this->constInfo = constInfo;

        loadData3DParams.l1W = QfaUseK128() ? 16U : GetC0Num<Q_T>();
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mStartPt = 0;
        loadData3DParams.kStartPt = 0;
        loadData3DParams.strideW = 1;
        loadData3DParams.strideH = 1;
        loadData3DParams.filterW = 1;
        loadData3DParams.filterSizeW = 0;
        loadData3DParams.filterH = 1;
        loadData3DParams.filterSizeH = 0;
        loadData3DParams.dilationFilterW = 1;
        loadData3DParams.dilationFilterH = 1;
        loadData3DParams.enTranspose = 0;
        loadData3DParams.fMatrixCtrl = 0;

        mm1LoadDataBTransposeToL0Params.startIndex = 0;
        mm1LoadDataBTransposeToL0Params.srcStride = 1;
        mm1LoadDataBTransposeToL0Params.dstGap = 0;
        mm1LoadDataBTransposeToL0Params.ifTranspose = false;

        mm2LoadDataBToL0Params.startIndex = 0;
        mm2LoadDataBToL0Params.repeatTimes = constInfo.headDim / GetC0Num<Q_T>(); // 列（N轴）方向上的分形数
        mm2LoadDataBToL0Params.dstGap = 0;
        mm2LoadDataBToL0Params.ifTranspose = true; // 将小Z格式转置为小N
    }

    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<MM1_GM_T>& mm1ResGm)
    {
        // mm1
        this->mm1ResGm.Init(mm1ResGm, constInfo.mmResUbSize);
    }

    __aicore__ inline void InitMm2GlobalTensor(const GlobalTensor<KV_T>& vec1ResGm,
                                               const GlobalTensor<MM2_GM_T>& mm2ResGm)
    {
        // mm2
        this->vec1ResGm.Init(vec1ResGm, constInfo.vec1ResUbSize);
        this->mm2ResGm.Init(mm2ResGm, constInfo.bmm2ResUbSize);
    }

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        // L1 Q
        pipe->InitBuffer(L1_Q, L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T));
        qL1Tensor.Init(L1_Q.Get<Q_T>());

        // L1 V
        pipe->InitBuffer(L1_V, L1_V_SIZE * L1_V_BUFCNT * sizeof(KV_T));
        vL1Tensor.Init(L1_V.Get<KV_T>());

        // L1 KP
        pipe->InitBuffer(L1_KP, (L1_KP_SIZE * L1_KP_BUFCNT + L1_P_SIZE * L1_P_BUFCNT) * sizeof(KV_T));
        kpL1Tensor.Init(L1_KP.Get<KV_T>());

        // L0A
        pipe->InitBuffer(L0_A, L0A_PP_SIZE * (HEAD_PAIR ? 4U : (D128_PANEL_LAYOUT ? 3U : 2U)) * sizeof(Q_T));
        aL0Tensor.Init(L0_A.Get<Q_T>());

        // L0B
        pipe->InitBuffer(L0_B, L0B_PP_SIZE * 2 * sizeof(KV_T));
        bL0Tensor.Init(L0_B.Get<KV_T>());

        // L0C
        pipe->InitBuffer(L0_C, L0C_PP_SIZE * 2 * sizeof(MM_OUT_T));
        cL0Tensor.Init(L0_C.Get<MM_OUT_T>());
    }

    __aicore__ inline void AllocEventID()
    {
        for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
            SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
        }
        for (uint32_t i = 0; i < L1_P_BUFCNT; ++i)
            SetFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + i);

        for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
            SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
        }

        SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
        SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        for (uint32_t i = 0; i < L1_KP_BUFCNT; ++i) {
            WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + i);
        }
        for (uint32_t i = 0; i < L1_P_BUFCNT; ++i)
            WaitFlag<HardEvent::MTE1_MTE2>(P_EVENT0 + i);

        for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
            WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + i);
        }

        WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }

    __aicore__ inline void CopyQToL1(uint32_t dstBufId, const RunInfo& info, uint32_t subMStart, uint32_t subMSize)
    {
        auto qL1Tensor = this->qL1Tensor[dstBufId];
        auto subMSizeAlign = AlignToPowerOfTwo<uint32_t>(
            subMSize,
            (uint32_t)
                BLOCK_CUBE); // 目的矩阵z分形的高固定为16，行为C0_SIZE(字节数)。目的矩阵的“高”将对齐为z分形的整数倍，所以将srcN按16对齐即为Z型矩阵相邻行起始地址之间的偏移
        uint32_t nopeDealSize =
            constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);

        FaL1Tensor<Q_T, L1Format::NZ> dstTensor{.tensor = qL1Tensor, .rowCount = subMSizeAlign};

        const uint32_t gSize = static_cast<uint32_t>(constInfo.gSize);
        const uint32_t c0 = GetC0Num<Q_T>();
        uint32_t row = 0U;
        while (row < subMSize) {
            const uint32_t mergedIdx = info.gS1Idx + subMStart + row;
            const uint32_t s1Idx = mergedIdx / gSize;
            const uint32_t gIdx = mergedIdx % gSize;
            uint32_t copyRows = gSize - gIdx;
            if (copyRows > subMSize - row) {
                copyRows = subMSize - row;
            }
            const uint64_t tIdx = static_cast<uint64_t>(QfaState().cuSeqLengthsQ[info.bIdx]) + s1Idx;
            const uint64_t qOffset = ((tIdx * constInfo.kvHeadNum + info.n2Idx) * gSize + gIdx) * nopeDealSize;
            CopySingleMatrixNDToNZ(dstTensor.tensor[c0 * row], queryGm[qOffset], copyRows, nopeDealSize, nopeDealSize,
                                   dstTensor.rowCount);
            row += copyRows;
        }
        return;
    }

    __aicore__ inline void CopyHeadPair(LocalTensor<KV_T> dst, GlobalTensor<KV_T> src, const RunInfo& info,
                                        uint32_t start, uint32_t count, uint64_t blockStride, uint64_t headStride,
                                        uint64_t tokenStride)
    {
        const uint32_t pitch = (count + 31U) & ~31U;
        for (uint32_t copied = 0U; copied < count;) {
            const uint32_t token = info.s2BatchOffset + start + copied;
            const uint32_t page = token / constInfo.kvCacheBlockSize;
            const uint32_t slot = token % constInfo.kvCacheBlockSize;
            const uint32_t rows =
                count - copied < constInfo.kvCacheBlockSize - slot ? count - copied : constInfo.kvCacheBlockSize - slot;
            const int32_t physical = blockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + page);
            const uint64_t offset =
                static_cast<uint64_t>(physical) * blockStride + (info.n2Idx & ~1U) * headStride + slot * tokenStride;
            CopySingleMatrixNDToNZ(dst[copied * 32U], src[offset], rows, 256U, tokenStride, pitch);
            copied += rows;
        }
    }

    __aicore__ inline void CopyKToL1(uint32_t dstBufId, const RunInfo& info, uint32_t subNStart, uint32_t subNSize)
    {
        auto kpL1Tensor = this->kpL1Tensor[dstBufId];
        if constexpr (HEAD_PAIR) {
            CopyHeadPair(kpL1Tensor, keyGm, info, subNStart, subNSize, constInfo.keyBnStride, QfaState().keyHeadStride,
                         constInfo.kvHeadNum * QfaState().keyHeadStride);
            return;
        }
        auto subNSizeAlign = AlignToPowerOfTwo(subNSize, GetC0Num<KV_T>());
        uint32_t nopeDealSize =
            constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);

        FaL1Tensor<KV_T, L1Format::NZ> dstTensor{.tensor = kpL1Tensor, .rowCount = subNSizeAlign};
        GmKvCoord gmCoord{.bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                          .n2Idx = info.n2Idx,
                          .s2Idx = (QFA_INT8 ? info.s2BatchOffset : info.s2Idx * constInfo.s2BaseSize) + subNStart,
                          .dIdx = 0,
                          .s2DealSize = subNSize,
                          .dDealSize = nopeDealSize};
        if constexpr (QFA_INT8 && KV_FORMAT == GmFormat::PA_NZ) {
            gmCoord.dDealSize = AlignToPowerOfTwo(gmCoord.dDealSize, GetC0Num<KV_T>());
        }
        copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
    }

    __aicore__ inline LocalTensor<KV_T> PL1(uint32_t slot)
    {
        if constexpr (D128_PANEL_LAYOUT || HEAD_PAIR)
            return L1_KP.Get<KV_T>()[L1_KP_SIZE * L1_KP_BUFCNT + slot * L1_P_SIZE];
        else
            return kpL1Tensor[slot];
    }

    template <CubeFormat Format>
    __aicore__ inline void CopyPToL1(uint32_t dstBufId, const RunInfo& info, uint32_t gmStride, uint32_t subMStart,
                                     uint32_t subMSize, uint32_t subKStart, uint32_t subKSize)
    {
        auto dstL1 = PL1(dstBufId);
        if constexpr (Format == CubeFormat::NZ) {
            auto srcGm =
                this->vec1ResGm[info.loop % CFG::PRELOAD_NUM][gmStride * subKStart + GetC0Num<KV_T>() * subMStart];
            uint32_t blockElementCnt = 32 / sizeof(T);
            DataCopyParams intriParams;
            intriParams.blockCount = subKSize / blockElementCnt;
            intriParams.blockLen = subMSize;
            intriParams.dstStride = 0;
            intriParams.srcStride = gmStride;
            DataCopy(dstL1, srcGm, intriParams);
        } else {
            auto srcGm = this->vec1ResGm[info.loop % CFG::PRELOAD_NUM]
                                        [(HEAD_PAIR ? pairHead_ * constInfo.vec1ResUbSize / 2U : 0U) +
                                         gmStride * subMStart + subKStart];
            auto subMSizeAlign = AlignToPowerOfTwo(subMSize, (uint32_t)BLOCK_CUBE);
            CopySingleMatrixNDToNZ(dstL1, srcGm, subMSize, subKSize, gmStride, subMSizeAlign);
        }
    }

    __aicore__ inline void CopyVToL1(uint32_t dstBufId, const RunInfo& info, uint32_t subKStart, uint32_t subKSize)
    {
        auto vL1Tensor = this->vL1Tensor[dstBufId];
        if constexpr (HEAD_PAIR) {
            CopyHeadPair(vL1Tensor, valueGm, info, subKStart, subKSize, constInfo.valueBnStride,
                         QfaState().valueHeadStride, constInfo.kvHeadNum * QfaState().valueHeadStride);
            return;
        }
        auto subKSizeAlign = AlignToPowerOfTwo(subKSize, GetC0Num<KV_T>());

        FaL1Tensor<KV_T, L1Format::NZ> dstTensor{.tensor = vL1Tensor, .rowCount = subKSizeAlign};
        GmKvCoord gmCoord{.bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                          .n2Idx = info.n2Idx,
                          .s2Idx = (QFA_INT8 ? info.s2BatchOffset : info.s2Idx * constInfo.s2BaseSize) + subKStart,
                          .dIdx = 0,
                          .s2DealSize = subKSize,
                          .dDealSize = (uint32_t)constInfo.headDim};
        if constexpr (QFA_INT8 && KV_FORMAT == GmFormat::PA_NZ) {
            gmCoord.dDealSize = AlignToPowerOfTwo(gmCoord.dDealSize, GetC0Num<KV_T>());
        }
        copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
    }

    template <uint32_t M_L1_SIZE, bool FORCE = false>
    __aicore__ inline void ResetLoad3DConfig()
    {
        if constexpr (!FORCE) {
            if (this->load3DL1SizeCfg == M_L1_SIZE) {
                return;
            }
        }

        this->load3DL1SizeCfg = M_L1_SIZE;
        constexpr uint8_t padList[] = {0, 0, 0, 255};
        SetFmatrix(M_L1_SIZE / BLOCK_CUBE, QfaUseK128() ? 16U : GetC0Num<Q_T>(), padList, FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue<Q_T>(0);
    }

    template <uint32_t M_L1_SPLIT_Size, bool RESIDENT_Q = false, typename T1>
    __aicore__ inline void LoadAToL0(uint32_t dstBufId, const LocalTensor<T1>& l1Tensor, uint32_t mL1Size,
                                     uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize)
    {
        auto srcTensor = l1Tensor[mL1Size * subKStart][GetC0Num<T1>() * subMStart];
        auto dstTensor = this->aL0Tensor[dstBufId];
        if constexpr (RESIDENT_Q)
            dstTensor = L0_A.Get<Q_T>()[L0A_PP_SIZE * (L0AB_BUFCNT + (HEAD_PAIR ? pairHead_ : 0U))];
        const uint32_t loadK = QfaUseK128() ? AlignToPowerOfTwo(subKSize, GetC0Num<T1>()) : subKSize;
        loadData3DParams.mExtension = subMSize;
        loadData3DParams.kExtension = loadK;
        loadData3DParams.channelSize = loadK;

        if (likely(mL1Size == M_L1_SPLIT_Size)) {
            LoadData<T1, LOAD3DV2_CONFIG>(dstTensor, srcTensor, loadData3DParams);
        } else {
            loadData3DParams.l1H = mL1Size / BLOCK_CUBE;
            LoadData(dstTensor, srcTensor, loadData3DParams);

            ResetLoad3DConfig<M_L1_SPLIT_Size, true>();
        }
    }

    __aicore__ inline void LoadBTransposeToL0(uint32_t dstBufId, const LocalTensor<KV_T>& l1Tensor, uint32_t nL1Size,
                                              uint32_t subNStart, uint32_t subNSize, uint32_t subKStart,
                                              uint32_t subKSize)
    {
        auto srcTensor = l1Tensor[nL1Size * subKStart][GetC0Num<KV_T>() * subNStart];
        auto dstTensor = this->bL0Tensor[dstBufId];
        if (nL1Size == subNSize) {
            mm1LoadDataBTransposeToL0Params.repeatTimes = AttentionCommon::CeilDiv(subKSize, (uint32_t)BLOCK_CUBE) *
                                                          AttentionCommon::CeilDiv(subNSize, GetC0Num<KV_T>());
            LoadData(dstTensor, srcTensor, mm1LoadDataBTransposeToL0Params);
        } else {
            mm1LoadDataBTransposeToL0Params.repeatTimes = AttentionCommon::CeilDiv(subNSize, (uint32_t)BLOCK_CUBE);
            uint32_t kLoops = subKSize / GetC0Num<KV_T>();
            for (uint32_t i = 0; i < kLoops; i++) {
                LoadData(dstTensor[subNSize * i * (QFA_INT8 ? GetC0Num<KV_T>() : BLOCK_CUBE)],
                         srcTensor[nL1Size * i * GetC0Num<KV_T>()], mm1LoadDataBTransposeToL0Params);
            }
        }
    }

    __aicore__ inline void LoadBToL0(uint32_t dstBufId, const LocalTensor<KV_T>& l1Tensor, uint32_t kL1Size,
                                     uint32_t subKStart, uint32_t subKSize, uint32_t subNStart, uint32_t subNSize)
    {
        mm2LoadDataBToL0Params.srcStride = kL1Size / BLOCK_CUBE;

        auto srcTensor = l1Tensor[kL1Size * subNStart][GetC0Num<KV_T>() * subKStart];
        auto dstTensor = this->bL0Tensor[dstBufId];

        constexpr uint32_t C0 = ONE_BLK_SIZE / sizeof(KV_T);

        const uint32_t nAligned = AttentionCommon::CeilDiv(subNSize, C0) * C0;
        LoadData2dTransposeParams p;
        p.startIndex = 0U;
        p.repeatTimes = nAligned / C0;
        p.srcStride = kL1Size / C0;
        p.dstGap = 1U;
        p.dstFracGap = 0U;
        const uint32_t kBlocks = AttentionCommon::CeilDiv(subKSize, C0);
        for (uint32_t ki = 0U; ki < kBlocks; ++ki) {
            LoadDataWithTranspose(dstTensor[ki * C0 * nAligned], srcTensor[ki * C0 * C0], p);
        }
        return;
    }

    template <CubeFormat GMFormat, typename DstT>
    __aicore__ inline void FixpipeCToGM(GlobalTensor<DstT>& mmResGm, uint32_t cL0BufId, uint32_t dstStride,
                                        uint32_t subMStart, uint32_t subMSize, uint32_t subNStart, uint32_t subNSize,
                                        uint64_t deqScalarBits = 0x3B800000)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = subNSize;
        fixParams.mSize = subMSize;
        fixParams.srcStride = AlignToPowerOfTwo(subMSize, (uint32_t)BLOCK_CUBE);
        fixParams.ndNum = 1;
        if constexpr (IsSameType<DstT, half>::value && IsSameType<MM_OUT_T, int32_t>::value) {
            fixParams.quantPre = QuantMode_t::DEQF16;
            fixParams.deqScalar = deqScalarBits;
        } else if constexpr (IsSameType<DstT, half>::value && IsSameType<MM_OUT_T, float>::value) {
            fixParams.quantPre = QuantMode_t::F322F16;
        }
        if constexpr (CFG::ENABLE_UNIFLAG) {
            fixParams.unitFlag = 3;
        }

        auto cL0Tensor = this->cL0Tensor[cL0BufId];
        if constexpr (GMFormat == CubeFormat::NZ) {
            ASCENDC_ASSERT(subNSize % BLOCK_CUBE == 0, {
                KERNEL_LOG(KERNEL_ERROR, "subNSize must be divisible by 16, current value is %u", subNSize);
            });
            fixParams.dstStride = dstStride * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE;
            auto dstTensor = mmResGm[dstStride * subNStart][BLOCK_CUBE * subMStart];
            if constexpr (QFA_INT8) {
                Fixpipe<DstT, MM_OUT_T, CFG_NZ>(dstTensor, cL0Tensor, fixParams);
            } else {
                Fixpipe<MM_OUT_T, T, CFG_NZ>(dstTensor, cL0Tensor, fixParams);
            }
        } else {
            fixParams.dstStride = dstStride;
            auto dstTensor = mmResGm[dstStride * subMStart + subNStart];
            if constexpr (QFA_INT8) {
                Fixpipe<DstT, MM_OUT_T, CFG_ROW_MAJOR>(dstTensor, cL0Tensor, fixParams);
            } else {
                Fixpipe<MM_OUT_T, T, CFG_ROW_MAJOR>(dstTensor, cL0Tensor, fixParams);
            }
        }
    }

    template <CubeFormat OutFormat = CubeFormat::ND, CubeFormat BFormat = CubeFormat::ND>
    __aicore__ inline void ComputeMm1(const RunInfo& info)
    {
        if constexpr (HEAD_PAIR) {
            for (uint32_t head = 0U; head < 2U; ++head) {
                pairHead_ = head;
                RunInfo part = info;
                part.n2Idx += head;
                ComputeMm1Head(part);
            }
        } else {
            ComputeMm1Head(info);
        }
    }

    template <CubeFormat OutFormat = CubeFormat::ND, CubeFormat BFormat = CubeFormat::ND>
    __aicore__ inline void ComputeMm1Head(const RunInfo& info)
    {
        Axis m(info.actMBaseSize);
        Axis n(info.actualSingleProcessSInnerSize);
        Axis k(constInfo.headDim + constInfo.headDimRope);
        auto mmResGm =
            this->mm1ResGm[info.loop % CFG::PRELOAD_NUM][HEAD_PAIR ? pairHead_ * constInfo.mmResUbSize / 2U : 0U];

        constexpr uint32_t M_SPLIT_SIZE = CFG::S2_BASICSIZE_IS_1024 ? 128 : 256;
        const uint32_t N_SPLIT_SIZE = D128_PANEL_LAYOUT || HEAD_PAIR ? 512U : ((k.sizeAct <= 128) ? 256U : 128U);

        constexpr uint32_t M_BASE = 128;
        const uint32_t K_BASE = QfaUseK128() ? 128U : (QFA_INT8 ? 32U : 128U);
        const uint32_t N_BASE = (QFA_INT8 && m.sizeAct <= 64U) ? 256U : 128U;

        ResetLoad3DConfig<M_SPLIT_SIZE>();

        auto mSlices = m.Split(M_SPLIT_SIZE);
        auto kL0Slices = k.Split(K_BASE);

        bool canFullLoadQ = (mSlices.size() <= L1_Q_BUFCNT);
        uint64_t qCoord = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | ((uint64_t)info.gS1Idx);
        bool reuseQBuf = canFullLoadQ && (qL1Snapshot.signature == qCoord);
        if (!reuseQBuf) {
            qL1Snapshot.bufCnt = 0;
            qL1Snapshot.firstBufId = this->qL1BufId;
            qL1Snapshot.signature = qCoord;
        }

        if constexpr (D128_PANEL_LAYOUT || HEAD_PAIR)
            this->kpL1BufId = info.loop & 1U;
        if constexpr (HEAD_PAIR) {
            reuseQBuf = pairQSignature_[pairHead_] == qCoord;
            qL1Snapshot.bufCnt = reuseQBuf ? 1U : 0U;
            qL1Snapshot.firstBufId = pairHead_;
            this->qL1BufId = pairHead_;
        }
        bool residentQLoaded = reuseQBuf;
        for (auto& nL1 : n.Split(N_SPLIT_SIZE)) {
            if (!HEAD_PAIR || pairHead_ == 0U) {
                WaitFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
                CopyKToL1(this->kpL1BufId, info, nL1.start, nL1.sizeAct);
                SetFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
                WaitFlag<HardEvent::MTE2_MTE1>(KP_EVENT0 + this->kpL1BufId);
            }

            int32_t mL1Id = -1;
            for (auto& mL1 : mSlices) {
                ++mL1Id;

                if (unlikely(mL1Id >= qL1Snapshot.bufCnt)) {
                    reuseQBuf = false;
                }

                uint32_t qBufId;
                if (unlikely(!reuseQBuf)) {
                    qBufId = this->qL1BufId;
                    // 在需要搬入Q前才去Set MTE1->MTE2事件，而不是在L0算完后就去Set，是考虑到Q_L1
                    // buf的生命周期可能跨越多轮MM1计算， 如果前一次MM1计算还未完成，还在复用Q_L1
                    // buf，后一次MM1计算就开始搬运，就会覆盖了前一次计算的数据
                    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + this->qL1BufId);
                    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId);
                    CopyQToL1(qBufId, info, mL1.start, mL1.sizeAct);

                    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId);
                    qL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadQ); // 不能全载Q时，不缓存到快照中
                } else {
                    qBufId = (qL1Snapshot.firstBufId + mL1Id) % L1_Q_BUFCNT;
                }

                auto nL0Slices = nL1.Split(N_BASE);
                for (auto& mL0 : mL1.Split(M_BASE)) {
                    for (auto& nL0 : nL0Slices) {
                        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

                        for (auto& kL0 : kL0Slices) {
                            WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                            constexpr uint32_t C0 = ONE_BLK_SIZE / sizeof(KV_T);
                            const uint32_t kL0LoadSize = AlignToPowerOfTwo(kL0.sizeAct, C0);

                            if constexpr (D128_PANEL_LAYOUT || HEAD_PAIR) {
                                if (!residentQLoaded) {
                                    // A new task may overwrite Q only after preceding MMAD readers finish.
                                    SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
                                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
                                    LoadAToL0<M_SPLIT_SIZE, true>(this->abL0BufId, qL1Tensor[qBufId], mL1.AlignedSize(),
                                                                  mL0.start, mL0.AlignedSize(), kL0.start, kL0LoadSize);
                                    residentQLoaded = true;
                                }
                            } else {
                                LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, qL1Tensor[qBufId], mL1.AlignedSize(),
                                                        mL0.start, mL0.AlignedSize(), kL0.start, kL0LoadSize);
                            }
                            LoadBTransposeToL0(this->abL0BufId, kpL1Tensor[this->kpL1BufId], nL1.AlignedSize<C0>(),
                                               nL0.start, QfaUseK128() ? nL0.AlignedSize() : nL0.AlignedSize<C0>(),
                                               kL0.start + (HEAD_PAIR ? pairHead_ * 128U : 0U), kL0LoadSize);

                            SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                            WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                            bool isLastK = kL0.IsTailOf(k);
                            MmadParams mmadParams;
                            mmadParams.m = mL0.AlignedSize();
                            mmadParams.n = QfaUseK128() ? nL0.AlignedSize() : nL0.sizeAct;
                            mmadParams.k = kL0.sizeAct;
                            mmadParams.cmatrixInitVal = (kL0.start == 0);
                            mmadParams.cmatrixSource = false;
                            if constexpr (CFG::ENABLE_UNIFLAG) {
                                mmadParams.unitFlag = isLastK ? 3 : 2;
                            }
                            auto qOperand = aL0Tensor[this->abL0BufId];
                            if constexpr (D128_PANEL_LAYOUT || HEAD_PAIR)
                                qOperand = L0_A.Get<Q_T>()[L0A_PP_SIZE * (L0AB_BUFCNT + (HEAD_PAIR ? pairHead_ : 0U))];
                            Mmad(cL0Tensor[this->cL0BufId], qOperand, bL0Tensor[this->abL0BufId], mmadParams);
                            if (likely(!isLastK)) {
                                AscendC::PipeBarrier<PIPE_M>();
                            }
                            SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                            this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
                        }
                        if constexpr (!CFG::ENABLE_UNIFLAG) {
                            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                        }
                        const uint32_t dstStride =
                            (OutFormat == CubeFormat::NZ) ? m.sizeAct : info.actualSingleProcessSInnerSizeAlign;
                        const uint32_t dstNSize = (OutFormat == CubeFormat::NZ) ? nL0.AlignedSize() : nL0.sizeAct;
                        if constexpr (QFA_INT8) {
                            FixpipeCToGM<OutFormat>(mmResGm, this->cL0BufId, dstStride, mL1.start + mL0.start,
                                                    mL0.sizeAct, nL1.start + nL0.start, dstNSize);

                        } else {
                            FixpipeCToGM<OutFormat>(mmResGm, this->cL0BufId, dstStride, mL1.start + mL0.start,
                                                    mL0.sizeAct, nL1.start + nL0.start, dstNSize);
                        }
                        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
                        this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
                    }
                }
                if (unlikely(!reuseQBuf)) {
                    this->qL1BufId = (this->qL1BufId + 1) % L1_Q_BUFCNT;
                    reuseQBuf = canFullLoadQ;
                }
            }
            if (!HEAD_PAIR || pairHead_ == 1U)
                SetFlag<HardEvent::MTE1_MTE2>(KP_EVENT0 + this->kpL1BufId);
            ++this->kpL1BufId;
            if (this->kpL1BufId >= L1_KP_BUFCNT) {
                this->kpL1BufId = 0;
            }
        }

        uint32_t qkSyncId = constInfo.syncC1V1;
        if constexpr (CFG::PRELOAD_NUM == 4U) {
            qkSyncId += info.loop % CFG::PRELOAD_NUM;
        }
        if constexpr (HEAD_PAIR)
            pairQSignature_[pairHead_] = qCoord;
        if (!HEAD_PAIR || pairHead_ == 1U)
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(qkSyncId);
    }

    template <CubeFormat OutFormat = CubeFormat::ND, CubeFormat AFormat = CubeFormat::ND>
    __aicore__ inline void ComputeMm2(const RunInfo& info)
    {
        if constexpr (HEAD_PAIR) {
            for (uint32_t head = 0U; head < 2U; ++head) {
                pairHead_ = head;
                RunInfo part = info;
                part.n2Idx += head;
                ComputeMm2Head(part);
            }
        } else {
            ComputeMm2Head(info);
        }
    }

    template <CubeFormat OutFormat = CubeFormat::ND, CubeFormat AFormat = CubeFormat::ND>
    __aicore__ inline void ComputeMm2Head(const RunInfo& info)
    {
        Axis m(info.actMBaseSize);
        Axis n(constInfo.headDim);
        Axis k(info.actualSingleProcessSInnerSize);
        auto KL1Slices = k.Split(256U);
        const bool canFullLoadV = KL1Slices.size() <= L1_V_BUFCNT;
        const bool preloadV =
            HEAD_PAIR ? true :
                        (D128_PANEL_LAYOUT ? info.isFirstSInnerLoop :
                                             QfaPolicy::HIGH_PRECISION && constInfo.headDim == 128U &&
                                                 info.actMBaseSize == 128U && k.sizeAct > 256U && canFullLoadV);
        const uint32_t panelVSlot = info.loop & 1U;
        vL1Snapshot.bufCnt = 0U;
        vL1Snapshot.firstBufId = this->vL1BufId;
        vL1Snapshot.signature = ((uint64_t)info.bIdx << 48) | ((uint64_t)info.n2Idx << 32) | info.s2BatchOffset;
        if constexpr (D128_PANEL_LAYOUT) {
            if (preloadV) {
                WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + panelVSlot);
                CopyVToL1(panelVSlot, info, 0U, k.sizeAct);
                SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + panelVSlot);
                WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + panelVSlot);
            }
        } else if (preloadV) {
            if constexpr (HEAD_PAIR) {
                this->vL1BufId = 0U;
                vL1Snapshot.firstBufId = 0U;
            }
            // V has no dependency on P. Consume each ready here; ComputeMm2 reuses the resident V slots.
            for (auto& kL1 : KL1Slices) {
                const uint32_t slot = this->vL1BufId;
                if (!HEAD_PAIR || pairHead_ == 0U) {
                    WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + slot);
                    CopyVToL1(slot, info, kL1.start, kL1.sizeAct);
                    SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + slot);
                    WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + slot);
                }
                ++vL1Snapshot.bufCnt;
                this->vL1BufId = (slot + 1U) % L1_V_BUFCNT;
            }
        }
        const bool useTileSync = CFG::ENABLE_V1C2_TILE_SYNC && (info.actMBaseSize <= ConstInfo::SYNC_V1_C2_TILE_MAX_M);
        const bool useHalfP = D128_PANEL_LAYOUT && QfaUseHalfP(info);
        if ((!HEAD_PAIR || pairHead_ == 0U) && !useTileSync && !useHalfP) {
            uint32_t vec1SyncId = constInfo.syncV1C2;
            if constexpr (CFG::PRELOAD_NUM == 4U) {
                vec1SyncId += info.loop % CFG::PRELOAD_NUM;
            }

            CrossCoreWaitFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_M>(vec1SyncId);
        }
        constexpr uint32_t M_SPLIT_SIZE = 128;
        const uint32_t K_BASE = QfaUseK128() ? 128U : (QFA_INT8 ? 32U : 128U);

        auto mmResGm =
            this->mm2ResGm[info.loop % CFG::PRELOAD_NUM][HEAD_PAIR ? pairHead_ * constInfo.bmm2ResUbSize / 2U : 0U];

        ResetLoad3DConfig<M_SPLIT_SIZE>();

        bool reuseVBuf = preloadV;

        uint32_t v1c2TileIdx = 0U;
        const uint32_t mStep = useHalfP || constInfo.headDim > 128U ? 64U : M_SPLIT_SIZE;
        for (auto& mL1 : m.Split(mStep)) {
            if (useHalfP)
                CrossCoreWaitFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_M>(mL1.start == 0U ? QFA_HALF_P_READY :
                                                                                       constInfo.syncV1C2);
            if (likely(useTileSync)) {
                if (v1c2TileIdx == 0U) {
                    CrossCoreWaitFlag(constInfo.syncV1C2);
                } else if (v1c2TileIdx == 1U) {
                    CrossCoreWaitFlag(ConstInfo::SYNC_V1_C2_TILE1_FLAG);
                } else if (v1c2TileIdx == 2U) {
                    CrossCoreWaitFlag(ConstInfo::SYNC_V1_C2_TILE2_FLAG);
                } else {
                    CrossCoreWaitFlag(ConstInfo::SYNC_V1_C2_TILE3_FLAG);
                }
            }
            ++v1c2TileIdx;
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);

            int32_t kL1Id = -1;
            for (auto& kL1 : KL1Slices) {
                ++kL1Id;
                const uint32_t pSlot = HEAD_PAIR         ? pairHead_ :
                                       D128_PANEL_LAYOUT ? (static_cast<uint32_t>(kL1Id) & 1U) :
                                                           this->kpL1BufId;
                const uint32_t pEvent = (D128_PANEL_LAYOUT || HEAD_PAIR ? P_EVENT0 : KP_EVENT0) + pSlot;
                if constexpr (D128_PANEL_LAYOUT) {
                    // Steady V loads follow P-ready, avoiding the previous panel's PV writeback burst.
                    if (!preloadV && mL1.start == 0U && kL1Id == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + panelVSlot);
                        CopyVToL1(panelVSlot, info, 0U, k.sizeAct);
                        SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + panelVSlot);
                        WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + panelVSlot);
                    }
                }
                WaitFlag<HardEvent::MTE1_MTE2>(pEvent);
                CopyPToL1<AFormat>(pSlot, info,
                                   (AFormat == CubeFormat::NZ) ? m.AlignedSize() :
                                   QFA_INT8                    ? constInfo.s2BaseSize :
                                                                 info.actualSingleProcessSInnerSizeAlign,
                                   mL1.start, mL1.sizeAct, kL1.start, kL1.sizeAct);

                SetFlag<HardEvent::MTE2_MTE1>(pEvent);
                WaitFlag<HardEvent::MTE2_MTE1>(pEvent);

                uint32_t vBufId;
                if constexpr (D128_PANEL_LAYOUT) {
                    vBufId = panelVSlot;
                } else {
                    if (unlikely(kL1Id >= vL1Snapshot.bufCnt))
                        reuseVBuf = false;
                    if (unlikely(!reuseVBuf)) {
                        vBufId = this->vL1BufId;
                        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                        CopyVToL1(vBufId, info, kL1.start, kL1.sizeAct);

                        SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                        WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId);
                        vL1Snapshot.bufCnt += static_cast<uint32_t>(canFullLoadV);
                    } else {
                        vBufId = (vL1Snapshot.firstBufId + kL1Id) % L1_V_BUFCNT;
                    }
                }

                for (auto& kL0 : kL1.Split(K_BASE)) {
                    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);

                    LoadAToL0<M_SPLIT_SIZE>(this->abL0BufId, PL1(pSlot), mL1.AlignedSize(), 0, mL1.AlignedSize(),
                                            kL0.start, kL0.AlignedSize());
                    constexpr uint32_t C0 = ONE_BLK_SIZE / sizeof(KV_T);
                    LoadBToL0(this->abL0BufId, vL1Tensor[vBufId],
                              D128_PANEL_LAYOUT ? k.AlignedSize<C0>() : kL1.AlignedSize<C0>(),
                              kL0.start + (D128_PANEL_LAYOUT ? kL1.start : 0U), kL0.AlignedSize<C0>(),
                              HEAD_PAIR ? pairHead_ * 128U : 0U, n.AlignedSize<C0>());
                    SetFlag<HardEvent::MTE1_M>(L0_READY_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(L0_READY_EVENT);

                    bool isLastK = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
                    MmadParams mmadParams;
                    mmadParams.m = mL1.AlignedSize();
                    mmadParams.n = QfaUseK128() ? n.AlignedSize<C0>() : n.sizeAct;
                    mmadParams.k = kL0.sizeAct;
                    mmadParams.cmatrixInitVal = (kL1.start == 0) && (kL0.start == 0);
                    mmadParams.cmatrixSource = false;
                    if constexpr (CFG::ENABLE_UNIFLAG) {
                        mmadParams.unitFlag = isLastK ? 3 : 2;
                    }
                    Mmad(cL0Tensor[this->cL0BufId], aL0Tensor[this->abL0BufId], bL0Tensor[this->abL0BufId], mmadParams);
                    if (likely(!isLastK)) {
                        AscendC::PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + this->abL0BufId);
                    this->abL0BufId = (this->abL0BufId + 1) % L0AB_BUFCNT;
                }
                SetFlag<HardEvent::MTE1_MTE2>(pEvent);
                if constexpr (!D128_PANEL_LAYOUT && !HEAD_PAIR) {
                    this->kpL1BufId = (this->kpL1BufId + 1U) % L1_KP_BUFCNT;
                }

                if (D128_PANEL_LAYOUT ? (kL1.IsTailOf(k) && mL1.IsTailOf(m)) : (!canFullLoadV || mL1.IsTailOf(m))) {
                    // 当可以复用V_L1 buf时，它的生命周期跨越整个mL1的迭代，在mL1迭代完成释放
                    if (!HEAD_PAIR || pairHead_ == 1U)
                        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId);
                }

                if constexpr (!D128_PANEL_LAYOUT) {
                    if (unlikely(!reuseVBuf)) {
                        this->vL1BufId = (this->vL1BufId + 1) % L1_V_BUFCNT;
                        reuseVBuf = canFullLoadV;
                    }
                }
            }
            if constexpr (!CFG::ENABLE_UNIFLAG) {
                SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
                WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + this->cL0BufId);
            }
            FixpipeCToGM<OutFormat>(mmResGm, this->cL0BufId,
                                    (OutFormat == CubeFormat::NZ) ?
                                        m.sizeAct :
                                        constInfo.headDimAlign, /* 输出ND时，mm2ResGm两行之间间隔元素个数按32对齐 */
                                    mL1.start, mL1.sizeAct, 0,
                                    (OutFormat == CubeFormat::NZ) ? n.AlignedSize() : n.sizeAct, 0x39800000);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + this->cL0BufId);
            this->cL0BufId = (this->cL0BufId + 1) % L0C_BUFCNT;
        }

        uint32_t pvSyncId = constInfo.syncC2V2;
        if constexpr (CFG::PRELOAD_NUM == 4U) {
            pvSyncId += info.loop % CFG::PRELOAD_NUM;
        }
        if (!HEAD_PAIR || pairHead_ == 1U)
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(pvSyncId);
    }
};

// QFA FP32 vector service. Generic queues and output support are inherited unchanged.
template <typename FIAT, bool OPTIMIZED_VECTOR_SERVICE = false, uint32_t ROW_BATCH = 0U, bool WIDE_ROW_SCALE = false,
          bool HEAD_PAIR = false, bool COMPACT_D256 = false>
class QfaHighPrecisionVec : private QfaHostPlanPolicy::VecState, public FiaBlockVecNonQuant<FIAT> {
    using Base = FiaBlockVecNonQuant<FIAT>;
    using QfaPolicy = QfaHostPlanPolicy;

public:
    static constexpr bool HOST_PLAN = true;
    static constexpr bool PAIRED_HEAD_SERVICE = HEAD_PAIR;
    static constexpr bool OPTIMIZED_SERVICE = OPTIMIZED_VECTOR_SERVICE;
    static constexpr bool HIGH_PRECISION = true;
    static constexpr bool QFA_INT8 = true;
    using T = float;
    using COMPUTE_T = float;
    using KV_T = int8_t;
    using OUT_T = bfloat16_t;
    using MM1_GM_T = int32_t;
    using MM2_GM_T = int32_t;
    static constexpr uint32_t QFA_BT_CACHE_ELEMS = 512U;
    static constexpr uint32_t QFA_QD_CACHE_ELEMS = COMPACT_D256 ? 256U : 512U;
    __aicore__ inline QfaHostPlanPolicy::VecState& QfaState()
    {
        return static_cast<QfaHostPlanPolicy::VecState&>(*this);
    }
    __aicore__ inline void InitQfaParams(const typename QfaPolicy::Params& p)
    {
        if constexpr (QFA_INT8) {
            QfaState().keyHeadStride = p.keyHeadStride;
            QfaState().valueHeadStride = p.valueHeadStride;
            QfaState().qfaKDescaleStrideBlock = p.qfaKDescaleStrideBlock;
            QfaState().qfaKDescaleStrideHead = p.qfaKDescaleStrideHead;
            QfaState().qfaKDescaleStrideSlot = p.qfaKDescaleStrideSlot;
            QfaState().qfaQDescaleStrideToken = p.qfaQDescaleStrideToken;
            QfaState().qfaVDescaleStrideHead = p.qfaVDescaleStrideHead;
            if constexpr (HIGH_PRECISION) {
                QfaState().hpRows = ROW_BATCH != 0U ? ROW_BATCH : p.hpRows;
                QfaState().hpVec2Rows = p.hpVec2Rows;
                QfaState().hpScratchBytes = p.hpScratchBytes;
                QfaState().hpStateRows = p.hpStateRows;
            }
        }
    }
    GlobalTensor<float> lseOut;
    bool writeLse = false;
    uint32_t firstVisibleQ = 0U;

    __aicore__ inline void WriteLse(LocalTensor<float> sums, LocalTensor<float> maxima, uint32_t count, uint32_t row,
                                    uint64_t headTokenBase)
    {
        if (firstVisibleQ == ~0U) // Empty KV rows were initialized before traversal.
            return;
        auto& st = QfaState();
        auto scratch = st.qfaFdLseBuf.template Get<float>();
        auto expanded = scratch[32U];
        Ln(scratch, sums, count);
        PipeBarrier<PIPE_V>();
        Add(scratch, scratch, maxima, count);
        PipeBarrier<PIPE_V>();
        const uint64_t firstRow = static_cast<uint64_t>(firstVisibleQ) * constInfo.gSize;
        if (row < firstRow) {
            const uint32_t invalid = firstRow - row < count ? firstRow - row : count;
            Duplicate(scratch, __builtin_inff(), invalid);
            PipeBarrier<PIPE_V>();
        }
        const uint32_t tokens = st.qfaQDescaleStrideToken;
        for (uint32_t i = 0; i < count; i += 8U) {
            // Pack one 32-byte source block for each strided NT destination.
            const uint32_t batch = count - i < 8U ? count - i : 8U;
            Brcb(expanded, scratch[i], 1U, {1U, 8U});
            SetFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
            for (uint32_t j = 0; j < batch;) {
                const uint32_t m = row + i + j;
                const uint32_t group = m % constInfo.gSize;
                const uint32_t left = constInfo.gSize - group;
                const uint32_t heads = batch - j < left ? batch - j : left;
                const uint64_t dst = headTokenBase + static_cast<uint64_t>(group) * tokens + m / constInfo.gSize;
                DataCopyExtParams cp(static_cast<uint16_t>(heads), sizeof(float), 0U, (tokens - 1U) * 4U, 0U);
                DataCopyPad(lseOut[dst], expanded[j * 8U], cp);
                j += heads;
            }
            SetFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
            WaitFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
        }
    }

    using Base::constInfo;
    using Base::mSplitInfo;
    using Base::tmpBuff1;
    using Base::inputQue1;
    using Base::outputQue1;
    using Base::outputQue2;
    using Base::softmaxMaxBuff;
    using Base::softmaxExpBuff;
    using Base::softmaxSumBuff;
    using Base::softmaxMaxDefaultBuff;
    using Base::softmaxSumDefaultBuff;
    using Base::softmaxMaxUb;
    using Base::softmaxSumUb;
    using Base::softmaxExpUb;
    using Base::softmaxMaxDefaultUb;
    using Base::softmaxSumDefaultUb;
    using Base::vec1ResGm;
    using Base::vec2ResGm;
    using Base::accumOutGm;
    using Base::lseMaxFdGm;
    using Base::lseSumFdGm;
    using Base::attentionOutGm;
    using Base::mm1ResGm;
    using Base::mm2ResGm;

    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<int8_t> p, GlobalTensor<int32_t> s)
    {
        this->vec1ResGm = p;
        // DMA preserves the raw INT32 bits through the native same-width view.
        this->mm1ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ int32_t*>(s.GetPhyAddr())));
    }
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<float> acc, GlobalTensor<int32_t> pv)
    {
        this->vec2ResGm = acc;
        this->mm2ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ int32_t*>(pv.GetPhyAddr())));
    }
    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        HpInit(pipe);
    }
    __aicore__ inline void AllocEventID()
    {
        HpAllocEvents();
    }
    __aicore__ inline void FreeEventID()
    {
        HpFreeEvents();
    }
    __aicore__ inline void ComputeVec1(const RunInfo& info, const RunInfo* next = nullptr)
    {
        SetMSplitInfo(info.actMBaseSize);
        HpVec1(info, next);
    }
    __aicore__ inline void ComputeVec2(const RunInfo& info)
    {
        SetMSplitInfo(info.actMBaseSize);
        CrossCoreWaitFlag(constInfo.syncC2V2);
        HpVec2(info);
    }
    __aicore__ inline void QfaJointPrefetchInitial(const RunInfo& info)
    {
        HpPrefetchInitial(info);
    }
    __aicore__ inline void QfaJointPrefetchScore(const RunInfo& info)
    {
        HpPrefetchScore(info);
    }
    __aicore__ inline void SetMSplitInfo(uint32_t rows)
    {
        mSplitInfo.nBufferIdx = 0U;
        mSplitInfo.nBufferStartM = 0U;
        mSplitInfo.nBufferDealM = rows;
        if constexpr (HEAD_PAIR) {
            mSplitInfo.vecStartM = 0U;
            mSplitInfo.vecDealM = rows;
            return;
        }
        mSplitInfo.vecDealM = QfaJointVec0Rows(rows);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
    }
    __aicore__ inline void BindQfaPlanFd(uint32_t prefix, uint32_t row, uint32_t rows)
    {
        if constexpr (QfaPolicy::HOST_PLAN) {
            QfaState().planFdPrefix = prefix;
            QfaState().planFdRow = row;
            QfaState().planFdRows = rows;
        }
    }
    __aicore__ inline uint64_t QfaFdPrefix(uint32_t task, uint32_t splits)
    {
        if constexpr (QfaPolicy::HOST_PLAN)
            return QfaState().planFdPrefix;
        else
            return static_cast<uint64_t>(task) * splits;
    }
    __aicore__ inline uint32_t QfaFdRows(uint32_t tokens)
    {
        if constexpr (QfaPolicy::HOST_PLAN)
            return QfaState().planFdRows;
        else
            return tokens * static_cast<uint32_t>(constInfo.gSize);
    }
    __aicore__ inline uint32_t QfaFdGlobalRow(uint32_t row)
    {
        if constexpr (QfaPolicy::HOST_PLAN)
            return row + QfaState().planFdRow;
        else
            return row;
    }

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengthsQ,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* deqScale1,
                                __gm__ uint8_t* quantScale1, __gm__ uint8_t* deqScale2, __gm__ uint8_t* quantScale2,
                                __gm__ uint8_t* quantOffset2, __gm__ uint8_t* antiquantScale,
                                __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix,
                                __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope,
                                __gm__ uint8_t* keyRope, __gm__ uint8_t* keyRopeAntiquantScale,
                                __gm__ uint8_t* learnableSink, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse)
    {
        Base::Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1,
                   deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize,
                   kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                   keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,
                   learnableSink, attentionOut, softmaxLse);
        QfaState().qfaQDescaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deqScale1));
        QfaState().qfaKDescaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(antiquantScale));
        QfaState().qfaVDescaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deqScale2));
        QfaState().qfaPScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale1));
        QfaState().qfaBlockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    }

    __aicore__ inline void SetQfaFdLayout(uint32_t coreIdx, uint32_t strideBytes)
    {
        QfaState().qfaFdCoreIdx_ = coreIdx;
        QfaState().qfaFdStrideElems_ = strideBytes / sizeof(float);
    }

    __aicore__ inline void SetQfaKScaleDense(const GlobalTensor<half>& denseGm, uint32_t strideElems, uint32_t kvHeads)
    {
        QfaState().qfaKScaleDenseGm = denseGm;
        QfaState().qfaKScaleDenseStride_ = strideElems;
        QfaState().qfaKScaleDenseKvHeads_ = kvHeads;
    }

    __aicore__ inline void QfaPrepareRows(uint32_t batch, uint32_t head, uint32_t rowStart, uint32_t rows,
                                          uint32_t kvLength, uint64_t qScaleBase)
    {
        if constexpr (QfaPolicy::HOST_PLAN) {
            if (!QfaState().qfaPScaleLoaded_) {
                QfaState().qfaPScaleCache_ = QfaState().qfaPScaleGm.GetValue(0U);
                QfaState().qfaPScaleLoaded_ = true;
            }
            QfaState().qfaVScaleCache_ = QfaState().qfaVDescaleGm.GetValue(head * QfaState().qfaVDescaleStrideHead);
            const uint32_t pages = (kvLength + constInfo.kvCacheBlockSize - 1U) / constInfo.kvCacheBlockSize;
            if (QfaState().planCachedBatch != batch) {
                QfaState().planCachedBatch = batch;
                QfaState().qfaTaskBtCount_ = pages <= QFA_BT_CACHE_ELEMS ? pages : 0U;
                if (QfaState().qfaTaskBtCount_ != 0U) {
                    DataCopyExtParams cp(1U, pages * sizeof(int32_t), 0U, 0U, 0U);
                    DataCopyPadExtParams<int32_t> pad(false, 0U, 0U, 0);
                    DataCopyPad(QfaState().qfaPreloadBuf.template Get<int32_t>(),
                                QfaState().qfaBlockTableGm[batch * constInfo.maxBlockNumPerBatch], cp, pad);
                }
            }
            const uint32_t gs = constInfo.gSize;
            const uint32_t firstQ = rowStart / gs;
            const uint32_t tokens = (rowStart + rows - 1U) / gs - firstQ + 1U;
            QfaState().qfaTaskQdStride_ = (tokens + 7U) & ~7U;
            auto qd = QfaState().qfaPreloadBuf.template Get<float>()[QFA_BT_CACHE_ELEMS];
            QfaState().qfaQdCached_ = gs * QfaState().qfaTaskQdStride_ <= QFA_QD_CACHE_ELEMS;
            if (QfaState().qfaQdCached_) {
                if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                    const uint64_t headStride = QfaState().qfaQDescaleStrideToken;
                    if (gs <= 65535U && headStride >= tokens && headStride - tokens <= 0x3FFFFFFFULL) {
                        DataCopyExtParams cp(gs, tokens * sizeof(float), (headStride - tokens) * sizeof(float), 0U, 0U);
                        DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0F);
                        DataCopyPad(qd, QfaState().qfaQDescaleGm[qScaleBase + firstQ], cp, pad);
                    } else {
                        QfaState().qfaQdCached_ = false;
                    }
                } else {
                    for (uint32_t g = 0U; g < gs; ++g) {
                        DataCopyExtParams cp(1U, tokens * sizeof(float), 0U, 0U, 0U);
                        DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0F);
                        DataCopyPad(
                            qd[g * QfaState().qfaTaskQdStride_],
                            QfaState().qfaQDescaleGm[qScaleBase + firstQ + g * QfaState().qfaQDescaleStrideToken], cp,
                            pad);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            SetMSplitInfo(rows);
            auto scales = QfaState().qfaJointQScaleBuf.template Get<float>();
            if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                if (QfaState().qfaQdCached_ && mSplitInfo.vecDealM != 0U) {
                    // Index and compact scales share existing scratch before any Softmax/Vec2 use.
                    auto indices = tmpBuff1.template Get<uint32_t>();
                    auto compact = tmpBuff1.template Get<float>()[2048U];
                    const uint32_t count = mSplitInfo.vecDealM;
                    SetFlag<HardEvent::V_S>(EVENT_ID2);
                    WaitFlag<HardEvent::V_S>(EVENT_ID2);
                    for (uint32_t r = 0U; r < count; ++r) {
                        const uint32_t m = rowStart + mSplitInfo.vecStartM + r;
                        const uint32_t index = (m % gs) * QfaState().qfaTaskQdStride_ + m / gs - firstQ;
                        indices.SetValue(r, index * sizeof(float));
                    }
                    SetFlag<HardEvent::S_V>(EVENT_ID2);
                    WaitFlag<HardEvent::S_V>(EVENT_ID2);
                    Gather(compact, qd, indices, 0U, count);
                    PipeBarrier<PIPE_V>();
                    Muls(compact, compact, constInfo.scaleValue, count);
                    PipeBarrier<PIPE_V>();
                    if ((count & 7U) != 0U) {
                        uint64_t tailMask[2] = {0xFFULL & ~((1ULL << (count & 7U)) - 1ULL), 0ULL};
                        Duplicate(compact[count & ~7U], 0.0F, tailMask, 1U, 1U, 0U);
                        PipeBarrier<PIPE_V>();
                    }
                    if constexpr (WIDE_ROW_SCALE) {
                        // Materialize the Q scales once; hot Mul reads distinct bank groups.
                        for (uint32_t col = 0U; col < 64U; col += 8U)
                            Brcb(scales[col], compact, (count + 7U) / 8U, {8U, 64U});
                    } else {
                        Brcb(scales, compact, (count + 7U) / 8U, {1U, 8U});
                    }
                    PipeBarrier<PIPE_V>();
                    return;
                }
            }
            for (uint32_t r = 0U; r < mSplitInfo.vecDealM; ++r) {
                const uint32_t m = rowStart + mSplitInfo.vecStartM + r;
                const float value = QfaState().qfaQdCached_ ?
                                        qd.GetValue((m % gs) * QfaState().qfaTaskQdStride_ + m / gs - firstQ) :
                                        QfaState().qfaQDescaleGm.GetValue(qScaleBase + m / gs +
                                                                          (m % gs) * QfaState().qfaQDescaleStrideToken);
                constexpr uint32_t scaleCols = WIDE_ROW_SCALE ? 64U : 8U;
                Duplicate(scales[r * scaleCols], value * constInfo.scaleValue, scaleCols);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void QfaWriteNeutralFdPartial(uint32_t rows, uint32_t dim)
    {
        if constexpr (!QFA_INT8) {
            return;
        }
        if (rows == 0U || dim == 0U) {
            return;
        }
        if (QfaState().qfaJointRoute_) {
            const uint32_t stride = constInfo.headDimAlign;
            const uint64_t slot = static_cast<uint64_t>(QfaState().qfaFdCoreIdx_) * QfaState().qfaFdStrideElems_;
            const uint32_t first = QfaJointVec0Rows(rows);
            LocalTensor<float> stage = tmpBuff1.template Get<float>();
            for (uint32_t owner = 0U; owner < 2U; ++owner) {
                const uint32_t lo = owner == 0U ? 0U : first;
                const uint32_t hi = owner == 0U ? first : rows;
                for (uint32_t row = lo; row < hi; row += 16U) {
                    const uint32_t count = hi - row < 16U ? hi - row : 16U;
                    Duplicate(stage, 0.0F, count * stride);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    const uint64_t out = slot + static_cast<uint64_t>(row) * stride;
                    DataCopy(accumOutGm[out], stage, count * stride);
                    SetFlag<HardEvent::MTE3_V>(QfaState().qfaJointOutDone_);
                    WaitFlag<HardEvent::MTE3_V>(QfaState().qfaJointOutDone_);
                    SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
                    WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
                    for (uint32_t off = 0U; off < count * stride; off += 16U) {
                        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                            accumOutGm[out + off]);
                    }
                    Duplicate(stage, -3.0E38F, 16U);
                    Duplicate(stage[16U], 0.0F, 16U);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    const uint64_t state = slot + QfaJointFdStateRow(row, rows);
                    DataCopyExtParams cp(1U, count * sizeof(float), 0U, 0U, 0U);
                    DataCopyPad(lseMaxFdGm[state], stage, cp);
                    DataCopyPad(lseSumFdGm[state], stage[16U], cp);
                    SetFlag<HardEvent::MTE3_V>(QfaState().qfaJointOutDone_);
                    WaitFlag<HardEvent::MTE3_V>(QfaState().qfaJointOutDone_);
                }
            }
            return;
        }
        const uint32_t rowStride = WIDE_ROW_SCALE ? 128U : constInfo.headDimAlign;
        const uint64_t slotBase = static_cast<uint64_t>(QfaState().qfaFdCoreIdx_) * QfaState().qfaFdStrideElems_;
        LocalTensor<float> stage = tmpBuff1.template Get<float>();
        Duplicate(stage, 0.0F, rows * rowStride);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(accumOutGm[slotBase], stage, rows * rowStride);
        for (uint32_t row = 0U; row < rows; ++row) {
            lseMaxFdGm.SetValue(slotBase + row, -3.0E38F);
            lseSumFdGm.SetValue(slotBase + row, 0.0F);
        }

        SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);

        for (uint32_t off = 0U; off < rows * rowStride; off += 16U) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                accumOutGm[slotBase + off]);
        }
        event_t evtSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtSToMte3);
        WaitFlag<HardEvent::S_MTE3>(evtSToMte3);
        for (uint32_t row = 0U; row < rows; row += 16U) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                lseMaxFdGm[slotBase + row]);
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                lseSumFdGm[slotBase + row]);
        }
    }

    __aicore__ inline void QfaFlashDecodeTask(uint32_t task, uint32_t tokenStart, uint32_t tokenCount, uint32_t kvHead,
                                              uint32_t splitCount, uint32_t rowBlockStart = 0U,
                                              uint32_t rowBlockCount = ~0U, uint32_t fdRowSize = 16U)
    {
        if constexpr (!QFA_INT8) {
            return;
        }
        const uint32_t groupSize = static_cast<uint32_t>(constInfo.gSize);
        const uint32_t headDim = WIDE_ROW_SCALE ? 128U : static_cast<uint32_t>(constInfo.headDim);
        if (groupSize == 0U || (!QfaPolicy::HOST_PLAN && groupSize > 16U) || headDim == 0U ||
            headDim > constInfo.headDimAlign || splitCount == 0U || tokenCount == 0U) {
            return;
        }
#if defined(QFA_A3_COMPILE_JOINT_ONLY)

        AscendC::ICachePreLoad(2);
#endif

        constexpr uint32_t QFA_FD_ROW_BLOCK = 16U;
        const uint32_t logicalRows = OPTIMIZED_VECTOR_SERVICE && fdRowSize == 8U ? 8U : 16U;
        const uint32_t rowStride = WIDE_ROW_SCALE ? 128U : constInfo.headDimAlign;
        const uint32_t mRows = QfaFdRows(tokenCount);
        const uint32_t qHeadNum = static_cast<uint32_t>(constInfo.qHeadNum);
        const uint64_t taskSlot = QfaFdPrefix(task, splitCount);

        LocalTensor<float> outAcc =
            COMPACT_D256 ? QfaState().hpPvBuf.template Get<float>() : tmpBuff1.template Get<float>();
        LocalTensor<float> denomRows = tmpBuff1.template Get<float>()[COMPACT_D256 ? 0U : QFA_FD_ROW_BLOCK * rowStride];
        LocalTensor<float> globalMax = softmaxExpUb;
        LocalTensor<float> localSum = softmaxExpUb[QFA_FD_ROW_BLOCK];
        LocalTensor<float> weight = softmaxMaxUb;
        LocalTensor<float> denom = softmaxSumUb;

        const uint32_t outPitch =
            ((headDim % (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T))) == 0U) ? headDim : rowStride;
        LocalTensor<OUT_T> outStage = QfaState().qfaAccBuf.template Get<OUT_T>();

        const uint32_t totalBlocks = (mRows + logicalRows - 1U) / logicalRows;
        uint32_t rowBlockEnd = totalBlocks;
        if (rowBlockCount != ~0U && rowBlockStart + rowBlockCount < totalBlocks) {
            rowBlockEnd = rowBlockStart + rowBlockCount;
        }
        const uint32_t rowLo = rowBlockStart * logicalRows;
        uint32_t rowHi = rowBlockEnd * logicalRows;
        if (rowHi > mRows) {
            rowHi = mRows;
        }
        if (rowLo >= rowHi) {
            return;
        }
        for (uint32_t rowBase = rowLo; rowBase < rowHi; rowBase += logicalRows) {
            uint32_t rowCount = rowHi - rowBase;
            if (rowCount > logicalRows) {
                rowCount = logicalRows;
            }
            Duplicate(outAcc, 0.0F, rowCount * rowStride);
            Duplicate(denom, 0.0F, rowCount);
            Duplicate(globalMax, -3.0E38F, rowCount);
            LocalTensor<float> lseUb = QfaState().qfaFdLseBuf.template Get<float>();
            DataCopyExtParams lseParams(1U, rowCount * sizeof(float), 0U, 0U, 0U);
            DataCopyPadExtParams<float> lsePad(false, 0U, 0U, 0.0F);
            if (QfaState().qfaJointRoute_) {
                PipeBarrier<PIPE_V>();
            } else {
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
            }
            if (QfaState().qfaJointRoute_) {
                LocalTensor<float> states = QfaState().qfaJointQScaleBuf.template Get<float>();
                QfaJointReadFdState(states, lseMaxFdGm, taskSlot * QfaState().qfaFdStrideElems_, rowBase, rowCount,
                                    mRows, splitCount);
                SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointKReady_);
                WaitFlag<HardEvent::MTE2_V>(QfaState().qfaJointKReady_);
                LocalTensor<float> partialOwner = inputQue1.template AllocTensor<float>();
                const uint32_t partStride = partialOwner.GetSize() / 2U;
                const uint32_t partElems = rowCount * rowStride;
                const uint32_t capacity = partStride / partElems;
                const uint32_t batch = capacity < splitCount ? capacity : splitCount;
                LocalTensor<float> part[2] = {partialOwner, partialOwner[partStride]};
                bool freePending[2] = {false, false};
                DataCopyExtParams partialCopy(batch, partElems * sizeof(float),
                                              (QfaState().qfaFdStrideElems_ - partElems) * sizeof(float), 0U, 0U);
                DataCopyPad(
                    part[0],
                    accumOutGm[taskSlot * QfaState().qfaFdStrideElems_ + static_cast<uint64_t>(rowBase) * rowStride],
                    partialCopy, lsePad);
                SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[0]);
                for (uint32_t split = 0U; split < splitCount; ++split) {
                    Max(globalMax, globalMax, states[split * 32U], rowCount);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                    const BinaryRepeatParams sameMax{1U, 1U, 1U, 4U, 4U, 0U};
                    const BinaryRepeatParams pairedSum{1U, 1U, 1U, 4U, 4U, 4U};
                    Sub(states, states, globalMax, static_cast<uint64_t>(rowCount), static_cast<uint8_t>(splitCount),
                        sameMax);
                    PipeBarrier<PIPE_V>();
                    Exp(states, states, static_cast<uint64_t>(rowCount), static_cast<uint8_t>(splitCount),
                        {1U, 1U, 4U, 4U});
                    PipeBarrier<PIPE_V>();
                    Mul(states, states, states[16U], static_cast<uint64_t>(rowCount), static_cast<uint8_t>(splitCount),
                        pairedSum);
                    PipeBarrier<PIPE_V>();
                }
                uint32_t inBatch = 0U, slot = 0U;
                for (uint32_t split = 0U; split < splitCount; ++split) {
                    if (inBatch == 0U) {
                        WaitFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[slot]);
                        if (split + batch < splitCount) {
                            const uint32_t next = slot ^ 1U;
                            if (freePending[next]) {
                                WaitFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[next]);
                                freePending[next] = false;
                            }
                            const uint32_t remaining = splitCount - split - batch;
                            partialCopy.blockCount = remaining < batch ? remaining : batch;
                            const uint64_t src = (taskSlot + split + batch) * QfaState().qfaFdStrideElems_ +
                                                 static_cast<uint64_t>(rowBase) * rowStride;
                            DataCopyPad(part[next], accumOutGm[src], partialCopy, lsePad);
                            SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[next]);
                        }
                    }
                    auto partial = part[slot][inBatch * partElems];
                    if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                        Adds(weight, states[split * 32U], 0.0F, rowCount);
                    } else {
                        Sub(weight, states[split * 32U], globalMax, rowCount);
                        PipeBarrier<PIPE_V>();
                        Exp(weight, weight, rowCount);
                        PipeBarrier<PIPE_V>();
                        Mul(weight, weight, states[split * 32U + 16U], rowCount);
                    }
                    PipeBarrier<PIPE_V>();
                    Add(denom, denom, weight, rowCount);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> scales =
                        tmpBuff1.template Get<float>()[(COMPACT_D256 ? 0U : QFA_FD_ROW_BLOCK * rowStride) +
                                                       QFA_FD_ROW_BLOCK * 8U];
                    Brcb(scales, weight, (rowCount + 7U) / 8U, {1U, 8U});
                    PipeBarrier<PIPE_V>();
                    fa_base_vector::RowMuls<float>(partial, partial, scales, rowCount, rowStride, headDim);
                    PipeBarrier<PIPE_V>();
                    Add(outAcc, outAcc, partial, rowCount * rowStride);
                    ++inBatch;
                    if (inBatch == batch || split + 1U == splitCount) {
                        SetFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[slot]);
                        freePending[slot] = true;
                        inBatch = 0U;
                        slot ^= 1U;
                    }
                }
                for (uint32_t slot = 0U; slot < 2U; ++slot) {
                    if (freePending[slot]) {
                        WaitFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[slot]);
                    }
                }
                inputQue1.FreeTensor(partialOwner);
            } else {
                for (uint32_t split = 0U; split < splitCount; ++split) {
                    const uint64_t slotBase = (taskSlot + split) * QfaState().qfaFdStrideElems_;
                    DataCopyPad(lseUb, lseMaxFdGm[slotBase + rowBase], lseParams, lsePad);
                    if (QfaState().qfaShortQRoute_) {
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        Max(globalMax, globalMax, lseUb, rowCount);
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    } else {
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
                        for (uint32_t row = 0U; row < rowCount; ++row) {
                            const float value = lseUb.GetValue(row);
                            if (value > globalMax.GetValue(row)) {
                                globalMax.SetValue(row, value);
                            }
                        }
                    }
                }

                for (uint32_t split = 0U; split < splitCount; ++split) {
                    const uint64_t slotBase = (taskSlot + split) * QfaState().qfaFdStrideElems_;
                    DataCopyPad(lseUb, lseMaxFdGm[slotBase + rowBase], lseParams, lsePad);
                    DataCopyPad(lseUb[64U], lseSumFdGm[slotBase + rowBase], lseParams, lsePad);
                    if (QfaState().qfaShortQRoute_) {
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
                        Sub(weight, lseUb, globalMax, rowCount);
                        PipeBarrier<PIPE_V>();
                        Exp(weight, weight, rowCount);
                        PipeBarrier<PIPE_V>();
                        Mul(weight, weight, lseUb[64U], rowCount);
                        PipeBarrier<PIPE_V>();
                        Add(denom, denom, weight, rowCount);
                        PipeBarrier<PIPE_V>();
                    } else {
                        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
                        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);
                        for (uint32_t row = 0U; row < rowCount; ++row) {
                            weight.SetValue(row, lseUb.GetValue(row) - globalMax.GetValue(row));
                            localSum.SetValue(row, lseUb.GetValue(64U + row));
                        }
                        SetFlag<HardEvent::S_V>(EVENT_ID4);
                        WaitFlag<HardEvent::S_V>(EVENT_ID4);
                        Exp(weight, weight, rowCount);
                        PipeBarrier<PIPE_V>();
                        Mul(weight, weight, localSum, rowCount);
                        PipeBarrier<PIPE_V>();
                        Add(denom, denom, weight, rowCount);
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_S>(EVENT_ID2);
                        WaitFlag<HardEvent::V_S>(EVENT_ID2);
                    }

                    LocalTensor<float> partial = inputQue1.template AllocTensor<float>();
                    DataCopy(partial, accumOutGm[slotBase + static_cast<uint64_t>(rowBase) * rowStride],
                             rowCount * rowStride);
                    inputQue1.EnQue(partial);
                    inputQue1.template DeQue<float>();
                    if (QfaState().qfaShortQRoute_) {
                        LocalTensor<float> rowScales =
                            tmpBuff1.template Get<float>()[(COMPACT_D256 ? 0U : QFA_FD_ROW_BLOCK * rowStride) +
                                                           QFA_FD_ROW_BLOCK * 8U];
                        Brcb(rowScales, weight, (rowCount + 7U) / 8U, {1, 8});
                        PipeBarrier<PIPE_V>();
                        fa_base_vector::RowMuls<float>(partial, partial, rowScales, rowCount, rowStride, headDim);
                        PipeBarrier<PIPE_V>();
                        Add(outAcc, outAcc, partial, rowCount * rowStride);
                    } else {
                        for (uint32_t row = 0U; row < rowCount; ++row) {
                            const uint32_t rowOffset = row * rowStride;
                            Muls(partial[rowOffset], partial[rowOffset], weight.GetValue(row), headDim);
                            PipeBarrier<PIPE_V>();
                            Add(outAcc[rowOffset], outAcc[rowOffset], partial[rowOffset], headDim);
                            PipeBarrier<PIPE_V>();
                        }
                    }
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    inputQue1.FreeTensor(partial);
                }
            }
            if (writeLse)
                WriteLse(denom, globalMax, rowCount, QfaFdGlobalRow(rowBase),
                         static_cast<uint64_t>(kvHead) * groupSize * QfaState().qfaQDescaleStrideToken + tokenStart);
            if constexpr (QfaPolicy::HOST_PLAN) {
                Maxs(denom, denom, 1.0E-10F, rowCount);
                PipeBarrier<PIPE_V>();
            }
            if (QfaState().qfaJointRoute_) {
                Brcb(denomRows, denom, (rowCount + 7U) / 8U, {1U, 8U});
            } else {
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
                for (uint32_t row = 0U; row < rowCount; ++row) {
                    Duplicate(denomRows[row * AttentionCommon::FP32_BLOCK_ELEMENT_NUM], denom.GetValue(row),
                              AttentionCommon::FP32_BLOCK_ELEMENT_NUM);
                }
            }
            PipeBarrier<PIPE_V>();
            fa_base_vector::RowDivs<float>(outAcc, outAcc, denomRows, rowCount, rowStride, headDim);
            PipeBarrier<PIPE_V>();

            for (uint32_t row = 0U; row < rowCount; ++row) {
                Cast(outStage[(rowBase + row) * outPitch], outAcc[row * rowStride], AscendC::RoundMode::CAST_RINT,
                     headDim);
                PipeBarrier<PIPE_V>();
            }
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if ((headDim % (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T))) == 0U) {
            uint32_t runStart = rowLo;
            for (uint32_t row = rowLo + 1U; row <= rowHi; ++row) {
                const bool flush =
                    (row == rowHi) || (QfaFdGlobalRow(row) / groupSize != QfaFdGlobalRow(row - 1U) / groupSize);
                if (!flush) {
                    continue;
                }
                const uint64_t outOffset =
                    (static_cast<uint64_t>(tokenStart + QfaFdGlobalRow(runStart) / groupSize) * qHeadNum +
                     kvHead * groupSize + QfaFdGlobalRow(runStart) % groupSize) *
                    headDim;
                DataCopyExtParams params(1U, static_cast<uint32_t>((row - runStart) * headDim * sizeof(OUT_T)), 0U, 0U,
                                         0U);
                DataCopyPad(attentionOutGm[outOffset], outStage[runStart * headDim], params);
                runStart = row;
            }
        } else {
            DataCopyExtParams params(1U, static_cast<uint32_t>(headDim * sizeof(OUT_T)), 0U, 0U, 0U);
            for (uint32_t row = rowLo; row < rowHi; ++row) {
                const uint64_t outOffset =
                    (static_cast<uint64_t>(tokenStart + QfaFdGlobalRow(row) / groupSize) * qHeadNum +
                     kvHead * groupSize + QfaFdGlobalRow(row) % groupSize) *
                    headDim;
                DataCopyPad(attentionOutGm[outOffset], outStage[row * rowStride], params);
            }
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline void QfaFlashDecodeColumns(uint32_t task, uint32_t tokenStart, uint32_t tokenCount,
                                                 uint32_t kvHead, uint32_t splitCount, uint32_t rowBlock,
                                                 uint32_t colStart)
    {
        if constexpr (WIDE_ROW_SCALE)
            return;
        // D256 source rows stay intact; each actor reduces one aligned D128 slice.
        // The facade admits only the private M32 / FD11 decode geometry.
        constexpr uint32_t ROW_BLOCK = 16U;
        constexpr uint32_t D = 128U;
        const uint32_t groupSize = static_cast<uint32_t>(constInfo.gSize);
        const uint32_t headDim = static_cast<uint32_t>(constInfo.headDim);
        const uint32_t srcStride = constInfo.headDimAlign;
        const uint32_t mRows = QfaFdRows(tokenCount);
        const uint32_t rowBase = rowBlock * ROW_BLOCK;
        if (rowBase >= mRows)
            return;
        const uint32_t R = mRows - rowBase < ROW_BLOCK ? mRows - rowBase : ROW_BLOCK;
        if (!QfaState().qfaJointRoute_ || headDim != 256U || srcStride != 256U || rowBase + R > mRows ||
            colStart + D > headDim || splitCount > (QfaPolicy::HOST_PLAN ? 32U : 16U)) {
            return;
        }
        AscendC::ICachePreLoad(2);
        const uint64_t taskSlot = QfaFdPrefix(task, splitCount);
        LocalTensor<float> outAcc =
            COMPACT_D256 ? QfaState().hpPvBuf.template Get<float>() : tmpBuff1.template Get<float>();
        LocalTensor<float> denomRows = tmpBuff1.template Get<float>()[COMPACT_D256 ? 0U : R * D];
        LocalTensor<float> scales = tmpBuff1.template Get<float>()[(COMPACT_D256 ? 0U : R * D) + R * 8U];
        LocalTensor<float> globalMax = softmaxExpUb;
        LocalTensor<float> weight = softmaxMaxUb;
        LocalTensor<float> denom = softmaxSumUb;
        LocalTensor<float> states = QfaState().qfaJointQScaleBuf.template Get<float>();
        Duplicate(outAcc, 0.0F, R * D);
        Duplicate(denom, 0.0F, R);
        Duplicate(globalMax, -3.0E38F, R);
        PipeBarrier<PIPE_V>();
        QfaJointReadFdState(states, lseMaxFdGm, taskSlot * QfaState().qfaFdStrideElems_, rowBase, R, mRows, splitCount);
        SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointKReady_);
        WaitFlag<HardEvent::MTE2_V>(QfaState().qfaJointKReady_);
        LocalTensor<float> owner = inputQue1.template AllocTensor<float>();
        LocalTensor<float> part[2] = {owner, owner[R * D]};
        bool freePending[2] = {false, false};
        DataCopyExtParams in(R, D * sizeof(float), (srcStride - D) * sizeof(float), 0U, 0U);
        DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0F);
        DataCopyPad(
            part[0],
            accumOutGm[taskSlot * QfaState().qfaFdStrideElems_ + static_cast<uint64_t>(rowBase) * srcStride + colStart],
            in, pad);
        SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[0]);
        for (uint32_t split = 0U; split < splitCount; ++split) {
            Max(globalMax, globalMax, states[split * 32U], R);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t split = 0U; split < splitCount; ++split) {
            const uint32_t slot = split & 1U;
            WaitFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[slot]);
            if (split + 1U < splitCount) {
                const uint32_t next = slot ^ 1U;
                if (freePending[next]) {
                    WaitFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[next]);
                    freePending[next] = false;
                }
                const uint64_t src = (taskSlot + split + 1U) * QfaState().qfaFdStrideElems_ +
                                     static_cast<uint64_t>(rowBase) * srcStride + colStart;
                DataCopyPad(part[next], accumOutGm[src], in, pad);
                SetFlag<HardEvent::MTE2_V>(QfaState().qfaJointOReady_[next]);
            }
            Sub(weight, states[split * 32U], globalMax, R);
            PipeBarrier<PIPE_V>();
            Exp(weight, weight, R);
            PipeBarrier<PIPE_V>();
            Mul(weight, weight, states[split * 32U + 16U], R);
            PipeBarrier<PIPE_V>();
            Add(denom, denom, weight, R);
            PipeBarrier<PIPE_V>();
            Brcb(scales, weight, (R + 7U) / 8U, {1U, 8U});
            PipeBarrier<PIPE_V>();
            fa_base_vector::RowMuls<float>(part[slot], part[slot], scales, R, D, D);
            PipeBarrier<PIPE_V>();
            Add(outAcc, outAcc, part[slot], R * D);
            SetFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[slot]);
            freePending[slot] = true;
        }
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (freePending[slot]) {
                WaitFlag<HardEvent::V_MTE2>(QfaState().qfaJointOFree_[slot]);
            }
        }
        inputQue1.FreeTensor(owner);
        if (writeLse && colStart == 0U)
            WriteLse(denom, globalMax, R, QfaFdGlobalRow(rowBase),
                     static_cast<uint64_t>(kvHead) * groupSize * QfaState().qfaQDescaleStrideToken + tokenStart);
        if constexpr (QfaPolicy::HOST_PLAN) {
            Maxs(denom, denom, 1.0E-10F, R);
            PipeBarrier<PIPE_V>();
        }
        Brcb(denomRows, denom, (R + 7U) / 8U, {1U, 8U});
        PipeBarrier<PIPE_V>();
        fa_base_vector::RowDivs<float>(outAcc, outAcc, denomRows, R, D, D);
        PipeBarrier<PIPE_V>();
        LocalTensor<OUT_T> outStage = QfaState().qfaAccBuf.template Get<OUT_T>();
        for (uint32_t row = 0U; row < R; ++row) {
            Cast(outStage[row * D], outAcc[row * D], AscendC::RoundMode::CAST_RINT, D);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t row = 0U; row < R;) {
            const uint32_t m = QfaFdGlobalRow(rowBase + row);
            const uint32_t count = (R - row < groupSize - m % groupSize) ? R - row : groupSize - m % groupSize;
            const uint64_t dst = (static_cast<uint64_t>(tokenStart + m / groupSize) * constInfo.qHeadNum +
                                  kvHead * groupSize + m % groupSize) *
                                     headDim +
                                 colStart;
            DataCopyExtParams out(count, D * sizeof(OUT_T), 0U, (headDim - D) * sizeof(OUT_T), 0U);
            DataCopyPad(attentionOutGm[dst], outStage[row * D], out);
            row += count;
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline uint32_t QfaJointVec0Rows(uint32_t rows) const
    {
        if constexpr (HEAD_PAIR)
            return rows;
        const uint32_t first = rows <= 8U ? rows : ((rows + 1U) / 2U + 7U) & ~7U;
        return first < rows ? first : rows;
    }

    __aicore__ inline uint32_t QfaJointFdStateRow(uint32_t row, uint32_t rows) const
    {
        const uint32_t first = QfaJointVec0Rows(rows);
        return row < first ? row : constInfo.mBaseSize / 2U + row - first;
    }

    __aicore__ inline void QfaJointReadFdState(LocalTensor<float> dst, GlobalTensor<float> src, uint64_t slot,
                                               uint32_t row, uint32_t count, uint32_t rows, uint32_t splits)
    {
        const uint32_t first = QfaJointVec0Rows(rows);
        uint32_t firstCount = count;
        if (row < first && row + count > first) {
            firstCount = first - row;
        }
        DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0F);
        for (uint32_t offset = 0U; offset < count;) {
            const uint32_t n = offset == 0U ? firstCount : count - firstCount;
            DataCopyExtParams cp(splits, n * sizeof(float), (QfaState().qfaFdStrideElems_ - n) * sizeof(float),
                                 (32U - ((n + 7U) & ~7U)) / 8U, 0U);
            const uint64_t start = slot + QfaJointFdStateRow(row + offset, rows);
            DataCopyPad(dst[offset], src[start], cp, pad);
            DataCopyPad(dst[16U + offset], src[start + constInfo.mBaseSize], cp, pad);
            offset += n;
        }
    }

    __aicore__ inline void HpInit(TPipe* pipe)
    {
        auto& st = QfaState();
        const uint32_t rowBatch = ROW_BATCH != 0U ? ROW_BATCH : st.hpRows;
        const uint32_t pvRows = WIDE_ROW_SCALE ? 16U : st.hpVec2Rows;
        const uint32_t dim = WIDE_ROW_SCALE ? 128U : constInfo.headDimAlign;
        const uint32_t rows = st.hpStateRows;
        const uint32_t scoreBytes = 2U * rowBatch * 512U * sizeof(int32_t);
        const uint32_t fdBytes = 2U * 16U * dim * sizeof(float);
        pipe->InitBuffer(inputQue1, 1, scoreBytes > fdBytes ? scoreBytes : fdBytes);
        pipe->InitBuffer(outputQue2, 1, 2U * rowBatch * 512U);
        pipe->InitBuffer(outputQue1, 1, pvRows * dim * sizeof(OUT_T));
        pipe->InitBuffer(st.hpPvBuf, 2U * pvRows * dim * sizeof(int32_t));
        st.hpPvOwner = st.hpPvBuf.template Get<int32_t>();
        pipe->InitBuffer(tmpBuff1, st.hpScratchBytes);
        pipe->InitBuffer(st.qfaAccBuf, rows * dim * sizeof(float));
        pipe->InitBuffer(st.qfaPreloadBuf, (QFA_BT_CACHE_ELEMS + QFA_QD_CACHE_ELEMS) * sizeof(float));
        pipe->InitBuffer(st.qfaFdLseBuf, 2U * 64U * sizeof(float));
        constexpr uint32_t scaleCols = WIDE_ROW_SCALE ? 64U : 8U;
        pipe->InitBuffer(st.qfaJointQScaleBuf,
                         COMPACT_D256 ?
                             2048U :
                             (rows * scaleCols * sizeof(float) > 4096U ? rows * scaleCols * sizeof(float) : 4096U));
        pipe->InitBuffer(st.qfaJointKScaleBuf, (2U * 512U + (WIDE_ROW_SCALE ? 64U : 0U)) * sizeof(float));
        pipe->InitBuffer(softmaxMaxBuff, 2U * rows * sizeof(float));
        pipe->InitBuffer(softmaxSumBuff, 2U * rows * sizeof(float));
        pipe->InitBuffer(softmaxExpBuff, 2U * rows * sizeof(float));
        pipe->InitBuffer(softmaxMaxDefaultBuff, rows * sizeof(float));
        pipe->InitBuffer(softmaxSumDefaultBuff, rows * sizeof(float));
        softmaxMaxUb = softmaxMaxBuff.template Get<float>();
        softmaxSumUb = softmaxSumBuff.template Get<float>();
        softmaxExpUb = softmaxExpBuff.template Get<float>();
        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.template Get<float>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.template Get<float>();
        Duplicate(softmaxMaxDefaultUb, -3.0E38F, rows);
        Duplicate(softmaxSumDefaultUb, 0.0F, rows);
        PipeBarrier<PIPE_V>();
        st.qfaJointAccOwner_ = st.qfaAccBuf.template Get<float>();
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(mm1ResGm);
        const auto boundary = static_cast<event_t>(pipe->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(boundary);
        WaitFlag<HardEvent::S_MTE2>(boundary);
    }

    __aicore__ inline void HpAllocEvents()
    {
        auto& st = QfaState();
        for (uint32_t j = 0; j < 2U; ++j) {
            st.qfaScoreReady_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            st.qfaScoreFree_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            st.hpKReady[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            st.hpKFree[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            st.qfaJointOReady_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
            st.qfaJointOFree_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
            st.qfaPReady_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
            st.qfaPFree_[j] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        }
        st.qfaJointKReady_ = st.hpKReady[0]; // FD reuses the drained scale event.
        st.qfaJointOutDone_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    }

    __aicore__ inline void HpFreeEvents()
    {
        auto& st = QfaState();
        for (uint32_t j = 0; j < 2U; ++j) {
            if (st.hpKUsed[j]) {
                WaitFlag<HardEvent::V_MTE2>(st.hpKFree[j]);
                st.hpKUsed[j] = false;
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(st.qfaScoreReady_[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(st.qfaScoreFree_[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(st.hpKReady[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(st.hpKFree[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(st.qfaJointOReady_[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(st.qfaJointOFree_[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(st.qfaPReady_[j]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(st.qfaPFree_[j]);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(st.qfaJointOutDone_);
    }

    __aicore__ inline void HpPrefetchScale(const RunInfo& info)
    {
        auto& st = QfaState();
        const uint32_t slot = info.loop & 1U;
        if (st.hpKTag[slot] == info.loop)
            return;
        if (st.hpKUsed[slot]) {
            WaitFlag<HardEvent::V_MTE2>(st.hpKFree[slot]);
            st.hpKUsed[slot] = false;
        }
        LocalTensor<float> ks = st.qfaJointKScaleBuf.template Get<float>()[slot * 512U + (WIDE_ROW_SCALE ? 64U : 0U)];
        auto bt = st.qfaPreloadBuf.template Get<int32_t>();
        uint32_t done = 0U;
        while (done < info.actualSingleProcessSInnerSize) {
            const uint32_t token = info.s2BatchOffset + done;
            const uint32_t page = token / constInfo.kvCacheBlockSize;
            const uint32_t off = token % constInfo.kvCacheBlockSize;
            uint32_t count = constInfo.kvCacheBlockSize - off;
            if (count > info.actualSingleProcessSInnerSize - done)
                count = info.actualSingleProcessSInnerSize - done;
            const int32_t physical = page < st.qfaTaskBtCount_ ?
                                         bt.GetValue(page) :
                                         st.qfaBlockTableGm.GetValue(info.bIdx * constInfo.maxBlockNumPerBatch + page);
            if (physical >= 0) {
                const uint64_t offset = static_cast<uint64_t>(physical) * st.qfaKDescaleStrideBlock +
                                        info.n2Idx * st.qfaKDescaleStrideHead + off * st.qfaKDescaleStrideSlot;
                DataCopyExtParams cp(1U, count * sizeof(float), 0U, 0U, 0U);
                DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0F);
                DataCopyPad(ks[done], st.qfaKDescaleGm[offset], cp, pad);
            } else {
                Duplicate(ks[done], 1.0F, count);
            }
            done += count;
        }
        SetFlag<HardEvent::MTE2_V>(st.hpKReady[slot]);
        st.hpKPending[slot] = true;
        st.hpKTag[slot] = info.loop;
    }

    __aicore__ inline void HpPrefetchInitial(const RunInfo& info)
    {
        auto& st = QfaState();
        SetMSplitInfo(info.actMBaseSize);
        if (mSplitInfo.vecDealM == 0U)
            return;
        st.hpScoreOwner = inputQue1.template AllocTensor<int32_t>();
        st.qfaJointPOwner_ = outputQue2.template AllocTensor<KV_T>();
        st.qfaJointGroupSeq_ = 0U;
        st.qfaJointPrefetchedPanel_ = ~0U;
        for (uint32_t j = 0; j < 2U; ++j) {
            st.qfaJointSFreePending_[j] = false;
            st.qfaJointPFreePending_[j] = false;
            st.hpKTag[j] = ~0U;
        }
        st.qfaJointActive_ = true;
        if constexpr (!OPTIMIZED_VECTOR_SERVICE)
            HpPrefetchScale(info);
    }

    __aicore__ inline void HpPrefetchScore(const RunInfo& info)
    {
        auto& st = QfaState();
        const uint32_t rowBatch = ROW_BATCH != 0U ? ROW_BATCH : st.hpRows;
        if (mSplitInfo.vecDealM == 0U || st.qfaJointPrefetchedPanel_ == info.loop)
            return;
        CrossCoreWaitFlag(constInfo.syncC1V1);
        const uint32_t cols = info.actualSingleProcessSInnerSizeAlign;
        const uint32_t m = mSplitInfo.vecStartM;
        const uint64_t src = (info.loop & 1U) * constInfo.mmResUbSize + static_cast<uint64_t>(m) * cols;
        const uint32_t slot = st.qfaJointGroupSeq_ & 1U;
        if (st.qfaJointSFreePending_[slot]) {
            WaitFlag<HardEvent::V_MTE2>(st.qfaScoreFree_[slot]);
            st.qfaJointSFreePending_[slot] = false;
        }
        const uint32_t count = mSplitInfo.vecDealM < rowBatch ? mSplitInfo.vecDealM : rowBatch;
        DataCopyParams cp(count, cols / 8U, 0U, (512U - cols) / 8U);
        DataCopy(st.hpScoreOwner[slot * rowBatch * 512U].template ReinterpretCast<float>(), mm1ResGm[src], cp);
        SetFlag<HardEvent::MTE2_V>(st.qfaScoreReady_[slot]);
        st.qfaJointPrefetchedPanel_ = info.loop;
    }

    template <uint32_t R, bool UPDATE>
    __aicore__ inline void HpBasicSoftmax(LocalTensor<float> sf, LocalTensor<float> sum, LocalTensor<float> max,
                                          LocalTensor<float> alpha, LocalTensor<float> inSum, LocalTensor<float> inMax)
    {
        constexpr SoftMaxShapeInfo shape{R, 512U, R, 512U};
        constexpr uint32_t scratchBytes = COMPACT_D256 ? 16384U : 32768U;
        constexpr auto tiling = SoftMaxFlashV2TilingFunc(shape, 4U, 4U, scratchBytes, UPDATE, true, false, true);
        auto scratch = tmpBuff1.template Get<uint8_t>();
        if constexpr (COMPACT_D256)
            scratch.SetSize(scratchBytes);
        SoftmaxFlashV2<float, UPDATE, false, true, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            sf, sum, max, sf, alpha, inSum, inMax, scratch, tiling, shape);
    }

    template <bool UPDATE>
    __aicore__ inline void HpApplySoftmax(LocalTensor<float> sf, LocalTensor<float> sum, LocalTensor<float> max,
                                          LocalTensor<float> alpha, LocalTensor<float> inSum, LocalTensor<float> inMax,
                                          uint32_t rows)
    {
        if constexpr (OPTIMIZED_VECTOR_SERVICE) {
            if constexpr (ROW_BATCH == 8U) {
                HpBasicSoftmax<8U, UPDATE>(sf, sum, max, alpha, inSum, inMax);
            } else if (rows > 8U) {
                HpBasicSoftmax<16U, UPDATE>(sf, sum, max, alpha, inSum, inMax);
            } else {
                HpBasicSoftmax<8U, UPDATE>(sf, sum, max, alpha, inSum, inMax);
            }
        } else {
            SoftMaxShapeInfo shape{rows, 512U, rows, 512U};
            auto scratch = tmpBuff1.template Get<uint8_t>();
            SoftmaxFlashV2<float, UPDATE, false, false, false, FIA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
                sf, sum, max, sf, alpha, inSum, inMax, scratch, QfaState().qfaSfTilingCache_, shape);
        }
    }

    __aicore__ inline void HpVec1(const RunInfo& info, const RunInfo* next)
    {
        auto& st = QfaState();
        const uint32_t rowBatch = ROW_BATCH != 0U ? ROW_BATCH : st.hpRows;
        const uint32_t rows = mSplitInfo.vecDealM;
        if (rows == 0U) {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
            return;
        }
        if (!st.qfaJointActive_)
            HpPrefetchInitial(info);
        HpPrefetchScale(info);
        const uint32_t kslot = info.loop & 1U;
        if constexpr (!OPTIMIZED_VECTOR_SERVICE) {
            WaitFlag<HardEvent::MTE2_V>(st.hpKReady[kslot]);
            st.hpKPending[kslot] = false;
        }
        const auto ks = st.qfaJointKScaleBuf.template Get<float>()[kslot * 512U + (WIDE_ROW_SCALE ? 64U : 0U)];
        const uint32_t cols = info.actualSingleProcessSInnerSizeAlign;
        const uint32_t mstart = mSplitInfo.vecStartM;
        const uint64_t sbase = (info.loop & 1U) * constInfo.mmResUbSize + static_cast<uint64_t>(mstart) * cols;
        const uint32_t stateOut = kslot * st.hpStateRows;
        const uint32_t stateIn = (kslot ^ 1U) * st.hpStateRows;
        const int64_t delta = static_cast<int64_t>(info.actS2Size) - info.actS1Size;
        const int64_t firstLimit = static_cast<int64_t>((info.gS1Idx + mstart) / constInfo.gSize) + delta + 1;
        const bool fullVisible =
            info.actualSingleProcessSInnerSize == 512U &&
            (constInfo.sparseMode != 3U || static_cast<int64_t>(info.s2BatchOffset) + 512 <= firstLimit);
        // Softmax produces exp(score - rowMax) in [0, 1]; these scales cannot saturate INT8.
        const bool clampP = !(st.qfaPScaleCache_ > 0.0F && st.qfaPScaleCache_ <= 127.0F);
        for (uint32_t base = 0U; base < rows; base += rowBatch) {
            const uint32_t rc = rows - base < rowBatch ? rows - base : rowBatch;
            const uint32_t slot = (st.qfaJointGroupSeq_ + base / rowBatch) & 1U;
            WaitFlag<HardEvent::MTE2_V>(st.qfaScoreReady_[slot]);
            if (base + rc < rows) {
                const uint32_t dst = slot ^ 1U;
                if (st.qfaJointSFreePending_[dst]) {
                    WaitFlag<HardEvent::V_MTE2>(st.qfaScoreFree_[dst]);
                    st.qfaJointSFreePending_[dst] = false;
                }
                const uint32_t count = rows - base - rc < rowBatch ? rows - base - rc : rowBatch;
                DataCopyParams cp(count, cols / 8U, 0U, (512U - cols) / 8U);
                DataCopy(st.hpScoreOwner[dst * rowBatch * 512U].template ReinterpretCast<float>(),
                         mm1ResGm[sbase + static_cast<uint64_t>(base + rc) * cols], cp);
                SetFlag<HardEvent::MTE2_V>(st.qfaScoreReady_[dst]);
            }
            auto si = st.hpScoreOwner[slot * rowBatch * 512U];
            auto sf = si.template ReinterpretCast<float>();
            Cast(sf, si, RoundMode::CAST_RINT, rc * 512U);
            PipeBarrier<PIPE_V>();
            const bool fixedRows = OPTIMIZED_VECTOR_SERVICE && ROW_BATCH != 8U && rc == 16U;
            if (fixedRows || WIDE_ROW_SCALE) {
                SetMaskNorm();
                SetVectorMask<float>(0ULL, 0xFFFFFFFFFFFFFFFFULL);
                const BinaryRepeatParams repeat{1U, 1U, WIDE_ROW_SCALE ? 1U : 0U, 64U, 64U, WIDE_ROW_SCALE ? 8U : 1U};
                auto qs = st.qfaJointQScaleBuf.template Get<float>()[base * (WIDE_ROW_SCALE ? 64U : 8U)];
                for (uint32_t col = 0U; col < 512U; col += 64U)
                    Mul<float, false>(sf[col], sf[col], qs, MASK_PLACEHOLDER, static_cast<uint8_t>(rc), repeat);
                ResetMask();
            } else {
                fa_base_vector::RowMuls<float>(sf, sf, st.qfaJointQScaleBuf.template Get<float>()[base * 8U], rc, 512U,
                                               512U);
            }
            // Independent Qscale segments share one completion boundary before Kscale reads S.
            PipeBarrier<PIPE_V>();
            if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                if (base == 0U) {
                    WaitFlag<HardEvent::MTE2_V>(st.hpKReady[kslot]);
                    st.hpKPending[kslot] = false;
                }
                if (fixedRows) {
                    SetVectorMask<float>(0ULL, 0xFFFFFFFFFFFFFFFFULL);
                    const BinaryRepeatParams repeat{1U, 1U, 1U, 64U, 64U, 0U};
                    for (uint32_t col = 0U; col < 512U; col += 64U)
                        Mul<float, false>(sf[col], sf[col], ks[col], MASK_PLACEHOLDER, static_cast<uint8_t>(rc),
                                          repeat);
                    ResetMask();
                } else {
                    for (uint32_t row = 0U; row < rc; ++row)
                        Mul(sf[row * 512U], sf[row * 512U], ks, 512U);
                }
            } else {
                for (uint32_t row = 0; row < rc; ++row)
                    Mul(sf[row * 512U], sf[row * 512U], ks, 512U);
            }
            PipeBarrier<PIPE_V>();
            if (base + rc == rows) {
                SetFlag<HardEvent::V_MTE2>(st.hpKFree[kslot]);
                st.hpKUsed[kslot] = true;
            }
            for (uint32_t row = 0; !fullVisible && row < rc; ++row) {
                uint32_t visible = info.actualSingleProcessSInnerSize;
                if (constInfo.sparseMode == 3U) {
                    const uint32_t q = (info.gS1Idx + mstart + base + row) / static_cast<uint32_t>(constInfo.gSize);
                    const int64_t last = static_cast<int64_t>(q) + delta + 1;
                    const uint32_t limit = last > 0 ? static_cast<uint32_t>(last) : 0U;
                    visible = limit <= info.s2BatchOffset ?
                                  0U :
                                  (limit - info.s2BatchOffset < visible ? limit - info.s2BatchOffset : visible);
                }
                uint32_t aligned = visible & ~7U;
                if (visible != aligned) {
                    uint64_t mask[2] = {0xFFULL & ~((1ULL << (visible - aligned)) - 1ULL), 0ULL};
                    Duplicate(sf[row * 512U + aligned], -3.0E38F, mask, 1U, 1U, 0U);
                    aligned += 8U;
                }
                if (aligned < 512U)
                    Duplicate(sf[row * 512U + aligned], -3.0E38F, 512U - aligned);
            }
            PipeBarrier<PIPE_V>();
            if (base == 0U && next != nullptr)
                HpPrefetchScale(*next);
            if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                const uint32_t paddedRows = ROW_BATCH == 8U || rc <= 8U ? 8U : 16U;
                if (paddedRows > rc) {
                    Duplicate(sf[rc * 512U], -3.0E38F, (paddedRows - rc) * 512U);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                SoftMaxShapeInfo shape{rc, 512U, rc, 512U};
                const uint32_t key = (rc << 1U) | (info.isFirstSInnerLoop ? 0U : 1U);
                if (st.qfaSfTilingKey0_ != key) {
                    st.qfaSfTilingCache_ =
                        SoftMaxFlashV2TilingFunc(shape, 4U, 4U, st.hpScratchBytes, !info.isFirstSInnerLoop, false);
                    st.qfaSfTilingKey0_ = key;
                }
            }
            auto sum = softmaxSumUb[stateOut + base];
            auto max = softmaxMaxUb[stateOut + base];
            auto alpha = softmaxExpUb[stateOut + base];
            if (info.isFirstSInnerLoop) {
                HpApplySoftmax<false>(sf, sum, max, alpha, softmaxSumDefaultUb, softmaxMaxDefaultUb, rc);
                PipeBarrier<PIPE_V>();
                Duplicate(alpha, 0.0F, rc);
            } else {
                HpApplySoftmax<true>(sf, sum, max, alpha, softmaxSumUb[stateIn + base], softmaxMaxUb[stateIn + base],
                                     rc);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (!OPTIMIZED_VECTOR_SERVICE) {
                if (st.qfaJointPFreePending_[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(st.qfaPFree_[slot]);
                    st.qfaJointPFreePending_[slot] = false;
                }
            }
            // A completely masked causal row has no P contribution to any PV or FD partial.
            if (firstVisibleQ != 0U) {
                const uint64_t firstRow = static_cast<uint64_t>(firstVisibleQ) * constInfo.gSize;
                const uint32_t start = info.gS1Idx + mstart + base;
                if (start < firstRow) {
                    const uint32_t invalid = firstRow - start < rc ? firstRow - start : rc;
                    Duplicate(sf, 0.0F, invalid * 512U);
                    PipeBarrier<PIPE_V>();
                }
            }
            Muls(sf, sf, st.qfaPScaleCache_, rc * 512U);
            if (clampP) {
                PipeBarrier<PIPE_V>();
                Mins(sf, sf, 127.0F, rc * 512U);
            }
            PipeBarrier<PIPE_V>();
            auto integerCarrier = tmpBuff1.template Get<half>();
            const bool exactHalfRound = OPTIMIZED_VECTOR_SERVICE && !clampP;
            if (exactHalfRound) {
                // All INT8 half-integer rounding boundaries in [0,127] are exact FP16 values.
                Cast(integerCarrier, sf, RoundMode::CAST_FLOOR, rc * 512U);
            } else {
                Cast(sf, sf, RoundMode::CAST_ROUND, rc * 512U);
                PipeBarrier<PIPE_V>();
                Cast(integerCarrier, sf, RoundMode::CAST_NONE, rc * 512U);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(st.qfaScoreFree_[slot]);
            st.qfaJointSFreePending_[slot] = true;
            auto p = st.qfaJointPOwner_[slot * rowBatch * 512U];
            if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                if (st.qfaJointPFreePending_[slot]) {
                    WaitFlag<HardEvent::MTE3_V>(st.qfaPFree_[slot]);
                    st.qfaJointPFreePending_[slot] = false;
                }
            }
            if (exactHalfRound)
                Cast(p, integerCarrier, RoundMode::CAST_ROUND, rc * 512U);
            else
                Cast(p, integerCarrier, RoundMode::CAST_TRUNC, rc * 512U);
            SetFlag<HardEvent::V_MTE3>(st.qfaPReady_[slot]);
            WaitFlag<HardEvent::V_MTE3>(st.qfaPReady_[slot]);
            const uint32_t pRow = WIDE_ROW_SCALE && QfaUseHalfP(info) ? QfaHalfPRow(mstart, base) : mstart + base;
            const uint64_t dst = (info.loop & 1U) * constInfo.vec1ResUbSize + static_cast<uint64_t>(pRow) * 512U;
            DataCopyParams cp(rc, cols / 32U, (512U - cols) / 32U, (512U - cols) / 32U);
            DataCopy(vec1ResGm[dst], p, cp);
            SetFlag<HardEvent::MTE3_V>(st.qfaPFree_[slot]);
            st.qfaJointPFreePending_[slot] = true;
            if constexpr (WIDE_ROW_SCALE) {
                if (QfaUseHalfP(info) && base + rc == 32U)
                    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(QFA_HALF_P_READY);
            }
        }
        st.qfaJointGroupSeq_ += (rows + rowBatch - 1U) / rowBatch;
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        if (next == nullptr) {
            for (uint32_t j = 0; j < 2U; ++j) {
                if (st.qfaJointPFreePending_[j])
                    WaitFlag<HardEvent::MTE3_V>(st.qfaPFree_[j]);
                if (st.qfaJointSFreePending_[j])
                    WaitFlag<HardEvent::V_MTE2>(st.qfaScoreFree_[j]);
            }
            outputQue2.FreeTensor(st.qfaJointPOwner_);
            inputQue1.FreeTensor(st.hpScoreOwner);
            st.qfaJointActive_ = false;
        }
    }

    __aicore__ inline void HpVec2(const RunInfo& info)
    {
        auto& st = QfaState();
        const uint32_t rows = mSplitInfo.vecDealM;
        if (rows == 0U)
            return;
        const uint32_t D = WIDE_ROW_SCALE ? 128U : constInfo.headDimAlign;
        const uint32_t R = WIDE_ROW_SCALE ? 16U : st.hpVec2Rows;
        float valueScale = 0.0F;
        if constexpr (OPTIMIZED_VECTOR_SERVICE)
            valueScale = st.qfaVScaleCache_ / st.qfaPScaleCache_;
        const bool deferValueScale = OPTIMIZED_VECTOR_SERVICE && valueScale > 0.0F && valueScale <= 3.402823466E38F;
        const uint32_t mstart = mSplitInfo.vecStartM;
        const bool packed = WIDE_ROW_SCALE && QfaUseHalfP(info);
        const uint64_t pvBase = (info.loop & 1U) * constInfo.bmm2ResUbSize;
        const uint64_t src = pvBase + static_cast<uint64_t>(packed ? QfaHalfPRow(mstart, 0U) : mstart) * D;
        const uint32_t first = rows < R ? rows : R;
        DataCopy(st.hpPvOwner.template ReinterpretCast<float>(), mm2ResGm[src], first * D);
        SetFlag<HardEvent::MTE2_V>(st.qfaJointOReady_[0]);
        for (uint32_t base = 0; base < rows; base += R) {
            const uint32_t rc = rows - base < R ? rows - base : R;
            const uint32_t slot = (base / R) & 1U;
            WaitFlag<HardEvent::MTE2_V>(st.qfaJointOReady_[slot]);
            if (base + rc < rows) {
                const uint32_t next = slot ^ 1U;
                if (base >= R)
                    WaitFlag<HardEvent::V_MTE2>(st.qfaJointOFree_[next]);
                const uint32_t count = rows - base - rc < R ? rows - base - rc : R;
                const uint32_t pvRow = packed ? QfaHalfPRow(mstart, base + rc) : mstart + base + rc;
                DataCopy(st.hpPvOwner[next * R * D].template ReinterpretCast<float>(), mm2ResGm[pvBase + pvRow * D],
                         count * D);
                SetFlag<HardEvent::MTE2_V>(st.qfaJointOReady_[next]);
            }
            auto ti = st.hpPvOwner[slot * R * D];
            auto tf = ti.template ReinterpretCast<float>();
            Cast(tf, ti, RoundMode::CAST_RINT, rc * D);
            PipeBarrier<PIPE_V>();
            if constexpr (OPTIMIZED_VECTOR_SERVICE) {
                if (!deferValueScale)
                    Muls(tf, tf, valueScale, rc * D);
            } else {
                Muls(tf, tf, st.qfaVScaleCache_ / st.qfaPScaleCache_, rc * D);
            }
            PipeBarrier<PIPE_V>();
            auto acc = st.qfaJointAccOwner_[base * D];
            if (info.isFirstSInnerLoop) {
                Adds(acc, tf, 0.0F, rc * D);
            } else {
                auto alpha = softmaxExpUb[(info.loop & 1U) * st.hpStateRows + base];
                auto brc = tmpBuff1.template Get<float>();
                Brcb(brc, alpha, (rc + 7U) / 8U, {1U, 8U});
                PipeBarrier<PIPE_V>();
                fa_base_vector::RowMuls<float>(acc, acc, brc, rc, D, D);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tf, rc * D);
            }
            SetFlag<HardEvent::V_MTE2>(st.qfaJointOFree_[slot]);
            if (info.isLastS2Loop) {
                if (deferValueScale) {
                    PipeBarrier<PIPE_V>();
                    Muls(acc, acc, valueScale, rc * D);
                    PipeBarrier<PIPE_V>();
                }
                auto sums = softmaxSumUb[(info.loop & 1U) * st.hpStateRows + base];
                auto brc = tmpBuff1.template Get<float>();
                Brcb(brc, sums, (rc + 7U) / 8U, {1U, 8U});
                PipeBarrier<PIPE_V>();
                fa_base_vector::RowDivs<float>(acc, acc, brc, rc, D, D);
                PipeBarrier<PIPE_V>();
                if (info.tndIsS2SplitCore == 0U) {
                    if (writeLse)
                        WriteLse(sums, softmaxMaxUb[(info.loop & 1U) * st.hpStateRows + base], rc,
                                 info.gS1Idx + mstart + base, info.tensorAOffset);
                    auto out = outputQue1.template AllocTensor<OUT_T>();
                    Cast(out, acc, RoundMode::CAST_RINT, rc * D);
                    SetFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
                    WaitFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
                    for (uint32_t row = 0; row < rc;) {
                        const uint32_t globalRow = info.gS1Idx + mstart + base + row;
                        const uint32_t remaining = static_cast<uint32_t>(constInfo.gSize) - globalRow % constInfo.gSize;
                        const uint32_t count =
                            D == constInfo.headDim ? (rc - row < remaining ? rc - row : remaining) : 1U;
                        const uint64_t offset = info.attenOutOffset +
                                                static_cast<uint64_t>(globalRow / constInfo.gSize) *
                                                    constInfo.qHeadNum * constInfo.headDim +
                                                static_cast<uint64_t>(globalRow % constInfo.gSize) * constInfo.headDim;
                        DataCopyExtParams cp(1U, count * constInfo.headDim * sizeof(OUT_T), 0U, 0U, 0U);
                        DataCopyPad(attentionOutGm[offset], out[row * D], cp);
                        row += count;
                    }
                    SetFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
                    WaitFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
                    outputQue1.FreeTensor(out);
                }
            }
        }
        if (info.isLastS2Loop && info.tndIsS2SplitCore != 0U) {
            const uint64_t fd = static_cast<uint64_t>(st.qfaFdCoreIdx_) * st.qfaFdStrideElems_;
            const uint64_t state = fd + QfaJointFdStateRow(mstart, info.actMBaseSize);
            const uint32_t stateBase = (info.loop & 1U) * st.hpStateRows;
            SetFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
            WaitFlag<HardEvent::V_MTE3>(st.qfaPReady_[0]);
            DataCopy(accumOutGm[fd + static_cast<uint64_t>(mstart) * D], st.qfaJointAccOwner_, rows * D);
            DataCopyExtParams cp(1U, rows * sizeof(float), 0U, 0U, 0U);
            DataCopyPad(lseMaxFdGm[state], softmaxMaxUb[stateBase], cp);
            DataCopyPad(lseSumFdGm[state], softmaxSumUb[stateBase], cp);
            SetFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
            WaitFlag<HardEvent::MTE3_V>(st.qfaJointOutDone_);
        }
        WaitFlag<HardEvent::V_MTE2>(st.qfaJointOFree_[0]);
        if (rows > R)
            WaitFlag<HardEvent::V_MTE2>(st.qfaJointOFree_[1]);
    }
};

constexpr uint32_t CUBE_M_TILE = 64U;
constexpr uint32_t QFA_KSCALE_PREP_CHUNK = 1024U;
constexpr uint32_t CUBE_SLOT_NUM = optiling::QFA_A8C8_CUBE_SLOT_NUM;
constexpr uint32_t CUBE_D_MAX = optiling::QFA_A8C8_CUBE_D_MAX;
constexpr uint32_t CUBE_QK_READY_ID = 7U;
constexpr uint32_t CUBE_SOFTMAX_READY_ID = 8U;
constexpr uint32_t CUBE_PV_READY_ID = 9U;

struct QfaCubeWorkspaceLayout {
    uint32_t mMax;
    uint32_t scoreOffset;
    uint32_t pOffset;
    uint32_t oOffset;
    uint32_t vec2Offset;
    uint32_t fdOutOffset;
    uint32_t fdMaxOffset;
    uint32_t fdSumOffset;
    uint32_t stride;
};

__aicore__ inline QfaCubeWorkspaceLayout QfaMakeWorkspaceLayout(const QuantFlashAttnA8C8TilingData* tiling,
                                                                uint32_t minimumRows = 0U)
{
    const uint32_t nativeRows =
        tiling->reservedNativeContext != 0U ?
            (tiling->reservedNativeContext > 64U ? tiling->reservedNativeContext : 64U) :
        tiling->shortQRoute != 0U && tiling->smallDQtile != 0U && (tiling->headDim == 64U || tiling->headDim == 128U) ?
            128U :
            CUBE_M_TILE;
    const uint32_t m = nativeRows > minimumRows ? nativeRows : minimumRows;
    const uint32_t scoreOffset = optiling::QFA_A8C8_CUBE_SEQ_Q_BYTES;
    const uint32_t pOffset = scoreOffset + m * tiling->kvTile * 4U * CUBE_SLOT_NUM;
    const uint32_t oOffset = pOffset + m * tiling->kvTile * 2U * CUBE_SLOT_NUM;
    const uint32_t vec2Offset = oOffset + m * CUBE_D_MAX * 4U * CUBE_SLOT_NUM;
    const uint32_t fdOutOffset = vec2Offset + m * CUBE_D_MAX * 4U * CUBE_SLOT_NUM;
    const uint32_t fdMaxOffset = fdOutOffset + m * CUBE_D_MAX * 4U;
    const uint32_t fdSumOffset = fdMaxOffset + m * 4U;
    return {m, scoreOffset, pOffset, oOffset, vec2Offset, fdOutOffset, fdMaxOffset, fdSumOffset, fdSumOffset + m * 4U};
}

// Both AIC and AIV consume these same host-owned QxG boundaries.
struct QfaPlanTask {
    uint32_t batch, head, row, rows, begin, end, kvLength, tokenBase, qLength, fdSlot;
};

__aicore__ inline void QfaDecodeBatchHead(uint32_t bn, const QuantFlashAttnA8C8TilingData* t, bool headMajor,
                                          uint32_t& batch, uint32_t& head)
{
    if (headMajor) {
        batch = bn % t->batchSize;
        head = bn / t->batchSize;
    } else {
        batch = bn / t->kvHeadNum;
        head = bn % t->kvHeadNum;
    }
}

class QfaPlanIterator {
public:
    __aicore__ inline QfaPlanIterator(__gm__ const optiling::QfaHostPlanData* p, const QuantFlashAttnA8C8TilingData* t,
                                      uint32_t core, __gm__ int32_t* cu, __gm__ int32_t* uq, __gm__ int32_t* uk)
        : p_(p),
          t_(t),
          cu_(cu),
          uq_(uq),
          uk_(uk),
          core_(core)
    {
        headMajor_ = p_->headMajorOrder != 0U;
        bn_ = core == 0U ? 0U : p_->bnEnd[core - 1U];
        gs_ = core == 0U ? 0U : p_->gsEnd[core - 1U];
        k_ = core == 0U ? 0U : p_->kvEnd[core - 1U];
        endBN_ = p_->bnEnd[core];
        endGS_ = p_->gsEnd[core];
        endK_ = p_->kvEnd[core];
    }
    __aicore__ inline bool Next(QfaPlanTask& out)
    {
        while (bn_ < t_->batchSize * t_->kvHeadNum) {
            if (bn_ > endBN_ || (bn_ == endBN_ && (gs_ > endGS_ || (gs_ == endGS_ && k_ >= endK_))))
                return false;
            uint32_t b, head;
            QfaDecodeBatchHead(bn_, t_, headMajor_, b, head);
            const uint32_t blocks = p_->qBlocks[b], panels = p_->kvPanels[b];
            if (gs_ >= blocks) {
                ++bn_;
                gs_ = 0U;
                k_ = 0U;
                continue;
            }
            if (k_ >= panels || panels == 0U) {
                ++gs_;
                k_ = 0U;
                continue;
            }
            if (cachedBatch_ != b) {
                cachedBatch_ = b;
                tokenBase_ = cu_[b];
                qLength_ = cu_[b + 1U] - tokenBase_;
                if (uq_ != nullptr && static_cast<uint32_t>(uq_[b]) < qLength_)
                    qLength_ = uq_[b];
                kvLength_ = uk_ == nullptr ? t_->maxSeqKv : static_cast<uint32_t>(uk_[b]);
            }
            const uint32_t stop = bn_ == endBN_ && gs_ == endGS_ ? endK_ : panels;
            out.batch = b;
            out.head = head;
            out.row = gs_ * p_->m;
            const uint32_t totalRows = qLength_ * t_->groupSize;
            out.rows = out.row < totalRows ? (totalRows - out.row < p_->m ? totalRows - out.row : p_->m) : 0U;
            out.begin = k_ * p_->kvUnit;
            out.end = stop * p_->kvUnit < kvLength_ ? stop * p_->kvUnit : kvLength_;
            out.kvLength = kvLength_;
            out.tokenBase = tokenBase_;
            out.qLength = qLength_;
            out.fdSlot = ~0U;
            const uint32_t groups[2] = {p_->group0[core_], p_->group1[core_]};
            for (uint32_t i = 0; i < 2U; ++i) {
                const uint32_t g = groups[i];
                if (g != ~0U && p_->fdBN[g] == bn_ && p_->fdGS[g] == gs_) {
                    out.fdSlot = i == 0U ? p_->slot0[core_] : p_->slot1[core_];
                }
            }
            ++gs_;
            k_ = 0U;
            return true;
        }
        return false;
    }

private:
    __gm__ const optiling::QfaHostPlanData* p_;
    const QuantFlashAttnA8C8TilingData* t_;
    __gm__ int32_t* cu_;
    __gm__ int32_t* uq_;
    __gm__ int32_t* uk_;
    bool headMajor_;
    uint32_t core_, bn_, gs_, k_, endBN_, endGS_, endK_;
    uint32_t cachedBatch_ = ~0U, tokenBase_ = 0U, qLength_ = 0U, kvLength_ = 0U;
};

template <bool IS_CUBE, typename Service>
__aicore__ inline void QfaExecutePlannedTask(Service& service, const QfaPlanTask& task,
                                             const QuantFlashAttnA8C8TilingData* t, uint32_t fdStride)
{
    if (task.rows == 0U)
        return;
    if constexpr (!IS_CUBE) {
        service.firstVisibleQ = t->maskMode == 3U && task.qLength > task.kvLength ? task.qLength - task.kvLength : 0U;
        service.SetQfaFdLayout(task.fdSlot, fdStride);
        if (task.begin >= task.end) {
            if (task.fdSlot != ~0U && (Service::PAIRED_HEAD_SERVICE || (GetBlockIdx() & 1U) == 0U))
                service.QfaWriteNeutralFdPartial(task.rows, t->headDim);
            return;
        }
        service.QfaPrepareRows(task.batch, task.head, task.row, task.rows, task.kvLength,
                               static_cast<uint64_t>(task.head * t->groupSize) * t->totalTokens + task.tokenBase);
    } else if (task.begin >= task.end) {
        return;
    }
    AttentionCommon::RunInfo info[3];
    const uint32_t panels = (task.end - task.begin + t->kvTile - 1U) / t->kvTile;
    if constexpr (Service::OPTIMIZED_SERVICE) {
        for (uint32_t slot = 0U; slot < (panels < 3U ? panels : 3U); ++slot) {
            auto& x = info[slot];
            x.isValid = true;
            x.bIdx = task.batch;
            x.n2Idx = task.head;
            x.gS1Idx = task.row;
            x.actMBaseSize = task.rows;
            x.actS1Size = task.qLength;
            x.actS2Size = task.kvLength;
            x.curSInnerLoopTimes = panels;
            x.tndIsS2SplitCore = task.fdSlot != ~0U;
            x.tensorAOffset = static_cast<uint64_t>(task.head * t->groupSize) * t->totalTokens + task.tokenBase;
            x.attenOutOffset =
                (static_cast<uint64_t>(task.tokenBase) * t->qHeadNum + task.head * t->groupSize) * t->headDim;
        }
    }
    for (uint32_t step = 0U; step < panels + 2U; ++step) {
        if (step < panels) {
            auto& x = info[step % 3U];
            if constexpr (!Service::OPTIMIZED_SERVICE) {
                x.isValid = true;
                x.bIdx = task.batch;
                x.n2Idx = task.head;
                x.gS1Idx = task.row;
                x.actMBaseSize = task.rows;
                x.actS1Size = task.qLength;
                x.actS2Size = task.kvLength;
                x.curSInnerLoopTimes = panels;
                x.tndIsS2SplitCore = task.fdSlot != ~0U;
                x.tensorAOffset = static_cast<uint64_t>(task.head * t->groupSize) * t->totalTokens + task.tokenBase;
                x.attenOutOffset =
                    (static_cast<uint64_t>(task.tokenBase) * t->qHeadNum + task.head * t->groupSize) * t->headDim;
            }
            x.loop = step;
            x.s2BatchOffset = task.begin + step * t->kvTile;
            x.s2Idx = x.s2BatchOffset / 128U;
            const uint32_t remain = task.end - x.s2BatchOffset;
            x.actualSingleProcessSInnerSize = remain < t->kvTile ? remain : t->kvTile;
            x.actualSingleProcessSInnerSizeAlign = (x.actualSingleProcessSInnerSize + 31U) & ~31U;
            x.isFirstSInnerLoop = step == 0U;
            x.isLastS2Loop = step + 1U == panels;
        }
        if constexpr (IS_CUBE) {
            if (step < panels)
                service.ComputeMm1(info[step % 3U]);
            if (step >= 1U && step - 1U < panels)
                service.ComputeMm2(info[(step + 2U) % 3U]);
        } else {
            if (step == 0U)
                service.QfaJointPrefetchInitial(info[0]);
            if (step >= 1U && step - 1U < panels) {
                auto& previous = info[(step + 2U) % 3U];
                service.QfaJointPrefetchScore(previous);
                service.ComputeVec1(previous, step < panels ? &info[step % 3U] : nullptr);
            }
            if (step >= 2U && step - 2U < panels)
                service.ComputeVec2(info[(step + 1U) % 3U]);
            if (step >= 1U && step < panels)
                service.QfaJointPrefetchScore(info[step % 3U]);
        }
    }
}

template <bool IS_CUBE, typename Service>
__aicore__ inline void QfaExecuteHostPlan(Service& service, __gm__ const optiling::QfaHostPlanData* plan,
                                          const QuantFlashAttnA8C8TilingData* t, uint32_t core, __gm__ int32_t* cu,
                                          __gm__ int32_t* uq, __gm__ int32_t* uk, uint32_t fdStride)
{
    QuantFlashAttnA8C8TilingData pairTiling;
    const auto* iteratorTiling = t;
    if constexpr (Service::PAIRED_HEAD_SERVICE) {
        pairTiling = *t;
        pairTiling.kvHeadNum /= 2U;
        iteratorTiling = &pairTiling;
    }
    QfaPlanIterator iterator(plan, iteratorTiling, core, cu, uq, uk);
    QfaPlanTask task;
    while (iterator.Next(task)) {
        if constexpr (Service::PAIRED_HEAD_SERVICE) {
            if constexpr (!IS_CUBE) {
                if (task.fdSlot != ~0U) {
                    uint32_t g = plan->group0[core];
                    if (g == ~0U || task.fdSlot >= plan->fdPrefix[g] + plan->fdSplits[g])
                        g = plan->group1[core];
                    task.fdSlot = 2U * plan->fdPrefix[g] + (GetBlockIdx() & 1U) * plan->fdSplits[g] + task.fdSlot -
                                  plan->fdPrefix[g];
                }
            }
            task.head = task.head * 2U + (IS_CUBE ? 0U : (GetBlockIdx() & 1U));
        }
        QfaExecutePlannedTask<IS_CUBE>(service, task, t, fdStride);
    }
    // Only plans with actual cross-core partials enter the FD phase.
    if (plan->fdGroups == 0U)
        return;
    if constexpr (!IS_CUBE) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::SyncAll<false>();
    if constexpr (!IS_CUBE) {
        const uint32_t actor = GetBlockIdx();
        if (actor >= plan->fdVectors)
            return;
        const uint32_t startGroup = actor == 0U ? 0U : plan->fdEndGroup[actor - 1U];
        const uint32_t startItem = actor == 0U ? 0U : plan->fdEndItem[actor - 1U] + 1U;
        const uint32_t endGroup = plan->fdEndGroup[actor], endItem = plan->fdEndItem[actor];
        for (uint32_t g = startGroup; g <= endGroup; ++g) {
            uint32_t b, head;
            QfaDecodeBatchHead(plan->fdBN[g], iteratorTiling, plan->headMajorOrder != 0U, b, head);
            const uint32_t tokenBase = cu[b];
            uint32_t qLength = cu[b + 1U] - tokenBase;
            if (uq != nullptr && static_cast<uint32_t>(uq[b]) < qLength)
                qLength = uq[b];
            const uint32_t rowStart = plan->fdGS[g] * plan->m;
            const uint32_t total = qLength * t->groupSize;
            if (rowStart >= total)
                continue;
            const uint32_t rows = total - rowStart < plan->m ? total - rowStart : plan->m;
            if (service.writeLse) {
                const uint32_t kvLength = uk == nullptr ? t->maxSeqKv : static_cast<uint32_t>(uk[b]);
                service.firstVisibleQ = t->maskMode == 3U && qLength > kvLength ? qLength - kvLength : 0U;
                if (kvLength == 0U)
                    service.firstVisibleQ = ~0U;
            }
            service.BindQfaPlanFd(plan->fdPrefix[g], rowStart, rows);
            const uint32_t lo = g == startGroup ? startItem : 0U;
            const uint32_t hi = g == endGroup ? endItem + 1U : plan->fdItems[g];
            for (uint32_t item = lo; item < hi; ++item) {
                uint32_t rowBlock = item / plan->fdColumns;
                const uint32_t fdRows = Service::OPTIMIZED_SERVICE ? plan->fdRows : 16U;
                uint32_t outputHead = head;
                if constexpr (Service::PAIRED_HEAD_SERVICE) {
                    const uint32_t blocks = (rows + fdRows - 1U) / fdRows;
                    const uint32_t child = rowBlock / blocks;
                    rowBlock %= blocks;
                    outputHead = 2U * head + child;
                    service.BindQfaPlanFd(2U * plan->fdPrefix[g] + child * plan->fdSplits[g], rowStart, rows);
                }
                if (rowBlock * fdRows >= rows)
                    continue;
                if (plan->fdColumns == 2U) {
                    service.QfaFlashDecodeColumns(g, tokenBase, 1U, head, plan->fdSplits[g], rowBlock,
                                                  (item & 1U) * 128U);
                } else {
                    service.QfaFlashDecodeTask(g, tokenBase, 1U, outputHead, plan->fdSplits[g], rowBlock, 1U, fdRows);
                }
            }
        }
    }
}

template <typename FiaCubeType, typename S_T, typename P_T, typename O_T>
class QuantFlashAttnA8C8FiaAic {
public:
    __gm__ uint8_t* reservedContext0_ = nullptr;
    __aicore__ inline void Run(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v, __gm__ uint8_t* blockTable,
                               __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedKv,
                               __gm__ uint8_t* out, __gm__ uint8_t* workspace, uint32_t coreIdx, uint32_t coreNum,
                               const QuantFlashAttnA8C8TilingData* tiling, AscendC::TPipe* pipe,
                               __gm__ const optiling::QfaHostPlanData* hostPlan = nullptr)
    {
#ifdef __DAV_C220_CUBE__
        cuSeqQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqLensQ));
        seqUsedQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqUsedQ));
        seqUsedKvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqUsedKv));
        hasSequsedQ_ = seqUsedQ != nullptr;
        hasSequsedKv_ = seqUsedKv != nullptr;
        tiling_ = tiling;
        workspaceLayout_ = QfaMakeWorkspaceLayout(tiling_, FiaCubeType::PAIRED_HEAD_SERVICE ? hostPlan->m * 2U : 0U);

        const uint64_t coreBase = static_cast<uint64_t>(coreIdx) * workspaceLayout_.stride;
        __gm__ uint8_t* coreWorkspace = workspace + coreBase;

        AttentionCommon::ConstInfo constInfo = BuildConstInfo();
        FiaCubeType cube;
        cube.InitQfaParams(BuildQfaParams());
        cube.InitParams(constInfo);
        cube.Init(q, k, v, nullptr, nullptr, cuSeqLensQ, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, blockTable, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, out, nullptr);

        AscendC::GlobalTensor<S_T> gS;
        gS.SetGlobalBuffer(reinterpret_cast<__gm__ S_T*>(coreWorkspace + workspaceLayout_.scoreOffset));
        AscendC::GlobalTensor<P_T> gP;
        gP.SetGlobalBuffer(reinterpret_cast<__gm__ P_T*>(coreWorkspace + workspaceLayout_.pOffset));
        AscendC::GlobalTensor<O_T> gO;
        gO.SetGlobalBuffer(reinterpret_cast<__gm__ O_T*>(coreWorkspace + workspaceLayout_.oOffset));
        cube.InitMm1GlobalTensor(gS);
        cube.InitMm2GlobalTensor(gP, gO);
        cube.InitBuffers(pipe);
        cube.AllocEventID();
        QfaExecuteHostPlan<true>(cube, hostPlan, tiling_, coreIdx, reinterpret_cast<__gm__ int32_t*>(cuSeqLensQ),
                                 reinterpret_cast<__gm__ int32_t*>(seqUsedQ),
                                 reinterpret_cast<__gm__ int32_t*>(seqUsedKv), workspaceLayout_.stride);
        cube.FreeEventID();

#else
        (void)q;
        (void)k;
        (void)v;
        (void)blockTable;
        (void)cuSeqLensQ;
        (void)seqUsedQ;
        (void)seqUsedKv;
        (void)out;
        (void)workspace;
        (void)coreIdx;
        (void)coreNum;
        (void)tiling;
        (void)pipe;
#endif
    }

private:
#ifdef __DAV_C220_CUBE__
    __aicore__ inline QfaInt8Params BuildQfaParams() const
    {
        QfaInt8Params p;
        p.keyHeadStride = tiling_->keyStrideHead;
        p.valueHeadStride = tiling_->valueStrideHead;
        p.qfaKDescaleStrideBlock = tiling_->kDescaleStrideBlock;
        p.qfaKDescaleStrideHead = tiling_->kDescaleStrideHead;
        p.qfaKDescaleStrideSlot = tiling_->kDescaleStrideSlot;
        p.qfaQDescaleStrideToken = tiling_->totalTokens;
        p.qfaVDescaleStrideHead = tiling_->vDescaleStrideHead;
        return p;
    }

    __aicore__ inline AttentionCommon::ConstInfo BuildConstInfo() const
    {
        AttentionCommon::ConstInfo constInfo{};
        constInfo.batchSize = tiling_->batchSize;
        constInfo.gSize = tiling_->groupSize;
        constInfo.qHeadNum = tiling_->qHeadNum;
        constInfo.kvHeadNum = tiling_->kvHeadNum;
        constInfo.headDim = tiling_->headDim;
        constInfo.headDimRope = 0U;
        constInfo.headDimAlign = (tiling_->headDim + 31U) & ~31U;
        constInfo.qSeqSize = tiling_->maxSeqQ;
        constInfo.kvSeqSize = tiling_->maxSeqKv;
        constInfo.kvCacheBlockSize = tiling_->blockSize;
        constInfo.maxBlockNumPerBatch = tiling_->maxBlockNum;
        constInfo.keyBnStride = tiling_->keyStrideBlock;
        constInfo.valueBnStride = tiling_->valueStrideBlock;

        constInfo.mmResUbSize = tiling_->kvTile * workspaceLayout_.mMax;
        constInfo.vec1ResUbSize = tiling_->kvTile * workspaceLayout_.mMax;
        constInfo.bmm2ResUbSize = constInfo.headDimAlign * workspaceLayout_.mMax;
        constInfo.preLoadNum = CUBE_SLOT_NUM;
        constInfo.nBufferMBaseSize = workspaceLayout_.mMax;
        constInfo.mBaseSize = workspaceLayout_.mMax;
        constInfo.s2BaseSize = tiling_->kvTile;
        constInfo.subBlockNum = 2U;
        constInfo.syncC1V1 = CUBE_QK_READY_ID;
        constInfo.syncV1C2 = CUBE_SOFTMAX_READY_ID;
        constInfo.syncC2V2 = CUBE_PV_READY_ID;
        constInfo.actualLenQDims = 0U;
        constInfo.actualLenDims = 0U;
        constInfo.batchContinuous = true;
        constInfo.outputLayout = AttentionCommon::FIA_LAYOUT::TND;
        return constInfo;
    }

#endif

    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedKvGm_;
    const QuantFlashAttnA8C8TilingData* tiling_ = nullptr;
    QfaCubeWorkspaceLayout workspaceLayout_{};
    bool hasSequsedQ_ = false;
    bool hasSequsedKv_ = false;
};

template <typename FiaVecType, typename S_T, typename P_T, typename O_T>
class QuantFlashAttnA8C8FiaVecAiv {
public:
    __gm__ uint8_t* reservedContext0_ = nullptr;
    __gm__ uint8_t* reservedContext1_ = nullptr;
    __aicore__ inline void Run(__gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale,
                               __gm__ uint8_t* blockTable, __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ,
                               __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* out,
                               __gm__ uint8_t* lse, __gm__ uint8_t* workspace, uint32_t coreIdx, uint32_t coreNum,
                               const QuantFlashAttnA8C8TilingData* tiling, AscendC::TPipe* pipe,
                               __gm__ const optiling::QfaHostPlanData* hostPlan = nullptr,
                               uint32_t returnSoftmaxLse = 0U)
    {
#ifdef __DAV_C220_VEC__
        cuSeqQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqLensQ));
        seqUsedQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqUsedQ));
        seqUsedKvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqUsedKv));
        hasSequsedQ_ = seqUsedQ != nullptr;
        hasSequsedKv_ = seqUsedKv != nullptr;
        tiling_ = tiling;
        workspaceLayout_ = QfaMakeWorkspaceLayout(tiling_, FiaVecType::PAIRED_HEAD_SERVICE ? hostPlan->m * 2U : 0U);
        // p3a: prepass inputs (PA-blocked fp32 kDescale + block table) and the dense
        // target region appended after the per-core workspace areas. tiling stores the
        // offset relative to this user workspace, not the raw system workspace.
        kDescaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kDescale));
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
        qfaKScaleDenseGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(workspace + tiling_->kScaleDenseOffset));
        workspace_ = workspace;
        coreNum_ = coreNum;
        if constexpr (AscendC::IsSameType<P_T, int8_t>::value) {
            if (tiling_->kScaleDenseStride != 0U && blockTable != nullptr) {
                pipe->InitBuffer(kScalePrepF32Buf_, QFA_KSCALE_PREP_CHUNK * sizeof(float));
                pipe->InitBuffer(kScalePrepHalfBuf_, QFA_KSCALE_PREP_CHUNK * sizeof(half));
            }
        }

        AttentionCommon::ConstInfo constInfo = BuildConstInfo();
        // The FIA TND output offset calculator needs a real cumulative cu_seqlens_q
        // (uint64) for batch token prefixes; convert the int32 input in workspace.
        __gm__ uint8_t* actualSeqQArg = nullptr;
        const uint64_t coreBase0 = static_cast<uint64_t>(coreIdx) * workspaceLayout_.stride;

        FiaVecType vec;
        vec.QfaState().qfaShortQRoute_ = tiling_->shortQRoute != 0U;
        auto qfaParams = BuildQfaParams();
        if constexpr (FiaVecType::HIGH_PRECISION) {
            qfaParams.hpRows = hostPlan->hpRows;
            qfaParams.hpVec2Rows = hostPlan->hpVec2Rows;
            qfaParams.hpScratchBytes = hostPlan->hpScratchBytes;
            const uint32_t m = hostPlan->m;
            const uint32_t first = m <= 8U ? m : (((m + 1U) / 2U + 7U) & ~7U);
            qfaParams.hpStateRows = FiaVecType::PAIRED_HEAD_SERVICE                        ? ((m + 7U) & ~7U) :
                                    (tiling_->headDim == 128U || tiling_->headDim == 256U) ? ((first + 7U) & ~7U) :
                                                                                             workspaceLayout_.mMax / 2U;
        }
        vec.InitQfaParams(qfaParams);
        vec.InitParams(constInfo);
        vec.Init(nullptr, nullptr, nullptr, nullptr, nullptr, actualSeqQArg, nullptr, qDescale, pScale, vDescale,
                 nullptr, nullptr, kDescale, nullptr, blockTable, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, out, nullptr);

        const uint64_t coreBase = static_cast<uint64_t>(coreIdx) * workspaceLayout_.stride;
        AscendC::GlobalTensor<S_T> gS;
        gS.SetGlobalBuffer(reinterpret_cast<__gm__ S_T*>(workspace + coreBase + workspaceLayout_.scoreOffset) +
                           (FiaVecType::PAIRED_HEAD_SERVICE ?
                                (GetBlockIdx() & 1U) * workspaceLayout_.mMax / 2U * tiling_->kvTile :
                                0U));
        AscendC::GlobalTensor<P_T> gP;
        gP.SetGlobalBuffer(reinterpret_cast<__gm__ P_T*>(workspace + coreBase + workspaceLayout_.pOffset) +
                           (FiaVecType::PAIRED_HEAD_SERVICE ?
                                (GetBlockIdx() & 1U) * workspaceLayout_.mMax / 2U * tiling_->kvTile :
                                0U));
        AscendC::GlobalTensor<O_T> gO;
        gO.SetGlobalBuffer(reinterpret_cast<__gm__ O_T*>(workspace + coreBase + workspaceLayout_.oOffset) +
                           (FiaVecType::PAIRED_HEAD_SERVICE ?
                                (GetBlockIdx() & 1U) * workspaceLayout_.mMax / 2U * constInfo.headDimAlign :
                                0U));
        AscendC::GlobalTensor<float> gVec2;
        gVec2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + coreBase + workspaceLayout_.vec2Offset));
        AscendC::GlobalTensor<float> gFdOut;
        gFdOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + workspaceLayout_.fdOutOffset));
        AscendC::GlobalTensor<float> gFdMax;
        gFdMax.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + workspaceLayout_.fdMaxOffset));
        AscendC::GlobalTensor<float> gFdSum;
        gFdSum.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + workspaceLayout_.fdSumOffset));
        vec.InitVec1GlobalTensor(gP, gS);
        vec.InitVec2GlobalTensor(gVec2, gO);
        vec.InitFlashDecodeGlobalTensor(gFdOut, gFdMax, gFdSum);
        vec.SetQfaFdLayout(coreIdx, workspaceLayout_.stride);
        if constexpr (AscendC::IsSameType<P_T, int8_t>::value) {
            // Bind the route before UB sizing; enabled prepass keeps the legacy layout.
            vec.SetQfaKScaleDense(qfaKScaleDenseGm_, blockTable != nullptr ? tiling_->kScaleDenseStride : 0U,
                                  tiling_->kvHeadNum);
        }
        vec.writeLse = returnSoftmaxLse != 0U;
        if (vec.writeLse)
            vec.lseOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
        vec.InitBuffers(pipe);
        vec.AllocEventID();
        if (vec.writeLse && (hasSequsedQ_ || hasSequsedKv_ || tiling_->maxSeqKv == 0U)) {
            auto fill = vec.QfaState().qfaFdLseBuf.template Get<float>();
            for (uint32_t bh = GetBlockIdx(); bh < tiling_->batchSize * tiling_->qHeadNum; bh += 2U * coreNum) {
                const uint32_t b = bh / tiling_->qHeadNum;
                const uint32_t start = cuSeqQGm_.GetValue(b);
                const uint32_t end = cuSeqQGm_.GetValue(b + 1U);
                uint32_t valid = hasSequsedQ_ ? seqUsedQGm_.GetValue(b) : end - start;
                if ((hasSequsedKv_ ? seqUsedKvGm_.GetValue(b) : tiling_->maxSeqKv) == 0U)
                    valid = 0U;
                if (valid >= end - start)
                    continue;
                Duplicate(fill, __builtin_inff(), 64U);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                const uint64_t head = static_cast<uint64_t>(bh % tiling_->qHeadNum) * tiling_->totalTokens;
                for (uint32_t pos = start + valid; pos < end;) {
                    const uint32_t n = end - pos < 64U ? end - pos : 64U;
                    DataCopyExtParams cp(1U, n * 4U, 0U, 0U, 0U);
                    DataCopyPad(vec.lseOut[head + pos], fill, cp);
                    pos += n;
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
        QfaExecuteHostPlan<false>(vec, hostPlan, tiling_, coreIdx, reinterpret_cast<__gm__ int32_t*>(cuSeqLensQ),
                                  reinterpret_cast<__gm__ int32_t*>(seqUsedQ),
                                  reinterpret_cast<__gm__ int32_t*>(seqUsedKv), workspaceLayout_.stride);
        vec.FreeEventID();

#else
        (void)qDescale;
        (void)kDescale;
        (void)vDescale;
        (void)blockTable;
        (void)pScale;
        (void)cuSeqLensQ;
        (void)seqUsedQ;
        (void)seqUsedKv;
        (void)out;
        (void)workspace;
        (void)coreIdx;
        (void)coreNum;
        (void)tiling;
        (void)pipe;
#endif
    }

private:
#ifdef __DAV_C220_VEC__
    __aicore__ inline QfaInt8Params BuildQfaParams() const
    {
        QfaInt8Params p;
        p.keyHeadStride = tiling_->keyStrideHead;
        p.valueHeadStride = tiling_->valueStrideHead;
        p.qfaKDescaleStrideBlock = tiling_->kDescaleStrideBlock;
        p.qfaKDescaleStrideHead = tiling_->kDescaleStrideHead;
        p.qfaKDescaleStrideSlot = tiling_->kDescaleStrideSlot;
        p.qfaQDescaleStrideToken = tiling_->totalTokens;
        p.qfaVDescaleStrideHead = tiling_->vDescaleStrideHead;
        return p;
    }

    __aicore__ inline AttentionCommon::ConstInfo BuildConstInfo() const
    {
        AttentionCommon::ConstInfo constInfo{};
        constInfo.batchSize = tiling_->batchSize;
        constInfo.gSize = tiling_->groupSize;
        constInfo.qHeadNum = tiling_->qHeadNum;
        constInfo.kvHeadNum = tiling_->kvHeadNum;
        constInfo.headDim = tiling_->headDim;
        constInfo.headDimAlign = (tiling_->headDim + 31U) & ~31U;
        constInfo.qSeqSize = tiling_->maxSeqQ;
        constInfo.kvSeqSize = tiling_->maxSeqKv;
        constInfo.kvCacheBlockSize = tiling_->blockSize;
        constInfo.maxBlockNumPerBatch = tiling_->maxBlockNum;
        constInfo.mmResUbSize = tiling_->kvTile * workspaceLayout_.mMax;
        constInfo.vec1ResUbSize = tiling_->kvTile * workspaceLayout_.mMax;
        constInfo.bmm2ResUbSize = constInfo.headDimAlign * workspaceLayout_.mMax;
        constInfo.preLoadNum = CUBE_SLOT_NUM;
        constInfo.mBaseSize = workspaceLayout_.mMax;
        constInfo.nBufferMBaseSize = workspaceLayout_.mMax;
        constInfo.s2BaseSize = tiling_->kvTile;
        constInfo.sparseMode = tiling_->maskMode;
        constInfo.scaleValue = tiling_->softmaxScale;

        constInfo.syncC1V1 = CUBE_QK_READY_ID;
        constInfo.syncV1C2 = CUBE_SOFTMAX_READY_ID;
        constInfo.syncC2V2 = CUBE_PV_READY_ID;
        constInfo.actualLenQDims = 0U;
        constInfo.actualLenDims = 0U;
        constInfo.batchContinuous = true;
        constInfo.outputLayout = AttentionCommon::FIA_LAYOUT::TND;
        return constInfo;
    }

#endif

    AscendC::GlobalTensor<int32_t> cuSeqQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedQGm_;
    AscendC::GlobalTensor<int32_t> seqUsedKvGm_;
    AscendC::GlobalTensor<float> kDescaleGm_;
    AscendC::GlobalTensor<int32_t> blockTableGm_;
    AscendC::GlobalTensor<half> qfaKScaleDenseGm_;
    TBuf<TPosition::VECCALC> kScalePrepF32Buf_;
    TBuf<TPosition::VECCALC> kScalePrepHalfBuf_;
    __gm__ uint8_t* workspace_ = nullptr;
    uint32_t coreNum_ = 0U;
    const QuantFlashAttnA8C8TilingData* tiling_ = nullptr;
    QfaCubeWorkspaceLayout workspaceLayout_{};
    bool hasSequsedQ_ = false;
    bool hasSequsedKv_ = false;
};

template <AttentionCommon::FIA_LAYOUT KV_LAYOUT, bool OPTIMIZED_VECTOR_SERVICE = false, uint32_t ROW_BATCH = 0U,
          bool WIDE_ROW_SCALE = false, bool HEAD_PAIR = false, uint32_t PAIR_MAX_M = 32U, bool COMPACT_D256 = false>
__aicore__ inline void QfaRunHostPlan(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v, __gm__ uint8_t* qd,
                                      __gm__ uint8_t* kd, __gm__ uint8_t* vd, __gm__ uint8_t* bt, __gm__ uint8_t* ps,
                                      __gm__ uint8_t* cu, __gm__ uint8_t* uq, __gm__ uint8_t* uk, __gm__ uint8_t* out,
                                      __gm__ uint8_t* lse, __gm__ uint8_t* ws,
                                      const QuantFlashAttnA8C8TilingData* tiling, __gm__ uint8_t* rawTiling,
                                      TPipe* pipe, __gm__ uint8_t* metadata = nullptr)
{
    using FIAT = AttentionCommon::FIAType<int8_t, int8_t, bfloat16_t, int8_t, true, false,
                                          AttentionCommon::FIA_LAYOUT::TND, 0U, false, KV_LAYOUT>;
    const auto* payload = reinterpret_cast<__gm__ const optiling::QfaHostPlanTilingPayload*>(rawTiling);
    auto* plan = &payload->plan;
    const uint32_t returnSoftmaxLse = plan->returnSoftmaxLse;
    (void)metadata;
    const uint32_t core = GetBlockIdx() / (g_coreType == AIV ? 2U : 1U);
    if ASCEND_IS_AIC {
        using Cube = QfaInt8Cube<FIAT, WIDE_ROW_SCALE, HEAD_PAIR, PAIR_MAX_M>;
        QuantFlashAttnA8C8FiaAic<Cube, int32_t, int8_t, int32_t> actor;
        actor.Run(q, k, v, bt, cu, uq, uk, out, ws, core, tiling->coreNum, tiling, pipe, plan);
    } else {
        using Vec =
            QfaHighPrecisionVec<FIAT, OPTIMIZED_VECTOR_SERVICE, ROW_BATCH, WIDE_ROW_SCALE, HEAD_PAIR, COMPACT_D256>;
        QuantFlashAttnA8C8FiaVecAiv<Vec, int32_t, int8_t, int32_t> actor;
        actor.Run(qd, kd, vd, bt, ps, cu, uq, uk, out, lse, ws, core, tiling->coreNum, tiling, pipe, plan,
                  returnSoftmaxLse);
    }
}

} // namespace QuantFlashAttnA8C8

#endif // QUANT_FLASH_ATTN_A8_C8_CUBE_H_
