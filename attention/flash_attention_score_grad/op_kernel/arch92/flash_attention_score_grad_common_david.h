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
 * \file common_david.h
 * \brief
 */

#pragma once

#include "kernel_operator.h"
#include "david/util_david.h"
#include <cstdint>

using namespace davidutil;
using AscendC::Cast;
using AscendC::RoundMode;

namespace commondef {
// 自主管理L1后，计算所需要的L1空间大小，分为preload和非preload两种模式
#define GET_L1_WITH_PRELOAD_USED_SIZE(S1_BASE, D_ALIGN, S2_BASE) \
    ((S1_BASE) * (D_ALIGN) * 20 + (S1_BASE) * (S2_BASE) * 4)
#define GET_L1_NO_PRELOAD_USED_SIZE(S1_BASE, D_ALIGN, S2_BASE) ((S1_BASE) * (D_ALIGN) * 16 + (S1_BASE) * (S2_BASE) * 4)
// 是否开启自主管理L1
#define IS_TSCM_REUSE(HEAD_DIM_ALIGN, T1, IS_DETER) \
    (((!IS_DETER && HEAD_DIM_ALIGN <= (uint32_t)DTemplateType::Aligned256) || \
      (IS_DETER && HEAD_DIM_ALIGN <= (uint32_t)DTemplateType::Aligned192)) && \
     (!IsSameType<T1, float>::value))
// 是否开启L1 preload，限制条件为D <= 192，非BN2模板，非确定性计算，非TND
#define IS_TSCM_PRELOAD(HEAD_DIM_ALIGN, T1, SPLIT_AXIS, IS_DETER, IS_TND) \
    (((HEAD_DIM_ALIGN) <= (uint32_t)DTemplateType::Aligned192) && (((SPLIT_AXIS) == 0) || ((SPLIT_AXIS) == 5)) && \
     ((IS_DETER) == 0) && (!IsSameType<T1, float>::value) && (IS_TND == 0))
// 计算所需要的L0C空间大小
#define GET_SHARED_C1_BUFFER_SZIE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? \
         ((CUBE_BASEM) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) : \
                                            (CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) : \
         ((CUBE_BASEM) > (CUBE_BASEN) ? (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float) : \
                                        (CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)))
// L0C自主管理相关
constexpr uint32_t MIN_L0C_BUF_NUM = 4;
constexpr int32_t DK_DV_L0C_BUF_NUM = 2;
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
#define GET_MM1_MM2_MM3_MAX_L0C_SIZE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) : \
                                       (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))
#define GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    ((L0C_MAX_SIZE - (CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float) * DK_DV_L0C_BUF_NUM)) / \
     (GET_MM1_MM2_MM3_MAX_L0C_SIZE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN)))

#define GET_L0C_BUF_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    (DK_DV_L0C_BUF_NUM + (int32_t)GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) <= 0 ? \
         1 : \
         DK_DV_L0C_BUF_NUM + (int32_t)GET_MAX_REMAIN_L0C_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN))

#define IS_L0C_REUSE(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN, IS_DETER, T1, IS_TND) \
    ((GET_L0C_BUF_NUM(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) >= MIN_L0C_BUF_NUM) && (!IS_DETER) && \
     (!IsSameType<T1, float>::value) && (!IS_TND) && \
     (!(IsSameType<T1, float8_e5m2_t>::value || IsSameType<T1, float8_e4m3_t>::value)))

constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
// 当前判断仅在FP32场景生效，后续需考虑FP16/BF16并结合L0DB开关
#define IS_L0_EXCEED(M, N, K, T1) (M * K * sizeof(T1) > L0_MAX_SIZE || K * N * sizeof(T1) > L0_MAX_SIZE);

#define FagTilingType \
    const __gm__ optiling::fag::FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2David<IS_DETER_NEW( \
        DETER_SPARSE_TYPE)>* __restrict

constexpr uint32_t RESERVED_WORKSPACE_SIZE = 64 * 1024;
constexpr bool INPUT_DISABLE = 0;
constexpr bool INPUT_ENABLE = 1;

constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;

// SPLIT_AXIS Enum
constexpr uint8_t BN2GS1S2 = 0;
constexpr uint8_t BN2 = 1;
constexpr uint8_t BN2S2 = 5;

constexpr uint32_t BSNGD = 1;
constexpr uint32_t SBNGD = 2;
constexpr uint32_t BNGSD = 3;
constexpr uint32_t TND = 4;

constexpr uint32_t MAX_SUM_BNS8 = 0;
constexpr uint32_t MAX_SUM_TND = 1;

constexpr uint32_t PREFIX_LENGTH = 64;
constexpr uint32_t SEQ_ARR_LENGTH = 256;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t INPUT_NUMS = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t FLOAT_BLOCK_SIZE = 32 / sizeof(float);
constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t PSE_PERFORMANCE_MODE = 0x12;
constexpr uint32_t MAX_CUBE_CORE_NUM = 36;
constexpr uint32_t MAX_SUM_REDUCE_AXIS_SIZE = 32;
constexpr uint32_t C0_SIZE = 16;
#if (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
constexpr uint32_t CV_CORE_RATIO = 1;
#else
constexpr uint32_t CV_CORE_RATIO = 2;
#endif
constexpr uint32_t GROUP_TSCM_MASK = 0x4;
constexpr uint32_t TSCM_BUF_NUM = 9;

// sparse mode
constexpr uint32_t NO_MASK = 0; // 未传入attenmask，不做mask操作
constexpr uint32_t ALL_MASK = 1;
constexpr uint32_t LEFT_UP_CAUSAL = 2;    // 左上角点划分的三角部分
constexpr uint32_t RIGHT_DOWN_CAUSAL = 3; // 右下角点划分的下三角部分
constexpr uint32_t BAND = 4;
constexpr uint32_t PREFIX = 5;
constexpr uint32_t PREFIX_COMPRESS = 6;
constexpr uint32_t RIGHT_DOWN_CASUAL_BAND = 7;
constexpr uint32_t BAND_LEFT_UP_CASUAL = 8;

// deter sparse type
// 非确定性
constexpr uint8_t NO_DETER = 0;
// 确定性老实现方案
constexpr uint8_t DETER_OLD = 1;
constexpr uint8_t DETER_DENSE = 2;
constexpr uint8_t DETER_CAUSAL = 3;
constexpr uint8_t DETER_BAND = 4;

// shape type
constexpr uint32_t attenMaskBN2GS1S2 = 0;
constexpr uint32_t attenMaskBS1S2 = 1;
constexpr uint32_t attenMaskS1S2 = 2;
constexpr uint32_t attenMaskTT = 99;

// pse shape type same as tiling
constexpr uint32_t PSE_SHAPE_TYPE_BNSS = 0;
constexpr uint32_t PSE_SHAPE_TYPE_BN1S = 1;
constexpr uint32_t PSE_SHAPE_TYPE_1NSS = 2;
constexpr uint32_t PSE_SHAPE_TYPE_BNHS = 3;
constexpr uint32_t PSE_SHAPE_TYPE_1NHS = 4;
constexpr uint32_t PSE_B_N2_G_SLOPE = 5;
constexpr uint32_t PSE_1_N2_G_SLOPE = 6;

constexpr uint32_t PSE_COMPRESS_H = 1024;
constexpr uint32_t VREG_SIZE = 256;

// swizzle
// 最小Swizzle块数量
constexpr uint32_t MIN_SWIZZLE_S1 = 16384;
// Swizzle块数量，16K对应8块，随S增大倍数增大
constexpr uint32_t BASE_SWIZZLE_BLOCK_NUM = 8;
constexpr uint32_t M_SWIZZLE_SIZE = 32768;
constexpr uint32_t N_SWIZZLE_SIZE = 32768;
constexpr uint32_t SWIZZLE_CONTINUOUS_BLOCK_NUM = 16;
constexpr uint32_t MAX_CONTINUOUS_BLOCK_NUM = 8;

// MM IDX
constexpr uint8_t DQ_IDX = 0;
constexpr uint8_t DK_IDX = 1;
constexpr uint8_t DV_IDX = 2;

struct DeterConstInfo {
    uint8_t usedCubeCoreNum;
    uint8_t usedVectorCoreNum;
    uint8_t eachVecCoreS1Offset;
    uint8_t eachVecCoreS2Offset;
    uint32_t dqEachVectorSize;
    uint32_t dkvEachVectorSize;
    // 确定性计算MTE3的stride
    int64_t deterBStride;
    int64_t deterN2Stride;
    int64_t deterGStride;
    int64_t deterS1oStride;
    uint32_t deterDqkSrcStride;
    uint32_t deterDvSrcStride;
    uint32_t deterDqDstStride;
    uint32_t deterDkDstStride;
    uint32_t deterDvDstStride;
    uint32_t deterVecCoreS1Offset;
    uint32_t deterDkVecCoreS2Offset;
    uint32_t deterDvVecCoreS2Offset;
    bool noNeedDeter;
    // event_id
    event_t eventIDScalarToMte2;
    event_t eventIDMte2ToScalar;
    event_t eventIDScalarToMte3;
    event_t eventIDMte3ToScalar;
    event_t eventIDMte3ToMte2;
    event_t eventIDMte2ToMte3;
};

struct SyncConstInfo {
    event_t mte3ToMte2DsOut;
    event_t mte3ToMte2POut;
};

struct RopeConstInfo {
    /* rope参数 */
    int64_t s1Dr;
    int64_t gS1Dr;
    int64_t n2GS1Dr;
    int64_t s2Dr;
    int64_t n2S2Dr;
    int64_t gDr;
    int64_t n2Dr;
    int64_t bN2Dr;
    int64_t n2GDr;
    int64_t bN2GDr;
    int64_t s2BaseN2Dr;
    int64_t s2BaseBN2Dr;
    int64_t s1BaseN2GDr;
    int64_t s1BaseBN2GDr;
    int64_t s1BaseDr;
    int64_t s2BaseDr;
    int64_t mm1RopeKa; /* rope matmul 跳读参数*/
    int64_t mm1RopeKb;
};

struct FagConstInfo {
    ConstInfo<false> commonConstInfo{0};
    DeterConstInfo deterConstInfo{0};
    SyncConstInfo syncConstInfo;
    RopeConstInfo ropeConstInfo;
    int64_t bSize;
    int64_t n2Size;
    float scaleValue;
    float attenMaskMinValue;

    // 轴的乘积
    int64_t gS1o;
    int64_t n2GS1o;
    int64_t n2GS1oS2o;
    int64_t gS1oS2o;
    int64_t s1oS2o;

    __gm__ uint8_t* seqS1_addr;
    __gm__ uint8_t* seqS2_addr;

    int64_t s1Token;
    int64_t s2Token;
    int64_t sparseMode;
    int64_t s1Outer;
    int64_t s2Outer;
    int64_t s1CvTail;
    int64_t s1Tail;
    int64_t s2Tail;
    float keepProb;
    uint32_t sfmgMaxLoopSize;
    uint32_t dAlignToBlock;
    int64_t mm2Ka;
    int64_t mm2Kb;
    int64_t dRopeSize = 64; // rope旋转的维度

    // swizzle相关
    uint32_t continuousBlockNum = 0; // 核内连续块数量
    int64_t tndSegmentStart = 0;
    int64_t tndSegmentEnd = 0;
    uint64_t totalValidS2TaskNum = 0;
    uint32_t validS2ActiveCoreNum = 0;
    uint32_t columnScheduleMode = 0;
    // 核数
    uint32_t aicCoreNum = 0;
    // sink相关
    uint32_t isSink = 0;
    uint64_t s1SinkOuter = 0;
    uint64_t s2SinkOuter = 0;
    // tnd max sum layout
    uint32_t tndMaxSumLayout = 0;
};

// fp8反量化因子
struct QuantScaleInfo {
    float deqScaleQValue = 1.0f;
    float deqScaleKValue = 1.0f;
    float deqScaleDyValue = 1.0f;
    float deqScaleVValue = 1.0f;
    float deqScaleDsValue = 1.0f;
};

struct FagRunInfo {
    RunInfo<false> commonRunInfo{0};
    QuantScaleInfo quantScaleInfo;
    int64_t s2oIdx;
    int64_t s2CvBegin;
    int64_t s2CvEnd;
    uint8_t taskPingPong;
    uint8_t kvPingPong{0};
    int32_t s1RealSizeAlign2;
    int32_t s2RealSizeAlign2;
    int64_t dAlign16;
    int32_t halfS2RealSize; // vector侧实际的s2基本块大小，如果Cube基本块=128，那么halfS2RealSize=64
    int32_t
        firstHalfS2RealSize; // 当s2RealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的s2 size
    uint8_t qDxPingPongIdx;
    uint8_t isS2IdxNoChange; // s2Idx是否变化
    uint8_t nextS2IdxNoChange{false};
    // BN2MULTIBLK模板使用
    bool isLastS2Outer = false;  // 标记BN2扩展模板中是否是S2轴要处理的最后一个s2outer
    bool isFirstS2Outer = false; // 标记BN2扩展模板中是否是S2轴要处理的第一个s2outer

    uint64_t dxGmOffset;
    uint64_t valueGmOffset;
    uint64_t valueGmOffsetForBn2Blk;
    uint64_t queryGmOffset;
    uint64_t queryGmOffsetForBn2Blk;
    uint64_t keyGmOffset;
    uint64_t keyGmOffsetForBn2Blk;
    uint64_t queryRopeGmOffset;
    uint64_t keyRopeGmOffset;
    uint64_t dqGmOffset;
    uint64_t dkGmOffset;

    // TND需要记录上一次的基本块的信息，用于优化scalar
    int64_t lastBatchIdx = 0;
    int64_t lastBatchTotalBaseIdx = 0;
    int64_t lastBatchTotalS1BOffset = 0;
    int64_t lastBatchTotalS1BOffsetForDv = 0;
    int64_t lastBatchTotalS2BOffset = 0;
    int64_t lastBatchTotalS2BOffsetForDv = 0;
    int64_t lastBatchTotalS1S2SizeAlign = 0;
    int64_t lastBatchTotalS1S2Size = 0;
    int64_t lastBatchTotalS2Size = 0;
    int64_t lastBatchTotalS1BRopeOffset = 0;
    int64_t lastBatchTotalS2BRopeOffset = 0;
    // 只有确定性计算使用
    bool completed = true;
    int64_t dsinkWorkSpaceOffset = 0;
    int64_t sinkN1Idx = 0;
};

template <TPosition pos, uint8_t bufferNum>
class BufferManager {
public:
    __aicore__ inline BufferManager() {}

    __aicore__ inline void Init()
    {
        for (uint8_t i = 0; i < bufferNum; i++) {
            if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
                SetFlag<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
                SetFlag<HardEvent::M_MTE1>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::CO1) {
                srcEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
                dstEventIds_[i] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();
                SetFlag<HardEvent::FIX_M>(dstEventIds_[i]);
            }
        }
    }

    __aicore__ inline uint8_t GetBuffer()
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE1_MTE2>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::M_MTE1>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::FIX_M>(dstEventIds_[bufferId_]);
        }
        return bufferId_;
    }

    __aicore__ inline uint8_t GetBuffer(uint32_t BufferId)
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE1_MTE2>(dstEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::M_MTE1>(dstEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::FIX_M>(dstEventIds_[BufferId]);
        }
        return BufferId;
    }

    __aicore__ inline void EnQue()
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE2_MTE1>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::MTE1_M>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::M_FIX>(srcEventIds_[bufferId_]);
        }
    }

    __aicore__ inline void EnQue(uint32_t BufferId)
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE2_MTE1>(srcEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::MTE1_M>(srcEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::M_FIX>(srcEventIds_[BufferId]);
        }
    }

    __aicore__ inline uint8_t DeQue()
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE2_MTE1>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::MTE1_M>(srcEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::M_FIX>(srcEventIds_[bufferId_]);
        }
        return bufferId_;
    }

    __aicore__ inline uint8_t DeQue(uint32_t BufferId)
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            WaitFlag<HardEvent::MTE2_MTE1>(srcEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            WaitFlag<HardEvent::MTE1_M>(srcEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::CO1) {
            WaitFlag<HardEvent::M_FIX>(srcEventIds_[BufferId]);
        }
        return BufferId;
    }

    __aicore__ inline void ReleaseBuffer(uint32_t BufferId)
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE1_MTE2>(dstEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::M_MTE1>(dstEventIds_[BufferId]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::FIX_M>(dstEventIds_[BufferId]);
        }
    }

    __aicore__ inline void ReleaseBuffer()
    {
        if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
            SetFlag<HardEvent::MTE1_MTE2>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
            SetFlag<HardEvent::M_MTE1>(dstEventIds_[bufferId_]);
        } else if constexpr (pos == TPosition::CO1) {
            SetFlag<HardEvent::FIX_M>(dstEventIds_[bufferId_]);
        }
        if constexpr (bufferNum == 2) {
            bufferId_ = bufferId_ ^ 1U;
        } else {
            bufferId_++;
            bufferId_ = bufferId_ % bufferNum;
        }
    }

    __aicore__ inline void Destory()
    {
        for (uint8_t i = 0; i < bufferNum; i++) {
            if constexpr (pos == TPosition::A1 || pos == TPosition::B1) {
                WaitFlag<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::A2 || pos == TPosition::B2) {
                WaitFlag<HardEvent::M_MTE1>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(dstEventIds_[i]);
            } else if constexpr (pos == TPosition::CO1) {
                WaitFlag<HardEvent::FIX_M>(dstEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(srcEventIds_[i]);
                GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(dstEventIds_[i]);
            }
        }
    }

private:
    TEventID srcEventIds_[bufferNum];
    TEventID dstEventIds_[bufferNum];
    uint8_t bufferId_ = 0;
};

constexpr SyncAllConfig syncAllConfigMte2ToMte2 = {PIPE_MTE2, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte2 = {PIPE_MTE3, PIPE_MTE2};
constexpr SyncAllConfig syncAllConfigMte3ToMte3 = {PIPE_MTE3, PIPE_MTE3};
constexpr SyncAllConfig syncAllConfigMte2ToMte3 = {PIPE_MTE2, PIPE_MTE3};

struct LoopInfo {
    int64_t bIdx{0};
    int64_t n2Idx{0};
    int64_t gIdx{0};
    int64_t s1oIdx{0};
    int64_t s2oIdx{0};
};

struct Bn2MultiBlkInfo {
    int64_t s2oDimIdx{0};
    int64_t s2OuterTmp{0}; // TND场景此值不可直接使用constinfo中的S2Outer
    int64_t s2SparseLeft{0};
    int64_t s2SparseRight{0};
};

__aicore__ inline uint32_t AlignTo(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

__aicore__ inline int64_t AlignTo16(int64_t num)
{
    return (num + 15) >> 4 << 4;
} // 4 is for 16 align

__aicore__ inline int64_t AlignTo32(int64_t num)
{
    return (num + 31) >> 5 << 5;
} // 5 is for 32 align

__aicore__ inline int64_t AlignTo64(int64_t num)
{
    return (num + 63) >> 6 << 6;
} // 6 is for 64 align

__aicore__ constexpr bool IS_DETER_OLD(const uint8_t deterSparseType)
{
    return deterSparseType == DETER_OLD;
}

__aicore__ constexpr bool IS_DETER_NEW(const uint8_t deterSparseType)
{
    return deterSparseType != NO_DETER && deterSparseType != DETER_OLD;
}

__aicore__ constexpr bool NEED_DETER_PREFIX(const uint8_t deterSparseType, bool isTnd)
{
    return isTnd && IS_DETER_NEW(deterSparseType);
}

// 判断DK/DV能否驻留在L0C的宏。
// 计算公式：max(mm1_size, mm2_size, mm3_size) + mm4_size + mm5_size <= L0C_MAX_SIZE
// 其中 mm*_size 对应不同矩阵乘的中间结果大小。
#define IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) \
    (((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) + ((CUBE_BASEN) * (HEAD_DIM_ALIGN) * sizeof(float)) + \
     ((CUBE_BASEN) > (HEAD_DIM_ALIGN) ? (CUBE_BASEM) * (CUBE_BASEN) * sizeof(float) : \
                                        (CUBE_BASEM) * (HEAD_DIM_ALIGN) * sizeof(float))) <= L0C_MAX_SIZE

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(T1) \
    X(T2)

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(IS_ATTEN_MASK, bool, false) \
    X(IS_PSE, bool, false) \
    X(IS_DROP, bool, false) \
    X(IS_TND, bool, false) \
    X(IS_BN2_MULTIBLK, bool, false) \
    X(DETER_SPARSE_TYPE, uint8_t, 0) \
    X(IS_N_EQUAL, bool, false) \
    X(IS_D_NO_EQUAL, bool, false) \
    X(SPLIT_AXIS, uint8_t, 0) \
    X(s1TemplateType, S1TemplateType, S1TemplateType::Aligned128) \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128) \
    X(IS_SWIZZLE, bool, false) \
    X(IS_ROPE, bool, false)

/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type(name) = (default_val),
#define TEMPLATES_DEF \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = \
                  true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
    template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                  CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值, 不带ChildClass的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

/* 4. 生成BASE的有默认值的Template, BASE带ChildClass*/
#define TEMPLATES_DEF_BASE \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 5. 生成BASE的没有默认值的Template, BASE带ChildClass */
#define TEMPLATES_DEF_BASE_NO_DEFAULT \
    template <typename ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
                                       CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 6. 生成BASE的BaseArgs, BASE带ChildClass */
#define TEMPLATE_BASE_ARGS \
    ChildClass, CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

} // namespace commondef
