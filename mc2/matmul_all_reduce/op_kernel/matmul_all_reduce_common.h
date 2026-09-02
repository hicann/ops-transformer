/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_all_reduce_common.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_COMMON_H
#define MATMUL_ALL_REDUCE_COMMON_H

#include "lib/hccl/hccl.h"
#ifdef __CCE_KT_TEST__
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

namespace AscendC {
// 代码多数据类型支持
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;

constexpr uint32_t AC_MAX_RANK_NUM = 32;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t HCCL_COMM_DOMAIN_KEY_MAX_LEN = 128;
constexpr uint32_t CAST_BF16_UB_FACTOR = 6; // 1 bf16 data needs 6 bytes tmpbuffer
constexpr int COMM_MODE_CCU = 0;
constexpr int COMM_MODE_AICPU = 1;

template <int commMode = COMM_MODE_CCU>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;
};

template <>
struct HcclTypeSelector<COMM_MODE_AICPU> {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>;
};

struct HcclSignalInfo {
    uint64_t resId; // 在代表event时为eventid，notify时为notifyid
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
};
// TP8卡
struct HcclCombinOpSignalParam {
    HcclSignalInfo noIpcNotifys[AC_MAX_RANK_NUM * 2];
    HcclSignalInfo ipcNotifys[AC_MAX_RANK_NUM * 4];
    HcclSignalInfo noIpcEvents[AC_MAX_RANK_NUM];
    HcclSignalInfo aicpuNotify;
    HcclSignalInfo aicpuOpNotify[2]; // 集合通信AICPU展开资源
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqIds;
};

struct HcclConfig {
    uint8_t determinism; // 确定性计算开关
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[AC_MAX_RANK_NUM];
    uint64_t windowsOut[AC_MAX_RANK_NUM];
    char hcomId[HCCL_COMM_DOMAIN_KEY_MAX_LEN];
    HcclStreamInfo streamInfo[AC_MAX_RANK_NUM];
    HcclCombinOpSignalParam signalInfo;
    HcclConfig config; // 配置参数
};

enum class AntiQuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
};

enum class DebugMode {
    MC2_DEBUG_ONLY_CUBE = 1,
    MC2_DEBUG_ONLY_VECTOR = 2,
    MC2_DEBUG_ONLY_AICPU = 4,
    MC2_DEBUG_WAIT_COMM = 8,
    MC2_DEBUG_TIME_TAKEN = 16,
};

enum MC2_BUFFER_TYPE {
    MC2_BUFFER_TYPE_DEFAULT = 0,
    MC2_BUFFER_TYPE_OUTPUT,
    MC2_BUFFER_TYPE_WINDOW_IN,
    MC2_BUFFER_TYPE_WINDOW_OUT,
    MC2_BUFFER_TYPE_WORKSPACE,
    MC2_BUFFER_TYPE_INPUT,
    MC2_BUFFER_TYPE_COMMOUT,
    MC2_BUFFER_TYPE_END
};

__aicore__ inline uint64_t CalcShapeOffset(uint64_t shapeTypeSize, uint64_t shapeLeftSize, uint64_t shapeRightSize)
{
    return shapeTypeSize * shapeLeftSize * shapeRightSize;
}

#if __CCE_AICORE__ == 200
using namespace matmul;
__aicore__ __inline__ GM_ADDR GetTailA(GM_ADDR aGM, AscendC::tiling::TCubeTiling &tiling, uint32_t size)
{
    uint64_t offset = CalcShapeOffset(sizeof(A_DTYPE), tiling.M, tiling.Ka);
    return aGM + offset * size;
}
__aicore__ __inline__ GM_ADDR GetTailC(GM_ADDR cGM, AscendC::tiling::TCubeTiling &tiling, uint32_t size)
{
    uint64_t offset = CalcShapeOffset(sizeof(C_DTYPE), tiling.M, tiling.N);
    return cGM + offset * size;
}

#else

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT16 || ORIG_DTYPE_X1 == DT_BF16))
#define MC2_NON_QUANT
#endif
#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_FLOAT16 || ORIG_DTYPE_RESIDUAL == DT_FLOAT16))
#define MC2_QUANT_FP16
#define MC2_QUANT
#endif
#if ((ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_RESIDUAL == DT_BF16))
#define MC2_QUANT_BF16
#define MC2_QUANT
#endif
#if ((ORIG_DTYPE_X1 != DT_INT8) && (ORIG_DTYPE_X2 == DT_INT8 || ORIG_DTYPE_X2 == DT_INT4))
#define MC2_WEIGHT_QUANT
#endif

#if (((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)) && \
     ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2) || (ORIG_DTYPE_X2 == DT_HIFLOAT8)))
#define WEIGHT_F8
#endif

#if (((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)) && \
     ((ORIG_DTYPE_X2 == DT_INT8) || (ORIG_DTYPE_X2 == DT_INT4)))
#define WEIGHT_W4_W8
#endif

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_FLOAT16))
#define DAVID_QUANT_INT8_OUT_FP16
#elif ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_INT8) && (ORIG_DTYPE_Y == DT_BF16))
#define DAVID_QUANT_INT8_OUT_BF16
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat X1_FORMAT = CubeFormat::NZ;
#else
constexpr CubeFormat X1_FORMAT = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat X2_FORMAT = CubeFormat::NZ;
#else
constexpr CubeFormat X2_FORMAT = CubeFormat::ND;
#endif

#if defined(FORMAT_Y)
constexpr CubeFormat Y_FORMAT = CubeFormat::ND;
#endif

#if defined(ORIG_DTYPE_X1) && (ORIG_DTYPE_X1 == DT_INT8)
#define DTYPE_LOC_LOCAL int32_t
#else
#define DTYPE_LOC_LOCAL float
#endif

#if ORIG_DTYPE_Y == DT_FLOAT16
constexpr HcclDataType HCCL_DATA_TYPE = AscendC::HCCL_DATA_TYPE_FP16;
#elif ORIG_DTYPE_Y == DT_BF16
constexpr HcclDataType HCCL_DATA_TYPE = AscendC::HCCL_DATA_TYPE_BFP16;
#else
constexpr HcclDataType HCCL_DATA_TYPE = AscendC::HCCL_DATA_TYPE_FP32;
#endif

#if (ORIG_DTYPE_X1 == DT_BF16)
using DTYPE_BIAS_FOR_MC2 = float;
#else
using DTYPE_BIAS_FOR_MC2 = DTYPE_Y;
#endif

struct MC2GmAddrs {
    GM_ADDR aGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR addGM;
    GM_ADDR cGM;
    GM_ADDR workspaceGM;
    GM_ADDR outputGM;
};

struct QuantGmAddrs {
    GM_ADDR antiquantScaleGM;
    GM_ADDR antiquantOffsetGM;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    GM_ADDR offsetGM;
#endif
    GM_ADDR dequantGM;
    GM_ADDR pertokenGM;
};

struct ArnGmAddrs {
    GM_ADDR residualGM;
    GM_ADDR gammaGM;
    GM_ADDR yGM;
    GM_ADDR normOutGM;
};

struct MC2TilingHeader {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    Mc2CcTiling mc2CcTilingComm;
#else
    Mc2Tiling::Mc2Msg msg;
#endif
    Mc2Tiling::RCSTiling param;
};

struct MC2TileInfo {
    AscendC::tiling::TCubeTiling *mmTiling;
    AscendC::HcclHandle hcclHandleId;
    uint64_t aOffset;
    uint64_t aAddrOffset;
    uint64_t cOffset;
    uint64_t cAddrOffset;
};

enum class Mc2CoreType {
    ON_CUBE_AND_VECTOR = 0,
    ON_VECTOR = 1,
    ON_CUBE = 2,
};

// david新增
#ifndef DTYPE_WEIGHT
#define DTYPE_WEIGHT DTYPE_X2
#endif

// for oom check
__aicore__ inline void OOMInit(__gm__ HcclCombinOpParam *context)
{
#ifndef __CCE_KT_TEST__
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(context->WorkSpace), context->WorkSpaceSize);
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(context->windowsIn[context->rankId]), context->winSize);
#endif
}

__aicore__ inline void CastBFtoFloatOnAiv0Impl(GM_ADDR dst, GM_ADDR src, uint32_t size,
                                               TBuf<TPosition::VECCALC> &tmpBuf)
{
    // 1. 初始化global tensor
    GlobalTensor<bfloat16_t> gmSrc;
    GlobalTensor<float> gmDst;
    gmSrc.SetGlobalBuffer((__gm__ bfloat16_t *)(src), size);
    gmDst.SetGlobalBuffer((__gm__ float *)(dst), size);

    // 2. 初始化local tensor
    LocalTensor<bfloat16_t> fullBf16 = tmpBuf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> xLocal = fullBf16[0];
    LocalTensor<float> yLocal = fullBf16[Ceil(size, UB_ALIGN_SIZE) * UB_ALIGN_SIZE].template ReinterpretCast<float>();

    // 3. GM数据拷贝至UB
    uint32_t cpInLen = size * sizeof(bfloat16_t);
    DataCopyExtParams cpInParams{1, cpInLen, 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0}; // 不需要填充数据
    DataCopyPad(xLocal, gmSrc, cpInParams, padParams);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    // 4. 进行cast转换
    Cast(yLocal, xLocal, RoundMode::CAST_NONE, size);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    // 5. UB数据拷贝至GM
    uint32_t cpOutLen = size * sizeof(float);
    DataCopyExtParams cpOutParams{1, cpOutLen, 0, 0, 0};
    DataCopyPad(gmDst, yLocal, cpOutParams);
}

__aicore__ inline void CastBFtoFloatOnAiv0(GM_ADDR dst, GM_ADDR src, uint32_t size, TBuf<TPosition::VECCALC> &tmpBuf)
{
    if (g_coreType == AIC || GetBlockIdx() != 0) {
        return;
    }

    auto tmpBufCount = TOTAL_UB_SIZE / CAST_BF16_UB_FACTOR;
    for (auto offset = 0; offset < size; offset += tmpBufCount) {
        auto calCount = (size - offset) > tmpBufCount ? tmpBufCount : (size - offset);
        CastBFtoFloatOnAiv0Impl(dst + offset * sizeof(float), src + offset * sizeof(bfloat16_t), calCount, tmpBuf);
        if (offset + calCount < size) {
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void WaitFlagDevLocal(int64_t flagID)
{
    CrossCoreWaitFlag(flagID);
}

template <Mc2CoreType type>
__aicore__ inline void Mc2SyncAll()
{
    if constexpr (type == Mc2CoreType::ON_CUBE_AND_VECTOR) {
        SyncAll<false>();
    } else if constexpr (type == Mc2CoreType::ON_VECTOR) {
        SyncAll();
    } else {
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x0, PIPE_FIX>(3);
        WaitFlagDevLocal(3);
    }
}

#endif
} // namespace AscendC
#endif // MATMUL_ALL_REDUCE_COMMON_H
