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
 * \file flash_attention_score_grad_block_cube.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_H

#include "cube_api/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "cube_api/mutex_buffers_policy.h"

namespace FagBaseApi {

constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;

template <bool IS_ROPE = false>
struct PreloadArgs;

template <>
struct PreloadArgs<false> {
    int64_t nextQueryOffset;
    int64_t nextDyOffset;
    int32_t nextMOrN;
    bool copyCurrent = true;
    bool copyNext = false;
};

template <>
struct PreloadArgs<true> {
    int64_t nextQueryOffset;
    int64_t nextDyOffset;
    int64_t nextQueryRopeOffset;
    int64_t nextKeyRopeOffset;
    int32_t nextMOrN;
    bool copyCurrent = true;
    bool copyNext = false;
};

/* ============确定Query的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD, uint8_t IS_SMALL_D_PRELOAD>
struct QL1BuffSelector {
    using TYPE =
        std::conditional_t<IS_SMALL_D_PRELOAD, MutexBuffersPolicy4buff<BufferType::L1>,
                           std::conditional_t<IS_L1_REUSE,
                                              std::conditional_t<IS_L1_PRELOAD, MutexBuffersPolicy3buff<BufferType::L1>,
                                                                 MutexBuffersPolicyDB<BufferType::L1>>,
                                              MutexBuffersPolicySingleBuffer<BufferType::L1>>>;
};

/* ============确定Key的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD, uint8_t IS_SMALL_D_PRELOAD>
struct KL1BuffSelector {
    using TYPE =
        std::conditional_t<IS_SMALL_D_PRELOAD, MutexBuffersPolicy3buff<BufferType::L1>,
                           std::conditional_t<IS_L1_REUSE,
                                              std::conditional_t<IS_L1_PRELOAD, MutexBuffersPolicyDB<BufferType::L1>,
                                                                 MutexBuffersPolicyDB<BufferType::L1>>,
                                              MutexBuffersPolicySingleBuffer<BufferType::L1>>>;
};

/* ============确定Dy的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD, uint8_t IS_SMALL_D_PRELOAD>
struct DyL1BuffSelector {
    using TYPE =
        std::conditional_t<IS_SMALL_D_PRELOAD, MutexBuffersPolicy4buff<BufferType::L1>,
                           std::conditional_t<IS_L1_REUSE,
                                              std::conditional_t<IS_L1_PRELOAD, MutexBuffersPolicy3buff<BufferType::L1>,
                                                                 MutexBuffersPolicyDB<BufferType::L1>>,
                                              MutexBuffersPolicySingleBuffer<BufferType::L1>>>;
};

/* ============确定mm1 mm2 mm3共用几块L0C buffer============= */
template <uint32_t HEAD_DIM_ALIGN>
struct mm1Mm2Mm3L0CBuffSelector {
    using TYPE =
        std::conditional_t<HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned128),
                           std::conditional_t<HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned64),
                                              MutexBuffersPolicy3buff<BufferType::L0C, SyncType::INNER_CORE_SYNC>,
                                              MutexBuffersPolicyDB<BufferType::L0C, SyncType::INNER_CORE_SYNC>>,
                           MutexBuffersPolicySingleBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC>>;
};

/* ============Store dqkv full headDim(>=192)，DKV不驻留L0C场景下开几块L0C buffer============= */
template <uint32_t HEAD_DIM_ALIGN>
struct CommonL0CBufSelector {
    using TYPE = std::conditional_t<HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned256),
                                    MutexBuffersPolicyDB<BufferType::L0C, SyncType::INNER_CORE_SYNC>,
                                    MutexBuffersPolicySingleBuffer<BufferType::L0C, SyncType::INNER_CORE_SYNC>>;
};

/* ============Store dqkv part headDim(>=256), DKV不驻留或仅Dv驻留下开几块L0C buffer============= */
template <bool IS_DV_RESIDENT_L0C>
struct PartHeadDimCommonL0CBufSelector {
    using TYPE = std::conditional_t<IS_DV_RESIDENT_L0C, MutexBuffersPolicyDB<BufferType::L0C>,
                                    MutexBuffersPolicy4buff<BufferType::L0C>>;
};

template <typename T, const int64_t CUBE_BASEM, const int64_t HEAD_DIM_ALIGN>
static constexpr uint32_t GET_DQ_L0_SPLIT_K()
{
    if constexpr (IsSameType<T, float>::value) {
        int64_t splitD = HEAD_DIM_ALIGN > 512 ? 512 : HEAD_DIM_ALIGN;
        return (std::min(L0_SINGLE_BUFFER_SIZE / (CUBE_BASEM * sizeof(float)),
                         L0_SINGLE_BUFFER_SIZE / (splitD * sizeof(float))) /
                C0_SIZE) *
               C0_SIZE;
    } else {
        return static_cast<uint16_t>(S2TemplateType::Aligned128);
    }
}

template <typename T, const int64_t CUBE_BASEN, const int64_t HEAD_DIM_ALIGN>
static constexpr uint32_t GET_DKV_L0_SPLIT_K()
{
    if constexpr (IsSameType<T, float>::value) {
        int64_t splitD = HEAD_DIM_ALIGN > 512 ? 512 : HEAD_DIM_ALIGN;
        return (std::min(L0_SINGLE_BUFFER_SIZE / (splitD * sizeof(float)),
                         L0_SINGLE_BUFFER_SIZE / (CUBE_BASEN * sizeof(float))) /
                C0_SIZE) *
               C0_SIZE;
    } else {
        return static_cast<uint16_t>(S1TemplateType::Aligned128);
    }
}

TEMPLATES_DEF
class FAGBlockCube {
public:
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t C0_SIZE = 16;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    constexpr static bool IS_L1_REUSE =
        GET_IS_L1_REUSE<INPUT_TYPE>(HEAD_DIM_ALIGN, IS_DETER_OLD(DETER_SPARSE_TYPE), false);
    constexpr static bool IS_L1_PRELOAD = GET_IS_L1_PRELOAD<INPUT_TYPE>(
        HEAD_DIM_ALIGN, SPLIT_AXIS, IS_DETER_OLD(DETER_SPARSE_TYPE), IS_TND, false, IS_ROPE);
    constexpr static bool IS_FP32_D_EXCEED_256 = IS_FP32_INPUT &&
                                                 HEAD_DIM_ALIGN > static_cast<uint16_t>(DTemplateType::Aligned256);
    constexpr static bool IS_DKV_RESIDENT_L0C =
        IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) && !IS_FP32_INPUT;
    constexpr static bool IS_SMALL_D_PRELOAD =
        IS_L1_PRELOAD && IS_L1_REUSE && HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned128) &&
        !IS_DROP && SPLIT_AXIS == BN2GS1S2 && DETER_SPARSE_TYPE == NO_DETER && !IS_FP32_INPUT;

    constexpr static uint32_t DQ_L0_SPLIT_K = GET_DQ_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEM, HEAD_DIM_ALIGN>();
    constexpr static uint32_t DKV_L0_SPLIT_K = GET_DKV_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEN, HEAD_DIM_ALIGN>();
    constexpr static bool ENABLE_UNITFLAG =
        HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned768) && !IS_DETER_OLD(DETER_SPARSE_TYPE);
    constexpr static bool IS_DV_RESIDENT_L0C = !IS_FP32_INPUT && !IS_ROPE && !IS_DETER_OLD(DETER_SPARSE_TYPE) &&
                                               (SPLIT_AXIS == BN2GS1S2 || SPLIT_AXIS == BN2S2) &&
                                               HEAD_DIM_ALIGN >= static_cast<uint16_t>(DTemplateType::Aligned192) &&
                                               HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned256);
    constexpr static uint32_t SPLIT_BASE_D =
        DETER_SPARSE_TYPE == DETER_OLD ?
            (HEAD_DIM_ALIGN > static_cast<uint32_t>(DTemplateType::Aligned512) ?
                 static_cast<uint32_t>(DTemplateType::Aligned256) :
                 static_cast<uint32_t>(DTemplateType::Aligned512)) :
            (IS_FP32_INPUT || HEAD_DIM_ALIGN > static_cast<uint32_t>(DTemplateType::Aligned256) ?
                 static_cast<uint32_t>(DTemplateType::Aligned512) :
                 static_cast<uint32_t>(DTemplateType::Aligned128));
    // input global mmemory
    GlobalTensor<INPUT_TYPE> queryGm, keyGm, valueGm, queryRopeGm, keyRopeGm;
    GlobalTensor<OUTDTYPE> dyGm;

    // output global mmemory
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;

    TPipe *pipe;
    FagTilingType tilingData;
    // l1 buffer manage
    MutexBufferManager<BufferType::L1> *l1BufferManagerPtr;
    typename std::conditional<IS_L1_REUSE,
                              typename DyL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, IS_SMALL_D_PRELOAD>::TYPE,
                              std::nullptr_t>::type dYL1Buf;
    typename std::conditional<IS_L1_REUSE, MutexBuffersPolicySingleBuffer<BufferType::L1>, std::nullptr_t>::type vL1Buf;
    typename std::conditional<IS_L1_REUSE,
                              typename QL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, IS_SMALL_D_PRELOAD>::TYPE,
                              std::nullptr_t>::type qL1Buf;
    typename std::conditional<IS_L1_REUSE,
                              typename KL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD, IS_SMALL_D_PRELOAD>::TYPE,
                              std::nullptr_t>::type kL1Buf;
    typename std::conditional<!IS_L1_REUSE, MutexBuffersPolicyDB<BufferType::L1>, std::nullptr_t>::type commonL1Buf;
    // fp32 d>256 l1buffer
    typename std::conditional<IS_FP32_D_EXCEED_256, MutexBuffersPolicySingleBuffer<BufferType::L1>,
                              std::nullptr_t>::type fp32L1Buf1;
    typename std::conditional<IS_FP32_D_EXCEED_256, MutexBuffersPolicySingleBuffer<BufferType::L1>,
                              std::nullptr_t>::type fp32L1Buf2;

    // l0ab buffer manage, double buffer
    MutexBufferManager<BufferType::L0A> l0aBufferManager;
    MutexBufferManager<BufferType::L0B> l0bBufferManager;
    MutexBuffersPolicyDB<BufferType::L0A> l0aBuf;
    MutexBuffersPolicyDB<BufferType::L0B> l0bBuf;

    // l0c buffer manage
    // l0c reuse, dk dv resident
    MutexBufferManager<BufferType::L0C> l0cBufferManager;
    using L0CType = typename mm1Mm2Mm3L0CBuffSelector<HEAD_DIM_ALIGN>::TYPE;
    typename std::conditional<IS_DKV_RESIDENT_L0C, L0CType, std::nullptr_t>::type mm1Mm2Mm3L0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, MutexBuffersPolicySingleBuffer<BufferType::L0C>,
                              std::nullptr_t>::type dkL0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, MutexBuffersPolicySingleBuffer<BufferType::L0C>,
                              std::nullptr_t>::type dvL0CBuf;
    // no l0c reuse, all mm res
    typename std::conditional<!IS_DKV_RESIDENT_L0C, typename CommonL0CBufSelector<HEAD_DIM_ALIGN>::TYPE,
                              std::nullptr_t>::type commonl0CBuf;
    typename PartHeadDimCommonL0CBufSelector<IS_DV_RESIDENT_L0C>::TYPE partHeadDimCommonL0CBuf;
    // special for dSize=192, dSizeV=128
    MutexBuffersPolicySingleBuffer<BufferType::L0C> mm1Mm2Mm3L0CSpecialBuf;
    MutexBuffersPolicySingleBuffer<BufferType::L0C> dkL0CSpecialBuf;
    MutexBuffersPolicySingleBuffer<BufferType::L0C> dvL0CSpecialBuf;
    // special for only Dv L0C Resident, headDim in [128, 256]
    MutexBuffersPolicySingleBuffer<BufferType::L0C> dvL0CResidentBuf;

    bool isDkvL0CResidentForD192Dv128 = false;
    bool isDvL0CResidentForD192ToD256 = false;

    __aicore__ inline FAGBlockCube(){};
    __aicore__ inline ~FAGBlockCube();
    __aicore__ inline bool IsDqkvSplitNEnabled(uint32_t s1Size, uint32_t s2Size);
    __aicore__ inline bool IsDvSplitNEnabled(uint32_t s1Size, uint32_t s2Size);
    constexpr static bool NeedLockL1EachNLoop();
    __aicore__ inline MutexBuffer<BufferType::L0C> GetMmL0CBuffer();
    __aicore__ inline MutexBuffer<BufferType::L0C> GetDkL0CBuffer();
    __aicore__ inline MutexBuffer<BufferType::L0C> GetDvL0CBuffer();
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              MutexBufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo);
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs); // mm1
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs); // mm2
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm3 dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm3 older dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKFixpout(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                               MutexBuffer<BufferType::L0C> mm3L0CBuffer, FagConstInfo &constInfo,
                                               FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo,
                                        bool isDqNeedDeter = false); // mm4 dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter = false); // mm4 older dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQFixpout(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                               MutexBuffer<BufferType::L0C> dkL0CBuffer, FagConstInfo &constInfo,
                                               FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset, bool isFixpOut);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm5 dv
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm5 dv
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyFixpout(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                               MutexBuffer<BufferType::L0C> &dvL0CBuffer, FagConstInfo &constInfo,
                                               FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset, bool isFixpOut);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCube<TEMPLATE_ARGS>::~FAGBlockCube() {}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool FAGBlockCube<TEMPLATE_ARGS>::IsDqkvSplitNEnabled(uint32_t s1Size, uint32_t s2Size)
{
    if constexpr ((SPLIT_AXIS != BN2GS1S2 && SPLIT_AXIS != BN2S2) || IS_ROPE) {
        return false;
    } else if constexpr (IS_FP32_INPUT) {
        return HEAD_DIM_ALIGN > static_cast<uint32_t>(DTemplateType::Aligned256);
    } else if constexpr (HEAD_DIM_ALIGN < static_cast<uint32_t>(DTemplateType::Aligned192)) {
        return false;
    } else if constexpr (HEAD_DIM_ALIGN == static_cast<uint32_t>(DTemplateType::Aligned192)) {
        return !isDkvL0CResidentForD192Dv128;
    } else if constexpr (HEAD_DIM_ALIGN > static_cast<uint32_t>(DTemplateType::Aligned256)) {
        return true;
    } else {
        return s1Size > static_cast<uint32_t>(S1TemplateType::Aligned64) &&
               s2Size > static_cast<uint32_t>(S2TemplateType::Aligned64);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool FAGBlockCube<TEMPLATE_ARGS>::IsDvSplitNEnabled(uint32_t s1Size, uint32_t s2Size)
{
    if constexpr (IS_DV_RESIDENT_L0C) {
        return false;
    } else {
        return IsDqkvSplitNEnabled(s1Size, s2Size);
    }
}

TEMPLATES_DEF_NO_DEFAULT
constexpr bool FAGBlockCube<TEMPLATE_ARGS>::NeedLockL1EachNLoop()
{
    return HEAD_DIM_ALIGN > static_cast<uint32_t>(DTemplateType::Aligned256) || IS_FP32_D_EXCEED_256;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline MutexBuffer<BufferType::L0C> FAGBlockCube<TEMPLATE_ARGS>::GetMmL0CBuffer()
{
    if constexpr (IS_DKV_RESIDENT_L0C) {
        return mm1Mm2Mm3L0CBuf.Get();
    } else {
        if (isDkvL0CResidentForD192Dv128) {
            return mm1Mm2Mm3L0CSpecialBuf.Get();
        } else if (isDvL0CResidentForD192ToD256) {
            return partHeadDimCommonL0CBuf.Get();
        } else {
            return commonl0CBuf.Get();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline MutexBuffer<BufferType::L0C> FAGBlockCube<TEMPLATE_ARGS>::GetDkL0CBuffer()
{
    if constexpr (IS_DKV_RESIDENT_L0C) {
        return dkL0CBuf.Get();
    } else {
        if (isDkvL0CResidentForD192Dv128) {
            return dkL0CSpecialBuf.Get();
        } else if (isDvL0CResidentForD192ToD256) {
            return partHeadDimCommonL0CBuf.Get();
        } else {
            return commonl0CBuf.Get();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline MutexBuffer<BufferType::L0C> FAGBlockCube<TEMPLATE_ARGS>::GetDvL0CBuffer()
{
    if constexpr (IS_DKV_RESIDENT_L0C) {
        return dvL0CBuf.Get();
    } else {
        if (isDkvL0CResidentForD192Dv128) {
            return dvL0CSpecialBuf.Get();
        } else if (isDvL0CResidentForD192ToD256) {
            return dvL0CResidentBuf.Get();
        } else {
            return commonl0CBuf.Get();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                       MutexBufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->l1BufferManagerPtr = l1BuffMgr;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                     GM_ADDR dy, GM_ADDR queryRope, GM_ADDR keyRope,
                                                                     GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                                     GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    queryRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &constInfo)
{
    isDkvL0CResidentForD192Dv128 =
        !IS_FP32_INPUT && ((SPLIT_AXIS == BN2GS1S2 || SPLIT_AXIS == BN2S2) &&
                           (HEAD_DIM_ALIGN == static_cast<uint32_t>(DTemplateType::Aligned192) &&
                            constInfo.commonConstInfo.dSizeV <= static_cast<uint32_t>(DTemplateType::Aligned128))) ||
        IS_ROPE;
    isDvL0CResidentForD192ToD256 =
        IS_DV_RESIDENT_L0C && !isDkvL0CResidentForD192Dv128 &&
        IsDqkvSplitNEnabled(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size);

    // init l1 buffer
    if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
        if constexpr (IS_ROPE) {
            // rope场景mm1 headDim固定为128
            dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * ROPE_D_128 * sizeof(INPUT_TYPE));
            vL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * ROPE_D_128 * sizeof(INPUT_TYPE));
        } else {
            dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
            vL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        }
        qL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        kL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            fp32L1Buf1.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
            fp32L1Buf2.Init(*l1BufferManagerPtr, CUBE_BASEN * SPLIT_BASE_D * sizeof(INPUT_TYPE));
        } else {
            commonL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        }
    }

    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE); // 256 * 1024
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm1Mm2Mm3L0CBuf.Init(l0cBufferManager, CUBE_BASEN > HEAD_DIM_ALIGN ?
                                                   CUBE_BASEM * CUBE_BASEN * sizeof(float) :
                                                   CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(float));
        dkL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
        dvL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
    } else {
        if (isDkvL0CResidentForD192Dv128) {
            mm1Mm2Mm3L0CSpecialBuf.Init(
                l0cBufferManager,
                CUBE_BASEM * static_cast<uint16_t>(DTemplateType::Aligned192) * sizeof(float)); // 98304
            dkL0CSpecialBuf.Init(l0cBufferManager, CUBE_BASEN * static_cast<uint16_t>(DTemplateType::Aligned192) *
                                                       sizeof(float)); // 98304
            dvL0CSpecialBuf.Init(l0cBufferManager, CUBE_BASEN * static_cast<uint16_t>(DTemplateType::Aligned128) *
                                                       sizeof(float)); // 65536
        } else if (isDvL0CResidentForD192ToD256) {
            partHeadDimCommonL0CBuf.Init(l0cBufferManager, CUBE_BASEM * CUBE_BASEN * sizeof(float));
            dvL0CResidentBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
        } else {
            if constexpr (HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned256)) {
                commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE / NUM_TWO);
            } else {
                commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                 PreloadArgs<IS_ROPE> &preloadArgs)
{
    MutexBuffer<BufferType::L1> dyL1Buffer;
    MutexBuffer<BufferType::L1> dyL1NextBuffer;
    MutexBuffer<BufferType::L1> vL1Buffer;
    Nd2NzParams nd2NzParams;

    // load left matrix to L1
    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            dyL1Buffer = dYL1Buf.Get();
        } else {
            dyL1Buffer = dYL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            dyL1NextBuffer = dYL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        dyL1Buffer = dYL1Buf.Get();
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            dyL1Buffer = fp32L1Buf1.Get();
        } else {
            dyL1Buffer = commonL1Buf.Get();
        }
    }
    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        dyL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1Buffer.template GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
        // L1->L0A，L1上的src stride为C0对齐后的SingleM
        nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dyL1Tensor, this->dyGm[runInfo.dyOffset], nd2NzParams);
        dyL1Buffer.UnlockProd();
    }
    // wait pre
    dyL1Buffer.LockCons();
    constexpr uint32_t baseN = (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) ? CUBE_BASEN / 2 : CUBE_BASEN;
    uint32_t nLoops = 1;
    uint32_t realN = runInfo.commonRunInfo.s2RealSize;
    if constexpr (IS_FP32_INPUT) {
        nLoops = (runInfo.commonRunInfo.s2RealSize + baseN - 1) / baseN; // 若为FP32,且Dtemplate>512，需要切分循环两次
        realN = baseN;
    }
    for (uint32_t n = 0; n < nLoops; ++n) {
        if constexpr (IS_FP32_INPUT) {
            if (n == nLoops - 1) {
                uint32_t tailSize = runInfo.commonRunInfo.s2RealSize % baseN;
                realN = tailSize ? tailSize : baseN;
            }
        }
        uint32_t gmNOffset = n * runInfo.vGmS2SplitOffset;
        uint32_t ubOffset = n * baseN;
        // load right matrix to L1
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isS2IdxNoChange;
            vL1Buffer = vL1Buf.Get();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                vL1Buffer = fp32L1Buf2.Get();
            } else {
                vL1Buffer = commonL1Buf.Get();
            }
        }

        if (isCopyRight) {
            vL1Buffer.LockProd();
            LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = realN;
            nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Kb;
            nd2NzParams.dstNzC0Stride = AlignTo16(realN);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(vL1Tensor, this->valueGm[runInfo.commonRunInfo.valueOffset + gmNOffset], nd2NzParams);
            vL1Buffer.UnlockProd();
        }

        // load next left matrix to L1
        if constexpr (IS_L1_PRELOAD) {
            if (preloadArgs.copyNext) {
                dyL1NextBuffer.LockProd();
                LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1NextBuffer.template GetTensor<INPUT_TYPE>();
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = preloadArgs.nextMOrN;
                nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
                // L1->L0A，L1上的src stride为C0对齐后的SingleM
                nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);

                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(dyL1Tensor, this->dyGm[preloadArgs.nextDyOffset], nd2NzParams);
                dyL1NextBuffer.UnlockProd();
            }
        }

        MutexBuffer<BufferType::L0C> mm1L0CBuffer = GetMmL0CBuffer();
        // load l1 to l0ab + mmad
        if constexpr (!ENABLE_UNITFLAG) {
            mm1L0CBuffer.LockProd(); // 反向同步
        }
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
                         (uint32_t)realN,                            // singleN
                         // (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleN
                         (uint32_t)constInfo.commonConstInfo.dSizeV, // singleK
                         false,                                      // isLeftTranspose
                         true,                                       // isRightTranspose
                         true, true, ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE};

        vL1Buffer.LockCons();
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                            L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
                dyL1Buffer.template GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
                mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                         L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
                dyL1Buffer.template GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
                mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);
        }

        vL1Buffer.UnlockCons();

        if constexpr (!ENABLE_UNITFLAG) {
            mm1L0CBuffer.UnlockProd();
            mm1L0CBuffer.LockCons();
        }

        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = CUBE_BASEN;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor[ubOffset], mm1L0CBuffer.GetTensor<CALC_TYPE>(),
                                                            fixpipeParams); // 将matmul结果从L0C搬运到UB
        if constexpr (!ENABLE_UNITFLAG) {
            mm1L0CBuffer.UnlockCons(); // 反向同步
        }
    }
    dyL1Buffer.UnlockCons();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                PreloadArgs<IS_ROPE> &preloadArgs)
{
    MutexBuffer<BufferType::L1> qL1Buffer;
    MutexBuffer<BufferType::L1> qL1NextBuffer;
    MutexBuffer<BufferType::L1> kL1Buffer;
    Nd2NzParams nd2NzParams;

    // load left matrix to L1
    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            qL1Buffer = qL1Buf.Get();
        } else {
            qL1Buffer = qL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            qL1NextBuffer = qL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        qL1Buffer = qL1Buf.Get();
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            qL1Buffer = fp32L1Buf1.Get();
        } else {
            qL1Buffer = commonL1Buf.Get();
        }
    }

    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        qL1Buffer.LockProd();
        LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
        if constexpr (IS_ROPE) {
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = ROPE_D_128;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            // L1->L0A，L1上的src stride为C0对齐后的SingleM
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;

            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.queryOffsetWithRopeForMm12], nd2NzParams);
            nd2NzParams.dValue = ROPE_D_64;
            nd2NzParams.srcDValue = constInfo.mm2Ka / NUM_TWO;
            DataCopy(qL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128],
                     this->queryRopeGm[runInfo.commonRunInfo.qRopeOffset], nd2NzParams);
        } else {
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            // L1->L0A，L1上的src stride为C0对齐后的SingleM
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset], nd2NzParams);
        }
        qL1Buffer.UnlockProd();
    }

    qL1Buffer.LockCons();
    constexpr uint32_t baseN = (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) ? CUBE_BASEN / 2 : CUBE_BASEN;
    uint32_t nLoops = 1;
    uint32_t realN = runInfo.commonRunInfo.s2RealSize;
    if constexpr (IS_FP32_INPUT) {
        nLoops = (runInfo.commonRunInfo.s2RealSize + baseN - 1) / baseN; // 若为FP32,且Dtemplate>512，需要切分循环两次
        realN = baseN;
    }
    for (uint32_t n = 0; n < nLoops; ++n) {
        if constexpr (IS_FP32_INPUT) {
            if (n == nLoops - 1) {
                uint32_t tailSize = runInfo.commonRunInfo.s2RealSize % baseN;
                realN = tailSize ? tailSize : baseN;
            }
        }
        uint32_t gmNOffset = n * runInfo.kGmS2SplitOffset;
        uint32_t ubOffset = n * baseN;
        // load right matrix to L1
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isS2IdxNoChange;
            if (isCopyRight) {
                kL1Buffer = kL1Buf.Get();
            } else {
                kL1Buffer = kL1Buf.GetPre();
            }
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                kL1Buffer = fp32L1Buf2.Get();
            } else {
                kL1Buffer = commonL1Buf.Get();
            }
        }

        if (isCopyRight) {
            kL1Buffer.LockProd();
            LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
            if constexpr (IS_ROPE) {
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
                nd2NzParams.dValue = ROPE_D_128;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Kb;
                nd2NzParams.dstNzC0Stride = runInfo.commonRunInfo.s2AlignedSize;
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(kL1Tensor, this->keyGm[runInfo.keyOffsetWithRopeForMm12], nd2NzParams);
                nd2NzParams.dValue = ROPE_D_64;
                nd2NzParams.srcDValue = constInfo.mm2Kb / NUM_TWO;
                DataCopy(kL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128],
                         this->keyRopeGm[runInfo.commonRunInfo.kRopeOffset], nd2NzParams);
            } else {
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = realN;
                nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Kb;
                nd2NzParams.dstNzC0Stride = AlignTo16(realN);
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(kL1Tensor, this->keyGm[runInfo.commonRunInfo.keyOffset + gmNOffset], nd2NzParams);
            }
            kL1Buffer.UnlockProd();
        }

        // load next left matrix to L1
        if constexpr (IS_L1_PRELOAD) {
            if (preloadArgs.copyNext) {
                qL1NextBuffer.LockProd();
                LocalTensor<INPUT_TYPE> qL1Tensor = qL1NextBuffer.template GetTensor<INPUT_TYPE>();
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = preloadArgs.nextMOrN;
                if constexpr (IS_ROPE) {
                    nd2NzParams.dValue = ROPE_D_128;
                } else {
                    nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
                }
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Ka;
                // L1->L0A，L1上的src stride为C0对齐后的SingleM
                nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(qL1Tensor, this->queryGm[preloadArgs.nextQueryOffset], nd2NzParams);
                if constexpr (IS_ROPE) {
                    nd2NzParams.dValue = ROPE_D_64;
                    nd2NzParams.srcDValue = constInfo.mm2Ka / NUM_TWO;
                    DataCopy(qL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128],
                             this->queryRopeGm[preloadArgs.nextQueryRopeOffset], nd2NzParams);
                }
                qL1NextBuffer.UnlockProd();
            }
        }

        MutexBuffer<BufferType::L0C> mm2L0CBuffer = GetMmL0CBuffer();
        // load l1 to l0ab + mmad
        if constexpr (!ENABLE_UNITFLAG) {
            mm2L0CBuffer.LockProd(); // 反向同步
        }
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
                         (uint32_t)realN,                            // singleN
                         (uint32_t)constInfo.commonConstInfo.dSize,  // singleK
                         false,                                      // isLeftTranspose
                         true,                                       // isRightTranspose
                         true,
                         true,
                         ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE};
        kL1Buffer.LockCons();
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                            L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
                qL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
                mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                         L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
                qL1Buffer.template GetTensor<INPUT_TYPE>(), kL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
                mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        kL1Buffer.UnlockCons();

        if constexpr (!ENABLE_UNITFLAG) {
            mm2L0CBuffer.UnlockProd();
            mm2L0CBuffer.LockCons();
        }

        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
        fixpipeParams.nSize = realN;
        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        // 源Nz矩阵中相邻大Z排布的起始地址偏移
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        // mmResUb上两行之间的间隔，单位：element。
        fixpipeParams.dstStride = CUBE_BASEN;
        // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor[ubOffset], mm2L0CBuffer.GetTensor<CALC_TYPE>(),
                                                            fixpipeParams); // 将matmul结果从L0C搬运到UB
        if constexpr (!ENABLE_UNITFLAG) {
            mm2L0CBuffer.UnlockCons();
        }
    }
    qL1Buffer.UnlockCons();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsKFixpout(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L0C> mm3L0CBuffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset)
{
    if constexpr (IS_WRITE_UB) {
        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        // 源Nz矩阵中相邻大Z排布的起始地址偏移
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        // mmResUb上两行之间的间隔，单位：element。
        fixpipeParams.dstStride = AlignTo16(constInfo.commonConstInfo.dSize);
        // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[gmNOffset], mm3L0CBuffer.GetTensor<CALC_TYPE>(),
                                                 fixpipeParams);
    } else {
        // fixp2GM
        if constexpr (!IS_NZ_OUT) {
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = realN;
            fixpipeParams.mSize = runInfo.commonRunInfo.s1RealSize;
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            fixpipeParams.dstStride = SPLIT_AXIS == BN2 ? AlignTo16(constInfo.commonConstInfo.dSize) : constInfo.mm3Ka;
            fixpipeParams.dualDstCtl = 0;
            fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
            bool needAtomic = (SPLIT_AXIS != BN2) || (IS_BN2_MULTIBLK && !runInfo.isFirstS1Outer);
            if (needAtomic) {
                SetAtomicAdd<CALC_TYPE>();
            }
            if constexpr (IS_ROPE) {
                int64_t offset = runInfo.queryOffsetWithRope;
                if constexpr (SPLIT_AXIS == BN2) {
                    offset = GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN;
                }
                Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[offset], mm3L0CBuffer.GetTensor<CALC_TYPE>(),
                                                         fixpipeParams);
            } else {
                int64_t offset =
                    SPLIT_AXIS == BN2 ? GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN : runInfo.commonRunInfo.queryOffset;
                if constexpr (IS_BN2_MULTIBLK) {
                    offset = GetBlockIdx() * (AlignTo128(constInfo.commonConstInfo.s1Size) * HEAD_DIM_ALIGN);
                    offset += runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN;
                }
                Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                         mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            }
            if (needAtomic) {
                SetAtomicNone();
            }
        } else {
            FixpipeParamsC310<CO2Layout::NZ> fixpipeParams;
            fixpipeParams.nSize = constInfo.dAlign16;
            fixpipeParams.mSize = runInfo.commonRunInfo.s1RealSize;
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            fixpipeParams.dstStride = fixpipeParams.mSize * C0_SIZE;
            fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
            constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::NZ, IS_WRITE_UB};
            SetAtomicAdd<CALC_TYPE>();
            int64_t offset = 0;
            if constexpr (IS_TND) {
                if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
                    offset = runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize *
                                 runInfo.commonRunInfo.actualS1Size * constInfo.dAlign16 +
                             runInfo.commonRunInfo.goIdx * runInfo.commonRunInfo.actualS1Size * constInfo.dAlign16 +
                             runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dAlign16;
                } else {
                    offset = ((__gm__ int64_t *)constInfo.seqS1_addr)[runInfo.commonRunInfo.boIdx - 1] *
                                 constInfo.n2Size * constInfo.commonConstInfo.gSize * constInfo.dAlign16 +
                             runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gSize *
                                 runInfo.commonRunInfo.actualS1Size * constInfo.dAlign16 +
                             runInfo.commonRunInfo.goIdx * runInfo.commonRunInfo.actualS1Size * constInfo.dAlign16 +
                             runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dAlign16;
                }
            } else {
                offset = runInfo.commonRunInfo.boIdx * constInfo.n2Size * constInfo.commonConstInfo.gS1 *
                             constInfo.dAlign16 +
                         runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.gS1 * constInfo.dAlign16 +
                         runInfo.commonRunInfo.goIdx * constInfo.commonConstInfo.s1Size * constInfo.dAlign16 +
                         runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * constInfo.dAlign16;
            }
            Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset], mm3L0CBuffer.GetTensor<CALC_TYPE>(),
                                                     fixpipeParams);
            SetAtomicNone();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsK(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    uint32_t dimSize = static_cast<uint32_t>(constInfo.commonConstInfo.dSize);
    bool isSplitN = IsDqkvSplitNEnabled(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size);
    uint32_t baseD = isSplitN ? SPLIT_BASE_D : dimSize;
    uint32_t nLoops = isSplitN ? (dimSize + SPLIT_BASE_D - 1) / SPLIT_BASE_D : 1;
    MutexBuffer<BufferType::L1> kL1Buffer;
    LocalTensor<INPUT_TYPE> kL1Tensor;
    for (uint32_t n = 0; n < nLoops; ++n) {
        uint32_t realN = dimSize;
        if (isSplitN) {
            realN = SPLIT_BASE_D;
            if (n == nLoops - 1) {
                uint32_t tailSize = dimSize % SPLIT_BASE_D;
                realN = tailSize ? tailSize : SPLIT_BASE_D;
            }
        }
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            if (n == 0) {
                kL1Buffer = kL1Buf.GetReused(runInfo.isNextS2IdxNoChange);
            }
            kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>()[n * runInfo.commonRunInfo.s2AlignedSize * baseD];
        } else {
            // load right matrix to L1
            if constexpr (IS_FP32_D_EXCEED_256) {
                kL1Buffer = fp32L1Buf2.Get();
            } else {
                kL1Buffer = commonL1Buf.Get();
            }
            kL1Buffer.LockProd();
            kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Kb;
            nd2NzParams.dstNzC0Stride = runInfo.commonRunInfo.s2AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(kL1Tensor, this->keyGm[runInfo.commonRunInfo.keyOffset + gmNOffset], nd2NzParams);
            kL1Buffer.UnlockProd();
        }
        if constexpr (NeedLockL1EachNLoop()) {
            kL1Buffer.LockCons();
        } else {
            if (n == 0) {
                kL1Buffer.LockCons();
            }
        }

        MutexBuffer<BufferType::L0C> mm3L0CBuffer = GetMmL0CBuffer();
        // load l1 to l0ab + mmad
        if constexpr (!ENABLE_UNITFLAG) {
            mm3L0CBuffer.LockProd(); // 反向同步
        }
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleK
                         false,                                      // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true,
                         ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                          mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                       mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                       mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        if constexpr (NeedLockL1EachNLoop()) {
            kL1Buffer.UnlockCons();
        } else {
            if (n == nLoops - 1) {
                kL1Buffer.UnlockCons();
            }
        }

        if constexpr (!ENABLE_UNITFLAG) {
            mm3L0CBuffer.UnlockProd();
            mm3L0CBuffer.LockCons();
        }
        IterateMmDsKFixpout<T, IS_WRITE_UB>(outTensor, mm3L0CBuffer, constInfo, runInfo, realN, gmNOffset);
        if constexpr (!ENABLE_UNITFLAG) {
            mm3L0CBuffer.UnlockCons();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsKOlderDeter(
    GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    constexpr uint32_t baseD = SPLIT_BASE_D;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSize + baseD - 1) / baseD; // 尾块处理
    uint32_t realN = baseD;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSize % baseD;
            realN = tailSize ? tailSize : baseD;
        }
        MutexBuffer<BufferType::L1> kL1Buffer;
        LocalTensor<INPUT_TYPE> kL1Tensor;
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            kL1Buffer = kL1Buf.GetReused(runInfo.isNextS2IdxNoChange);
            kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
        } else {
            // load right matrix to L1
            if constexpr (IS_FP32_D_EXCEED_256) {
                kL1Buffer = fp32L1Buf2.Get();
            } else {
                kL1Buffer = commonL1Buf.Get();
            }
            kL1Buffer.LockProd();
            kL1Tensor = kL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Kb;
            nd2NzParams.dstNzC0Stride = runInfo.commonRunInfo.s2AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(kL1Tensor, this->keyGm[runInfo.commonRunInfo.keyOffset + gmNOffset], nd2NzParams);
            kL1Buffer.UnlockProd();
        }
        kL1Buffer.LockCons();

        MutexBuffer<BufferType::L0C> mm3L0CBuffer = GetMmL0CBuffer();
        // load l1 to l0ab + mmad
        mm3L0CBuffer.LockProd();                                     // 反向同步
        MMParam param = {(uint32_t)CUBE_BASEM,                       // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleK
                         false,                                      // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                          mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                       mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf,
                                       mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        }

        kL1Buffer.UnlockCons();

        mm3L0CBuffer.UnlockProd();
        mm3L0CBuffer.LockCons();

        // fixp2GM
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
        fixpipeParams.mSize = CUBE_BASEM;
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        // 源Nz矩阵中相邻大Z排布的起始地址偏移
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        // mmResUb上两行之间的间隔，单位：element。
        fixpipeParams.dstStride = HEAD_DIM_ALIGN;
        // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};

        Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[gmNOffset], mm3L0CBuffer.GetTensor<CALC_TYPE>(),
                                                 fixpipeParams);
        mm3L0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQFixpout(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L0C> dkL0CBuffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset, bool isFixpOut)
{
    if constexpr (IS_WRITE_UB) {
        // fixp2ub
        if (isFixpOut) {
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
            fixpipeParams.mSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            fixpipeParams.dstStride = AlignTo16(constInfo.commonConstInfo.dSize);
            fixpipeParams.dualDstCtl = 1;
            fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
            Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[gmNOffset], dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                                     fixpipeParams);
        }
    } else {
        // fixp2gm
        if (isFixpOut) {
            if constexpr (!IS_NZ_OUT) {
                bool needAtomic = SPLIT_AXIS == BN2GS1S2 ||
                                  (!IS_DKV_RESIDENT_L0C && !isDkvL0CResidentForD192Dv128 && runInfo.isS2IdxNoChange) ||
                                  (SPLIT_AXIS == BN2S2 && !runInfo.isFirstBlock && (runInfo.specialS2Index != -1));
                if constexpr (IS_BN2_MULTIBLK) {
                    needAtomic = (!IS_DKV_RESIDENT_L0C && runInfo.isS2IdxNoChange);
                }
                FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
                fixpipeParams.nSize = realN;
                fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
                fixpipeParams.srcStride = runInfo.commonRunInfo.s2AlignedSize;
                fixpipeParams.dstStride =
                    SPLIT_AXIS != BN2GS1S2 ? AlignTo16(constInfo.commonConstInfo.dSize) : constInfo.mm4Kb;
                fixpipeParams.dualDstCtl = 0;
                fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
                fixpipeParams.params.ndNum = 1;
                fixpipeParams.params.srcNdStride = 0;
                fixpipeParams.params.dstNdStride = 0;
                constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
                if (needAtomic) {
                    SetAtomicAdd<CALC_TYPE>();
                }
                if constexpr (IS_ROPE) {
                    int64_t offset = runInfo.keyOffsetWithRope;
                    if constexpr (SPLIT_AXIS == BN2) {
                        offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                    }
                    Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[offset], dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                                             fixpipeParams);
                } else {
                    int64_t offset = 0;
                    if constexpr (SPLIT_AXIS == BN2S2) {
                        if (runInfo.specialS2Index != -1) {
                            offset = runInfo.specialS2Index * CUBE_BASEM * HEAD_DIM_ALIGN * NUM_TWO;
                        } else {
                            offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                        }
                    } else if constexpr (SPLIT_AXIS == BN2) {
                        offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                    } else {
                        offset = runInfo.commonRunInfo.keyOffset;
                    }

                    Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                             dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                }
                if (needAtomic) {
                    SetAtomicNone();
                }
            } else {
                FixpipeParamsC310<CO2Layout::NZ> fixpipeParams;
                fixpipeParams.nSize = constInfo.dAlign16;
                fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
                fixpipeParams.srcStride = runInfo.commonRunInfo.s2AlignedSize;
                fixpipeParams.dstStride = fixpipeParams.mSize * C0_SIZE;
                fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
                constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::NZ, false};
                SetAtomicAdd<CALC_TYPE>();
                int64_t offset = 0;
                if constexpr (IS_TND) {
                    if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
                        offset =
                            runInfo.commonRunInfo.n2oIdx * runInfo.commonRunInfo.actualS2Size * constInfo.dAlign16 +
                            runInfo.s2oIdx * CUBE_BASEN * constInfo.dAlign16;
                    } else {
                        offset =
                            ((__gm__ int64_t *)constInfo.seqS2_addr)[runInfo.commonRunInfo.boIdx - 1] *
                                constInfo.n2Size * constInfo.dAlign16 +
                            runInfo.commonRunInfo.n2oIdx * runInfo.commonRunInfo.actualS2Size * constInfo.dAlign16 +
                            runInfo.s2oIdx * CUBE_BASEN * constInfo.dAlign16;
                    }
                } else {
                    offset = runInfo.commonRunInfo.boIdx * constInfo.n2Size * constInfo.commonConstInfo.s2Size *
                                 constInfo.dAlign16 +
                             runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Size * constInfo.dAlign16 +
                             runInfo.s2oIdx * CUBE_BASEN * constInfo.dAlign16;
                }

                Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                         dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                SetAtomicNone();
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQ(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter)
{
    LocalTensor<INPUT_TYPE> dsL1Tensor = dSL1Buffer.GetTensor<INPUT_TYPE>();
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
        if (isDqNeedDeter) {
            dsL1Tensor.SetAddrWithOffset(dsL1Tensor, CUBE_BASEM * CUBE_BASEN);
        }
    }
    uint32_t dimSize = static_cast<uint32_t>(constInfo.commonConstInfo.dSize);
    bool isSplitN = IsDqkvSplitNEnabled(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size);
    uint32_t baseD = isSplitN ? SPLIT_BASE_D : dimSize;
    // d=192,dv=128场景使能了L0C累加，此处动态关闭SPLIT_N
    uint32_t nLoops = isSplitN ? (dimSize + SPLIT_BASE_D - 1) / SPLIT_BASE_D : 1; // 尾块处理
    MutexBuffer<BufferType::L1> qL1Buffer;
    LocalTensor<INPUT_TYPE> qL1Tensor;
    for (uint32_t n = 0; n < nLoops; ++n) {
        uint32_t realN = dimSize;
        if (isSplitN) {
            realN = SPLIT_BASE_D;
            if (n == nLoops - 1) {
                uint32_t tailSize = dimSize % SPLIT_BASE_D;
                realN = tailSize ? tailSize : SPLIT_BASE_D;
            }
        }
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            if (n == 0) {
                qL1Buffer = qL1Buf.GetReused();
            }
            qL1Tensor =
                qL1Buffer.template GetTensor<INPUT_TYPE>()[n * runInfo.s1AlignedSize * baseD]; // 切d场景存在偏移
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                qL1Buffer = fp32L1Buf1.Get();
            } else {
                qL1Buffer = commonL1Buf.Get();
            }
            qL1Buffer.LockProd();
            qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset + gmNOffset], nd2NzParams);
            qL1Buffer.UnlockProd();
        }
        if constexpr (NeedLockL1EachNLoop()) {
            qL1Buffer.LockCons();
        } else {
            if (n == 0) {
                qL1Buffer.LockCons();
            }
        }

        MutexBuffer<BufferType::L0C> dkL0CBuffer = GetDkL0CBuffer();
        // load l1 to l0ab + mmad
        if constexpr (!ENABLE_UNITFLAG) {
            dkL0CBuffer.LockProd(); // 反向同步
        }
        bool enPartialSum = false; // control l0c accumulate sum
        bool isFixpOut = true;

        if constexpr (!IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            if ((IS_DKV_RESIDENT_L0C || isDkvL0CResidentForD192Dv128) && !(SPLIT_AXIS == BN2)) {
                enPartialSum = runInfo.isS2IdxNoChange;
                isFixpOut = !runInfo.isNextS2IdxNoChange;
            }
            if constexpr (IS_BN2_MULTIBLK && IS_DKV_RESIDENT_L0C) {
                enPartialSum = runInfo.isS2IdxNoChange;
                isFixpOut = !runInfo.isNextS2IdxNoChange;
            }
        }
        MMParam param = {(uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         !enPartialSum,
                         !ENABLE_UNITFLAG ? UNITFLAG_DISABLE : (isFixpOut ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE)};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                          param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                       param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                       param);
        }
        if constexpr (NeedLockL1EachNLoop()) {
            qL1Buffer.UnlockCons();
        } else {
            if (n == nLoops - 1) {
                qL1Buffer.UnlockCons();
            }
        }

        if constexpr (!ENABLE_UNITFLAG) {
            dkL0CBuffer.UnlockProd();
            dkL0CBuffer.LockCons();
        }
        IterateMmDsQFixpout<T, IS_WRITE_UB>(outTensor, dkL0CBuffer, constInfo, runInfo, realN, gmNOffset, isFixpOut);
        if constexpr (!ENABLE_UNITFLAG) {
            dkL0CBuffer.UnlockCons();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQOlderDeter(
    GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter)
{
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    LocalTensor<INPUT_TYPE> dsL1Tensor = dSL1Buffer.GetTensor<INPUT_TYPE>();
    if (isDqNeedDeter) {
        dsL1Tensor.SetAddrWithOffset(dsL1Tensor, CUBE_BASEM * CUBE_BASEN);
    }
    constexpr uint32_t baseD = SPLIT_BASE_D;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSize + baseD - 1) / baseD; // 尾块处理
    uint32_t realN = baseD;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSize % baseD;
            realN = tailSize ? tailSize : baseD;
        }
        MutexBuffer<BufferType::L1> qL1Buffer;
        LocalTensor<INPUT_TYPE> qL1Tensor;
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            qL1Buffer = qL1Buf.GetReused();
            qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                qL1Buffer = fp32L1Buf1.Get();
            } else {
                qL1Buffer = commonL1Buf.Get();
            }
            qL1Buffer.LockProd();
            qL1Tensor = qL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset + gmNOffset], nd2NzParams);
            qL1Buffer.UnlockProd();
        }
        qL1Buffer.LockCons();

        MutexBuffer<BufferType::L0C> dkL0CBuffer = GetDkL0CBuffer();
        // load l1 to l0ab + mmad
        dkL0CBuffer.LockProd();                                      // 反向同步
        MMParam param = {(uint32_t)CUBE_BASEN,                       // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                          param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                       param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(dsL1Tensor, qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                       param);
        }

        qL1Buffer.UnlockCons();

        dkL0CBuffer.UnlockProd();
        dkL0CBuffer.LockCons();
        // fixp2gm
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = CUBE_BASEN;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = HEAD_DIM_ALIGN;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[gmNOffset], dkL0CBuffer.GetTensor<CALC_TYPE>(),
                                                 fixpipeParams);
        dkL0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDyFixpout(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L0C> &dvL0CBuffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo, uint32_t realN, uint64_t gmNOffset, bool isFixpOut)
{
    if constexpr (!IS_NZ_OUT) {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr ((SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) || IS_FP32_INPUT) {
            fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
            fixpipeParams.nSize = realN;
        } else {
            fixpipeParams.mSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1;
            fixpipeParams.nSize = realN;
        }
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = (SPLIT_AXIS == BN2S2 || IS_BN2_MULTIBLK) ?
                                      AlignTo16(constInfo.commonConstInfo.dSizeV) :
                                      constInfo.commonConstInfo.mm1Kb;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        if constexpr (!IS_WRITE_UB) {
            // S1S2模板与BN2模板的dv均会输出到gm
            if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
                if constexpr (IsSameType<T, half>::value) {
                    fixpipeParams.quantPre = QuantMode_t::F322F16;
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    fixpipeParams.quantPre = QuantMode_t::F322BF16;
                }
                Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.commonRunInfo.valueOffset + gmNOffset],
                                                         dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            } else if constexpr (IS_BN2_MULTIBLK) {
                if (isFixpOut) {
                    bool needAtomic = ((!IS_DKV_RESIDENT_L0C) && runInfo.isS2IdxNoChange);
                    if (needAtomic) {
                        SetAtomicAdd<CALC_TYPE>();
                    }
                    int64_t offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[offset], dvL0CBuffer.GetTensor<CALC_TYPE>(),
                                                             fixpipeParams);
                    if (needAtomic) {
                        SetAtomicNone();
                    }
                }
            } else if constexpr (SPLIT_AXIS == BN2GS1S2) {
                if (isFixpOut) {
                    SetAtomicAdd<CALC_TYPE>();
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.commonRunInfo.valueOffset + gmNOffset],
                                                             dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                    SetAtomicNone();
                }
            } else { // BNS2
                if (isFixpOut) {
                    bool needAtomic = ((!IS_DKV_RESIDENT_L0C) && runInfo.isS2IdxNoChange &&
                                       !isDkvL0CResidentForD192Dv128 && !isDvL0CResidentForD192ToD256) ||
                                      (!runInfo.isFirstBlock && (runInfo.specialS2Index != -1));
                    if (needAtomic) {
                        SetAtomicAdd<CALC_TYPE>();
                    }
                    int64_t offset = (runInfo.specialS2Index != -1) ?
                                         (runInfo.specialS2Index * CUBE_BASEN * HEAD_DIM_ALIGN * NUM_TWO +
                                          CUBE_BASEN * HEAD_DIM_ALIGN) :
                                         GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN;
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                             dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                    if (needAtomic) {
                        SetAtomicNone();
                    }
                }
            }
        } else {
            if (isFixpOut) {
                fixpipeParams.nSize = (realN + 7) >> 3 << 3;
                Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[gmNOffset], dvL0CBuffer.GetTensor<CALC_TYPE>(),
                                                         fixpipeParams);
            }
        }
    } else {
        FixpipeParamsC310<CO2Layout::NZ> fixpipeParams;
        fixpipeParams.nSize = constInfo.dvAlign16;
        fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
        fixpipeParams.srcStride = runInfo.commonRunInfo.s2AlignedSize;
        fixpipeParams.dstStride = fixpipeParams.mSize * C0_SIZE;
        fixpipeParams.unitFlag = ENABLE_UNITFLAG ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::NZ, false};
        SetAtomicAdd<CALC_TYPE>();
        if (isFixpOut) {
            int64_t offset = 0;
            if constexpr (IS_TND) {
                if (unlikely(runInfo.commonRunInfo.boIdx == 0)) {
                    offset = runInfo.commonRunInfo.n2oIdx * runInfo.commonRunInfo.actualS2Size * constInfo.dvAlign16 +
                             runInfo.s2oIdx * CUBE_BASEN * constInfo.dvAlign16;
                } else {
                    offset = ((__gm__ int64_t *)constInfo.seqS2_addr)[runInfo.commonRunInfo.boIdx - 1] *
                                 constInfo.n2Size * constInfo.dvAlign16 +
                             runInfo.commonRunInfo.n2oIdx * runInfo.commonRunInfo.actualS2Size * constInfo.dvAlign16 +
                             runInfo.s2oIdx * CUBE_BASEN * constInfo.dvAlign16;
                }
            } else {
                offset = runInfo.commonRunInfo.boIdx * constInfo.n2Size * constInfo.commonConstInfo.s2Size *
                             constInfo.dvAlign16 +
                         runInfo.commonRunInfo.n2oIdx * constInfo.commonConstInfo.s2Size * constInfo.dvAlign16 +
                         runInfo.s2oIdx * CUBE_BASEN * constInfo.dvAlign16;
            }
            Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset], dvL0CBuffer.GetTensor<CALC_TYPE>(),
                                                     fixpipeParams);
        }
        SetAtomicNone();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDy(
    typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor, MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    uint32_t dimSize = static_cast<uint32_t>(constInfo.commonConstInfo.dSizeV);
    bool isSplitN = IsDvSplitNEnabled(constInfo.commonConstInfo.s1Size, constInfo.commonConstInfo.s2Size);
    uint32_t baseD = isSplitN ? SPLIT_BASE_D : dimSize;
    uint32_t nLoops = isSplitN ? (dimSize + SPLIT_BASE_D - 1) / SPLIT_BASE_D : 1;
    MutexBuffer<BufferType::L1> dYL1Buffer;
    LocalTensor<OUTDTYPE> dYL1Tensor;
    for (uint32_t n = 0; n < nLoops; ++n) {
        uint32_t realN = dimSize;
        if (isSplitN) {
            realN = SPLIT_BASE_D;
            if (n == nLoops - 1) {
                uint32_t tailSize = dimSize % SPLIT_BASE_D;
                realN = tailSize ? tailSize : SPLIT_BASE_D;
            }
        }
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            if (n == 0) {
                dYL1Buffer = dYL1Buf.GetReused();
            }
            dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>()[n * runInfo.s1AlignedSize * baseD];
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                dYL1Buffer = fp32L1Buf2.Get();
            } else {
                dYL1Buffer = commonL1Buf.Get();
            }
            dYL1Buffer.LockProd();
            dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(dYL1Tensor, this->dyGm[runInfo.dyOffset + gmNOffset], nd2NzParams);
            dYL1Buffer.UnlockProd();
        }
        if constexpr (NeedLockL1EachNLoop()) {
            dYL1Buffer.LockCons();
        } else {
            if (n == 0) {
                dYL1Buffer.LockCons();
            }
        }

        MutexBuffer<BufferType::L0C> dvL0CBuffer = GetDvL0CBuffer();
        // load l1 to l0ab + mmad
        if constexpr (!ENABLE_UNITFLAG) {
            dvL0CBuffer.LockProd(); // 反向同步
        }
        bool enPartialSum = false; // control l0c accumulate sum
        bool isFixpOut = true;
        if constexpr (!IS_DETER_OLD(DETER_SPARSE_TYPE)) {
            if ((IS_DKV_RESIDENT_L0C || isDkvL0CResidentForD192Dv128 || isDvL0CResidentForD192ToD256) &&
                !(SPLIT_AXIS == BN2)) {
                enPartialSum = runInfo.isS2IdxNoChange;
                isFixpOut = !runInfo.isNextS2IdxNoChange;
            }
            if constexpr (IS_BN2_MULTIBLK && IS_DKV_RESIDENT_L0C) {
                enPartialSum = runInfo.isS2IdxNoChange;
                isFixpOut = !runInfo.isNextS2IdxNoChange;
            }
        }

        MMParam param = {(uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         !enPartialSum,
                         !ENABLE_UNITFLAG ? UNITFLAG_DISABLE : (isFixpOut ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE)};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                          dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                       dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                       dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        if constexpr (NeedLockL1EachNLoop()) {
            dYL1Buffer.UnlockCons();
        } else {
            if (n == nLoops - 1) {
                dYL1Buffer.UnlockCons();
            }
        }

        if constexpr (!ENABLE_UNITFLAG) {
            dvL0CBuffer.UnlockProd();
            dvL0CBuffer.LockCons();
        }
        IterateMmPDyFixpout<T, IS_WRITE_UB>(outTensor, dvL0CBuffer, constInfo, runInfo, realN, gmNOffset, isFixpOut);
        if constexpr (!ENABLE_UNITFLAG) {
            dvL0CBuffer.UnlockCons();
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDyOlderDeter(
    GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
    FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    MutexBuffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = pL1Buf.Get();
    constexpr uint32_t baseD = SPLIT_BASE_D;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSizeV + baseD - 1) / baseD; // 尾块处理
    uint32_t realN = baseD;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSizeV % baseD;
            realN = tailSize ? tailSize : baseD;
        }
        MutexBuffer<BufferType::L1> dYL1Buffer;
        LocalTensor<OUTDTYPE> dYL1Tensor;
        uint64_t gmNOffset = n * baseD;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            dYL1Buffer = dYL1Buf.GetReused();
            dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                dYL1Buffer = fp32L1Buf2.Get();
            } else {
                dYL1Buffer = commonL1Buf.Get();
            }
            dYL1Buffer.LockProd();
            dYL1Tensor = dYL1Buffer.template GetTensor<OUTDTYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
            nd2NzParams.dstNzC0Stride = runInfo.s1AlignedSize;
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(dYL1Tensor, this->dyGm[runInfo.dyOffset + gmNOffset], nd2NzParams);
            dYL1Buffer.UnlockProd();
        }
        dYL1Buffer.LockCons();

        MutexBuffer<BufferType::L0C> dvL0CBuffer = GetDvL0CBuffer();
        // load l1 to l0ab + mmad
        dvL0CBuffer.LockProd();                                      // 反向同步
        MMParam param = {(uint32_t)CUBE_BASEN,                       // singleM
                         realN,                                      // singleN
                         (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
                         true,                                       // isLeftTranspose
                         false,                                      // isRightTranspose
                         true,
                         true};
        if constexpr (HEAD_DIM_ALIGN <= CUBE_BASEN && !IS_FP32_INPUT) {
            MatmulFullMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                            ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                          dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else if constexpr (IS_FP32_INPUT) {
            MatmulKMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                       dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulNMutex<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK,
                         ABLayout::KN>(pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf,
                                       dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        }

        dYL1Buffer.UnlockCons();

        dvL0CBuffer.UnlockProd();
        dvL0CBuffer.LockCons();

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = CUBE_BASEN;
        fixpipeParams.nSize = realN;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = HEAD_DIM_ALIGN;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[gmNOffset], dvL0CBuffer.GetTensor<CALC_TYPE>(),
                                                 fixpipeParams);
        dvL0CBuffer.UnlockCons();
    }
}

TEMPLATES_DEF
class FAGBlockCubeDummy {
public:
    __aicore__ inline FAGBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              MutexBufferManager<BufferType::L1> *l1BuffMgr) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace) {};
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo) {};
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs) {};
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs) {};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // mm3 older dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo,
                                        bool isDqNeedDeter = false) {}; // dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo, bool isDqNeedDeter = false) {}; // mm4 older dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        MutexBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // dv
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyOlderDeter(
        GlobalTensor<CALC_TYPE> outTensor, MutexBuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
        FagConstInfo &constInfo, FagRunInfo &runInfo) {}; // mm5 older dv
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)

} // namespace FagBaseApi
#endif
