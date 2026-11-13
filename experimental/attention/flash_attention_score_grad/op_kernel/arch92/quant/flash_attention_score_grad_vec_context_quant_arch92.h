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
 * \file flash_attention_score_grad_vec_context_quant_arch92.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_VEC_CONTEXT_QUANT_ARCH92_H_
#define FLASH_ATTENTION_SCORE_GRAD_VEC_CONTEXT_QUANT_ARCH92_H_

#include "flash_attention_score_grad_common_quant_arch92.h"
namespace FagBaseApi {
namespace quant {

TEMPLATES_DEF
class FAGBlockVecQuantContext {
public:
    __aicore__ inline FAGBlockVecQuantContext(){};
    __aicore__ inline void SetVecBlockParams(TPipe* pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo& constInfo);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax,
                                            GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV,
                                            GM_ADDR deqScaleDy, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                            int64_t sfmgWorkSpaceOffset);
    __aicore__ inline void InitUbBuffer(const LocalTensor<CALC_TYPE>& commonTensors);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    constexpr static bool IS_FP8_INPUT = IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ||
                                         IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ||
                                         IsSameType<INPUT_TYPE, hifloat8_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static float FP8_MAX =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value ? 57344 : (IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value ? 448 : 32768);
    constexpr static uint32_t DETER_OFFSET_UB_SIZE = 1024 * 3;
    constexpr static uint32_t CUBE_BASEM = static_cast<uint32_t>(s1TemplateType);
    constexpr static uint32_t CUBE_BASEN = static_cast<uint32_t>(s2TemplateType);
    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t VECTOR_BASEM = CUBE_BASEM / CV_CORE_RATIO;
    constexpr static uint32_t VECTOR_BASEN = CUBE_BASEN;
    constexpr static uint32_t INNER_VECTOR_BASEM = 128;
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_INPUT_DTYPE = 32 / sizeof(INPUT_TYPE);
    constexpr static uint32_t INPUT_BLOCK_NUM_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t FRACTAL_NZ_C0_SIZE_FOR_OUT_DTYPE = 32 / sizeof(OUTDTYPE);
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP16 = 32 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D256 = 16 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE_FP32_D512 = 64 * 1024;
    constexpr static uint32_t DETER_DQ_UB_SIZE =
        IS_FP32_INPUT ? (HEAD_DIM_ALIGN > 256 ? DETER_DQ_UB_SIZE_FP32_D512 : DETER_DQ_UB_SIZE_FP32_D256) :
                        DETER_DQ_UB_SIZE_FP16;

    constexpr static uint8_t PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[4] = {0, 1, 2, 3};
    constexpr static uint8_t PDS_PDS_COPY_TO_L1_INNER_CORE_SYNC_EVENTS[2] = {0, 1};
    constexpr static uint8_t DQKV_UB_TO_GM_INNER_CORE_SYNC_EVENTS[3] = {4, 5, 6};

    // vector gm addr
    GlobalTensor<INPUT_TYPE> valueGm, dyGm;
    GlobalTensor<OUTDTYPE> yGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm;
    GlobalTensor<float> deqScaleQGm, deqScaleKGm, deqScaleVGm, deqScaleDyGm;
    GlobalTensor<float> sfmgWorkspaceGm;

    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm;

    // ub buffer
    TBuf<> vecQue;
    LocalTensor<float> maxTensor[4];  // 64 * 4 * 4
    LocalTensor<float> sumTensor[4];  // 64 * 4 * 4
    LocalTensor<float> dTensor[4];    // 64 * 4 * 4
    LocalTensor<uint16_t> permTensor; // 128 * 2

    TPipe* pipe;

    uint32_t vBlockIdx;
    uint32_t cBlockIdx;
    uint32_t vSubBlockIdx;

    uint32_t maxsumIdx = 0;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuantContext<TEMPLATE_ARGS>::SetVecBlockParams(TPipe* pipe, FagTilingType tilingData,
                                                                                 uint32_t vBlockIdx, uint32_t cBlockIdx,
                                                                                 uint32_t vSubBlockIdx,
                                                                                 FagConstInfo& constInfo)
{
    this->pipe = pipe;
    this->vBlockIdx = vBlockIdx;
    this->cBlockIdx = cBlockIdx;
    this->vSubBlockIdx = vSubBlockIdx;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuantContext<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK,
    GM_ADDR deqScaleV, GM_ADDR deqScaleDy, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
    int64_t sfmgWorkSpaceOffset)
{
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE*)value);
    dyGm.SetGlobalBuffer((__gm__ INPUT_TYPE*)dy);
    yGm.SetGlobalBuffer((__gm__ OUTDTYPE*)y);

    softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmaxMax);
    softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmaxSum);
    deqScaleQGm.SetGlobalBuffer((__gm__ float*)deqScaleQ);
    deqScaleKGm.SetGlobalBuffer((__gm__ float*)deqScaleK);
    deqScaleVGm.SetGlobalBuffer((__gm__ float*)deqScaleV);
    deqScaleDyGm.SetGlobalBuffer((__gm__ float*)deqScaleDy);

    dqGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dq);
    dkGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dk);
    dvGm.SetGlobalBuffer((__gm__ OUTDTYPE*)dv);

    sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float*)workspace + sfmgWorkSpaceOffset / sizeof(float));
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuantContext<TEMPLATE_ARGS>::InitUbBuffer(const LocalTensor<CALC_TYPE>& commonTensors)
{
    uint32_t ubOffset = QUANT_S1_BASE_COUNT;
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        maxTensor[i] = commonTensors[ubOffset];
        sumTensor[i] = commonTensors[ubOffset + QUANT_S1_BASE_COUNT_DB];
        dTensor[i] = commonTensors[ubOffset + QUANT_S1_BASE_COUNT_DB * NUM_TWO];
        ubOffset += QUANT_S1_BASE_COUNT_DB * NUM_THREE;
    }

    // 256B
    permTensor = commonTensors[ubOffset].template ReinterpretCast<uint16_t>();
    for (uint16_t i = 0; i < BLOCK_SIZE; i++) {
        permTensor.SetValue(i * NUM_FOUR + 0, 0 * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + 1, 1 * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + NUM_TWO, NUM_TWO * BLOCK_SIZE + i);
        permTensor.SetValue(i * NUM_FOUR + NUM_THREE, NUM_THREE * BLOCK_SIZE + i);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuantContext<TEMPLATE_ARGS>::AllocEventID()
{
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        SetFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[i]);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockVecQuantContext<TEMPLATE_ARGS>::FreeEventID()
{
    for (int64_t i = 0; i < NUM_FOUR; i++) {
        WaitFlag<AscendC::HardEvent::V_MTE2>(PDS_COPY_IN_MAXSUMD_INNER_CORE_SYNC_EVENTS[i]);
    }
}

TEMPLATES_DEF
class FAGBlockVecQuantContextDummy {
public:
    __aicore__ inline void InitUbBuffer(const LocalTensor<CALC_TYPE>& commonTensors) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR value, GM_ADDR dy, GM_ADDR y, GM_ADDR softmaxMax,
                                            GM_ADDR softmaxSum, GM_ADDR deqScaleQ, GM_ADDR deqScaleK, GM_ADDR deqScaleV,
                                            GM_ADDR deqScaleDy, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                            int64_t sfmgWorkSpaceOffset) {};
    __aicore__ inline void SetVecBlockParams(TPipe* pipe, FagTilingType tilingData, uint32_t vBlockIdx,
                                             uint32_t cBlockIdx, uint32_t vSubBlockIdx, FagConstInfo& constInfo) {};

    __aicore__ inline void AllocEventID() {};
    __aicore__ inline void FreeEventID() {};
};

} // namespace quant
} // namespace FagBaseApi

#endif
