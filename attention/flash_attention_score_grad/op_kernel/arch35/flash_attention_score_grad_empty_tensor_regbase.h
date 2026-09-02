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
 * \file flash_attention_score_grad_empty_tensor_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_EMPTY_TENSOR_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_EMPTY_TENSOR_REGBASE_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

using AscendC::InitOutput;

template <typename T>
class FlashAttentionScoreGradEmptyTensorRegbase {
public:
    __aicore__ inline FlashAttentionScoreGradEmptyTensorRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse,
                                __gm__ uint8_t *dqRope, __gm__ uint8_t *dkRope,
                                const FlashAttentionScoreGradEmptyTensorTilingDataRegbase *__restrict tilingData);
    __aicore__ inline void Process();

protected:
    uint64_t m_blockIdx;
    AscendC::GlobalTensor<T> m_dqGm, m_dkGm, m_dvGm, m_dpseGm, m_dqRopeGm, m_dkRopeGm;
    const FlashAttentionScoreGradEmptyTensorTilingDataRegbase *__restrict m_tilingData;
};

template <typename T>
__aicore__ inline void FlashAttentionScoreGradEmptyTensorRegbase<T>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dpse, __gm__ uint8_t *dqRope,
    __gm__ uint8_t *dkRope, const FlashAttentionScoreGradEmptyTensorTilingDataRegbase *__restrict tilingData)
{
    m_blockIdx = AscendC::GetBlockIdx();
    m_tilingData = tilingData;
    m_dqGm.SetGlobalBuffer((__gm__ T *)dq);
    m_dkGm.SetGlobalBuffer((__gm__ T *)dk);
    m_dvGm.SetGlobalBuffer((__gm__ T *)dv);
    m_dpseGm.SetGlobalBuffer((__gm__ T *)dpse);
    if (m_tilingData->isRope) {
        m_dqRopeGm.SetGlobalBuffer((__gm__ T *)dqRope);
        m_dkRopeGm.SetGlobalBuffer((__gm__ T *)dkRope);
    }
}

template <typename T>
__aicore__ inline void FlashAttentionScoreGradEmptyTensorRegbase<T>::Process()
{
    if (m_tilingData->singleCoreDqNum > 0) {
        if (m_blockIdx < m_tilingData->formerDqNum) {
            InitOutput<T>(m_dqGm[m_blockIdx * m_tilingData->singleCoreDqNum], m_tilingData->singleCoreDqNum, 0);
        } else {
            InitOutput<T>(m_dqGm[m_tilingData->formerDqNum * m_tilingData->singleCoreDqNum +
                                 (m_blockIdx - m_tilingData->formerDqNum) * m_tilingData->tailCoreDqNum],
                          m_tilingData->tailCoreDqNum, 0);
        }
    }
    if (m_tilingData->singleCoreDkNum > 0) {
        if (m_blockIdx < m_tilingData->formerDkNum) {
            InitOutput<T>(m_dkGm[m_blockIdx * m_tilingData->singleCoreDkNum], m_tilingData->singleCoreDkNum, 0);
        } else {
            InitOutput<T>(m_dkGm[m_tilingData->formerDkNum * m_tilingData->singleCoreDkNum +
                                 (m_blockIdx - m_tilingData->formerDkNum) * m_tilingData->tailCoreDkNum],
                          m_tilingData->tailCoreDkNum, 0);
        }
    }
    if (m_tilingData->singleCoreDvNum > 0) {
        if (m_blockIdx < m_tilingData->formerDvNum) {
            InitOutput<T>(m_dvGm[m_blockIdx * m_tilingData->singleCoreDvNum], m_tilingData->singleCoreDvNum, 0);
        } else {
            InitOutput<T>(m_dvGm[m_tilingData->formerDvNum * m_tilingData->singleCoreDvNum +
                                 (m_blockIdx - m_tilingData->formerDvNum) * m_tilingData->tailCoreDvNum],
                          m_tilingData->tailCoreDvNum, 0);
        }
    }
    if (m_tilingData->singleCoreDpseNum > 0) {
        if (m_blockIdx < m_tilingData->formerDpseNum) {
            InitOutput<T>(m_dpseGm[m_blockIdx * m_tilingData->singleCoreDpseNum], m_tilingData->singleCoreDpseNum, 0);
        } else {
            InitOutput<T>(m_dpseGm[m_tilingData->formerDpseNum * m_tilingData->singleCoreDpseNum +
                                   (m_blockIdx - m_tilingData->formerDpseNum) * m_tilingData->tailCoreDpseNum],
                          m_tilingData->tailCoreDpseNum, 0);
        }
    }
    if (m_tilingData->isRope && m_tilingData->singleCoreDqRopeNum > 0) {
        if (m_blockIdx < m_tilingData->formerDqRopeNum) {
            InitOutput<T>(m_dqRopeGm[m_blockIdx * m_tilingData->singleCoreDqRopeNum], m_tilingData->singleCoreDqRopeNum,
                          0);
        } else {
            InitOutput<T>(m_dqRopeGm[m_tilingData->formerDqRopeNum * m_tilingData->singleCoreDqRopeNum +
                                     (m_blockIdx - m_tilingData->formerDqRopeNum) * m_tilingData->tailCoreDqRopeNum],
                          m_tilingData->tailCoreDqRopeNum, 0);
        }
    }
    if (m_tilingData->isRope && m_tilingData->singleCoreDkRopeNum > 0) {
        if (m_blockIdx < m_tilingData->formerDkRopeNum) {
            InitOutput<T>(m_dkRopeGm[m_blockIdx * m_tilingData->singleCoreDkRopeNum], m_tilingData->singleCoreDkRopeNum,
                          0);
        } else {
            InitOutput<T>(m_dkRopeGm[m_tilingData->formerDkRopeNum * m_tilingData->singleCoreDkRopeNum +
                                     (m_blockIdx - m_tilingData->formerDkRopeNum) * m_tilingData->tailCoreDkRopeNum],
                          m_tilingData->tailCoreDkRopeNum, 0);
        }
    }
}

#endif // FLASH_ATTENTION_SCORE_GRAD_EMPTY_H_
