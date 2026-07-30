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
 * \file stem_oam_prep_paged_kv_base.h
 * \brief
 */

#ifndef STEM_OAM_PREP_PAGED_KV_BASE_H
#define STEM_OAM_PREP_PAGED_KV_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

using namespace AscendC;

constexpr float FLT_MIN = 1.1754943508222875e-38F;

constexpr static AscendC::MicroAPI::CastTrait castTraitFp8ToFloat = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTraitFloatTo16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

template <typename PARAM_T>
__aicore__ inline void CopyIn(const LocalTensor<PARAM_T> &dstTensor, const GlobalTensor<PARAM_T> &srcTensor,
                              int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<PARAM_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                               static_cast<PARAM_T>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
}

template <typename PARAM_T>
__aicore__ inline void CopyOut(const GlobalTensor<PARAM_T> &dstTensor, const LocalTensor<PARAM_T> &srcTensor,
                               int64_t repTime, int64_t dataLen, uint32_t ubStride, uint32_t gmStride)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(repTime), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                    static_cast<uint32_t>(ubStride), static_cast<uint32_t>(gmStride),
                                    static_cast<uint32_t>(0)};
    DataCopyPad(dstTensor, srcTensor, copyParams);
}

template <AscendC::HardEvent VAR_T>
__aicore__ inline void EventMsg()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(VAR_T));
    SetFlag<VAR_T>(event);
    WaitFlag<VAR_T>(event);
}

#endif //  STEM_OAM_PREP_PAGED_KV_BASE_H
