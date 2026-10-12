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
 * \file apply_rotary_pos_emb_base.h
 * \brief
 */
#ifndef APPLY_ROTARY_POS_EMB_BASE_H
#define APPLY_ROTARY_POS_EMB_BASE_H

#include "kernel_operator.h"

namespace ApplyRotaryPosEmb {
using namespace AscendC;

template <typename T>
class ApplyRotaryPosEmbBase {
public:
    __aicore__ inline ApplyRotaryPosEmbBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline void LocalTensor2NewTensor(LocalTensor<T1>& tensor_new, const LocalTensor<T2>& tensor_old)
    {
        tensor_new = tensor_old.template ReinterpretCast<T1>();
    };
};
} // namespace ApplyRotaryPosEmb

#endif // APPLY_ROTARY_POS_EMB_BASE_H
