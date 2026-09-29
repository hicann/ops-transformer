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
 * \file qfa_checker_mxfp8_softmax_fp16.h
 * \brief QfaMxfp8SoftmaxFp16Checker 声明
 */

#ifndef QFA_CHECKER_MXFP8_SOFTMAX_FP16_H
#define QFA_CHECKER_MXFP8_SOFTMAX_FP16_H

#include <register/op_impl_registry.h>

namespace optiling {
namespace quant_flash_attn {

class QfaTilingInfo;

class QfaMxfp8SoftmaxFp16Checker {
public:
    QfaMxfp8SoftmaxFp16Checker() = default;
    ~QfaMxfp8SoftmaxFp16Checker() = default;

    ge::graphStatus Init(const QfaTilingInfo& qfaInfo);
    ge::graphStatus Process(const QfaTilingInfo& qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling

#endif // QFA_CHECKER_MXFP8_SOFTMAX_FP16_H
