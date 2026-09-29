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
 * \file qfa_checker_mxfp8_softmax_fp16.cpp
 * \brief QfaMxfp8SoftmaxFp16Checker 实现
 */

#include "../qfa_tiling_info.h"
#include "qfa_checker_mxfp8_softmax_fp16.h"

namespace optiling {
namespace quant_flash_attn {

ge::graphStatus QfaMxfp8SoftmaxFp16Checker::Init(const QfaTilingInfo& qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaMxfp8SoftmaxFp16Checker::Process(const QfaTilingInfo& qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
