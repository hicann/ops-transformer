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
 * \file qmla_checker.h
 * \brief QuantFlashMlaWithKvcache 参数校验
 */

#ifndef QMLA_CHECKER_H
#define QMLA_CHECKER_H

#include <vector>
#include "tiling/tiling_api.h"
#include "../qmla_tiling_info.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class QmlaChecker {
public:
    QmlaChecker() = default;
    ~QmlaChecker() = default;

    ge::graphStatus Init(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus Process(const QmlaTilingInfo& qmlaInfo);

private:
    ge::graphStatus CheckAxisPara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSeqLenPara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckMaskPara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckQuantPara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckLayoutPara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckSoftmaxLsePara(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus CheckNonContiguousSupport(const QmlaTilingInfo& qmlaInfo);
    // 从最后一维向前推算连续期望stride, 返回第一个不连续维的下标
    ge::graphStatus CheckTensorContiguous(const gert::Shape& inputShape, const gert::Stride* strides,
                                          int32_t& index) const;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling

#endif // QMLA_CHECKER_H
