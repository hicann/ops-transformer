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
 * \file qfmla_checker.h
 * \brief QmlaChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef QFMLA_CHECKER_H
#define QFMLA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "qfmla_base_checker.h"
#include "../qmla_tiling_info.h"

#include "qfmla_common_checker.h"
#include "qfmla_metadata_checker.h"
#include "qfmla_quant_checker.h"
#include "qfmla_mask_checker.h"
#include "qfmla_seq_len_checker.h"
#include "qfmla_paged_attention_checker.h"
#include "qfmla_softmax_lse_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class QmlaChecker {
public:
    QmlaChecker() = default;
    ~QmlaChecker() = default;

    ge::graphStatus Init(const QmlaTilingInfo& qmlaInfo);
    ge::graphStatus Process(const QmlaTilingInfo& qmlaInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(QmlaBaseChecker*, const QmlaTilingInfo&)>;

    ge::graphStatus RunCheck(const CheckMethod& method, const QmlaTilingInfo& qmlaInfo);
    void RegisterCheckers();

    std::vector<std::unique_ptr<QmlaBaseChecker>> checkers_;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
#endif // QFMLA_CHECKER_H
