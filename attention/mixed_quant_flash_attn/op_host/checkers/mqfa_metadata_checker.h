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
 * \file mqfa_metadata_checker.h
 * \brief Checker for metadata parameter (ÎÄµµ: INT32, shape=(max_schedule_size,))
 */

#ifndef MQFA_METADATA_CHECKER_H
#define MQFA_METADATA_CHECKER_H

#include "./mqfa_base_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {
class MetadataChecker : public FABaseChecker {
public:
    MetadataChecker() = default;
    ~MetadataChecker() = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo) override;
};
} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MQFA_METADATA_CHECKER_H
