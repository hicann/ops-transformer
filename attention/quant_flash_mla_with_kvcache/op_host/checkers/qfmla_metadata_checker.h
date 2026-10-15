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
 * \file qfmla_metadata_checker.h
 * \brief Checker for metadata parameter ( 公共参数组 - metadata)
 */

#ifndef QFMLA_METADATA_CHECKER_H
#define QFMLA_METADATA_CHECKER_H

#include "qfmla_base_checker.h"

namespace optiling {
namespace quant_flash_mla_with_kvcache {

class MetadataChecker : public QmlaBaseChecker {
public:
    MetadataChecker() = default;
    ~MetadataChecker() = default;

    ge::graphStatus CheckSinglePara(const QmlaTilingInfo& qmlaInfo) override;
    ge::graphStatus CheckParaExistence(const QmlaTilingInfo& qmlaInfo) override;
};

} // namespace quant_flash_mla_with_kvcache
} // namespace optiling
#endif // QFMLA_METADATA_CHECKER_H
