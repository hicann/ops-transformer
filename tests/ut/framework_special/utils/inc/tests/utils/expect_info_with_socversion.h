/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file expect_info.h
 * \brief 测试用例执行结果期望信息.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include "platform.h"

namespace ops::adv::tests::utils {

class ExpectInfoWithSocversion {
public:
    static constexpr uint64_t kInvalidTilingKey = std::numeric_limits<uint64_t>::max();
    static constexpr int64_t kInvalidTilingBlockDim = 0;
    static constexpr int64_t kFullTilingBlockDim = -1;

public:
    /**
     * 期望该用例成功
     */
    bool mSuccess = true;

    /**
     * 期望 TilingKeys 取值，区分不同的soc
     */
    uint64_t mTilingKeys[(uint64_t)Platform::SocVersion::SocVersionBottom] = {0};

    /**
     * 期望 TilingBlockDim 取值
     */
    int64_t mTilingBlockDims[(uint64_t)Platform::SocVersion::SocVersionBottom] = {0};

public:
    ExpectInfoWithSocversion() noexcept;
    ExpectInfoWithSocversion(bool success, uint64_t tilingKey, int64_t tilingBlockDim);
    ExpectInfoWithSocversion(bool success, const std::map<int32_t, uint64_t> &expTilingKeys,
            int64_t tilingBlockDim);
    ExpectInfoWithSocversion(bool success, const std::map<int32_t, uint64_t> &expTilingKeys,
            const std::map<int32_t, int64_t> &expTilingBlockDims);
};

} // namespace ops::adv::tests::utils
