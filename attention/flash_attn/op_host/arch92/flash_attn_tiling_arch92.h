/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_TILING_IMPL_H_ARCH92
#define FLASH_ATTN_TILING_IMPL_H_ARCH92

#include "../flash_attn_tiling_base.h"

namespace optiling {
namespace flash_attn {

class FlashAttnTilingImplArch92 : public FlashAttnTilingBase {
public:
    explicit FlashAttnTilingImplArch92(gert::TilingContext* context)
        : FlashAttnTilingBase(context)
    {}
    ~FlashAttnTilingImplArch92() override = default;

protected:
    void SplitPolicy() override;
    void UpdateTilingKeyConfig() override;
    void UpdateTilingKeyInfo() override;

    uint32_t templateId_ = FA_Template_ND;
};

} // namespace flash_attn
} // namespace optiling
#endif // FLASH_ATTN_TILING_IMPL_H_ARCH92
