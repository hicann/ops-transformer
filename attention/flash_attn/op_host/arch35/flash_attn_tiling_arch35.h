/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_TILING_IMPL_H_ARCH35
#define FLASH_ATTN_TILING_IMPL_H_ARCH35

#include "../flash_attn_tiling_base.h"

namespace optiling {
namespace flash_attn {

class FlashAttnTilingImplArch35 : public FlashAttnTilingBase {
public:
    explicit FlashAttnTilingImplArch35(gert::TilingContext* context)
        : FlashAttnTilingBase(context)
    {}
    ~FlashAttnTilingImplArch35() override = default;

protected:
    void SplitPolicy() override;
    void UpdateTilingKeyConfig() override;
    void UpdateTilingKeyInfo() override;
    void UpdateTilingKeyTemplateId() override;
};

} // namespace flash_attn
} // namespace optiling
#endif // FLASH_ATTN_TILING_IMPL_H_ARCH35
