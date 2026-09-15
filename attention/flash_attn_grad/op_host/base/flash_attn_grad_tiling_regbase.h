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
 * \file flash_attn_grad_tiling_regbase.h
 * \brief 目前唯一的 tiling 模板类：regbase(DAV_3510) 上的全部 layout。
 *
 * 命名取 regbase 而不是 Padded/Varlen，是因为现在 BSND/BNSD/TND
 * 都由这一个类接（TND 只影响 tilingkey bit2 与分核口径，不换 C++ 类）。等 TND
 * 真要走独立实现时再拆成两个类，那时 IsCapable 分别按 layout 收窄，DoTiling 的
 * GRAPH_PARAM_INVALID 返回值就是为那一步准备的。
 *
 * 能力边界只有一条：SOC 必须是 regbase。其余"接不接"的判断在两个地方 ——
 * 参数合法性在 op_host/checkers/，模板/swizzle/D 分档这类性能路由在
 * plan/flash_attn_grad_tiling_route.cpp 的候选表里，都不在这里。
 */

#ifndef FLASH_ATTN_GRAD_TILING_REGBASE_H_
#define FLASH_ATTN_GRAD_TILING_REGBASE_H_

#include "flash_attn_grad_tiling_base.h"

namespace optiling {

class FlashAttnGradTilingRegbase : public FagTilingBase {
public:
    explicit FlashAttnGradTilingRegbase(gert::TilingContext *context)
        : FagTilingBase(context)
    {}

protected:
    void InitTilingInfo(const FagParsedInfo &info) override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;

private:
    // 写 pypto codegen 出来的 TilingData。基类不封装这一步（见 flash_attn_grad_tiling_base.h
    // 的说明）：那个结构体只在带 force-include 的 *_tiling*.cpp 里可见。
    ge::graphStatus WriteTilingData(const FagParsedInfo &info);

    // 指向入口栈上的解析结果。本类不拥有它，也不回写（策略层对 info 只读）。
    const FagParsedInfo *info_ = nullptr;
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_TILING_REGBASE_H_
