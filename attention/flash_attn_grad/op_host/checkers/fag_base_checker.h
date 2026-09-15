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
 * \file fag_base_checker.h
 * \brief 校验层基类与共享上下文。
 *
 * 形态参照正向 attention/flash_attn/op_host/checkers/base_checker_flash_attn.h 的
 * 组合模式。
 *
 * D3：校验代码本算子自持，不与正向共用、不抽到 attention/common。
 * 口径漂移靠文件头对照注释兜住，不靠共享实现。
 *
 * 与正向的一处结构差异：正向的 checker 吃已解析好的 FaTilingInfo，而 FAGrad 的
 * 校验发生在 parser 之前（入口顺序是 CheckParams -> ParseInfo），所以这里的
 * checker 直接读 gert::TilingContext。属性只在 FagAttrRangeChecker 里读一次，
 * 存进 FagCheckCtx 供后续 checker 复用，避免每个 checker 各读一遍。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_BASE_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_BASE_CHECKER_H_

#include <string>

#include "exe_graph/runtime/tiling_context.h"

#include "../info/flash_attn_grad_tiling_info.h"

namespace optiling {

// checker 之间传递的共享状态。属性字段由 FagAttrRangeChecker 填写，在它之后
// 注册的 checker 才可以读；默认值与 flash_attn_grad_def.cpp 的属性默认值一致。
struct FagCheckCtx {
    gert::TilingContext *context = nullptr;
    const char *opName = nullptr;

    int64_t maskMode = MASK_MODE_NO_MASK;
    int64_t winLeft = -1;
    int64_t winRight = -1;
    int64_t maxSeqlenQ = -1;
    int64_t maxSeqlenKv = -1;
    std::string layoutQ = "BSND";
    std::string layoutKv = "BSND";
    std::string layoutOut = "BSND";
};

class FagBaseChecker {
public:
    virtual ~FagBaseChecker() = default;

    // 校验失败必须返回非 SUCCESS 并自己打过 OP_LOGE；组合器只负责补一条
    // "哪个 checker 失败了"，不重复打原因。
    virtual ge::graphStatus Check(FagCheckCtx &ctx) = 0;

    // 用于失败时定位是哪一环，不参与判定。
    virtual const char *Name() const = 0;
};

// 可选输入是否真的传进来了：shape 非空且维度数大于 0。图模式下未接的可选输入
// 可能拿到非空但 0 维的 shape，所以两个条件都要看。
bool IsOptionalTensorExist(gert::TilingContext *context, size_t index);

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_BASE_CHECKER_H_
