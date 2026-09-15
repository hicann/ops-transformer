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
 * \file fag_checker.h
 * \brief 组合器：按注册顺序依次跑各 checker，任一失败即返回。
 *
 * 形态参照正向 attention/flash_attn/op_host/checkers/fa_checker.h。
 *
 * 与正向的一处取舍：正向分 CheckParaExistence / CheckSinglePara / CheckMultiPara /
 * CheckFeature 四轮，是因为它有八个 checker、多数要参与多轮。FAGrad 只有六个
 * checker，每个都只落在一轮里，做成四轮会让每个 checker 实现一个方法、另外三个
 * 返回 SUCCESS —— 正是 §8.2 列为反面写法的空虚函数。所以这里退成"单个 Check +
 * 有序列表"，顺序由 RegisterCheckers 显式表达，等 checker 数量涨上来再分轮。
 */

#ifndef FLASH_ATTN_GRAD_CHECKERS_FAG_CHECKER_H_
#define FLASH_ATTN_GRAD_CHECKERS_FAG_CHECKER_H_

#include <memory>
#include <vector>

#include "fag_base_checker.h"

namespace optiling {

class FagChecker {
public:
    FagChecker();
    ~FagChecker() = default;

    ge::graphStatus Process(gert::TilingContext *context);

private:
    std::vector<std::unique_ptr<FagBaseChecker>> checkers_;
};

} // namespace optiling

#endif // FLASH_ATTN_GRAD_CHECKERS_FAG_CHECKER_H_
