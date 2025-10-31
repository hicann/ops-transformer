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
 * \file aclnn_fag_case.h
 * \brief
 */

#include "aclnn_fa_case.h"

namespace ops::adv::tests::fag {

class AclnnFagCase : public ops::adv::tests::fa::AclnnFaCase {
public:
    AclnnFagCase();
    AclnnFagCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion reverse, AclnnFaParam param);

    bool Run() override;
};

} // namespace ops::adv::tests::fag
