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
 * \file ts_nsa_selected_attention.h
 * \brief NsaSelectedAttention UTest 基类定义.
 */

#include "tests/utest/ts.h"
#include "nsa_selected_attention_case.h"

using NsaSelectedAttentionCase = ops::adv::tests::nsa_selected_attention_ns::NsaSelectedAttentionCase;
class Ts_NsaSelectedAttention : public Ts<NsaSelectedAttentionCase> {};