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
 * \file ts_ffn.h
 * \brief FFN UTest 相关基类定义.
 */

#ifndef UTEST_TS_FFN_H
#define UTEST_TS_FFN_H

#include "tests/utest/ts.h"
#include "ffn_case.h"
#include "aclnn_ffn_case.h"

using ops::adv::tests::ffn::AclnnFFNCase;
using ops::adv::tests::ffn::GenTensor;
using AclnnFFNParam = ops::adv::tests::ffn::AclnnFFNParam;
using FunctionType = ops::adv::tests::ffn::AclnnFFNParam::FunctionType;
using AclnnFFNVersion = ops::adv::tests::ffn::AclnnFFNParam::AclnnFFNVersion;

class Ts_Aclnn_FFN_WithParam : public Ts_WithParam<AclnnFFNCase> {};
class Ts_Aclnn_FFN_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<AclnnFFNCase> {};
class Ts_Aclnn_FFN_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<AclnnFFNCase> {};
class Ts_Aclnn_FFN_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<AclnnFFNCase> {};

#endif // UTEST_TS_FFN_H