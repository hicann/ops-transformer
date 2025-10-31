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
 * \file ts_fa.h
 * \brief FlashAttentionScore / FlashAttentionScoreGrad UTest 相关基类定义.
 */

#pragma once

#include "tests/utest/ts.h"
#include "fa_case.h"
#include "aclnn_fa_case.h"

using FaParam = ops::adv::tests::fa::FaParam;
using PseShapeType = ops::adv::tests::fa::FaParam::PseShapeType;
using DropMaskShapeType = ops::adv::tests::fa::FaParam::DropMaskShapeType;
using PaddingMaskShapeType = ops::adv::tests::fa::FaParam::PaddingMaskShapeType;
using AttenMaskShapeType = ops::adv::tests::fa::FaParam::AttenMaskShapeType;
using PrefixShapeType = ops::adv::tests::fa::FaParam::PrefixShapeType;
using LayoutType = ops::adv::tests::fa::FaParam::LayoutType;
using FaCase = ops::adv::tests::fa::FaCase;
using ExpectInfoWithSocversion = ops::adv::tests::utils::ExpectInfoWithSocversion;
using OpInfoWithSocversion = ops::adv::tests::utils::OpInfoWithSocversion;

class Ts_Fa : public Ts<FaCase> {};
class Ts_Fa_Ascend910B1 : public Ts_Ascend910B1<FaCase> {};
class Ts_Fa_Ascend910B2 : public Ts_Ascend910B2<FaCase> {};
class Ts_Fa_Ascend910B3 : public Ts_Ascend910B3<FaCase> {};
class Ts_Fa_Ascend910_9591 : public Ts_Ascend910_9591<FaCase> {};

class Ts_Fa_WithParam : public Ts_WithParam<FaCase> {};
class Ts_Fa_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<FaCase> {};
class Ts_Fa_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<FaCase> {};
class Ts_Fa_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<FaCase> {};
