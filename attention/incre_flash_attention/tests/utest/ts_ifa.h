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
 * \file ts_ifa.h
 * \brief IncreFlashAttention UTest 相关基类定义.
 */

#include "tests/utest/ts.h"
#include "ifa_case.h"

using IfaCase = ops::adv::tests::ifa::IfaCase;
using AntiQuantShapeType = IfaCase::AntiQuantShapeType;
using QuantShapeType = IfaCase::QuantShapeType;
using AttenMaskShapeType = IfaCase::AttenMaskShapeType;
using PseShiftShapeType = IfaCase::PseShiftShapeType;

class Ts_Ifa : public Ts<IfaCase> {};
class Ts_Ifa_Ascend910B2 : public Ts_Ascend910B2<IfaCase> {};
class Ts_Ifa_Ascend310P3 : public Ts_Ascend310P3<IfaCase> {};
class Ts_Ifa_Ascend910_9591 : public Ts_Ascend910_9591<IfaCase> {};

class Ts_Ifa_WithParam : public Ts_WithParam<IfaCase> {};
class Ts_Ifa_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<IfaCase> {};
class Ts_Ifa_WithParam_Ascend310P3 : public Ts_WithParam_Ascend310P3<IfaCase> {};
class Ts_Ifa_WithParam_Ascend910_9591 : public Ts_WithParam_Ascend910_9591<IfaCase> {};
