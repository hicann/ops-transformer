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
 * \file ts_mla_prolog_v3.h
 * \brief MlaPrologV3 UTest 相关基类定义.
 */

#ifndef TS_MLA_PROLOG_V3_H
#define TS_MLA_PROLOG_V3_H

#include "tests/utest/ts.h"
#include "mla_prolog_v3_case.h"

using MlaPrologV3Param = ops::adv::tests::MlaPrologV3::MlaPrologV3Param;
using CacheModeType = ops::adv::tests::MlaPrologV3::MlaPrologV3Param::CacheModeType;
using MlaPrologV3Case = ops::adv::tests::MlaPrologV3::MlaPrologV3Case;

class Ts_MlaPrologV3 : public Ts<MlaPrologV3Case> {};
class Ts_MlaPrologV3_Ascend910B1 : public Ts_Ascend910B1<MlaPrologV3Case> {};
class Ts_MlaPrologV3_Ascend910B2 : public Ts_Ascend910B2<MlaPrologV3Case> {};
class Ts_MlaPrologV3_Ascend910B3 : public Ts_Ascend910B3<MlaPrologV3Case> {};

class Ts_MlaPrologV3_WithParam : public Ts_WithParam<MlaPrologV3Case> {};
class Ts_MlaPrologV3_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<MlaPrologV3Case> {};
class Ts_MlaPrologV3_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<MlaPrologV3Case> {};
class Ts_MlaPrologV3_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<MlaPrologV3Case> {};

#endif