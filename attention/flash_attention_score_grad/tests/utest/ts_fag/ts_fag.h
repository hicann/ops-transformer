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
 * \file ts_fag.h
 * \brief FlashAttentionScoreGrad UTest 基类定义.
 */

#include "ts_fa/ts_fa.h"
#include "fag_case.h"

using FagCase = ops::adv::tests::fag::FagCase;
using ExpectInfoWithSocversion = ops::adv::tests::utils::ExpectInfoWithSocversion;
using OpInfoWithSocversion = ops::adv::tests::utils::OpInfoWithSocversion;

class Ts_Fag : public Ts<FagCase> {};
class Ts_Fag_Ascend910B1 : public Ts_Ascend910B1<FagCase> {};
class Ts_Fag_Ascend910B2 : public Ts_Ascend910B2<FagCase> {};
class Ts_Fag_Ascend910B3 : public Ts_Ascend910B3<FagCase> {};

class Ts_Fag_WithParam : public Ts_WithParam<FagCase> {};
class Ts_Fag_WithParam_Ascend910B1 : public Ts_WithParam_Ascend910B1<FagCase> {};
class Ts_Fag_WithParam_Ascend910B2 : public Ts_WithParam_Ascend910B2<FagCase> {};
class Ts_Fag_WithParam_Ascend910B3 : public Ts_WithParam_Ascend910B3<FagCase> {};
