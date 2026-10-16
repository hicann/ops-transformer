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
 * \file op_const_def_transformer.h
 * \brief data type constants definition
 */

#pragma once

#include "graph/types.h"

namespace Ops {
namespace Transformer {
constexpr ge::DataType DT_FLOAT = static_cast<ge::DataType>(0);
constexpr ge::DataType DT_FLOAT16 = static_cast<ge::DataType>(1);
constexpr ge::DataType DT_INT8 = static_cast<ge::DataType>(2);
constexpr ge::DataType DT_INT32 = static_cast<ge::DataType>(3);
constexpr ge::DataType DT_UINT8 = static_cast<ge::DataType>(4);
constexpr ge::DataType DT_INT16 = static_cast<ge::DataType>(6);
constexpr ge::DataType DT_UINT16 = static_cast<ge::DataType>(7);
constexpr ge::DataType DT_UINT32 = static_cast<ge::DataType>(8);
constexpr ge::DataType DT_INT64 = static_cast<ge::DataType>(9);
constexpr ge::DataType DT_UINT64 = static_cast<ge::DataType>(10);
constexpr ge::DataType DT_DOUBLE = static_cast<ge::DataType>(11);
constexpr ge::DataType DT_BOOL = static_cast<ge::DataType>(12);
constexpr ge::DataType DT_STRING = static_cast<ge::DataType>(13);
constexpr ge::DataType DT_DUAL_SUB_INT8 = static_cast<ge::DataType>(14);
constexpr ge::DataType DT_DUAL_SUB_UINT8 = static_cast<ge::DataType>(15);
constexpr ge::DataType DT_COMPLEX64 = static_cast<ge::DataType>(16);
constexpr ge::DataType DT_COMPLEX128 = static_cast<ge::DataType>(17);
constexpr ge::DataType DT_QINT8 = static_cast<ge::DataType>(18);
constexpr ge::DataType DT_QINT16 = static_cast<ge::DataType>(19);
constexpr ge::DataType DT_QINT32 = static_cast<ge::DataType>(20);
constexpr ge::DataType DT_QUINT8 = static_cast<ge::DataType>(21);
constexpr ge::DataType DT_QUINT16 = static_cast<ge::DataType>(22);
constexpr ge::DataType DT_RESOURCE = static_cast<ge::DataType>(23);
constexpr ge::DataType DT_STRING_REF = static_cast<ge::DataType>(24);
constexpr ge::DataType DT_DUAL = static_cast<ge::DataType>(25);
constexpr ge::DataType DT_VARIANT = static_cast<ge::DataType>(26);
constexpr ge::DataType DT_BF16 = static_cast<ge::DataType>(27);
constexpr ge::DataType DT_UNDEFINED = static_cast<ge::DataType>(28);
constexpr ge::DataType DT_INT4 = static_cast<ge::DataType>(29);
constexpr ge::DataType DT_UINT1 = static_cast<ge::DataType>(30);
constexpr ge::DataType DT_INT2 = static_cast<ge::DataType>(31);
constexpr ge::DataType DT_UINT2 = static_cast<ge::DataType>(32);
constexpr ge::DataType DT_COMPLEX32 = static_cast<ge::DataType>(33);
constexpr ge::DataType DT_HIFLOAT8 = static_cast<ge::DataType>(34);
constexpr ge::DataType DT_FLOAT8_E5M2 = static_cast<ge::DataType>(35);
constexpr ge::DataType DT_FLOAT8_E4M3FN = static_cast<ge::DataType>(36);
constexpr ge::DataType DT_FLOAT8_E8M0 = static_cast<ge::DataType>(37);
constexpr ge::DataType DT_FLOAT6_E3M2 = static_cast<ge::DataType>(38);
constexpr ge::DataType DT_FLOAT6_E2M3 = static_cast<ge::DataType>(39);
constexpr ge::DataType DT_FLOAT4_E2M1 = static_cast<ge::DataType>(40);
constexpr ge::DataType DT_FLOAT4_E1M2 = static_cast<ge::DataType>(41);
constexpr ge::DataType DT_HIFLOAT4 = static_cast<ge::DataType>(42);
constexpr ge::DataType DT_HIFLOAT4_SCALE = static_cast<ge::DataType>(43);
} // namespace Transformer
} // namespace Ops
