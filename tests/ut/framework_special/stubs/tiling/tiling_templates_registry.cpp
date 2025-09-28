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
 * \file tiling_templates_registry.cpp
 * \brief
 */

#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;

TilingRegistry &TilingRegistry::GetInstance()
{
    static TilingRegistry registry_impl_;
    return registry_impl_;
}

TilingRegistryNew &TilingRegistryNew::GetInstance()
{
    static TilingRegistryNew registry_impl_;
    return registry_impl_;
}
