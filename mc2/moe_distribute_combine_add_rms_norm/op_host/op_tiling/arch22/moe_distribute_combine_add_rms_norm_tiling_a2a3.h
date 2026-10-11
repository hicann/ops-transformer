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
 * \file moe_distribute_combine_add_rms_norm_tiling_a2a3.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_A2A3_H
#define MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_TILING_A2A3_H

#include "../moe_distribute_combine_add_rms_norm_tiling_base.h"

namespace optiling {
class MoeDistributeCombineAddRmsNormTilingFuncA2A3 : public MoeDistributeCombineAddRmsNormTilingFuncBase {
public:
    ge::graphStatus MoeDistributeCombineV2TilingFunc(gert::TilingContext* context) override;
};
} // namespace optiling
#endif
