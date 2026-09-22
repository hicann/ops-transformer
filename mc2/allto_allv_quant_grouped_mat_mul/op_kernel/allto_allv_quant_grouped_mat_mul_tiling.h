/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
#define __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__

#include "mc2_templates/common/a2av_common_tiling.h"

#pragma pack(push, 8)
struct QuantAlltoAllvGroupedMatmulTilingData {
    MC2KernelTemplate::HcclA2avTilingInfo hcclA2avTilingInfo;
    MC2KernelTemplate::TaskTilingInfo taskTilingInfo;
    bool isPermuteOut = false;
    bool isNeedMM = false;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData gmmQuantTilingData;
    Mc2GroupedMatmulTilingData::GMMQuantTilingData mmQuantTilingData;
};
#pragma pack(pop)

#endif // __ALL_TO_ALLV_GROUPED_MAT_MUL_TILING_H__
