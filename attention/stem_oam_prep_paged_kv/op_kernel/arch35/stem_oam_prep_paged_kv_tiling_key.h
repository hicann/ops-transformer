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
 * \file stem_oam_prep_paged_kv_tiling_key.h
 * \brief StemOamPrepPagedKV tiling key definition (arch35)
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_KEY_H
#define STEM_OAM_PREP_PAGED_KV_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"
#include "stem_oam_prep_paged_kv_tiling_data.h"

ASCENDC_TPL_ARGS_DECL(StemOamPrepPagedKv, ASCENDC_TPL_BOOL_DECL(TILINGKEY, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(StemOamPrepPagedKvTilingData)))

#endif // STEM_OAM_PREP_PAGED_KV_TILING_KEY_H
