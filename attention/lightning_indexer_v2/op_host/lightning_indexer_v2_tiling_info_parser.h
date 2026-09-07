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
 * \file lightning_indexer_v2_tiling_info_parser.h
 * \brief LightningIndexerV2 tiling info parser
 */

#ifndef LIGHTNING_INDEXER_V2_TILING_INFO_PARSER_H_
#define LIGHTNING_INDEXER_V2_TILING_INFO_PARSER_H_

#include "lightning_indexer_v2_tiling.h"

namespace optiling {
ge::graphStatus ParseAndCheckLIV2Arch35(gert::TilingContext *context, LIV2TilingInfo &tilingInfo);
} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_TILING_INFO_PARSER_H_
