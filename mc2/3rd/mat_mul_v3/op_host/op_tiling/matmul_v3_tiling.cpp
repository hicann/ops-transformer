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
 * \file matmul_v3_tiling.cc
 * \brief
 */
#include "matmul_v3_tiling.h"

#include <type_traits>

#include "runtime/tiling_context.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "op_cache_tiling.h"
#include "matmul_v3_base_tiling.h"
#include "matmul_v3_compile_info.h"
#include "matmul_v3_simplifiedkey.h"
#include "matmul_v3_platform_common.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace optiling::matmul_v3;
using Ops::Transformer::OpTiling::TilingRegistry;

namespace {
static const size_t DEST_MAX = 100;
static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
static const int32_t INPUT0_INDEX = 0;
static const int32_t INPUT1_INDEX = 1;
static const int32_t BIAS_INDEX = 2;
}

namespace optiling {

REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);

static ge::graphStatus MatmulV3TilingFunc(gert::TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMatmulV3(gert::TilingParseContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"), return ge::GRAPH_FAILED);
  fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
  OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                  return ge::GRAPH_FAILED);

  auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
  OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                  return ge::GRAPH_FAILED);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
  std::string val;
  std::string dataMoveL12Bt;
  platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
  platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
  compileInfoPtr->supportL0c2out = !val.empty();
  compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
  compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
  compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
  compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                       // 1024 is btSize
  compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize;  // 4096 is btSize
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

  if(!TilingPrepareForOpCache(context)) {
      return ge::GRAPH_FAILED;
  }
  OP_LOGI(
      context->GetNodeName(),
      "parse compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
      static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->l1Size, compileInfoPtr->l2Size,
      compileInfoPtr->aicNum, compileInfoPtr->supportL0c2out, compileInfoPtr->supportL12BtBf16);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatMulV3)
    .Tiling(MatmulV3TilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForMatmulV3)
    .GenSimplifiedKey(GenSimplifiedKey);
}
