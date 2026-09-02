/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_H
#define TRANSFORMER_GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_H

#include "version/cann_version.h"

#define GROUPED_MATMUL_GRAPH_FUSION_SUPPORT_VERSION 90000000

#if CANN_VERSION_NUM >= GROUPED_MATMUL_GRAPH_FUSION_SUPPORT_VERSION
#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {

class __attribute__((visibility("default"))) GroupedMatmulTransFusionPass : public ge::fusion::FusionBasePass {
protected:
    ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext) override;
};

} // namespace ops

#endif // CANN_VERSION_NUM >= GROUPED_MATMUL_GRAPH_FUSION_SUPPORT_VERSION
#endif // TRANSFORMER_GROUPED_MATMUL_TRANSPOSE_FUSION_PASS_H
