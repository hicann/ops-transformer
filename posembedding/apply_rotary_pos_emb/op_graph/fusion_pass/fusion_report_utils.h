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
 * \file fusion_report_utils.h
 * \brief Compatibility wrappers for GE fusion result reporting APIs.
 *
 * GE 9.1.0+ provides GraphFuseInspectorUtils::ReportFuse and GE 9.2.0+ provides ReportMatch.
 * The declarations are kept local (aligned with GE, weak symbols) so that the pass still
 * compiles and runs against older GE runtimes: compile-time version macros gate the API
 * usage, and null weak-symbol checks give a silent runtime fallback.
 * Do not include ge/fusion/graph_fuse_inspector_utils.h in the same translation unit.
 */

#ifndef TRANSFORMER_APPLY_ROTARY_POS_EMB_FUSION_REPORT_UTILS_H_
#define TRANSFORMER_APPLY_ROTARY_POS_EMB_FUSION_REPORT_UTILS_H_

#include <vector>

#include "graph/graph.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

// Keep these declarations aligned with GE. Do not include graph_fuse_inspector_utils.h
// in the same translation unit.
namespace ge {
namespace fusion {
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
#if GE_COMPILER_VERSION_NUM >= 90200000U
    static Status ReportMatch(const std::vector<GNode>& matchedNodes, CustomPassContext& ctx) __attribute__((weak));
#endif
};
#endif
} // namespace fusion
} // namespace ge

namespace ops {
namespace fusion {

/**
 * @brief Report a matched node set when the runtime provides the reporting API (GE 9.2.0+).
 * @return Reporting status, or SUCCESS when reporting is unavailable.
 */
inline ge::Status ReportMatchNodes(const std::vector<ge::GNode>& nodes, ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90200000U)
    if (ge::fusion::GraphFuseInspectorUtils::ReportMatch != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportMatch(nodes, context);
    }
#else
    (void)nodes;
    (void)context;
#endif
    return ge::SUCCESS;
}

/**
 * @brief Report fusion after rewiring and before removing the original nodes (GE 9.1.0+).
 *        An empty @p after means nodes are deleted without new nodes being added.
 * @return Reporting status, or SUCCESS when reporting is unavailable.
 */
inline ge::Status ReportFuseNodes(const std::vector<ge::GNode>& before, const std::vector<ge::GNode>& after,
                                  ge::CustomPassContext& context)
{
#if defined(GE_COMPILER_VERSION_NUM) && (GE_COMPILER_VERSION_NUM >= 90100000U)
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportFuse(before, after, context);
    }
#else
    (void)before;
    (void)after;
    (void)context;
#endif
    return ge::SUCCESS;
}

} // namespace fusion
} // namespace ops

#endif // TRANSFORMER_APPLY_ROTARY_POS_EMB_FUSION_REPORT_UTILS_H_
