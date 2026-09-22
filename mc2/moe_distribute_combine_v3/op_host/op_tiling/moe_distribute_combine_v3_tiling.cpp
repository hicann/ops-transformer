/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v3_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h"
#include "mc2_hcom_topo_info.h"
#include "../../../moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.h"
#include "mc2_exception_dump.h"
#include "moe_distribute_combine_v3_tiling_base.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
namespace {
// combineV2 算子原型输入张量在 ir 定义中的位置索引
constexpr uint32_t INPUT_TP_SEND_COUNTS_INDEX = 6;
constexpr uint32_t INPUT_X_ACTIVE_MASK_INDEX = 7;
constexpr uint32_t INPUT_ACTIVATION_SCALE_INDEX = 8;
constexpr uint32_t INPUT_WEIGHT_SCALE_INDEX = 9;
constexpr uint32_t INPUT_GROUP_LIST_INDEX = 10;
constexpr uint32_t INPUT_SHARED_EXPERT_X_INDEX = 12;
constexpr uint32_t INPUT_ELASTIC_INFO_INDEX = 13;
constexpr uint32_t INPUT_ORI_X_INDEX = 14;
constexpr uint32_t INPUT_CONST_EXPERT_ALPHA1_INDEX = 15;
constexpr uint32_t INPUT_CONST_EXPERT_ALPHA2_INDEX = 16;
constexpr uint32_t INPUT_CONST_EXPERT_V_INDEX = 17;
constexpr uint32_t INPUT_PERFORMANCE_INFO_INDEX = 18;
// combineV2 算子原型属性在 ir 定义中的位置索引
constexpr uint32_t ATTR_ZERO_EXPERT_NUM_INDEX = 14;
constexpr uint32_t ATTR_COPY_EXPERT_NUM_INDEX = 15;
constexpr uint32_t ATTR_CONST_EXPERT_NUM_INDEX = 16;
} // namespace

ge::graphStatus MoeDistributeCombineV3TilingFuncBase::MoeDistributeCombineV3TilingFunc(gert::TilingContext *context)
{
    CombineV2Config config;
    config.contextIndex = 0;      // 0: 根据combineV2算子原型标志位初始化context索引
    config.expandXIndex = 1;      // 1: 根据combineV2算子原型标志位初始化expandX索引
    config.expertIdsIndex = 2;    // 2: 根据combineV2算子原型标志位初始化expertIds索引
    config.assistInfoIndex = 3;   // 3: 根据combineV2算子原型标志位初始化assitInfo索引
    config.epSendCountIndex = 4;  // 4: 根据combineV2算子原型标志位初始化epSendCount索引
    config.expertScalesIndex = 5; // 5: 根据combineV2算子原型标志位初始化expertScales索引
    config.tpSendCountsIndex = INPUT_TP_SEND_COUNTS_INDEX;
    config.xActiveMaskIndex = INPUT_X_ACTIVE_MASK_INDEX;
    config.activationScaleIndex = INPUT_ACTIVATION_SCALE_INDEX;
    config.weightScaleIndex = INPUT_WEIGHT_SCALE_INDEX;
    config.groupListIndex = INPUT_GROUP_LIST_INDEX;
    config.sharedExpertXIndex = INPUT_SHARED_EXPERT_X_INDEX;
    config.elasticInfoIndex = INPUT_ELASTIC_INFO_INDEX;
    config.oriXIndex = INPUT_ORI_X_INDEX;
    config.constExpertAlpha1Index = INPUT_CONST_EXPERT_ALPHA1_INDEX;
    config.constExpertAlpha2Index = INPUT_CONST_EXPERT_ALPHA2_INDEX;
    config.constExpertVIndex = INPUT_CONST_EXPERT_V_INDEX;
    config.performanceInfoIndex = INPUT_PERFORMANCE_INFO_INDEX;
    config.outputXIndex = 0;           // 根据combineV2算子原型标志位设置outputX索引
    config.attrEpWorldSizeIndex = 0;   // 0: 根据combineV2算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 1;      // 1: 根据combineV2算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 2;  // 2: 根据combineV2算子原型标志位初始化moeExpertNum索引
    config.attrCclBufferSizeIndex = 3; // 3: 根据combineV2算子原型标志位初始化attrCclBufferSizeIndex索引
    config.attrTpWorldSizeIndex = 4;   // 4: 根据combineV2算子原型标志位初始化attrTpWorldSizeIndex索引
    config.attrTpRankIdIndex = 5;      // 5: 根据combineV2算子原型标志位初始化attrTpRankIdIndex索引
    config.attrExpertSharedTypeIndex = 6; // 6: 根据combineV2算子原型标志位初始化attrExpertSharedTypeIndex索引
    config.attrSharedExpertNumIndex = 7; // 7: 根据combineV2算子原型标志位初始化attrSharedExpertNumIndex索引
    config.attrSharedExpertRankNumIndex = 8; // 8: 根据combineV2算子原型标志位初始化attrSharedExpertRankNumIndex索引
    config.attrGlobalBsIndex = 9;            // 9: 根据combineV2算子原型标志位初始化attrGlobalBsIndex索引
    config.attrOutDTypeIndex = 10;           // 10: 根据combineV2算子原型标志位初始化attrOutDTypeIndex索引
    config.attrCommQuantModeIndex = 11; // 11: 根据combineV2算子原型标志位初始化attrCommQuantModeIndex索引
    config.attrGroupListTypeIndex = 12; // 12: 根据combineV2算子原型标志位初始化attrGroupListTypeIndex索引
    config.attrCommAlgIndex = 13;       // 13: 根据combineV2算子原型标志位初始化attrCommAlgIndex索引
    config.attrZeroExpertNumIndex = ATTR_ZERO_EXPERT_NUM_INDEX;
    config.attrCopyExpertNumIndex = ATTR_COPY_EXPERT_NUM_INDEX;
    config.attrConstExpertNumIndex = ATTR_CONST_EXPERT_NUM_INDEX;
    config.hasAddRmsNorm = false;
    config.isMc2Context = true;

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeDispatchV3 tiling");
    ge::graphStatus ret = MoeDistributeCombineV2TilingFuncNew(context, config);
    return ret;
}
} // namespace optiling

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeCombineV3ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MoeDistributeCombineV3");
}

__attribute__((constructor)) static void RegisterMoeDistributeCombineV3ExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeCombineV3", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeCombineV3", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeDistributeCombineV3",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MoeDistributeCombineV3).ExceptionDumpParseFunc(MoeDistributeCombineV3ExceptionImplWrapper);
}
#endif
