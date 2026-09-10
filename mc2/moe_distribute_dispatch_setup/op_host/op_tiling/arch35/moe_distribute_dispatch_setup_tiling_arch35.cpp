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
 * \file moe_distribute_dispatch_setup_tiling_arch35.cpp
 * \brief
 */

#include "moe_distribute_dispatch_setup_tiling_arch35.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace {

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal;
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};
} // namespace

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(MoeDistributeDispatchSetup, MoeDistributeDispatchSetupTilingA5, 0);
void MoeDistributeDispatchSetupTilingA5::SetHcommCfg()
{
    OP_LOGD(nodeName_, "MoeDistributeCombineSetup groupEp = %s", groupEp_.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    uint8_t aivEngineValue = npuArch_ == NpuArch::DAV_3510 ? mc2tiling::AIV_ENGINE : mc2tiling::AIV_ENGINE;

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp_, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(aivEngineValue); // AIV_UB-MEM or AIV_URMA
    mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData_->mc2CcTiling)->protocal = 1; // 0: UB-MEM, 1: URMA
}

bool MoeDistributeDispatchSetupTilingA5::IsCapable()
{
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_LOGD(nodeName_, "Do MoeDistributeDispatchSetupTilingA5 tiling.");
        return true;
    }
    return false;
}

ge::graphStatus MoeDistributeDispatchSetupTilingA5::DoOpTiling()
{
    return MoeDistributeDispatchSetupTilingFuncImpl();
}
} // namespace optiling
