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
 * \file moe_distribute_combine_tiling_a5.cpp
 * \brief
 */

#include "moe_distribute_combine_tiling_a5.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"

#include "op_host/op_tiling/moe_tiling_base.h"
#include "mc2_hcom_topo_info.h"

using namespace Ops::Transformer::OpTiling;
using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace {
constexpr uint32_t INT8_COMM_QUANT = 2U;
}

namespace optiling {

ge::graphStatus MoeDistributeCombineTilingA5::MoeDistributeCombineA5TilingCheckAttr(gert::TilingContext *context,
                                                                                    uint32_t &commQuantMode)
{
    const char *nodeName = context->GetNodeName();
    MoeDistributeCombineTilingData *tilingData = context->GetTilingData<MoeDistributeCombineTilingData>();
    std::string groupEp = "";
    std::string groupTp = "";
    uint32_t tpWorldSize = 1;
    bool isShared = true;
    uint32_t localMoeExpertNum = 1;
    commQuantMode = 0U;

    // 获取入参属性
    OP_TILING_CHECK(
        GetAttrAndSetTilingData(context, *tilingData, nodeName, groupEp, groupTp, commQuantMode) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);

    // 检查输入输出的dim、format、dataType
    OP_TILING_CHECK(
        MoeDistributeCombineTilingHelper::TilingCheckMoeDistributeCombine(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling check params failed"), return ge::GRAPH_FAILED);

    // 检查属性的取值是否合法
    OP_TILING_CHECK(!CheckAttrs(context, *tilingData, nodeName, localMoeExpertNum),
                    OP_LOGE(nodeName, "attr check failed."), return ge::GRAPH_FAILED);

    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineInfo.sharedExpertRankNum;
    uint32_t epRankId = tilingData->moeDistributeCombineInfo.epRankId;
    if (epRankId >= sharedExpertRankNum) { // 本卡为moe专家
        isShared = false;
    }

    // 检查shape各维度并赋值h,k
    OP_TILING_CHECK(!CheckTensorShape(context, *tilingData, nodeName, isShared, localMoeExpertNum),
                    OP_LOGE(nodeName, "param dim check failed."), return ge::GRAPH_FAILED);

    // 校验win区大小
    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName, localMoeExpertNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkspace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetHCommCfg(context, tilingData, groupEp, groupTp, tpWorldSize) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineTilingA5::MoeDistributeCombineA5TilingFuncImpl(gert::TilingContext *context)
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);
    MC2_CHECK_LOG_RET(context->GetNodeName(), ret);

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeCombine Tiling func");
    MoeDistributeCombineTilingData *tilingData = context->GetTilingData<MoeDistributeCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    uint32_t commQuantMode;

    // 统一调用所有校验逻辑
    OP_TILING_CHECK(MoeDistributeCombineA5TilingCheckAttr(context, commQuantMode) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "CombineSA5Tiling attr check failed."), return ge::GRAPH_FAILED);

    uint32_t quantMode = TILINGKEY_NO_QUANT;
    uint32_t layeredMode = TILINGKEY_TPL_MTE; // A2
    if (commQuantMode == INT8_COMM_QUANT) {
        quantMode = TILINGKEY_INT8_QUANT;
    }
    uint32_t archTag = (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510) ? TILINGKEY_TPL_A5 : TILINGKEY_TPL_A3;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(quantMode, layeredMode, archTag);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
    uint32_t numBlocks = 1U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->moeDistributeCombineInfo.aivNum = aivNum;
    tilingData->moeDistributeCombineInfo.totalUbSize = ubSize;
    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %lu, ubsize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeCombineTilingA5::CheckEpWorldSize(const char *nodeName, uint32_t epWorldSize)
{
    // 为支持在 A5 上的验证，放开 epWorldSize 为 2 或 4 的校验
    // 检验epWorldSize是否是2的倍数
    OP_TILING_CHECK(epWorldSize % 2 != 0,
                    OP_LOGE(nodeName, "epWorldSize should be divisible by 2, but got epWorldSize = %u.", epWorldSize),
                    return false);

    OP_TILING_CHECK((256 % epWorldSize != 0) && (epWorldSize % 144 != 0),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              "2, 4, 8, 16, 32, 64, 128, 144, 256, or 288"),
                    return false);

    return true;
}

ge::graphStatus MoeDistributeCombineTilingA5::MoeDistributeCombineTilingFuncImpl(std::string &socVersion,
                                                                                 gert::TilingContext *context)
{
    ge::graphStatus ret = MoeDistributeCombineA5TilingFuncImpl(context);
    return ret;
}

// A5 采用 MTE 方式通信，复用 A3 tiling
REGISTER_OPS_TILING_TEMPLATE(MoeDistributeCombine, MoeDistributeCombineTilingA5, 0);

} // namespace optiling
