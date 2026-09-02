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
 * \file grouped_matmul_finalize_routing_weight_quant_tiling.h
 * \brief
 */
#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_H

#include <graph/utils/type_utils.h>
#include <sstream>
#include <map>
#include <unordered_set>
#include <vector>
#include <functional>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

#include "./grouped_matmul_finalize_routing_quant_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h"
#include "../../../op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h"
#include "../../grouped_matmul_finalize_routing_tiling.h"
#include "../../../../grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
using namespace GMMFinalizeRoutingArch35Tiling;

namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling {
constexpr static int64_t GMMFR_WEIGHT_QUANT_TILING_VEC_ANTIQUANT = 2;

constexpr uint32_t AIC_AIV_CORE_RATIO = 2;

enum class ScenarioType {
    NONE = 0,
    MX_A8W4_WEIGHT_NZ = 1,
};

struct GMMFRWeightQuantInputParams {
    std::string opName;
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType wDtype = ge::DT_INT8;
    ge::Format wFormat = ge::FORMAT_ND;

    bool xTrans = false;
    bool wTrans = false;

    float sharedInputWeight = 0.0f;
    int64_t shareInputOffset = 0;

    int64_t mSize = -1;
    int64_t kSize = -1;
    int64_t nSize = -1;
    int64_t groupListType = -1;
    int64_t groupNum = -1;
    int64_t outputBS = -1;
    int64_t outputDtype = -1;

    bool hasBias = false;
    int64_t sharedInputLen = 0;
    float residualScale = 0.0;
};

class TilingKeyConfigure {
public:
    uint8_t socVersionType = 0;
    uint8_t quantizationScenario = 0;
    uint8_t algorithm = 0;
    uint8_t transposeSituation = 0;
    uint8_t antiquantType = 0;
    uint8_t quantType = 0;
    uint8_t optionInputSituation = 0;
    uint8_t weightFormat = 0;
    uint8_t templateCustom = 0;
    uint8_t apiConstexpr = 0;
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "socVersionType: " << static_cast<uint32_t>(this->socVersionType)
           << " quantizationScenario: " << static_cast<uint32_t>(this->quantizationScenario)
           << " algorithm: " << static_cast<uint32_t>(this->algorithm)
           << " transposeSituation: " << static_cast<uint32_t>(this->transposeSituation)
           << " antiquantType: " << static_cast<uint32_t>(this->antiquantType)
           << " quantType: " << static_cast<uint32_t>(this->quantType)
           << " optionInputSituation: " << static_cast<uint32_t>(this->optionInputSituation)
           << " weightFormat: " << static_cast<uint32_t>(this->weightFormat)
           << " templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << " apiConstexpr: " << static_cast<uint32_t>(this->apiConstexpr);
        OP_LOGI("GMMWeightQuantBatchMatmul", "tilingKeyConfigure: %s", ss.str().c_str());
        return;
    }
};

template <typename T>
auto CeilDiv(T a, T b) -> T
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
auto CeilAlign(T a, T b) -> T
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

enum class DataSize {
    B4_DATA_SIZE = 0,
    B8_DATA_SIZE = 1,
    RESERVED = 999,
};

DataSize GetSizeByDataType(ge::DataType dType);

class GMMFRWeightQuantTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GMMFRWeightQuantTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~GMMFRWeightQuantTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

    GMMFRWeightQuantInputParams inputParams_;

private:
    const GroupedMatmulFinalizeRoutingCompileInfo *compileInfoPtr_;
    ScenarioType scenarioType_ = ScenarioType::NONE;
    std::vector<std::function<bool(gert::TilingContext *context)>> checkConditionFuncs_;
    std::vector<std::function<void(gert::TilingContext *context, GMMFRWeightQuantInputParams &inputParams_)>>
        SetInputFuncs_;
    GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData tilingData_;

    void RunSetInputFunc();
    bool SetMxA8W4NzInputFunc();
    bool SetMxA8W4NzConditionFunc();
    bool CheckCoreNum() const;
    bool RunCheckFunc();
    bool InferScenario();

    int64_t coreNum_ = 0;
    int64_t aivNum_ = 0;
};
} // namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling
} // namespace optiling

#endif // GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILING_H
