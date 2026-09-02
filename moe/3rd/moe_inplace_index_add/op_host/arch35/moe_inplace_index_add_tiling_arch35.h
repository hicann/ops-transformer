/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_tiling_arch35.h
 * \brief
 */

#ifndef MOE_INPLACE_INDEX_ADD_TILING_ARCH35_H_
#define MOE_INPLACE_INDEX_ADD_TILING_ARCH35_H_

#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "log/log.h"
#include <nlohmann/json.hpp>
#include "../../../inc/moe_log.h"
#include "platform/platform_info.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MoeInplaceIndexAddForAscendcTilingData)
TILING_DATA_FIELD_DEF(int64_t, preAxis);
TILING_DATA_FIELD_DEF(int64_t, varInAxis);
TILING_DATA_FIELD_DEF(int64_t, updatesInAxis);
TILING_DATA_FIELD_DEF(int64_t, afterAxis);
TILING_DATA_FIELD_DEF(int64_t, ubFactor);
TILING_DATA_FIELD_DEF(int64_t, colUbFactor);
TILING_DATA_FIELD_DEF(int64_t, indicesUbFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeInplaceIndexAdd, MoeInplaceIndexAddForAscendcTilingData)

/*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
inline std::unique_ptr<nlohmann::json> MoeGetCompileInfoJson(gert::TilingParseContext *context)
{
    auto json_str = context->GetCompiledJson();
    MOE_OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo =
        std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
    return parsed_object_cinfo;
}

/*
 * @brief: read one compile item from json object
 * @param [in] all_vars: json object
 * @param [in] name: item name to read
 * @param [out] value: item value to load
 * @return bool: success or failed
 */
template <typename T>
bool MoeReadCompileItem(const nlohmann::json &all_vars, const std::string &name, T &value)
{
    if (all_vars.empty()) {
        return false;
    }

    if (all_vars.count(name) == 0) {
        return false;
    }

    value = all_vars[name].get<T>();
    return true;
}

template <typename T1, typename T2>
void MoeReadCompileItem(const nlohmann::json &all_vars, const std::string &name, T1 &value, const T2 default_value)
{
    if (!MoeReadCompileItem(all_vars, name, value)) {
        value = static_cast<T1>(default_value);
    }
}

/*
 * @brief: get running core number.
 * @param [in] TilingParseContext: context
 * @param [out] int64_t: running core number
 * @return bool: result
 */
bool MoeGetTilingCoreNum(const gert::TilingParseContext *context, uint32_t &core_num);

ge::graphStatus MoeInplaceIndexAddTilingForAscendC(gert::TilingContext *context);

class MoeInplaceIndexAddTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeInplaceIndexAddTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~MoeInplaceIndexAddTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override {}
    void SelTemplateByInput();
    ge::graphStatus CheckInputDtype();
    ge::graphStatus CheckInputShape();
    bool CompareShape(const gert::Shape &shape1, const gert::Shape &shape2, int64_t dim = static_cast<int64_t>(-1));
    void CombineAxis(const gert::Shape &varShape, const gert::Shape &updatesShape);
    void GetCastTypeSize();
    uint32_t GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend);
    void GetCastTypeForSort();

public:
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubFactor_ = 0;
    int64_t colUbFactor_ = 0;
    int64_t indicesUbFactor_ = 1;
    int64_t dim_ = 0;
    int64_t rank_ = 1;
    int64_t varAxis_ = 1;
    int64_t indicesAxis_ = 0;
    int64_t updatesAxis_ = 1;
    int64_t preAxis_ = 1;
    int64_t varInAxis_ = 1;
    int64_t updatesInAxis_ = 1;
    int64_t afterAxis_ = 1;
    int64_t castTypeSize_ = 0;
    int64_t varTypeSize_ = 0;
    int64_t indicesTypeSize_ = 0;
    uint64_t withAlpha_ = 0;
    uint64_t tilingKey_ = 0;
    int64_t ubIndexFactor_ = 0;
    int64_t ubVarFactor_ = 0;
    int64_t afterAxisFactor_ = 1;
    int64_t ubQuantaIndxFactor_ = 0;
    int64_t usedCoreNumBefore_ = 0;
    int64_t usedCoreNumAfter_ = 0;
    int64_t eachCorePreAxisCount_ = 0;
    int64_t tailCorePreAxisCount_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t tailCoreAfterAxisCount_ = 0;
    int64_t updateLoopSize_ = 0;
    int64_t updateTailNum_ = 0;
    int64_t indicesLoopSize_ = 0;
    int64_t indiceAxisTailNum_ = 0;
    int64_t isSplitPreAxis_ = 0;

    int64_t mainCoreIndicesLoop_ = 0;
    int64_t tailCoreIndicesLoop_ = 0;
    int64_t mainCoreTailIndices_ = 0;
    int64_t tailCoreTailIndices_ = 0;

    int64_t tailUpdateLoopSize_ = 0;
    int64_t tailUpdateAxisNum_ = 0;
    int64_t isSplitAfterAxis_ = 0;
    int64_t isSplitIndicesAxis_ = 0;

    int64_t tailBlockIndicesLoopSize_ = 0;
    int64_t sortShareBufSize_ = 0;
    int64_t normalUpdatesPreNum_ = 0;
    int64_t updatesPreLoop_ = 0;
    int64_t tailUpdatesPreNum_ = 0;

    int64_t eachCoreIndexCount_ = 0;
    int64_t tailCoreIndexCount_ = 0;
    int64_t eachCoreVarCount_ = 0;
    int64_t tailCoreVarCount_ = 0;
    int64_t isDeterminstic_ = 0;
    int64_t isSimdSort_ = 0;
    int64_t isSimdNoSort_ = 0;
    int64_t isSimtSort_ = 0;
    int64_t isSimtNoSort_ = 0;
    int64_t ubVarOptiFactor_ = 0;
    int64_t isOpti_ = 0;
    int64_t indicesStride_ = 1;
    uint64_t indicesCastMode_ = 0; // 0: 不Cast; 1：int32 Cast int16; 2：int64 Cast int32; 3：int64 Cast int16; 4:int32
                                   // Cast uint8; 5:int64 Cast uint8.
    int64_t indicesCastDtypeSize_ = 0;

    ge::DataType dtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    ge::DataType indicesCastDtype_ = ge::DT_UNDEFINED;
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INPLACE_INDEX_ADD_TILING_H_
