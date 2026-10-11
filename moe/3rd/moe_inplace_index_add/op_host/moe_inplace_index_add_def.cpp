/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> varSupportedTypes = {
    ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_UINT8, ge::DT_INT8,    ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64,   ge::DT_BOOL,  ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_UINT8, ge::DT_INT8,    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,   ge::DT_BOOL};

static const std::vector<ge::DataType> indicesSupportedTypes = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class MoeInplaceIndexAdd : public OpDef {
public:
    explicit MoeInplaceIndexAdd(const char *name) : OpDef(name) {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("alpha")
            .ParamType(OPTIONAL)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(varSupportedTypes)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("axis").AttrType(REQUIRED).Int();
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(MoeInplaceIndexAdd);
}
