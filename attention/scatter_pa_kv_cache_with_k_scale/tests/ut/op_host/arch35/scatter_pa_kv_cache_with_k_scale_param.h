/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_PARAM_H
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_PARAM_H

#include "op_host_csv_case_loader.h"
#include <sstream>

namespace ScatterPaKvCacheWithKScaleUT {

struct ScatterPaKvCacheWithKScaleHostUtParamBase : public HostUtParamBase {
    std::string cache_layout;

    ScatterPaKvCacheWithKScaleHostUtParamBase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        this->cache_layout = ReadMap(csvMap, "cache_layout", "BNBD");
    }
};

struct ScatterPaKvCacheWithKScaleTilingUtParam : public ScatterPaKvCacheWithKScaleHostUtParamBase {
    gert::TilingContextPara::TensorDescription key = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_cache = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value_cache = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription slot_mapping = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_scale_cache = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription key_cache_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value_cache_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_scale_cache_out = TD_DEFAULT;

    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    ScatterPaKvCacheWithKScaleTilingUtParam(const csv_map &csvMap)
        : ScatterPaKvCacheWithKScaleHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_cache_shape", "key_cache_dtype", "key_cache_format", this->key_cache));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_cache_shape", "value_cache_dtype", "value_cache_format", this->value_cache));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "slot_mapping_shape", "slot_mapping_dtype", "slot_mapping_format", this->slot_mapping));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_scale_shape", "key_scale_dtype", "key_scale_format", this->key_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_scale_cache_shape", "key_scale_cache_dtype",
                                                     "key_scale_cache_format", this->key_scale_cache));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "key_cache_out_shape", "key_cache_out_dtype",
                                                      "key_cache_out_format", this->key_cache_out));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "value_cache_out_shape", "value_cache_out_dtype",
                                                      "value_cache_out_format", this->value_cache_out));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "key_scale_cache_out_shape", "key_scale_cache_out_dtype",
                                                      "key_scale_cache_out_format", this->key_scale_cache_out));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct ScatterPaKvCacheWithKScaleInferShapeUtParam : public ScatterPaKvCacheWithKScaleHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription key = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_cache = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value_cache = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription slot_mapping = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_scale_cache = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription key_cache_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value_cache_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_scale_cache_out = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    ScatterPaKvCacheWithKScaleInferShapeUtParam(const csv_map &csvMap)
        : ScatterPaKvCacheWithKScaleHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_cache_shape", "key_cache_dtype", "key_cache_format", this->key_cache));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_cache_shape", "value_cache_dtype", "value_cache_format", this->value_cache));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "slot_mapping_shape", "slot_mapping_dtype", "slot_mapping_format", this->slot_mapping));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_scale_shape", "key_scale_dtype", "key_scale_format", this->key_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_scale_cache_shape", "key_scale_cache_dtype",
                                                     "key_scale_cache_format", this->key_scale_cache));

        this->outputInstance.emplace_back(GetTensorGE(csvMap, "key_cache_out_shape", "key_cache_out_dtype",
                                                      "key_cache_out_format", this->key_cache_out));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "value_cache_out_shape", "value_cache_out_dtype",
                                                      "value_cache_out_format", this->value_cache_out));
        this->outputInstance.emplace_back(GetTensorGE(csvMap, "key_scale_cache_out_shape", "key_scale_cache_out_dtype",
                                                      "key_scale_cache_out_format", this->key_scale_cache_out));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "key_cache_out_shape")),
                                       GetShapeArr(ReadMap(csvMap, "value_cache_out_shape")),
                                       GetShapeArr(ReadMap(csvMap, "key_scale_cache_out_shape"))};
        }
    }
};

} // namespace ScatterPaKvCacheWithKScaleUT

#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_PARAM_H
