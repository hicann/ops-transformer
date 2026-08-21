/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_PARAM_H
#define QUANT_FLASH_ATTN_PARAM_H

#include <memory>
#include <sstream>
#include "op_host_csv_case_loader.h"

namespace QuantFlashAttnUT {

struct QuantFlashAttnHostUtParamBase : public HostUtParamBase {
    int64_t quant_compute_mode;
    float softmax_scale;
    int64_t mask_mode;
    int64_t win_left;
    int64_t win_right;
    int64_t max_seqlen_q;
    int64_t max_seqlen_kv;
    std::string layout_q;
    std::string layout_q_descale;
    std::string layout_kv;
    std::string layout_out;
    bool return_softmax_lse;

    QuantFlashAttnHostUtParamBase(const csv_map &csvMap)
        : HostUtParamBase(csvMap)
    {
        this->quant_compute_mode = std::stoll(ReadMap(csvMap, "quant_compute_mode"));
        this->softmax_scale = std::stof(ReadMap(csvMap, "softmax_scale"));
        this->mask_mode = std::stoll(ReadMap(csvMap, "mask_mode"));
        this->win_left = std::stoll(ReadMap(csvMap, "win_left"));
        this->win_right = std::stoll(ReadMap(csvMap, "win_right"));
        this->max_seqlen_q = std::stoll(ReadMap(csvMap, "max_seqlen_q"));
        this->max_seqlen_kv = std::stoll(ReadMap(csvMap, "max_seqlen_kv"));
        this->layout_q = ReadMap(csvMap, "layout_q");
        this->layout_q_descale = ReadMap(csvMap, "layout_q_descale");
        this->layout_kv = ReadMap(csvMap, "layout_kv");
        this->layout_out = ReadMap(csvMap, "layout_out");
        this->return_softmax_lse = std::stoi(ReadMap(csvMap, "return_softmax_lse"));
    }
};

struct QuantFlashAttnTilingUtParam : public QuantFlashAttnHostUtParamBase {
    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription v = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription q_descale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k_descale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription v_descale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription p_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_kv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_kv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sinks = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription attn_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;

    std::shared_ptr<std::vector<int32_t>> cu_seqlens_q_data;
    std::shared_ptr<std::vector<int32_t>> cu_seqlens_kv_data;
    std::shared_ptr<std::vector<int32_t>> seqused_q_data;
    std::shared_ptr<std::vector<int32_t>> seqused_kv_data;

    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    // QFA const data inputs use int32 (DT_INT32), unlike FIA which uses int64.
    static void ApplyConstData(const csv_map &csvMap, const std::string &name,
                               gert::TilingContextPara::TensorDescription &desc,
                               std::shared_ptr<std::vector<int32_t>> &buf)
    {
        std::string dataStr = ReadMap(csvMap, name + "_data");
        if (!dataStr.empty()) {
            buf = std::make_shared<std::vector<int32_t>>();
            std::istringstream iss(dataStr);
            int32_t val;
            while (iss >> val) {
                buf->push_back(val);
            }
            desc.isConst_ = true;
            desc.constValue_ = buf->data();
        }
    }

    QuantFlashAttnTilingUtParam(const csv_map &csvMap)
        : QuantFlashAttnHostUtParamBase(csvMap)
    {
        // 输入顺序与 quant_flash_attn_def.cpp 中 Input() 注册顺序保持一致
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "k_shape", "k_dtype", "k_format", this->k));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "v_shape", "v_dtype", "v_format", this->v));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_descale_shape", "q_descale_dtype", "q_descale_format", this->q_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_descale_shape", "k_descale_dtype", "k_descale_format", this->k_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "v_descale_shape", "v_descale_dtype", "v_descale_format", this->v_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "p_scale_shape", "p_scale_dtype", "p_scale_format", this->p_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", this->cu_seqlens_q));
        ApplyConstData(csvMap, "cu_seqlens_q", this->cu_seqlens_q, this->cu_seqlens_q_data);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "cu_seqlens_kv_shape", "cu_seqlens_kv_dtype",
                                                     "cu_seqlens_kv_format", this->cu_seqlens_kv));
        ApplyConstData(csvMap, "cu_seqlens_kv", this->cu_seqlens_kv, this->cu_seqlens_kv_data);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        ApplyConstData(csvMap, "seqused_q", this->seqused_q, this->seqused_q_data);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_kv_shape", "seqused_kv_dtype", "seqused_kv_format", this->seqused_kv));
        ApplyConstData(csvMap, "seqused_kv", this->seqused_kv, this->seqused_kv_data);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "sinks_shape", "sinks_dtype", "sinks_format", this->sinks));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        // 输出
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            const std::string &tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
            // 留空时使用 SKIP_TILING_KEY_VALIDATION(UINT64_MAX) 跳过 tiling key 校验
            this->expectTilingKey = tilingKeyStr.empty() ? UINT64_MAX : stoull(tilingKeyStr);
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct QuantFlashAttnInferShapeUtParam : public QuantFlashAttnHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription v = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription q_descale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k_descale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription v_descale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription block_table = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription p_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cu_seqlens_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cu_seqlens_kv = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription seqused_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription seqused_kv = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sinks = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attn_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription metadata = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription attn_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    QuantFlashAttnInferShapeUtParam(const csv_map &csvMap)
        : QuantFlashAttnHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "k_shape", "k_dtype", "k_format", this->k));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "v_shape", "v_dtype", "v_format", this->v));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_descale_shape", "q_descale_dtype", "q_descale_format", this->q_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_descale_shape", "k_descale_dtype", "k_descale_format", this->k_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "v_descale_shape", "v_descale_dtype", "v_descale_format", this->v_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "p_scale_shape", "p_scale_dtype", "p_scale_format", this->p_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", this->cu_seqlens_q));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "cu_seqlens_kv_shape", "cu_seqlens_kv_dtype",
                                                     "cu_seqlens_kv_format", this->cu_seqlens_kv));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_kv_shape", "seqused_kv_dtype", "seqused_kv_format", this->seqused_kv));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "sinks_shape", "sinks_dtype", "sinks_format", this->sinks));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "attn_out_shape")),
                                       GetShapeArr(ReadMap(csvMap, "softmax_lse_shape"))};
        }
    }
};

struct QuantFlashAttnInferDTypeUtParam : public QuantFlashAttnHostUtParamBase {
    ge::DataType q = ge::DT_UNDEFINED;
    ge::DataType k = ge::DT_UNDEFINED;
    ge::DataType v = ge::DT_UNDEFINED;
    ge::DataType q_descale = ge::DT_UNDEFINED;
    ge::DataType k_descale = ge::DT_UNDEFINED;
    ge::DataType v_descale = ge::DT_UNDEFINED;
    ge::DataType block_table = ge::DT_UNDEFINED;
    ge::DataType p_scale = ge::DT_UNDEFINED;
    ge::DataType cu_seqlens_q = ge::DT_UNDEFINED;
    ge::DataType cu_seqlens_kv = ge::DT_UNDEFINED;
    ge::DataType seqused_q = ge::DT_UNDEFINED;
    ge::DataType seqused_kv = ge::DT_UNDEFINED;
    ge::DataType sinks = ge::DT_UNDEFINED;
    ge::DataType attn_mask = ge::DT_UNDEFINED;
    ge::DataType metadata = ge::DT_UNDEFINED;

    ge::DataType attn_out = ge::DT_UNDEFINED;
    ge::DataType softmax_lse = ge::DT_UNDEFINED;

    QuantFlashAttnInferDTypeUtParam(const csv_map &csvMap)
        : QuantFlashAttnHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_dtype", this->q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "k_dtype", this->k));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "v_dtype", this->v));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_descale_dtype", this->q_descale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "k_descale_dtype", this->k_descale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "v_descale_dtype", this->v_descale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "block_table_dtype", this->block_table));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "p_scale_dtype", this->p_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "cu_seqlens_q_dtype", this->cu_seqlens_q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "cu_seqlens_kv_dtype", this->cu_seqlens_kv));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "seqused_q_dtype", this->seqused_q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "seqused_kv_dtype", this->seqused_kv));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "sinks_dtype", this->sinks));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "attn_mask_dtype", this->attn_mask));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "metadata_dtype", this->metadata));

        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "attn_out_dtype", this->attn_out));
        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "softmax_lse_dtype", this->softmax_lse));
    }
};

} // namespace QuantFlashAttnUT

#endif // QUANT_FLASH_ATTN_PARAM_H
