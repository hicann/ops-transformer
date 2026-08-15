/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_HOST_UT_PARAM_H
#define QUANT_BLOCK_SPARSE_ATTN_HOST_UT_PARAM_H

#include <sstream>
#include "op_host_csv_case_loader.h"

namespace QuantBlockSparseAttnUT {

struct QuantBlockSparseAttnHostUtParamBase {
    std::string case_name;
    float softmax_scale = 1.0f;
    int64_t sparse_q_block_size = 128;
    int64_t sparse_kv_block_size = 128;
    std::string layout_kv = "PA_BNBD";
    std::string layout_q = "TND";
    std::string layout_sparse_indices = "B_N_Qb_Kb";
    std::string layout_out = "TND";
    int64_t quant_mode = 1;
    int64_t mask_mode = 3;
    bool return_softmax_lse = false;
    ge::graphStatus expectResult = ge::GRAPH_FAILED;

    explicit QuantBlockSparseAttnHostUtParamBase(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "caseName");
        this->softmax_scale = std::stof(ReadMap(csvMap, "softmax_scale", "1.0"));
        this->sparse_q_block_size = std::stoll(ReadMap(csvMap, "sparse_q_block_size", "128"));
        this->sparse_kv_block_size = std::stoll(ReadMap(csvMap, "sparse_kv_block_size", "128"));
        this->layout_kv = ReadMap(csvMap, "layout_kv", "PA_BNBD");
        this->layout_q = ReadMap(csvMap, "layout_q", "TND");
        this->layout_sparse_indices = ReadMap(csvMap, "layout_sparse_indices", "B_N_Qb_Kb");
        this->layout_out = ReadMap(csvMap, "layout_out", "TND");
        this->quant_mode = std::stoll(ReadMap(csvMap, "quant_mode", "1"));
        this->mask_mode = std::stoll(ReadMap(csvMap, "mask_mode", "3"));
        this->return_softmax_lse = StrToBoolIgnoreCase(ReadMap(csvMap, "return_softmax_lse", "false"));
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectStatus"));
    }
};

inline std::ostream &operator<<(std::ostream &os, const QuantBlockSparseAttnHostUtParamBase &param)
{
    return os << param.case_name;
}

struct QuantBlockSparseAttnTilingUtParam : public QuantBlockSparseAttnHostUtParamBase {
    gert::TilingContextPara::TensorDescription query = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription qDescale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kDescale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription vDescale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription pScale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cuSeqlensQ = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cuSeqlensKv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sequsedQ = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sequsedKv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sparseIndices = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sparseSeqLen = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription blockTable = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attenMask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attentionOut = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmaxLse = TD_DEFAULT;
    bool provideFp8Strides = true;
    bool provideFp8Metadata = true;
    bool mismatchValueStride = false;

    explicit QuantBlockSparseAttnTilingUtParam(const csv_map &csvMap)
        : QuantBlockSparseAttnHostUtParamBase(csvMap)
    {
        GetTensorGE(csvMap, "queryShape", "queryDtype", "queryFormat", this->query);
        GetTensorGE(csvMap, "keyShape", "keyDtype", "keyFormat", this->key);
        GetTensorGE(csvMap, "valueShape", "valueDtype", "valueFormat", this->value);
        GetTensorGE(csvMap, "qDescaleShape", "qDescaleDtype", "qDescaleFormat", this->qDescale);
        GetTensorGE(csvMap, "kDescaleShape", "kDescaleDtype", "kDescaleFormat", this->kDescale);
        GetTensorGE(csvMap, "vDescaleShape", "vDescaleDtype", "vDescaleFormat", this->vDescale);
        GetTensorGE(csvMap, "pScaleShape", "pScaleDtype", "pScaleFormat", this->pScale);
        GetTensorGE(csvMap, "cuSeqlensQShape", "cuSeqlensQDtype", "cuSeqlensQFormat", this->cuSeqlensQ);
        GetTensorGE(csvMap, "cuSeqlensKvShape", "cuSeqlensKvDtype", "cuSeqlensKvFormat", this->cuSeqlensKv);
        GetTensorGE(csvMap, "sequsedQShape", "sequsedQDtype", "sequsedQFormat", this->sequsedQ);
        GetTensorGE(csvMap, "sequsedKvShape", "sequsedKvDtype", "sequsedKvFormat", this->sequsedKv);
        GetTensorGE(csvMap, "sparseIndicesShape", "sparseIndicesDtype", "sparseIndicesFormat", this->sparseIndices);
        GetTensorGE(csvMap, "sparseSeqLenShape", "sparseSeqLenDtype", "sparseSeqLenFormat", this->sparseSeqLen);
        GetTensorGE(csvMap, "blockTableShape", "blockTableDtype", "blockTableFormat", this->blockTable);
        GetTensorGE(csvMap, "attenMaskShape", "attenMaskDtype", "attenMaskFormat", this->attenMask);
        GetTensorGE(csvMap, "metadataShape", "metadataDtype", "metadataFormat", this->metadata);
        GetTensorGE(csvMap, "attentionOutShape", "attentionOutDtype", "attentionOutFormat", this->attentionOut);
        GetTensorGE(csvMap, "softmaxLseShape", "softmaxLseDtype", "softmaxLseFormat", this->softmaxLse);
        this->provideFp8Strides = StrToBoolIgnoreCase(ReadMap(csvMap, "provideFp8Strides", "true"));
        this->provideFp8Metadata = StrToBoolIgnoreCase(ReadMap(csvMap, "provideFp8Metadata", "true"));
        this->mismatchValueStride = StrToBoolIgnoreCase(ReadMap(csvMap, "mismatchValueStride", "false"));
    }
};

struct QuantBlockSparseAttnInferShapeUtParam : public QuantBlockSparseAttnHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription query = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription qDescale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kDescale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription vDescale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pScale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cuSeqlensQ = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cuSeqlensKv = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sequsedQ = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sequsedKv = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sparseIndices = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sparseSeqLen = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription blockTable = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attenMask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription metadata = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attentionOut = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmaxLse = ID_DEFAULT;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit QuantBlockSparseAttnInferShapeUtParam(const csv_map &csvMap)
        : QuantBlockSparseAttnHostUtParamBase(csvMap)
    {
        GetTensorGE(csvMap, "queryShape", "queryDtype", "queryFormat", this->query);
        GetTensorGE(csvMap, "keyShape", "keyDtype", "keyFormat", this->key);
        GetTensorGE(csvMap, "valueShape", "valueDtype", "valueFormat", this->value);
        GetTensorGE(csvMap, "qDescaleShape", "qDescaleDtype", "qDescaleFormat", this->qDescale);
        GetTensorGE(csvMap, "kDescaleShape", "kDescaleDtype", "kDescaleFormat", this->kDescale);
        GetTensorGE(csvMap, "vDescaleShape", "vDescaleDtype", "vDescaleFormat", this->vDescale);
        GetTensorGE(csvMap, "pScaleShape", "pScaleDtype", "pScaleFormat", this->pScale);
        GetTensorGE(csvMap, "cuSeqlensQShape", "cuSeqlensQDtype", "cuSeqlensQFormat", this->cuSeqlensQ);
        GetTensorGE(csvMap, "cuSeqlensKvShape", "cuSeqlensKvDtype", "cuSeqlensKvFormat", this->cuSeqlensKv);
        GetTensorGE(csvMap, "sequsedQShape", "sequsedQDtype", "sequsedQFormat", this->sequsedQ);
        GetTensorGE(csvMap, "sequsedKvShape", "sequsedKvDtype", "sequsedKvFormat", this->sequsedKv);
        GetTensorGE(csvMap, "sparseIndicesShape", "sparseIndicesDtype", "sparseIndicesFormat", this->sparseIndices);
        GetTensorGE(csvMap, "sparseSeqLenShape", "sparseSeqLenDtype", "sparseSeqLenFormat", this->sparseSeqLen);
        GetTensorGE(csvMap, "blockTableShape", "blockTableDtype", "blockTableFormat", this->blockTable);
        GetTensorGE(csvMap, "attenMaskShape", "attenMaskDtype", "attenMaskFormat", this->attenMask);
        GetTensorGE(csvMap, "metadataShape", "metadataDtype", "metadataFormat", this->metadata);
        GetTensorGE(csvMap, "attentionOutShape", "attentionOutDtype", "attentionOutFormat", this->attentionOut);
        GetTensorGE(csvMap, "softmaxLseShape", "softmaxLseDtype", "softmaxLseFormat", this->softmaxLse);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string outputShapeStr = ReadMap(csvMap, "expectOutputShape");
            std::istringstream iss(outputShapeStr);
            std::string token;
            while (std::getline(iss, token, '|')) {
                this->expectOutputShape.emplace_back(GetShapeArr(token));
            }
        }
    }
};

struct QuantBlockSparseAttnInferDTypeUtParam : public QuantBlockSparseAttnHostUtParamBase {
    ge::DataType query = ge::DT_UNDEFINED;
    ge::DataType key = ge::DT_UNDEFINED;
    ge::DataType value = ge::DT_UNDEFINED;
    ge::DataType qDescale = ge::DT_UNDEFINED;
    ge::DataType kDescale = ge::DT_UNDEFINED;
    ge::DataType vDescale = ge::DT_UNDEFINED;
    ge::DataType pScale = ge::DT_UNDEFINED;
    ge::DataType cuSeqlensQ = ge::DT_UNDEFINED;
    ge::DataType cuSeqlensKv = ge::DT_UNDEFINED;
    ge::DataType sequsedQ = ge::DT_UNDEFINED;
    ge::DataType sequsedKv = ge::DT_UNDEFINED;
    ge::DataType sparseIndices = ge::DT_UNDEFINED;
    ge::DataType sparseSeqLen = ge::DT_UNDEFINED;
    ge::DataType blockTable = ge::DT_UNDEFINED;
    ge::DataType attenMask = ge::DT_UNDEFINED;
    ge::DataType metadata = ge::DT_UNDEFINED;
    ge::DataType expectAttentionOutDtype = ge::DT_UNDEFINED;
    ge::DataType expectSoftmaxLseDtype = ge::DT_UNDEFINED;

    explicit QuantBlockSparseAttnInferDTypeUtParam(const csv_map &csvMap)
        : QuantBlockSparseAttnHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "queryDtype", this->query);
        GetDataTypeGE(csvMap, "keyDtype", this->key);
        GetDataTypeGE(csvMap, "valueDtype", this->value);
        GetDataTypeGE(csvMap, "qDescaleDtype", this->qDescale);
        GetDataTypeGE(csvMap, "kDescaleDtype", this->kDescale);
        GetDataTypeGE(csvMap, "vDescaleDtype", this->vDescale);
        GetDataTypeGE(csvMap, "pScaleDtype", this->pScale);
        GetDataTypeGE(csvMap, "cuSeqlensQDtype", this->cuSeqlensQ);
        GetDataTypeGE(csvMap, "cuSeqlensKvDtype", this->cuSeqlensKv);
        GetDataTypeGE(csvMap, "sequsedQDtype", this->sequsedQ);
        GetDataTypeGE(csvMap, "sequsedKvDtype", this->sequsedKv);
        GetDataTypeGE(csvMap, "sparseIndicesDtype", this->sparseIndices);
        GetDataTypeGE(csvMap, "sparseSeqLenDtype", this->sparseSeqLen);
        GetDataTypeGE(csvMap, "blockTableDtype", this->blockTable);
        GetDataTypeGE(csvMap, "attenMaskDtype", this->attenMask);
        GetDataTypeGE(csvMap, "metadataDtype", this->metadata);
        this->expectAttentionOutDtype = Str2DTypeGE(ReadMap(csvMap, "expectAttentionOutDtype"));
        this->expectSoftmaxLseDtype = Str2DTypeGE(ReadMap(csvMap, "expectSoftmaxLseDtype"));
    }
};

} // namespace QuantBlockSparseAttnUT

#endif // QUANT_BLOCK_SPARSE_ATTN_HOST_UT_PARAM_H
