/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_API_UT_PARAM_H
#define QUANT_BLOCK_SPARSE_ATTN_API_UT_PARAM_H

#include <sstream>
#include "op_api_csv_case_loader.h"

namespace QuantBlockSparseAttnUT {

struct QuantBlockSparseAttnApiUtParam {
    std::string case_name;
    TensorDesc query;
    TensorDesc key;
    TensorDesc value;
    TensorDesc qDescale;
    TensorDesc kDescale;
    TensorDesc vDescale;
    TensorDesc pScale;
    TensorDesc cuSeqlensQ;
    TensorDesc cuSeqlensKv;
    TensorDesc sequsedQ;
    TensorDesc sequsedKv;
    TensorDesc sparseIndices;
    TensorDesc sparseSeqLen;
    TensorDesc blockTable;
    TensorDesc attenMask;
    TensorDesc metadata;
    TensorDesc attentionOut;
    TensorDesc softmaxLse;
    int64_t maxSeqlenQ;
    int64_t maxSeqlenKv;
    double softmaxScale;
    int64_t sparseQBlockSize;
    int64_t sparseKvBlockSize;
    int64_t paBlockStride;
    std::string layoutKv;
    std::string layoutQ;
    std::string layoutSparseIndices;
    std::string layoutOut;
    int64_t quantMode;
    int64_t maskMode;
    bool returnSoftmaxLse;
    op::SocVersion soc;
    aclnnStatus expectResult;

    explicit QuantBlockSparseAttnApiUtParam(const csv_map &csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->query = GetTensorACL(csvMap, "query_shape", "query_dtype", "query_format");
        this->key = GetTensorACL(csvMap, "key_shape", "key_dtype", "key_format");
        this->value = GetTensorACL(csvMap, "value_shape", "value_dtype", "value_format");
        this->qDescale = GetTensorACL(csvMap, "q_descale_shape", "q_descale_dtype", "q_descale_format");
        this->kDescale = GetTensorACL(csvMap, "k_descale_shape", "k_descale_dtype", "k_descale_format");
        this->vDescale = GetTensorACL(csvMap, "v_descale_shape", "v_descale_dtype", "v_descale_format");
        this->pScale = GetTensorACL(csvMap, "p_scale_shape", "p_scale_dtype", "p_scale_format");
        this->cuSeqlensQ = GetTensorACL(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format");
        this->cuSeqlensKv = GetTensorACL(csvMap, "cu_seqlens_kv_shape", "cu_seqlens_kv_dtype", "cu_seqlens_kv_format");
        this->sequsedQ = GetTensorACL(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format");
        this->sequsedKv = GetTensorACL(csvMap, "seqused_kv_shape", "seqused_kv_dtype", "seqused_kv_format");
        this->sparseIndices =
            GetTensorACL(csvMap, "sparse_indices_shape", "sparse_indices_dtype", "sparse_indices_format");
        this->sparseSeqLen =
            GetTensorACL(csvMap, "sparse_seq_len_shape", "sparse_seq_len_dtype", "sparse_seq_len_format");
        this->blockTable = GetTensorACL(csvMap, "block_table_shape", "block_table_dtype", "block_table_format");
        this->attenMask = GetTensorACL(csvMap, "atten_mask_shape", "atten_mask_dtype", "atten_mask_format");
        this->metadata = GetTensorACL(csvMap, "metadata_shape", "metadata_dtype", "metadata_format");
        this->attentionOut = GetTensorACL(csvMap, "attention_out_shape", "attention_out_dtype", "attention_out_format");
        this->softmaxLse = GetTensorACL(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format");
        this->maxSeqlenQ = stoll(ReadMap(csvMap, "max_seqlen_q", "0"));
        this->maxSeqlenKv = stoll(ReadMap(csvMap, "max_seqlen_kv", "0"));
        this->softmaxScale = stod(ReadMap(csvMap, "softmax_scale", "1.0"));
        this->sparseQBlockSize = stoll(ReadMap(csvMap, "sparse_q_block_size", "128"));
        this->sparseKvBlockSize = stoll(ReadMap(csvMap, "sparse_kv_block_size", "128"));
        this->paBlockStride = stoll(ReadMap(csvMap, "pa_block_stride", "0"));
        this->layoutKv = ReadMap(csvMap, "layout_kv", "PA_BNSD");
        this->layoutQ = ReadMap(csvMap, "layout_q", "TND");
        this->layoutSparseIndices = ReadMap(csvMap, "layout_sparse_indices", "B_N_Qb_Kb");
        this->layoutOut = ReadMap(csvMap, "layout_out", "TND");
        this->quantMode = stoll(ReadMap(csvMap, "quant_mode", "1"));
        this->maskMode = stoll(ReadMap(csvMap, "mask_mode", "3"));
        this->returnSoftmaxLse = ReadMap(csvMap, "return_softmax_lse", "0") == "1";
        this->soc = GetCaseSocVersion(csvMap, "soc");
        this->expectResult = GetAclnnRet(csvMap, "expect_result");
    }
};

inline std::ostream &operator<<(std::ostream &os, const QuantBlockSparseAttnApiUtParam &param)
{
    return os << param.case_name;
}

} // namespace QuantBlockSparseAttnUT

#endif // QUANT_BLOCK_SPARSE_ATTN_API_UT_PARAM_H
