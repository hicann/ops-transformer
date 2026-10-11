/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "../../common/inc/kernel/mc2_profiling.h"
#include <torch/library.h>
#include <ATen/Operators.h>
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "op_host/incre_flash_attention_tiling_impl.h"
#include "op_kernel/incre_flash_attention_arch22.h"
namespace ascend_ops {
namespace custom {
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(R"(npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *,
                                            Tensor? query_rope=None, Tensor? key_rope=None,
                                            Tensor? pse_shift=None, Tensor? atten_mask=None,
                                            Tensor? actual_seq_qlen=None, Tensor? actual_seq_kvlen=None,
                                            Tensor? block_table=None, Tensor? dequant_scale_query=None,
                                            Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None,
                                            Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None,
                                            Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None,
                                            Tensor? quant_offset_out=None, Tensor? learnable_sink=None,
                                            Tensor? metadata=None,
                                            int num_query_heads=1, int num_key_value_heads=0,
                                            float softmax_scale=1.0, int pre_tokens=2147483647,
                                            int next_tokens=2147483647, str input_layout='BSH',
                                            int sparse_mode=0, int block_size=0,
                                            int query_quant_mode=0, int key_quant_mode=0,
                                            int value_quant_mode=0, int inner_precise=1,
                                            bool return_softmax_lse=False, int? query_dtype=None,
                                            int? key_dtype=None, int? value_dtype=None,
                                            int? query_rope_dtype=None, int? key_rope_dtype=None,
                                            int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None,
                                            int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None,
                                            int? dequant_scale_value_dtype=None,
                                            int? dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor))");

    m.def(R"(npu_fused_infer_attention_score_metadata(int batch_size,
                                                    int query_seq_size,
                                                    int query_head_num,
                                                    int key_head_num,
                                                    int head_dim,
                                                    int block_size,
                                                    int max_block_num_per_batch,
                                                    Tensor actual_seq_lengths_kv=None,
                                                    *,
                                                    str layout_query='BSND') -> Tensor)");
}

#define LAUNCH_INCRE_FA(FD, LAYOUT, ANTIQ) \
    incre_flash_attention<FD, LAYOUT, ANTIQ><<<blockDim, nullptr, aclstream>>>( \
        (GM_ADDR)(query.data_ptr()), (GM_ADDR)(key.data_ptr()), (GM_ADDR)(value.data_ptr()), \
        (GM_ADDR)(pse_shift.has_value() ? pse_shift->data_ptr() : nullptr), \
        (GM_ADDR)(atten_mask.has_value() ? atten_mask->data_ptr() : nullptr), \
        (GM_ADDR)(actual_seq_qlen.has_value() ? actual_seq_qlen->data_ptr() : nullptr), \
        (GM_ADDR)(actual_seq_kvlen.has_value() ? actual_seq_kvlen->data_ptr() : nullptr), \
        (GM_ADDR)(nullptr), /* deqScale1 */ \
        (GM_ADDR)(nullptr), /* quantScale1 */ \
        (GM_ADDR)(nullptr), /* deqScale2 */ \
        (GM_ADDR)(nullptr), /* quantScale2 */ \
        (GM_ADDR)(nullptr), /* quantOffset2 */ \
        (GM_ADDR)(nullptr), /* antiquantScale */ \
        (GM_ADDR)(nullptr), /* antiquantOffset */ \
        (GM_ADDR)(block_table.has_value() ? block_table->data_ptr() : nullptr), \
        (GM_ADDR)(nullptr), /* queryPaddingSize */ \
        (GM_ADDR)(nullptr), /* kvPaddingSize */ \
        (GM_ADDR)(dequant_scale_key.has_value() ? dequant_scale_key->data_ptr() : nullptr), \
        (GM_ADDR)(nullptr), /* keyAntiquantOffset */ \
        (GM_ADDR)(dequant_scale_value.has_value() ? dequant_scale_value->data_ptr() : nullptr), \
        (GM_ADDR)(nullptr), /* valueAntiquantOffset */ \
        (GM_ADDR)(nullptr), /* keySharedPrefix */ \
        (GM_ADDR)(nullptr), /* valueSharedPrefix */ \
        (GM_ADDR)(nullptr), /* actualSharedPrefixLen */ \
        (GM_ADDR)(query_rope.has_value() ? query_rope->data_ptr() : nullptr), \
        (GM_ADDR)(key_rope.has_value() ? key_rope->data_ptr() : nullptr), \
        (GM_ADDR)(dequant_scale_key_rope.has_value() ? dequant_scale_key_rope->data_ptr() : nullptr), \
        (GM_ADDR)(dequant_scale_query.has_value() ? dequant_scale_query->data_ptr() : nullptr), \
        (GM_ADDR)(metadata.has_value() ? metadata->data_ptr() : nullptr), (GM_ADDR)(output.data_ptr()), \
        (GM_ADDR)(softmax_lse.data_ptr()), (GM_ADDR)(workspaceTensor.data_ptr()), tilingData)

const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
constexpr uint32_t MIX_AIC = 4;

using namespace at_npu::native;
using namespace optiling;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor& query, const at::Tensor& value, std::string input_layout_str,
    const c10::optional<at::Tensor>& quant_scale_out, const c10::optional<at::Tensor>& block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
    bool return_softmax_lse, const c10::optional<at::Tensor>& query_rope)
{
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    at::Tensor tmp_output = at::empty_like(query);
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = at::empty({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = at::empty({query.size(DIM_1), query.size(DIM_0), query.size(DIM_2), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = at::empty({query.size(DIM_2), query.size(DIM_0), query.size(DIM_1), query.size(DIM_3)},
                               query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output =
            at::empty({query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)}, query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    }

    if (quant_scale_out.has_value()) {
        output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor& query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = at::empty(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = at::empty(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = at::empty_like(tmp_output);
    }

    auto lse_opts = output.options().dtype(c10::ScalarType::Float);
    at::Tensor softmax_lse;
    softmax_lse = at::empty({batchSize, num_query_heads, qsSize, 1}, lse_opts);

    if (!return_softmax_lse) {
        softmax_lse = at::empty({0}, lse_opts);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

RequiredParaInfo ToRequiredParaInfo(const at::Tensor& t)
{
    return RequiredParaInfo{.data = t.defined() ? t.data_ptr() : nullptr, .shape = t.sizes(), .dType = t.scalar_type()};
}

OptionalTensorParaInfo ToOptionalTensorParaInfo(const c10::optional<at::Tensor>& ot)
{
    OptionalTensorParaInfo info;
    info.hasValue = ot.has_value();
    if (info.hasValue && ot->defined()) {
        info.data = ot->data_ptr();
        info.shape = ot->sizes();
        info.dType = ot->scalar_type();
    } else {
        info.data = nullptr;
    }
    return info;
}

void ConvertContextToParamsIFA(
    IFAContext& ifaContext, const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& query_rope, const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& pse_shift, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& actual_seq_qlen, const c10::optional<at::Tensor>& actual_seq_kvlen,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& dequant_scale_key, const c10::optional<at::Tensor>& dequant_offset_key,
    const c10::optional<at::Tensor>& dequant_scale_value, const c10::optional<at::Tensor>& dequant_offset_value,
    const c10::optional<at::Tensor>& dequant_scale_key_rope, const c10::optional<at::Tensor>& quant_scale_out,
    const c10::optional<at::Tensor>& quant_offset_out, const c10::optional<at::Tensor>& learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    const char* input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode,
    int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const c10::optional<int64_t>& query_dtype,
    const c10::optional<int64_t>& key_dtype, const c10::optional<int64_t>& value_dtype,
    const c10::optional<int64_t>& query_rope_dtype, const c10::optional<int64_t>& key_rope_dtype,
    const c10::optional<int64_t>& key_shared_prefix_dtype, const c10::optional<int64_t>& value_shared_prefix_dtype,
    const c10::optional<int64_t>& dequant_scale_query_dtype, const c10::optional<int64_t>& dequant_scale_key_dtype,
    const c10::optional<int64_t>& dequant_scale_value_dtype, const c10::optional<int64_t>& dequant_scale_key_rope_dtype,
    const at::Tensor& attenOut, const at::Tensor& lseOut)
{
    // required input
    ifaContext.opName = "FusedInferAttentionScore";
    ifaContext.query = ToRequiredParaInfo(query);
    ifaContext.key = ToRequiredParaInfo(key);
    ifaContext.value = ToRequiredParaInfo(value);
    // optional input
    ifaContext.pseShift = ToOptionalTensorParaInfo(pse_shift);
    ifaContext.attenMask = ToOptionalTensorParaInfo(atten_mask);
    ifaContext.actualSeqLengthsQ = ToOptionalTensorParaInfo(actual_seq_qlen);
    ifaContext.actualSeqLengths = ToOptionalTensorParaInfo(actual_seq_kvlen);
    ifaContext.deqScale1 = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.quantScale1 = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.deqScale2 = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.quantScale2 = ToOptionalTensorParaInfo(quant_scale_out);
    ifaContext.quantOffset2 = ToOptionalTensorParaInfo(quant_offset_out);
    ifaContext.antiquantScale = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.antiquantOffset = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.blockTable = ToOptionalTensorParaInfo(block_table);
    ifaContext.queryPaddingSize = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.kvPaddingSize = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.keyAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_key);
    ifaContext.keyAntiquantOffset = ToOptionalTensorParaInfo(dequant_offset_key);
    ifaContext.valueAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_value);
    ifaContext.valueAntiquantOffset = ToOptionalTensorParaInfo(dequant_offset_value);
    ifaContext.keySharedPrefix = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.valueSharedPrefix = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.actualSharedPrefixLen = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.queryRope = ToOptionalTensorParaInfo(query_rope);
    ifaContext.keyRope = ToOptionalTensorParaInfo(key_rope);
    ifaContext.keyRopeAntiquantScale = ToOptionalTensorParaInfo(dequant_scale_key_rope);
    ifaContext.dequantScaleQuery = ToOptionalTensorParaInfo(dequant_scale_query);
    ifaContext.qStartIdx = ToOptionalTensorParaInfo(c10::nullopt);
    ifaContext.kvStartIdx = ToOptionalTensorParaInfo(c10::nullopt);
    // attr
    ifaContext.numHeads = num_query_heads;
    ifaContext.preToken = pre_tokens;
    ifaContext.nextToken = next_tokens;
    ifaContext.scaleValue = softmax_scale;
    ifaContext.kvHeadNums = num_key_value_heads;
    ifaContext.layOut = input_layout;
    ifaContext.blockSize = block_size;
    ifaContext.innerPrecise = inner_precise;
    ifaContext.antiquantMode = 0;
    ifaContext.softmaxLseFlag = return_softmax_lse;
    ifaContext.keyAntiquantMode = key_quant_mode;
    ifaContext.valueAntiquantMode = value_quant_mode;
    ifaContext.sparseMode = sparse_mode;
    ifaContext.queryQuantMode = query_quant_mode;
    ifaContext.pseType = 0;
    ifaContext.windowSize = 0;
    // output
    ifaContext.attenOut = ToRequiredParaInfo(attenOut);
    ifaContext.lseOut = ToRequiredParaInfo(lseOut);

    ifaContext.kCache.resize(1);
    ifaContext.vCache.resize(1);
    at::IntArrayRef keyShapeRef = ifaContext.kCache[0] = key.sizes();
    ifaContext.vCache[0] = value.sizes();
}

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_npu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& query_rope, const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& pse_shift, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& actual_seq_qlen, const c10::optional<at::Tensor>& actual_seq_kvlen,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& dequant_scale_key, const c10::optional<at::Tensor>& dequant_offset_key,
    const c10::optional<at::Tensor>& dequant_scale_value, const c10::optional<at::Tensor>& dequant_offset_value,
    const c10::optional<at::Tensor>& dequant_scale_key_rope, const c10::optional<at::Tensor>& quant_scale_out,
    const c10::optional<at::Tensor>& quant_offset_out, const c10::optional<at::Tensor>& learnable_sink,
    const c10::optional<at::Tensor>& metadata, int64_t num_query_heads, int64_t num_key_value_heads,
    double softmax_scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t sparse_mode,
    int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse, c10::optional<int64_t> query_dtype,
    c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype, c10::optional<int64_t> query_rope_dtype,
    c10::optional<int64_t> key_rope_dtype, c10::optional<int64_t> key_shared_prefix_dtype,
    c10::optional<int64_t> value_shared_prefix_dtype, c10::optional<int64_t> dequant_scale_query_dtype,
    c10::optional<int64_t> dequant_scale_key_dtype, c10::optional<int64_t> dequant_scale_value_dtype,
    c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output =
        construct_fia_output_tensor_v2(query, value, input_layout_str, quant_scale_out, block_table, num_query_heads,
                                       num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char input_layout_ptr[20];
    strncpy(input_layout_ptr, input_layout_str.c_str(), 20 - 1);

    IFAContext ifaContext;
    ConvertContextToParamsIFA(
        ifaContext, query, key, value, query_rope, key_rope, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen,
        block_table, dequant_scale_query, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, dequant_scale_key_rope, quant_scale_out, quant_offset_out, learnable_sink,
        num_query_heads, num_key_value_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, sparse_mode,
        block_size, query_quant_mode, key_quant_mode, value_quant_mode, inner_precise, return_softmax_lse, query_dtype,
        key_dtype, value_dtype, query_rope_dtype, key_rope_dtype, key_shared_prefix_dtype, value_shared_prefix_dtype,
        dequant_scale_query_dtype, dequant_scale_key_dtype, dequant_scale_value_dtype, dequant_scale_key_rope_dtype,
        output, softmax_lse);

    IFATiling ifaTiling;
    if (ifaTiling.DoSubOpTiling(ifaContext) == ::custom::graphStatus::GRAPH_FAILED) {
        throw std::runtime_error("Tiling operation failed. Please check the logs for more details. "
                                 "To enable stdout logging, set: [export ASCEND_SLOG_PRINT_TO_STDOUT=1]");
    }
    // stream
    int devidx = query.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream(false);
    // workspace
    auto workspaceTensor = at::empty({static_cast<long>(ifaContext.workSpaceSize)},
                                     at::TensorOptions().dtype(at::kByte).device(query.options().device()));
    // tilingdata
    optiling::IncreFlashAttentionTilingData& tilingData = ifaContext.tilingData.tilingBase;
    uint8_t fdFlag = ifaContext.fdFlag;
    uint8_t layoutVal = ifaContext.layoutVal;
    uint8_t antiquantMode = ifaContext.antiquantMode_;
    // blockdim
    uint32_t blockDim = tilingData.increFlashAttentionSingleCoreParams.get_usedCoreNum();

    aclprofTensorInfo tensorInfo;
    aclprofTensor emptyTensor = {0, 0, 0, 0, {0}};
    // 本FA demo只支持伪量化场景，非支持场景会被tiling拦截，此处只上报支持范围内的所有参数shape
    INIT_ACL_PROF_TENSOR_INFO(
        "fused_infer_attention_score", "FusedInferAttentionScore", blockDim, MIX_AIC, tensorInfo, aclstream,
        INPUT(query), INPUT(key), INPUT(value), INPUT(atten_mask.value_or(at::empty({}))),
        atten_mask.has_value() ? INPUT(atten_mask.value()) : emptyTensor,
        block_table.has_value() ? INPUT(block_table.value()) : emptyTensor,
        actual_seq_kvlen.has_value() ? INPUT(actual_seq_kvlen.value()) : emptyTensor,
        dequant_scale_key.has_value() ? INPUT(dequant_scale_key.value()) : emptyTensor,
        dequant_scale_value.has_value() ? INPUT(dequant_scale_value.value()) : emptyTensor, OUTPUT(output));
    aclprofEventAttributes attrs = {1, sizeof(aclprofEventAttributes::message), 0, &tensorInfo};
    aclprofRangePushEx(&attrs);

    if (fdFlag == 0 && layoutVal == 0 && antiquantMode == 0) {
        LAUNCH_INCRE_FA(0, 0, 0);
    } else if (fdFlag == 0 && layoutVal == 0 && antiquantMode == 1) {
        LAUNCH_INCRE_FA(0, 0, 1);
    } else if (fdFlag == 0 && layoutVal == 1 && antiquantMode == 0) {
        LAUNCH_INCRE_FA(0, 1, 0);
    } else if (fdFlag == 0 && layoutVal == 1 && antiquantMode == 1) {
        LAUNCH_INCRE_FA(0, 1, 1);
    } else if (fdFlag == 1 && layoutVal == 0 && antiquantMode == 0) {
        LAUNCH_INCRE_FA(1, 0, 0);
    } else if (fdFlag == 1 && layoutVal == 0 && antiquantMode == 1) {
        LAUNCH_INCRE_FA(1, 0, 1);
    } else if (fdFlag == 1 && layoutVal == 1 && antiquantMode == 0) {
        LAUNCH_INCRE_FA(1, 1, 0);
    } else if (fdFlag == 1 && layoutVal == 1 && antiquantMode == 1) {
        LAUNCH_INCRE_FA(1, 1, 1);
    } else {
        printf("invalid flags: fd=%d layout=%d antiquantMode=%d\n", fdFlag, layoutVal, antiquantMode);
    }

    aclprofRangePop();
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_meta(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const c10::optional<at::Tensor>& query_rope, const c10::optional<at::Tensor>& key_rope,
    const c10::optional<at::Tensor>& pse_shift, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& actual_seq_qlen, const c10::optional<at::Tensor>& actual_seq_kvlen,
    const c10::optional<at::Tensor>& block_table, const c10::optional<at::Tensor>& dequant_scale_query,
    const c10::optional<at::Tensor>& dequant_scale_key, const c10::optional<at::Tensor>& dequant_offset_key,
    const c10::optional<at::Tensor>& dequant_scale_value, const c10::optional<at::Tensor>& dequant_offset_value,
    const c10::optional<at::Tensor>& dequant_scale_key_rope, const c10::optional<at::Tensor>& quant_scale_out,
    const c10::optional<at::Tensor>& quant_offset_out, const c10::optional<at::Tensor>& learnable_sink,
    const c10::optional<at::Tensor>& metadata, int64_t num_query_heads, int64_t num_key_value_heads,
    double softmax_scale, int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout, int64_t sparse_mode,
    int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse, c10::optional<int64_t> query_dtype,
    c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype, c10::optional<int64_t> query_rope_dtype,
    c10::optional<int64_t> key_rope_dtype, c10::optional<int64_t> key_shared_prefix_dtype,
    c10::optional<int64_t> value_shared_prefix_dtype, c10::optional<int64_t> dequant_scale_query_dtype,
    c10::optional<int64_t> dequant_scale_key_dtype, c10::optional<int64_t> dequant_scale_value_dtype,
    c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    std::string input_layout_str = std::string(input_layout);
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output =
        construct_fia_output_tensor_v2(query, value, input_layout_str, quant_scale_out, block_table, num_query_heads,
                                       num_key_value_heads, return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_npu);
}

// 为META设备注册前向实现
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("npu_fused_infer_attention_score", &custom::npu_fused_infer_attention_score_meta);
}

} // namespace custom
} // namespace ascend_ops
