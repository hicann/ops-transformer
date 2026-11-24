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
 * \file quant_lightning_indexer.cpp
 * \brief
 */

#include <iterator>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using namespace at_npu::native;

inline TensorWrapper MakeWrapper(const at::Tensor& tensor)
{
    return {tensor, ConvertToAclDataType(tensor.scalar_type())};
}

inline aclIntArray* MakeDcpParamsArray(int64_t cpWorld, int64_t cpRank, int64_t cpInterleaveSize)
{
    const int64_t data[] = {cpWorld, cpRank, cpInterleaveSize};
    return ::ConvertType(at::IntArrayRef(data, std::size(data)));
}

inline bool IsMxQuantMode(int64_t quantMode)
{
    return quantMode == 3 || quantMode == 5;
}

inline bool IsE8M0Tensor(const at::Tensor& tensor)
{
    return tensor.scalar_type() == at::kFloat8_e8m0fnu;
}

inline bool IsFp4CompatibleTensor(const at::Tensor& tensor)
{
    return tensor.scalar_type() == at::kFloat4_e2m1fn_x2 || tensor.scalar_type() == at::kByte;
}

// V3 metadata（candidate 与 DCP 共用）固定输出 1024 个 int32
constexpr int64_t QLI_V3_METADATA_SIZE = 1024;
constexpr int64_t E8M0_SCALE_PACK_NUM = 2;
constexpr uint32_t QUANT_MODE_INT8 = 2;
constexpr int64_t MASK_MODE_RIGHT_DOWN_CAUSAL = 3;
constexpr uint32_t QUANT_MODE_HIFLOAT8 = 4;
constexpr uint32_t QUANT_MODE_MXFP4 = 5;
constexpr uint32_t QUANT_MODE_HIFLOAT4 = 6;
constexpr uint32_t QUANT_MODE_INT4 = 7;

// quant_mode 表达量化模式（scale 粒度/形态），DAV_2201 支持 per-token-head：
//   quant_mode=2：per-token-head W8A8，q/k 以 torch.int8 承载；
//   quant_mode=7：per-token-head W4A4 signed int4。
// torch 侧无 int4 dtype，q/k 以 nibble 打包的 torch.uint8 承载（低半字节=偶下标、补码
// -8..7、zero_point=0），物理 shape D 轴按存储（headDim/2）给出；本 extension 按
// quant_mode=7 将 uint8 映射为 ACL_INT4，框架按 2 逻辑元素/字节折算为逻辑 D=headDim。
// 真实 int4 tensor（ACL_INT4 直传、逻辑 D=headDim）仅支持 aclnn C++ API 场景，
// 两种载体在 tiling 层统一为 DT_INT4 + 逻辑 D shape（tiling 内部归一化 7 到 2 下发）。
inline bool IsDav2201Platform()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        return false;
    }
    return std::strstr(socName, "Ascend910B") != nullptr || std::strstr(socName, "Ascend910_93") != nullptr;
}

inline void FixQLIV2AclDtypes(int64_t quantMode, TensorWrapper& queryWrapper, TensorWrapper& keyWrapper,
                              TensorWrapper& queryScaleWrapper, TensorWrapper& keyScaleWrapper)
{
    if (quantMode == QUANT_MODE_HIFLOAT8) {
        TORCH_CHECK(queryWrapper.tensor_.scalar_type() == at::kByte, "When quant_mode is 4, query must be hifp8 type");
        TORCH_CHECK(keyWrapper.tensor_.scalar_type() == at::kByte, "When quant_mode is 4, key must be hifp8 type");
        queryWrapper.dtype = ACL_HIFLOAT8;
        keyWrapper.dtype = ACL_HIFLOAT8;
        return;
    }

    if (quantMode == QUANT_MODE_MXFP4) {
        TORCH_CHECK(IsFp4CompatibleTensor(queryWrapper.tensor_),
                    "When quant_mode is 5, query must be torch.float4_e2m1fn_x2 or packed torch.uint8");
        TORCH_CHECK(IsFp4CompatibleTensor(keyWrapper.tensor_),
                    "When quant_mode is 5, key must be torch.float4_e2m1fn_x2 or packed torch.uint8");
        queryWrapper.dtype = ACL_FLOAT4_E2M1;
        keyWrapper.dtype = ACL_FLOAT4_E2M1;
    }

    if (quantMode == QUANT_MODE_INT8 && IsDav2201Platform()) {
        // A2/A3（DAV_2201）quant_mode=2（per-token-head W8A8）：q/k 必须为 torch.int8。
        TORCH_CHECK(queryWrapper.tensor_.scalar_type() == at::kChar && keyWrapper.tensor_.scalar_type() == at::kChar,
                    "When quant_mode is 2 on Ascend910B/910_93, query and key must be torch.int8 (W8A8)");
        TORCH_CHECK(
            queryScaleWrapper.tensor_.scalar_type() == at::kHalf && keyScaleWrapper.tensor_.scalar_type() == at::kHalf,
            "When quant_mode is 2, query_dequant_scale and key_dequant_scale must be torch.float16");
        return;
    }

    if (quantMode == QUANT_MODE_INT4 && IsDav2201Platform()) {
        // A2/A3（DAV_2201）quant_mode=7（per-token-head W4A4 signed int4）：torch 侧无 int4
        // dtype，q/k 以 nibble 打包的 torch.uint8 承载（低半字节=偶下标，补码 -8..7，
        // zero_point=0）。uint8 物理 shape D 轴按存储（headDim/2）给出，映射 ACL_INT4 后
        // 框架按 2 逻辑元素/字节折算为逻辑 D=headDim。真实 int4 tensor（ACL_INT4 直传）
        // 仅支持 aclnn C++ API 场景，不经过本 torch extension。
        TORCH_CHECK(queryWrapper.tensor_.scalar_type() == at::kByte && keyWrapper.tensor_.scalar_type() == at::kByte,
                    "When quant_mode is 7 on Ascend910B/910_93, query and key must be packed torch.uint8 "
                    "(signed int4, low nibble = even index); real int4 tensor is only supported via aclnn C++ API");
        TORCH_CHECK(
            queryScaleWrapper.tensor_.scalar_type() == at::kHalf && keyScaleWrapper.tensor_.scalar_type() == at::kHalf,
            "When quant_mode is 7, query_dequant_scale and key_dequant_scale must be torch.float16");
        // ACL_INT4 为 acl 全局枚举（acl_base_rt.h，=29），非 cann_ops_transformer 扩展枚举
        queryWrapper.dtype = ACL_INT4;
        keyWrapper.dtype = ACL_INT4;
        return;
    }

    if (quantMode == QUANT_MODE_HIFLOAT4) {
        // quant_mode=6 HIFLOAT4（仅 DAV_9201/9202/Ascend950），scale 为按 D 分组的 float32。
        TORCH_CHECK(queryWrapper.tensor_.scalar_type() == at::kByte,
                    "When quant_mode is 6, query must be packed torch.uint8");
        TORCH_CHECK(keyWrapper.tensor_.scalar_type() == at::kByte,
                    "When quant_mode is 6, key must be packed torch.uint8");
        TORCH_CHECK(queryScaleWrapper.tensor_.scalar_type() == at::kFloat &&
                        keyScaleWrapper.tensor_.scalar_type() == at::kFloat,
                    "When quant_mode is 6, query_dequant_scale and key_dequant_scale must be torch.float32");
        queryWrapper.dtype = cann_ops_transformer::ACL_HIFLOAT4;
        keyWrapper.dtype = cann_ops_transformer::ACL_HIFLOAT4;
    }

    if (IsMxQuantMode(quantMode)) {
        TORCH_CHECK(IsE8M0Tensor(queryScaleWrapper.tensor_),
                    "When quant_mode is 3 or 5, query_dequant_scale must be torch.float8_e8m0fnu");
        TORCH_CHECK(IsE8M0Tensor(keyScaleWrapper.tensor_),
                    "When quant_mode is 3 or 5, key_dequant_scale must be torch.float8_e8m0fnu");
        // Cube loads E8M0 scales in two-byte groups through a bfloat16_t view.
        TORCH_CHECK(queryScaleWrapper.tensor_.storage_offset() % E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 3 or 5, query_dequant_scale storage offset must satisfy 2-element E8M0 packing "
                    "alignment, but got ",
                    queryScaleWrapper.tensor_.storage_offset());
        TORCH_CHECK(keyScaleWrapper.tensor_.storage_offset() % E8M0_SCALE_PACK_NUM == 0,
                    "When quant_mode is 3 or 5, key_dequant_scale storage offset must satisfy 2-element E8M0 packing "
                    "alignment, but got ",
                    keyScaleWrapper.tensor_.storage_offset());
        queryScaleWrapper.dtype = ACL_FLOAT8_E8M0;
        keyScaleWrapper.dtype = ACL_FLOAT8_E8M0;
    }
}

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

at::Tensor QuantLightningIndexerMetadata(
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor>& cuSeqlensQ, const c10::optional<at::Tensor>& cuSeqlensK,
    const c10::optional<at::Tensor>& sequsedQ, const c10::optional<at::Tensor>& sequsedK,
    const c10::optional<at::Tensor>& cmpResidualK, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
    c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio, int64_t metadataSize)
{
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    }

    at::Tensor output = torch::empty({metadataSize}, torch::dtype(torch::kInt32).device(outputDevice));
    auto cuSeqlensQVal = get_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKVal = get_valid_tensor(cuSeqlensK, outputDevice);
    auto sequsedQVal = get_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKVal = get_valid_tensor(sequsedK, outputDevice);
    auto cmpResidualKVal = get_valid_tensor(cmpResidualK, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);

    ACLNN_CMD(aclnnQuantLightningIndexerV2Metadata, cuSeqlensQVal, cuSeqlensKVal, sequsedQVal, sequsedKVal,
              cmpResidualKVal, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK,
              layoutQStr, layoutKStr, maskMode, cmpRatio, output);
    return output;
}

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> ConstructQuantLightningIndexerOutputTensor(
    const at::Tensor& query, const at::Tensor& key, int64_t sparseCount, std::string queryLayoutStr,
    std::string keyLayoutStr, int64_t returnValue)
{
    at::SmallVector<int64_t, SIZE> outputSize;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0,
                    "All values within key's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparseCount > 0, "sparse count should be greater than 0, but now is ", sparseCount);
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparseCount};
    } else {
        int nDimIndex = 0;
        nDimIndex = (keyLayoutStr == "TND") ? DIM_1 : DIM_2;
        outputSize = {query.size(DIM_0), key.size(nDimIndex), sparseCount};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue) {
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kBFloat16));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kBFloat16));
    }

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor> QuantLightningIndexer(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, const at::Tensor& queryDequantScale,
    const at::Tensor& keyDequantScale, int64_t topk, int64_t quantMode, const c10::optional<at::Tensor>& cuSeqlensQ,
    const c10::optional<at::Tensor>& cuSeqlensK, const c10::optional<at::Tensor>& sequsedQ,
    const c10::optional<at::Tensor>& sequsedK, const c10::optional<at::Tensor>& cmpResidualK,
    const c10::optional<at::Tensor>& blockTable, const c10::optional<at::Tensor>& outputIdxOffset,
    const c10::optional<at::Tensor>& metadata, int64_t maxSeqlenQ, c10::string_view layoutQ, c10::string_view layoutK,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> quantLightningIndexerOutput =
        ConstructQuantLightningIndexerOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(quantLightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(quantLightningIndexerOutput);

    auto queryWrapper = MakeWrapper(query);
    auto keyWrapper = MakeWrapper(key);
    auto queryScaleWrapper = MakeWrapper(queryDequantScale);
    auto keyScaleWrapper = MakeWrapper(keyDequantScale);
    FixQLIV2AclDtypes(quantMode, queryWrapper, keyWrapper, queryScaleWrapper, keyScaleWrapper);

    ACLNN_CMD(aclnnQuantLightningIndexerV2, queryWrapper, keyWrapper, weights, queryScaleWrapper, keyScaleWrapper,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset, metadata, topk,
              quantMode, maxSeqlenQ, queryLayoutStr, keyLayoutStr, maskMode, cmpRatio, returnValue, sparseIndicesOut,
              sparseValuesOut);

    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}
// Candidate entry points require the new ACLNN symbols; the legacy ABI is unchanged.
at::Tensor QuantLightningIndexerMetadataCandidate(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                                  int64_t quantMode, const c10::optional<at::Tensor>& cuSeqlensQ,
                                                  const c10::optional<at::Tensor>& cuSeqlensK,
                                                  const c10::optional<at::Tensor>& sequsedQ,
                                                  const c10::optional<at::Tensor>& sequsedK,
                                                  const c10::optional<at::Tensor>& cmpResidualK, int64_t batchSize,
                                                  int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ,
                                                  c10::string_view layoutK, int64_t maskMode, int64_t cmpRatio,
                                                  int64_t candidateTopkBlocks, int64_t candidateBlockSize)
{
    TORCH_CHECK(
        (candidateTopkBlocks == -1 && candidateBlockSize == -1) || (candidateTopkBlocks > 0 && candidateBlockSize > 0),
        "candidate_topk_blocks and candidate_block_size must both be -1 or both be positive");
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (cuSeqlensK.has_value()) {
        outputDevice = cuSeqlensK.value().device();
    } else if (sequsedQ.has_value()) {
        outputDevice = sequsedQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (cmpResidualK.has_value()) {
        outputDevice = cmpResidualK.value().device();
    }

    at::Tensor output = torch::empty({QLI_V3_METADATA_SIZE}, torch::dtype(torch::kInt32).device(outputDevice));

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);

    c10::optional<at::Tensor> causalSeqlensKvGlobalNull = c10::nullopt;
    const aclIntArray* dcpParamsNone = nullptr;
    ACLNN_CMD(aclnnQuantLightningIndexerV3Metadata, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,
              causalSeqlensKvGlobalNull, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ,
              maxSeqlenK, layoutQStr, layoutKStr, maskMode, cmpRatio, candidateTopkBlocks, candidateBlockSize,
              dcpParamsNone, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> QuantLightningIndexerCandidate(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, const at::Tensor& queryDequantScale,
    const at::Tensor& keyDequantScale, int64_t topk, int64_t quantMode, const c10::optional<at::Tensor>& cuSeqlensQ,
    const c10::optional<at::Tensor>& cuSeqlensK, const c10::optional<at::Tensor>& sequsedQ,
    const c10::optional<at::Tensor>& sequsedK, const c10::optional<at::Tensor>& cmpResidualK,
    const c10::optional<at::Tensor>& blockTable, const c10::optional<at::Tensor>& outputIdxOffset,
    const c10::optional<at::Tensor>& metadata, int64_t maxSeqlenQ, c10::string_view layoutQ, c10::string_view layoutK,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue, int64_t candidateTopkBlocks, int64_t candidateBlockSize)
{
    const bool candidateEnabled = candidateTopkBlocks > 0 && candidateBlockSize > 0;
    TORCH_CHECK(candidateEnabled || (candidateTopkBlocks == -1 && candidateBlockSize == -1),
                "candidate_topk_blocks and candidate_block_size must both be -1 or both be positive");
    TORCH_CHECK(layoutQ == "BSND" || layoutQ == "TND", "layout_q must be BSND or TND");
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> quantLightningIndexerOutput =
        ConstructQuantLightningIndexerOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(quantLightningIndexerOutput);
    at::Tensor sparseValuesOut = std::get<1>(quantLightningIndexerOutput);
    auto lengthShape = sparseIndicesOut.sizes().vec();
    lengthShape.pop_back();
    auto candidateShape = lengthShape;
    candidateShape.push_back(candidateTopkBlocks);
    at::Tensor candidateBlockIndicesOut = candidateEnabled ?
                                              at::empty(candidateShape, query.options().dtype(at::kInt)) :
                                              at::empty({0}, query.options().dtype(at::kInt));
    at::Tensor candidateBlockLengthOut = candidateEnabled ? at::empty(lengthShape, query.options().dtype(at::kInt)) :
                                                            at::empty({0}, query.options().dtype(at::kInt));

    auto queryWrapper = MakeWrapper(query);
    auto keyWrapper = MakeWrapper(key);
    auto queryScaleWrapper = MakeWrapper(queryDequantScale);
    auto keyScaleWrapper = MakeWrapper(keyDequantScale);
    FixQLIV2AclDtypes(quantMode, queryWrapper, keyWrapper, queryScaleWrapper, keyScaleWrapper);

    c10::optional<at::Tensor> causalSeqlensKvGlobalNull = c10::nullopt;
    const aclIntArray* dcpParamsNone = nullptr;
    ACLNN_CMD(aclnnQuantLightningIndexerV3, queryWrapper, keyWrapper, weights, queryScaleWrapper, keyScaleWrapper,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset, metadata,
              causalSeqlensKvGlobalNull, topk, quantMode, maxSeqlenQ, queryLayoutStr, keyLayoutStr, maskMode, cmpRatio,
              returnValue, candidateTopkBlocks, candidateBlockSize, dcpParamsNone, sparseIndicesOut, sparseValuesOut,
              candidateBlockIndicesOut, candidateBlockLengthOut);

    return std::make_tuple(sparseIndicesOut, sparseValuesOut, candidateBlockIndicesOut, candidateBlockLengthOut);
}

std::tuple<at::Tensor, at::Tensor> ConstructDcpOutputTensor(const at::Tensor& query, const at::Tensor& key,
                                                            int64_t sparseCount, std::string queryLayoutStr,
                                                            std::string keyLayoutStr, int64_t returnValue)
{
    int64_t keyHeadNum = (keyLayoutStr == "TND") ? key.size(DIM_1) : key.size(DIM_2);
    at::SmallVector<int64_t, SIZE> outputSize;
    if (queryLayoutStr == "BSND") {
        outputSize = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparseCount};
    } else {
        outputSize = {query.size(DIM_0), keyHeadNum, sparseCount};
    }
    at::Tensor sparseIndicesOut = at::empty(outputSize, query.options().dtype(at::kInt));
    at::Tensor sparseValuesOut;
    if (returnValue != 0) {
        // DCP score 输出 BF16（与 arch35 对齐，跨 rank 合并前由框架按 BF16 处理）
        sparseValuesOut = at::empty(outputSize, query.options().dtype(at::kBFloat16));
    } else {
        sparseValuesOut = at::empty({0}, query.options().dtype(at::kBFloat16));
    }
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

std::tuple<at::Tensor, at::Tensor> QuantLightningIndexerDcp(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, const at::Tensor& queryDequantScale,
    const at::Tensor& keyDequantScale, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor>& causalSeqlensKvGlobal, const c10::optional<at::Tensor>& cuSeqlensQ,
    const c10::optional<at::Tensor>& cuSeqlensK, const c10::optional<at::Tensor>& sequsedQ,
    const c10::optional<at::Tensor>& sequsedK, const c10::optional<at::Tensor>& cmpResidualK,
    const c10::optional<at::Tensor>& blockTable, const c10::optional<at::Tensor>& outputIdxOffset,
    const c10::optional<at::Tensor>& metadata, int64_t maxSeqlenQ, c10::string_view layoutQ, c10::string_view layoutK,
    int64_t maskMode, int64_t cmpRatio, int64_t returnValue, int64_t cpWorld, int64_t cpRank, int64_t cpInterleaveSize)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")
    TORCH_CHECK(cpWorld > 1, "DCP requires cp_world > 1, but got ", cpWorld)
    TORCH_CHECK(cpRank >= 0 && cpRank < cpWorld, "DCP requires 0 <= cp_rank < cp_world, but got ", cpRank)
    TORCH_CHECK(cpInterleaveSize >= 1, "DCP requires cp_kv_cache_interleave_size >= 1")
    TORCH_CHECK(causalSeqlensKvGlobal.has_value(), "DCP requires causal_seqlens_kv_global")
    TORCH_CHECK(maskMode == MASK_MODE_RIGHT_DOWN_CAUSAL, "DCP requires mask_mode=3, but got ", maskMode)
    TORCH_CHECK(outputIdxOffset == c10::nullopt, "DCP requires output_idx_offset to be null")
    TORCH_CHECK(returnValue == 1, "DCP requires return_value=1")

    std::string queryLayoutStr = std::string(layoutQ);
    std::string keyLayoutStr = std::string(layoutK);
    std::tuple<at::Tensor, at::Tensor> output =
        ConstructDcpOutputTensor(query, key, topk, queryLayoutStr, keyLayoutStr, returnValue);
    at::Tensor sparseIndicesOut = std::get<0>(output);
    at::Tensor sparseValuesOut = std::get<1>(output);

    auto queryWrapper = MakeWrapper(query);
    auto keyWrapper = MakeWrapper(key);
    auto queryScaleWrapper = MakeWrapper(queryDequantScale);
    auto keyScaleWrapper = MakeWrapper(keyDequantScale);
    FixQLIV2AclDtypes(quantMode, queryWrapper, keyWrapper, queryScaleWrapper, keyScaleWrapper);

    // candidate 关闭：candidate_topk_blocks / candidate_block_size 固定 -1，
    // 两个候选输出传 nullptr（V3 aclnn 关闭态允许为空）
    int64_t candidateTopkBlocksOff = -1;
    int64_t candidateBlockSizeOff = -1;
    // 候选输出传空 optional（ConvertType 对空 optional 生成 nullptr）
    c10::optional<at::Tensor> candidateBlockIndicesNull = c10::nullopt;
    c10::optional<at::Tensor> candidateBlockLengthNull = c10::nullopt;
    aclIntArray* dcpParams = MakeDcpParamsArray(cpWorld, cpRank, cpInterleaveSize);
    TORCH_CHECK(dcpParams != nullptr, "Failed to create dcp_params array.")
    const aclIntArray* dcpParamsArg = dcpParams;
    ACLNN_CMD(aclnnQuantLightningIndexerV3, queryWrapper, keyWrapper, weights, queryScaleWrapper, keyScaleWrapper,
              cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, blockTable, outputIdxOffset, metadata,
              causalSeqlensKvGlobal, topk, quantMode, maxSeqlenQ, queryLayoutStr, keyLayoutStr, maskMode, cmpRatio,
              returnValue, candidateTopkBlocksOff, candidateBlockSizeOff, dcpParamsArg, sparseIndicesOut,
              sparseValuesOut, candidateBlockIndicesNull, candidateBlockLengthNull);
    ::Release(dcpParams);
    return std::tuple<at::Tensor, at::Tensor>(sparseIndicesOut, sparseValuesOut);
}

at::Tensor QuantLightningIndexerMetadataDcp(
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t quantMode,
    const c10::optional<at::Tensor>& causalSeqlensKvGlobal, const c10::optional<at::Tensor>& cuSeqlensQ,
    const c10::optional<at::Tensor>& cuSeqlensK, const c10::optional<at::Tensor>& sequsedQ,
    const c10::optional<at::Tensor>& sequsedK, const c10::optional<at::Tensor>& cmpResidualK, int64_t batchSize,
    int64_t maxSeqlenQ, int64_t maxSeqlenK, c10::string_view layoutQ, c10::string_view layoutK, int64_t maskMode,
    int64_t cmpRatio, int64_t metadataSize, int64_t cpWorld, int64_t cpRank, int64_t cpInterleaveSize)
{
    TORCH_CHECK(cpWorld > 1, "DCP requires cp_world > 1, but got ", cpWorld)
    TORCH_CHECK(causalSeqlensKvGlobal.has_value(), "DCP requires causal_seqlens_kv_global")
    TORCH_CHECK(maskMode == MASK_MODE_RIGHT_DOWN_CAUSAL, "DCP requires mask_mode=3, but got ", maskMode)
    at::Device outputDevice = at::Device(std::string("npu"));
    if (cuSeqlensQ.has_value()) {
        outputDevice = cuSeqlensQ.value().device();
    } else if (sequsedK.has_value()) {
        outputDevice = sequsedK.value().device();
    } else if (causalSeqlensKvGlobal.has_value()) {
        outputDevice = causalSeqlensKvGlobal.value().device();
    }
    at::Tensor output = torch::empty({metadataSize}, torch::dtype(torch::kInt32).device(outputDevice));
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKStr = std::string(layoutK);

    int64_t candidateTopkBlocksOff = -1;
    int64_t candidateBlockSizeOff = -1;
    aclIntArray* dcpParams = MakeDcpParamsArray(cpWorld, cpRank, cpInterleaveSize);
    TORCH_CHECK(dcpParams != nullptr, "Failed to create dcp_params array.")
    const aclIntArray* dcpParamsArg = dcpParams;
    // 可选输入保持 nullopt：上游 V3Metadata 会显式拒绝非空的 cu_seqlens_k 等参数，
    // 不能像 V2 那样用空 tensor 兜底（空 tensor 指针非空）。
    ACLNN_CMD(aclnnQuantLightningIndexerV3Metadata, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,
              causalSeqlensKvGlobal, numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK,
              layoutQStr, layoutKStr, maskMode, cmpRatio, candidateTopkBlocksOff, candidateBlockSizeOff, dcpParamsArg,
              output);
    ::Release(dcpParams);
    return output;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_lightning_indexer_metadata", &QuantLightningIndexerMetadata, "quant_lightning_indexer_metadata");
    m.def("quant_lightning_indexer", &QuantLightningIndexer, "quant_lightning_indexer");
    m.def("quant_lightning_indexer_candidate", &QuantLightningIndexerCandidate, "quant_lightning_indexer_candidate");
    m.def("quant_lightning_indexer_metadata_candidate", &QuantLightningIndexerMetadataCandidate,
          "quant_lightning_indexer_metadata_candidate");
    m.def("quant_lightning_indexer_dcp", &QuantLightningIndexerDcp, "quant_lightning_indexer_dcp");
    m.def("quant_lightning_indexer_metadata_dcp", &QuantLightningIndexerMetadataDcp,
          "quant_lightning_indexer_metadata_dcp");
}
} // namespace op_api
