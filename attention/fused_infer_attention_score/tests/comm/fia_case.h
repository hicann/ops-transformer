/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_case.h
 * \brief FusedInferAttentionScore 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include "graph/types.h"
#include <functional>
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"
#include "tests/utils/context_with_template_tilingkey.h"
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fia/tiling_data.h"
#include "tiling/fia/tiling_stub.h"
#define __NPU_HOST__

#define FIA_KERNEL_PARAM_                                                                        \
    uint8_t * query, uint8_t * key, uint8_t * value, uint8_t * pse_shift,                        \
    uint8_t * attenMask, uint8_t * actualSeqLengths, uint8_t * actualSeqLengthsKV,               \
    uint8_t * deq_scale1, uint8_t * quant_scale1, uint8_t * deq_scale2,                          \
    uint8_t * quant_scale2, uint8_t * quant_offset2, uint8_t * antiquantScale,                   \
    uint8_t * antiquantOffset, uint8_t * blocktable, uint8_t * queryPaddingSize,                 \
    uint8_t * kvPaddingSize, uint8_t * keyAntiquantScale, uint8_t * keyAntiquantOffset,          \
    uint8_t * valueAntiquantScale, uint8_t * valueAntiquantOffset, uint8_t * keySharedPrefix,    \
    uint8_t * valueSharedPrefix, uint8_t * actualSharedPrefixLen, uint8_t * attentionOut,        \
    uint8_t * softmaxLse, uint8_t * workspace, uint8_t * tiling

#define FIA_INPUT_DTYPE                                     \
    uint8_t * , uint8_t * , uint8_t * , uint8_t * ,         \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * ,                     \
    uint8_t * , uint8_t * , uint8_t * 

#define FIA_INPUT_PARAMS                                           \
    query, key, value, pse_shift,                                  \
    attenMask, actualSeqLengths, actualSeqLengthsKV,               \
    deq_scale1, quant_scale1, deq_scale2,                          \
    quant_scale2, quant_offset2, antiquantScale,                   \
    antiquantOffset, blocktable, queryPaddingSize,                 \
    kvPaddingSize, keyAntiquantScale, keyAntiquantOffset,          \
    valueAntiquantScale, valueAntiquantOffset, keySharedPrefix,    \
    valueSharedPrefix, actualSharedPrefixLen, attentionOut,        \
    softmaxLse, workspace, tiling

namespace ops::adv::tests::fia {
enum class CaseMode : uint32_t {
    MLA_NOQUANT = 0,
    MLA_ANTIQUANT = 1,
    MLA_FULLQUANT = 2,
    GQA_NOQUANT = 3,
    GQA_ANTIQUANT = 4,
    GQA_FULLQUANT = 5,
    DEFAULT = 6
};

enum class CaseKvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

struct ShapeParam {
    int64_t b = 0;
    int64_t s = 0;
    int64_t n = 0;
    int64_t d = 0;
    int64_t t = 0;
};

class FiaCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using ContextWithTemplateTilingKey = ops::adv::tests::utils::ContextWithTemplateTilingKey<FIA_INPUT_DTYPE>;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t qs = 1;
        int64_t d = 0;
        int64_t h = 0;
        int64_t t = 0;
        int64_t numHeads = 0;
        float scaleValue = 1.0f;
        int64_t pre_tokens = 2147483647;
        int64_t next_tokens = 0;
        std::string layout = "BSH";
        int64_t kvNumHeads = 0;
        int64_t sparse_mode = 0;
        int64_t innerPrecise = 1;
        int64_t blockSize = 0;
        int64_t antiquant_mode = 0;
        int64_t softmax_lse_flag = 0;
        int64_t key_antiquant_mode = 0;
        int64_t value_antiquant_mode = 0;

        int64_t queryQuantMode = 0;
        int64_t pseType = 0;
        int64_t qkHeadDim = 0;
        int64_t vHeadDim = 0;
        int64_t ropeHeadDim = 0;
        CaseMode mode = CaseMode::DEFAULT;
        CaseKvStorageMode storageMode = CaseKvStorageMode::PAGE_ATTENTION;

        ge::DataType qDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kDataType = ge::DataType::DT_FLOAT16;
        ge::DataType vDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;
        std::vector<int64_t> actualSeqLength = {};
        std::vector<int64_t> actualSeqLengthKV = {};
        std::vector<int64_t> actualSharedPrefixLens = {};

        Param();
    };

    class DoTilingParam {
        public:
        gert::TilingContext* ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor* actualSeqLengthsTensor = nullptr;
        gert::Tensor* actualSeqLengthsKVTensor = nullptr;
        gert::Tensor* actualSharedPrefixLen = nullptr;
    };

    int64_t h;
    TensorList key, value;
    Tensor query, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deqScale1, quantScale1,
        deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, queryPaddinSize,
        kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
        keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, 
        dequantScaleQuery, keyRopeAntiquantScale, qStartIdx, kvStartIdx, attentionOut, softmaxLse;
    OpInfo mOpInfo;
    ContextWithTemplateTilingKey mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc fiaTilingFunc = nullptr;
    std::function<void(FIA_INPUT_DTYPE)> FiaKernelTemplateFunc;
    FiaCase();
    FiaCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    FiaCase(const char *name, bool enable, const char *dbgInfo, 
            const std::function<void(FIA_INPUT_DTYPE)>& templatekeyKernelFunc,
            OpInfo incre, Param param);
    bool Run() override;
    bool IsMla() const;
    Tensor ConstructTensor(std::string name, ShapeParam shapeParam, std::string layout,
        ge::DataType dtype, ge::Format format = ge::FORMAT_ND) const;
    TensorList ConstructTensorList(std::string name, ShapeParam shapeParam, std::string layout,
        ge::DataType dtype, ge::Format format = ge::FORMAT_ND) const;
    std::string GetQueryLayout(std::string layout) const;
    std::string GetOutLayout(std::string layout) const;
    std::string GetKvLayout(std::string layout) const;
    bool InitEnhanceParamActualSeqLenQ();
    bool InitEnhanceParamActualSeqLenKv();
    bool InitInOutParam();
    bool InitEnhanceParam();
    bool InitBasicParam();
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
    bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::fia
