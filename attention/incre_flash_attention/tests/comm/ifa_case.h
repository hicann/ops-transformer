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
 * \file ifa_case.h
 * \brief IncreFlashAttention 测试用例.
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
#include "tiling/ifa/tiling_data.h"
#include "tiling/ifa/tiling_stub.h"
#include "../../op_kernel/incre_flash_attention_tiling.h"
#define __NPU_HOST__

#define IFA_KERNEL_PARAM_                                                                        \
    uint8_t * query, uint8_t * key, uint8_t * value, uint8_t * pseShift,                         \
    uint8_t * attenMask, uint8_t * actualSeqLengths, uint8_t * deqScale1,                        \
    uint8_t * quantScale1, uint8_t * deqScale2, uint8_t * quantScale2,                           \
    uint8_t * quantOffset2, uint8_t * antiquantScale, uint8_t * antiquantOffset,                 \
    uint8_t * blocktable, uint8_t * kvPaddingSize, uint8_t * attentionOut,                       \
    uint8_t * workspace, uint8_t * tiling

#define IFA_INPUT_DTYPE                                     \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *,             \
    uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t *, uint8_t *,                        \
    uint8_t *, uint8_t * 

#define IFA_INPUT_PARAM                                             \
    query, key, value, pseShift,                                    \
    attenMask, actualSeqLengths, deqScale1,                         \
    quantScale1, deqScale2, quantScale2,                            \
    quantOffset2, antiquantScale, antiquantOffset,                  \
    blocktable, kvPaddingSize, attentionOut,                        \
    workspace, tiling

namespace ops::adv::tests::ifa {
class IfaCase : public ops::adv::tests::utils::Case {
    using OpInfo = ops::adv::tests::utils::OpInfo;
    using Context = ops::adv::tests::utils::Context;
    using ContextWithTemplateTilingKey = ops::adv::tests::utils::ContextWithTemplateTilingKey<IFA_INPUT_DTYPE>;
    using Tensor = ops::adv::tests::utils::Tensor;
    using TensorList = ops::adv::tests::utils::TensorList;

public:
    enum class PseShiftShapeType {
        NONE,
        B_N_1_S,
        _1_N_1_S,
    };
    enum class AttenMaskShapeType {
        NONE,
        B_N_1_S,
        B_1_S,
    };

    enum class QuantShapeType {
        NONE,
        PER_1,
        POST_1,
        ALL_1,
    };

    enum class AntiQuantShapeType {
        NONE,
        _2_H,
        _2_N_1_D,
        _2_N_D,
    };

    class Param {
    public:
        int64_t b = 0;
        int64_t n = 0;
        int64_t s = 0;
        int64_t d = 0;
        std::string layout = "BSH";
        int64_t numHeads = 1;
        int64_t kvNumHeads = 0;
        float scaleValue = 1.0f;
        int64_t blockSize = 0;
        int64_t innerPrecise = 1;
        ge::DataType qDataType = ge::DataType::DT_FLOAT16;
        ge::DataType kvDataType = ge::DataType::DT_FLOAT16;
        ge::DataType outDataType = ge::DataType::DT_FLOAT16;
        PseShiftShapeType pseShiftType = PseShiftShapeType::NONE;
        AttenMaskShapeType attenMaskType = AttenMaskShapeType::NONE;
        std::vector<int64_t> actualSeqLength = {};
        std::vector<int64_t> blocktable = {};
        QuantShapeType quantType = QuantShapeType::NONE;
        AntiQuantShapeType antiQuantType = AntiQuantShapeType::NONE;
        bool pageAttentionFlag = false;
        bool enbaleKvPaing = false;
        int64_t kvPaddingSize = 0;
        Param();
        Param(int64_t pB, int64_t pN, int64_t pS, int64_t pD, std::string pLayout, int64_t pNumHeads,
              int64_t pKvNumHeads, float pScaleValue, int64_t pBlockSize, int64_t pInnerPrecise,
              std::vector<int64_t> pActualSeqLength);
    };
    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actualSeqLengthsTensor = nullptr;
    };

    int64_t h;
    Tensor query, key, value, pseShift, attenMask, actualSeqLengths, deqScale1, quantScale1, deqScale2, quantScale2,
        quantOffset2, antiquantScale, antiquantOffset, blocktable, kvPaddingSize, attentionOut;
    OpInfo mOpInfo;
    ContextWithTemplateTilingKey mCtx;
    Param mParam;
    gert::OpImplRegisterV2::TilingKernelFunc ifaTilingFunc = nullptr;
    std::function<void(IFA_INPUT_DTYPE)> IfaKernelTemplateFunc;
    IfaCase();
    IfaCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
    IfaCase(const char *name, bool enable, const char *dbgInfo, 
            const std::function<void(IFA_INPUT_DTYPE)>& templatekeyKernelFunc,
            OpInfo incre, Param param);
    bool Run() override;
    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitCurrentCasePtr() override;
     bool DoOpTiling(DoTilingParam& tilingParam);
};

} // namespace ops::adv::tests::ifa
