/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_grad_tiling_data.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_GRAD_TILING_DATA_H_
#define LIGHTNING_INDEXER_GRAD_TILING_DATA_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "../op_kernel/lightning_indexer_grad_tiling.h"
namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    BnBsND = 2
};

struct LigParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo dy = {nullptr, nullptr};
    TilingRequiredParaInfo sparseIndices = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsK = {nullptr, nullptr};
    TilingRequiredParaInfo dQuery = {nullptr, nullptr};
    TilingRequiredParaInfo dKey = {nullptr, nullptr};
    TilingRequiredParaInfo dWeights = {nullptr, nullptr};

    int64_t headNum = 0;
    const char *layout = nullptr;
    int64_t sparseMode = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    bool deterministic = false;
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t DY_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INDEX = 3;
constexpr uint32_t WEIGTHS_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6;

// Outputs Index
constexpr uint32_t DQ_INDEX = 0;
constexpr uint32_t DK_INDEX = 0;
constexpr uint32_t DWEIGHTS_INDEX = 0;

// Attributes Index
constexpr uint32_t ATTR_HEADNUM_INDEX = 0;
constexpr uint32_t ATTR_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSEMODE_INDEX = 2;
constexpr uint32_t ATTR_PRETOKENS_INDEX = 3;
constexpr uint32_t ATTR_NEXTTOKENS_INDEX = 4;
constexpr uint32_t ATTR_DETERMINSTIC_INDEX = 5;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 0;
constexpr uint32_t DIM_IDX_TWO = 1;
constexpr uint32_t DIM_IDX_THREE = 2;
constexpr uint32_t DIM_IDX_FOUR = 3;

// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

// -----------算子CompileInfo定义-------------------
struct LIGCompileInfo {};

// ---------------算子Tiling类---------------
class LightningIndexerGradTiling {
public:
    explicit LightningIndexerGradTiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling();

    uint32_t batch = 0;
    uint32_t seqlenQ = 0;
    uint32_t seqlenK = 0;
    uint32_t topK = 0;
    uint32_t headNumQ = 0;
    uint32_t headNumK = 0;
    uint32_t groupNum = 0;
    uint32_t headDim = 0;
    uint32_t usedCoreNum = 0;
    int64_t dkSize = 0;
    int64_t dkWorkSpaceOffset = 0;
    int64_t dkCoreWorkspaceOffset = 0;
    int64_t keyGatherWorkspaceOffset = 0;
    int64_t reluInWorkspaceOffset = 0;
    int64_t reluGradWorkspaceOffset = 0;
    int64_t scatterAddWorkspaceOffset = 0;
    uint64_t sparseMode;
    bool deterministic = false;
    ge::DataType queryDataType = ge::DT_FLOAT16;

private:
    fe::PlatFormInfos *platformInfo_;
    const char *opName_;
    gert::TilingContext *context_ = nullptr;
    LIGTilingData *tilingData_ = context_->GetTilingData<LIGTilingData>();

    LigParaInfo opParamInfo;
};
} // namespace optiling
#endif // LIGHTNING_INDEXER_GRAD_TILING_DATA_H_
