/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include "tiling/tiling_api.h"
#include "text_gated_residual_grad_nn_base_tiling.h"
#include "text_gated_residual_grad_nn_a5_tiling.h"
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_base.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "../text_gated_residual_grad_tiling.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_tiling_data_arch35.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_tiling_key_arch35.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_pipeline_arch35.h"

namespace optiling {
namespace {
namespace hcgrad = TextGatedResidualGrad;
constexpr size_t INPUT_RANK = 3;
constexpr size_t MATRIX_RANK = 2;
constexpr size_t BATCH_DIM = 0;
constexpr size_t SEQUENCE_DIM = 1;
constexpr size_t FEATURE_DIM = 2;
constexpr size_t MATRIX_ROW_DIM = 0;
constexpr size_t MATRIX_COLUMN_DIM = 1;
constexpr int64_t FLOAT_BYTES = sizeof(float);
constexpr int64_t SPLIT_K_CORE_WAVES = 4;
constexpr int64_t SPLIT_K_TARGET_ELEMENTS = 1152;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1U;
constexpr size_t WORKSPACE_COUNT = 1;
constexpr size_t USER_WORKSPACE_INDEX = 0;
constexpr uint32_t TILING_TEMPLATE_PRIORITY = 0;

constexpr int64_t SINGLE_TASK_COUNT = 1;
constexpr int64_t CHUNK_PIPELINE_MIN_TASKS = 2;
constexpr int FULL_GEMM_PLAN = 0;
constexpr int GEMM_PLAN_COUNT = 2;
constexpr int A2_GATE_STAGE = static_cast<int>(hcgrad::HC_GRAD_NN_STAGE_GATE);
constexpr int A2_GEMM_STAGE_COUNT = 5;
constexpr uint32_t A5_GEMM_STAGE_COUNT = 4U;
constexpr uint32_t NO_GEMM_STAGES = 0U;
constexpr uint64_t NO_NN_SCRATCH = 0U;
constexpr int64_t MATMUL_TILING_ERROR = -1;

constexpr int64_t TUNED_A2_AIC_COUNT = 20;
constexpr int64_t TUNED_A2_AIV_COUNT = 40;
constexpr int64_t A2_ND2NZ_MIN_ROWS = 1024;
constexpr uint32_t A2_C1A_N_TILES = 2U;
constexpr uint32_t A2_C1A_ND2NZ_K_ROWS_PER_TILE = 128U;
constexpr uint64_t A2_ND2NZ_SHORT_K_STEP = 4U;
constexpr uint64_t A2_ND2NZ_LONG_K_STEP = 8U;
constexpr uint32_t A2_L2_SINGLE_TILE = 1U;
constexpr uint32_t A2_L2_ALL_CACHE_ENABLED = 1U;
constexpr uint32_t A2_L2_NO_CACHE_HINTS = 0U;
constexpr int64_t A2_DX_TAIL_L2_SPLIT_MIN_ROWS = 3072;
constexpr uint32_t A2_DX_TAIL_L2_N_TILES = 2U;
constexpr uint32_t GEMM_STAGE_MASK_BIT = 1U;

struct GemmShape {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t aRowStride{};
};

struct L2TileGrid {
    uint32_t m;
    uint32_t n;
};

constexpr L2TileGrid A2_FULL_CHUNK_L2_TILES[A2_GATE_STAGE] = {
    {10U, 1U}, // C1a
    {10U, 1U}, // C1b
    {1U, 8U},  // G56
    {2U, 2U},  // DX
};

constexpr size_t IDX_GRAD_MIXED = 0;
constexpr size_t IDX_X = 1;
constexpr size_t IDX_W_DOWN = 2;
constexpr size_t IDX_W_UP = 3;
constexpr size_t IDX_INV_RMS = 4;
constexpr size_t IDX_DOWN_PRE = 5;
constexpr size_t IDX_UP = 6;
constexpr size_t IDX_GRAD_INJECT = 7;
constexpr size_t IDX_W_GATE = 8;
constexpr size_t IDX_GAMMA = 9;
constexpr size_t IDX_INJECT = 10;
constexpr size_t IDX_ATTR_CHANNELS = 0;
static_assert(IDX_X == HC_GRAD_TPL_INPUT_X, "Template dtype binding must use the registered x input.");
constexpr size_t OUT_DW_GATE = 3;
constexpr size_t OUT_DGAMMA = 4;

// Shared tiling for A5 (DAV_3510) and A2/A3 (DAV_2201), with platform-specific Cube tiles, UB budgets and transpose
// scratch.
const std::vector<int32_t> supportedNpuArch = {HC_GRAD_ARCH_A2, HC_GRAD_ARCH_A5};

ge::graphStatus CheckInputShape(gert::TilingContext* context, size_t irIndex, std::initializer_list<int64_t> dims,
                                const char* name, bool optional = false)
{
    auto shape = optional ? context->GetOptionalInputShape(irIndex) : context->GetRequiredInputShape(irIndex);
    if (shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context), name);
        return ge::GRAPH_FAILED;
    }
    const auto& storage = shape->GetStorageShape();
    if (storage.GetDimNum() != dims.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(Ops::Base::GetOpInfo(context), name, std::to_string(storage.GetDimNum()),
                                     std::to_string(dims.size()));
        return ge::GRAPH_FAILED;
    }
    size_t i = 0;
    for (int64_t expect : dims) {
        if (storage.GetDim(i) != expect) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context), name, Ops::Base::ToString(storage),
                                                  "dim[" + std::to_string(i) + "] must be " + std::to_string(expect));
            return ge::GRAPH_FAILED;
        }
        ++i;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputDtype(gert::TilingContext* context, size_t irIndex, ge::DataType expect, const char* name,
                                bool optional = false)
{
    auto desc = optional ? context->GetOptionalInputDesc(irIndex) : context->GetRequiredInputDesc(irIndex);
    if (desc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context), name);
        return ge::GRAPH_FAILED;
    }
    if (desc->GetDataType() != expect) {
        OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context), name, Ops::Base::ToString(desc->GetDataType()),
                                  Ops::Base::ToString(expect));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

const gert::StorageShape* GetOutputShapeByIrIndex(gert::TilingContext* context, size_t irIndex)
{
    const auto* instance = context->GetIrOutputInstanceInfo(irIndex);
    if (instance == nullptr || instance->GetInstanceNum() == 0) {
        return nullptr;
    }
    return context->GetOutputShape(instance->GetInstanceStart());
}
} // namespace

class TextGatedResidualGradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit TextGatedResidualGradTiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        Reset();
    }
    ~TextGatedResidualGradTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITHOUT_REPORT(context_, "Tiling platform info is missing."),
                    return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
        aivNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        npuArch_ = static_cast<int64_t>(ascendcPlatform.GetCurNpuArch());
        // Match token scheduling to each backend's C1a output tile.
        // A2/A3 use 128x256 FP32 tiles, fitting their 128 KiB L0C.
        cubeTileM_ = (npuArch_ == HC_GRAD_ARCH_A2) ? hcgrad::HC_GRAD_CUBE_TILE_M_A2 : hcgrad::HC_GRAD_CUBE_TILE_M_A5;
        cubeTileN_ = (npuArch_ == HC_GRAD_ARCH_A2) ? hcgrad::HC_GRAD_CUBE_TILE_N_A2 : hcgrad::HC_GRAD_C1A_TILE_N;
        OP_CHECK_IF(aicNum_ <= 0 || aivNum_ <= 0 || aivNum_ > TextGatedResidualGrad::HC_GRAD_MAX_AIV,
                    OP_LOGE_WITHOUT_REPORT(context_,
                                           "Unsupported core counts for arch=%ld: AIC=%ld, AIV=%ld, "
                                           "expected positive counts and at most %ld AIV cores.",
                                           npuArch_, aicNum_, aivNum_, static_cast<int64_t>(hcgrad::HC_GRAD_MAX_AIV)),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        // Shape planning runs before the framework's platform hook; budgets need the SoC first.
        if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        int64_t b = 0;
        int64_t s = 0;
        ge::DataType xDtype = ge::DT_UNDEFINED;
        if (ReadInputDimensions(b, s) != ge::GRAPH_SUCCESS || ReadProjectionDimension() != ge::GRAPH_SUCCESS ||
            CheckChannelCoreCount() != ge::GRAPH_SUCCESS || ConfigureVectorBuffers() != ge::GRAPH_SUCCESS ||
            CheckRequiredShapes(b, s) != ge::GRAPH_SUCCESS || CheckRequiredDtypes(xDtype) != ge::GRAPH_SUCCESS ||
            ConfigureSplitK() != ge::GRAPH_SUCCESS || CheckGamma() != ge::GRAPH_SUCCESS ||
            CheckGate(b, s, xDtype) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ConfigureSchedule();
    }

    ge::graphStatus DoOpTiling() override
    {
        auto* tilingData = context_->GetTilingData<TextGatedResidualGradTilingDataArch35>();
        OP_CHECK_IF(tilingData == nullptr, OP_LOGE_WITHOUT_REPORT(context_, "Tiling data buffer is unavailable."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(splitkFactor_ <= 0,
                    OP_LOGE_WITHOUT_REPORT(context_, "Invalid split-K factor %ld for tokens=%ld and taskRows=%ld.",
                                           splitkFactor_, n_, taskRows_),
                    return ge::GRAPH_FAILED);
        tilingData->taskRows = taskRows_;
        tilingData->fullBatchRows = UseChunkedFullBatch() ? ChunkRows() : 0;
        tilingData->recomputeDirect = RecomputeDirectInV6() ? 1 : 0;
        // V6 combines DX with the direct path before both RMS backprop and dgamma.
        tilingData->lowpDx = 0;
        tilingData->nnMatmul = nnMatmul_ ? 1 : 0;
        tilingData->nnScratchOffset = tilingData->nnScratchBytes = tilingData->nnUbBackupOffset = NO_NN_SCRATCH;
        tilingData->n = n_;
        tilingData->c = c_;
        tilingData->h = h_;
        tilingData->r = r_;
        tilingData->d = d_;
        tilingData->taskCount = taskCount_;
        tilingData->workspaceSlots = workspaceSlots_;
        tilingData->splitkFactor = splitkFactor_;
        tilingData->hasGamma = hasGamma_;
        tilingData->hasGate = hasGate_;
        tilingData->aicNum = aicNum_;
        tilingData->aivNum = aivNum_;
        tilingData->scheduleBeatCount = scheduleBeatCount_;
        tilingData->workspaceSize = static_cast<int64_t>(WorkspaceLayout().totalBytes);
        tilingData->vectorUbBytes = ubBytes_;
        tilingData->pipeStages = pipeStages_;
        tilingData->vectorRowBlock = rowBlock_;
        tilingData->scheduleMode = UsesA5FullKSchedule() ?
                                       hcgrad::HC_GRAD_A5_FULL_K_SCHEDULE :
                                       (UseChunkedFullBatch() ? hcgrad::HC_GRAD_FULL_BATCH_SCHEDULE : 1U);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        if (npuArch_ == HC_GRAD_ARCH_A5 && n_ > 0) {
            return TileA5Nn();
        }
        if (npuArch_ != HC_GRAD_ARCH_A2 || n_ == 0) {
            return ge::GRAPH_SUCCESS;
        }
        auto* data = context_->GetTilingData<TextGatedResidualGradTilingDataArch35>();
        std::fill_n(data->nnBaseMask, GEMM_PLAN_COUNT, NO_GEMM_STAGES);
        std::fill_n(data->nnMixMask, GEMM_PLAN_COUNT, NO_GEMM_STAGES);
        data->nnWeightBatchMask = NO_GEMM_STAGES;
        data->nnScratchOffset = data->nnScratchBytes = NO_NN_SCRATCH;
        auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        const int64_t gateRows = hcgrad::HcGradAlign16(c_);
        const bool chunked = UseChunkedFullBatch();
        const int64_t planRows = chunked ? ChunkRows() : taskRows_;
        const int64_t remainderRows = n_ % planRows;
        const bool direct = !chunked && taskCount_ == SINGLE_TASK_COUNT && !nnMatmul_;
        const bool tuneNd2Nz = chunked && IsTunedModel() && aicNum_ == TUNED_A2_AIC_COUNT &&
                               aivNum_ == TUNED_A2_AIV_COUNT && hasGate_ && hasGamma_;
        for (int task = FULL_GEMM_PLAN; task < GEMM_PLAN_COUNT; ++task) {
            const int64_t rows =
                task == FULL_GEMM_PLAN ? std::min(n_, planRows) : (remainderRows ? remainderRows : planRows);
            const GemmShape shapes[A2_GEMM_STAGE_COUNT] = {{rows, r_, d_},
                                                           {d_, r_, rows},
                                                           {r_ + (hasGate_ && !direct ? gateRows : 0), d_, rows},
                                                           {rows, d_, r_ + (hasGate_ ? c_ : 0)},
                                                           {gateRows, d_, rows}};
            for (int stage = hcgrad::HC_GRAD_NN_STAGE_C1A; stage < A2_GEMM_STAGE_COUNT; ++stage) {
                const auto& shape = shapes[stage];
                const bool weightGrad = stage == hcgrad::HC_GRAD_NN_STAGE_C1B || stage == hcgrad::HC_GRAD_NN_STAGE_G56;
                const bool transA = weightGrad || stage == A2_GATE_STAGE;
                auto& nn = data->nnBase[task][stage];
                nn = {};
                HcGradNnBase::Policy policy;
                auto cfg = policy.Make(shape.m, shape.n, shape.k, transA, aicNum_, l1, l0a, l0b, l0c);
                const bool converted = tuneNd2Nz && rows >= A2_ND2NZ_MIN_ROWS && stage < A2_GATE_STAGE;
                if (converted) {
                    // The fixed model uses a full-K tile and converts just one operand to NZ.
                    const bool weightUp = stage == hcgrad::HC_GRAD_NN_STAGE_C1B;
                    cfg.baseM = weightUp ? hcgrad::HC_GRAD_C1B_TILE_M_A2 : hcgrad::HC_GRAD_CUBE_TILE_M_A2;
                    cfg.baseN = weightUp ? hcgrad::HC_GRAD_C1B_TILE_N_A2 : hcgrad::HC_GRAD_CUBE_TILE_N_A2;
                    cfg.baseK = hcgrad::HC_GRAD_CUBE_TILE_K;
                    if (stage == hcgrad::HC_GRAD_NN_STAGE_C1A) {
                        const uint32_t rowTiles = static_cast<uint32_t>(aicNum_) / A2_C1A_N_TILES;
                        const uint32_t rowsPerTile = HcGradNnBase::MathUtil::CeilDivision(rows, rowTiles);
                        cfg.baseM = std::min(hcgrad::HC_GRAD_CUBE_TILE_M_A2, hcgrad::HcGradAlign16(rowsPerTile));
                    }
                    cfg.singleCoreM = cfg.baseM;
                    cfg.singleCoreN = cfg.baseN;
                    cfg.stepKa = weightUp ? A2_ND2NZ_SHORT_K_STEP : A2_ND2NZ_LONG_K_STEP;
                    cfg.stepKb = weightUp ? A2_ND2NZ_LONG_K_STEP : A2_ND2NZ_SHORT_K_STEP;
                    cfg.depthA1 = HcGradNnBase::DB_SIZE * cfg.stepKa;
                    cfg.depthB1 = HcGradNnBase::DB_SIZE * cfg.stepKb;
                    cfg.iterateOrder = HcGradNnBase::ITER_COL_FIRST;
                    cfg.dbL0c = HcGradNnBase::DB_OFF_SIZE;
                }
                matmul_tiling::MultiCoreMatmulTiling mm(platform);
                const auto dtype = isBf16_ ? matmul_tiling::DataType::DT_BFLOAT16 : matmul_tiling::DataType::DT_FLOAT16;
                mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, transA);
                mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dtype, false);
                mm.SetCType(
                    matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    (stage == hcgrad::HC_GRAD_NN_STAGE_DX && data->lowpDx) ? dtype : matmul_tiling::DataType::DT_FLOAT);
                mm.SetDim(aicNum_);
                mm.SetShape(shape.m, shape.n, shape.k);
                mm.SetOrgShape(shape.m, shape.n, shape.k);
                // The local policy owns the per-core tile and the full K extent.
                mm.SetSingleShape(cfg.singleCoreM, cfg.singleCoreN, shape.k);
                mm.SetBufferSpace(l1, l0c, ub);
                mm.SetFixSplit(cfg.baseM, cfg.baseN, cfg.baseK);
                auto& t = nn.matmulTiling;
                const auto tilingRet = mm.GetTiling(t);
                OP_CHECK_IF(tilingRet == MATMUL_TILING_ERROR,
                            OP_LOGE_WITHOUT_REPORT(context_,
                                                   "NN base MatMul tiling failed for task=%d, stage=%d, "
                                                   "M=%ld, N=%ld, K=%ld, rc=%ld.",
                                                   task, stage, shape.m, shape.n, shape.k, tilingRet),
                            return ge::GRAPH_FAILED);
                t.usedCoreNum = aicNum_;
                t.singleCoreM = cfg.singleCoreM;
                t.singleCoreN = cfg.singleCoreN;
                t.singleCoreK = shape.k;
                t.baseM = cfg.baseM;
                t.baseN = cfg.baseN;
                t.baseK = cfg.baseK;
                t.stepM = cfg.stepM;
                t.stepN = cfg.stepN;
                t.depthA1 = cfg.depthA1;
                t.depthB1 = cfg.depthB1;
                t.stepKa = cfg.stepKa;
                t.stepKb = cfg.stepKb;
                t.iterateOrder = cfg.iterateOrder;
                t.dbL0C = cfg.dbL0c;
                nn.matmulRunInfo.transA = transA;
                nn.tileL2cacheTiling.mTileCntL2 = A2_L2_SINGLE_TILE;
                nn.tileL2cacheTiling.nTileCntL2 = A2_L2_SINGLE_TILE;
                nn.l2cacheUseInfo.l2CacheFlag = A2_L2_ALL_CACHE_ENABLED;
                if (converted) {
                    t.dbL0A = t.dbL0B = HcGradNnBase::DB_SIZE;
                    const bool convertA = stage == hcgrad::HC_GRAD_NN_STAGE_G56 || stage == hcgrad::HC_GRAD_NN_STAGE_DX;
                    const uint32_t rowSlice = HcGradNnBase::MathUtil::CeilDivision(rows, aivNum_);
                    if (convertA) {
                        nn.matmulRunInfo.nd2nzA = true;
                        nn.baseAN = rowSlice;
                        nn.baseAD = hcgrad::HcGradAlign16(transA ? t.M : t.Ka);
                    } else {
                        nn.matmulRunInfo.nd2nzB = true;
                        nn.baseBN = stage == hcgrad::HC_GRAD_NN_STAGE_C1A ? A2_C1A_ND2NZ_K_ROWS_PER_TILE : rowSlice;
                        nn.baseBD = hcgrad::HcGradAlign16(t.N);
                    }

                    auto& l2 = nn.tileL2cacheTiling;
                    if (rows == hcgrad::HC_GRAD_CHUNK_ROWS) {
                        l2.mTileCntL2 = A2_FULL_CHUNK_L2_TILES[stage].m;
                        l2.nTileCntL2 = A2_FULL_CHUNK_L2_TILES[stage].n;
                    } else if (stage == hcgrad::HC_GRAD_NN_STAGE_DX && rows >= A2_DX_TAIL_L2_SPLIT_MIN_ROWS) {
                        l2.nTileCntL2 = A2_DX_TAIL_L2_N_TILES;
                    }
                    l2.mTileBlock = HcGradNnBase::MathUtil::CeilDivision(t.M, t.baseM * l2.mTileCntL2);
                    l2.nTileBlock = HcGradNnBase::MathUtil::CeilDivision(t.N, t.baseN * l2.nTileCntL2);
                    nn.l2cacheUseInfo.l2CacheFlag = A2_L2_NO_CACHE_HINTS;

                    const uint64_t convertedExtent = hcgrad::HcGradAlign16(convertA ? t.M : t.N);
                    const uint64_t convertedK = hcgrad::HcGradAlign16(t.Ka);
                    extraNnWorkspace_ = std::max(extraNnWorkspace_, convertedExtent * convertedK * sizeof(uint16_t));
                    const uint32_t stageMask = GEMM_STAGE_MASK_BIT << stage;
                    data->nnBaseMask[task] |= stageMask;
                    data->nnMixMask[task] |= stageMask;
                }
                OP_CHECK_IF(
                    weightGrad && t.singleCoreK != shape.k,
                    OP_LOGE_WITHOUT_REPORT(context_, "dW tiling for task=%d, stage=%d needs full K=%ld; got %ld.", task,
                                           stage, shape.k, static_cast<int64_t>(t.singleCoreK)),
                    return ge::GRAPH_FAILED);
            }
        }
        if (extraNnWorkspace_ != NO_NN_SCRATCH) {
            data->nnScratchOffset = WorkspaceLayout().totalBytes;
            data->nnScratchBytes = extraNnWorkspace_;
            data->workspaceSize = data->nnScratchOffset + data->nnScratchBytes;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TileA5Nn()
    {
        auto* data = context_->GetTilingData<TextGatedResidualGradTilingDataArch35>();
        const int64_t gateRows = hcgrad::HcGradAlign16(c_);
        const int64_t packedStride = hcgrad::HcGradAlign16(r_ + gateRows);
        const bool chunked = UseChunkedFullBatch();
        const bool tuned = chunked && aicNum_ == hcgrad::HC_GRAD_A5_TUNED_AIC_COUNT;
        const int64_t planRows = chunked ? ChunkRows() : taskRows_;
        const int64_t remainderRows = n_ % planRows;
        for (int task = FULL_GEMM_PLAN; task < GEMM_PLAN_COUNT; ++task) {
            const int64_t rows =
                task == FULL_GEMM_PLAN ? std::min(n_, planRows) : (remainderRows ? remainderRows : planRows);
            const int64_t weightK = UsesA5FullKSchedule() ? n_ : rows;
            const GemmShape shapes[A5_GEMM_STAGE_COUNT] = {{rows, r_, d_, d_},
                                                           {d_, r_, weightK, d_},
                                                           {r_ + (hasGate_ ? gateRows : 0), d_, weightK, packedStride},
                                                           {rows, d_, r_ + (hasGate_ ? c_ : 0), packedStride}};
            for (uint32_t stage = hcgrad::HC_GRAD_NN_STAGE_C1A; stage < A5_GEMM_STAGE_COUNT; ++stage) {
                const auto& shape = shapes[stage];
                data->nnA5[task][stage] =
                    HcGradA5Tiling::MakePlan(shape.m, shape.n, shape.k, shape.aRowStride, aicNum_, stage, tuned);
                const bool transA = stage == hcgrad::HC_GRAD_NN_STAGE_C1B || stage == hcgrad::HC_GRAD_NN_STAGE_G56;
                data->nnA5Key[task][stage] =
                    transA ? (hcgrad::HC_GRAD_A5_BASIC_GEMM_KEY | hcgrad::HC_GRAD_A5_TRANSPOSE_A_KEY_BIT) :
                             hcgrad::HC_GRAD_A5_BASIC_GEMM_KEY;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        OP_CHECK_IF(platformInfo == nullptr,
                    OP_LOGE_WITHOUT_REPORT(context_, "Platform info is missing for workspace."),
                    return ge::GRAPH_FAILED);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        userWorkspaceSize_ = WorkspaceLayout().totalBytes + extraNnWorkspace_;
        workspaceSize_ = userWorkspaceSize_ + ascendcPlatform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(static_cast<uint32_t>(aicNum_));
        // All platforms synchronize all cores. Batch scheduling avoids mutual core waits across concurrent streams.
        context_->SetScheduleMode(BATCH_SCHEDULE_MODE);
        size_t* ws = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
        OP_CHECK_IF(
            ws == nullptr,
            OP_LOGE_WITHOUT_REPORT(context_, "Workspace size storage is unavailable for %zu entries.", WORKSPACE_COUNT),
            return ge::GRAPH_FAILED);
        ws[USER_WORKSPACE_INDEX] = workspaceSize_;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        const uint32_t dtypeKey = isBf16_ ? static_cast<uint32_t>(ge::DT_BF16) : static_cast<uint32_t>(ge::DT_FLOAT16);
        return GET_TPL_TILING_KEY(dtypeKey, static_cast<uint32_t>(hasGamma_));
    }

    void Reset()
    {
        n_ = 0;
        c_ = 0;
        h_ = 0;
        r_ = 0;
        d_ = 0;
        taskRows_ = hcgrad::HC_GRAD_TASK_ROWS;
        nnMatmul_ = false;
        taskCount_ = 0;
        workspaceSlots_ = 0;
        splitkFactor_ = 0;
        hasGamma_ = 0;
        hasGate_ = 0;
        aicNum_ = 0;
        aivNum_ = 0;
        npuArch_ = 0;
        cubeTileM_ = hcgrad::HC_GRAD_CUBE_TILE_M_A5;
        cubeTileN_ = hcgrad::HC_GRAD_C1A_TILE_N;
        scheduleBeatCount_ = 0;
        isBf16_ = true;
        workspaceSize_ = 0;
        userWorkspaceSize_ = 0;
        extraNnWorkspace_ = 0;
    }

private:
    ge::graphStatus ReadInputDimensions(int64_t& b, int64_t& s)
    {
        auto attrs = context_->GetAttrs();
        const auto* channelsPtr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(IDX_ATTR_CHANNELS);
        OP_CHECK_IF(channelsPtr == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_), "channels",
                                                             "the required integer attribute is missing"),
                    return ge::GRAPH_FAILED);
        c_ = *channelsPtr;

        auto xShape = context_->GetRequiredInputShape(IDX_X);
        OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context_), "x"),
                    return ge::GRAPH_FAILED);
        const auto& xStorage = xShape->GetStorageShape();
        OP_CHECK_IF(xStorage.GetDimNum() != INPUT_RANK,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(Ops::Base::GetOpInfo(context_), "x",
                                                 std::to_string(xStorage.GetDimNum()), std::to_string(INPUT_RANK)),
                    return ge::GRAPH_FAILED);
        b = xStorage.GetDim(BATCH_DIM);
        s = xStorage.GetDim(SEQUENCE_DIM);
        d_ = xStorage.GetDim(FEATURE_DIM);
        OP_CHECK_IF(
            b < 0 || s < 0 || (s > 0 && b > std::numeric_limits<int64_t>::max() / s),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context_), "x", Ops::Base::ToString(xStorage),
                                                  "B and S must be non-negative and B*S must fit in int64"),
            return ge::GRAPH_FAILED);
        n_ = b * s;
        OP_CHECK_IF(c_ <= 0 || c_ > hcgrad::HC_GRAD_MAX_CHANNELS,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "channels", std::to_string(c_),
                        "must be in [1, " + std::to_string(hcgrad::HC_GRAD_MAX_CHANNELS) + "]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(d_ <= 0 || d_ % c_ != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "x", Ops::Base::ToString(xStorage),
                        "the last dimension must be positive and divisible by channels=" + std::to_string(c_)),
                    return ge::GRAPH_FAILED);
        h_ = d_ / c_;
        OP_CHECK_IF(
            d_ > std::numeric_limits<uint32_t>::max() || h_ > std::numeric_limits<uint32_t>::max(),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context_), "x", Ops::Base::ToString(xStorage),
                                                  "D and H must fit in the kernel 32-bit shape range"),
            return ge::GRAPH_FAILED);
        // Each channel transfers H 16-bit elements, so H*2 must be a multiple of 32 bytes.
        // Misaligned DataCopyPad block lengths cause AIVEC traps, observed for H=1688/1692.
        OP_CHECK_IF(h_ % hcgrad::HC_GRAD_VECTOR_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "x", Ops::Base::ToString(xStorage),
                        "the last dimension divided by channels=" + std::to_string(c_) +
                            " gives H=" + std::to_string(h_) + ", which must be a multiple of 16"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ReadProjectionDimension()
    {
        auto wDownShape = context_->GetRequiredInputShape(IDX_W_DOWN);
        OP_CHECK_IF(wDownShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context_), "w_down"),
                    return ge::GRAPH_FAILED);
        const auto& wDownStorage = wDownShape->GetStorageShape();
        OP_CHECK_IF(wDownStorage.GetDimNum() != MATRIX_RANK || wDownStorage.GetDim(MATRIX_COLUMN_DIM) != d_,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "w_down", Ops::Base::ToString(wDownStorage),
                        "must have two dimensions [R, C*H] with C*H=" + std::to_string(d_)),
                    return ge::GRAPH_FAILED);
        r_ = wDownStorage.GetDim(MATRIX_ROW_DIM);
        OP_CHECK_IF(r_ <= 0 || r_ > hcgrad::HC_GRAD_MAX_PROJECTION,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "w_down", Ops::Base::ToString(wDownStorage),
                        "the first dimension R must be in [1, " + std::to_string(hcgrad::HC_GRAD_MAX_PROJECTION) + "]"),
                    return ge::GRAPH_FAILED);
        // Cube uses R as dwDown's M dimension and R+C as dx's K dimension, with 16-element tail alignment.
        // Misaligned R causes MTE alignment errors on A2/A3 and incorrect dwDown/dgamma/dx on A5.
        // Observed failures at R=248/250/252/260 require rejecting this configuration on both architectures.
        OP_CHECK_IF(r_ % hcgrad::HC_GRAD_GATE_ALIGN != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(Ops::Base::GetOpInfo(context_), "w_down",
                                                          Ops::Base::ToString(wDownStorage),
                                                          "the first dimension R must be a multiple of 16"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckChannelCoreCount()
    {
        // Each AIV handles one channel and retains H gamma/dgamma/Kahan elements.
        // Reject C greater than the AIV count: multiple channels per core cannot share this resident layout.
        {
            int64_t aivNum = aivNum_;
            if (aivNum <= 0) {
                auto platformInfo = context_->GetPlatformInfo();
                OP_CHECK_IF(platformInfo == nullptr,
                            OP_LOGE_WITHOUT_REPORT(context_, "Platform info is missing for the AIV channel check."),
                            return ge::GRAPH_FAILED);
                platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
                aivNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
            }
            OP_CHECK_IF(c_ > aivNum,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            Ops::Base::GetOpInfo(context_), "channels", std::to_string(c_),
                            "must not exceed the available " + std::to_string(aivNum) + " AIV cores"),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ConfigureVectorBuffers()
    {
        uint64_t ubSizeDetected = 0;
        {
            auto platformInfo = context_->GetPlatformInfo();
            OP_CHECK_IF(platformInfo == nullptr,
                        OP_LOGE_WITHOUT_REPORT(context_, "Platform info is missing for the vector UB budget."),
                        return ge::GRAPH_FAILED);
            platform_ascendc::PlatformAscendC probePlatform(platformInfo);
            probePlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeDetected);
        }
        const int64_t fallbackUbBytes =
            (npuArch_ == HC_GRAD_ARCH_A2) ? hcgrad::HC_GRAD_VECTOR_UB_A2 : hcgrad::HC_GRAD_VECTOR_UB_FALLBACK;
        const int64_t kUbBytes = (ubSizeDetected > 0) ? static_cast<int64_t>(ubSizeDetected) : fallbackUbBytes;
        ubBytes_ = kUbBytes;
        // pipeStages=0 selects serial blocks. V1/V6 buffer budgets depend on this setting.
        pipeStages_ = hcgrad::HC_GRAD_PIPE_MAX_STAGES;
        int64_t rowBlock = 1;
        if (h_ > 0) {
            while (rowBlock < hcgrad::HC_GRAD_MAX_ROW_BLOCK && PoolBytes(rowBlock + 1) + InvBytes(rowBlock + 1) +
                                                                       hcgrad::HC_GRAD_GAMMA_RESIDENT_BYTES * h_ +
                                                                       FixedBytes(rowBlock + 1) <=
                                                                   kUbBytes) {
                ++rowBlock;
            }
        }
        rowBlock_ = rowBlock;
        const int64_t fullPool = PoolBytes(rowBlock);
        const int64_t fullBytes =
            fullPool + InvBytes(rowBlock) + hcgrad::HC_GRAD_GAMMA_RESIDENT_BYTES * h_ + FixedBytes(rowBlock);
        OP_CHECK_IF(fullBytes > kUbBytes,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Ops::Base::GetOpInfo(context_), "x, w_down, channels",
                        "H=" + std::to_string(h_) + ", R=" + std::to_string(r_) + ", channels=" + std::to_string(c_) +
                            " require " + std::to_string(fullBytes) + " vector UB bytes; available " +
                            std::to_string(kUbBytes) + " bytes"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckRequiredShapes(int64_t b, int64_t s)
    {
        OP_CHECK_IF(n_ > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_), "x",
                                                             "B*S=" + std::to_string(n_) +
                                                                 " exceeds the kernel scheduler limit " +
                                                                 std::to_string(std::numeric_limits<uint32_t>::max())),
                    return ge::GRAPH_FAILED);
        if (CheckInputShape(context_, IDX_GRAD_MIXED, {b, s, h_}, "grad_mixed") != ge::GRAPH_SUCCESS ||
            CheckInputShape(context_, IDX_W_UP, {d_, r_}, "w_up") != ge::GRAPH_SUCCESS ||
            CheckInputShape(context_, IDX_INV_RMS, {b, s, c_}, "inv_rms") != ge::GRAPH_SUCCESS ||
            CheckInputShape(context_, IDX_DOWN_PRE, {b, s, r_}, "down_pre_act") != ge::GRAPH_SUCCESS ||
            CheckInputShape(context_, IDX_UP, {b, s, d_}, "up") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ConfigureSplitK()
    {
        nnMatmul_ = npuArch_ == HC_GRAD_ARCH_A5;
        // Fusion owns the bounded ring batch. Each GEMM uses its own local tile plan.
        // Size a task from the local C1a tile and the actual launched AIC count.
        // Both backends accumulate a complete K per output tile; R=320 on 950 retains its specialization.
        const bool coreSizedTasks = npuArch_ == HC_GRAD_ARCH_A5 || npuArch_ == HC_GRAD_ARCH_A2;
        taskRows_ = hcgrad::HC_GRAD_TASK_ROWS;
        if (nnMatmul_ || coreSizedTasks) {
            const bool tuned = nnMatmul_ && aicNum_ == hcgrad::HC_GRAD_A5_TUNED_AIC_COUNT && IsTunedModel();
            const uint32_t tileM = tuned ? hcgrad::HC_GRAD_C1A_UB_TILE_M : static_cast<uint32_t>(cubeTileM_);
            const uint32_t tileN = tuned ? hcgrad::HC_GRAD_C1A_UB_TILE_N : static_cast<uint32_t>(cubeTileN_);

            taskRows_ = hcgrad::ChooseHcGradTaskRows(static_cast<uint32_t>(n_), static_cast<uint32_t>(aicNum_),
                                                     static_cast<uint32_t>(r_), tileM, tileN);
            // Wide N already provides many parallel output tiles. Keep enough tokens per task
            // to amortize weight-gradient partials, instead of shrinking below the original batch.
            const uint32_t minimumRows = HcGradNnBase::ops::CeilAlign(hcgrad::HC_GRAD_TASK_ROWS, tileM);
            if (taskRows_ < minimumRows) {
                taskRows_ = minimumRows;
            }
            if (n_ > 0 && n_ < taskRows_) {
                taskRows_ = HcGradNnBase::ops::CeilAlign(n_, hcgrad::HC_GRAD_VECTOR_ALIGN);
            }
        }
        OP_CHECK_IF(
            taskRows_ <= 0,
            OP_LOGE_WITHOUT_REPORT(context_, "Invalid taskRows=%ld for tokens=%ld, projection=%ld.", taskRows_, n_, r_),
            return ge::GRAPH_FAILED);
        taskCount_ = (n_ + taskRows_ - 1) / taskRows_;
        workspaceSlots_ = taskCount_ < TextGatedResidualGrad::HC_GRAD_RING_SLOTS ?
                              taskCount_ :
                              TextGatedResidualGrad::HC_GRAD_RING_SLOTS;
        if (nnMatmul_ || coreSizedTasks) {
            splitkFactor_ = 1;
            return ge::GRAPH_SUCCESS;
        }
        const int64_t dactBlockNum =
            static_cast<int64_t>((static_cast<uint32_t>(taskRows_) + static_cast<uint32_t>(cubeTileM_) - 1U) /
                                 static_cast<uint32_t>(cubeTileM_)) *
            ((r_ + cubeTileN_ - 1) / cubeTileN_);
        const int64_t kTileNum = (d_ + hcgrad::HC_GRAD_CUBE_TILE_K - 1) / hcgrad::HC_GRAD_CUBE_TILE_K;
        int64_t aicNum = aicNum_;
        if (aicNum <= 0) {
            auto platformInfo = context_->GetPlatformInfo();
            OP_CHECK_IF(platformInfo == nullptr,
                        OP_LOGE_WITHOUT_REPORT(context_, "Platform info is missing for split-K planning."),
                        return ge::GRAPH_FAILED);
            platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
            aicNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
        }
        // Long FP32 accumulation chains at K=10240 amplify cancellation error (observed dact error around 1e-2).
        // Increase split-K to shorten each partial's accumulation chain.
        // Choose the split count using one AIC wave's capacity and ceil(D/1152),
        // then cap it by four AIC waves and the number of K tiles. This does not impose a fixed maximum chain length.
        const int64_t splitkCap = dactBlockNum > 0 ? (aicNum * SPLIT_K_CORE_WAVES) / dactBlockNum : 1;
        // Estimate splits per AIC wave from a full 1024-row task, rounding down.
        const int64_t splitkFill = dactBlockNum > 0 ? aicNum / dactBlockNum : 1;
        const int64_t splitkChain = (d_ + SPLIT_K_TARGET_ELEMENTS - 1) / SPLIT_K_TARGET_ELEMENTS;
        splitkFactor_ = splitkFill > splitkChain ? splitkFill : splitkChain;
        if (splitkFactor_ > splitkCap) {
            splitkFactor_ = splitkCap;
        }
        if (splitkFactor_ < 1) {
            splitkFactor_ = 1;
        }
        if (splitkFactor_ > kTileNum) {
            splitkFactor_ = kTileNum;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckRequiredDtypes(ge::DataType& xDtype)
    {
        auto xDesc = context_->GetRequiredInputDesc(IDX_X);
        OP_CHECK_IF(xDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(Ops::Base::GetOpInfo(context_), "x"),
                    return ge::GRAPH_FAILED);
        xDtype = xDesc->GetDataType();
        OP_CHECK_IF(
            xDtype != ge::DataType::DT_BF16 && xDtype != ge::DataType::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE(Ops::Base::GetOpInfo(context_), "x", Ops::Base::ToString(xDtype), "BF16 or FP16"),
            return ge::GRAPH_FAILED);
        isBf16_ = xDtype == ge::DataType::DT_BF16;
        if (CheckInputDtype(context_, IDX_GRAD_MIXED, xDtype, "grad_mixed") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(context_, IDX_W_DOWN, xDtype, "w_down") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(context_, IDX_W_UP, xDtype, "w_up") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(context_, IDX_DOWN_PRE, xDtype, "down_pre_act") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(context_, IDX_UP, xDtype, "up") != ge::GRAPH_SUCCESS ||
            CheckInputDtype(context_, IDX_INV_RMS, ge::DataType::DT_FLOAT, "inv_rms") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckGamma()
    {
        auto gammaTensor = context_->GetOptionalInputTensor(IDX_GAMMA);
        hasGamma_ = gammaTensor != nullptr && gammaTensor->GetShapeSize() > 0 ? 1 : 0;
        OP_CHECK_IF(hasGamma_ != 0 && GetOutputShapeByIrIndex(context_, OUT_DGAMMA) == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_), "dgamma",
                                                             "the output is required when gamma is provided"),
                    return ge::GRAPH_FAILED);
        if (hasGamma_ != 0 &&
            (CheckInputShape(context_, IDX_GAMMA, {d_}, "gamma", true) != ge::GRAPH_SUCCESS ||
             CheckInputDtype(context_, IDX_GAMMA, ge::DataType::DT_FLOAT, "gamma", true) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        // Kahan compensation has the same size as the dgamma accumulator and remains resident in UB.
        // ConfigureVectorBuffers already budgets 12H bytes for accumulator, compensation and gamma cache; verify the
        // fit.
        if (hasGamma_ != 0) {
            const int64_t kahanBytes =
                PoolBytes(rowBlock_) + hcgrad::HC_GRAD_KAHAN_RESIDENT_BYTES * h_ + FixedBytes(rowBlock_);
            OP_CHECK_IF(kahanBytes > ubBytes_,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            Ops::Base::GetOpInfo(context_), "gamma",
                            "dgamma with H=" + std::to_string(h_) + ", R=" + std::to_string(r_) + " requires " +
                                std::to_string(kahanBytes) + " vector UB bytes; available " + std::to_string(ubBytes_) +
                                " bytes"),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckGate(int64_t b, int64_t s, ge::DataType xDtype)
    {
        auto wGateTensor = context_->GetOptionalInputTensor(IDX_W_GATE);
        hasGate_ = wGateTensor != nullptr && wGateTensor->GetShapeSize() > 0 ? 1 : 0;
        OP_CHECK_IF(hasGate_ != 0 && GetOutputShapeByIrIndex(context_, OUT_DW_GATE) == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_), "dw_gate",
                                                             "the output is required when w_gate is provided"),
                    return ge::GRAPH_FAILED);
        if (hasGate_ != 0 &&
            (CheckInputShape(context_, IDX_W_GATE, {c_, d_}, "w_gate", true) != ge::GRAPH_SUCCESS ||
             CheckInputShape(context_, IDX_GRAD_INJECT, {b, s, c_}, "grad_inject", true) != ge::GRAPH_SUCCESS ||
             CheckInputShape(context_, IDX_INJECT, {b, s, c_}, "inject", true) != ge::GRAPH_SUCCESS ||
             CheckInputDtype(context_, IDX_W_GATE, xDtype, "w_gate", true) != ge::GRAPH_SUCCESS ||
             CheckInputDtype(context_, IDX_GRAD_INJECT, xDtype, "grad_inject", true) != ge::GRAPH_SUCCESS ||
             CheckInputDtype(context_, IDX_INJECT, xDtype, "inject", true) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ConfigureSchedule()
    {
        if (UseChunkedFullBatch()) {
            if (nnMatmul_ && taskCount_ == SINGLE_TASK_COUNT) {
                // Two aligned tasks keep gamma resident during V1 and V6 even when one chunk holds all rows.
                const auto tokensPerTask = HcGradNnBase::MathUtil::CeilDivision(n_, CHUNK_PIPELINE_MIN_TASKS);
                taskRows_ = hcgrad::HcGradAlign16(static_cast<uint32_t>(tokensPerTask));
                taskCount_ = (n_ + taskRows_ - 1) / taskRows_;
                workspaceSlots_ = taskCount_;
            }
            return ge::GRAPH_SUCCESS;
        }
        if (UsesA5FullKSchedule()) {
            scheduleBeatCount_ = TextGatedResidualGrad::CountHcGradBeats<false, false, true>(
                static_cast<uint32_t>(taskCount_), static_cast<uint32_t>(workspaceSlots_), hasGamma_ != 0);
        } else if (MaterializeDupT()) {
            scheduleBeatCount_ = TextGatedResidualGrad::CountHcGradBeats<true>(
                static_cast<uint32_t>(taskCount_), static_cast<uint32_t>(workspaceSlots_), hasGamma_ != 0);
        } else {
            scheduleBeatCount_ = TextGatedResidualGrad::CountHcGradBeats<false, false>(
                static_cast<uint32_t>(taskCount_), static_cast<uint32_t>(workspaceSlots_), hasGamma_ != 0);
        }
        OP_CHECK_IF(n_ > 0 && scheduleBeatCount_ == 0,
                    OP_LOGE_WITHOUT_REPORT(context_,
                                           "CV dependency scheduler failed to drain for tokens=%ld, "
                                           "taskRows=%ld, tasks=%ld, slots=%ld, hasGate=%ld, hasGamma=%ld.",
                                           n_, taskRows_, taskCount_, workspaceSlots_, hasGate_, hasGamma_),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // Allocate the pool per row block: V1/V6 process rowBlock rows at a time.
    // With pipeStages=2, each input has two stages and outputs reuse their stage's input slots.
    // V1 and V6 run separately and share the pool; allocate the larger of their layouts:
    //   V1: five block-sized FP32 buffers (three scratch, two direct outputs) and six 16-bit input/output buffers
    //       = 32 * bufCount bytes.
    //   V6: two stages of direct/dx2, x and row-sized scratch; dx reuses the 16-bit x slot
    //       = 24 * bufCount + 16 * H bytes.
    // Serial execution (pipeStages=0) uses six FP32 and three 16-bit buffers = 30 * bufCount bytes.
    // Resident storage is 4H bytes each for dgamma, Kahan compensation and gamma cache.
    // Both platforms read packed directly as ColumnMajor; A2/A3 retain scratch for the transpose fallback.
    // Size scalar storage from the actual rowBlock instead of reserving the 128-row maximum.
    // Use detected UB capacity, with architecture-specific fallbacks: 248 KiB on A5 and 192 KiB on A2/A3.
    int64_t FixedBytes(int64_t rowBlock) const
    {
        return hcgrad::HC_GRAD_SIGMOID_SCRATCH +
               hcgrad::HC_GRAD_SCALAR_SLOTS * rowBlock * hcgrad::HC_GRAD_DATA_BLOCK_BYTES +
               ((npuArch_ == HC_GRAD_ARCH_A2) ? hcgrad::HC_GRAD_TRANSPOSE_SCRATCH : 0);
    }

    int64_t InvBytes(int64_t rows) const
    {
        return ((FLOAT_BYTES * c_ * rows + hcgrad::HC_GRAD_DATA_BLOCK_MASK) / hcgrad::HC_GRAD_DATA_BLOCK_BYTES *
                hcgrad::HC_GRAD_DATA_BLOCK_BYTES) *
               ((pipeStages_ != 0) ? pipeStages_ : 1);
    }

    int64_t PoolBytes(int64_t rowBlock) const
    {
        const int64_t elems = (rowBlock * h_ > r_) ? rowBlock * h_ : r_;
        const int64_t buf =
            (elems + hcgrad::HC_GRAD_VECTOR_ALIGN_MASK) / hcgrad::HC_GRAD_VECTOR_ALIGN * hcgrad::HC_GRAD_VECTOR_ALIGN;
        if (pipeStages_ == 0) {
            return hcgrad::HC_GRAD_SERIAL_POOL_BYTES * buf;
        }
        const int64_t v1 = hcgrad::HC_GRAD_V1_POOL_BYTES * buf;
        int64_t v6 = hcgrad::HC_GRAD_V6_POOL_BYTES * buf + hcgrad::HC_GRAD_V6_ROW_BYTES * h_;
        if ((npuArch_ == HC_GRAD_ARCH_A2 || npuArch_ == HC_GRAD_ARCH_A5) && IsTunedModel()) {
            v6 += hcgrad::HC_GRAD_V6_DIRECT_RECOMP_EXTRA_BYTES * buf;
        }
        return v1 > v6 ? v1 : v6;
    }

    // The transpose fallback materializes dup^T in Vector and requires both workspace and scheduling dependencies.
    bool MaterializeDupT() const
    {
        return npuArch_ == HC_GRAD_ARCH_A2 && TextGatedResidualGrad::HC_GRAD_USE_DUP_T;
    }

    bool UsesA5FullKSchedule() const
    {
        // Retaining every token for a deferred dW GEMM makes workspace O(BS).
        // A5 uses the bounded per-task schedule; eligible A2 shapes use fixed full-batch chunks.
        return false;
    }

    int64_t ChunkRows() const
    {
        return n_ < hcgrad::HC_GRAD_CHUNK_ROWS ? n_ : hcgrad::HC_GRAD_CHUNK_ROWS;
    }

    bool IsTunedModel() const
    {
        return c_ == hcgrad::HC_GRAD_TUNED_CHANNELS && h_ == hcgrad::HC_GRAD_TUNED_HIDDEN_SIZE &&
               r_ == hcgrad::HC_GRAD_TUNED_PROJECTION_SIZE;
    }

    bool RecomputeDirectInV6() const
    {
        return UseChunkedFullBatch() && IsTunedModel() && taskCount_ >= CHUNK_PIPELINE_MIN_TASKS && pipeStages_ != 0;
    }

    bool UseChunkedFullBatch() const
    {
        if (npuArch_ == HC_GRAD_ARCH_A2) {
            return n_ >= taskRows_ && n_ % hcgrad::HC_GRAD_VECTOR_ALIGN == 0 &&
                   r_ % hcgrad::HC_GRAD_VECTOR_ALIGN == 0 && !MaterializeDupT() &&
                   ubBytes_ >= hcgrad::HC_GRAD_A2_NN_UB_BYTES;
        }
        // A5 uses the same bounded retained-chunk layout, but only for the tuned production shape.
        return npuArch_ == HC_GRAD_ARCH_A5 && IsTunedModel() && hasGate_ != 0 && hasGamma_ != 0 &&
               ((taskCount_ >= CHUNK_PIPELINE_MIN_TASKS && n_ >= taskRows_) ||
                (taskCount_ == SINGLE_TASK_COUNT && n_ >= hcgrad::HC_GRAD_TASK_ROWS / CHUNK_PIPELINE_MIN_TASKS &&
                 n_ % hcgrad::HC_GRAD_VECTOR_ALIGN == 0)) &&
               r_ % hcgrad::HC_GRAD_VECTOR_ALIGN == 0 && pipeStages_ != 0 && ubBytes_ >= hcgrad::HC_GRAD_A2_NN_UB_BYTES;
    }

    TextGatedResidualGrad::HcGradWorkspaceLayout WorkspaceLayout() const
    {
        const bool fullBatch = UseChunkedFullBatch();
        const uint32_t retainedRows =
            fullBatch ? static_cast<uint32_t>(ChunkRows()) : (UsesA5FullKSchedule() ? static_cast<uint32_t>(n_) : 0U);
        // Both input dtypes use the first R+C entries of each aligned G56 row for DX.
        const uint32_t dxColumns =
            npuArch_ == HC_GRAD_ARCH_A5 && fullBatch ?
                hcgrad::HcGradAlign16(static_cast<uint32_t>(r_) + hcgrad::HcGradAlign16(static_cast<uint32_t>(c_))) :
                static_cast<uint32_t>(r_ + (hasGate_ != 0 ? c_ : 0));
        return TextGatedResidualGrad::BuildHcGradWorkspaceLayout(
            static_cast<uint32_t>(workspaceSlots_), static_cast<uint32_t>(splitkFactor_),
            static_cast<uint32_t>(aivNum_), static_cast<uint32_t>(c_), static_cast<uint32_t>(h_),
            static_cast<uint32_t>(r_), static_cast<uint32_t>(d_), sizeof(uint16_t), hasGamma_ != 0, MaterializeDupT(),
            static_cast<uint32_t>(taskRows_), retainedRows, fullBatch, dxColumns, RecomputeDirectInV6(), fullBatch,
            fullBatch, fullBatch);
    }

    int64_t n_{0};
    int64_t c_{0};
    int64_t h_{0};
    int64_t r_{0};
    int64_t d_{0};
    int64_t taskRows_{hcgrad::HC_GRAD_TASK_ROWS};
    bool nnMatmul_{false};
    int64_t taskCount_{0};
    int64_t workspaceSlots_{0};
    int64_t splitkFactor_{0};
    int64_t hasGamma_{0};
    int64_t hasGate_{0};
    int64_t aicNum_{0};
    int64_t aivNum_{0};
    int64_t npuArch_{0};
    int64_t cubeTileM_{hcgrad::HC_GRAD_CUBE_TILE_M_A5};
    int64_t cubeTileN_{hcgrad::HC_GRAD_C1A_TILE_N};
    int64_t scheduleBeatCount_{0};
    int64_t ubBytes_{hcgrad::HC_GRAD_VECTOR_UB_FALLBACK};
    int64_t pipeStages_{0};
    int64_t rowBlock_{1};
    bool isBf16_{true};
    size_t workspaceSize_{0};
    uint64_t userWorkspaceSize_{0};
    uint64_t extraNnWorkspace_{0};
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(TextGatedResidualGrad, TextGatedResidualGradTiling, supportedNpuArch,
                                   TILING_TEMPLATE_PRIORITY);
} // namespace optiling
