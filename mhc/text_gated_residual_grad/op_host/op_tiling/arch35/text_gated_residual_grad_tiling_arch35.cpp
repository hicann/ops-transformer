/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <initializer_list>
#include <limits>
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
constexpr int64_t CHANNEL_SHAPE_LIMIT = 4096;
constexpr int64_t FLOAT_BYTES = sizeof(float);
constexpr int64_t SPLIT_K_CORE_WAVES = 4;
constexpr int64_t SPLIT_K_TARGET_ELEMENTS = 1152;
constexpr uint32_t BATCH_SCHEDULE_MODE = 1U;
constexpr size_t WORKSPACE_COUNT = 1;
constexpr size_t USER_WORKSPACE_INDEX = 0;
constexpr uint32_t TILING_TEMPLATE_PRIORITY = 0;

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
const std::vector<int32_t> supportedNpuArch = {static_cast<int32_t>(NpuArch::DAV_2201),
                                               static_cast<int32_t>(NpuArch::DAV_3510)};

ge::graphStatus CheckInputShape(gert::TilingContext* context, size_t irIndex, std::initializer_list<int64_t> dims,
                                const char* name, bool optional = false)
{
    auto shape = optional ? context->GetOptionalInputShape(irIndex) : context->GetRequiredInputShape(irIndex);
    if (shape == nullptr) {
        OP_LOGE(context, "%s shape is null.", name);
        return ge::GRAPH_FAILED;
    }
    const auto& storage = shape->GetStorageShape();
    if (storage.GetDimNum() != dims.size()) {
        OP_LOGE(context, "%s rank should be %zu, but got %zu.", name, dims.size(), storage.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    size_t i = 0;
    for (int64_t expect : dims) {
        if (storage.GetDim(i) != expect) {
            OP_LOGE(context, "%s dim[%zu] should be %ld, but got %ld.", name, i, expect, storage.GetDim(i));
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
        OP_LOGE(context, "%s desc is null.", name);
        return ge::GRAPH_FAILED;
    }
    if (desc->GetDataType() != expect) {
        OP_LOGE(context, "%s dtype mismatch, expect %d but got %d.", name, static_cast<int32_t>(expect),
                static_cast<int32_t>(desc->GetDataType()));
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
        OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAic());
        aivNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        npuArch_ = static_cast<int64_t>(ascendcPlatform.GetCurNpuArch());
        // Match the split-K wave estimate to kernel baseMNK: A5 C1a uses 256x128;
        // A2/A3 have 128 KiB L0C and use 128x128x64 C1/DX tiles, consuming 64 KiB per tile.
        cubeTileM_ = (npuArch_ == static_cast<int64_t>(NpuArch::DAV_2201)) ? hcgrad::HC_GRAD_CUBE_TILE_M_A2 :
                                                                             hcgrad::HC_GRAD_CUBE_TILE_M_A5;
        cubeTileN_ = hcgrad::HC_GRAD_C1A_TILE_N;
        OP_CHECK_IF(aicNum_ <= 0 || aivNum_ <= 0 || aivNum_ > TextGatedResidualGrad::HC_GRAD_MAX_AIV,
                    OP_LOGE(context_, "unsupported AIC/AIV core count"), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        // Shape planning runs before the framework's platform hook; budgets need the SoC first.
        OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get platform info failed"),
                    return ge::GRAPH_FAILED);
        int64_t b = 0;
        int64_t s = 0;
        ge::DataType xDtype = ge::DT_UNDEFINED;
        if (ReadInputDimensions(b, s) != ge::GRAPH_SUCCESS || ReadProjectionDimension() != ge::GRAPH_SUCCESS ||
            CheckChannelCoreCount() != ge::GRAPH_SUCCESS || ConfigureVectorBuffers() != ge::GRAPH_SUCCESS ||
            CheckRequiredShapes(b, s) != ge::GRAPH_SUCCESS || ConfigureSplitK() != ge::GRAPH_SUCCESS ||
            CheckRequiredDtypes(xDtype) != ge::GRAPH_SUCCESS || CheckGamma() != ge::GRAPH_SUCCESS ||
            CheckGate(b, s, xDtype) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ConfigureSchedule();
    }

    ge::graphStatus DoOpTiling() override
    {
        auto* tilingData = context_->GetTilingData<TextGatedResidualGradTilingDataArch35>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
        OP_CHECK_IF(splitkFactor_ <= 0, OP_LOGE(context_, "splitkFactor is not initialized"), return ge::GRAPH_FAILED);
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
        // Enable bounded producer-consumer scheduling by default; mode 0 uses SyncAll at every step.
        tilingData->scheduleMode = 1;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto platformInfo = context_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        userWorkspaceSize_ = WorkspaceLayout().totalBytes;
        workspaceSize_ = userWorkspaceSize_ + ascendcPlatform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(static_cast<uint32_t>(aicNum_));
        // All platforms synchronize all cores. Batch scheduling avoids mutual core waits across concurrent streams.
        context_->SetScheduleMode(BATCH_SCHEDULE_MODE);
        size_t* ws = context_->GetWorkspaceSizes(WORKSPACE_COUNT);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ws);
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
    }

private:
    ge::graphStatus ReadInputDimensions(int64_t& b, int64_t& s)
    {
        auto attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        auto channelsPtr = attrs->GetAttrPointer<int64_t>(IDX_ATTR_CHANNELS);
        OP_CHECK_NULL_WITH_CONTEXT(context_, channelsPtr);
        c_ = *channelsPtr;

        auto xShape = context_->GetRequiredInputShape(IDX_X);
        OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
        const auto& xStorage = xShape->GetStorageShape();
        OP_CHECK_IF(xStorage.GetDimNum() != INPUT_RANK, OP_LOGE(context_, "x must have shape [B,S,C*H]"),
                    return ge::GRAPH_FAILED);
        b = xStorage.GetDim(BATCH_DIM);
        s = xStorage.GetDim(SEQUENCE_DIM);
        d_ = xStorage.GetDim(FEATURE_DIM);
        OP_CHECK_IF(b < 0 || s < 0 || (s > 0 && b > std::numeric_limits<int64_t>::max() / s),
                    OP_LOGE(context_, "invalid B/S dimensions"), return ge::GRAPH_FAILED);
        n_ = b * s;
        OP_CHECK_IF(c_ <= 0 || d_ <= 0 || d_ % c_ != 0, OP_LOGE(context_, "x last dim must be divisible by channels"),
                    return ge::GRAPH_FAILED);
        h_ = d_ / c_;
        OP_CHECK_IF(d_ > std::numeric_limits<uint32_t>::max() || h_ > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context_, "D/H exceeds the kernel 32-bit shape range"), return ge::GRAPH_FAILED);
        // Each channel transfers H 16-bit elements, so H*2 must be a multiple of 32 bytes.
        // Misaligned DataCopyPad block lengths cause AIVEC traps, observed for H=1688/1692.
        OP_CHECK_IF(h_ % hcgrad::HC_GRAD_VECTOR_ALIGN != 0,
                    OP_LOGE(context_, "H must be a multiple of 16 (got H=%ld)", h_), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ReadProjectionDimension()
    {
        auto wDownShape = context_->GetRequiredInputShape(IDX_W_DOWN);
        OP_CHECK_NULL_WITH_CONTEXT(context_, wDownShape);
        const auto& wDownStorage = wDownShape->GetStorageShape();
        OP_CHECK_IF(wDownStorage.GetDimNum() != MATRIX_RANK || wDownStorage.GetDim(MATRIX_COLUMN_DIM) != d_,
                    OP_LOGE(context_, "w_down must have shape [R,C*H]"), return ge::GRAPH_FAILED);
        r_ = wDownStorage.GetDim(MATRIX_ROW_DIM);
        OP_CHECK_IF(r_ <= 0, OP_LOGE(context_, "R must be positive"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(r_ > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context_, "R exceeds the kernel 32-bit shape range"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(c_ <= 0 || c_ > CHANNEL_SHAPE_LIMIT, OP_LOGE(context_, "channels out of range"),
                    return ge::GRAPH_FAILED);
        // Cube uses R as dwDown's M dimension and R+C as dx's K dimension, with 16-element tail alignment.
        // Misaligned R causes MTE alignment errors on A2/A3 and incorrect dwDown/dgamma/dx on A5.
        // Observed failures at R=248/250/252/260 require rejecting this configuration on both architectures.
        OP_CHECK_IF(r_ % hcgrad::HC_GRAD_GATE_ALIGN != 0,
                    OP_LOGE(context_, "R must be a multiple of 16 (got R=%ld)", r_), return ge::GRAPH_FAILED);
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
                OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
                platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
                aivNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
            }
            OP_CHECK_IF(c_ > aivNum,
                        OP_LOGE(context_, "channels %ld exceeds the AIV core count %ld; each AIV must own one channel",
                                c_, aivNum),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ConfigureVectorBuffers()
    {
        uint64_t ubSizeDetected = 0;
        {
            auto platformInfo = context_->GetPlatformInfo();
            OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
            platform_ascendc::PlatformAscendC probePlatform(platformInfo);
            probePlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeDetected);
        }
        const int64_t fallbackUbBytes = (npuArch_ == static_cast<int64_t>(NpuArch::DAV_2201)) ?
                                            hcgrad::HC_GRAD_VECTOR_UB_A2 :
                                            hcgrad::HC_GRAD_VECTOR_UB_FALLBACK;
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
        OP_CHECK_IF(
            fullBytes > kUbBytes,
            OP_LOGE(context_,
                    "H=%ld (R=%ld) with C=%ld exceeds the vector UB budget: "
                    "pool(%ld) + alignedInv(%ld) + 12*H + scratch(%ld) must be <= %ld bytes "
                    "(pipeStages=%ld, rowBlock=%ld)",
                    h_, r_, c_, fullPool, InvBytes(rowBlock), FixedBytes(rowBlock), kUbBytes, pipeStages_, rowBlock),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckRequiredShapes(int64_t b, int64_t s)
    {
        OP_CHECK_IF(n_ > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context_, "token count exceeds the kernel scheduler range"), return ge::GRAPH_FAILED);
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
        taskCount_ = (n_ + TextGatedResidualGrad::HC_GRAD_TASK_ROWS - 1) / TextGatedResidualGrad::HC_GRAD_TASK_ROWS;
        workspaceSlots_ = taskCount_ < TextGatedResidualGrad::HC_GRAD_RING_SLOTS ?
                              taskCount_ :
                              TextGatedResidualGrad::HC_GRAD_RING_SLOTS;
        const int64_t dactBlockNum =
            static_cast<int64_t>((TextGatedResidualGrad::HC_GRAD_TASK_ROWS + static_cast<uint32_t>(cubeTileM_) - 1U) /
                                 static_cast<uint32_t>(cubeTileM_)) *
            ((r_ + cubeTileN_ - 1) / cubeTileN_);
        const int64_t kTileNum = (d_ + hcgrad::HC_GRAD_CUBE_TILE_K - 1) / hcgrad::HC_GRAD_CUBE_TILE_K;
        int64_t aicNum = aicNum_;
        if (aicNum <= 0) {
            auto platformInfo = context_->GetPlatformInfo();
            OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
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
        OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
        xDtype = xDesc->GetDataType();
        OP_CHECK_IF(xDtype != ge::DataType::DT_BF16 && xDtype != ge::DataType::DT_FLOAT16,
                    OP_LOGE(context_, "x dtype only supports bf16/fp16"), return ge::GRAPH_FAILED);
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
                    OP_LOGE(context_, "dgamma output is required when gamma is present"), return ge::GRAPH_FAILED);
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
                        OP_LOGE(context_, "H=%ld exceeds the dgamma Kahan UB budget (R=%ld)", h_, r_),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckGate(int64_t b, int64_t s, ge::DataType xDtype)
    {
        auto wGateTensor = context_->GetOptionalInputTensor(IDX_W_GATE);
        hasGate_ = wGateTensor != nullptr && wGateTensor->GetShapeSize() > 0 ? 1 : 0;
        OP_CHECK_IF(hasGate_ != 0 && GetOutputShapeByIrIndex(context_, OUT_DW_GATE) == nullptr,
                    OP_LOGE(context_, "dw_gate output is required when w_gate is present"), return ge::GRAPH_FAILED);
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
        if (MaterializeDupT()) {
            scheduleBeatCount_ = TextGatedResidualGrad::CountHcGradBeats<true>(
                static_cast<uint32_t>(taskCount_), static_cast<uint32_t>(workspaceSlots_), hasGamma_ != 0);
        } else {
            scheduleBeatCount_ = TextGatedResidualGrad::CountHcGradBeats<false, false>(
                static_cast<uint32_t>(taskCount_), static_cast<uint32_t>(workspaceSlots_), hasGamma_ != 0);
        }
        OP_CHECK_IF(n_ > 0 && scheduleBeatCount_ == 0, OP_LOGE(context_, "CV dependency scheduler failed to drain"),
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
               ((npuArch_ == static_cast<int64_t>(NpuArch::DAV_2201)) ? hcgrad::HC_GRAD_TRANSPOSE_SCRATCH : 0);
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
        const int64_t v6 = hcgrad::HC_GRAD_V6_POOL_BYTES * buf + hcgrad::HC_GRAD_V6_ROW_BYTES * h_;
        return v1 > v6 ? v1 : v6;
    }

    // The transpose fallback materializes dup^T in Vector and requires both workspace and scheduling dependencies.
    bool MaterializeDupT() const
    {
        return npuArch_ == static_cast<int64_t>(NpuArch::DAV_2201) && TextGatedResidualGrad::HC_GRAD_USE_DUP_T;
    }

    TextGatedResidualGrad::HcGradWorkspaceLayout WorkspaceLayout() const
    {
        return TextGatedResidualGrad::BuildHcGradWorkspaceLayout(
            static_cast<uint32_t>(workspaceSlots_), static_cast<uint32_t>(splitkFactor_),
            static_cast<uint32_t>(aivNum_), static_cast<uint32_t>(c_), static_cast<uint32_t>(h_),
            static_cast<uint32_t>(r_), static_cast<uint32_t>(d_), sizeof(uint16_t), hasGamma_ != 0, MaterializeDupT());
    }

    int64_t n_{0};
    int64_t c_{0};
    int64_t h_{0};
    int64_t r_{0};
    int64_t d_{0};
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
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(TextGatedResidualGrad, TextGatedResidualGradTiling, supportedNpuArch,
                                   TILING_TEMPLATE_PRIORITY);
} // namespace optiling
