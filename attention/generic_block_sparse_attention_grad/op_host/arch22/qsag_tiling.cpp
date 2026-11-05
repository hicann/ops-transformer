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
 * \file qsag_tiling.cpp
 * \brief Arch22 tiling for GenericBlockSparseAttentionGrad (QSAG, layoutSparsePattern=0 / residualBlockMode=1).
 *        Uses the arch22-specific QsagTilingDataArch22 struct and tiling keys 200/201.
 */

#include "../generic_block_sparse_attention_grad_tiling.h"
#include "../qsag_validation.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include <algorithm>
#include <string>
#include <vector>
#include "base/err_msg.h"

namespace optiling {
namespace {
class QsagTilingArch22 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit QsagTilingArch22(gert::TilingContext* ctx)
        : TilingBaseClass(ctx)
    {}

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        const auto* attrs = context_->GetAttrs();
        if (!attrs)
            return Fail("QSAG attributes are missing");
        const auto* layout = attrs->GetAttrPointer<int64_t>(3);
        if (!layout)
            return Fail("QSAG layout_sparse_pattern is missing; provide the operator attribute.",
                        "layout_sparse_pattern");
        if (*layout != 0)
            return ge::GRAPH_PARAM_INVALID;
        std::array<qsag::TensorInfo, 17> tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto* shape = i < 8 ?
                                    context_->GetInputShape(i) :
                                    (i < 14 ? context_->GetOptionalInputShape(i) : context_->GetOutputShape(i - 14));
            const auto* desc = i < 8 ? context_->GetInputDesc(i) :
                                       (i < 14 ? context_->GetOptionalInputDesc(i) : context_->GetOutputDesc(i - 14));
            auto& t = tensors[i];
            if (i >= 8 && i < 14 && ((shape == nullptr) != (desc == nullptr)))
                return Fail("optional input shape/descriptor presence mismatch");
            t.present = shape != nullptr && desc != nullptr;
            if (!t.present)
                continue;
            t.dtype = desc->GetDataType();
            const auto& storage = shape->GetStorageShape();
            for (size_t j = 0; j < storage.GetDimNum(); ++j)
                t.shape.push_back(storage.GetDim(j));
        }
        qsag::Config cfg;
        const auto *q = attrs->GetAttrPointer<char>(1), *kv = attrs->GetAttrPointer<char>(2);
        if (!q || !kv)
            return Fail("Q/KV layouts are missing; supply matching TND, BNSD or BSND layouts.", "layout_q,layout_kv");
        cfg.layoutQ = q;
        cfg.layoutKv = kv;
        const auto* block = attrs->GetListInt(0);
        if (block)
            cfg.block.assign(block->GetData(), block->GetData() + block->GetSize());
        const auto* scale = attrs->GetAttrPointer<float>(4);
        const auto *mask = attrs->GetAttrPointer<int64_t>(5), *precision = attrs->GetAttrPointer<int64_t>(6);
        const auto *left = attrs->GetAttrPointer<int64_t>(7), *right = attrs->GetAttrPointer<int64_t>(8);
        const auto* res = attrs->GetAttrPointer<int64_t>(9);
        const auto* topk = attrs->GetAttrPointer<bool>(10);
        if (!scale || !mask || !precision || !left || !right || !res || !topk)
            return Fail("Required QSAG attributes are missing; check scale_value, mask_type, softmax_precision, "
                        "win_left, win_right, residual_block_mode and is_consistent_topk.",
                        "QSAG attributes");
        cfg.scale = *scale;
        cfg.mask = *mask;
        cfg.precision = *precision;
        cfg.left = *left;
        cfg.right = *right;
        cfg.residual = *res;
        cfg.consistent = *topk;
        auto error =
            qsag::Validate(tensors, cfg, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, s_);
        if (!error.empty())
            return Fail(error.c_str(), "QSAG inputs and attributes");
        // Architecture-specific kernel limits stay outside the shared descriptor validator.
        for (const auto& tensor : tensors) {
            if (tensor.present &&
                std::any_of(tensor.shape.begin(), tensor.shape.end(), [](int64_t dimension) { return dimension == 0; }))
                return Fail("QSAG arch22 does not support zero-length tensor dimensions.", "QSAG tensors");
        }
        if (s_.group > 256 || s_.d > 256)
            return Fail((std::string("QSAG arch22 requires G<=256 and D<=256; actual G=") + std::to_string(s_.group) +
                         ", D=" + std::to_string(s_.d))
                            .c_str(),
                        "query,key");
        if (s_.layout == 0 && s_.qcap < s_.sq / s_.batch + (s_.sq % s_.batch != 0))
            return Fail((std::string("TND requires Qcap>=ceil(T1/B); actual Qcap=") + std::to_string(s_.qcap) +
                         ", T1=" + std::to_string(s_.sq) + ", B=" + std::to_string(s_.batch) +
                         "; enlarge the sparse query capacity.")
                            .c_str(),
                        "query,sparse_block_idx,sparse_block_count");
        const uint64_t baseKtile = s_.group <= 128 ? 128 : 64;
        if (s_.bs > baseKtile)
            return Fail((std::string("QSAG arch22 requires Bs<=") + std::to_string(baseKtile) +
                         "; actual Bs=" + std::to_string(s_.bs) + ", G=" + std::to_string(s_.group))
                            .c_str(),
                        "block_shape,query,key");
        // Arch22 UB capacity limits Ktile to 128 even for the small-group case.
        s_.ktile = s_.dtile = baseKtile;
        s_.dp = (s_.d + 15) / 16 * 16;
        s_.gcap = 2 * (((s_.group + 1) / 2 + 15) / 16) * 16;
        s_.slots = s_.group <= 64 ? 2 : 1;
        dtype_ = tensors[0].dtype;
        data_.set_batchNum(s_.batch);
        data_.set_qSeqLen(s_.sq);
        data_.set_kvSeqLen(s_.sk);
        data_.set_qHeadNum(s_.nq);
        data_.set_kvHeadNum(s_.nk);
        data_.set_headDim(s_.d);
        data_.set_headDimAlign(s_.dp);
        data_.set_qGroup(s_.group);
        data_.set_qCapacity(s_.qcap);
        data_.set_topK(s_.topk);
        data_.set_blockSize(s_.bs);
        data_.set_kTile(s_.ktile);
        data_.set_dTile(s_.dtile);
        data_.set_gCapacity(s_.gcap);
        data_.set_qDoSlots(s_.slots);
        data_.set_inputLayout(s_.layout);
        data_.set_maskType(cfg.mask);
        data_.set_softmaxScale(cfg.scale);
        data_.set_hasUsedQ(tensors[12].present);
        data_.set_hasUsedKv(tensors[13].present);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        platform_ascendc::PlatformAscendC platform(context_->GetPlatformInfo());
        uint64_t cores = platform.GetCoreNumAic();
        if (!cores)
            return Fail("QSAG requires AIC cores");
        data_.set_cubeCoreNum(cores);
        context_->SetBlockDim(cores);
        context_->SetScheduleMode(1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        uint64_t totalQueryTokens = s_.sq, totalKeyValueTokens = s_.sk;
        if (s_.layout != 0 && (!qsag::Product(totalQueryTokens, s_.batch, totalQueryTokens) ||
                               !qsag::Product(totalKeyValueTokens, s_.batch, totalKeyValueTokens)))
            return Fail("token count overflow");
        const uint64_t cubeCoreCount = data_.get_cubeCoreNum();
        uint64_t workspaceOffset = 0, partitionOffset = 0;
        auto reserveWorkspace = [&](std::initializer_list<uint64_t> dimensions) {
            uint64_t partitionBytes = 1;
            for (auto dimension : dimensions)
                if (!qsag::Product(partitionBytes, dimension, partitionBytes))
                    return false;
            const auto maxWorkspaceBytes = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
            if (partitionBytes > maxWorkspaceBytes - 255 ||
                workspaceOffset > maxWorkspaceBytes - ((partitionBytes + 255) / 256 * 256))
                return false;
            partitionOffset = workspaceOffset;
            workspaceOffset += (partitionBytes + 255) / 256 * 256;
            return true;
        };
        if (!reserveWorkspace({cubeCoreCount, 2, s_.ktile, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_selectedKOffset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.ktile, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_selectedVOffset(partitionOffset);
        // Align element count to 256, as required by the Di layout (eight floats per row).
        uint64_t diElementCount = 0;
        if (!qsag::Product(totalQueryTokens, s_.nq, diElementCount) ||
            !qsag::Product(diElementCount, 8, diElementCount) || diElementCount > INT64_MAX - 255)
            return Fail("Di overflow");
        if (!reserveWorkspace({(diElementCount + 255) / 256 * 256, 4}))
            return Fail("workspace overflow");
        data_.set_diOffset(partitionOffset);
        if (!reserveWorkspace({totalQueryTokens, s_.nq, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dqOffset(partitionOffset);
        if (!reserveWorkspace({totalKeyValueTokens, s_.nk, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dkOffset(partitionOffset);
        if (!reserveWorkspace({totalKeyValueTokens, s_.nk, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dvOffset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.ktile, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_mm4Offset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.ktile, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_mm5Offset(partitionOffset);
        // AIV 先按 causal 可见范围过滤稀疏索引，主循环只消费过滤后的前缀。
        if (!reserveWorkspace({s_.batch, s_.nk, s_.qcap, s_.topk, sizeof(int32_t)}))
            return Fail("filtered indices workspace overflow");
        data_.set_filteredIndicesOffset(partitionOffset);
        if (!reserveWorkspace({s_.batch, s_.nk, s_.qcap, sizeof(int32_t)}))
            return Fail("filtered counts workspace overflow");
        data_.set_filteredCountsOffset(partitionOffset);
        if (!reserveWorkspace({s_.d != s_.dp ? totalQueryTokens : 0, s_.nq, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_qPadOffset(partitionOffset);
        if (!reserveWorkspace({s_.d != s_.dp ? totalQueryTokens : 0, s_.nq, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_dOPadOffset(partitionOffset);
        // A5 用 UB 直传 AIV；arch22 的 AIC/AIV 物理分离改为 GM 中转：
        // score/grad 为 float ND，probability/dsProbability 为 T（fp16/bf16）ND。
        // 布局为双缓冲 [cubeCoreCount][2][gcap][ktile]（slot 0/1），AIC 写 2*halfRows 行，AIV 按半核写 halfRows 行；
        // padding 行 [2*halfRows_, gcap) 由双 AIV 在 Pre() 幂等清零（防 NaN 毒化 MM4/MM5）。
        if (!reserveWorkspace({cubeCoreCount, 2, s_.gcap, s_.ktile, 4}))
            return Fail("workspace overflow");
        data_.set_scoreGmOffset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.gcap, s_.ktile, 4}))
            return Fail("workspace overflow");
        data_.set_gradGmOffset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.gcap, s_.ktile, 2}))
            return Fail("workspace overflow");
        data_.set_probabilityGmOffset(partitionOffset);
        if (!reserveWorkspace({cubeCoreCount, 2, s_.gcap, s_.ktile, 2}))
            return Fail("workspace overflow");
        data_.set_dsProbabilityGmOffset(partitionOffset);
        platform_ascendc::PlatformAscendC platform(context_->GetPlatformInfo());
        uint64_t systemWorkspaceBytes = platform.GetLibApiWorkSpaceSize();
        if (systemWorkspaceBytes > static_cast<uint64_t>(INT64_MAX) - workspaceOffset)
            return Fail("total workspace overflow");
        auto* workspaceSizes = context_->GetWorkspaceSizes(1);
        if (!workspaceSizes)
            return Fail("GetWorkspaceSizes(1) returned null; check the Tiling workspace output slot.");
        workspaceSizes[0] = systemWorkspaceBytes + workspaceOffset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        auto* raw = context_->GetRawTilingData();
        if (!raw || raw->GetCapacity() < data_.GetDataSize())
            return Fail((std::string("Tiling buffer is unavailable or too small; capacity=") +
                         std::to_string(raw == nullptr ? 0U : raw->GetCapacity()) +
                         ", required=" + std::to_string(data_.GetDataSize()) + "; check Tiling output buffers.")
                            .c_str());
        data_.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(data_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return dtype_ == ge::DT_BF16 ? 200 : 201;
    }

private:
    ge::graphStatus Fail(const char* message, const char* paramName = nullptr)
    {
        const char* opName =
            context_->GetNodeName() == nullptr ? "GenericBlockSparseAttentionGrad" : context_->GetNodeName();
        if (paramName != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, paramName, message);
        } else {
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{opName, message}));
        }
        return ge::GRAPH_FAILED;
    }
    struct Arch22ShapeInfo : qsag::ShapeInfo {
        uint64_t dp = 0, ktile = 0, dtile = 0, gcap = 0, slots = 0;
    };
    Arch22ShapeInfo s_{};
    QsagTilingDataArch22 data_;
    int dtype_ = ge::DT_UNDEFINED;
};
REGISTER_TILING_TEMPLATE_WITH_ARCH(GenericBlockSparseAttentionGrad, QsagTilingArch22,
                                   std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 0);
} // namespace
} // namespace optiling
