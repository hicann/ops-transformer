/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../generic_block_sparse_attention_grad_tiling.h"
#include "../qsag_validation.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include <algorithm>

namespace optiling {
namespace {
struct QsagShapeInfoArch35 : qsag::ShapeInfo {
    uint64_t dp, ktile, dtile, gcap, slots;
};
class QsagTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit QsagTiling(gert::TilingContext* ctx)
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
            return ge::GRAPH_FAILED;
        const auto* layout = attrs->GetAttrPointer<int64_t>(3);
        if (!layout || *layout != 0)
            return ge::GRAPH_PARAM_INVALID;
        if (context_->GetDeterministic() != 0)
            return Fail("QSAG does not support deterministic execution", "deterministic");
        std::array<qsag::TensorInfo, 17> tensors;
        constexpr const char* optionalInputNames[] = {"metadata",          "atten_mask", "cu_seq_lengths_q",
                                                      "cu_seq_lengths_kv", "seqused_q",  "seqused_kv"};
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto* shape = i < 8 ?
                                    context_->GetInputShape(i) :
                                    (i < 14 ? context_->GetOptionalInputShape(i) : context_->GetOutputShape(i - 14));
            const auto* desc = i < 8 ? context_->GetInputDesc(i) :
                                       (i < 14 ? context_->GetOptionalInputDesc(i) : context_->GetOutputDesc(i - 14));
            auto& t = tensors[i];
            if (i >= 8 && i < 14 && ((shape == nullptr) != (desc == nullptr)))
                return Fail("optional input shape/descriptor presence mismatch", optionalInputNames[i - 8]);
            // The shared L0 wrapper represents omitted optionals with rank-0 descriptors.
            // User-provided scalar inputs are rejected by ACLNN validation before L0.
            if (i >= 8 && i < 14 && shape && shape->GetStorageShape().GetDimNum() == 0)
                continue;
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
            return Fail("missing Q/KV layout", q == nullptr ? "layout_q" : "layout_kv");
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
            return Fail("missing QSAG attributes", !scale     ? "scale_value" :
                                                   !mask      ? "mask_type" :
                                                   !precision ? "softmax_precision" :
                                                   !left      ? "win_left" :
                                                   !right     ? "win_right" :
                                                   !res       ? "residual_block_mode" :
                                                                "is_consistent_topk");
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
            return Fail(
                error.c_str(),
                "query, key, value, dout, out, lse, sparse_block_idx, sparse_block_count, metadata, atten_mask, "
                "cu_seq_lengths_q, cu_seq_lengths_kv, seqused_q, seqused_kv, dQuery, dKey, dValue, "
                "block_shape, layout_q, layout_kv, scale_value, mask_type, softmax_precision, win_left, "
                "win_right, residual_block_mode, is_consistent_topk");
        if (s_.group > 256)
            return Fail((std::string("QSAG arch35 requires G<=256; actual G=") + std::to_string(s_.group)).c_str(),
                        "query, key");
        const uint64_t baseKtile = qsag::BaseKTile(s_.group);
        s_.ktile = s_.dtile = baseKtile;
        s_.dp = (s_.d + 15) / 16 * 16;
        if (s_.group <= 16 && s_.dp <= 128) {
            s_.ktile = 256;
            s_.dtile = 128;
        }
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
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (!ubSize)
            return Fail("QSAG requires a valid UB capacity");
        data_.set_ubSize(ubSize);
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
        uint64_t qt = s_.sq, kt = s_.sk;
        if (s_.layout != 0 && (!qsag::Product(qt, s_.batch, qt) || !qsag::Product(kt, s_.batch, kt)))
            return Fail("token count overflow");
        const uint64_t cores = data_.get_cubeCoreNum();
        uint64_t offset = 0, start = 0;
        auto reserve = [&](std::initializer_list<uint64_t> dims) {
            uint64_t size = 1;
            for (auto d : dims)
                if (!qsag::Product(size, d, size))
                    return false;
            const auto limit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
            if (size > limit - 255 || offset > limit - ((size + 255) / 256 * 256))
                return false;
            start = offset;
            offset += (size + 255) / 256 * 256;
            return true;
        };
        if (!reserve({cores, 2, s_.ktile, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_selectedKOffset(start);
        if (!reserve({cores, 2, s_.ktile, s_.dp, 2}))
            return Fail("workspace overflow");
        data_.set_selectedVOffset(start);
        // Align element count to 256, as required by the Di layout (eight floats per row).
        uint64_t di = 0;
        if (!qsag::Product(qt, s_.nq, di) || !qsag::Product(di, 8, di) || di > INT64_MAX - 255)
            return Fail("Di overflow");
        if (!reserve({(di + 255) / 256 * 256, 4}))
            return Fail("workspace overflow");
        data_.set_diOffset(start);
        if (!reserve({qt, s_.nq, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dqOffset(start);
        if (!reserve({kt, s_.nk, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dkOffset(start);
        if (!reserve({kt, s_.nk, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_dvOffset(start);
        if (!reserve({cores, 2, s_.ktile, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_mm4Offset(start);
        if (!reserve({cores, 2, s_.ktile, s_.dp, 4}))
            return Fail("workspace overflow");
        data_.set_mm5Offset(start);
        // AIV compacts the actual sparse prefix after per-index causal masking.
        if (!reserve({s_.batch, s_.nk, s_.qcap, s_.topk, sizeof(int32_t)}))
            return Fail("filtered indices workspace overflow");
        data_.set_filteredIndicesOffset(start);
        // Each row stores the retained count after per-index causal masking.
        if (!reserve({s_.batch, s_.nk, s_.qcap, sizeof(int32_t)}))
            return Fail("filtered counts workspace overflow");
        data_.set_filteredCountsOffset(start);
        platform_ascendc::PlatformAscendC platform(context_->GetPlatformInfo());
        uint64_t sys = platform.GetLibApiWorkSpaceSize();
        if (sys > static_cast<uint64_t>(INT64_MAX) - offset)
            return Fail("total workspace overflow");
        auto* sizes = context_->GetWorkspaceSizes(1);
        if (!sizes)
            return Fail("missing workspace descriptor");
        sizes[0] = sys + offset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        auto* raw = context_->GetRawTilingData();
        if (!raw || raw->GetCapacity() < data_.GetDataSize())
            return Fail("tiling buffer too small");
        data_.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(data_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return dtype_ == ge::DT_BF16 ? 2000 : 2001;
    }

private:
    ge::graphStatus Fail(const char* message, const char* parameters = nullptr)
    {
        if (parameters != nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), parameters, message);
        } else {
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{context_->GetNodeName(), message}));
        }
        return ge::GRAPH_FAILED;
    }
    QsagShapeInfoArch35 s_{};
    QsagTilingDataArch35 data_;
    int dtype_ = ge::DT_UNDEFINED;
};
REGISTER_TILING_TEMPLATE_WITH_ARCH(GenericBlockSparseAttentionGrad, QsagTiling, static_cast<int32_t>(NpuArch::DAV_3510),
                                   0);
} // namespace
} // namespace optiling
