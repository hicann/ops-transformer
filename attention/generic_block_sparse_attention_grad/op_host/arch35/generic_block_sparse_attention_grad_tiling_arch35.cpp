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
 * \file generic_block_sparse_attention_grad_tiling_arch35.cpp
 * \brief Arch35 tiling for GenericBlockSparseAttentionGrad (design §4.7 / §6).
 */

#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../generic_block_sparse_attention_grad_tiling.h"

namespace optiling {
namespace GSAG_ARC35 {

static constexpr int64_t HEAD_DIM = 128;
static constexpr int64_t BASE_M = 128;
static constexpr int64_t MAX_HEAD_NUM = 128;
static constexpr uint32_t SUPPORTED_MASK_TYPE = 1;

int64_t AlignTo(int64_t x, int64_t align)
{
    return (x + align - 1) / align * align;
}

class GenericBlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GenericBlockSparseAttentionGradArch35Tiling(gert::TilingContext *context_)
        : TilingBaseClass(context_)
    {}

protected:
    bool IsCapable() override
    {
        if (context_->GetDeterministic() == 1) {
            OP_LOGE(context_->GetNodeName(), "GenericBlockSparseAttentionGrad does not support Deterministic.");
            return false;
        }
        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE(context_->GetNodeName(), "only support DT_FLOAT16 and DT_BF16.");
            return false;
        }
        if (headDim_ != HEAD_DIM) {
            OP_LOGE(context_->GetNodeName(), "only support head_dim 128.");
            return false;
        }
        if (blockShapeX_ != 1) {
            OP_LOGE(context_->GetNodeName(), "only support block_shape[0] == 1.");
            return false;
        }
        if (blockShapeY_ < 128 || blockShapeY_ % 64 != 0) {
            OP_LOGE(context_->GetNodeName(), "block_shape[1] must be >= 128 and 64-aligned.");
            return false;
        }
        // Cube tile baseN shares UB/L1 with baseM; current layout budgets baseN<=128.
        if (blockShapeY_ > BASE_M) {
            OP_LOGE(context_->GetNodeName(), "block_shape[1]=%ld exceeds supported cube baseN=%ld (UB/L1 budget).",
                    blockShapeY_, BASE_M);
            return false;
        }
        if (isPackedGqa_ != 1) {
            OP_LOGE(context_->GetNodeName(), "only support is_packed_gqa == 1.");
            return false;
        }
        if (qHeadNum_ <= 0 || qHeadNum_ > MAX_HEAD_NUM) {
            OP_LOGE(context_->GetNodeName(), "qHeadNum=%ld must be in [1, %ld].", qHeadNum_, MAX_HEAD_NUM);
            return false;
        }
        if (kvHeadNum_ <= 0 || kvHeadNum_ > MAX_HEAD_NUM) {
            OP_LOGE(context_->GetNodeName(), "kvHeadNum=%ld must be in [1, %ld].", kvHeadNum_, MAX_HEAD_NUM);
            return false;
        }
        if (maskType_ != SUPPORTED_MASK_TYPE) {
            OP_LOGE(context_->GetNodeName(), "only support mask_type == %u, got %u.", SUPPORTED_MASK_TYPE, maskType_);
            return false;
        }
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        // Inputs: query=0, key=1, value=2, ..., rsvd_block_idx=6, rsvd_block_count=7, metadata=8
        // Optional: atten_mask=9, cu_seq_lengths=10, cu_seq_lengths_kv=11, seqused_q=12, seqused_kv=13.
        auto qInputDesc = context_->GetInputDesc(0);
        const gert::StorageShape *queryShape = context_->GetInputShape(0);
        const gert::StorageShape *keyShape = context_->GetInputShape(1);
        const gert::StorageShape *valueShape = context_->GetInputShape(2);
        const gert::StorageShape *idxShape = context_->GetInputShape(6);
        const gert::StorageShape *cntShape = context_->GetInputShape(7);
        const gert::StorageShape *metaShape = context_->GetInputShape(8);

        const auto *attrs = context_->GetAttrs();
        if (attrs == nullptr || qInputDesc == nullptr || queryShape == nullptr || keyShape == nullptr ||
            valueShape == nullptr || idxShape == nullptr || cntShape == nullptr || metaShape == nullptr) {
            OP_LOGE(context_->GetNodeName(), "required inputs/attrs are null.");
            return ge::GRAPH_FAILED;
        }

        dataType_ = qInputDesc->GetDataType();

        // Attrs: block_shape, is_packed_gqa, q_input_layout, kv_input_layout, scale_value,
        //        mask_type, softmax_precision, window_size_left, window_size_right
        const auto *blockShapeList = attrs->GetListInt(0);
        isPackedGqa_ = static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>(1));
        qLayout_ = attrs->GetAttrPointer<char>(2);
        kvLayout_ = attrs->GetAttrPointer<char>(3);
        softmaxScale_ = *attrs->GetAttrPointer<float>(4);
        maskType_ = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
        windowSizeLeft_ = static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>(7));
        windowSizeRight_ = static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>(8));

        if (qLayout_ == nullptr || kvLayout_ == nullptr || strcmp(qLayout_, kvLayout_) != 0) {
            OP_LOGE(context_->GetNodeName(), "q_input_layout must equal kv_input_layout.");
            return ge::GRAPH_FAILED;
        }

        if (blockShapeList != nullptr && blockShapeList->GetSize() >= 2) {
            const int64_t *data = blockShapeList->GetData();
            blockShapeX_ = static_cast<int32_t>(data[0]);
            blockShapeY_ = static_cast<int32_t>(data[1]);
        } else {
            blockShapeX_ = 1;
            blockShapeY_ = 128;
        }

        if (idxShape->GetOriginShape().GetDimNum() != 4 || cntShape->GetOriginShape().GetDimNum() != 3) {
            OP_LOGE(context_->GetNodeName(), "rsvd_block_idx must be 4D and rsvd_block_count must be 3D.");
            return ge::GRAPH_FAILED;
        }
        batchNum_ = idxShape->GetStorageShape().GetDim(0);
        kvHeadNum_ = idxShape->GetStorageShape().GetDim(1);
        numJ_ = idxShape->GetStorageShape().GetDim(2);
        maxS1_ = idxShape->GetStorageShape().GetDim(3);

        if (strcmp(qLayout_, TND_STR) == 0) {
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE(context_->GetNodeName(), "TND query must be 3D.");
                return ge::GRAPH_FAILED;
            }
            auto cuQ = context_->GetOptionalInputTensor(10);
            auto cuKv = context_->GetOptionalInputTensor(11);
            if (cuQ == nullptr || cuKv == nullptr) {
                OP_LOGE(context_->GetNodeName(), "TND requires cu_seq_lengths and cu_seq_lengths_kv.");
                return ge::GRAPH_FAILED;
            }
            // seqused_q/kv (inputs 12/13) are dynamic device values. They are
            // consumed by the kernel, while cu tensors retain packed-TND offsets.
            qSeqLen_ = queryShape->GetStorageShape().GetDim(0);
            qHeadNum_ = queryShape->GetStorageShape().GetDim(1);
            headDim_ = queryShape->GetStorageShape().GetDim(2);
            kvSeqLen_ = keyShape->GetStorageShape().GetDim(0);
            if (static_cast<int64_t>(keyShape->GetStorageShape().GetDim(1)) != kvHeadNum_) {
                OP_LOGE(context_->GetNodeName(), "key N2 mismatch with rsvd_block_idx.");
                return ge::GRAPH_FAILED;
            }
        } else if (strcmp(qLayout_, BSND_STR) == 0) {
            batchNum_ = queryShape->GetStorageShape().GetDim(0);
            qSeqLen_ = queryShape->GetStorageShape().GetDim(1);
            qHeadNum_ = queryShape->GetStorageShape().GetDim(2);
            headDim_ = queryShape->GetStorageShape().GetDim(3);
            kvSeqLen_ = keyShape->GetStorageShape().GetDim(1);
            kvHeadNum_ = keyShape->GetStorageShape().GetDim(2);
        } else if (strcmp(qLayout_, BNSD_STR) == 0) {
            batchNum_ = queryShape->GetStorageShape().GetDim(0);
            qHeadNum_ = queryShape->GetStorageShape().GetDim(1);
            qSeqLen_ = queryShape->GetStorageShape().GetDim(2);
            headDim_ = queryShape->GetStorageShape().GetDim(3);
            kvHeadNum_ = keyShape->GetStorageShape().GetDim(1);
            kvSeqLen_ = keyShape->GetStorageShape().GetDim(2);
        } else {
            OP_LOGE(context_->GetNodeName(), "layout must be TND/BSND/BNSD.");
            return ge::GRAPH_FAILED;
        }

        if (kvHeadNum_ <= 0 || qHeadNum_ % kvHeadNum_ != 0) {
            OP_LOGE(context_->GetNodeName(), "qHeadNum must be divisible by kvHeadNum.");
            return ge::GRAPH_FAILED;
        }
        qGroup_ = qHeadNum_ / kvHeadNum_;

        tilingData_.set_batchNum(batchNum_);
        tilingData_.set_qSeqLen(qSeqLen_);
        tilingData_.set_kvSeqLen(kvSeqLen_);
        tilingData_.set_qGroup(qGroup_);
        tilingData_.set_qHeadNum(qHeadNum_);
        tilingData_.set_kvHeadNum(kvHeadNum_);
        tilingData_.set_headDim(headDim_);
        tilingData_.set_softmaxScale(softmaxScale_);
        tilingData_.set_maxS1(maxS1_);
        tilingData_.set_numJ(numJ_);
        tilingData_.set_maskType(maskType_);
        tilingData_.set_isPackedGqa(static_cast<uint32_t>(isPackedGqa_));
        tilingData_.set_windowSizeLeft(windowSizeLeft_);
        tilingData_.set_windowSizeRight(windowSizeRight_);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        auto cubeCoreNum = ascendcPlatform.GetCoreNumAic();
        uint32_t baseM = BASE_M;
        uint32_t baseN = static_cast<uint32_t>(blockShapeY_);

        tilingData_.set_BlockX(static_cast<uint32_t>(blockShapeX_));
        tilingData_.set_BlockY(static_cast<uint32_t>(blockShapeY_));
        tilingData_.set_cubeCoreNum(cubeCoreNum);
        tilingData_.set_baseM(baseM);
        tilingData_.set_baseN(baseN);
        context_->SetBlockDim(cubeCoreNum);
        context_->SetScheduleMode(1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        constexpr uint32_t elementNum = 8 * 1024;
        uint32_t processS1Size = elementNum / static_cast<uint32_t>(headDim_);
        auto inputShape = ge::Shape({processS1Size, static_cast<int64_t>(headDim_)});
        uint32_t tmpSpaceSize = AscendC::GetSoftMaxGradMaxTmpSize(inputShape, sizeof(float), true, true);
        AscendC::SoftMaxGradTilingFunc(inputShape, sizeof(float), tmpSpaceSize, tilingData_.softmaxGradFrontTilingData,
                                       true);
        tilingData_.set_sftgTmpSpaceSize(tmpSpaceSize);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        // Design §4.7: sftg → dq → dk → dv (fp32 user workspace)
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t usrOffset = 0;
        uint64_t dqSize = 0;
        uint64_t dkSize = 0;
        uint64_t sftgSize = 0;

        if (strcmp(qLayout_, TND_STR) == 0) {
            dqSize = static_cast<uint64_t>(qSeqLen_) * qHeadNum_ * headDim_;
            dkSize = static_cast<uint64_t>(kvSeqLen_) * kvHeadNum_ * headDim_;
            // Pad sftg so DataCopy of last EvenCore tail cannot write past the buffer
            // (T=1024 / 56 AIVs leaves a non-16-aligned tail that may round up).
            sftgSize = static_cast<uint64_t>(AlignTo(static_cast<int64_t>(qSeqLen_) * qHeadNum_ * 8, 256));
        } else {
            dqSize = static_cast<uint64_t>(batchNum_) * qSeqLen_ * qHeadNum_ * headDim_;
            dkSize = static_cast<uint64_t>(batchNum_) * kvSeqLen_ * kvHeadNum_ * headDim_;
            sftgSize = static_cast<uint64_t>(AlignTo(static_cast<int64_t>(batchNum_) * qHeadNum_ * qSeqLen_ * 8, 256));
        }

        tilingData_.set_dqSize(dqSize);
        tilingData_.set_dkSize(dkSize);
        tilingData_.set_sftgWorkspaceOffset(usrOffset);
        usrOffset += sftgSize * sizeof(float);
        tilingData_.set_dqWorkspaceOffset(usrOffset);
        usrOffset += dqSize * sizeof(float);
        tilingData_.set_dkWorkspaceOffset(usrOffset);
        usrOffset += dkSize * sizeof(float);
        tilingData_.set_dvWorkspaceOffset(usrOffset);
        usrOffset += dkSize * sizeof(float);
        // Per-cube dQ_sel scratch (2 ping-pong slots): Fixpipe [baseM, D] then Vector scatter.
        tilingData_.set_dqSelWorkspaceOffset(usrOffset);
        usrOffset += tilingData_.get_cubeCoreNum() * 2 * tilingData_.get_baseM() * headDim_ * sizeof(float);

        context_->GetWorkspaceSizes(1)[0] = sysWorkspaceSize + usrOffset;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        // 1000 BF16 BSND, 1001 FP16 BSND, 1002 BF16 BNSD, 1003 FP16 BNSD, 1004 BF16 TND, 1005 FP16 TND
        uint64_t key = 1000;
        key = (dataType_ == ge::DT_BF16) ? key : key + 0b001;
        if (strcmp(qLayout_, BSND_STR) == 0) {
            key += 0b000;
        } else if (strcmp(qLayout_, BNSD_STR) == 0) {
            key += 0b010;
        } else if (strcmp(qLayout_, TND_STR) == 0) {
            key += 0b100;
        }
        return key;
    }

private:
    GenericBlockSparseAttentionGradTilingDataArch35 tilingData_;
    ge::DataType dataType_{ge::DT_FLOAT16};
    const char *qLayout_{nullptr};
    const char *kvLayout_{nullptr};
    static constexpr const char *BSND_STR = "BSND";
    static constexpr const char *BNSD_STR = "BNSD";
    static constexpr const char *TND_STR = "TND";
    int64_t batchNum_{0};
    int64_t qSeqLen_{0};
    int64_t kvSeqLen_{0};
    int64_t qHeadNum_{0};
    int64_t kvHeadNum_{0};
    int64_t qGroup_{0};
    int64_t headDim_{0};
    int64_t maxS1_{0};
    int64_t numJ_{0};
    int32_t blockShapeX_{1};
    int32_t blockShapeY_{128};
    int32_t isPackedGqa_{1};
    uint32_t maskType_{0};
    float softmaxScale_{1.0f};
    int32_t windowSizeLeft_{-1};
    int32_t windowSizeRight_{-1};
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(GenericBlockSparseAttentionGrad, GenericBlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

} // namespace GSAG_ARC35
} // namespace optiling
