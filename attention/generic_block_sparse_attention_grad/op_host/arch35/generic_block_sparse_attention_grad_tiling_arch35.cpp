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
static constexpr int64_t METADATA_HEADER_SIZE = 80;
static constexpr int64_t TASK_ENTRY_SIZE = 4;

int64_t AlignTo(int64_t x, int64_t align)
{
    return (x + align - 1) / align * align;
}

class GenericBlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GenericBlockSparseAttentionGradArch35Tiling(gert::TilingContext* context_)
        : TilingBaseClass(context_)
    {}

protected:
    bool IsCapable() override
    {
        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "query, key, value",
                                                   std::to_string(static_cast<int64_t>(dataType_)),
                                                   "supported dtypes are FP16 and BF16 for the current template");
            return false;
        }
        if (headDim_ != HEAD_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query", std::to_string(headDim_),
                                                     "query head dimension must be 128 for the current template");
            return false;
        }
        if (blockShapeX_ != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape", std::to_string(blockShapeX_),
                                                  "the first block dimension must be 1 for the current template");
            return false;
        }
        if (blockShapeY_ < 128 || blockShapeY_ % 64 != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "block_shape", std::to_string(blockShapeY_),
                "the second block dimension must be at least 128 and a multiple of 64");
            return false;
        }
        if (qHeadNum_ <= 0 || qHeadNum_ > MAX_HEAD_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "query", std::to_string(qHeadNum_),
                                                     "head number must be in the range [1, 128]");
            return false;
        }
        if (kvHeadNum_ <= 0 || kvHeadNum_ > MAX_HEAD_NUM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "key", std::to_string(kvHeadNum_),
                                                     "head number must be in the range [1, 128]");
            return false;
        }
        if (maskType_ != SUPPORTED_MASK_TYPE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "mask_type", std::to_string(maskType_),
                                                  "only 1 is supported by the current template");
            return false;
        }
        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        auto qInputDesc = context_->GetInputDesc(0);
        const gert::StorageShape* queryShape = context_->GetInputShape(0);
        const gert::StorageShape* keyShape = context_->GetInputShape(1);
        const gert::StorageShape* valueShape = context_->GetInputShape(2);
        const gert::StorageShape* idxShape = context_->GetInputShape(6);
        const gert::StorageShape* cntShape = context_->GetInputShape(7);
        const gert::StorageShape* metaShape = context_->GetOptionalInputShape(8);

        const auto* attrs = context_->GetAttrs();
        if (attrs == nullptr || qInputDesc == nullptr || queryShape == nullptr || keyShape == nullptr ||
            valueShape == nullptr || idxShape == nullptr || cntShape == nullptr) {
            if (attrs == nullptr) {
                const std::vector<const char*> keys = {"opName", "reason"};
                const std::vector<const char*> values = {
                    context_->GetNodeName(), "GetShapeAttrsInfo: attribute container is unavailable in TilingContext"};
                REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
            } else {
                const char* missingInput = qInputDesc == nullptr || queryShape == nullptr ? "query" :
                                           keyShape == nullptr                            ? "key" :
                                           valueShape == nullptr                          ? "value" :
                                           idxShape == nullptr                            ? "sparse_block_idx" :
                                                                                            "sparse_block_count";
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), missingInput,
                                                         "required input descriptor or shape is missing");
            }
            return ge::GRAPH_FAILED;
        }
        if (metaShape == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "metadata", "null",
                                                  "is required by the current sparse pattern");
            return ge::GRAPH_FAILED;
        }

        dataType_ = qInputDesc->GetDataType();

        const auto* blockShapeList = attrs->GetListInt(0);
        qLayout_ = attrs->GetAttrPointer<char>(1);
        kvLayout_ = attrs->GetAttrPointer<char>(2);
        const auto* sparseLayout = attrs->GetAttrPointer<int64_t>(3);
        const auto* residualMode = attrs->GetAttrPointer<int64_t>(9);
        layoutSparsePattern_ = sparseLayout != nullptr ? *sparseLayout : 1;
        residualBlockMode_ = residualMode != nullptr ? *residualMode : 0;
        if (layoutSparsePattern_ != 1 || residualBlockMode_ != 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context_->GetNodeName(), "layout_sparse_pattern and residual_block_mode",
                std::to_string(layoutSparsePattern_) + " and " + std::to_string(residualBlockMode_),
                "supported combination is sparse_pattern=1 with residual_block_mode=0");
            return ge::GRAPH_FAILED;
        }
        // is_consistent_topk (10) needs no specialization: each block uses its actual count.
        softmaxScale_ = *attrs->GetAttrPointer<float>(4);
        maskType_ = static_cast<uint32_t>(*attrs->GetAttrPointer<int64_t>(5));
        winLeft_ = static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>(7));
        winRight_ = static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>(8));

        if (qLayout_ == nullptr || kvLayout_ == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "layout_q, layout_kv",
                                                     "both layout attributes must be provided");
            return ge::GRAPH_FAILED;
        }
        if (strcmp(qLayout_, kvLayout_) != 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "layout_q and layout_kv",
                                                   std::string(qLayout_) + " and " + std::string(kvLayout_),
                                                   "layout_q must equal layout_kv");
            return ge::GRAPH_FAILED;
        }

        if (blockShapeList == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "block_shape",
                                                     "block_shape attribute must be provided");
            return ge::GRAPH_FAILED;
        }

        if (blockShapeList->GetSize() != 2) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape",
                                                  std::to_string(blockShapeList->GetSize()),
                                                  "block_shape must have exactly 2 elements");
            return ge::GRAPH_FAILED;
        }
        const int64_t* data = blockShapeList->GetData();
        blockShapeX_ = static_cast<int32_t>(data[0]);
        blockShapeY_ = static_cast<int32_t>(data[1]);
        if (blockShapeY_ <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_shape", std::to_string(blockShapeY_),
                                                  "block_shape[1] must be greater than 0");
            return ge::GRAPH_FAILED;
        }

        if (idxShape->GetOriginShape().GetDimNum() != 4 || cntShape->GetOriginShape().GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(),
                                                      "sparse_block_idx and sparse_block_count",
                                                      std::to_string(idxShape->GetOriginShape().GetDimNum()) + " and " +
                                                          std::to_string(cntShape->GetOriginShape().GetDimNum()),
                                                      "sparse_block_idx must be 4D and sparse_block_count must be 3D");
            return ge::GRAPH_FAILED;
        }
        const auto& idxStorage = idxShape->GetStorageShape();
        const auto& cntStorage = cntShape->GetStorageShape();
        const int64_t idxBatchNum = idxStorage.GetDim(0);
        const int64_t idxKvHeadNum = idxStorage.GetDim(1);
        numJ_ = idxStorage.GetDim(2);
        maxS1_ = idxStorage.GetDim(3);
        if (cntStorage.GetDim(0) != idxBatchNum || cntStorage.GetDim(1) != idxKvHeadNum ||
            cntStorage.GetDim(2) != numJ_) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "sparse_block_idx and sparse_block_count",
                "[" + std::to_string(cntStorage.GetDim(0)) + ", " + std::to_string(cntStorage.GetDim(1)) + ", " +
                    std::to_string(cntStorage.GetDim(2)) + "]",
                "sparse_block_count must match sparse_block_idx [B, N2, J] = [" + std::to_string(idxBatchNum) + ", " +
                    std::to_string(idxKvHeadNum) + ", " + std::to_string(numJ_) + "]");
            return ge::GRAPH_FAILED;
        }
        batchNum_ = idxBatchNum;
        kvHeadNum_ = idxKvHeadNum;

        if (strcmp(qLayout_, TND_STR) == 0) {
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "query, key, value",
                                                          std::to_string(queryShape->GetOriginShape().GetDimNum()),
                                                          "TND requires query/key/value to be 3-D");
                return ge::GRAPH_FAILED;
            }
            const gert::StorageShape* cuQShape = context_->GetOptionalInputShape(10);
            const gert::StorageShape* cuKvShape = context_->GetOptionalInputShape(11);
            if (cuQShape == nullptr || cuKvShape == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "cu_seq_lengths_q/cu_seq_lengths_kv",
                                                      "null", "required when layout is TND");
                return ge::GRAPH_FAILED;
            }
            const int64_t cuQElems = cuQShape->GetStorageShape().GetShapeSize();
            const int64_t cuKvElems = cuKvShape->GetStorageShape().GetShapeSize();
            if (cuQElems != idxBatchNum + 1 || cuKvElems != idxBatchNum + 1) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context_->GetNodeName(), "cu_seq_lengths_q and cu_seq_lengths_kv",
                    std::to_string(cuQElems) + " and " + std::to_string(cuKvElems),
                    "must hold B+1=" + std::to_string(idxBatchNum + 1) + " prefix offsets");
                return ge::GRAPH_FAILED;
            }
            for (int32_t idx = 12; idx <= 13; ++idx) {
                const gert::StorageShape* shape = context_->GetOptionalInputShape(idx);
                if (shape != nullptr && shape->GetStorageShape().GetShapeSize() != idxBatchNum) {
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        context_->GetNodeName(), "seqused", std::to_string(shape->GetStorageShape().GetShapeSize()),
                        "input " + std::to_string(idx) + " must hold B=" + std::to_string(idxBatchNum) + " lengths");
                    return ge::GRAPH_FAILED;
                }
            }
            qSeqLen_ = queryShape->GetStorageShape().GetDim(0);
            qHeadNum_ = queryShape->GetStorageShape().GetDim(1);
            headDim_ = queryShape->GetStorageShape().GetDim(2);
            kvSeqLen_ = keyShape->GetStorageShape().GetDim(0);
            kvHeadNum_ = keyShape->GetStorageShape().GetDim(1);
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
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "layout", qLayout_,
                                                  "supported values are TND, BSND and BNSD");
            return ge::GRAPH_FAILED;
        }

        if (kvHeadNum_ <= 0 || qHeadNum_ % kvHeadNum_ != 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "query and key",
                std::to_string(qHeadNum_) + " and " + std::to_string(kvHeadNum_),
                "key head number must be positive and query head number must be divisible by key head number");
            return ge::GRAPH_FAILED;
        }
        qGroup_ = qHeadNum_ / kvHeadNum_;

        if (batchNum_ != idxBatchNum || kvHeadNum_ != idxKvHeadNum) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "sparse_block_idx, query and key",
                "[" + std::to_string(idxBatchNum) + ", " + std::to_string(idxKvHeadNum) + "]",
                "sparse_block_idx [B, N2] must match query/key [B, N2] = [" + std::to_string(batchNum_) + ", " +
                    std::to_string(kvHeadNum_) + "]");
            return ge::GRAPH_FAILED;
        }
        const bool isTnd = strcmp(qLayout_, TND_STR) == 0;
        const int64_t expectNumJ = (kvSeqLen_ + blockShapeY_ - 1) / blockShapeY_;
        if (isTnd ? (numJ_ > expectNumJ) : (numJ_ != expectNumJ)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "sparse_block_idx", std::to_string(numJ_),
                "J must " + std::string(isTnd ? "be <=" : "equal") + " ceilDiv(kvSeqLen=" + std::to_string(kvSeqLen_) +
                    ", BlockY=" + std::to_string(blockShapeY_) + ")=" + std::to_string(expectNumJ));
            return ge::GRAPH_FAILED;
        }
        if (!isTnd && maxS1_ < qSeqLen_) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "sparse_block_idx and query",
                std::to_string(maxS1_) + " and " + std::to_string(qSeqLen_),
                "sparse_block_idx maxS1 must be greater than or equal to the query sequence length");
            return ge::GRAPH_FAILED;
        }
        const int64_t metaElems = metaShape->GetStorageShape().GetShapeSize();
        const int64_t requiredMetaElems = METADATA_HEADER_SIZE + batchNum_ * qHeadNum_ * numJ_ * TASK_ENTRY_SIZE;
        if (metaElems < requiredMetaElems) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "metadata", std::to_string(metaElems) + " elements",
                "must hold at least " + std::to_string(requiredMetaElems) + " int32 elements (80 + B * N1 * J * 4)");
            return ge::GRAPH_FAILED;
        }

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
        tilingData_.set_layoutSparsePattern(static_cast<uint32_t>(layoutSparsePattern_));
        tilingData_.set_winLeft(winLeft_);
        tilingData_.set_winRight(winRight_);
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
        uint32_t baseN = BASE_M;

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
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t usrOffset = 0;
        uint64_t dqSize = 0;
        uint64_t dkSize = 0;
        uint64_t sftgSize = 0;
        const bool deterministic = (context_->GetDeterministic() == 1);

        if (strcmp(qLayout_, TND_STR) == 0) {
            dqSize = static_cast<uint64_t>(qSeqLen_) * qHeadNum_ * headDim_;
            dkSize = static_cast<uint64_t>(kvSeqLen_) * kvHeadNum_ * headDim_;
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
        // Deterministic reduction writes dq directly; no intermediate FP32 dq is needed.
        if (!deterministic) {
            usrOffset += dqSize * sizeof(float);
        }
        tilingData_.set_dkWorkspaceOffset(usrOffset);
        usrOffset += dkSize * sizeof(float);
        tilingData_.set_dvWorkspaceOffset(usrOffset);
        usrOffset += dkSize * sizeof(float);
        tilingData_.set_dqSelWorkspaceOffset(usrOffset);
        usrOffset += tilingData_.get_cubeCoreNum() * 2 * tilingData_.get_baseM() * headDim_ * sizeof(float);

        if (deterministic) {
            uint64_t detCoreNum = tilingData_.get_cubeCoreNum();
            const uint64_t groupUpper = static_cast<uint64_t>(batchNum_ > 0 ? batchNum_ : 1) *
                                        static_cast<uint64_t>(kvHeadNum_ > 0 ? kvHeadNum_ : 1) *
                                        static_cast<uint64_t>(numJ_ > 0 ? numJ_ : 1);
            if (groupUpper < detCoreNum) {
                detCoreNum = groupUpper;
            }
            if (detCoreNum == 0) {
                detCoreNum = 1;
            }
            const uint64_t maxGroupElems = static_cast<uint64_t>(qSeqLen_) * (qHeadNum_ / kvHeadNum_) * headDim_;
            const bool compactDq = static_cast<uint64_t>(batchNum_) * kvHeadNum_ > 1 &&
                                   (strcmp(qLayout_, TND_STR) != 0 || dqSize >= 2 * 1024 * 1024);
            const uint64_t detElems = compactDq ? dqSize + (detCoreNum - 1) * maxGroupElems : detCoreNum * dqSize;
            const uint64_t detDqSize = detElems * sizeof(float);
            tilingData_.set_detCoreNum(detCoreNum);
            tilingData_.set_detDqWorkspaceOffset(usrOffset);
            tilingData_.set_detDqWorkspaceSize(detDqSize);
            usrOffset += detDqSize;
        } else {
            tilingData_.set_detCoreNum(0);
            tilingData_.set_detDqWorkspaceOffset(0);
            tilingData_.set_detDqWorkspaceSize(0);
        }

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
        // 1000-1005: non-deter; 1010-1015: deter (+10)
        uint64_t key = 1000;
        key = (dataType_ == ge::DT_BF16) ? key : key + 0b001;
        if (strcmp(qLayout_, BSND_STR) == 0) {
            key += 0b000;
        } else if (strcmp(qLayout_, BNSD_STR) == 0) {
            key += 0b010;
        } else if (strcmp(qLayout_, TND_STR) == 0) {
            key += 0b100;
        }
        if (context_->GetDeterministic() == 1) {
            key += 10;
        }
        return key;
    }

private:
    GenericBlockSparseAttentionGradTilingDataArch35 tilingData_;
    ge::DataType dataType_{ge::DT_FLOAT16};
    const char* qLayout_{nullptr};
    const char* kvLayout_{nullptr};
    static constexpr const char* BSND_STR = "BSND";
    static constexpr const char* BNSD_STR = "BNSD";
    static constexpr const char* TND_STR = "TND";
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
    int64_t layoutSparsePattern_{1};
    int64_t residualBlockMode_{0};
    uint32_t maskType_{0};
    float softmaxScale_{1.0f};
    int32_t winLeft_{-1};
    int32_t winRight_{-1};
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(GenericBlockSparseAttentionGrad, GenericBlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);

} // namespace GSAG_ARC35
} // namespace optiling
