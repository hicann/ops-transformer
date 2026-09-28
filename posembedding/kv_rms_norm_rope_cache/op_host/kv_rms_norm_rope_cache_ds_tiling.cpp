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
 * \file kv_rms_norm_rope_cache_ds_tiling.cpp
 * \brief
 */
#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t RMS_NORM_LENGTHS[2] = {512, 192};
int64_t RMS_NORM_LENGTH = 512;
constexpr int64_t ROPE_LENGTH = 64;
constexpr int64_t MAX_BLOCK_DIM = 65535;
constexpr int64_t V_LENGTH = 128;
constexpr int64_t D_LENGTH = 576;
constexpr uint64_t TLING_KEY_5011 = 5011;
constexpr uint64_t TLING_KEY_5010 = 5010;
constexpr uint64_t TLING_KEY_5000 = 5000;
constexpr uint64_t TLING_KEY_4011 = 4011;
constexpr uint64_t TLING_KEY_4001 = 4001;
constexpr uint64_t TLING_KEY_4010 = 4010;
constexpr uint64_t TLING_KEY_4000 = 4000;
constexpr uint64_t TLING_KEY_5001 = 5001;
constexpr uint64_t TLING_KEY_3001 = 3001;
constexpr uint64_t TLING_KEY_3000 = 3000;
constexpr uint64_t TLING_KEY_3010 = 3010;
constexpr uint64_t TLING_KEY_2000 = 2000;
constexpr uint64_t TLING_KEY_2001 = 2001;
constexpr uint64_t TLING_KEY_1000 = 1000;
constexpr uint64_t TLING_KEY_1010 = 1010;
constexpr uint64_t TLING_KEY_1001 = 1001;
constexpr uint64_t TLING_KEY_1011 = 1011;

bool KvRmsNormRopeCacheTilingDs::CheckScaleValid(const gert::TilingContext *context)
{
    auto scale1Shape = context->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context->GetOptionalInputShape(C_KV_SCALE_IDX);

    bool isValid = true;
    isValid = isValid && ((scale1Shape != nullptr) || (scale2Shape != nullptr));
    if ((scale1Shape != nullptr) && (scale2Shape != nullptr)) {
        isValid = isValid && (scale1Shape->GetStorageShape().GetDimNum() == scale2Shape->GetStorageShape().GetDimNum());
    }
    isValid = isValid && (((scale1Shape != nullptr) && (scale1Shape->GetStorageShape().GetDimNum() <= DIM_TWO)) ||
                          ((scale2Shape != nullptr) && (scale2Shape->GetStorageShape().GetDimNum() <= DIM_TWO)));
    if (!isValid) {
        return false;
    }
    if (methodMode_ == 0) {
        if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == ROPE_LENGTH);
        } else if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(1) == ROPE_LENGTH);
        }

        if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }
    } else {
        if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (scale1Shape != nullptr && scale1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (scale1Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }

        if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == V_LENGTH);
        } else if (scale2Shape != nullptr && scale2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (scale2Shape->GetStorageShape().GetDim(1) == V_LENGTH);
        }
    }

    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::CheckOffsetValid(const gert::TilingContext *context)
{
    auto offset1Shape = context->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context->GetOptionalInputShape(C_KV_OFFSET_IDX);

    bool isValid = true;
    isValid = isValid && ((offset1Shape != nullptr) || (offset2Shape != nullptr));
    if ((offset1Shape != nullptr) && (offset2Shape != nullptr)) {
        isValid =
            isValid && (offset1Shape->GetStorageShape().GetDimNum() == offset2Shape->GetStorageShape().GetDimNum());
    }
    isValid = isValid && (((offset1Shape != nullptr) && (offset1Shape->GetStorageShape().GetDimNum() <= DIM_TWO)) ||
                          ((offset2Shape != nullptr) && (offset2Shape->GetStorageShape().GetDimNum() <= DIM_TWO)));
    if (!isValid) {
        return false;
    }
    if (methodMode_ == 0) {
        if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == ROPE_LENGTH);
        } else if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(1) == ROPE_LENGTH);
        }

        if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == DIM_ONE);
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }
    } else {
        if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == RMS_NORM_LENGTH);
        } else if (offset1Shape != nullptr && offset1Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (offset1Shape->GetStorageShape().GetDim(1) == RMS_NORM_LENGTH);
        }

        if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_ONE) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == V_LENGTH);
        } else if (offset2Shape != nullptr && offset2Shape->GetStorageShape().GetDimNum() == DIM_TWO) {
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(0) == tilingData_.get_numHead());
            isValid = isValid && (offset2Shape->GetStorageShape().GetDim(1) == V_LENGTH);
        }
    }
    return isValid;
}

bool KvRmsNormRopeCacheTilingDs::IsCapable()
{
    return !isRegbase_;
}

void KvRmsNormRopeCacheTilingDs::DoOpTilingPaBlkNz()
{
    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();
    int64_t numHead = tilingData_.get_numHead();
    int64_t bns = batchSize * numHead * seqLen;
    int64_t blockFactor = (bns + coreNum_ - 1) / coreNum_;
    int64_t numBlocks = (bns + blockFactor - 1) / blockFactor;
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_numBlocks(numBlocks);

    int64_t maxUbFactor = (methodMode_ == 1) ? 32 : 16;
    constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
    if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
        tilingData_.set_ubFactor(maxUbFactor);
    } else {
        tilingData_.set_ubFactor(1);
    }
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::DoOpTiling()
{
    RMS_NORM_LENGTH = RMS_NORM_LENGTHS[methodMode_];
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    tilingData_.set_batchSize(std::get<SHAPE_IDX_B>(kvShapeTuple));
    tilingData_.set_numHead(std::get<SHAPE_IDX_N>(kvShapeTuple));
    tilingData_.set_seqLength(std::get<SHAPE_IDX_S>(kvShapeTuple));
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_methodMode(methodMode_);

    if (isOutputKv_) {
        tilingData_.set_isOutputKv(1);
    } else {
        tilingData_.set_isOutputKv(0);
    }

    if ((!isRegbase_) && (quantMode_ == QUANT_MODE)) {
        OP_CHECK_IF(!CheckScaleValid(context_), OP_LOGE(context_->GetNodeName(), "quant scale shape check failed."),
                    return ge::GRAPH_FAILED);
    }

    if (((!isRegbase_) && (quantMode_ > 1))) {
        OP_CHECK_IF(!CheckOffsetValid(context_), OP_LOGE(context_->GetNodeName(), "quant offset shape check failed."),
                    return ge::GRAPH_FAILED);
    }
    if ((!isRegbase_) && (dk_ != ROPE_LENGTH)) {
        auto cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
        std::string reasonMsg = "The D-dimension of input cos must be equal to " + std::to_string(ROPE_LENGTH) +
                                ", where D is the last axis of cos";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos", ToString(cosShape).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((!isRegbase_) && (dv_ != RMS_NORM_LENGTH)) {
        auto gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
        std::string reasonMsg = "The 0th axis of input gamma must be equal to " + std::to_string(RMS_NORM_LENGTH);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "gamma", ToString(gammaShape).c_str(),
                                              reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (methodMode_ == 0) {
        OP_CHECK_IF((!isRegbase_) && (currentCacheMode_ == CacheMode::Norm) && (quantMode_ == QUANT_MODE),
                    OP_LOGE(context_->GetNodeName(), "CacheMode::Norm do not support quant!"), return ge::GRAPH_FAILED);

        if ((!isRegbase_) && (kv_ != D_LENGTH)) {
            auto kvShape = context_->GetInputShape(KV_INDEX)->GetStorageShape();
            std::string reasonMsg = "The D-dimension of input kv must be equal to " + std::to_string(D_LENGTH) +
                                    ", where D is the last axis of kv";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv", ToString(kvShape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if ((!isRegbase_) && (vlen_ != V_LENGTH)) {
            auto vShape = context_->GetOptionalInputShape(V_IDX)->GetStorageShape();
            std::string reasonMsg = "The D-dimension of input v must be equal to " + std::to_string(V_LENGTH) +
                                    ", where D is the last axis of v";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "v", ToString(vShape).c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        OP_CHECK_IF((!isRegbase_) && (quantMode_ != NON_QUANT_MODE && quantMode_ != QUANT_MODE),
                    OP_LOGE(context_->GetNodeName(), "Only Support QUANT or NON_QUANT."), return ge::GRAPH_FAILED);
    }

    auto scale1Shape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
    if (scale1Shape != nullptr) {
        tilingData_.set_isKQuant(1);
    } else {
        tilingData_.set_isKQuant(0);
    }
    if (scale2Shape != nullptr) {
        tilingData_.set_isVQuant(1);
    } else {
        tilingData_.set_isVQuant(0);
    }

    if (currentCacheMode_ == CacheMode::PA && quantMode_ != NON_QUANT_MODE) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5011;
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_BNSD) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_5010;
        } else {
            tilingKey_ = TLING_KEY_5000;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4011;
        } else {
            tilingKey_ = TLING_KEY_4001;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (currentCacheMode_ == CacheMode::PA_BLK_NZ) {
        DoOpTilingPaBlkNz();
        if (quantMode_ != NON_QUANT_MODE) {
            tilingKey_ = TLING_KEY_4010;
        } else {
            tilingKey_ = TLING_KEY_4000;
        }
        return ge::GRAPH_SUCCESS;
    }

    int8_t outputKvValue = tilingData_.get_isOutputKv();
    bool outputKv = outputKvValue == 0 ? false : true;
    if (outputKv && currentCacheMode_ == CacheMode::PA) {
        DoOpTilingPaBlkNz();
        tilingKey_ = TLING_KEY_5001;
        return ge::GRAPH_SUCCESS;
    }

    int64_t batchSize = tilingData_.get_batchSize();
    int64_t seqLen = tilingData_.get_seqLength();
    int64_t numHead = tilingData_.get_numHead();

    if (IsB1SD(context_)) {
        int64_t bns = batchSize * numHead * seqLen;
        int64_t blockFactor = (bns + coreNum_ - 1) / coreNum_;
        int64_t numBlocks = (bns + blockFactor - 1) / blockFactor;
        tilingData_.set_blockFactor(blockFactor);
        tilingData_.set_numBlocks(numBlocks);

        int64_t maxUbFactor = (methodMode_ == 1) ? 32 : 16;
        constexpr static int64_t needUbSize = static_cast<int64_t>(170) * static_cast<int64_t>(1024);
        if (static_cast<int64_t>(ubSize_) >= static_cast<int64_t>(needUbSize)) {
            tilingData_.set_ubFactor(maxUbFactor);
        } else {
            tilingData_.set_ubFactor(1);
        }
    } else {
        if (batchSize % BATCHES_FOR_EACH_CORE == 0) {
            tilingData_.set_numBlocks(batchSize / BATCHES_FOR_EACH_CORE);
            tilingData_.set_rowsPerBlock(BATCHES_FOR_EACH_CORE);
        } else {
            tilingData_.set_numBlocks(batchSize);
            tilingData_.set_rowsPerBlock(1);
        }
        // Check numBlocks <= MAX_BLOCK_DIM
        OP_CHECK_IF(tilingData_.get_numBlocks() > MAX_BLOCK_DIM,
                    OP_LOGE(context_->GetNodeName(), "numBlocks must be smaller than 65535."), return ge::GRAPH_FAILED);
    }

    if (methodMode_ == 1) {
        if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
            tilingData_.set_isOutputKv(0);
        }
        if (IsB1SD(context_)) {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_3001;
            } else if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_3000;
            } else {
                tilingKey_ = TLING_KEY_3010;
            }
        } else {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_2000;
            } else if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_1000;
            } else if (!isPagedAttention_ && quantMode_ > NON_QUANT_MODE) {
                tilingKey_ = TLING_KEY_1010;
            }
            if (isMTP_) {
                tilingKey_ += 1;
            }
        }
    } else {
        if (IsB1SD(context_)) {
            if (!isPagedAttention_ && quantMode_ == NON_QUANT_MODE) {
                tilingData_.set_isOutputKv(0);
            }
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_3001;
            } else {
                tilingKey_ = TLING_KEY_3000;
            }
        } else {
            if (isPagedAttention_) {
                tilingKey_ = TLING_KEY_2000;
            } else {
                tilingKey_ = TLING_KEY_1000;
            }
            if (isMTP_) {
                tilingKey_ += 1;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheTilingDs::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_numBlocks());
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(KvRmsNormRopeCache, KvRmsNormRopeCacheTilingDs, TEMPLATE_DS_PRIORITY);
} // namespace optiling
