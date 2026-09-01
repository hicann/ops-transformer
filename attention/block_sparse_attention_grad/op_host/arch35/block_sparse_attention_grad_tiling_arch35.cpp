/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../block_sparse_attention_grad_tiling.h"

namespace optiling {
namespace BSA_ARC35 {

int64_t AlignTo(int64_t x, int64_t align)
{
    return (x + align - 1) / align * align;
}

template <typename... Args>
bool IsSameShape(Args... shapes)
{
    std::array<int32_t, sizeof...(Args)> arr = {shapes...};
    if (arr.size() <= 1)
        return true;
    int32_t first = arr[0];
    if (first <= 0) {
        return false;
    }

    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i] != first)
            return false;
    }
    return true;
}

ge::graphStatus InputCheck(const std::vector<int32_t> &q_shape, const std::vector<int32_t> &k_shape,
                           const std::vector<int32_t> &v_shape, const int32_t q_group, gert::TilingContext *context)
{
    /* function: 根据固定格式传入shape检查输入是否符合要求
     * shape：[b, n, s, d]
     */
    if (!IsSameShape(q_shape[0], k_shape[0], v_shape[0])) {
        OP_LOGE(context->GetNodeName(), "q_batch must be same as k_batch and v_batch.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[1], k_shape[1] * q_group, v_shape[1] * q_group)) {
        OP_LOGE(context->GetNodeName(), "q_group * kv_head_num must same as q_head_num.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[2], k_shape[2], v_shape[2])) {
        OP_LOGE(context->GetNodeName(), "q_seq_len must be same as k_seq_len and v_seq_len.");
        return ge::GRAPH_FAILED;
    } else if (!IsSameShape(q_shape[3], k_shape[3], v_shape[3])) {
        OP_LOGE(context->GetNodeName(), "q_head_dim must be same as k_head_dim and v_head_dim.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

class BlockSparseAttentionGradArch35Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit BlockSparseAttentionGradArch35Tiling(gert::TilingContext *context_) : TilingBaseClass(context_)
    {
    }


protected:
    bool IsCapable() override
    {
        if (context_->GetDeterministic() == 1) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad not support Deterministic.");
            return false;
        }

        if (dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support DT_FLOAT16 and DT_BF16.");
            return false;
        }

        if (head_dim_ != 128) {
            OP_LOGE(context_->GetNodeName(), "BlockSparseAttentionGrad only support head_dim 128.");
            return false;
        }

        if (blockShapeX_ % 16 != 0 || blockShapeY_ % 16 != 0) {
            OP_LOGE(context_->GetNodeName(),
                    "BlockSparseAttentionGrad only supports blockShapeX and blockShapeY that are multiples of 16.");
            return false;
        }

        return true;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        int32_t q_batch_num, q_group, q_head_num, q_seq_len, q_head_dim;
        int32_t k_batch_num, k_head_num, k_seq_len, k_head_dim;
        int32_t v_batch_num, v_head_num, v_seq_len, v_head_dim;
        auto qInputDesc = context_->GetInputDesc(1);
        const gert::StorageShape *queryShape = context_->GetInputShape(1);
        const gert::StorageShape *keyShape = context_->GetInputShape(2);
        const gert::StorageShape *valueShape = context_->GetInputShape(3);
        const char *qInputLayout = context_->GetAttrs()->GetAttrPointer<char>(0);
        const char *kvInputLayout = context_->GetAttrs()->GetAttrPointer<char>(1);
        const float softmaxScale = *context_->GetAttrs()->GetAttrPointer<float>(4);
        auto blockShapeOptional = context_->GetInputTensor(8);
        if (blockShapeOptional == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }
        const int64_t *blockShapeList = blockShapeOptional->GetData<int64_t>();
        if (blockShapeList == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Block shape tensor is null");
            return ge::GRAPH_FAILED;
        }

        if (qInputDesc == nullptr) {
            OP_LOGE(context_->GetNodeName(), "Query inputDesc is null");
            return ge::GRAPH_FAILED;
        } else {
            dataType_ = qInputDesc->GetDataType();
        }

        if (strcmp(qInputLayout, kvInputLayout) != 0) {
            OP_LOGE(context_->GetNodeName(), "kvInputLayout must same as qInputLayout.");
            return ge::GRAPH_FAILED;
        }

        if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr) {
            OP_LOGE(context_->GetNodeName(), "queryShape or keyShape or valueShape is nullptr.");
            return ge::GRAPH_FAILED;
        }

        if (strcmp(qInputLayout, BSND_STR) == 0) {
            // BSND layout
            if (queryShape->GetOriginShape().GetDimNum() != 4) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 4 dimension.");
                return ge::GRAPH_FAILED;
            }
            // query
            q_batch_num = queryShape->GetStorageShape().GetDim(0);
            q_seq_len = queryShape->GetStorageShape().GetDim(1);
            q_head_num = queryShape->GetStorageShape().GetDim(2);
            q_head_dim = queryShape->GetStorageShape().GetDim(3);
            // key
            k_batch_num = keyShape->GetStorageShape().GetDim(0);
            k_seq_len = keyShape->GetStorageShape().GetDim(1);
            k_head_num = keyShape->GetStorageShape().GetDim(2);
            k_head_dim = keyShape->GetStorageShape().GetDim(3);
            // value
            v_batch_num = valueShape->GetStorageShape().GetDim(0);
            v_seq_len = valueShape->GetStorageShape().GetDim(1);
            v_head_num = valueShape->GetStorageShape().GetDim(2);
            v_head_dim = valueShape->GetStorageShape().GetDim(3);
            q_group = q_head_num / k_head_num;
        } else if (strcmp(qInputLayout, BNSD_STR) == 0) {
            // BNSD layout
            if (queryShape->GetOriginShape().GetDimNum() != 4) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 4 dimension.");
                return ge::GRAPH_FAILED;
            }
            // query
            q_batch_num = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            q_seq_len = queryShape->GetStorageShape().GetDim(2);
            q_head_dim = queryShape->GetStorageShape().GetDim(3);
            // key
            k_batch_num = keyShape->GetStorageShape().GetDim(0);
            k_head_num = keyShape->GetStorageShape().GetDim(1);
            k_seq_len = keyShape->GetStorageShape().GetDim(2);
            k_head_dim = keyShape->GetStorageShape().GetDim(3);
            // value
            v_batch_num = valueShape->GetStorageShape().GetDim(0);
            v_head_num = valueShape->GetStorageShape().GetDim(1);
            v_seq_len = valueShape->GetStorageShape().GetDim(2);
            v_head_dim = valueShape->GetStorageShape().GetDim(3);
            q_group = q_head_num / k_head_num;
        } else if (strcmp(qInputLayout, TND_STR) == 0) {
            // TND layout
            if (queryShape->GetOriginShape().GetDimNum() != 3) {
                OP_LOGE(context_->GetNodeName(), "The shape of query must be 3 dimension.");
                return ge::GRAPH_FAILED;
            }
            auto actualSeqLengths = context_->GetOptionalInputTensor(9);
            auto actualKVSeqLengths = context_->GetOptionalInputTensor(10);
            if (actualSeqLengths == nullptr || actualKVSeqLengths == nullptr) {
                OP_LOGE(context_->GetNodeName(),
                        "The layout is TND, actualSeqLengths or actualKVSeqLengths must not be nullptr.");
                return ge::GRAPH_FAILED;
            }
            q_batch_num = static_cast<uint32_t>(actualSeqLengths->GetShapeSize());
            k_batch_num = static_cast<uint32_t>(actualKVSeqLengths->GetShapeSize());
            v_batch_num = k_batch_num;
            // query
            q_seq_len = queryShape->GetStorageShape().GetDim(0);
            q_head_num = queryShape->GetStorageShape().GetDim(1);
            q_head_dim = queryShape->GetStorageShape().GetDim(2);
            // key
            k_seq_len = keyShape->GetStorageShape().GetDim(0);
            k_head_num = keyShape->GetStorageShape().GetDim(1);
            k_head_dim = keyShape->GetStorageShape().GetDim(2);
            // value
            v_seq_len = valueShape->GetStorageShape().GetDim(0);
            v_head_num = valueShape->GetStorageShape().GetDim(1);
            v_head_dim = valueShape->GetStorageShape().GetDim(2);
            q_group = q_head_num / k_head_num;
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSND or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_batchNum(q_batch_num);
        tilingData_.set_qSeqLen(q_seq_len);  // TND时含义为total_seq_len
        tilingData_.set_kvSeqLen(k_seq_len); // TND时含义为total_seq_len
        tilingData_.set_qGroup(q_group);
        tilingData_.set_qHeadNum(q_head_num);
        tilingData_.set_kvHeadNum(k_head_num);
        tilingData_.set_headDim(q_head_dim);
        tilingData_.set_softmaxScale(softmaxScale);
        batch_num_ = q_batch_num;
        q_seq_len_ = q_seq_len;
        kv_seq_len_ = k_seq_len;
        q_head_num_ = q_head_num;
        q_group_ = q_group;
        kv_head_num_ = k_head_num;
        head_dim_ = q_head_dim;
        layout_ = qInputLayout;
        blockShapeX_ = blockShapeList[0];
        blockShapeY_ = blockShapeList[1];

        return InputCheck({q_batch_num, q_head_num, q_seq_len, q_head_dim},
                          {k_batch_num, k_head_num, k_seq_len, k_head_dim},
                          {v_batch_num, v_head_num, v_seq_len, v_head_dim}, q_group, context_);
    }

    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoOpTiling() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        auto cubeCoreNum = ascendcPlatform.GetCoreNumAic();

        uint32_t baseM = blockShapeX_ <= 128 ? blockShapeX_ : 128;
        uint32_t baseN = blockShapeY_ <= 128 ? blockShapeY_ : 128;
        uint32_t singleM = AlignTo(2048, baseM);
        // Keep all Q tiles for the current KV block/head on one cube core when
        // possible so DK/DV can accumulate in L0 and K/V stays cached in L1.

        tilingData_.set_BlockX(blockShapeX_);
        tilingData_.set_BlockY(blockShapeY_);
        tilingData_.set_cubeCoreNum(cubeCoreNum);
        tilingData_.set_baseM(baseM);
        tilingData_.set_baseN(baseN);
        tilingData_.set_singleM(singleM);
        context_->SetBlockDim(cubeCoreNum);
        context_->SetScheduleMode(1);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        // softmaxGrad tiling，一次最多处理16K个元素个数
        constexpr uint32_t element_num = 8 * 1024;
        uint32_t process_s1_size = element_num / head_dim_;
        auto input_shape = ge::Shape({process_s1_size, head_dim_});
        uint32_t tmp_space_size = AscendC::GetSoftMaxGradMaxTmpSize(input_shape, sizeof(float), true, true);
        AscendC::SoftMaxGradTilingFunc(input_shape, sizeof(float), tmp_space_size,
                                       tilingData_.softmaxGradFrontTilingData, true);
        tilingData_.set_sftgTmpSpaceSize(tmp_space_size); // 元素个数=8K时需要50K的临时空间
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        uint64_t dq_size, dk_size, dv_size, sftg_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t sys_workspace_size = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t usr_workspace_offset = 0;
        if (strcmp(layout_, BSND_STR) == 0 || strcmp(layout_, BNSD_STR) == 0) {
            dq_size = batch_num_ * q_seq_len_ * q_head_num_ * head_dim_;
            dk_size = batch_num_ * kv_seq_len_ * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = batch_num_ * q_head_num_ * q_seq_len_ * 8;
        } else if (strcmp(layout_, TND_STR) == 0) {
            dq_size = q_seq_len_ * q_head_num_ * head_dim_;
            dk_size = kv_seq_len_ * kv_head_num_ * head_dim_;
            dv_size = dk_size;
            sftg_size = q_seq_len_ * q_head_num_ * 8; // 存放逻辑按[B,N,S,8]
        } else {
            OP_LOGE(context_->GetNodeName(), "Invaild Input Layout: InputLayout must be BSDN or BNSD or TND.");
            return ge::GRAPH_FAILED;
        }

        tilingData_.set_dqSize(dq_size);
        tilingData_.set_dkSize(dk_size);
        tilingData_.set_sftgWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += sftg_size * sizeof(float);
        tilingData_.set_dqWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dq_size * sizeof(float);
        tilingData_.set_dkWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dk_size * sizeof(float);
        tilingData_.set_dvWorkspaceOffset(usr_workspace_offset);
        usr_workspace_offset += dv_size * sizeof(float);

        context_->GetWorkspaceSizes(1)[0] = sys_workspace_size + usr_workspace_offset;
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
        uint64_t default_key = 1000;
        /*
         *  000: BF16 BSND
         *  001: FP16 BSND
         *  010: BF16 BNSD
         *  011: FP16 BNSD
         *  100: BF16 TND
         *  101: FP16 TND
         */
        default_key = (dataType_ == ge::DT_BF16) ? default_key : default_key + 0b001;

        if (strcmp(layout_, BSND_STR) == 0) {
            default_key += 0b000;
        } else if (strcmp(layout_, BNSD_STR) == 0) {
            default_key += 0b010;
        } else if (strcmp(layout_, TND_STR) == 0) {
            default_key += 0b100;
        }

        return default_key;
    }

private:
    const char *layout_;
    ge::DataType dataType_;
    uint64_t tilingKey_ = 1000; // arc35 默认tilingkey从1000开始
    BlockSparseAttentionGradTilingDataArch35 tilingData_;
    static constexpr const char *BSND_STR = "BSND";
    static constexpr const char *BNSD_STR = "BNSD";
    static constexpr const char *TND_STR = "TND";
    int32_t batch_num_ = 0;
    int32_t q_seq_len_ = 0;
    int32_t kv_seq_len_ = 0;
    int32_t q_head_num_ = 0;
    int32_t q_group_ = 0;
    int32_t kv_head_num_ = 0;
    int32_t head_dim_ = 0;
    int32_t blockShapeX_ = 0;
    int32_t blockShapeY_ = 0;
    int32_t max_q_seq_len_ = 0;
    int32_t max_kv_seq_len_ = 0;
};


REGISTER_TILING_TEMPLATE_WITH_ARCH(BlockSparseAttentionGrad, BlockSparseAttentionGradArch35Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 1);
} // namespace BSA_ARC35
} // namespace optiling
