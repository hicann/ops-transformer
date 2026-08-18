/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_score_tiling_regbase_host.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
namespace FA {
class FlashAttentionScoreTilingBasic : public FlashAttentionScoreTilingRegbase {
public:
    explicit FlashAttentionScoreTilingBasic(gert::TilingContext *context) :
        FlashAttentionScoreTilingRegbase(context)
    {
        this->templateName = "S1S2Const";
        this->regbase = true;
    }
    ~FlashAttentionScoreTilingBasic() override = default;

protected:
    STemplateType s1TemplateType = STemplateType::STEMPLATEBOTTOM;
    STemplateType s2TemplateType = STemplateType::STEMPLATEBOTTOM;

    ge::graphStatus CheckContext() override {
        FlashAttentionScoreTilingRegbase::CheckContext();
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcTotalSize() override {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        if (totalSize < aicNum && implMode != ImplMode::AA_INVALID_LINE_HIGH_PRECISION && !hasRope &&
            inputDtypeBytes != DATA_TYPE_FP32 && inputDtypeBytes != DATA_TYPE_FP8 && dBasicBlock <= NUM_256) {
            if (s2Size > NUM_1024 && !hasAttenMask && !hasPse && !hasDropOut) {
                s2TemplateType = STemplateType::ALIGNED_256;
                s2BasicBlock = NUM_256;
            }
            s1TemplateType = STemplateType::ALIGNED_64;
            s1BasicBlock = NUM_64;
            multiCoreParamsRegbase_->set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
            totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        }
        return totalSize;
    }

    void CalcDBasicBlock() override {
        /* 先确定D的基本块，确定的逻辑是按照64来分档 */
        dBasicBlock = AlignUp(dSize + dSizeRope, D_TEMPLATE_SPLIT_SIZE);
        /* dBasicBlock > 256 直接令D轴大小为768 */
        if (dBasicBlock > NUM_256) {
            dTemplateType = DTemplateType::ALIGNED_768;
            return ; // 直接返回不往下走
        }
        /* dBasicBlock <= 256, 根据对齐大小选择*/
        switch (dBasicBlock) {
            case NUM_64:
                dTemplateType = DTemplateType::ALIGNED_64;
                break;
            case NUM_128:
                dTemplateType = DTemplateType::ALIGNED_128;
                break;
            case NUM_192:
                dTemplateType = DTemplateType::ALIGNED_192;
                break;
            case NUM_256:
                dTemplateType = DTemplateType::ALIGNED_256;
                break;
            default:
                dTemplateType = DTemplateType::DTEMPLATEBOTTOM;
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key",
                    std::to_string(dBasicBlock).c_str(),
                    "The value of dBasicBlock must be in range (0, 768]");
        }
    }

    void CalcS1S2BasicBlock() override
    {
        /* s2 = 64 && d == 64使能dn优化 */
        
        if ((inputDtype == ge::DT_HIFLOAT8) && !hasAttenMask && !hasPse && !hasDropOut && !hasRope) {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_512;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_512;
        } else if ((dSize == DN_D_64 && dSizeV == DN_D_64 &&
            (s1Size % DN_S1_128 == 0) &&
            (s2Size % MIN_DN_S2 == 0) &&
            !hasAttenMask && !hasPse && !hasDropOut && (inputDtypeBytes != DATA_TYPE_FP32) && 
            (inputDtypeBytes != DATA_TYPE_FP8) && !hasRope) ||
            ((inputDtype == ge::DT_FLOAT8_E5M2 || inputDtype == ge::DT_FLOAT8_E4M3FN) &&
              !hasAttenMask && !hasPse && !hasDropOut && !hasRope)) {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_256;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_256;
        } else if (dSize > NUM_256) {
            if (inputDtypeBytes == DATA_TYPE_FP32) {
                s1TemplateType = STemplateType::ALIGNED_64;
                s1BasicBlock = NUM_64;
            } else {
                s1TemplateType = STemplateType::ALIGNED_128;
                s1BasicBlock = NUM_128;
            }
            s2TemplateType = STemplateType::ALIGNED_128;
            s2BasicBlock = NUM_128;
        } else {
            s1TemplateType = STemplateType::ALIGNED_128;
            s2TemplateType = STemplateType::ALIGNED_128;
            s1BasicBlock = NUM_128;
            s2BasicBlock = NUM_128;
        }
    }

    bool SetPseAlibiParamsRegbase() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            return true;
        }
        // 2: pre last axiss
        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PseEncodeType::PSE_ENCODE_NONE;
        if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE) {
            if (s1Size == s2Size) {
                if (inputParamsRegbase_->get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL)) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparse_type",
                        std::to_string(static_cast<int64_t>(inputParamsRegbase_->get_sparseType())).c_str(),
                        "The value of sparse_type must be CAUSAL when pse alibi is used with "
                        "s1Size > 1024 and s1Size == s2Size");
                    return false;
                }
                pseEncodeType = PseEncodeType::PSE_ENCODE_ALIBI_S2_FULL;
            } else {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName,
                    "s1Size and s2Size", (std::to_string(s1Size) + " and " + std::to_string(s2Size)).c_str(),
                    "The values of s1Size and s2Size must be the same "
                    "when pse alibi is used with s1Size > 1024");
                return false;
            }
        }
        inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(pseEncodeType));
        inputParamsRegbase_->set_pseS1Size(pseS1Size);
        inputParamsRegbase_->set_pseS2Size(pseS2Size);
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        uint8_t pseMode = hasPse ? static_cast<uint8_t>(pseType) : static_cast<uint8_t>(PseType::PSE_NONE_TYPE);
        OP_LOGD(opName, "TilingKey info is implMode:%d, s1TemplateType:%d, s2TemplateType:%d, dTemplateType:%d,"
            "dVTemplateType:%d, pseMode:%d, hasAttenMask:%d, hasDropOut:%d, hasRope:%d, outDtype:%d, regbase:%d,"
            "optionalDn:%d", static_cast<uint8_t>(implMode), static_cast<uint16_t>(s1TemplateType),
            static_cast<uint16_t>(s2TemplateType), static_cast<uint16_t>(dTemplateType),
            static_cast<uint16_t>(dVTemplateType), pseMode, hasAttenMask, hasDropOut, hasRope,
            static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase), optionalDn);

        // Const 128
        if (dTemplateType == dVTemplateType) {
            return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
                static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
                static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(DTemplateType::NONALIGNED),
                pseMode, hasAttenMask, hasDropOut, hasRope, static_cast<uint8_t>(outDtype),
                static_cast<uint8_t>(regbase), optionalDn);
        }
        return GET_TPL_TILING_KEY(0, static_cast<uint8_t>(implMode), static_cast<uint8_t>(tilingKeyLayout),
            static_cast<uint16_t>(s1TemplateType), static_cast<uint16_t>(s2TemplateType),
            static_cast<uint16_t>(dTemplateType), static_cast<uint16_t>(dVTemplateType), pseMode, hasAttenMask,
            hasDropOut, hasRope, static_cast<uint8_t>(outDtype), static_cast<uint8_t>(regbase), optionalDn);
    }

    void AnalyzeOptionalDn() override
    {
        if ((hasAttenMask && attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)
            && attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) ||
            !hasAttenMask || hasPse || hasRope || hasDropOut || inputDtypeBytes == DATA_TYPE_FP32 ||
            inputDtypeBytes == DATA_TYPE_FP8 || dTemplateType > DTemplateType::ALIGNED_256 ||
            dTemplateType != dVTemplateType || s1TemplateType == STemplateType::ALIGNED_64 ||
            s1Size < NUM_1536 || s2Size < NUM_1536) {
            return;
        }
        optionalDn = true;
    }
    
    bool IsCapable() override
    {
        if (npuArch != NpuArch::DAV_3510) {
            OP_LOGD(opName, "Current npu arch is not dav-3510.");
            return false;
        }

        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // 当前只有当D较大时需要开启workspace存放Bmm2的数据
        int64_t bmm2Bytes = 0;
        int64_t vec2Bytes = 0;
        int64_t bmm2ResBlockSize = dVBasicBlock;
        if (dTemplateType > DTemplateType::ALIGNED_256) {
            bmm2ResBlockSize = static_cast<int64_t>(dVTemplateType);
        }
        bool useDn = (!hasPse && !hasAttenMask && !hasDropOut && s1BasicBlock != NUM_64
                      && dVBasicBlock <= NUM_256 && !hasRope);
        if ((!useDn && dSize > MIN_D_TO_USE_WORKSPACE) ||
            (useDn && dSize > DN_MIN_D_TO_USE_WORKSPACE)) {
            bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
            if (dTemplateType > DTemplateType::ALIGNED_256) {
                vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
            }
        }
        bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
        vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
        workspaces[0] = static_cast<size_t>((bmm2Bytes + vec2Bytes) * PING_PONG_VALUE *
            multiCoreParamsRegbase_->get_coreNum());
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttentionScoreTilingRegbase::PostTiling();
        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingBasic, static_cast<int32_t>(NpuArch::DAV_3510), 83);
} // namespace FA
} // namespace optiling
