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
class FlashAttentionScoreTilingVarLen : public FlashAttentionScoreTilingRegbase {
public:
    explicit FlashAttentionScoreTilingVarLen(gert::TilingContext *context)
        : FlashAttentionScoreTilingRegbase(context)
    {
        this->templateName = "VarLenConst";
        this->regbase = true;
    }
    ~FlashAttentionScoreTilingVarLen() override = default;

protected:
    int64_t s2SizeLimitMax = 128;

    STemplateType s1TemplateType = STemplateType::STEMPLATEBOTTOM;
    STemplateType s2TemplateType = STemplateType::STEMPLATEBOTTOM;

    ge::graphStatus CheckContext() override
    {
        FlashAttentionScoreTilingRegbase::CheckContext();
        return ge::GRAPH_SUCCESS;
    }

    int64_t CalcTotalSize() override
    {
        int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
        if (totalSize < static_cast<int64_t>(aicNum) && implMode != ImplMode::AA_INVALID_LINE_HIGH_PRECISION &&
            inputDtypeBytes != DATA_TYPE_FP32 && dBasicBlock <= NUM_256 && !hasRope) {
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

    void CalcDBasicBlock() override
    {
        /* 先确定D的基本块，确定的逻辑是按照64来分档 */
        dBasicBlock = AlignUp(dSize + dSizeRope, D_TEMPLATE_SPLIT_SIZE);
        /* dBasicBlock > 256 直接令D轴大小为768 */
        if (dBasicBlock > NUM_256) {
            dTemplateType = DTemplateType::ALIGNED_768;
            return; // 直接返回不往下走
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
                                                      "The value of dBasicBlock must be within range (0, 768]");
        }
    }

    void CalcS1S2BasicBlock() override
    {
        /* 无可选输入使能dn优化 */
        if (dSize > NUM_256) {
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

    bool SetBandLeftUpCausalPseParamsRegbase()
    {
        for (int64_t i = 0L; i < bSize; ++i) {
            if (i == 0) {
                if (actualSeqLenData[0] - actualSeqLenKvData[0] + qStartIdx - kvStartIdx == 0) {
                    continue;
                } else {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),
                                                           "actual_seq_qlen, q_start_idx, actual_seq_kvlen and kv_start_idx",
                                                           (std::to_string(actualSeqLenData[0]) + ", " + std::to_string(qStartIdx) + ", " +
                                                            std::to_string(actualSeqLenKvData[0]) + " and " + std::to_string(kvStartIdx))
                                                               .c_str(),
                                                           "The following constraint must be met: "
                                                           "actualSeqQLen[0] - actualSeqKvLen[0] + q_start_idx - kv_start_idx == 0,"
                                                           "when the scenario is inner pse and sparse mode is 8");
                    return false;
                }
            }
            if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),
                                                       "actual_seq_qlen and actual_seq_kvlen",
                                                       (std::to_string(actualSeqLenData[i]) + " and " + std::to_string(actualSeqLenKvData[i])).c_str(),
                                                       "The following constraint must be met: actual_seq_qlen == actual_seq_kvlen, "
                                                       "when the scenario is inner pse and sparse mode is 8");
                return false;
            }
        }
        return true;
    }

    bool SetPseAlibiParamsRegbase() override
    {
        auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
        if (pseShape == nullptr) {
            return true;
        }
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL_BAND)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparse_type",
                                                      std::to_string(static_cast<int64_t>(inputParamsRegbase_->get_sparseType())).c_str(),
                                                      "The value of sparse_type cannot be RIGHT_DOWN_CAUSAL_BAND when pseType is 2 or 3");
                return false;
            }
            if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::BAND_LEFT_UP_CAUSAL)) {
                if (!SetBandLeftUpCausalPseParamsRegbase()) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pse",
                                                          std::to_string(static_cast<int64_t>(inputParamsRegbase_->get_sparseType())).c_str(),
                                                          "The parameters of input pse should be valid when sparse_mode is BAND_LEFT_UP_CAUSAL "
                                                          "and pseType is 2 or 3");
                    return false;
                }
                return true;
            }
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),
                                                           "actual_seq_qlen and actual_seq_kvlen",
                                                           (std::to_string(actualSeqLenData[i]) + " and " +
                                                            std::to_string(actualSeqLenKvData[i]))
                                                               .c_str(),
                                                           "The following constraint must be met: actual_seq_qlen == actual_seq_kvlen, "
                                                           "when the scenario is inner pse alibi");
                    return false;
                }
            }
            return true;
        }

        auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - DIM_NUM_2);
        auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);

        PseEncodeType pseEncodeType = PseEncodeType::PSE_ENCODE_NONE;
        OP_LOGD(context_, "[%s] pseS1Size:%ld, pseS2Size:%ld.", templateName, pseS1Size, pseS2Size);
        if (pseS1Size == PSE_ALIBI_S_SIZE) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] != actualSeqLenKvData[i]) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(),
                                                           "actual_seq_qlen and actual_seq_kvlen",
                                                           (std::to_string(actualSeqLenData[i]) + " and " +
                                                            std::to_string(actualSeqLenKvData[i]))
                                                               .c_str(),
                                                           "The following constraint must be met: actual_seq_qlen == actual_seq_kvlen "
                                                           "when the scenario is pse alibi");
                    return false;
                }
            }
            if (inputParamsRegbase_->get_sparseType() != static_cast<uint8_t>(SparseEnum::CAUSAL) &&
                inputParamsRegbase_->get_sparseType() !=
                    static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparse_type",
                                                      std::to_string(static_cast<int64_t>(inputParamsRegbase_->get_sparseType())).c_str(),
                                                      "The value of sparse_type must be CAUSAL or RIGHT_DOWN_CAUSAL when pse alibi is used");
                return false;
            }
            pseEncodeType = PseEncodeType::PSE_ENCODE_ALIBI_S2_FULL;
            OP_LOGD(context_, "[%s] PSE_ENCODE_ALIBI_S2_FULL.", templateName);
        }
        inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(pseEncodeType));
        inputParamsRegbase_->set_pseS1Size(pseS1Size);
        inputParamsRegbase_->set_pseS2Size(pseS2Size);
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        uint8_t pseMode = hasPse ? static_cast<uint8_t>(pseType) : static_cast<uint8_t>(PseType::PSE_NONE_TYPE);
        OP_LOGD(opName, "TND TilingKey info is implMode:%d, s1TemplateType:%d, s2TemplateType:%d, dTemplateType:%d,"
                        "pseMode:%d, hasAttenMask:%d, hasDropOut:%d, hasRope:%d, regbase:%d, optionalDn:%d",
                static_cast<uint8_t>(implMode), static_cast<uint16_t>(s1TemplateType),
                static_cast<uint16_t>(s2TemplateType), static_cast<uint16_t>(dTemplateType), pseMode, hasAttenMask,
                hasDropOut, hasRope, static_cast<uint8_t>(regbase), optionalDn);

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
                                  hasDropOut, hasRope, 0, static_cast<uint8_t>(regbase), optionalDn);
    }

    void AnalyzeOptionalDn() override
    {
        if ((hasAttenMask && attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) && attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) ||
            !hasAttenMask || hasPse || hasRope || hasDropOut || inputDtypeBytes == DATA_TYPE_FP32 ||
            inputDtypeBytes == DATA_TYPE_FP8 || dTemplateType > DTemplateType::ALIGNED_256 ||
            dTemplateType != dVTemplateType || s1TemplateType == STemplateType::ALIGNED_64) {
            return;
        }
        int64_t s1TotalLen = 0;
        int64_t s2TotalLen = 0;
        for (int64_t i = 0; i < bSize; ++i) {
            s1TotalLen += actualSeqLenData[i];
            s2TotalLen += actualSeqLenKvData[i];
        }

        int64_t s1AvgLen = s1TotalLen / bSize;
        int64_t s2AvgLen = s2TotalLen / bSize;
        if (s1AvgLen < NUM_1536 || s2AvgLen < NUM_1536) {
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
        if (tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return false;
        }
        return true;
    }

    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum) override
    {
        (void)coreNum;
        int64_t accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; ++i) {
            if (actualSeqLenKvData[i] == 0) {
                continue;
            }
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        totalSize = accumS1BlockNum * n2Size * gSize;
        // 全部batch的KV长度均为0时, totalSize为0, 保留1个no-op核避免kernel侧totalSize/(coreNum*2)除0,
        // 其余多核参数置0, kernel空跑后安全返回
        if (totalSize == 0) {
            multiCoreParamsRegbase_->set_coreNum(1);
            multiCoreParamsRegbase_->set_totalSize(0);
            multiCoreParamsRegbase_->set_splitFactorSize(0);
            multiCoreParamsRegbase_->set_splitFactorTailSize(0);
            return;
        }
        int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(aicNum));
        multiCoreParamsRegbase_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
        multiCoreParamsRegbase_->set_totalSize(totalSize);
        multiCoreParamsRegbase_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
        multiCoreParamsRegbase_->set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParamsRegbase_->get_splitFactorSize()));
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        // 当前只有当D较大时需要开启workspace存放Bmm2的数据
        int64_t bmm2Bytes = 0;
        int64_t vec2Bytes = 0;
        int64_t bmm2ResBlockSize = dBasicBlock;
        if (dTemplateType > DTemplateType::ALIGNED_256) {
            bmm2ResBlockSize = static_cast<int64_t>(dVTemplateType);
        }
        if (dSize > MIN_D_TO_USE_WORKSPACE) {
            bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
            if (dTemplateType > DTemplateType::ALIGNED_256) {
                vec2Bytes = s1BasicBlock * dBasicBlock * calcTypeSize;
            }
        }
        bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
        vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
        workspaces[0] = static_cast<size_t>(((bmm2Bytes + vec2Bytes) * PING_PONG_VALUE) *
                                            multiCoreParamsRegbase_->get_coreNum());
        return ge::GRAPH_SUCCESS;
    }

    bool CheckBandPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (preTokens < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pre_tokens",
                                                  std::to_string(preTokens).c_str(), "The value of pre_tokens must be greater than or equal to zero");
            return false;
        }
        if (nextTokens < 0 && preTokens + nextTokens < 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "pre_tokens and next_tokens",
                                                   (std::to_string(preTokens) + " and " + std::to_string(nextTokens)).c_str(),
                                                   "The following constraint must be met: next_tokens >= 0 or pre_tokens + next_tokens >= 0");
            return false;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i]) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "next_tokens",
                                                      std::to_string(nextTokens).c_str(),
                                                      "The following constraint must be met for each batch: "
                                                      "actual_seq_len_q - next_tokens <= actual_seq_len_kv");
                return false;
            }
        }
        if (preTokens >= s2Size && nextTokens == 0) {
            preTokens = s2Size;
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else {
            sparseType = SparseEnum::BAND_COMPRESS;
        }
        return true;
    }

    bool CheckRightDownCausalBandPretokenAndNexttoken(SparseEnum &sparseType)
    {
        int64_t lastS2 = actualSeqLenKvData[bandIndex];
        if (preTokens < lastS2 || nextTokens > 0) {
            std::string paramMsg = "pre_tokens and next_tokens";
            std::string valMsg = std::to_string(preTokens) + " and " + std::to_string(nextTokens);
            std::string reasonMsg = "In RightDownCausal_Band mode, pre_tokens must not "
                                    "be smaller than last valid s2(" +
                                    std::to_string(lastS2) + ") "
                                                             "and next_tokens must not be greater than 0";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramMsg.c_str(),
                                                   valMsg.c_str(), reasonMsg.c_str());
            return false;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                std::string shapeMsg = std::to_string(actualSeqLenData[i]) + " and " +
                                       std::to_string(actualSeqLenKvData[i]);
                std::string reasonMsg = "The value of actual_seq_qlen of batch " + std::to_string(i) +
                                        " must not be greater than actual_seq_kvlen";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "actual_seq_qlen and actual_seq_kvlen",
                                                       shapeMsg.c_str(), reasonMsg.c_str());
                return false;
            }
            if ((i == bandIndex) && (actualSeqLenData[i] - nextTokens > actualSeqLenKvData[i])) {
                OP_LOGE(context_, "Batch[%ld], s1[%ld], s2[%ld], next_tokens[%ld], has invalid row.", i,
                        actualSeqLenData[i], actualSeqLenKvData[i], nextTokens);
                return false;
            }
        }
        sparseType = SparseEnum::RIGHT_DOWN_CAUSAL_BAND;
        return true;
    }

    bool CheckPretokenAndNexttoken(SparseEnum &sparseType)
    {
        if (sparseMode == static_cast<int64_t>(SparseMode::ALL_MASK)) {
            if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
                OP_LOGW(context_,
                        "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                        "preTokens and nextTokens will be reset max int value.",
                        preTokens, nextTokens, sparseMode);
                preTokens = std::numeric_limits<int32_t>::max();
                nextTokens = std::numeric_limits<int32_t>::max();
            }
            sparseType = static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
        } else if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            for (int64_t i = 0L; i < bSize; ++i) {
                if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                    std::string shapeMsg = std::to_string(actualSeqLenData[i]) + " and " +
                                           std::to_string(actualSeqLenKvData[i]);
                    std::string reasonMsg = "The value of actual_seq_qlen of batch " + std::to_string(i) +
                                            " must not be greater than actual_seq_kvlen";
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, "actual_seq_qlen and actual_seq_kvlen",
                                                           shapeMsg.c_str(), reasonMsg.c_str());
                    return false;
                }
            }
            preTokens = s2Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
            sparseType = SparseEnum::RIGHT_DOWN_CAUSAL;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
            OP_CHECK_IF(!CheckBandPretokenAndNexttoken(sparseType),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "pre_tokens and next_tokens",
                                                               (std::to_string(preTokens) + " and " + std::to_string(nextTokens)).c_str(),
                                                               "The values of pre_tokens and next_tokens must satisfy the band mode constraints"),
                        return false);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
            OP_CHECK_IF(!CheckRightDownCausalBandPretokenAndNexttoken(sparseType),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "pre_tokens and next_tokens",
                                                               (std::to_string(preTokens) + " and " + std::to_string(nextTokens)).c_str(),
                                                               "The values of pre_tokens and next_tokens must satisfy "
                                                               "the right down causal band mode constraints"),
                        return false);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            if (actualSeqLenData[bandIndex] - nextTokens > actualSeqLenKvData[bandIndex]) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "next_tokens",
                                                      std::to_string(nextTokens).c_str(),
                                                      "The following constraint must be met for the band index batch: "
                                                      "actual_seq_len_q - next_tokens <= actual_seq_len_kv");
                return false;
            }
            int64_t firstS2 = actualSeqLenKvData[bandIndex];
            if (preTokens < firstS2) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pre_tokens",
                                                      std::to_string(preTokens).c_str(),
                                                      "The value of pre_tokens must be greater than the first "
                                                      "valid actual_seq_len_kv in BAND_LEFT_UP_CAUSAL mode");
                return false;
            }
            sparseType = SparseEnum::BAND_LEFT_UP_CAUSAL;
        }
        return true;
    }

    bool SparseNoMaskModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS2Val,
                               SparseEnum &sparseType)
    {
        if (nextTokens < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "next_tokens",
                                                  std::to_string(nextTokens).c_str(), "To have valid data blocks, the value of "
                                                                                      "next_tokens must be greater than or equal to zero");
            return false;
        }
        if (preTokens >= maxS1Val && nextTokens >= maxS2Val) {
            return true;
        }
        for (int64_t i = 0L; i < bSize; ++i) {
            if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
                continue;
            }
            if (actualSeqLenKvData[i] + preTokens < actualSeqLenData[i]) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pre_tokens",
                                                      std::to_string(preTokens).c_str(),
                                                      "The following constraint must be met for each batch: "
                                                      "actual_seq_len_kv + pre_tokens >= actual_seq_len_q");
                return false;
            }
        }
        if (preTokens >= 0) {
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            sparseType = SparseEnum::BAND;
            return true;
        }

        if (preTokens < 0) {
            int64_t absPreTokens = std::abs(preTokens);
            if (nextTokens >= absPreTokens) {
                s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
                s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
                sparseType = SparseEnum::BAND;
                return true;
            } else {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "pre_tokens and next_tokens",
                                                       (std::to_string(preTokens) + " and " + std::to_string(nextTokens)).c_str(),
                                                       "The following constraint must be met: "
                                                       "pre_tokens + next_tokens >= min(actual_seq_len_kv) when pre_tokens < 0");
                return false;
            }
        }
        return true;
    }

    bool VarLenGetPrefixNList(SparseEnum &sparseType)
    {
        auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
        if (prefixN == nullptr) {
            OP_LOGE(context_, "[%s] prefixN is null pointer while sparse mode is prefix compress", templateName);
            return false;
        }

        auto &prefixShape = prefixN->GetShape().GetStorageShape();
        if (prefixShape.GetDimNum() != 1) {
            std::string dimStr = std::to_string(prefixShape.GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "prefix",
                                                     dimStr.c_str(), "The shape dim of input prefix must be 1D");
            return false;
        }
        if (prefixShape.GetDim(0) != bSize) {
            std::string sizeStr = std::to_string(prefixShape.GetDim(0));
            std::string reasonMsg = "The shape size of input prefix should be equal to batch size (" +
                                    std::to_string(bSize) + ")";
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, "prefix",
                                                      sizeStr.c_str(), reasonMsg.c_str());
            return false;
        }

        /* Get Data from tensor. */
        prefixNData = prefixN->GetData<int64_t>();
        if (prefixNData == nullptr) {
            OP_LOGE(context_, "[%s] prefixN data is null pointer", templateName);
            return false;
        }

        for (int64_t i = 0; i < bSize; ++i) {
            if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                if (prefixNData[i] < 0 || prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "prefix_n",
                                                          std::to_string(prefixNData[i]).c_str(),
                                                          "The value of prefix_n must be [0, actual_seq_len_kv] "
                                                          "when actual_seq_len_q > actual_seq_len_kv");
                    return false;
                }
                if (prefixNData[i] == 0) {
                    implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
                    OP_LOGD(context_, "Enable invalid line impl mode.");
                }
            } else {
                if (prefixNData[i] < actualSeqLenKvData[i] - actualSeqLenData[i] ||
                    prefixNData[i] > actualSeqLenKvData[i]) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "prefix_n",
                                                          std::to_string(prefixNData[i]).c_str(),
                                                          "The value of prefix_n must be [actual_seq_len_kv - actual_seq_len_q, "
                                                          "actual_seq_len_kv] when actual_seq_len_q <= actual_seq_len_kv");
                    return false;
                }
            }
        }

        sparseType = SparseEnum::PREFIX;
        return true;
    }

    bool VarLenSparseModeProcess(SparseEnum &sparseType)
    {
        if (!CheckPretokenAndNexttoken(sparseType)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "pre_tokens and next_tokens",
                                                   (std::to_string(preTokens) + " and " + std::to_string(nextTokens)).c_str(),
                                                   "The values of pre_tokens and next_tokens must satisfy the sparse mode constraints");
            return false;
        }

        if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            if (!VarLenGetPrefixNList(sparseType)) {
                return false;
            }
        }
        return true;
    }

    bool GetSparseInfo(SparseEnum &sparseType) override
    {
        OP_LOGD(context_,
                "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExist[%d]",
                preTokens, nextTokens, s1Size, s2Size, hasAttenMask);
        if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX) ||
            sparseMode > static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparse_mode",
                                                  std::to_string(sparseMode).c_str(), "The value of sparse_mode must be within "
                                                                                      "the supported range for VarLen mode");
            return false;
        }

        if (!hasAttenMask || tilingKeyLayout != LayoutType::LAYOUT_TND) {
            return true;
        }

        // if sparseMode is NoMask, preTokens and nextTokens start from top left vertex;
        // if sparseMode is Band, preTokens and nextTokens start from bottom right vertex.
        if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
            if (preTokens >= s1Size && nextTokens == 0) {
                sparseType = SparseEnum::CAUSAL;
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            } else {
                if (preTokens >= s1Size && nextTokens >= s2Size) {
                    return true;
                }
                int64_t minS2Val = *std::min_element(actualSeqLenKvData.begin(), actualSeqLenKvData.begin() + bSize);
                if (!SparseNoMaskModeCheck(s1Size, s2Size, minS2Val, sparseType)) {
                    return false;
                }
            }
        } else {
            if (!VarLenSparseModeProcess(sparseType)) {
                return false;
            }
        }
        return true;
    }

    bool IsUseSplitCoreMode(SparseMode inputSparseMode) override
    {
        if (inputSparseMode == SparseMode::LEFT_UP_CAUSAL || inputSparseMode == SparseMode::RIGHT_DOWN_CAUSAL ||
            inputSparseMode == SparseMode::ALL_MASK) {
            for (auto i = 0; i < bSize; i++) {
                // 当前采用保守判断条件，当同batch中S1、S2均超过阈值时开启分核优化
                int compareValue = actualSeqLenKvData[i];
                if (inputSparseMode == SparseMode::LEFT_UP_CAUSAL) {
                    compareValue = std::min(actualSeqLenData[i], actualSeqLenKvData[i]);
                }
                if (compareValue > thresholdS2Size) {
                    return true;
                }
            }
        }

        return false;
    }

    void SetSplitCoreModeParam() override
    {
        CalcThresholdForS2Size();

        // 索引从0开始，需要将基本块个数减1
        if ((sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) &&
            IsUseSplitCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        } else if ((sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) &&
                   IsUseSplitCoreMode(SparseMode::RIGHT_DOWN_CAUSAL)) {
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::ALL_MASK) &&
                   IsUseSplitCoreMode(SparseMode::ALL_MASK)) {
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        } else if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
            if (!hasAttenMask && IsUseSplitCoreMode(SparseMode::ALL_MASK)) {
                splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
            } else if (preTokens >= s1Size && nextTokens == 0 && IsUseSplitCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
                splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
            }
        }

        multiCoreParamsRegbase_->set_splitCoreMode(static_cast<uint8_t>(splitCoreMode));
        multiCoreParamsRegbase_->set_firstFullLoadS1OuterIdx(firstFullLoadS1OuterIdx);

        OP_LOGD(context_, "sparseMode: %ld, firstFullLoadS1OuterIdx: %ld, splitCoreMode: %d, s2SizeThreshold: %d.",
                sparseMode, firstFullLoadS1OuterIdx, splitCoreMode, thresholdS2Size);
    }

    int64_t GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
    {
        int64_t s2RealSize = s2Size;
        if (sparseType == static_cast<uint8_t>(SparseEnum::CAUSAL)) {
            s2RealSize = s1BasicBlock * (s1OutIdx + 1);
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::RIGHT_DOWN_CAUSAL)) {
            s2RealSize = s1BasicBlock * (s1OutIdx + 1) + actualSeqLenKvData[bOutIdx] - actualSeqLenData[bOutIdx];
        } else if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
            s2RealSize = std::max(s1BasicBlock * (s1OutIdx + 1) - s1Size + s2Size, prefixNData[bOutIdx]);
        }
        return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
    }

    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx) override
    {
        (void)bIdx;
        OP_CHECK_IF(sparseValidArray.size() == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "sparseValidArray", "0",
                                                          "The value of size of sparseValidArray must be greater than 0"),
                    return false);

        // 特殊系数, 代表s2Size=[128, 256, 384, 512, 640, 768, 896, 1024] 对应的真实耗时膨胀值
        // 如 {256, 384, 512, 640, 768, 896, 960, 1024}; {384, 512, 640, 768, 832, 896, 960, 1024};
        // 和 {256, 256, 512, 512, 768, 768, 1024, 1024};
        // cof数组可变化，以下为当前选择参数，可优化不对齐场景负载均衡
        const int64_t cof[] = {128, 256, 384, 512, 640, 768, 896, 960, 1024};
        uint8_t sparseType = inputParamsRegbase_->get_sparseType();
        int64_t localAccumS1BlockNum = 0;
        for (int32_t i = 0; i < bSize; i++) {
            if (actualSeqLenKvData[i] == 0) {
                continue;
            }
            int64_t n2G = n2Size * gSize;
            int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
            for (int64_t j = 0; j < s1BlockNum; j++) {
                // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                int64_t s1RealSize = 1;
                int64_t s2RealSize = GetS2RealSize(sparseType, i, j);
                // 新增一个系数, 解决理论和实际的差异
                int64_t s2RemainSize = s2RealSize % s2SizeLimitMax;
                s2RealSize = (s2RealSize / s2SizeLimitMax) * s2SizeLimitMax;
                int64_t s2SizeOffset = ((s2RemainSize > 0) ? cof[CeilDivision(s2RemainSize, 128L) - 1] : 0);
                s2RealSize += s2SizeOffset;

                // 每个s1方向上切分块的计算量
                for (int64_t k = 0; k < n2G; k++) {
                    sparseValidArray[localAccumS1BlockNum + k * s1BlockNum + j] = s1RealSize * s2RealSize;
                }
            }
            localAccumS1BlockNum += (s1BlockNum * n2G);
        }

        return true;
    }

    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray, MultiCoreParamsRegbase &multiCoreParamsRegbase,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx)
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()),
                                       static_cast<int64_t>(aicNum));
        int64_t totalSize = multiCoreParamsRegbase.get_totalSize();
        int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
        int64_t tmpMaxVal = maxVal;

        // 从前往后遍历
        for (int64_t idx = 1; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
                localValue[idx - 1] += sparseValidArray[start];
                localValue[idx] -= sparseValidArray[start];
                sparseStartIdx[idx] += 1;
            } else if (start == totalSize) {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        // 从后往前遍历
        for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
            int64_t start = sparseStartIdx[idx];
            if (start == totalSize) {
                if (sparseStartIdx[idx - 1] == totalSize) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] = sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else if (start > 0) {
                if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] += sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        return (tmpMaxVal >= maxVal) ? false : true;
    }

    inline bool InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum, int64_t totalSize,
                              const std::vector<int64_t> &sparseStartIdx, std::vector<int64_t> &localValue)
    {
        for (int64_t idx = 0; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
            if (start < totalSize) {
                localValue[idx] =
                    std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
            } else {
                break;
            }
        }
        return true;
    }

    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, MultiCoreParamsRegbase &multiCoreParamsRegbase,
                           int64_t maxCoreNum) override
    {
        (void)maxCoreNum;
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()),
                                       static_cast<int64_t>(aicNum));
        int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
        int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();

        OP_CHECK_IF(totalSize <= 0, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "totalSize", std::to_string(totalSize).c_str(), "The value of totalSize must be greater than 0"),
                    return false);

        // initLoad: 使用均分策略, 保证后续不会比均分差
        int64_t splitFactorSize = multiCoreParamsRegbase.get_splitFactorSize();
        std::vector<int64_t> localSparseStartIdx(aicNum, totalSize);
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAivNum, 0);
        InitLoadValue(sparseValidArray, validAivNum, totalSize, localSparseStartIdx, localValue);

        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAivNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(aicNum, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < static_cast<int64_t>(aicNum); ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                             (tmpsparseStartIdx[idx] - 1) :
                                             (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                  singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
            }
        }

        InitLoadValue(sparseValidArray, validAivNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, multiCoreParamsRegbase, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }
        for (int64_t idx = 0; idx < static_cast<int64_t>(aicNum); ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }

        return true;
    }

    void SetSparseParamsRegbase(int64_t maxCoreNum) override
    {
        std::vector<int64_t> sparseValidArray(multiCoreParamsRegbase_->get_totalSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);

        inputParamsRegbase_->set_s1Size(realT1Size);
        inputParamsRegbase_->set_s1SparseValidSize(s1SparseValidSize);
        inputParamsRegbase_->set_s2SparseValidSize(s2SparseValidSize);
    }

    ge::graphStatus PostTiling() override
    {
        FlashAttentionScoreTilingRegbase::PostTiling();
        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingVarLen, static_cast<int32_t>(NpuArch::DAV_3510), 82);
} // namespace FA
} // namespace optiling
