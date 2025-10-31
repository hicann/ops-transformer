/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ts_fag_tc_unpadded_attention.cpp
 * \brief FlashAttentionScoreGrad 算子 UnpaddedAttention UTest 用例.
 */

#include "ts_fag.h"
#include "../../../op_kernel/flash_attention_score_grad.cpp"

class Ts_Fag_Ascend910B2_UnpaddedAttention : public Ts_WithParam_Ascend910B2<FagCase> {};

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_UnpaddedAttention_BatchCase = ::testing::Values(
    FagCase("Fag_UnpaddedAttention_Case_TND_S1_pingpong_000", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 4, 0, 3, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              8512564UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 1, 1, 2050, 130, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.125f, 1.0f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2050, 2050},                             /* ActualSeqQLenList */
                    {130, 130}),                               /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_S1_pingpong_001", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 4, 0, 3, 3, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              42066996UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 1, 1, 2050, 130, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.125f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2050, 2050},                             /* ActualSeqQLenList */
                    {130, 130}),                               /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_S1_pingpong_002", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 4, 0, 1, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              134325300UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 1, 1, 128, 128, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,      /* Dtype, Layout */
                    0.125f, 1.0f, 65535, 0,                         /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,              /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                          /* PrefixShapeType */
                    {16, 16},                                             /* PrefixTensorData */
                    {64, 128},                                   /* ActualSeqQLenList */
                    {64, 128}),                                    /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_000", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 130, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 4096, 4096},                             /* ActualSeqQLenList */
                    {130, 130, 130}),                               /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_001", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 64, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 4096, 4096},                             /* ActualSeqQLenList */
                    {64, 64, 64}),                                  /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_002", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              193061940UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 64, 72,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 4096, 4096},                             /* ActualSeqQLenList */
                    {64, 64, 64}),                                  /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 13, 16},                                   /* ActualSeqQLenList */
                    {28, 18, 10}),                                  /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_001", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              167900212UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2800, 2800, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, -10, 100,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2800, 1300, 1600},                             /* ActualSeqQLenList */
                    {2800, 1800, 1000}),                            /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_002", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 13, 16},                                   /* ActualSeqQLenList */
                    {28, 18, 10}),                                  /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_003", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 13, 16},                                   /* ActualSeqQLenList */
                    {28, 18, 10}),                                  /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_004", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 13, 16},                                   /* ActualSeqQLenList */
                    {28, 18, 10}),                                  /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_005", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              235004980UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_ALIBI_S1_S2,                 /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_006", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              235004980UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_ALIBI_S1_S2,                /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_007", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_008", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(5, 2, 3, 100, 50, 64,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {100, 20, 10, 100, 4},                          /* ActualSeqQLenList */
                    {50, 10, 5, 50, 2}),                            /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_009", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 7,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_010", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 8,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_011", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(4, 1, 1, 128, 128, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {32, 64, 32, 64},                               /* PrefixTensorData */
                    {64, 128, 64, 128},                             /* ActualSeqQLenList */
                    {64, 128, 64, 128}),                            /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_012", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              235004980UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::TND_1S,                           /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_013", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              235004980UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 2,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::TND_SS,                           /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_014", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, -10, -10,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_015", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              167887924UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 2048, 2048, 64,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::TND,         /* Dtype, Layout */
                    0.08838f, 0.8f, -100, -100,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                     /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2048, 273, 194},                               /* ActualSeqQLenList */
                    {2048, 273, 194}),                              /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_016", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176161076UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 64, 32,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {64},                                           /* ActualSeqQLenList */
                    {64}),                                          /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_017", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 130, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 4096, 4096},                             /* ActualSeqQLenList */
                    {130, 130, 130}),                               /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_018", true,             /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              176284724UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 4096, 64, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {4096, 4096, 4096},                             /* ActualSeqQLenList */
                    {64, 64, 64}),                                  /* ActualSeqKVLenList */

            FaCase::kTilingTemplatePriority_Invalid /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_TND_019", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 4, 0, 3, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 1, 1, 2, 2, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 65535,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {2,2,2},                                           /* ActualSeqQTensorData */
                    {2,0,0}),                                          /* ActualSeqKVTensorData */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_020", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 128, 128, 32,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::TND_SS,                           /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {128},                                          /* ActualSeqQLenList */
                    {128}),                                         /* ActualSeqKVLenList */

            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2 /* TilingTemplatePriority */
            ),
    FagCase("Fag_UnpaddedAttention_Case_022", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 4, 512, 512, 128,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,        /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {512},                                         /* ActualSeqQLenList */
                    {512}),                                        /* ActualSeqKVLenList */
    
            FagCase::kTemplatePriority_Tnd_Mla /* TilingTemplatePriority */
        )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_UnpaddedAttention, Tc_Fag_UnpaddedAttention_BatchCase);

class Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam : public Ts_Fag_Ascend910B2_UnpaddedAttention {};

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam, Tc_seqlen_nullptr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.actualSeqQLen = Tensor("actualSeqQLen", {}, "None", ge::DataType::DT_INT64, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam, Tc_seqlen_dimerr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.actualSeqQLen =
        Tensor("actualSeqQLen", {case_->mParam.b + 1}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_UnpaddedAttention_InvalidParam_BatchCase = ::testing::Values(

    FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(3, 2, 3, 28, 28, 64,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65535, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE,                          /* PrefixShapeType */
                    {},                                             /* PrefixTensorData */
                    {28, 13, 16},                                   /* ActualSeqQLenList */
                    {28, 18, 10}),                                  /* ActualSeqKVLenList */
            FaCase::kTilingTemplatePriority_Invalid                 /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_UnpaddedAttention_InvalidParam,
                         Tc_Fag_UnpaddedAttention_InvalidParam_BatchCase);

TEST_F(Ts_Fag_Ascend910B2, Tc_seqlen_dim_long)
{
    auto cur = FagCase("Fag_UnpaddedAttention_Case_000", true,                 /* CaseName, Enable */
                       "",                                                     /* DebugInfo */
                       [](FAG_KERNEL_PARAM_){
                              ::flash_attention_score_grad<4, 3, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
                       OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                              ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                                         176161076UL,               /* ExpectTilingKey */
                                         ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                       FaParam(2049, 2, 1, 32, 32, 32,                         /* B, N2, G, S1, S2, D */
                               ge::DataType::DT_FLOAT16, LayoutType::TND,      /* Dtype, Layout */
                               0.08838f, 0.8f, 65535, 0,          /* Scale, KeepProb, PreTokens, NxtTokens */
                               1, 0,                              /* InnerPrecise, SparseMode */
                               PseShapeType::NONE,                /* PseShapeType */
                               DropMaskShapeType::B_N1_S1_S2DIV8, /* DropMaskShapeType */
                               PaddingMaskShapeType::S1_S2,       /* PaddingMaskShapeType */
                               AttenMaskShapeType::S1_S2,         /* AttentionMaskShapeType */
                               ge::DataType::DT_BOOL,             /* AttentionMaskDtype */
                               PrefixShapeType::NONE,             /* PrefixShapeType */
                               {},                                /* PrefixTensorData */
                               {},                                /* ActualSeqQLenList */
                               {}),                               /* ActualSeqKVLenList */
                       FaCase::kTilingTemplatePriority_Invalid    /* TilingTemplatePriority */
    );
    int64_t tmpData = 32;
    cur.mParam.actualSeqQLenList.push_back(tmpData);
    cur.mParam.actualSeqKVLenList.push_back(tmpData);
    for (int64_t i = 0; i < 2048; i++) {
        cur.mParam.actualSeqQLenList.push_back(0);
        cur.mParam.actualSeqKVLenList.push_back(0);
    }
    ASSERT_TRUE(cur.Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(cur.Run(), (cur.mReverse.mExp.mSuccess));
}
