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
 * \file ts_fag_tc_us1s2_bbn2.cpp
 * \brief FlashAttentionScoreGrad 算子 Us1s2Bbn2 模板 UTest 用例.
 */

#include "ts_fag.h"
#include "../../../op_kernel/flash_attention_score_grad.cpp"

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalAttenMaskShape_001)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalAttenMaskShape_001", true,         /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
               OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                 ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 512, 512, 128,                        /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.attenMask = Tensor("atten_mask", {10000, 10000, cs.mParam.s1, cs.mParam.s2}, "X_X_S1_S2",
                                 cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalAttenMaskShape_002)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalAttenMaskShape_002", true,         /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
               OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                 ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 512, 512, 128,                        /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.attenMask = Tensor("atten_mask", {cs.mParam.b}, "B", cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalPseShape_001)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalPseShape_001", true,               /* CaseName, Enable */
               "not support pseShape",                                 /* DebugInfo */
               OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                 ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 512, 512, 128,                        /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.pse =
        Tensor("pse", {10000, 10000, cs.mParam.s1, cs.mParam.s2}, "X_X_S1_S2", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Us1s2Bbn2_IllegalPseShape_002)
{
    FagCase cs("Tc_Us1s2Bbn2_IllegalPseShape_002", true,               /* CaseName, Enable */
               "not support pseShape",                                 /* DebugInfo */
               OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                      ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                 ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                 ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
               FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                       ge::DataType::DT_FLOAT, LayoutType::BNSD,       /* Dtype, Layout */
                       0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                       1, 0,                                           /* InnerPrecise, SparseMode */
                       PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                       DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                       PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                       AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                       ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.pse = Tensor("pse", {cs.mParam.b}, "B", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

class Ts_Fag_Ascend910B2_Us1s2Bbn2 : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_L1Manage_001", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              293617972UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 257, 257, 72,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSH,        /* Dtype, Layout */
                    0.11785f, 1, 65536, 65536,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                       /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_L1Manage_002", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              293617972UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 257, 257, 88,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSH,        /* Dtype, Layout */
                    0.1066f, 1, 65536, 65536,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                       /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_L1Manage_003", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              276840756UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 507, 507, 128,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSH,        /* Dtype, Layout */
                    0.08839f, 1, 65536, 65536,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                       /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::NONE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_000", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              201367860UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(128, 18, 1, 34, 34, 16,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,        /* Dtype, Layout */
                    0.08838f, 1, 65536, 65536,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_S1_S2,                       /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_fuzz_000", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              134258996UL,               /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1, 256, 72,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::SBH,        /* Dtype, Layout */
                    0.11785f, 1, 65536, 65536,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                       /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::SPARSE,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_001", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 40, 1, 256, 128, 128,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 8,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_002", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 2, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 25, 1, 15, 15, 72,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_004", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 25, 1, 15, 129, 72,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_005", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 25, 1, 15, 129, 32,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_006", true,                         /* CaseName, Enable */         // 192938292
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, RunKernelNotInPr),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 50, 1, 3, 65, 88,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_007", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 20, 2, 128, 128, 32,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_008", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 20, 1, 1023, 128, 32,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_009", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 32,                           /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT, LayoutType::BSND,       /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_010", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 32,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::SBH,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_011", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            [](FAG_KERNEL_PARAM_){
              ::flash_attention_score_grad<4, 3, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0>(FAG_INPUT_PARAMS);},
            OpInfoWithSocversion(ControlInfo(true, true),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 32,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_012", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 72,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSND,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_013", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 72,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            ),
    FagCase("Fag_Us1s2Bbn2_Case_014", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),             /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 64, 128, 16,                            /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.9f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )
);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2, Tc_Fag_Us1s2Bbn2_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErr)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {case_->mParam.s1, 1}, "S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErrDim)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask = Tensor("attenMask", {1, case_->mParam.s1, 1, case_->mParam.s2}, "1_S1_1_S2(Invalid)",
                                     case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_AttenmaskErrDimNum)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {1, case_->mParam.s1, 1}, "1_S1_1(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2_InvalidShape_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_Case_000", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 512, 512, 128,                        /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidShape, Tc_Fag_Us1s2Bbn2_InvalidShape_BatchCase);

class Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape : public Ts_Fag_Ascend910B2_Us1s2Bbn2 {};

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape, Tc_AttenmaskErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.attenMask =
        Tensor("attenMask", {2048, 2048}, "not_3072_2048(Invalid)", case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

TEST_P(Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape, Tc_PrefixErrShapeForPrefixCompress)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    case_->mParam.prefix = Tensor("prefix", {110, 110, 110, 110, 110, 110, 110}, "prefixN_gt_B(Invalid)",
                                  case_->mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Us1s2Bbn2_InvalidPrefixCompressShape_BatchCase = ::testing::Values(

    FagCase("Fag_Us1s2Bbn2_Case_001", true,                         /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(6, 10, 10, 512, 512, 128,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 0.8f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 6,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::PREFIXCOMPRESS,             /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::B,                             /* PrefixShapeType */
                    {332, 482, 196, 245, 177, 71},                  /* PrefixTensorData */
                    {},                                             /* ActualSeqQLenList */
                    {}),                                            /* ActualSeqKVLenList */
            FagCase::kTemplatePriority_Us1s2_Bbn2                   /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Us1s2Bbn2_InvalidPrefixCompressShape,
                         Tc_Fag_Us1s2Bbn2_InvalidPrefixCompressShape_BatchCase);
