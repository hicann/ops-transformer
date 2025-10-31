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
 * \file ts_fag_tc_us1s2_bbn2gs1s2_sab.cpp
 * \brief FlashAttentionScoreGrad 算子 Sab 模板 UTest 用例.
 */

#include "ts_fag.h"

TEST_F(Ts_Fag_Ascend910B2, Tc_Sab_IllegalAttenMaskShape_001)
{
    FagCase cs("Tc_Sab_IllegalAttenMaskShape_001", true,               /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
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
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.attenMask = Tensor("atten_mask", {10000, 10000, cs.mParam.s1, cs.mParam.s2}, "X_X_S1_S2",
                                 cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Sab_IllegalAttenMaskShape_002)
{
    FagCase cs("Tc_Sab_IllegalAttenMaskShape_002", true,               /* CaseName, Enable */
               "not support attenmask shape type",                     /* DebugInfo */
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
                       AttenMaskShapeType::B_N1_S1_S2,                 /* AttentionMaskShapeType */
                       ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                       PrefixShapeType::NONE),                         /* PrefixShapeType */
               FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.attenMask = Tensor("atten_mask", {cs.mParam.b}, "B", cs.mParam.attenMaskDtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}

TEST_F(Ts_Fag_Ascend910B2, Tc_Sab_IllegalPseShape_001)
{
    FagCase cs("Tc_Sab_IllegalPseShape_001", true,                     /* CaseName, Enable */
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
               FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
    );

    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mParam.pse =
        Tensor("pse", {10000, 10000, cs.mParam.s1, cs.mParam.s2}, "X_X_S1_S2", cs.mParam.dtype, ge::FORMAT_ND);
    ASSERT_EQ(cs.Run(), cs.mReverse.mExp.mSuccess);
}


class Ts_Fag_Ascend910B2_Sab : public Ts_Fag_WithParam_Ascend910B2 {};

TEST_P(Ts_Fag_Ascend910B2_Sab, Tc_BatchCase)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_EQ(case_->Run(), case_->mReverse.mExp.mSuccess);
}

const auto Tc_Fag_Sab_BatchCase = ::testing::Values(

    FagCase("Fag_Sab_Case_000", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_001", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 40, 1, 2048, 2048, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BSND,        /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    0, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_002", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(24, 10, 1, 2304, 2304, 64,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 0,                        /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::_1_N1_S1_S2,                      /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_UINT8,                         /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_003", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 9216, 77, 64,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 65504, 65504,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_004", true,                               /* CaseName, Enable */
            "compress atten_mask mode not support s1 s2 2048 2048", /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 128,                      /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 2048, 2048,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 3,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_005", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 2048, 2048, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, 512,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 4,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_006", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, 512, -256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_007", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(2, 30, 2, 1024, 1024, 64,                       /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype, Layout */
                    0.08838f, 0.8f, -128, 256,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::B_N1_1_S2,                        /* PseShapeType */
                    DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                    PaddingMaskShapeType::S1_S2,                    /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_008", false,                              /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling, RunKernel */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(6, 10, 10, 2048, 1024, 128,                     /* B, N2, G, S1, S2, D */
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
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_009", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 2, 144, 1024, 72,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_010", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 2, 32, 1024, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_011", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 2, 32, 1024, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, -2,                      /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_012", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 2, 32, 1024, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, -100, 1000,                     /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_013", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, true),                  /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 2, 32, 2049, 64,                          /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype, Layout */
                    0.08838f, 1.0f, 65536, 0,                       /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 1,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            ),
    FagCase("Fag_Sab_Case_014", true,                               /* CaseName, Enable */
            "",                                                     /* DebugInfo */
            OpInfoWithSocversion(ControlInfo(true, false, false),                 /* RunTiling, RunKernel, Deterministic */
                   ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                              ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                              ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
            FaParam(1, 1, 1, 1024, 1024, 128,                         /* B, N2, G, S1, S2, D */
                    ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype, Layout */
                    0.07217f, 1.0f, 65536, 65536,                   /* Scale, KeepProb, PreTokens, NxtTokens */
                    1, 0,                                           /* InnerPrecise, SparseMode */
                    PseShapeType::NONE,                             /* PseShapeType */
                    DropMaskShapeType::NONE,                        /* DropMaskShapeType */
                    PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                    AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                    ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                    PrefixShapeType::NONE),                         /* PrefixShapeType */
            FagCase::kTemplatePriority_Us1s2_Bbn2gs1s2_sab          /* TilingTemplatePriority */
            )

);
INSTANTIATE_TEST_SUITE_P(Fag, Ts_Fag_Ascend910B2_Sab, Tc_Fag_Sab_BatchCase);
