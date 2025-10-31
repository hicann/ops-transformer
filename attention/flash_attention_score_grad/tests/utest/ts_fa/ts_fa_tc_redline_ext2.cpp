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
 * \file ts_fa_tc_redline_ext.cpp
 * \brief FlashAttention 自动反向红线用例.
 */

#include "ts_fa.h"

class Ts_Fa_Ascend910B2_Redline_ExtCase_2 : public Ts_Fa_WithParam_Ascend910B2 {};
class Ts_Fa_Ascend910B3_Redline_ExtCase_2 : public Ts_Fa_WithParam_Ascend910B3 {};

TEST_P(Ts_Fa_Ascend910B2_Redline_ExtCase_2, Tc_Redline)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(case_->Run());
}

TEST_P(Ts_Fa_Ascend910B3_Redline_ExtCase_2, Tc_Redline)
{
    ASSERT_TRUE(case_->Init(static_cast<int32_t>(this->socVersion_)));
    ASSERT_TRUE(case_->Run());
}

const auto Tc_Fa_Redline_Case_ExtCase_2 = ::testing::Values(

    FaCase("ExtCase_000", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 65536, 65536, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_001", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_002", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_003", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_004", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_005", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_006", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_007", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_008", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_009", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_010", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_011", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::NONE,                       /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_012", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_013", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_014", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_015", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(8, 10, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_016", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(16, 10, 1, 2048, 2048, 128,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_017", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_018", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_019", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_020", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(8, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_021", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(16, 10, 1, 4096, 4096, 128,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_022", false,                                   /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 1, 8, 8192, 8192, 120,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.09128f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_023", false,                                   /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 4, 2, 8192, 8192, 120,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.09128f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_024", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 52, 1, 2048, 2048, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_025", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_026", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_027", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_028", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_029", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 1, 5, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_030", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 1, 5, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_031", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(4, 1, 5, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_032", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(6, 1, 5, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_033", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 40, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_034", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 40, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_035", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 5, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_036", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 5, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_037", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 10, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_038", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 32, 1, 4096, 4096, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BNSD,     /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_039", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_040", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_041", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 8192, 8192, 128,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_042", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 8, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::SBH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 4096, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_043", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 8192, 8192, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_044", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 32768, 32768, 128,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_045", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 4, 1, 8192, 8192, 256,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_046", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 8192, 8192, 256,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_047", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 16, 1, 8192, 8192, 256,                      /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_048", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 4, 1, 16384, 16384, 256,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_049", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 16384, 16384, 256,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_050", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 16, 1, 16384, 16384, 256,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_051", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 4, 1, 32768, 32768, 256,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_052", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 32768, 32768, 256,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_053", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 16, 1, 32768, 32768, 256,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::BSH,      /* Dtype,Layout */
                   0.0625f, 0.9f, 65536, 0,                        /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_054", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(8, 4, 1, 4096, 4096, 128,                       /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_FLOAT16, LayoutType::SBH,      /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_055", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 8, 1, 16384, 16384, 128,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 16384, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_056", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(2, 8, 1, 16384, 16384, 128,                     /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BNSD,        /* Dtype,Layout */
                   0.08838f, 0.9f, 16384, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::S1_S2,                      /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_057", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 32768, 32768, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_058", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 32768, 32768, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_059", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 48, 1, 32768, 32768, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_060", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 12, 1, 65536, 65536, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           ),
    FaCase("ExtCase_061", true,                                    /* CaseName,Enable */
           "",                                                     /* DebugInfo */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                  ExpectInfoWithSocversion(true,                                 /* ExpectSuccess */
                             ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                             ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
           FaParam(1, 24, 1, 65536, 65536, 128,                    /* B,N2,G,S1,S2,D */
                   ge::DataType::DT_BF16, LayoutType::BSH,         /* Dtype,Layout */
                   0.08838f, 0.9f, 65536, 0,                       /* Scale,KeepProb,PreTokens,NxtTokens */
                   0, 0,                                           /* InnerPrecise,SparseMode */
                   PseShapeType::NONE,                             /* PseShapeType */
                   DropMaskShapeType::B_N1_S1_S2DIV8,              /* DropMaskShapeType */
                   PaddingMaskShapeType::NONE,                     /* PaddingMaskShapeType */
                   AttenMaskShapeType::B_1_S1_S2,                  /* AttentionMaskShapeType */
                   ge::DataType::DT_BOOL,                          /* AttentionMaskDtype */
                   PrefixShapeType::NONE)                          /* PrefixShapeType */
           )

);
INSTANTIATE_TEST_SUITE_P(Fa, Ts_Fa_Ascend910B2_Redline_ExtCase_2, Tc_Fa_Redline_Case_ExtCase_2);
INSTANTIATE_TEST_SUITE_P(Fa, Ts_Fa_Ascend910B3_Redline_ExtCase_2, Tc_Fa_Redline_Case_ExtCase_2);
