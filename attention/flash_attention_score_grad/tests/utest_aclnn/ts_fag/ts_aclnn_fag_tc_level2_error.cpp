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
 * \file ts_aclnn_fag_tc_level2_error.cpp
 * \brief FlashAttentionScore / FlashAttentionScoreGrad ACLNN 测试用例.
 */

#include "ts_aclnn_fag.h"

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err01)
{
    AclnnFagCase cs("Test_001", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0,                                      /* InnerPrecise,SparseMode */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "BSH";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2, 2048}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return;
        }
        if (t->IsOutput()) {
            continue;
        }
    }
    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err02)
{
    AclnnFagCase cs("Test_002", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 4096, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0,                                      /* InnerPrecise,SparseMode */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE,                     /* PrefixShapeType */
                                 {},                                        /* PrefixTensorData */
                                 {25, 50},                                  /* ActualSeqQLenList */
                                 {50, 25})                                  /* ActualSeqKVLenList */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "TND";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return;
        }
        if (t->IsOutput()) {
            continue;
        }
    }
    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err03)
{
    AclnnFagCase cs("Test_003", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "BSH";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2, 2048}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return;
        }
        if (t->IsOutput()) {
            continue;
        }
    }
    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err04)
{
    AclnnFagCase cs("Test_004", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 4096, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE,                     /* PrefixShapeType */
                                 {},                                        /* PrefixTensorData */
                                 {25, 50},                                  /* ActualSeqQLenList */
                                 {50, 25})                                  /* ActualSeqKVLenList */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "TND";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
        if (t->GetExpDataSize() <= 0) {
            continue;
        }
        auto *devData = t->AllocDevData(0, 0);
        if (devData == nullptr) {
            return;
        }
        if (t->IsOutput()) {
            continue;
        }
    }
    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err05)
{
    AclnnFagCase cs("Test_005", false,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));

    ASSERT_EQ(cs.Run(), true);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err06)
{
    AclnnFagCase cs("Test_006", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 4096, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE,                     /* PrefixShapeType */
                                 {},                                        /* PrefixTensorData */
                                 {25, 50},                                  /* ActualSeqQLenList */
                                 {50, 25})                                  /* ActualSeqKVLenList */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err07)
{
    AclnnFagCase cs("Test_007", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0,                                      /* InnerPrecise,SparseMode */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "BSH";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2, 2048}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err08)
{
    AclnnFagCase cs("Test_008", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 4096, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0,                                      /* InnerPrecise,SparseMode */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE,                     /* PrefixShapeType */
                                 {},                                        /* PrefixTensorData */
                                 {25, 50},                                  /* ActualSeqQLenList */
                                 {50, 25})                                  /* ActualSeqKVLenList */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "TND";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err09)
{
    AclnnFagCase cs("Test_009", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "BSH";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2, 2048}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err010)
{
    AclnnFagCase cs("Test_0010", true,                                      /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, true),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 4096, 2048, 128,                  /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::TND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE,                     /* PrefixShapeType */
                                 {},                                        /* PrefixTensorData */
                                 {25, 50},                                  /* ActualSeqQLenList */
                                 {50, 25})                                  /* ActualSeqKVLenList */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    for (auto *t : {&cs.mAclnnParam.aclnnQuery}) {
        t->FreeDevData();
    }
    std::string layout1 = "TND";
    ops::adv::tests::utils::Tensor query1 = Tensor("query1", {2}, layout1.c_str(), ge::DataType::DT_FLOAT16,
                                                   ge::FORMAT_ND, Tensor::TensorType::REQUIRED_INPUT);
    cs.mAclnnParam.aclnnQuery = ops::adv::tests::utils::AclnnTensor(query1);

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err011)
{
    AclnnFagCase cs("Test_011", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 71,                   /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err012)
{
    AclnnFagCase cs("Test_012", true,                                        /* CaseName,Enable */
                    "",                                                      /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, false),                         /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                 /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,         /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)),  /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 71,                    /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::BSND, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                 /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                    /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                         /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,          /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                 /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                   /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                      /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                      /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err013)
{
    AclnnFagCase cs("Test_013", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(2, 5, 1, 2048, 2048, 71,                   /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));

    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err014)
{
    AclnnFagCase cs("Test_014", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(1, 1, 1, 1024, 1024, 1,                    /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 1,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mAclnnParam.n1 = 100;
    ASSERT_EQ(cs.Run(), false);
}

TEST_F(Ts_Aclnn_Fag_Ascend910B2, Tc_Fag_Level2_Err015)
{
    AclnnFagCase cs("Test_015", true,                                       /* CaseName,Enable */
                    "",                                                     /* DebugInfo */
                    OpInfoWithSocversion(ControlInfo(true, false),                        /* RunTiling,RunKernel */
                           ExpectInfoWithSocversion(false,                                /* ExpectSuccess */
                                      ExpectInfoWithSocversion::kInvalidTilingKey,        /* ExpectTilingKey */
                                      ExpectInfoWithSocversion::kInvalidTilingBlockDim)), /* ExpectTilingBlockDim */
                    AclnnFaParam(1, 1, 1, 1024, 1024, 1,                    /* B,N2,G,S1,S2,D */
                                 ge::DataType::DT_FLOAT16, LayoutType::SBH, /* Dtype,Layout */
                                 0.125f, 0.9f, 65536, 65536,                /* Scale,KeepProb,PreTokens,NxtTokens */
                                 0, 0, 2,                                   /* InnerPrecise,SparseMode,pseType */
                                 PseShapeType::NONE,                        /* PseShapeType */
                                 DropMaskShapeType::B_N1_S1_S2DIV8,         /* DropMaskShapeType */
                                 PaddingMaskShapeType::NONE,                /* PaddingMaskShapeType */
                                 AttenMaskShapeType::NONE,                  /* AttentionMaskShapeType */
                                 ge::DataType::DT_BOOL,                     /* AttentionMaskDtype */
                                 PrefixShapeType::NONE)                     /* PrefixShapeType */
    );
    ASSERT_TRUE(cs.Init(static_cast<int32_t>(this->socVersion_)));
    cs.mAclnnParam.n1 = 100;
    ASSERT_EQ(cs.Run(), false);
}