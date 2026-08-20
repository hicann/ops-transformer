/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../op_host/checkers/checker_adapter.h"
#include "../../../op_host/checkers/mask_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/paged_attention_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/seq_len_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/softmax_lse_checker_sparse_flash_mla.h"
#include "../../../op_host/checkers/sparse_compression_checker.h"

namespace {
using optiling::sparse_mla_checker::CheckContext;
using optiling::sparse_mla_checker::Layout;
using optiling::sparse_mla_checker::MaskChecker;
using optiling::sparse_mla_checker::OperatorVariant;
using optiling::sparse_mla_checker::PagedAttentionChecker;
using optiling::sparse_mla_checker::SeqLenChecker;
using optiling::sparse_mla_checker::SoftmaxLseChecker;
using optiling::sparse_mla_checker::SparseCompressionChecker;

struct ShapeOnlyOptionalParam {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Tensor *tensor = nullptr;
    const gert::StorageShape *shape = nullptr;
};

CheckContext MakeSparseContext()
{
    CheckContext context;
    context.opName = "SparseFlashMla";
    context.variant = OperatorVariant::SPARSE;
    context.cmpRatio = 1;
    context.oriMaskMode = 0;
    context.cmpMaskMode = 0;
    context.oriWinLeft = -1;
    context.oriWinRight = -1;
    return context;
}
} // namespace

TEST(SparseFlashMlaChecker, AcceptsPagedAttentionSequsedWithShapeButNoTensorPointer)
{
    gert::StorageShape shape = {{2}, {2}};
    ShapeOnlyOptionalParam param;
    param.shape = &shape;

    const auto tensorParam = optiling::sparse_mla_checker::MakeOptionalTensor(param);
    EXPECT_TRUE(tensorParam.present);
    ASSERT_NE(tensorParam.shape, nullptr);
    EXPECT_EQ(tensorParam.shape->GetDimNum(), 1U);
    EXPECT_EQ(tensorParam.shape->GetDim(0), 2);

    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriBlockTable.present = true;
    context.sequsedOriKv = tensorParam;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, TreatsMissingOptionalInputAsAbsent)
{
    const ShapeOnlyOptionalParam param;

    const auto tensorParam = optiling::sparse_mla_checker::MakeOptionalTensor(param);

    EXPECT_FALSE(tensorParam.present);
    EXPECT_EQ(tensorParam.shape, nullptr);
}

TEST(SparseFlashMlaChecker, AcceptsOriSparseIndicesWithMatchingTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;
    context.oriTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, RejectsOriSparseIndicesWithoutTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.oriSparseIndices.present = true;

    SparseCompressionChecker compressionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
}

TEST(SparseFlashMlaChecker, RejectsCmpSparseIndicesWithoutTopkLength)
{
    CheckContext context = MakeSparseContext();
    context.cmpKv.present = true;
    context.cmpSparseIndices.present = true;

    SparseCompressionChecker compressionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_FAILED);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthAsSequsedReplacementInPaModeWithoutSparseIndices)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriTopkLength.present = true;
    context.oriBlockTable.present = true;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsOriCmpSparseTopkLengthPairs)
{
    CheckContext context = MakeSparseContext();
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;

    SparseCompressionChecker compressionChecker;
    MaskChecker maskChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(maskChecker.CheckFeature(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, SkipsSoftmaxLseChecksWhenReturnIsDisabled)
{
    CheckContext context = MakeSparseContext();
    context.returnSoftmaxLse = false;
    context.softmaxLse.present = true;

    SoftmaxLseChecker checker;
    EXPECT_EQ(checker.CheckSinglePara(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiPara(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthsWithoutSequsedKvInOriCmpSparsePaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;

    SparseCompressionChecker compressionChecker;
    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsOriTopkLengthWithoutSequsedOriKvInOriSparsePaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.oriSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.oriBlockTable.present = true;

    SparseCompressionChecker compressionChecker;
    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(compressionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsCmpTopkLengthAsSequsedReplacementInPaMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.cmpSparseIndices.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;
    context.sequsedOriKv.present = true;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}

TEST(SparseFlashMlaChecker, AcceptsTopkLengthsAsSequsedReplacementRegardlessOfMaskMode)
{
    CheckContext context = MakeSparseContext();
    context.kvLayout = Layout::PA_BBND;
    context.cmpKv.present = true;
    context.oriSparseIndices.present = true;
    context.cmpSparseIndices.present = true;
    context.oriTopkLength.present = true;
    context.cmpTopkLength.present = true;
    context.oriBlockTable.present = true;
    context.cmpBlockTable.present = true;
    context.cmpMaskMode = 3;

    SeqLenChecker seqLenChecker;
    PagedAttentionChecker pagedAttentionChecker;
    EXPECT_EQ(seqLenChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pagedAttentionChecker.CheckParaExistence(context), ge::GRAPH_SUCCESS);
}
