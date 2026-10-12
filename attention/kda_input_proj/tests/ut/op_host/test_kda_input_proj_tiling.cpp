/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kda_input_proj_tiling.cpp
 * \brief Host tiling UT for KdaInputProj.
 *
 * Tiling-key 笛卡尔积：TRANS_WEIGHT_{QKV,BETA,GATE,G} 各 0/1 → 16 keys。
 * 其余分支按源码条件反解：
 *   MxQuant  full-row          colBlocks <= maxUbBlockNum
 *   MxQuant  col-split 512/256 小 UB + 不同 T
 *   QMM      nAlign 16 vs 32    trans_weight_qkv true/false 且 N 非 32 对齐
 *   QMM      bMustHitL2         mCnt>1 (T=257) vs mCnt==1 (T=8)
 *   QMM      dbL0C 2 vs 1       小 T vs 256 基本块
 *   MmBgg    trans mismatch     beta/gate/g trans 不一致
 *   Sigmoid  FillAivSplit       小 elemNum (T=1) / 大 elemNum (T=512)
 *   MxQuant  列/行尾块          K=1088 col-split；T=41 full-row rowTail
 *   QMM      N 尾块             nQkv=241 → baseN=256
 */

#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/kda_input_proj_tiling.h"
#include "../../../op_kernel/kda_input_proj_template_tiling_key.h"
#include "../../../op_kernel/kda_input_proj_tiling_data.h"
#include "../../../op_kernel/kda_input_proj_workspace.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;
using namespace ge;
using namespace optiling;

namespace {
constexpr int64_t kMxScaleBlock = 64;
constexpr int64_t kScalePack = 2;
constexpr uint64_t kTilingDataSize = 4096;
constexpr uint64_t kDefaultUb = 253952;
constexpr uint64_t kSmallUb = 8192;
constexpr uint64_t kDefaultL1 = 524288;

constexpr int64_t kProdK = 7168;
constexpr int64_t kProdQkv = 4608;
constexpr int64_t kProdBeta = 12;
constexpr int64_t kProdGate = 1536;
constexpr int64_t kProdG = 1536;

constexpr int64_t kSmallK = 256;
constexpr int64_t kSmallQkv = 128;
constexpr int64_t kSmallBeta = 16;
constexpr int64_t kSmallGate = 64;
constexpr int64_t kSmallG = 64;

static optiling::KdaInputProjCompileInfo gCompileInfo{};

gert::StorageShape Shape2D(int64_t d0, int64_t d1)
{
    return {{d0, d1}, {d0, d1}};
}

gert::StorageShape Shape3D(int64_t d0, int64_t d1, int64_t d2)
{
    return {{d0, d1, d2}, {d0, d1, d2}};
}

gert::StorageShape WeightShape(int64_t nSize, int64_t kSize, bool trans)
{
    return trans ? Shape2D(nSize, kSize) : Shape2D(kSize, nSize);
}

gert::StorageShape ScaleShape(int64_t nSize, int64_t kSize, bool transQkv)
{
    const int64_t mxK = (kSize + kMxScaleBlock - 1) / kMxScaleBlock;
    return transQkv ? Shape3D(nSize, mxK, kScalePack) : Shape3D(mxK, nSize, kScalePack);
}

std::string MakeSocInfo(uint64_t ubSize, uint64_t l1Size, const char* socVersion)
{
    std::ostringstream oss;
    oss << R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1",)"
        << R"("Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,)"
        << R"("Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,)"
        << "\"UB_SIZE\":" << ubSize << ","
        << "\"L2_SIZE\":201326592,\"L1_SIZE\":" << l1Size << ","
        << R"("L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,)"
        << R"("CORE_NUM":20,"cube_core_cnt":20,"vector_core_cnt":40,)"
        << R"("cube_freq":1800,"ddr_rate":64,"l2_rate":512,"socVersion":")" << socVersion << "\"}}";
    return oss.str();
}

struct ShapeCfg {
    int64_t t = 8;
    int64_t k = kSmallK;
    int64_t nQkv = kSmallQkv;
    int64_t nBeta = kSmallBeta;
    int64_t nGate = kSmallGate;
    int64_t nG = kSmallG;
    bool transQkv = true;
    bool transBeta = true;
    bool transGate = true;
    bool transG = true;
    bool withAttrs = true;
    uint64_t ubSize = kDefaultUb;
    uint64_t l1Size = kDefaultL1;
    const char* socVersion = "Ascend950";
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType wQkvDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType wBetaDtype = ge::DT_BF16;
    ge::DataType wGateDtype = ge::DT_BF16;
    ge::DataType wGDtype = ge::DT_BF16;
    ge::DataType scaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType qkvOutDtype = ge::DT_BF16;
    ge::DataType betaOutDtype = ge::DT_FLOAT;
    ge::DataType gateOutDtype = ge::DT_BF16;
    ge::DataType gOutDtype = ge::DT_BF16;
    int64_t xRankExtra = 0;      // >0 时 x 改成 3D
    int64_t weightRankExtra = 0; // >0 时 weight_qkv 改成 3D
    int64_t scaleRank = 3;
    int64_t scaleDim2 = kScalePack;
    bool mismatchScale = false;
};

std::vector<gert::TilingContextPara::OpAttr> MakeAttrs(const ShapeCfg& cfg)
{
    if (!cfg.withAttrs) {
        return {};
    }
    return {
        {"trans_weight_qkv", Ops::Transformer::AnyValue::CreateFrom<bool>(cfg.transQkv)},
        {"trans_weight_beta", Ops::Transformer::AnyValue::CreateFrom<bool>(cfg.transBeta)},
        {"trans_weight_gate", Ops::Transformer::AnyValue::CreateFrom<bool>(cfg.transGate)},
        {"trans_weight_g", Ops::Transformer::AnyValue::CreateFrom<bool>(cfg.transG)},
    };
}

gert::TilingContextPara BuildPara(const ShapeCfg& cfg)
{
    gert::StorageShape xShape = cfg.xRankExtra > 0 ? Shape3D(cfg.t, cfg.k, 1) : Shape2D(cfg.t, cfg.k);
    gert::StorageShape scale = ScaleShape(cfg.nQkv, cfg.k, cfg.transQkv);
    if (cfg.mismatchScale) {
        scale = ScaleShape(cfg.nQkv, cfg.k, !cfg.transQkv);
    } else if (cfg.scaleDim2 != kScalePack) {
        const int64_t mxK = (cfg.k + kMxScaleBlock - 1) / kMxScaleBlock;
        scale = cfg.transQkv ? Shape3D(cfg.nQkv, mxK, cfg.scaleDim2) : Shape3D(mxK, cfg.nQkv, cfg.scaleDim2);
    }

    gert::StorageShape wQkv =
        cfg.weightRankExtra > 0 ? Shape3D(cfg.nQkv, cfg.k, 1) : WeightShape(cfg.nQkv, cfg.k, cfg.transQkv);
    if (cfg.scaleRank == 2) {
        const int64_t mxK = (cfg.k + kMxScaleBlock - 1) / kMxScaleBlock;
        scale = cfg.transQkv ? Shape2D(cfg.nQkv, mxK) : Shape2D(mxK, cfg.nQkv);
    }
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {xShape, cfg.xDtype, ge::FORMAT_ND},
        {wQkv, cfg.wQkvDtype, ge::FORMAT_ND},
        {WeightShape(cfg.nBeta, cfg.k, cfg.transBeta), cfg.wBetaDtype, ge::FORMAT_ND},
        {WeightShape(cfg.nGate, cfg.k, cfg.transGate), cfg.wGateDtype, ge::FORMAT_ND},
        {WeightShape(cfg.nG, cfg.k, cfg.transG), cfg.wGDtype, ge::FORMAT_ND},
        {scale, cfg.scaleDtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {Shape2D(cfg.t, cfg.nQkv), cfg.qkvOutDtype, ge::FORMAT_ND},
        {Shape2D(cfg.t, cfg.nBeta), cfg.betaOutDtype, ge::FORMAT_ND},
        {Shape2D(cfg.t, cfg.nGate), cfg.gateOutDtype, ge::FORMAT_ND},
        {Shape2D(cfg.t, cfg.nG), cfg.gOutDtype, ge::FORMAT_ND},
    };
    const std::string socInfo = MakeSocInfo(cfg.ubSize, cfg.l1Size, cfg.socVersion);
    return gert::TilingContextPara("KdaInputProj", inputs, outputs, MakeAttrs(cfg), &gCompileInfo, cfg.socVersion,
                                   socInfo, kTilingDataSize);
}

const KdaInputProjTilingData* AsTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(KdaInputProjTilingData));
    return reinterpret_cast<const KdaInputProjTilingData*>(info.tilingData.get());
}

void CheckCommon(const TilingInfo& info, const ShapeCfg& cfg)
{
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(cfg.transQkv, cfg.transBeta, cfg.transGate, cfg.transG));
    EXPECT_GT(info.blockNum, 0U);
    const auto* td = AsTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->baseParams.tSize, static_cast<uint32_t>(cfg.t));
    EXPECT_EQ(td->baseParams.hiddenSize, static_cast<uint32_t>(cfg.k));
    EXPECT_EQ(td->baseParams.qkvSize, static_cast<uint32_t>(cfg.nQkv));
    EXPECT_EQ(td->baseParams.betaSize, static_cast<uint32_t>(cfg.nBeta));
    EXPECT_EQ(td->baseParams.gateSize, static_cast<uint32_t>(cfg.nGate));
    EXPECT_EQ(td->baseParams.gSize, static_cast<uint32_t>(cfg.nG));

    EXPECT_GT(td->mmBggParams.tTile, 0U);
    EXPECT_GT(td->mmBggParams.betaTile, 0U);
    EXPECT_EQ(td->mmBggParams.betaTile, td->mmBggParams.gateTile);
    EXPECT_EQ(td->mmBggParams.betaTile, td->mmBggParams.gTile);
    EXPECT_GT(td->mmBggParams.hiddenstatesTile, 0U);
    EXPECT_GT(td->mmBggParams.hiddenstatesL0Tile, 0U);
    EXPECT_GT(td->mmBggParams.tDim, 0U);
    EXPECT_GT(td->mmBggParams.numBetaTile + td->mmBggParams.numGateTile + td->mmBggParams.numGTile, 0U);

    EXPECT_EQ(td->mxQuantParams.rowNum, cfg.t);
    EXPECT_EQ(td->mxQuantParams.colNum, cfg.k);
    EXPECT_EQ(td->mxQuantParams.blockSize, 32);
    EXPECT_GT(td->mxQuantParams.usedCoreNum, 0);
    EXPECT_GT(td->mxQuantParams.maxUbBlockNum, 0);

    EXPECT_GT(td->qmmQkvParams.baseM, 0U);
    EXPECT_GT(td->qmmQkvParams.baseN, 0U);
    EXPECT_GT(td->qmmQkvParams.baseK, 0U);
    EXPECT_GT(td->qmmQkvParams.kL1, 0U);
    EXPECT_TRUE(td->qmmQkvParams.nBufferNum == 2 || td->qmmQkvParams.nBufferNum == 3 ||
                td->qmmQkvParams.nBufferNum == 4);

    EXPECT_EQ(td->sigmoidParams.elemNum, static_cast<uint32_t>(cfg.t * cfg.nBeta));
    EXPECT_GT(td->sigmoidParams.aivNum, 0U);
    EXPECT_GT(td->sigmoidParams.blockTile, 0U);
    EXPECT_GT(td->sigmoidParams.ubTile, 0U);

    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GE(info.workspaceSizes[0], KdaInputProj::KdaInputProjWorkspace::TotalBytes(static_cast<uint32_t>(cfg.t),
                                                                                      static_cast<uint32_t>(cfg.k)));
}

bool RunSuccess(const ShapeCfg& cfg, TilingInfo& info)
{
    auto para = BuildPara(cfg);
    return ExecuteTiling(para, info);
}
} // namespace

class KdaInputProjTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KdaInputProjTilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "KdaInputProjTilingTest TearDown" << std::endl;
    }
};

TEST_F(KdaInputProjTilingTest, tiling_success_prod_all_trans_true)
{
    ShapeCfg cfg;
    cfg.t = 8;
    cfg.k = kProdK;
    cfg.nQkv = kProdQkv;
    cfg.nBeta = kProdBeta;
    cfg.nGate = kProdGate;
    cfg.nG = kProdG;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->mxQuantParams.colTileNum, 1); // full-row：K/32=224 能进默认 UB
    EXPECT_EQ(td->qmmQkvParams.bMustHitL2, 0);  // T=8 → mCnt=1
    EXPECT_EQ(td->qmmQkvParams.dbL0C, 2);       // 16x256 L0C 双缓冲放得下
}

TEST_F(KdaInputProjTilingTest, tiling_success_prod_all_trans_false)
{
    ShapeCfg cfg;
    cfg.t = 8;
    cfg.k = kProdK;
    cfg.nQkv = kProdQkv;
    cfg.nBeta = kProdBeta;
    cfg.nGate = kProdGate;
    cfg.nG = kProdG;
    cfg.transQkv = false;
    cfg.transBeta = false;
    cfg.transGate = false;
    cfg.transG = false;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
}

TEST_F(KdaInputProjTilingTest, tiling_success_all_16_tiling_keys)
{
    std::set<uint64_t> keys;
    for (int mask = 0; mask < 16; ++mask) {
        ShapeCfg cfg;
        cfg.transQkv = (mask & 1) != 0;
        cfg.transBeta = (mask & 2) != 0;
        cfg.transGate = (mask & 4) != 0;
        cfg.transG = (mask & 8) != 0;
        SCOPED_TRACE(mask);
        TilingInfo info;
        ASSERT_TRUE(RunSuccess(cfg, info));
        CheckCommon(info, cfg);
        keys.insert(static_cast<uint64_t>(info.tilingKey));
    }
    EXPECT_EQ(keys.size(), 16U);
}

TEST_F(KdaInputProjTilingTest, tiling_success_t1_min_path)
{
    ShapeCfg cfg;
    cfg.t = 1;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->sigmoidParams.elemNum, static_cast<uint32_t>(cfg.nBeta));
}

TEST_F(KdaInputProjTilingTest, tiling_success_t257_qmm_mcnt_hit_l2)
{
    // T=257 → baseM=256 → mCnt=2 → bMustHitL2=1；256x256 L0C 双缓冲放不下 → dbL0C=1
    ShapeCfg cfg;
    cfg.t = 257;
    cfg.k = kProdK;
    cfg.nQkv = kProdQkv;
    cfg.nBeta = kProdBeta;
    cfg.nGate = kProdGate;
    cfg.nG = kProdG;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->qmmQkvParams.bMustHitL2, 1);
    EXPECT_EQ(td->qmmQkvParams.dbL0C, 1);
    EXPECT_EQ(td->qmmQkvParams.baseM, 256U);
}

TEST_F(KdaInputProjTilingTest, tiling_success_t512_sigmoid_multicore)
{
    ShapeCfg cfg;
    cfg.t = 512;
    cfg.k = kProdK;
    cfg.nQkv = kProdQkv;
    cfg.nBeta = kProdBeta;
    cfg.nGate = kProdGate;
    cfg.nG = kProdG;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_GT(td->sigmoidParams.aivNum, 1U);
}

TEST_F(KdaInputProjTilingTest, tiling_success_qmm_n_align_trans_true_vs_false)
{
    // N=240：trans=true nAlign=16 → baseN=240；trans=false nAlign=32 → baseN=256
    ShapeCfg cfgTrue;
    cfgTrue.nQkv = 240;
    TilingInfo infoTrue;
    ASSERT_TRUE(RunSuccess(cfgTrue, infoTrue));
    CheckCommon(infoTrue, cfgTrue);
    EXPECT_EQ(AsTilingData(infoTrue)->qmmQkvParams.baseN, 240U);

    ShapeCfg cfgFalse = cfgTrue;
    cfgFalse.transQkv = false;
    TilingInfo infoFalse;
    ASSERT_TRUE(RunSuccess(cfgFalse, infoFalse));
    CheckCommon(infoFalse, cfgFalse);
    EXPECT_EQ(AsTilingData(infoFalse)->qmmQkvParams.baseN, 256U);
}

TEST_F(KdaInputProjTilingTest, tiling_success_mxquant_col_split_slice256)
{
    // UB=8192 → maxUbBlockNum≈24；K=1024 colBlocks=32 > 24 走列切。
    // T=1 时 row*colTile < 0.75*aiv → sliceSize 从 512 降到 256。
    ShapeCfg cfg;
    cfg.t = 1;
    cfg.k = 1024;
    cfg.ubSize = kSmallUb;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_GT(td->mxQuantParams.colTileNum, 1);
    EXPECT_EQ(td->mxQuantParams.colNum, 1024);
    // sliceSize=256 → colNormalBlockNum 以 256 为单位，整除后为 1
    EXPECT_EQ(td->mxQuantParams.colNormalBlockNum, 1);
}

TEST_F(KdaInputProjTilingTest, tiling_success_mxquant_col_split_slice512)
{
    // T=8、K=2048：8 * CeilDiv(2048,512)=32 >= 0.75*40，保持 512 切片。
    // SplitCores 对 8×4 网格会切列（colTileNum=4），colNormalBlockNum=512/256=2。
    ShapeCfg cfg;
    cfg.t = 8;
    cfg.k = 2048;
    cfg.ubSize = kSmallUb;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_GT(td->mxQuantParams.colTileNum, 1);
    EXPECT_EQ(td->mxQuantParams.colNormalBlockNum, 2);
}

TEST_F(KdaInputProjTilingTest, tiling_success_qmm_nbuffer_2_3_4)
{
    // N=256、K=256：kL1 取满 256。4/3/2 buffer 的 L1 占用约 283KB / 213KB / 144KB。
    struct Case {
        uint64_t l1Size;
        uint8_t nBuffer;
    };
    const Case cases[] = {{180000, 2}, {250000, 3}, {kDefaultL1, 4}};
    for (const auto& c : cases) {
        ShapeCfg cfg;
        cfg.nQkv = 256;
        cfg.l1Size = c.l1Size;
        SCOPED_TRACE(c.l1Size);
        TilingInfo info;
        ASSERT_TRUE(RunSuccess(cfg, info));
        CheckCommon(info, cfg);
        EXPECT_EQ(AsTilingData(info)->qmmQkvParams.nBufferNum, c.nBuffer);
    }
}

TEST_F(KdaInputProjTilingTest, tiling_success_mmbgg_trans_mismatch_warning)
{
    ShapeCfg cfg;
    cfg.transQkv = true;
    cfg.transBeta = false;
    cfg.transGate = true;
    cfg.transG = false;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
}

TEST_F(KdaInputProjTilingTest, tiling_success_default_attrs_trans_true)
{
    ShapeCfg cfg;
    cfg.withAttrs = false;
    cfg.transQkv = true;
    cfg.transBeta = true;
    cfg.transGate = true;
    cfg.transG = true;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
}

TEST_F(KdaInputProjTilingTest, tiling_success_k64_min_hidden)
{
    ShapeCfg cfg;
    cfg.k = 64;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
}

TEST_F(KdaInputProjTilingTest, tiling_success_t225_t240_qmm_l1_regression)
{
    for (int64_t t : {225, 240}) {
        ShapeCfg cfg;
        cfg.t = t;
        cfg.k = kProdK;
        cfg.nQkv = kProdQkv;
        cfg.nBeta = kProdBeta;
        cfg.nGate = kProdGate;
        cfg.nG = kProdG;
        SCOPED_TRACE(t);
        TilingInfo info;
        ASSERT_TRUE(RunSuccess(cfg, info));
        CheckCommon(info, cfg);
    }
}

TEST_F(KdaInputProjTilingTest, tiling_fail_soc_910b)
{
    ShapeCfg cfg;
    cfg.socVersion = "Ascend910B";
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_x_dtype)
{
    ShapeCfg cfg;
    cfg.xDtype = ge::DT_FLOAT16;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_weight_qkv_dtype)
{
    ShapeCfg cfg;
    cfg.wQkvDtype = ge::DT_BF16;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_scale_dtype)
{
    ShapeCfg cfg;
    cfg.scaleDtype = ge::DT_FLOAT;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_x_rank3)
{
    ShapeCfg cfg;
    cfg.xRankExtra = 1;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_t_zero)
{
    ShapeCfg cfg;
    cfg.t = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_hidden_not_multiple_of_32)
{
    ShapeCfg cfg;
    cfg.k = 48;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_scale_layout_mismatch)
{
    ShapeCfg cfg;
    cfg.mismatchScale = true;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_scale_pack_dim)
{
    ShapeCfg cfg;
    cfg.scaleDim2 = 1;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_zero_qkv_features)
{
    ShapeCfg cfg;
    cfg.nQkv = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_zero_beta_features)
{
    ShapeCfg cfg;
    cfg.nBeta = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_zero_gate_features)
{
    ShapeCfg cfg;
    cfg.nGate = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_zero_g_features)
{
    ShapeCfg cfg;
    cfg.nG = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_hidden_zero)
{
    ShapeCfg cfg;
    cfg.k = 0;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_weight_beta_dtype)
{
    ShapeCfg cfg;
    cfg.wBetaDtype = ge::DT_FLOAT;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_weight_gate_dtype)
{
    ShapeCfg cfg;
    cfg.wGateDtype = ge::DT_FLOAT16;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_weight_g_dtype)
{
    ShapeCfg cfg;
    cfg.wGDtype = ge::DT_FLOAT;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_qkv_out_dtype)
{
    ShapeCfg cfg;
    cfg.qkvOutDtype = ge::DT_FLOAT;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_beta_out_dtype)
{
    ShapeCfg cfg;
    cfg.betaOutDtype = ge::DT_BF16;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_gate_out_dtype)
{
    ShapeCfg cfg;
    cfg.gateOutDtype = ge::DT_FLOAT;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_g_out_dtype)
{
    ShapeCfg cfg;
    cfg.gOutDtype = ge::DT_FLOAT16;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_weight_qkv_rank3)
{
    ShapeCfg cfg;
    cfg.weightRankExtra = 1;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_scale_rank2)
{
    ShapeCfg cfg;
    cfg.scaleRank = 2;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_ub_too_small_for_mxquant)
{
    // RESERVED_UB=2048，budget=0 → MxQuant CalcMaxUbBlockNum 失败
    ShapeCfg cfg;
    cfg.ubSize = 2048;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_fail_l1_too_small_for_qmm)
{
    ShapeCfg cfg;
    cfg.nQkv = 256;
    cfg.l1Size = 8192;
    ExecuteTestCase(BuildPara(cfg), ge::GRAPH_FAILED);
}

TEST_F(KdaInputProjTilingTest, tiling_success_mxquant_col_split_col_tail)
{
    // K=1088 非 256 整除，列切后 colTailLen 覆盖尾块
    ShapeCfg cfg;
    cfg.t = 1;
    cfg.k = 1088;
    cfg.ubSize = kSmallUb;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_GT(td->mxQuantParams.colTileNum, 1);
    EXPECT_GT(td->mxQuantParams.colTailLen, 0);
    EXPECT_NE(td->mxQuantParams.colTailLen, td->mxQuantParams.colNum);
}

TEST_F(KdaInputProjTilingTest, tiling_success_t41_fullrow_row_tail)
{
    // T=41、K=256 走 full-row；rowNormal=2、rowTail=1
    ShapeCfg cfg;
    cfg.t = 41;
    cfg.k = kSmallK;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    const auto* td = AsTilingData(info);
    EXPECT_EQ(td->mxQuantParams.colTileNum, 1);
    EXPECT_EQ(td->mxQuantParams.rowNormalBlockNum, 2);
    EXPECT_EQ(td->mxQuantParams.rowTailLen, 1);
}

TEST_F(KdaInputProjTilingTest, tiling_success_qmm_n_tail_241)
{
    // N=241 trans=true → nAlign=16 → baseN=256，覆盖 QMM N 尾块
    ShapeCfg cfg;
    cfg.nQkv = 241;
    TilingInfo info;
    ASSERT_TRUE(RunSuccess(cfg, info));
    CheckCommon(info, cfg);
    EXPECT_EQ(AsTilingData(info)->qmmQkvParams.baseN, 256U);
}
