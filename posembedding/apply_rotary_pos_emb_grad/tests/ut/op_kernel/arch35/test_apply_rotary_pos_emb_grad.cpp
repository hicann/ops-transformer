/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "test_apply_rotary_pos_emb_grad.h"

// kernel 入口为模板函数且源文件带 _apt 后缀(UT 框架按 op_kernel/{op_name}.cpp 无法 glob 到),
// 由测试编译单元直接包含 kernel 源文件完成模板实例化
#include "../../../../op_kernel/apply_rotary_pos_emb_grad_apt.cpp"

using namespace std;

namespace {
// ApplyRopeGradDxTilingKey: BAB=203 / AB=204 / A=205 (见 op_host tiling 头)
constexpr uint32_t TILING_KEY_AB = 204;
// ReduceSch 模板实参: 连续布局 / 非 BatchInvariant / ARA(20) 中轴 reduce;
// LoopARCount 十进制编码 loopACount(高位)=1, loopRCount(低位)=0 → ProcessNormal + A 轴分核
// (factorATotalCnt=64 块 / 32 核, 见 gen_tiling.py), 与 reduceTiling.shape=[128,4,D] 排布一致
constexpr uint32_t REDUCE_PATTERN_ARA = 20;
constexpr uint32_t LOOP_AR_COUNT = 10;
constexpr uint32_t LOOP_INNER_AR_COUNT = 0;
constexpr uint32_t DCOS_FLAG = 1;

// kernel 共 12 个 GM_ADDR 参数
using ApplyRopeGradKernelFunc = void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
                                         GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);

// AB 模板(SBND, DxTilingKey=204) + dCosFlag=1 用例公共流程:
//   B=2 S=64 nQ=nK=4, cos/sin=[B,S,1,D] 与 grad 的 B/S 轴一致(N 轴广播) → 内部 layout=SBND
//   blockDim = ropeGradABParams.blockNumBS * blockNumN = 43 * 1 = 43
void RunAbSbndCase(const string &caseName, int64_t dimD)
{
    const int64_t B = 2;
    const int64_t S = 64;
    const int64_t N = 4;
    const int64_t D = dimD;
    size_t gradByteSize = B * S * N * D * sizeof(float); // grad_query/key_embed 与 query/key
    // cos/sin 及 grad_cos/grad_sin: [B, S, 1, D] (AB/SBND 模板, cos 的 B 轴与 grad 一致非广播)
    size_t cosByteSize = B * S * 1 * D * sizeof(float);
    size_t tilingDataSize = sizeof(ApplyRopeGradTilingData);

    uint8_t *gradQueryEmbed = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *gradKeyEmbed = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *cos = (uint8_t *)AscendC::GmAlloc(cosByteSize);
    uint8_t *sin = (uint8_t *)AscendC::GmAlloc(cosByteSize);
    uint8_t *query = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *key = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *gradQueryOut = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *gradKeyOut = (uint8_t *)AscendC::GmAlloc(gradByteSize);
    uint8_t *gradCosOut = (uint8_t *)AscendC::GmAlloc(cosByteSize);
    uint8_t *gradSinOut = (uint8_t *)AscendC::GmAlloc(cosByteSize);
    // workspace: dcos/dsin 两份 fp32 部分积(B*S*maxN*D*4B*2) + reduce 暂存, 预留 16MB
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 43; // ropeGradABParams.blockNumBS * ropeGradABParams.blockNumN

    // 生成输入与 tiling (gen_data.py: dx/dy→grad_query/key_embed, x/y→query/key; cos/sin=[B,S,1,D])
    string cmdPrefix = "cd ./apply_rotary_pos_emb_grad_data/ && ";
    system("cp -r ../../../../../posembedding/apply_rotary_pos_emb_grad/tests/ut/op_kernel/arch35/"
           "apply_rotary_pos_emb_grad_data ./");
    system("chmod -R 755 ./apply_rotary_pos_emb_grad_data/");
    system("cd ./apply_rotary_pos_emb_grad_data/ && rm -rf ./*bin");
    system((cmdPrefix + "python3 gen_data.py 2 64 4 " + to_string(D) + " float32").c_str());
    system((cmdPrefix + "python3 gen_tiling.py " + caseName).c_str());
    char *path_ = get_current_dir_name();
    string path(path_);
    ApplyRopeGradTilingData *tilingDatafromBin = reinterpret_cast<ApplyRopeGradTilingData *>(tiling);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/dx.bin", gradByteSize, gradQueryEmbed, gradByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/dy.bin", gradByteSize, gradKeyEmbed, gradByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/cos.bin", cosByteSize, cos, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/sin.bin", cosByteSize, sin, cosByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/x.bin", gradByteSize, query, gradByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/y.bin", gradByteSize, key, gradByteSize);
    ReadFile(path + "/apply_rotary_pos_emb_grad_data/tiling.bin", tilingDataSize, tilingDatafromBin, tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // 模板实参列表含逗号, 直接展开进 ICPU_RUN_KF 宏会破坏宏参数解析, 先绑定到函数指针
    ApplyRopeGradKernelFunc kernelFunc = apply_rotary_pos_emb_grad<true, false, REDUCE_PATTERN_ARA, LOOP_AR_COUNT,
                                                                   LOOP_INNER_AR_COUNT, TILING_KEY_AB, DCOS_FLAG>;
    ICPU_SET_TILING_KEY(TILING_KEY_AB);
    ICPU_RUN_KF(kernelFunc, blockDim, gradQueryEmbed, gradKeyEmbed, cos, sin, query, key, gradQueryOut, gradKeyOut,
                gradCosOut, gradSinOut, workspace, (uint8_t *)(tilingDatafromBin));

    // 输出落盘, 供与 numpy/python golden 比对
    WriteFile(path + "/apply_rotary_pos_emb_grad_data/grad_query_out.bin", gradQueryOut, gradByteSize);
    WriteFile(path + "/apply_rotary_pos_emb_grad_data/grad_key_out.bin", gradKeyOut, gradByteSize);
    WriteFile(path + "/apply_rotary_pos_emb_grad_data/grad_cos_out.bin", gradCosOut, cosByteSize);
    WriteFile(path + "/apply_rotary_pos_emb_grad_data/grad_sin_out.bin", gradSinOut, cosByteSize);

    AscendC::GmFree(gradQueryEmbed);
    AscendC::GmFree(gradKeyEmbed);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(query);
    AscendC::GmFree(key);
    AscendC::GmFree(gradQueryOut);
    AscendC::GmFree(gradKeyOut);
    AscendC::GmFree(gradCosOut);
    AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
} // namespace

class apply_rotary_pos_emb_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        cout << "apply_rotary_pos_emb_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "apply_rotary_pos_emb_grad_test TearDown\n" << endl;
    }
};

// case0: D=48 (非 32B 对齐, kernel 走 dAlign=64 padding 路径)
TEST_F(apply_rotary_pos_emb_grad_test, test_case0_ab_sbnd_fp32_d48_dcos1)
{
    RunAbSbndCase("case0", 48);
}

// case1: D=64 (天然 32B 对齐, 无 dAlign padding), 与 case0 对照隔离 padding 路径差异
TEST_F(apply_rotary_pos_emb_grad_test, test_case1_ab_sbnd_fp32_d64_dcos1)
{
    RunAbSbndCase("case1", 64);
}
