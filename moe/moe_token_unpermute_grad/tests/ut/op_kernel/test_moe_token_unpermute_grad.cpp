/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "test_moe_token_unpermute_grad.h"

extern "C" __global__ __aicore__ void moe_token_unpermute_grad(GM_ADDR permuted_tokens, GM_ADDR unpermuted_output_d,
                                                               GM_ADDR sorted_indices, GM_ADDR probs,
                                                               GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                                               GM_ADDR workspace, GM_ADDR tiling);

namespace {
class MoeTokenUnpermuteGradTest : public testing::Test {
protected:
    std::string dataDir;
    std::vector<uint8_t*> buffers;

    void SetUp() override
    {
        char directory[] = "/tmp/moe_grad_ut_XXXXXX";
        char* created = mkdtemp(directory);
        ASSERT_NE(created, nullptr);
        dataDir = created;
    }

    void TearDown() override
    {
        for (auto* buffer : buffers) {
            AscendC::GmFree(buffer);
        }
        if (dataDir.empty()) {
            return;
        }
        for (const auto* name : {"input_permuted_tokens.bin", "input_sorted_indices.bin", "input_probs.bin",
                                 "input_unpermuted_output_d.bin", "output_unpermuted_tokens.bin",
                                 "output_permuted_tokens_grad.bin", "output_probs_grad.bin"}) {
            std::remove((dataDir + "/" + name).c_str());
        }
        rmdir(dataDir.c_str());
    }

    uint8_t* Allocate(size_t bytes)
    {
        auto* buffer = static_cast<uint8_t*>(AscendC::GmAlloc(bytes + 32));
        if (buffer != nullptr) {
            // An unwritten output must fail comparison rather than look like a zero golden.
            std::memset(buffer, 0xA5, bytes + 32);
            buffers.push_back(buffer);
        }
        return buffer;
    }

    bool Load(const std::string& name, void* buffer, size_t bytes)
    {
        size_t actual = 0;
        return buffer != nullptr && ReadFile(dataDir + "/" + name, actual, buffer, bytes) && actual == bytes;
    }

    template <typename T>
    void Check(const std::string& name, const uint8_t* output, size_t count)
    {
        std::vector<T> expected(count);
        ASSERT_TRUE(Load(name, expected.data(), count * sizeof(T))) << name;
        const auto* actual = reinterpret_cast<const T*>(output);
        for (size_t i = 0; i < count; ++i) {
            ASSERT_EQ(static_cast<float>(actual[i]), static_cast<float>(expected[i])) << name << " at " << i;
        }
    }

    void Run(bool haveProbs, int64_t topK, int64_t hiddenSize)
    {
        constexpr int64_t TOKENS = 5;
        constexpr uint32_t BLOCK_DIM = 2;
        const int64_t total = TOKENS * topK;
        const std::string source(__FILE__);
        const std::string script =
            source.substr(0, source.find_last_of('/')) + "/moe_token_unpermute_grad_data/gen_data.py";
        const std::string command = "python3 \"" + script + "\" " + std::to_string(TOKENS) + " " +
                                    std::to_string(topK) + " " + std::to_string(hiddenSize) + " bfloat16 float32 " +
                                    (haveProbs ? "True" : "False") + " --output-dir \"" + dataDir + "\"";
        ASSERT_EQ(std::system(command.c_str()), 0) << command;

        const size_t tokenBytes = total * hiddenSize * sizeof(bfloat16_t);
        const size_t gradBytes = TOKENS * hiddenSize * sizeof(bfloat16_t);
        auto* tokens = Allocate(tokenBytes);
        auto* outputGrad = Allocate(gradBytes);
        auto* indices = Allocate(total * sizeof(int32_t));
        auto* probs = Allocate(total * sizeof(float));
        auto* tokenGrad = Allocate(tokenBytes);
        auto* probsGrad = Allocate(total * sizeof(float));
        auto* workspace = Allocate(32);
        auto* tiling = Allocate(sizeof(MoeTokenUnpermuteGradTilingData));
        ASSERT_NE(tokenGrad, nullptr);
        ASSERT_NE(probsGrad, nullptr);
        ASSERT_NE(workspace, nullptr);
        ASSERT_NE(tiling, nullptr);
        ASSERT_TRUE(Load("input_permuted_tokens.bin", tokens, tokenBytes));
        ASSERT_TRUE(Load("input_unpermuted_output_d.bin", outputGrad, gradBytes));
        ASSERT_TRUE(Load("input_sorted_indices.bin", indices, total * sizeof(int32_t)));
        if (haveProbs) {
            ASSERT_TRUE(Load("input_probs.bin", probs, total * sizeof(float)));
        }

        auto& data = *reinterpret_cast<MoeTokenUnpermuteGradTilingData*>(tiling);
        data = {};
        data.tokensNum = TOKENS;
        data.topK = topK;
        data.hiddenSize = hiddenSize;
        data.numOutTokens = total;
        data.formerCoreNum = TOKENS % BLOCK_DIM;
        data.tailCoreNum = BLOCK_DIM - data.formerCoreNum;
        data.tokenNumTailCore = TOKENS / BLOCK_DIM;
        data.tokenNumEachCore = data.tokenNumTailCore + 1;
        data.rowIdMapEachCore = data.tokenNumEachCore * topK;
        data.rowIdMapTailCore = data.tokenNumTailCore * topK;
        // BF16 hidden slices reserve 512 bytes, as in the host tiler. The
        // reduction also addresses the next vector when its valid count is zero.
        data.hiddenSizeAlign = 256;
        data.hiddenSizeLoopTimes = (hiddenSize + data.hiddenSizeAlign - 1) / data.hiddenSizeAlign;
        data.hiddenSizeTail = hiddenSize - (data.hiddenSizeLoopTimes - 1) * data.hiddenSizeAlign;
        data.indicesReserveNum = haveProbs ? 2 * topK : 8;
        data.inputReserveNum = haveProbs ? 1 : data.indicesReserveNum;
        data.indicesReserveNumAlign = (data.indicesReserveNum + 15) / 16 * 16;
        data.totalUbSize = 196352;
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(haveProbs ? 1 : 0);
        ICPU_RUN_KF(moe_token_unpermute_grad, BLOCK_DIM, tokens, outputGrad, indices, haveProbs ? probs : nullptr,
                    tokenGrad, haveProbs ? probsGrad : nullptr, workspace, tiling);
        Check<bfloat16_t>("output_permuted_tokens_grad.bin", tokenGrad, total * hiddenSize);
        if (haveProbs) {
            Check<float>("output_probs_grad.bin", probsGrad, total);
        }
    }
};

// The test adapter compiles BF16 tokens and FP32 probabilities. Every allocation
// and generated file must match those kernel template arguments.
TEST_F(MoeTokenUnpermuteGradTest, Bf16WithoutProbs)
{
    Run(false, 1, 64);
}

TEST_F(MoeTokenUnpermuteGradTest, Bf16WithoutProbsHiddenTail)
{
    Run(false, 1, 270);
}

TEST_F(MoeTokenUnpermuteGradTest, Bf16WithFp32Probs)
{
    Run(true, 3, 64);
}

TEST_F(MoeTokenUnpermuteGradTest, Bf16WithFp32ProbsHiddenTail)
{
    Run(true, 3, 270);
}
} // namespace
