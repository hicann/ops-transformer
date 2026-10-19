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
#include "test_moe_token_permute_with_ep_grad.h"

extern "C" __global__ __aicore__ void moe_token_permute_with_ep_grad(GM_ADDR permuted_output_grad,
                                                                     GM_ADDR sorted_indices,
                                                                     GM_ADDR permuted_probs_output_grad,
                                                                     GM_ADDR token_grad, GM_ADDR probs_grad,
                                                                     GM_ADDR workspace, GM_ADDR tiling);

namespace {
class MoeTokenPermuteWithEpGradTest : public testing::Test {
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

    void Run(bool haveProbs, int64_t topK, int64_t hiddenSize, int64_t start = 0, int64_t end = -1)
    {
        constexpr int64_t TOKENS = 5;
        constexpr uint32_t BLOCK_DIM = 2;
        const int64_t total = TOKENS * topK;
        end = end < 0 ? total : end;
        const std::string source(__FILE__);
        const std::string script =
            source.substr(0, source.find_last_of('/')) + "/moe_token_permute_with_ep_grad_data/gen_data.py";
        const std::string command = "python3 \"" + script + "\" " + std::to_string(TOKENS) + " " +
                                    std::to_string(topK) + " " + std::to_string(hiddenSize) + " bfloat16 " +
                                    (haveProbs ? "True" : "False") + " --start " + std::to_string(start) + " --end " +
                                    std::to_string(end) + " --output-dir \"" + dataDir + "\"";
        ASSERT_EQ(std::system(command.c_str()), 0) << command;

        const size_t inputBytes = (end - start) * hiddenSize * sizeof(bfloat16_t);
        auto* tokens = Allocate(inputBytes);
        auto* indices = Allocate(total * sizeof(int32_t));
        auto* probs = Allocate((end - start) * sizeof(bfloat16_t));
        auto* tokenGrad = Allocate(TOKENS * hiddenSize * sizeof(bfloat16_t));
        auto* probsGrad = Allocate(total * sizeof(bfloat16_t));
        auto* workspace = Allocate(32);
        auto* tiling = Allocate(sizeof(MoeTokenPermuteWithEpGradTilingData));
        ASSERT_NE(tokenGrad, nullptr);
        ASSERT_NE(probsGrad, nullptr);
        ASSERT_NE(workspace, nullptr);
        ASSERT_NE(tiling, nullptr);
        ASSERT_TRUE(Load("input_permuted_tokens.bin", tokens, inputBytes));
        ASSERT_TRUE(Load("input_sorted_indices.bin", indices, total * sizeof(int32_t)));
        if (haveProbs) {
            ASSERT_TRUE(Load("input_probs.bin", probs, (end - start) * sizeof(bfloat16_t)));
        }

        auto& data = *reinterpret_cast<MoeTokenPermuteWithEpGradTilingData*>(tiling);
        data = {};
        data.hidden_size = hiddenSize;
        data.permuted_probs_grad_length = 1;
        data.top_k = topK;
        data.start = start;
        data.end = end;
        data.num_out_tokens = TOKENS;
        data.hidden_splited_length = std::min<int64_t>(hiddenSize, 64);
        data.hidden_splited_num = hiddenSize / data.hidden_splited_length;
        data.hidden_splited_remain = hiddenSize % data.hidden_splited_length;
        data.tokens_core_length = TOKENS / BLOCK_DIM;
        data.tokens_core_remain = TOKENS % BLOCK_DIM;
        data.tokens_splited_length = data.tokens_core_length + 1;
        data.tokens_splited_num = 0;
        data.tokens_splited_remain = data.tokens_core_length;
        data.buffer_num = 2;
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(haveProbs ? 1 : 0);
        ICPU_RUN_KF(moe_token_permute_with_ep_grad, BLOCK_DIM, tokens, indices, haveProbs ? probs : nullptr, tokenGrad,
                    haveProbs ? probsGrad : nullptr, workspace, tiling);
        Check<bfloat16_t>("output_unpermuted_tokens.bin", tokenGrad, TOKENS * hiddenSize);
        if (haveProbs) {
            Check<bfloat16_t>("output_probs_grad.bin", probsGrad, total);
        }
    }
};

TEST_F(MoeTokenPermuteWithEpGradTest, Bf16WithoutProbs)
{
    Run(false, 1, 64);
}

TEST_F(MoeTokenPermuteWithEpGradTest, Bf16WithProbs)
{
    Run(true, 3, 64);
}

TEST_F(MoeTokenPermuteWithEpGradTest, Bf16RangeAndHiddenTail)
{
    Run(true, 3, 70, 2, 13);
}
} // namespace
