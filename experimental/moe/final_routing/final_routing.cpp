/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace npu_ops_transformer_ext {
namespace FinalRouting {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t UB_MAX_BYTES = 184*1024;
constexpr int64_t BUFFER_NUM = 1;

class final_routing {
public:
    __aicore__ inline final_routing() {}

    __aicore__ inline void Init(GM_ADDR in, GM_ADDR token_table, GM_ADDR score_table, GM_ADDR out,
            const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        gbExpertNum_ = expert_num;
        gbTokenNum_  = token_num;
        gbCopyByte_  = copy_byte;
        gbCopyNum_   = gbCopyByte_ / sizeof(bfloat16_t);

        gbTokenNumAlign_  = AlignUp(gbTokenNum_, 64 / sizeof(int32_t));
        gbExpertNumAlign_ = AlignUp(gbExpertNum_, 64 / sizeof(int32_t));

        bkExpertNum_ = 1;
        bkTokenNum_  = 1;
        bkCopyByte_         = gbCopyByte_;
        bkCopyNum_          = bkCopyByte_ / sizeof(bfloat16_t);
        bkCopyByteAlign_    = AlignUp(bkCopyByte_, 64);
        bkCopyNumAlign_     = bkCopyByteAlign_ / sizeof(bfloat16_t);

        gmIn_.SetGlobalBuffer((__gm__ bfloat16_t*)(in), gbExpertNum_ * gbTokenNum_ * gbCopyNum_);
        gmTokenTable_.SetGlobalBuffer((__gm__ int32_t*)(token_table), gbTokenNum_ * gbExpertNum_);
        gmScoreTable_.SetGlobalBuffer((__gm__ bfloat16_t*)(score_table), gbTokenNum_ * gbExpertNum_);
        gmOut_.SetGlobalBuffer((__gm__ bfloat16_t*)(out), gbTokenNum_ * gbCopyNum_);

        pipe_.InitBuffer(inQueIn_, BUFFER_NUM, bkCopyByteAlign_ * 2);
        pipe_.InitBuffer(inQueTokenTable_, BUFFER_NUM, gbExpertNumAlign_ * sizeof(int32_t));
        pipe_.InitBuffer(inQueScoreTable_, BUFFER_NUM, gbExpertNumAlign_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(outQueOut_,  BUFFER_NUM, bkCopyByteAlign_ * 2);
    }

    __aicore__ inline void Process()
    {
        for (int64_t tis = 0; tis < gbTokenNum_; tis += blockNum_) {
            int64_t token_idx = tis + blockIdx_;

            if (token_idx < gbTokenNum_) {
                LocalTensor<int32_t> local_token_table = inQueTokenTable_.AllocTensor<int32_t>();
                LocalTensor<bfloat16_t> local_score_table = inQueScoreTable_.AllocTensor<bfloat16_t>();

                DataCopyParams copy_pas{1, (uint16_t)(gbExpertNum_ * sizeof(int32_t)), 0, 0};
                DataCopyPadParams pad_pas;

                DataCopyPad(local_token_table, gmTokenTable_[token_idx * gbExpertNum_], copy_pas, pad_pas);
                copy_pas.blockLen = (uint16_t)(gbExpertNum_ * sizeof(bfloat16_t));
                DataCopyPad(local_score_table, gmScoreTable_[token_idx * gbExpertNum_], copy_pas, pad_pas);

                inQueTokenTable_.EnQue(local_token_table);
                inQueScoreTable_.EnQue(local_score_table);

                lmTokenTable_ = inQueTokenTable_.DeQue<int32_t>();
                lmScoreTable_ = inQueScoreTable_.DeQue<bfloat16_t>();

                lmOut_ = outQueOut_.AllocTensor<bfloat16_t>();
                lmOutFloat_ = lmOut_.ReinterpretCast<float>();
                Duplicate(lmOutFloat_, (float)0.0, bkCopyNumAlign_);

                for (int64_t expert_idx = 0; expert_idx < gbExpertNum_; expert_idx++) {
                    int64_t routing_token_idx = lmTokenTable_.GetValue(expert_idx);
                    if (routing_token_idx >= 0) {
                        CopyIn(routing_token_idx);
                        Compute(AscendC::ToFloat(lmScoreTable_.GetValue(expert_idx)));
                    }
                }

                Cast(lmOut_, lmOutFloat_, RoundMode::CAST_ROUND, bkCopyNum_);
                outQueOut_.EnQue(lmOut_);
                outQueOut_.DeQue<bfloat16_t>();
                CopyOut(token_idx);

                inQueTokenTable_.FreeTensor(local_token_table);
                inQueScoreTable_.FreeTensor(local_score_table);
                outQueOut_.FreeTensor(lmOut_);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t routing_token_idx)
    {
        LocalTensor<bfloat16_t> local_in = inQueIn_.AllocTensor<bfloat16_t>();

        DataCopyExtParams copy_pas{1, (uint32_t)(bkCopyByte_), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad_pas;

        int64_t offset = routing_token_idx * gbCopyNum_;
        DataCopyPad(local_in[bkCopyNumAlign_], gmIn_[offset], copy_pas, pad_pas);
        inQueIn_.EnQue(local_in);
    }

    __aicore__ inline void Compute(float score)
    {
        LocalTensor<bfloat16_t> local_in = inQueIn_.DeQue<bfloat16_t>();
        LocalTensor<float> local_in_float = local_in.ReinterpretCast<float>();

        Cast(local_in_float, local_in[bkCopyNumAlign_], RoundMode::CAST_NONE, bkCopyNum_);

        Muls(local_in_float, local_in_float, score, bkCopyNum_);
        Add(lmOutFloat_, lmOutFloat_, local_in_float, bkCopyNum_);

        inQueIn_.FreeTensor(local_in);
    }

    __aicore__ inline void CopyOut(int64_t token_idx)
    {
        DataCopyExtParams copy_pas{1, (uint32_t)(bkCopyByte_), 0, 0, 0};
        DataCopyPad(gmOut_[token_idx * bkCopyNum_], lmOut_, copy_pas);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_, inQueTokenTable_, inQueScoreTable_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;

    GlobalTensor<bfloat16_t> gmIn_, gmOut_;
    GlobalTensor<int32_t> gmTokenTable_;
    GlobalTensor<bfloat16_t> gmScoreTable_;

    LocalTensor<int32_t> lmTokenTable_;
    LocalTensor<bfloat16_t> lmScoreTable_;
    LocalTensor<bfloat16_t> lmOut_;
    LocalTensor<float> lmOutFloat_;

    int64_t blockNum_, blockIdx_;

    int64_t gbExpertNum_, gbTokenNum_, gbCopyByte_, gbCopyNum_;
    int64_t gbExpertNumAlign_, gbTokenNumAlign_;

    int64_t bkExpertNum_, bkTokenNum_, bkCopyByte_, bkCopyNum_;
    int64_t bkCopyByteAlign_, bkCopyNumAlign_;
};

extern "C" __global__ __aicore__ void compute_final_routing(
        GM_ADDR in, GM_ADDR token_table, GM_ADDR score_table, GM_ADDR out,
        const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
{
    final_routing op;
    op.Init(in, token_table, score_table, out, expert_num, token_num, copy_byte);
    op.Process();
}

void final_routing_kernel_lanuch(int64_t block_dim, void* stream,
    uint8_t* in, uint8_t* token_table, uint8_t* score_table, uint8_t* out,
    const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
{
    compute_final_routing<<<block_dim, nullptr, stream>>>(in, token_table, score_table, out,
            expert_num, token_num, copy_byte);
}

inline int64_t align_up(const int64_t number, const int64_t alignSize)
{
    if (number % alignSize == 0) {
        return number;
    }

    return ((number / alignSize + 1) * alignSize);
}


int judge_final_routing_lanuch(const int64_t expert_num, const int64_t token_num, const int64_t copy_byte,
    const int64_t ubSize, const int64_t vCores)
{
    (void)(vCores);

    constexpr int64_t BUFFER_NUM = 1;

    int64_t gbExpertNum_;
    int64_t gbTokenNum_;
    int64_t gbCopyByte_;
    int64_t gbExpertNumAlign_;
    int64_t gbTokenNumAlign_;

    int64_t bkCopyByte_;
    int64_t bkCopyByteAlign_;

    gbExpertNum_ = expert_num;
    gbTokenNum_  = token_num;
    gbCopyByte_  = copy_byte;

    gbTokenNumAlign_  = align_up(gbTokenNum_, 64 / sizeof(int32_t));
    gbExpertNumAlign_ = align_up(gbExpertNum_, 64 / sizeof(int32_t));

    bkCopyByte_  = gbCopyByte_;
    bkCopyByteAlign_ = align_up(bkCopyByte_, 64);

    float use_byte = BUFFER_NUM * bkCopyByteAlign_ * 2;
    use_byte += gbTokenNumAlign_ * sizeof(int32_t);
    use_byte += gbExpertNumAlign_ * sizeof(int32_t);
    use_byte += gbExpertNumAlign_ * sizeof(int16_t);

    if (use_byte > ubSize) {
        std::cout << __FUNCTION__ << ": " << use_byte/1024 << " KB" << std::endl;
        return 1;
    }
    return 0;
}

int final_routing_lanuch(int64_t block_dim, void* stream,
    uint8_t* in, uint8_t* token_table, uint8_t* score_table, uint8_t* out,
    const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
{
    int64_t ubSize = 184 * 1024;
    int64_t vCores = 40;

    int ret = judge_final_routing_lanuch(expert_num, token_num, copy_byte, ubSize, vCores);
    if (ret == 0) {
        final_routing_kernel_lanuch(block_dim, stream, in, token_table, score_table,
            out, expert_num, token_num, copy_byte);
        return 0;
    }

    std::cout << __FUNCTION__ << ": " << "UB size is limited, please check!" << std::endl;
    return 1;
}

int64_t final_routing_npu(int64_t block_dim, torch::Tensor &in, torch::Tensor &token_table,
    torch::Tensor &score_table, torch::Tensor &out)
{
    TORCH_CHECK(torch_npu::utils::is_npu(in), "input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(token_table), "token table tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(score_table), "score table tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(out), "output tensor must be on NPU device");

    const int64_t token_num = token_table.size(0);
    const int64_t expert_num = token_table.size(1);
    const int64_t copy_byte = in.element_size() * in.size(1);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int launchStatus = 0;
    auto acl_call = [=, &launchStatus]() -> int {
        launchStatus = final_routing_lanuch(block_dim, stream, (uint8_t *)in.data_ptr(),
        (uint8_t *)token_table.data_ptr(), (uint8_t *)score_table.data_ptr(), (uint8_t *)out.data_ptr(),
        expert_num, token_num, copy_byte);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("FinalRouting", acl_call);
    return launchStatus;
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("final_routing", final_routing_npu);
}
}
}
