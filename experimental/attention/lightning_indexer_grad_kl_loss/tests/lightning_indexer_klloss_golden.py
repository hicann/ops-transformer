# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import os
import importlib

importlib.import_module(
    os.environ.get("CANN_OPS_TRANSFORMER_PACKAGE", "cann_ops_transformer")
)
# import torchair
# import custom_ops
import numpy as np
import torch.nn as nn
import math
import random
import hashlib

# from precision_compare import data_compare
TEST_FAILURES = []
FILL_VAL = -float("inf")
DEVICE_ID = int(os.environ.get("NPU_DEVICE", "11"))
torch_npu.npu.set_device(int(DEVICE_ID))

np.set_printoptions(
    precision=4,  # 保留小数点后4位
    linewidth=1536 / 2,  # 每行显示128个数（约1536字符）
    threshold=np.inf,  # 完整打印，不截断
    suppress=False,  # 不使用科学计数法（可选）
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


def check_result(golden_output, output, name=None):
    rtol = 5 * 1e-03
    atol = 5 * 1e-03
    diff = golden_output - output
    abs_diff = torch.abs(diff)
    print(
        "[INFO] LIG {} max diff is {}, mean diff is {}".format(
            name, torch.max(abs_diff), torch.mean(abs_diff)
        )
    )

    result_check = torch.allclose(golden_output, output, rtol=rtol, atol=atol)
    if not result_check:
        tolerance = atol + rtol * torch.abs(golden_output)
        mismatched_positions = torch.nonzero(abs_diff > tolerance)

        if golden_output.dim() == 1:
            mismatched_indices = torch.nonzero(mismatched_positions).flatten()
        else:
            mismatched_indices = torch.nonzero(mismatched_positions)

        # print(mismatched_positions.cpu().numpy())
        mismatched_golden = golden_output[mismatched_positions.split(1, dim=1)]
        mismatched_actual = output[mismatched_positions.split(1, dim=1)]
        mismatched_diffs = diff[mismatched_positions.split(1, dim=1)]

        if mismatched_diffs.numel() > (golden_output.numel() * 0.001):
            flatten_m_golden = torch.flatten(mismatched_golden)
            flatten_m_actual = torch.flatten(mismatched_actual)
            flatten_m_diff = torch.flatten(mismatched_diffs)
            print_n = min(100, mismatched_diffs.numel())
            for i in range(print_n):
                print(
                    "index: {}, expect: {}, actual: {}, diff is:{}".format(
                        i, flatten_m_golden[i], flatten_m_actual[i], flatten_m_diff[i]
                    )
                )
            print(
                "[INFO] FlashAttention BWD {} Check Acuraccy, diff num is: {}, percentage is: {}".format(
                    name,
                    mismatched_diffs.numel(),
                    mismatched_diffs.numel() * 1.0 / output.numel(),
                )
            )
            TEST_FAILURES.append(
                f"{name}: {mismatched_diffs.numel() / output.numel():.6%} exceed the original tolerance"
            )
        else:
            print("[INFO] LIG {} Check Acuraccy Pass".format(name))
    else:
        print("[INFO] LIG {} Check Acuraccy Pass".format(name))


def getCaseInfo(q, k, q_index, k_index, sparse_indices):
    D = q.shape[3]
    B = q.shape[0]
    S1 = q.shape[1]
    S2 = k.shape[1]
    N1 = q.shape[2]
    N2 = k.shape[2]
    G = N1 // N2
    Nqi = q_index.shape[2]
    Nki = k_index.shape[2]
    G_index = Nqi // Nki
    D_index = q_index.shape[3]
    topKSize = sparse_indices.shape[3]
    print("topKSize", topKSize)

    return B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize


def getTNDCaseInfo(
    q,
    k,
    q_index,
    k_index,
    sparse_indices,
    query_rope,
    key_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
):
    print(f"actual_seq_lengths_query is {actual_seq_lengths_query}")
    print(f"actual_seq_lengths_key is {actual_seq_lengths_key}")
    B = len(actual_seq_lengths_query)
    T1 = actual_seq_lengths_query[-1]
    T2 = actual_seq_lengths_key[-1]
    NQuery = q.shape[1]
    N2 = k.shape[1]
    NQueryIndex = q_index.shape[1]
    D1 = q.shape[2] if query_rope is None else q.shape[2] + query_rope.shape[2]
    D2 = q_index.shape[2]
    return B, T1, T2, NQuery, N2, NQueryIndex, D1, D2


def get_tnd_idx(acc_q_len, t_idx):
    b_idx = 0
    while t_idx >= acc_q_len[b_idx + 1]:
        b_idx += 1
    s1_offset = acc_q_len[b_idx]

    s1_idx = t_idx - s1_offset

    return b_idx, s1_idx


class LightningIndicesKLLoss:
    # input     1. query, key, topk_index, scale, max, sum
    #           2. query_index, key_index, weight
    # output    1. KLLoss
    #           2. dW, dQ, dK
    def __init__(
        self,
        q,
        k,
        topk_index,
        scale,
        layout,
        query_index,
        key_index,
        weight,
        q_rope=None,
        k_rope=None,
        actual_seq_q_len=None,
        actual_seq_kv_len=None,
        block_size=1,
    ):
        self.layout = layout
        if layout == "TND":
            self.B = len(actual_seq_q_len)
            self.N1 = q.shape[1]
            self.N2 = k.shape[1]
            self.Nqi = query_index.shape[1]
            self.Nki = key_index.shape[1]
            self.G = self.N1 // self.N2
            self.G_index = self.Nqi // self.Nki
            self.T1 = actual_seq_q_len[-1]
            self.T2 = actual_seq_kv_len[-1]
            self.D1 = q.shape[2]
            self.D2 = key_index.shape[2]
        else:  # BSND
            self.B = q.shape[0]
            self.N1 = q.shape[2]
            self.N2 = k.shape[2]
            self.Nqi = query_index.shape[2]
            self.Nki = key_index.shape[2]
            self.G = self.N1 // self.N2
            self.G_index = self.Nqi // self.Nki
            self.S1 = q.shape[1]
            self.S2 = k.shape[1]
            self.D1 = q.shape[3]
            self.D2 = key_index.shape[3]

        self.q = q
        self.k = k
        self.topk_index = topk_index
        self.scale = scale
        self.q_index = query_index
        self.k_index = key_index
        self.weight = weight
        self.block_size = block_size

        self.q_rope = q_rope
        self.k_rope = k_rope
        self.actual_seq_q_len = actual_seq_q_len
        self.actual_seq_kv_len = actual_seq_kv_len

    def get_spoint_by_bIdx(self, bIdx):
        Sq, Skv = 0, 0
        if bIdx == 0:
            Sq = self.actual_seq_q_len[0]
            Skv = self.actual_seq_kv_len[0]
        elif bIdx > 0:
            Sq = self.actual_seq_q_len[bIdx]
            Skv = self.actual_seq_kv_len[bIdx]
        return Sq, Skv

    def run_unpad(self):
        if self.layout != "TND":
            raise ValueError(
                f"[error] case layout is not invalid!, case layout is {self.layout}, please check correct interface!"
            )
        # 初始化zeros tensor 接收每个B的输出
        B, T1, T2, NQuery, N2, NQueryIndex, DQuery, DQuerIndex = getTNDCaseInfo(
            self.q,
            self.k,
            self.q_index,
            self.k_index,
            self.topk_index,
            self.q_rope,
            self.k_rope,
            self.actual_seq_q_len,
            self.actual_seq_kv_len,
        )
        # print('DQuery', DQuery)
        # print('DQuerIndex', DQuerIndex)
        d_query_index = torch.zeros(T1, NQueryIndex, DQuerIndex)
        d_key_index = torch.zeros(T2, N2, DQuerIndex)

        d_weight = torch.zeros(T1, NQueryIndex)
        loss = torch.zeros(1)
        # p_max = torch.zeros(N2, T1, NQuery)
        # p_sum = torch.zeros(N2, T1, NQuery)
        p_max = None
        p_sum = None

        for bIdx in range(B):
            # print(f"==================batch index = {bIdx}==================")
            q_start, kv_start = self.get_spoint_by_bIdx(bIdx - 1)
            q_end, kv_end = self.get_spoint_by_bIdx(bIdx)
            if q_start == q_end:
                continue
            # transpose to BSND
            q_i = self.q[q_start:q_end, :, :].unsqueeze(
                0
            )  # (T1, N1, DQuery) -> (1, S1, N1, DQuery)
            k_i = self.k[kv_start:kv_end, :, :].unsqueeze(
                0
            )  # (T1, N2, DQuery) -> (1, S2, N2, DQuery)
            q_index_i = self.q_index[q_start:q_end, :, :].unsqueeze(0)
            k_index_i = self.k_index[kv_start:kv_end, :, :].unsqueeze(0)
            # [N,T] -> [N,S,1]

            # TBD:
            weight = self.weight[q_start:q_end, :].unsqueeze(0)
            # print(f"topk shape in run_unpad is {self.topk_index.shape}")
            # print(self.topk_index[:, 0, 0:32])
            # print(f"q_start = {q_start} q_end =  {q_end}")
            sparse_indices = self.topk_index[q_start:q_end, :, :].unsqueeze(
                0
            )  # [S1, N2, K] -> [1, S1, N2, K]

            (
                d_query_index_i,
                d_key_index_i,
                d_weight_i,
                loss_i,
                p_max_i,
                p_sum_i,
                key_topk,
                bmm2_res,
                bmm1_res,
                reluRes,
                sReduceGRes,
                sSoftmaxRes,
                pReduceGRes,
            ) = self.golden(
                q_i, k_i, q_index_i, k_index_i, self.scale, weight, sparse_indices
            )
            # 填入数据
            # print(f"q_start q_end is {q_start} : {q_end}", d_query_index_i.shape)

            d_query_index[q_start:q_end, :, :] = d_query_index_i
            d_key_index[kv_start:kv_end, :, :] = d_key_index_i
            # print(f"bmm1_res shape is {bmm1_res.shape}")
            # print(f"bmm2_res shape is {bmm2_res.shape}")
            # print(f"d_key_index shape is {d_key_index.shape}")
            # print(f"key_topk shape is {key_topk.shape}")
            # print(f"q_start q_end is {q_start} : {q_end}")
            # d_key_index[q_start:q_end, :, : ] = bmm1_res # TEST bmm1 out
            # d_key_index[q_start:q_end, :, : ] = reluRes # TEST bmm2Res
            # d_key_index[q_start:q_end, :, : ] = reluRes # TEST reluRes
            # d_key_index[q_start:q_end, :, : ] = key_topk # TEST key_topk 2048 * 576
            # d_key_index[q_start:q_end, :, : ] = sSoftmaxRes # TEST sSoftmaxRes

            d_weight[q_start:q_end, :] = d_weight_i

            # 确认维度 1 done
            loss[0] += loss_i

            if p_max is None:
                p_max = p_max_i
                p_sum = p_sum_i
            else:
                p_max = torch.cat((p_max, p_max_i), dim=0)
                p_sum = torch.cat((p_sum, p_sum_i), dim=0)
        p_max.unsqueeze(0)
        p_sum.unsqueeze(0)
        # print('p_max', p_max.shape)
        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum

    def safe_gather(self, key, topk_index, S1):
        S2 = key.size(1)
        mask = (topk_index < 0) | (topk_index >= S2)  # 标记非法索引

        # 先安全限制索引范围
        safe_index = topk_index.clamp(min=0, max=S2 - 1)  # (1 S1 TOPK D)

        # 安全 gather
        key_expanded = key.unsqueeze(1).expand(-1, S1, -1, -1)  # (1 S1 S2 D)
        key_topk = torch.gather(key_expanded, 2, safe_index)

        # 把非法部分置为 0（或其他你想要的默认值）
        key_topk = key_topk.masked_fill(mask, 0.0)
        return key_topk

    def safe_scatter_add(self, beforeScatterAdd, sparse_indices, d_key_index):
        B, S1, S2, D_index = d_key_index.shape
        # 1️⃣ 生成 mask，标记哪些索引是有效的
        mask = sparse_indices != -1  # True 表示合法，False 表示 -1

        # 2️⃣ 将 -1 替换成 0（以防 scatter 越界）
        safe_indices = sparse_indices.clone()
        safe_indices[~mask] = 0

        # 3️⃣ 扩展索引维度以匹配 scatter_add 需求
        index_expanded = safe_indices.unsqueeze(-1).expand(
            -1, -1, -1, D_index
        )  # [B, S1, K, D]

        # 4️⃣ 对应无效位置的 beforeScatterAdd 清零（跳过聚合）
        masked_beforeScatterAdd = beforeScatterAdd * mask.unsqueeze(-1)

        # 5️⃣ scatter_add 聚合
        d_key_index.scatter_add_(2, index_expanded, masked_beforeScatterAdd)
        return d_key_index

    def deal_invalid_token(self, B, S1, S2, topKSize, src_tensor):
        dst_tensor = src_tensor
        for b_idx in range(B):
            for s1_idx in range(S1):
                invalid_len = 0
                s2_realsize = (int)((S2 - S1) + s1_idx + 1)
                if s2_realsize <= 0:
                    s2_realsize = S2
                if s2_realsize > topKSize:
                    block_remain = s2_realsize % self.block_size
                    if self.block_size != 1 and block_remain != 0:
                        s2_realsize = topKSize - (self.block_size - block_remain)
                        dst_tensor[b_idx, s1_idx, :, s2_realsize:] = -float("inf")
                else:
                    invalid_len = topKSize - s2_realsize
                    if invalid_len > 0:
                        dst_tensor[b_idx, s1_idx, :, s2_realsize:] = -float("inf")
        return dst_tensor

    def golden(self, q, k, q_index, k_index, scale, weight, sparse_indices):
        # P 和S'Y的N实际是不同的, 需要额外处理  done
        # sparse 处理 done
        # case info
        B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize = getCaseInfo(
            q, k, q_index, k_index, sparse_indices
        )
        # print("==================in golden==================")
        # print(f"q.shape = {q.shape}")
        # print(f"k.shape = {k.shape}")
        # print(f"q_index.shape = {q_index.shape}")
        # print(f"k_index.shape = {k_index.shape}")
        # print(f"weight.shape = {weight.shape}")
        # print(f"sparse_indices.shape = {sparse_indices.shape}")
        # print(sparse_indices[:, :, :, 0:32])
        # calc P
        # print("==================计算 P==================")
        query = q.to(torch.float)  # (B, S1, N1, D)
        key = k.squeeze(2).to(torch.float)  # N2轴压缩
        sparse_indices = sparse_indices.squeeze(2).to(torch.int64)  # (1 T1 TOPK)
        topk_index = sparse_indices.unsqueeze(-1).expand(-1, -1, -1, D)  # (1 T1 TOPK D)
        # print(f"key shape is {key.shape}")
        # print(f"topk_index shape is {topk_index.shape}")

        key_topk = self.safe_gather(key, topk_index, S1)  # B,S1,K,D
        p = torch.matmul(
            query, key_topk.permute(0, 1, 3, 2).contiguous()
        )  # (B, S1, N1, topK)
        # TODO scale mul 需要补
        p = p * scale
        p = self.deal_invalid_token(B, S1, S2, topKSize, p)
        pSoftmaxRes, p_max, p_sum = self.tsoftmax(p)
        # simplesoftmax
        reduceGShape = [B, S1, N2, G, topKSize]
        hideGShape = [B, S1, N2, topKSize]
        # print(f"reduceGShape shape is {reduceGShape}")
        pSoftmaxRes = pSoftmaxRes.reshape(reduceGShape)
        pReduceGRes = torch.sum(
            pSoftmaxRes, dim=3
        )  # (B, N1, S1, topK) -> (B, N2, G, S1, topK) -> (B, N2, S1, topK)
        pReduceGRes = pReduceGRes / G  #

        # calc S', Y
        # print("==================计算 SY==================")
        query_index = q_index.to(torch.float)  # (B, S1, N1, D)
        key_index = k_index.squeeze(2).to(torch.float)  # -> [B,S2,D]
        topk_index = sparse_indices.unsqueeze(-1).expand(
            -1, -1, -1, D_index
        )  # -> [B,S1,K,D]
        # print(f"topk_index SY shape is {topk_index.shape}")

        # key_index_topk = torch.gather(key_index.unsqueeze(1).expand(-1,S1,-1,-1), 2, topk_index)    # key_index [B,S2,D] -> [B,S1,S2,D]
        key_index_topk = self.safe_gather(key_index, topk_index, S1)
        s = torch.matmul(
            query_index, key_index_topk.permute(0, 1, 3, 2).contiguous()
        )  # (B, S1, N1, topK)

        # relu res 复用
        reluRes = torch.relu(s)  #
        weight = weight.unsqueeze(-1).to(
            torch.float
        )  # (B,S1,N1) -> (B,N1,S1) -> (B,N1,S1,1)
        # print(weight[0, 0, 0, 0])
        # print("reluRes=================:")
        # print(reluRes[0, 0, 0, 0 : 32])
        sReluRes = reluRes * weight  #
        # print("sReluRes=================:")
        # print(sReluRes[0, 0, 0, 0 : 32])
        reduceGShape = [B, S1, Nki, G_index, topKSize]
        hideGShape = [B, S1, Nki, topKSize]
        sReduceGRes = torch.sum(sReluRes.reshape(reduceGShape), dim=3).reshape(
            hideGShape
        )  #
        # S2无效行场景 kSize 选择的行大于S2实际的行 末尾赋值-1 把矩阵末尾的数据修改成-inf
        sReduceGRes = self.deal_invalid_token(B, S1, S2, topKSize, sReduceGRes)
        # torch.set_printoptions(
        #     threshold=float('inf'),  # 元素阈值设为无限大（超过默认1000也不省略）
        #     edgeitems=torch.inf      # 首尾显示元素数量设为无限大（默认只显示3个）
        # )
        softmax_func = nn.Softmax(dim=3)
        sSoftmaxRes = softmax_func(sReduceGRes)  # (B, S1, N2, topK)
        # print(f"sSoftmaxRes shape is {sSoftmaxRes.shape}")
        # KLLoss part
        # print("==================计算 KLLoss==================")
        # pReduceGRes is true, sSoftmaxRes is predict
        min_value = torch.tensor([1e-8])
        pReduceGResClip = torch.max(pReduceGRes, min_value)
        sSoftmaxResClip = torch.max(sSoftmaxRes, min_value)
        log_p = torch.log(pReduceGResClip)
        log_s = torch.log(sSoftmaxResClip)
        sub_res = log_p - log_s
        mul_res = sub_res * pReduceGRes
        loss = torch.sum(mul_res)

        # dW, dQ, dK
        # print("==================计算 dW, dQ, dK==================")
        q_sub_p = sSoftmaxRes - pReduceGRes  # (B, N2, S1, topK) TOCHECK
        # print('p', pReduceGRes.shape, pReduceGRes)
        # print('y', sSoftmaxRes.shape, sSoftmaxRes)
        # print('di', q_sub_p.shape, q_sub_p)
        # bmm3,4 is simplified to matmul
        d_weight = torch.matmul(
            q_sub_p, reluRes.permute(0, 1, 3, 2).contiguous()
        ).squeeze(2)  # (B,S1,1,K) * (B,S1,N,K)

        # fake_relu = torch.ones_like(reluRes) * 100
        # d_weight = torch.matmul(q_sub_p, fake_relu.permute(0,1,3,2).contiguous()).squeeze(2)          # (B,S1,1,K) * (B,S1,N,K)

        # fake_di = torch.ones_like(q_sub_p) * 0.1
        # d_weight = torch.matmul(fake_di, reluRes.permute(0,1,3,2).contiguous()).squeeze(2)          # (B,S1,1,K) * (B,S1,N,K)

        # reluGrad (x > 0) = 1; (x <= 0) = 0
        relu_grad = (
            q_sub_p * weight * (reluRes > 0).to(torch.bfloat16).to(torch.float)
        )  # reshape 1, (B, S1, 1, topK) * (B,S1,N1,1)

        d_query_index = torch.matmul(
            relu_grad, key_index_topk
        )  # (B, S1, N1, topK) * (B, S1, K, D)
        beforeScatterAdd = torch.matmul(
            relu_grad.permute(0, 1, 3, 2).contiguous(), query_index
        )  # [B,S1,N,D]

        # for B:
        #     for S1:
        #         取K index
        #         add->d_key_index
        # index扩展到[...D]
        # 创建目标张量
        d_key_index_before = torch.zeros(
            B,
            S1,
            S2,
            D_index,
            dtype=beforeScatterAdd.dtype,
            device=beforeScatterAdd.device,
        )
        d_key_index = self.safe_scatter_add(
            beforeScatterAdd, sparse_indices, d_key_index_before
        )
        # 如果目标K轴要变成1，则聚合掉K轴
        d_key_index = (
            d_key_index.sum(1, keepdim=True).squeeze(1).unsqueeze(2)
        )  # [B, S1, S2, D] -> [B, 1, S2, D] -> [B, S2, 1, D]
        # print('p_max 0', p_max.shape)
        p_max = p_max.squeeze(0)
        p_sum = p_sum.squeeze(0)
        p_max = p_max.squeeze(-1)
        p_sum = p_sum.squeeze(-1)
        # print('p_max 1', p_max.shape)
        # print(f"d_key_index shape is {d_key_index.shape}")
        # print(f"sSoftmaxRes shape is {sSoftmaxRes.shape}")
        # print(d_key_index[0, :, 0, 0:1])
        # return d_query_index, d_key_index, d_weight, loss, p_max, p_sum
        s = s.squeeze(0)
        p = p.squeeze(0)
        d_query_index = d_query_index.squeeze(0)
        d_key_index = d_key_index.squeeze(0)
        key_index_topk = key_index_topk.squeeze(0)
        sSoftmaxRes = sSoftmaxRes.squeeze(0)
        sReduceGRes = sReduceGRes.squeeze(0)
        return (
            d_query_index,
            d_key_index,
            d_weight,
            loss,
            p_max,
            p_sum,
            key_index_topk,
            s,
            p,
            reluRes,
            sReduceGRes,
            sSoftmaxRes,
            pReduceGRes,
        )

    def run(self):
        if self.layout != "BSND":
            raise ValueError(
                f"[error] case layout is not invalid!, case layout is {self.layout}, please check correct interface!"
            )
        return self.golden(
            self.q,
            self.k,
            self.q_index,
            self.k_index,
            self.scale,
            self.weight,
            self.topk_index,
        )


def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(B)]
    kv_len = [q_len[i] + np.random.randint(1, 8192) for i in range(B)]
    # kv_len = [q_len[i] + np.random.randint(128*1024, 128*1024+1) for i in range(B)]
    # kv_len = [6666]
    return q_len, kv_len


def print_diff(name, tensor_gpu, tensor_npu):
    diff = torch.abs(tensor_npu.float() - tensor_gpu.float())
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f"{name} max diff: %.9f" % max_diff)
    print(f"{name} mean diff: %.9f" % mean_diff)
    max_diff_flat_idx = torch.argmax(diff)
    max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff.shape)
    print(
        f"{name} max diff pos tensor value: %.9f, %.9f"
        % (tensor_npu[max_diff_idx], tensor_gpu[max_diff_idx])
    )
    return max_diff, mean_diff


def check_tensor(x):
    print("Any NaN?", x.isnan().any())
    print("Any Inf?", x.isinf().any())


import functools
from typing import Callable, Any


def convert_half_to_float(func: Callable) -> Callable:
    """
    装饰器：自动将BF16和FP16类型的输入tensor转换为FP32
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 处理位置参数
        converted_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                converted_args.append(arg.to(torch.float32))
            else:
                converted_args.append(arg)

        # 处理关键字参数
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                converted_kwargs[key] = value.to(torch.float32)
            else:
                converted_kwargs[key] = value

        # 调用原始函数
        return func(*converted_args, **converted_kwargs)

    return wrapper


@convert_half_to_float
def torch_sparse_dsa_chunked(
    q,
    kv,
    indices,
    sm_scale=192**-0.5,
    kv_lora_rank=512,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    chunk_size_h=8,
    chunk_size_s=1024,
):
    """
    切块计算版本，减少显存占用

    Args:
        chunk_size_h: head 维度的切块大小
        chunk_size_s: sequence 维度的切块大小
    """

    if cu_seqlens_q is not None:
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        out_list = []
        p_list = []
        sm_sum_list = []
        sm_max_list = []
        for i in range(len(cu_seqlens_q) - 1):
            q_bos, q_eos = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            k_bos, k_eos = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            out, p, sm_max, sm_sum = torch_sparse_dsa_chunked(
                q[q_bos:q_eos],
                kv[k_bos:k_eos],
                indices[q_bos:q_eos],
                sm_scale,
                kv_lora_rank,
                chunk_size_h=chunk_size_h,
                chunk_size_s=chunk_size_s,
            )
            out_list.append(out)
            p_list.append(p)
            sm_max_list.append(sm_max)
            sm_sum_list.append(sm_sum)
        out = torch.cat(out_list, 0)
        p = torch.cat(p_list, 0)
        sm_max = torch.cat(sm_max_list, 0)
        sm_sum = torch.cat(sm_sum_list, 0)
        return out, p, sm_max, sm_sum

    # 准备数据
    topk = indices.size(-1)
    H = q.size(1)  # num heads
    S = q.size(0)  # query length
    L = kv.size(0)  # key length
    D = q.size(2)  # head dim

    indices = indices.unsqueeze(0).repeat_interleave(H // kv.size(1), 0)
    kv = kv.repeat_interleave(H // kv.size(1), 1)

    # 预先计算 causal mask 的索引（只计算一次）
    diff = L - S
    q_idx = torch.arange(S, device=q.device) + diff
    k_idx = torch.arange(L, device=q.device)
    mask_causal = q_idx[:, None] < k_idx[None, :]  # [S, L]

    # 预先计算 index mask（只计算一次）
    # 注意：这里假设所有 head 的 indices 相同，否则需要在循环内计算

    # 初始化输出
    o = torch.zeros(S, H, kv_lora_rank, dtype=q.dtype, device=q.device)
    select_p_sum = torch.zeros(S, topk, dtype=torch.float32, device=q.device)
    sm_max_all = torch.zeros(H, S, dtype=torch.float32, device=q.device)
    sm_sum_all = torch.zeros(H, S, dtype=torch.float32, device=q.device)

    # 按 head 维度切块
    for h_start in range(0, H, chunk_size_h):
        h_end = min(h_start + chunk_size_h, H)
        h_chunk = h_end - h_start

        # 按 query sequence 维度切块
        for s_start in range(0, S, chunk_size_s):
            s_end = min(s_start + chunk_size_s, S)
            s_chunk = s_end - s_start

            # 切块计算 attention scores
            # s_chunk: [h_chunk, s_chunk, L]
            q_chunk = q[s_start:s_end, h_start:h_end, :]  # [s_chunk, h_chunk, D]
            kv_chunk = kv[:, h_start:h_end, :]  # [L, h_chunk, D]

            # 计算 QK^T
            s_chunk_score = torch.einsum(
                "shd,lhd->hsl", q_chunk.float(), kv_chunk.float()
            )  # [h_chunk, s_chunk, L]

            # 应用 scale 和 causal mask
            mask_causal_chunk = mask_causal[s_start:s_end, :]  # [s_chunk, L]
            score_chunk = (s_chunk_score * sm_scale).masked_fill(
                mask_causal_chunk.unsqueeze(0), FILL_VAL
            )  # [h_chunk, s_chunk, L]

            # Padding to topk
            if topk - score_chunk.size(-1) > 0:
                score_chunk = torch.nn.functional.pad(
                    score_chunk, (0, topk - score_chunk.size(-1)), value=FILL_VAL
                )

            # 应用 sparse mask
            indices_chunk = indices[
                h_start:h_end, s_start:s_end, :
            ]  # [h_chunk, s_chunk, topk]
            mask_chunk = index_to_mask(indices_chunk, score_chunk)
            score_chunk = score_chunk + mask_chunk

            # 计算 softmax
            sm_max_chunk = score_chunk.max(-1)[0]  # [h_chunk, s_chunk]
            sm_sum_chunk = (
                (score_chunk - sm_max_chunk.unsqueeze(-1)).exp().sum(-1)
            )  # [h_chunk, s_chunk]
            p_chunk = score_chunk.softmax(-1)  # [h_chunk, s_chunk, topk]

            # 保存 softmax 统计信息
            sm_max_all[h_start:h_end, s_start:s_end] = sm_max_chunk
            sm_sum_all[h_start:h_end, s_start:s_end] = sm_sum_chunk

            # 计算 attention output (PV)
            v_chunk = kv_chunk[..., :kv_lora_rank]  # [L, h_chunk, kv_lora_rank]
            o_chunk = torch.einsum(
                "hsl,lhd->shd", p_chunk[..., : v_chunk.size(0)], v_chunk
            )  # [s_chunk, h_chunk, kv_lora_rank]
            o[s_start:s_end, h_start:h_end, :] = o_chunk

            # 累积 select_p（用于计算 topk indices 的概率）
            indices2_chunk = torch.where(indices_chunk == -1, 0, indices_chunk)
            select_p_chunk = torch.gather(p_chunk, -1, indices2_chunk).sum(
                0
            )  # [s_chunk, topk]
            select_p_sum[s_start:s_end, :] += select_p_chunk

            # 释放中间变量
            del s_chunk_score, score_chunk, p_chunk, o_chunk
            torch.cuda.empty_cache()  # 如果使用 CUDA

    # 平均 select_p
    select_p = select_p_sum / H
    select_p = torch.where(indices[0] == -1, 0.0, select_p)

    return o, select_p, sm_max_all.transpose(0, 1), sm_sum_all.transpose(0, 1)


@convert_half_to_float
def compute_topk(
    index_q,
    index_k,
    weights,
    topk=2048,
    topk_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    """
    input:
    index_q: [T, H2, D2]
    index_k: [T, 1, D2]
    w: [T, H]

    output:
    topk_score: [T]
    """
    if cu_seqlens_q is not None:
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        topk_score_list = []
        topk_indices_list = []
        for i in range(len(cu_seqlens_q) - 1):
            q_bos, q_eos = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            k_bos, k_eos = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            # print(q_bos, q_eos, k_bos, k_eos)
            _topk_score, _topk_indices = compute_topk(
                index_q[q_bos:q_eos],
                index_k[k_bos:k_eos],
                weights[q_bos:q_eos],
                topk,
                topk_indices[q_bos:q_eos] if topk_indices is not None else None,
            )
            topk_score_list.append(_topk_score)
            topk_indices_list.append(_topk_indices)
            # print(topk_indices.shape)
        topk_score = torch.cat(topk_score_list, 0)
        topk_indices = torch.cat(topk_indices_list, 0)
        return topk_score, topk_indices

    # print(index_q.shape, index_k.shape, weights.shape)
    index_k = index_k.repeat_interleave(index_q.size(1) // index_k.size(1), 1)
    s = torch.einsum("shd,lhd->hsl", index_q.float(), index_k.float())
    diff = s.size(2) - s.size(1)
    q_idx = torch.arange(s.size(1), device=s.device) + diff
    k_idx = torch.arange(s.size(2), device=s.device)
    s = torch.nn.functional.relu(s)
    s = s * weights.unsqueeze(-1).transpose(0, 1)
    s = s.sum(0)
    # 使用 masked_fill 替代 torch.where，保持梯度流动
    mask = q_idx[:, None] < k_idx[None, :]
    index_score = s.masked_fill(mask, FILL_VAL)
    if topk - index_score.size(-1) > 0:
        index_score = torch.nn.functional.pad(
            index_score, (0, topk - index_score.size(-1)), value=FILL_VAL
        )
    if topk_indices is None:
        topk_indices = torch.arange(topk, device=index_score.device)
        topk_score, topk_indices = index_score.topk(
            min(topk, index_score.size(-1)), dim=-1
        )
        # 这里的 where 不会影响 topk_score 的梯度，因为它只修改 indices
        topk_indices = torch.where(topk_score == FILL_VAL, -1, topk_indices)
    else:
        topk_indices = topk_indices
        topk_score = index_score.gather(-1, topk_indices)
    return topk_score, topk_indices


@torch.no_grad()
def index_to_mask(indices, score):
    mask = torch.full_like(score, FILL_VAL)
    indices2 = torch.where(
        indices == -1, indices.to(torch.int64).max(-1)[0].unsqueeze(-1), indices
    )
    mask = mask.scatter(-1, indices2, 0)
    return mask


@convert_half_to_float
def torch_sparse_dsa(
    q,
    kv,
    indices,
    sm_scale=192**-0.5,
    kv_lora_rank=512,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
):
    if cu_seqlens_q is not None:
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        out_list = []
        p_list = []
        sm_sum_list = []
        sm_max_list = []
        for i in range(len(cu_seqlens_q) - 1):
            q_bos, q_eos = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            k_bos, k_eos = cu_seqlens_k[i], cu_seqlens_k[i + 1]
            out, p, sm_max, sm_sum = torch_sparse_dsa(
                q[q_bos:q_eos],
                kv[k_bos:k_eos],
                indices[q_bos:q_eos],
                sm_scale,
                kv_lora_rank,
            )
            out_list.append(out)
            p_list.append(p)
            sm_max_list.append(sm_max)
            sm_sum_list.append(sm_sum)
        out = torch.cat(out_list, 0)
        p = torch.cat(p_list, 0)
        sm_max = torch.cat(sm_max_list, 0)
        sm_sum = torch.cat(sm_sum_list, 0)
        return out, p, sm_max, sm_sum

    topk = indices.size(-1)
    indices = indices.unsqueeze(0).repeat_interleave(q.size(1) // kv.size(1), 0)
    kv = kv.repeat_interleave(q.size(1) // kv.size(1), 1)
    s = torch.einsum("shd,lhd->hsl", q.float(), kv.float())
    diff = s.size(2) - s.size(1)
    q_idx = torch.arange(s.size(1), device=s.device) + diff
    k_idx = torch.arange(s.size(2), device=s.device)
    # 使用 masked_fill 替代 torch.where，保持梯度流动
    mask_causal = q_idx[:, None] < k_idx[None, :]
    score = (s * sm_scale).masked_fill(mask_causal, FILL_VAL)
    if topk - score.size(-1) > 0:
        score = torch.nn.functional.pad(
            score, (0, topk - score.size(-1)), value=FILL_VAL
        )
    mask = index_to_mask(indices, score)
    score = score + mask
    sm_max = score.max(-1)[0]
    sm_sum = (score - sm_max.unsqueeze(-1)).exp().sum(-1)
    p = score.softmax(-1)
    indices2 = torch.where(indices == -1, 0, indices)
    select_p = torch.gather(p, -1, indices2).sum(0) / p.size(0)
    select_p = torch.where(indices[0] == -1, 0.0, select_p)

    v = kv[..., :kv_lora_rank]
    o = torch.einsum("hsl,lhd->shd", p[..., : v.size(0)], v)
    return o, select_p, sm_max.transpose(0, 1), sm_sum.transpose(0, 1)


class KLLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, index_score):
        index_p = index_score.softmax(-1, dtype=torch.float32)
        clip_p = torch.clamp(p, 1e-8, 1.0)
        clip_index_p = torch.clamp(index_p, 1e-8, 1.0)
        loss = p * (torch.log(clip_p) - torch.log(clip_index_p))
        loss = loss.sum(-1).view(-1)
        dindex_score = index_p - p
        ctx.save_for_backward(dindex_score)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (dindex_score,) = ctx.saved_tensors
        # print(dindex_score.shape, grad_output.shape)
        dindex_score = dindex_score * grad_output[:, None]
        return None, dindex_score


def kl_loss_sparse(select_p, topk_score):
    return KLLossFunction.apply(select_p, topk_score)


def _call_migrated_grad(*args):
    if os.environ.get("LIG_USE_SKIP_PADDING") == "1":
        mask = torch.ones((args[0].shape[0],), dtype=torch.int8, device=args[0].device)
        print("Testing skip_padding with an all-valid mask", flush=True)
        return (
            torch.ops.cann_ops_transformer.lightning_indexer_grad_kl_loss_skip_padding(
                *args[:10], mask, *args[10:], validTokenNum=args[0].shape[0]
            )
        )
    return torch.ops.cann_ops_transformer.lightning_indexer_grad_kl_loss(*args)


def test(B, T1, blockCount, blockSize, T2):
    failures_before = len(TEST_FAILURES)
    device = torch.device(f"npu:{DEVICE_ID}")
    torch.npu.set_device(device)
    dtype = torch.bfloat16
    isDetermin = True

    headNum = 64
    headNumK = 1
    headNumIdx = 32
    headNumKKIdx = 1
    headDimP = 512
    headDimSY = 128
    scale = headNumIdx**-0.5 * (headDimSY**-0.5)
    sm_scale = 192**-0.5
    layout = "TND"

    # actual_seq_lengths_query, actual_seq_lengths_key = gen_seq_len(B, T1)
    actual_seq_lengths_query = [T1]
    actual_seq_lengths_key = [T2]
    T1 = np.sum(actual_seq_lengths_query)  # 重新更新T1
    T2 = np.sum(actual_seq_lengths_key)  # 重新更新T2
    cu_actual_seq_lengths_query = list(np.cumsum(actual_seq_lengths_query))
    cu_actual_seq_lengths_key = list(np.cumsum(actual_seq_lengths_key))
    print(
        "B, T1, T2, cu_actual_seq_lengths_query, blockCount: ",
        B,
        T1,
        T2,
        cu_actual_seq_lengths_query,
        blockCount,
    )

    # inputs
    q_absorb = (
        torch.rand(T1, headNum, headDimP, device=device, dtype=dtype).requires_grad_(
            True
        )
        * 20
        - 10
    )
    q_pe = (
        torch.rand(T1, headNum, 64, device=device, dtype=dtype).requires_grad_(True)
        * 20
        - 10
    )
    ct_kv = (
        torch.rand(T2, headNumK, headDimP, device=device, dtype=dtype).requires_grad_(
            True
        )
        * 20
        - 10
    )
    k_pe = (
        torch.rand(T2, headNumK, 64, device=device, dtype=dtype).requires_grad_(True)
        * 20
        - 10
    )
    q = torch.cat([q_absorb, q_pe], dim=-1)
    kv = torch.cat([ct_kv, k_pe], dim=-1)

    query_index = (
        torch.rand(T1, headNumIdx, headDimSY, device=device, dtype=dtype) * 20 - 10
    )
    key_index = (
        torch.rand(T2, headNumKKIdx, headDimSY, device=device, dtype=dtype) * 20 - 10
    )
    # weights = torch.rand(T1, headNumIdx, dtype=torch.float) * 2 - 1 # 用float类型，cast 2 bf16，精度影响有多大？
    weights = torch.rand(T1, headNumIdx, device=device, dtype=torch.float) * 5 * scale

    # 在使用前设置requires_grad
    query_index.requires_grad_(True)
    key_index.requires_grad_(True)
    weights.requires_grad_(True)
    do = torch.randn(T1, headNum, headDimP, device=device, dtype=dtype)

    cu_actual_seq_lengths_query0 = [0] + list(np.cumsum(actual_seq_lengths_query))
    cu_actual_seq_lengths_key0 = [0] + list(np.cumsum(actual_seq_lengths_key))
    cu_q0 = (
        torch.tensor(cu_actual_seq_lengths_query0, dtype=torch.int64)
        .to(device)
        .contiguous()
    )
    cu_k0 = (
        torch.tensor(cu_actual_seq_lengths_key0, dtype=torch.int64)
        .to(device)
        .contiguous()
    )
    print("cu_q0 dtype is ", cu_q0.dtype)
    print("cu_k0 dtype is ", cu_k0.dtype)
    topk_index, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query_index,
        key_index,
        weights,
        cur_seq_lengths_query=cu_q0,
        cur_seq_lengths_key=cu_k0,
        block_table=None,
        layout_query="TND",
        layout_key="TND",
        sparse_count=blockCount,
        kv_block_len=1,
        q_block_len=1,
        init_num=16,
        local_num=0,
        sparse_mode=3,
        pre_tokens=512 * 1024,
        next_tokens=512 * 1024,
        return_value=False,
    )
    print("topk_index", topk_index)

    # 注意：topk_indices 是从 NPU 算子返回的，需要 detach 避免影响梯度计算
    ref_topk_score, ref_topk_indices = compute_topk(
        query_index,
        key_index,
        weights,
        topk=2048,
        topk_indices=topk_index.squeeze(1).detach(),
        cu_seqlens_q=cu_q0,
        cu_seqlens_k=cu_k0,
    )
    print("ref_topk_indices ", ref_topk_indices)
    # The teacher attention does not depend on indexer parameters once TopK
    # indices are detached. Its backward graph cannot contribute to dq_index,
    # dk_index or d_weight, and retaining it exhausts HBM for long sequences.
    with torch.no_grad():
        ref_o, ref_select_p, ref_softmax_max, ref_softmax_sum = torch_sparse_dsa(
            q, kv, ref_topk_indices.detach(), sm_scale, headDimP, cu_q0, cu_k0
        )
    # ref_o, ref_select_p, ref_softmax_max, ref_softmax_sum = torch_sparse_dsa_chunked(
    #     q, kv, ref_topk_indices.detach(), scale, headDimP, cu_q0, cu_k0,
    #     chunk_size_h=1,     # 每次处理 8 个 head
    #     chunk_size_s=16   # 每次处理 1024 个 query tokens
    # )
    ref_kl_loss = kl_loss_sparse(ref_select_p, ref_topk_score).sum()

    # 执行反向传播
    ref_kl_loss.backward()

    print(f"[DEBUG] query_index.grad is None: {query_index.grad is None}")
    print(f"[DEBUG] key_index.grad is None: {key_index.grad is None}")
    print(f"[DEBUG] weights.grad is None: {weights.grad is None}")
    # 调用SFA，使用它返回的softmax max sum，而不是用golden产生的
    stats_source = os.environ.get("LIG_STATS_SOURCE", "sfa")
    print("softmax statistics source:", stats_source, flush=True)
    if stats_source == "reference":
        sfa_max, sfa_sum = (
            ref_softmax_max.detach().contiguous(),
            ref_softmax_sum.detach().contiguous(),
        )
    elif stats_source == "sfa":
        _, sfa_max, sfa_sum = torch_npu.npu_sparse_flash_attention(
            q_absorb,
            ct_kv,
            ct_kv,
            topk_index,
            sm_scale,
            block_table=None,
            actual_seq_lengths_query=cu_q0.to(torch.int32),
            actual_seq_lengths_kv=cu_k0.to(torch.int32),
            query_rope=q_pe,
            key_rope=k_pe,
            sparse_block_size=1,
            layout_query=layout,
            layout_kv=layout,
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=True,
        )
    else:
        raise ValueError("LIG_STATS_SOURCE must be sfa or reference")
    print("cu_actual_seq_lengths_query0 is ", cu_actual_seq_lengths_query0)
    print("cu_actual_seq_lengths_key0 is ", cu_actual_seq_lengths_key0)
    real_dq, real_dk, real_dw, real_loss = _call_migrated_grad(
        q_absorb,
        ct_kv,
        query_index,
        key_index,
        weights,
        topk_index,
        sfa_max,
        sfa_sum,
        q_pe,
        k_pe,
        cu_actual_seq_lengths_query0,
        cu_actual_seq_lengths_key0,
        sm_scale,
        layout,
        3,  # sparse_mode
        512 * 1024,
        512 * 1024,
        blockSize,
        isDetermin,
    )
    torch.npu.synchronize()

    ## check results
    print_diff("dq", query_index.grad, real_dq)
    check_result(query_index.grad, real_dq, "dq")

    print_diff("dk", key_index.grad, real_dk)
    check_result(key_index.grad, real_dk, "dk")

    print_diff("dw", weights.grad, real_dw)
    check_result(weights.grad, real_dw, "dw")

    # print_diff('loss', ref_kl_loss.cpu(), real_loss.cpu())
    print("loss value", ref_kl_loss, real_loss)
    print("loss diff ", (ref_kl_loss - real_loss).cpu())
    if not torch.isfinite(real_loss).all() or not torch.allclose(
        real_loss.detach(),
        ref_kl_loss.detach().reshape_as(real_loss),
        rtol=5e-3,
        atol=5e-3,
    ):
        TEST_FAILURES.append("loss exceeds rtol=atol=5e-3 or contains NaN/Inf")
    else:
        print("[INFO] LIG loss Check Acuraccy Pass", flush=True)
    outcome = "PASS" if len(TEST_FAILURES) == failures_before else "FAIL"
    print(f"CASE {outcome}: T1={T1}, T2={T2}, topk={blockCount}", flush=True)

    # # perf time
    # import time
    # for _ in range(10):
    #     real_dq, real_dk, real_dw, real_loss = _call_migrated_grad(
    #         npu_q,
    #         npu_k,
    #         npu_query_index,
    #         npu_key_index,
    #         npu_weights,
    #         npu_topk_index,
    #         sfa_max,
    #         sfa_sum,
    #         npu_q_rope,
    #         npu_k_rope,
    #         cu_actual_seq_lengths_query,
    #         cu_actual_seq_lengths_key,
    #         scale,
    #         layout,
    #         3, # sparse_mode
    #         512*1024,
    #         512*1024,
    #         blockSize,
    #         isDetermin
    #     )
    # torch.npu.synchronize()

    # start = time.time()
    # for _ in range(50):
    #     real_dq, real_dk, real_dw, real_loss = _call_migrated_grad(
    #         npu_q,
    #         npu_k,
    #         npu_query_index,
    #         npu_key_index,
    #         npu_weights,
    #         npu_topk_index,
    #         sfa_max,
    #         sfa_sum,
    #         npu_q_rope,
    #         npu_k_rope,
    #         cu_actual_seq_lengths_query,
    #         cu_actual_seq_lengths_key,
    #         scale,
    #         layout,
    #         3, # sparse_mode
    #         512*1024,
    #         512*1024,
    #         blockSize,
    #         isDetermin
    #     )
    # torch.npu.synchronize()
    # end = time.time()
    # print('avg time(ms): ', (end - start) / 50 * 1000)

    # determin test
    # md5_hash_dq = tensor_md5(real_dq)
    # print(f"[INFO]: dq计算的 MD5 校验和: {md5_hash_dq}")
    # md5_hash_dk = tensor_md5(real_dk)
    # print(f"[INFO]: dk计算的 MD5 校验和: {md5_hash_dk}")
    # md5_hash_dw = tensor_md5(real_dw)
    # print(f"[INFO]: dw计算的 MD5 校验和: {md5_hash_dw}")
    # md5_hash_loss = tensor_md5(real_loss)
    # print(f"[INFO]: loss计算的 MD5 校验和: {md5_hash_loss}")
    # torch.npu.synchronize()
    # for _ in range(100):
    #     real_dq, real_dk, real_dw, real_loss = _call_migrated_grad(
    #         npu_q,
    #         npu_k,
    #         npu_query_index,
    #         npu_key_index,
    #         npu_weights,
    #         npu_topk_index,
    #         sfa_max,
    #         sfa_sum,
    #         npu_q_rope,
    #         npu_k_rope,
    #         cu_actual_seq_lengths_query0,
    #         cu_actual_seq_lengths_key0,
    #         scale,
    #         layout,
    #         3, # sparse_mode
    #         512*1024,
    #         512*1024,
    #         blockSize,
    #         isDetermin
    #     )
    #     md5_hash_dq2 = tensor_md5(real_dq)
    #     if md5_hash_dq2 != md5_hash_dq:
    #         print("dq fail")
    #     md5_hash_dk2 = tensor_md5(real_dk)
    #     if md5_hash_dk2 != md5_hash_dk:
    #         print("dk fail")
    #     md5_hash_dw2 = tensor_md5(real_dw)
    #     if md5_hash_dw2 != md5_hash_dw:
    #         print("dw fail")
    #     md5_hash_loss2 = tensor_md5(real_loss)
    #     if md5_hash_loss2 != md5_hash_loss:
    #         print("loss fail")


def tensor_md5(tensor):
    """计算 PyTorch 张量的 MD5 值（支持 BFloat16）"""
    # 处理 BFloat16 类型
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)

    # 确保张量在 CPU 上且内存连续
    tensor = tensor.detach().cpu().contiguous()

    # 转换为字节数据
    byte_data = tensor.numpy().tobytes()

    return hashlib.md5(byte_data).hexdigest()


if __name__ == "__main__":
    t1 = int(os.environ.get("LIG_T1", "16"))
    topk = int(os.environ.get("LIG_TOPK", "2048"))
    # Bound this environment's golden validation, not the operator's API.
    requested_t2_values = [
        int(x) for x in os.environ.get("LIG_T2_LIST", "4096,8192").split(",")
    ]
    if any(value <= 0 for value in requested_t2_values):
        raise ValueError("LIG_T2_LIST must contain positive sequence lengths")
    t2_values = [value for value in requested_t2_values if value <= 8192]
    skipped_t2_values = [value for value in requested_t2_values if value > 8192]
    if skipped_t2_values:
        print(
            f"Skipping seq_k > 8192 outside the current golden validation range: {skipped_t2_values}",
            flush=True,
        )
    if not t2_values:
        raise ValueError(
            "No test case remains: current golden validation requires seq_k <= 8192"
        )
    for T2 in t2_values:
        test(1, t1, topk, 1, T2)
    if TEST_FAILURES:
        raise AssertionError("; ".join(TEST_FAILURES))
    print(f"OVERALL PASS: {len(t2_values)} cases", flush=True)
    # j=0
    # blocksize=1
    # for B in range(1, 3):
    #     for T1 in range(B*7, 10240, 4444):
    #         for topk in range(1024, 4096, 128):
    #             print('case:', j)
    #             j+=1
    #             test(B, T1, topk, 1)
    #             print('---------------------------')

    # #blocksize=4
    # for B in range(5, 6):
    #     for T1 in range(B*7, 10240, 5555):
    #         for topk in range(2560, 3072, 128):
    #             print('case:', j)
    #             j+=1
    #             test(B, T1, topk, 4)
    #             print('---------------------------')
    # j=0
    # for i in range(1):
    #     print(i)
    #     # test(1, 8192, 2560, 1)
    #     for B in range(1, 5):
    #         for T1 in [2048]:
    #             for T2 in range(8192, 40000, 1111):
    #                 for topk in [2048]:
    #                     print('case:', j)
    #                     j += 1
    #                     test(B, T1, topk, 1)
    #                     print('---------------------------')
