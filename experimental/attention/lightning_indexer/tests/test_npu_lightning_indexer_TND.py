# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import importlib
import os
import json
from pathlib import Path

importlib.import_module(
    os.environ.get("CANN_OPS_TRANSFORMER_PACKAGE", "cann_ops_transformer")
)
# import torchair
# import custom_ops
import numpy as np
import torch.nn as nn
import time
import math

# torch.set_printoptions(profile="default", threshold=float("inf"), precision=9, sci_mode=False)
torch.set_printoptions(profile="default", precision=9, sci_mode=False)
# Q_BLOCK_LEN=2
# INIT_NUM=0
# LOCAL_NUM=0
DEFAULT_SCORE = -8888888
MAGIC_NUMBER = 8866
DIGONAL_VALUE = 9999999
WINDOW_VALUE = 1111111
RANDOM_BLOCK_TABLE = 0


def split_to_kv_cache(input_tensor, block_size, cur_kv_seqlen, max_seq_len, end_batch):
    device = input_tensor.device
    # T = input_tensor.shape[0]
    B = len(cur_kv_seqlen)
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        total_seq_block_cnt += seq_block_cnt
    block_table_list = (
        torch.randperm(total_seq_block_cnt)
        if RANDOM_BLOCK_TABLE
        else torch.arange(0, total_seq_block_cnt, 1)
    )
    B2 = B + end_batch
    block_tables = torch.zeros(B2, math.ceil(max_seq_len / block_size)).to(torch.int32)
    input_shape = input_tensor.shape
    cache_tensor = (
        torch.zeros(
            B2 * math.ceil(max_seq_len / block_size),
            block_size,
            input_shape[-2],
            input_shape[-1],
        )
        .to(input_tensor.dtype)
        .to(device)
    )

    start = 0
    spilt_tensors_list = []
    total_seq_block_cnt = 0
    for i in range(B):
        seq_block_cnt = math.ceil(cur_kv_seqlen[i] / block_size)
        if seq_block_cnt == 0:
            seq_block_cnt = 1
        block_tables[i, :seq_block_cnt] = block_table_list[
            total_seq_block_cnt : seq_block_cnt + total_seq_block_cnt
        ]
        kv_len = cur_kv_seqlen[i]
        if kv_len == 0:
            kv_len = 1
        for block_idx in range(seq_block_cnt):
            block_offset = block_idx * block_size
            seq_offset = block_offset + start
            if block_idx == seq_block_cnt - 1:
                remain_size = kv_len - block_idx * block_size
                cache_tensor[block_tables[i][block_idx], :remain_size] = input_tensor[
                    seq_offset : seq_offset + remain_size
                ]
            else:
                cache_tensor[block_tables[i][block_idx], :block_size] = input_tensor[
                    seq_offset : seq_offset + block_size
                ]

        total_seq_block_cnt += seq_block_cnt
        tensor = input_tensor[start : start + kv_len]
        split_tensor = torch.split(tensor, block_size, dim=0)
        remaining_size = kv_len % block_size
        if remaining_size != 0:
            last_split = torch.narrow(
                tensor, 0, tensor.shape[0] - remaining_size, remaining_size
            )
            split_tensor = split_tensor[:-1] + (last_split,)
        start += kv_len
        spilt_tensors_list += split_tensor
    return cache_tensor, block_tables.to(device).detach()


def compare_token_indices(actual, reference, seq_k):
    """Compare every token's index array, with separate order/set diagnostics."""
    actual = actual.detach().cpu()
    reference = reference.detach().cpu()
    assert actual.shape == reference.shape and actual.dim() == 2, (
        f"Index shape mismatch: {actual.shape} vs {reference.shape}"
    )
    different = actual != reference
    ordered_bad = different.any(dim=1)
    # Sorting preserves repeated indices and padding multiplicity.
    sorted_bad = (actual.sort(dim=1).values != reference.sort(dim=1).values).any(dim=1)
    report = {
        "seq_k": seq_k,
        "tokens": actual.shape[0],
        "topk": actual.shape[1],
        "ordered_mismatch_tokens": int(ordered_bad.sum()),
        "index_multiset_mismatch_tokens": int(sorted_bad.sum()),
        "order_only_mismatch_tokens": int((ordered_bad & ~sorted_bad).sum()),
        "mismatch_positions": int(different.sum()),
        "total_positions": actual.numel(),
        "mismatches": [],
    }
    for token in ordered_bad.nonzero(as_tuple=False).flatten().tolist():
        a, b = actual[token], reference[token]
        positions = different[token].nonzero(as_tuple=False).flatten().tolist()
        report["mismatches"].append(
            {
                "token": token,
                "same_index_multiset": not bool(sorted_bad[token]),
                "mismatch_positions": positions,
                "actual_indices": a.tolist(),
                "reference_indices": b.tolist(),
                "actual_only": a[~torch.isin(a, b)].tolist(),
                "reference_only": b[~torch.isin(b, a)].tolist(),
            }
        )
    print(
        f"TOKEN INDEX CHECK seq_k={seq_k}: "
        f"ordered_mismatch_tokens={report['ordered_mismatch_tokens']}/{report['tokens']}, "
        f"index_multiset_mismatch_tokens={report['index_multiset_mismatch_tokens']}/{report['tokens']}, "
        f"order_only_mismatch_tokens={report['order_only_mismatch_tokens']}, "
        f"mismatch_positions={report['mismatch_positions']}/{report['total_positions']}",
        flush=True,
    )
    for item in report["mismatches"][:8]:
        print(
            f"  token={item['token']} same_index_multiset={item['same_index_multiset']} "
            f"first_positions={item['mismatch_positions'][:8]} "
            f"actual_only={item['actual_only'][:8]} reference_only={item['reference_only'][:8]}",
            flush=True,
        )
    report_dir = os.environ.get("LI_REPORT_DIR")
    if report_dir:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        (report_path / f"indices_seq_k_{seq_k}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        # Save indices only; returned values are deliberately not inspected.
        torch.save(
            {"actual": actual, "reference": reference},
            report_path / f"indices_seq_k_{seq_k}.pt",
        )
    return report


DEVICE_ID = int(os.environ.get("LI_NPU_DEVICE", "11"))
torch_npu.npu.set_device(int(DEVICE_ID))


def _lightning_indexer(
    query,
    key,
    weights,
    cur_seq_lengths_query,
    cur_seq_lengths_key,
    layout_query="TND",
    sparse_count=2048,
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    return_value=False,
    BLOCK_LEN=1,
    INIT_NUM=0,
    LOCAL_NUM=0,
    Q_BLOCK_LEN=1,
):
    batch_size = cur_seq_lengths_query.shape[0] - 1
    out_shape = list(query.shape)
    n2 = 1
    N = query.shape[-2]
    D = query.shape[-1]
    out_shape[-1] = sparse_count
    out_shape[-2] = n2
    # indice初始化为全-1
    out = (
        torch.zeros(out_shape, dtype=torch.int32, device=query.device).reshape(
            -1, n2, sparse_count
        )
        - 1
    )
    # value初始化为全0
    valuesOut = torch.zeros(
        out_shape, dtype=torch.float32, device=query.device
    ).reshape(-1, n2, sparse_count)
    act_s1 = 0
    act_s2 = 0
    process_q_len = 0
    process_kv_len = 0
    for batch_id in range(batch_size):
        act_s1 = cur_seq_lengths_query[batch_id + 1] - cur_seq_lengths_query[batch_id]
        act_s2 = cur_seq_lengths_key[batch_id + 1] - cur_seq_lengths_key[batch_id]
        if act_s1 < 0:
            break
        if act_s1 == 0:
            if act_s2 == 0:
                act_s1 = 1
                act_s2 = 1
                process_q_len += act_s1
                process_kv_len += act_s2
                continue
            else:
                process_q_len += 1
                process_kv_len += act_s2
                continue
        now_q = (
            query.reshape(-1, N, D)[process_q_len : process_q_len + act_s1, :, :]
            .transpose(0, 1)
            .to(torch.float32)
        )
        now_k = (
            key.reshape(-1, D)[process_kv_len : process_kv_len + act_s2, :]
            .transpose(0, 1)
            .to(torch.float32)
        )
        now_weights = (
            weights.reshape(-1, N, 1)[process_q_len : process_q_len + act_s1, :, :]
            .transpose(0, 1)
            .to(torch.float32)
        )
        process_q_len += act_s1
        process_kv_len += act_s2
        # N,s1,D @ D,s2 -> N,s1,s2
        relu_out = torch.maximum(torch.matmul(now_q, now_k), torch.tensor(0))
        weight_out = relu_out * now_weights
        # N,s1,s2 -> s1,s2
        reduce_out = torch.sum(weight_out, dim=0)
        # sparse场景下三角置为-inf
        tmp_s1 = reduce_out.shape[0]
        tmp_s2 = reduce_out.shape[1]
        # print('tmp_s1, tmp_s2', tmp_s1, tmp_s2)

        atten_mask_u = torch.triu(
            torch.ones([act_s1, act_s2], dtype=torch.uint8),
            diagonal=(act_s2 - act_s1) + 1,
        )
        reduce_out = reduce_out.masked_fill(atten_mask_u.to(torch.bool), DEFAULT_SCORE)
        # reduce_out.diagonal(offset=act_s2-act_s1)[:] = DIGONAL_VALUE  # 直接赋值给对角线的视图

        act_s2_align = (act_s2 + BLOCK_LEN - 1) // BLOCK_LEN * BLOCK_LEN
        reduce_out_tmp = torch.zeros([act_s1, act_s2_align], dtype=torch.float)
        reduce_out_tmp[:, :act_s2] = reduce_out
        reduce_out_tmp[:, act_s2:] = DEFAULT_SCORE
        # print('reduce_out_tmp', reduce_out_tmp)

        reduce_out_tmp = reduce_out_tmp.view(
            act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN
        )
        reduce_out_tmp = torch.sum(reduce_out_tmp, dim=-1, keepdim=False) / BLOCK_LEN
        reduce_out_tmp = torch.clip(
            reduce_out_tmp, max=WINDOW_VALUE / 2
        )  # logits做max clip，防止冲撞WINDOW_VALUE和DIGONAL_VALUE
        block_init_num = INIT_NUM // BLOCK_LEN
        block_local_num = LOCAL_NUM // BLOCK_LEN
        reduce_out_tmp[:, :block_init_num] = WINDOW_VALUE
        for s1 in range(act_s1):
            valid_s2_len = act_s2 - act_s1 + s1 + 1
            valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
            start_idx = (
                (valid_s2_block - block_local_num)
                if (valid_s2_block - block_local_num) > 0
                else 0
            )
            reduce_out_tmp[s1, start_idx:valid_s2_block] = (
                WINDOW_VALUE  # 指定窗口区域赋予第二大分数
            )
            reduce_out_tmp[s1, valid_s2_block - 1] = (
                DIGONAL_VALUE  # 对角线上元素赋予最大分数
            )
            reduce_out_tmp[s1, valid_s2_block:] = (
                DEFAULT_SCORE  # 对角线右侧直接赋予最小分数
            )
        # print('reduce_out_tmp')
        # print(reduce_out_tmp)

        atten_mask_s2_scale_u = torch.triu(
            torch.ones([act_s1, act_s2_align], dtype=torch.uint8),
            diagonal=(act_s2 - act_s1) + 1,
        )
        atten_mask_s2_scale_u = atten_mask_s2_scale_u.view(
            act_s1, act_s2_align // BLOCK_LEN, BLOCK_LEN
        )
        atten_mask_s2_scale_u = torch.sum(atten_mask_s2_scale_u, dim=-1, keepdim=False)
        atten_mask_s2_scale_u = (atten_mask_s2_scale_u == BLOCK_LEN).to(torch.uint8)
        # print('atten_mask_s2_scale_u')
        # print(atten_mask_s2_scale_u)
        # exit(0)

        reduce_out = reduce_out_tmp.contiguous()
        # print('reduce_out scale', reduce_out)

        sorted_value, sorted_indices = torch.sort(
            reduce_out, dim=1, descending=True, stable=True
        )
        sorted_indices = sorted_indices.masked_fill(
            atten_mask_s2_scale_u.to(torch.bool), -1
        )
        # print('sorted_indices', sorted_indices.shape)
        # print(sorted_indices)
        if Q_BLOCK_LEN > 1:
            for s1 in range(act_s1):
                if s1 % Q_BLOCK_LEN == 0:
                    # 对角线block当前在队首位置，我们把它放回到对角线上
                    valid_s2_len = act_s2 - act_s1 + s1 + 1
                    valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                    valid_s2_block = (
                        valid_s2_block
                        if valid_s2_block < sparse_count
                        else sparse_count
                    )
                    tmp = sorted_indices[s1, 0].clone()
                    sorted_indices[s1, 0] = sorted_indices[s1, valid_s2_block - 1]
                    sorted_indices[s1, valid_s2_block - 1] = tmp
                    tmp = sorted_value[s1, 0].clone()
                    sorted_value[s1, 0] = sorted_value[s1, valid_s2_block - 1]
                    sorted_value[s1, valid_s2_block - 1] = tmp

                    for q in range(1, Q_BLOCK_LEN):
                        if s1 + q >= act_s1:
                            break
                        valid_s2_len = act_s2 - act_s1 + s1 + 1 + q
                        valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                        if valid_s2_block < sparse_count:
                            sorted_indices[s1 + q, :] = sorted_indices[
                                s1 + q - 1, :
                            ].clone()
                            sorted_indices[s1 + q, valid_s2_block - 1] = (
                                valid_s2_block - 1
                            )
                        else:
                            sorted_indices[s1 + q, :] = sorted_indices[
                                s1 + q - 1, :
                            ].clone()
                            for i in range(q + 1):
                                if i == q:
                                    tmp = sorted_indices[
                                        s1 + q, sparse_count - 1 - i
                                    ].clone()
                                sorted_indices[s1 + q, sparse_count - 1 - i] = (
                                    valid_s2_block - 1 - i
                                )
                                if i == q:
                                    sorted_indices[s1 + q, 0] = tmp
                else:
                    continue
        else:
            for s1 in range(act_s1):
                valid_s2_len = act_s2 - act_s1 + s1 + 1
                valid_s2_block = (valid_s2_len + BLOCK_LEN - 1) // BLOCK_LEN
                valid_s2_block = (
                    valid_s2_block if valid_s2_block < sparse_count else sparse_count
                )
                tmp = sorted_indices[s1, 0].clone()
                sorted_indices[s1, 0] = sorted_indices[s1, valid_s2_block - 1]
                sorted_indices[s1, valid_s2_block - 1] = tmp
                tmp = sorted_value[s1, 0].clone()
                sorted_value[s1, 0] = sorted_value[s1, valid_s2_block - 1]
                sorted_value[s1, valid_s2_block - 1] = tmp

        # print('sorted_value', sorted_value)
        # print('sorted_indices', sorted_indices)
        return_s2 = min(sparse_count, act_s2_align // BLOCK_LEN)
        out[process_q_len - act_s1 : process_q_len, 0, :return_s2] = sorted_indices.to(
            torch.int32
        )[:, :return_s2]
        if return_value:
            valuesOut[process_q_len - act_s1 : process_q_len, 0, :return_s2] = (
                sorted_value[:, :return_s2]
            )
        # print('out', out)
        # print('valuesOut', valuesOut)

    out = out.reshape(out_shape)
    valuesOut = valuesOut.reshape(out_shape)
    return out, valuesOut


def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(B)]
    kv_len = [q_len[i] + np.random.randint(1000, 8000) for i in range(B)]
    # kv_len = [q_len[i] + 2622 for i in range(B)]
    # kv_len = [q_len[i]*4 for i in range(B)]
    return q_len, kv_len


def calculate_special_sum(lst):
    """
    计算特殊列表的和：
    - 0元素长度为1（即每个0贡献1到总和）
    - 非零元素为它本身的值

    参数:
        lst: 输入列表，如 [0, 0, 1, 10]

    返回:
        计算后的总和，如 [0, 0, 1, 10] 返回 13 (0->1 + 0->1 + 1->1 + 10->10)

    示例:
        >>> calculate_special_sum([0, 0, 1, 10])
        13
        >>> calculate_special_sum([1, 2, 3])
        6
        >>> calculate_special_sum([0, 0, 0])
        3
    """
    if not lst:  # 处理空列表
        return 0

    total = 0
    for element in lst:
        if element == 0:
            total += 1  # 0元素长度为1
        else:
            total += element  # 非零元素为它本身的值

    return total


def test_tnd_lightning_indexer_eager(
    B, T1, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT, T2
):
    assert INIT_NUM % BLOCK_LEN == 0, "INIT_NUM invalid"
    assert LOCAL_NUM % BLOCK_LEN == 0, "LOCAL_NUM invalid"
    # actual_q_len, actual_kv_len = gen_seq_len(B, T1)
    actual_q_len = [T1]
    actual_kv_len = [T2]
    T1 = calculate_special_sum(actual_q_len)  # np.sum(actual_q_len) #重新更新T1
    T2 = calculate_special_sum(actual_kv_len)  # np.sum(actual_kv_len) #重新更新T2
    print(
        "B, T1, T2, N, BLOCK_LEN, INIT_NUM, LOCAL_NUM, Q_BLOCK_LEN, SPARSE_COUNT: ",
        B,
        T1,
        T2,
        N,
        BLOCK_LEN,
        INIT_NUM,
        LOCAL_NUM,
        Q_BLOCK_LEN,
        SPARSE_COUNT,
    )
    print("actual_q_len", actual_q_len)
    print("actual_kv_len", actual_kv_len)
    D = 128
    block_size = 128
    max_seq_len = max(actual_kv_len)
    layout_query = "TND"
    layout_key = "PA_BSND"
    dType = torch.bfloat16
    np.random.seed(3)

    query = torch.tensor(np.random.uniform(-2, 2, (T1, N, D))).to(dType)
    key = torch.tensor(np.random.uniform(-2, 2, (T2, 1, D))).to(dType)  # T 1 D
    weights = torch.tensor(np.random.uniform(-1, 1, (T1, N))).to(torch.float)
    # TND格式下，actual_seq_lengths_query为前缀和表示
    end_batch = 2
    cur_q_len = [0] + list(np.cumsum(actual_q_len)) + [-1] * end_batch
    cur_kv_len = [0] + list(np.cumsum(actual_kv_len)) + [-1] * end_batch
    cur_seq_lengths_query = torch.tensor(cur_q_len).to(torch.int32)
    cur_seq_lengths_key = torch.tensor(cur_kv_len).to(torch.int32)
    print("cu_q_len", cur_q_len)
    print("cu_kv_len", cur_kv_len)

    sparse_count = SPARSE_COUNT
    # sparse_count = 32
    sparse_mode = 3
    pre_tokens = 2147483647
    next_tokens = 2147483647  # TODO: 设成0还是？
    return_value = False
    cpu_out, cpu_valuesOut = _lightning_indexer(
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        layout_query,
        sparse_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
        BLOCK_LEN,
        INIT_NUM,
        LOCAL_NUM,
        Q_BLOCK_LEN,
    )

    torch_npu.npu.set_device(int(DEVICE_ID))
    query = query.to("npu:%s" % DEVICE_ID)
    key = key.to("npu:%s" % DEVICE_ID)
    weights = weights.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_query = cur_seq_lengths_query.to("npu:%s" % DEVICE_ID)
    cur_seq_lengths_key = cur_seq_lengths_key.to("npu:%s" % DEVICE_ID)

    # start run custom ops
    if layout_key == "TND":
        npu_out, npu_valuesOut = torch.ops.cann_ops_transformer.lightning_indexer(
            query,
            key,
            weights,
            cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key,
            block_table=None,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            kv_block_len=BLOCK_LEN,
            q_block_len=Q_BLOCK_LEN,
            init_num=INIT_NUM,
            local_num=LOCAL_NUM,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            return_value=return_value,
        )
    else:
        key_cache, key_block_table = split_to_kv_cache(
            key, block_size, actual_kv_len, max_seq_len, end_batch
        )
        npu_out, npu_valuesOut = torch.ops.cann_ops_transformer.lightning_indexer(
            query,
            key_cache,
            weights,
            cur_seq_lengths_query=cur_seq_lengths_query,
            cur_seq_lengths_key=cur_seq_lengths_key,
            block_table=key_block_table,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_count=sparse_count,
            kv_block_len=BLOCK_LEN,
            q_block_len=Q_BLOCK_LEN,
            init_num=INIT_NUM,
            local_num=LOCAL_NUM,
            sparse_mode=sparse_mode,
        )

    # Only compare indices. Values are neither copied to CPU nor checked.
    cpu_out = cpu_out.reshape(-1, sparse_count).cpu()
    npu_out = npu_out.reshape(-1, sparse_count).cpu()
    return compare_token_indices(npu_out, cpu_out, T2)


if __name__ == "__main__":
    # Keep the source test's inputs and golden, bounded to the requested range.
    seq_k_values = [
        int(value) for value in os.environ.get("LI_T2_LIST", "4096,8192").split(",")
    ]
    if not seq_k_values or any(value <= 0 or value > 8192 for value in seq_k_values):
        raise ValueError("This validation script accepts 0 < seq_k <= 8192")
    reports = []
    for seq_k in seq_k_values:
        print(
            f"CASE START: query=TND key=PA_BSND BF16 seq_q=1024 seq_k={seq_k} heads=64 topk=128",
            flush=True,
        )
        report = test_tnd_lightning_indexer_eager(1, 1024, 64, 1, 4, 32, 1, 128, seq_k)
        reports.append(report)
        status = "PASS" if report["ordered_mismatch_tokens"] == 0 else "FAIL"
        print(
            f"CASE {status}: seq_k={seq_k}; per-token exact index comparison (including order)",
            flush=True,
        )
    torch.npu.synchronize()
    failed = [report for report in reports if report["ordered_mismatch_tokens"]]
    if failed:
        raise AssertionError(
            "Per-token exact index comparison failed: "
            + "; ".join(
                f"seq_k={report['seq_k']}, ordered mismatch tokens={report['ordered_mismatch_tokens']}, "
                f"index multiset mismatch tokens={report['index_multiset_mismatch_tokens']}"
                for report in failed
            )
        )
    print(
        f"OVERALL PASS: {len(reports)} cases; every token's index array matches exactly; values not checked",
        flush=True,
    )
